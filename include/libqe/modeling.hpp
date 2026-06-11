/**
 * @file modeling.hpp
 * @brief ENA modeling utilities: centering, correlation, confidence intervals,
 *        and node-position solvers for undirected and directed ENA networks.
 */
#ifndef LIBQE_MODELING_HPP
#define LIBQE_MODELING_HPP

#include <armadillo>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <libqe/linalg_fallback.hpp>

namespace qe {

// ---------------------------------------------------------------------------
/// @name Return types
/// @{
// ---------------------------------------------------------------------------

/**
 * @brief Aggregated result returned by every node-position solver.
 *
 * All matrices share the same dimensional conventions:
 * rows index units (or nodes), columns index rotated dimensions.
 */
struct NodePositions {
    arma::mat nodes;      ///< Solved node locations, n_codes × n_dims.
    arma::mat centroids;  ///< Per-unit centroids projected onto the node space, n_units × n_dims.
    arma::mat weights;    ///< Normalised half-edge weight per unit per node, n_units × n_codes.
    arma::mat points;     ///< Input rotated points echoed back unchanged, n_units × n_dims.
};

/// @}

// ---------------------------------------------------------------------------
/// @name Centering
/// @{
// ---------------------------------------------------------------------------

/**
 * @brief Subtract column means so that every dimension is centred at the origin.
 *
 * @param[in] values  Matrix of ENA unit points (n_units × n_dims).
 * @returns A copy of @p values with each column mean subtracted (centre-to-origin).
 *
 * @note Equivalent to `center_data_c()` in rENA/ena.cpp.
 */
inline arma::mat center_points(arma::mat values) {
    return values.each_row() - arma::mean(values);
}

/// @}

/// @cond INTERNAL

// ---------------------------------------------------------------------------
// Normal quantile (probit) — pure C++, no R dependency
// Rational approximation by Peter Acklam; max abs error < 1.15e-9.
// Used internally by ena_correlation() so we don't need Rcpp::qnorm.
// ---------------------------------------------------------------------------

inline double normal_quantile(double p) {
    static const double a[] = {
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00
    };
    static const double b[] = {
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01
    };
    static const double c[] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00
    };
    static const double d[] = {
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00
    };
    const double p_low  = 0.02425;
    const double p_high = 1.0 - p_low;
    double q, r;
    if (p < p_low) {
        q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= p_high) {
        q = p - 0.5; r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
}

// ---------------------------------------------------------------------------
// Internal helpers: regularized incomplete beta + t-distribution quantile
// ---------------------------------------------------------------------------
//
// These live in qe::detail so they don't pollute the public qe namespace.
// They are used by mean_ci() and are not part of the public API.

namespace detail {

// Lentz continued-fraction evaluation for the regularized incomplete beta.
// Ported from Numerical Recipes in C, §6.4.
inline double betacf(double a, double b, double x) {
    const int    MAXIT = 200;
    const double EPS   = std::numeric_limits<double>::epsilon();
    const double FPMIN = std::numeric_limits<double>::min() / EPS;
    double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::abs(d) < FPMIN) d = FPMIN;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= MAXIT; ++m) {
        const int m2 = 2 * m;
        // even step
        double aa = static_cast<double>(m) * (b - static_cast<double>(m)) * x
                    / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d; if (std::abs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c; if (std::abs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d; h *= d * c;
        // odd step
        aa = -(a + static_cast<double>(m)) * (qab + static_cast<double>(m)) * x
             / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d; if (std::abs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c; if (std::abs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::abs(del - 1.0) <= EPS) break;
    }
    return h;
}

// Regularized incomplete beta I_x(a, b).
inline double betai(double a, double b, double x) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    const double lbab = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b);
    const double bt   = std::exp(lbab + a * std::log(x) + b * std::log(1.0 - x));
    if (x < (a + 1.0) / (a + b + 2.0))
        return bt * betacf(a, b, x) / a;
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
}

// Quantile function (inverse CDF) for the t-distribution with `df` degrees
// of freedom.  Uses the relation:
//
//   P(T ≤ t | df)  =  1 - I_x(df/2, 1/2) / 2,   x = df / (df + t^2)
//
// Inverted via Newton–Raphson on I_x with bisection fallback; normal-quantile
// plus one Cornish–Fisher correction term provides the starting guess.
inline double t_quantile(double p, double df) {
    const double INF = std::numeric_limits<double>::infinity();
    if (p <= 0.0) return -INF;
    if (p >= 1.0) return  INF;
    if (p == 0.5) return  0.0;
    if (df <= 0.0) return INF;   // df = 0 → t-distribution undefined → treat as ∞

    // Work with pp > 0.5 for numerical stability; negate at the end if needed.
    const bool   flip = (p < 0.5);
    const double pp   = flip ? 1.0 - p : p;
    const double a    = 0.5 * df;
    const double bv   = 0.5;

    // We need x in (0,1) such that I_x(a, bv) = 2*(1-pp).
    const double target = 2.0 * (1.0 - pp);

    // Initial guess via normal quantile + single Cornish–Fisher correction.
    const double z  = normal_quantile(pp);
    const double t0 = z + (z * z * z + z) / (4.0 * df);
    double x = df / (df + t0 * t0);
    x = std::max(1e-12, std::min(1.0 - 1e-12, x));

    // Newton–Raphson with a bisection bracket.
    // f(x) = I_x(a,bv) - target;  f'(x) = x^(a-1)*(1-x)^(bv-1) / B(a,bv)
    const double lbab = std::lgamma(a + bv) - std::lgamma(a) - std::lgamma(bv);
    double xlo = 0.0, xhi = 1.0;
    for (int iter = 0; iter < 100; ++iter) {
        const double fx = betai(a, bv, x) - target;
        // betai is increasing in x, so fx < 0 → x too small → raise lower bound
        if (fx < 0.0) xlo = x; else xhi = x;

        // Derivative: may underflow near boundaries — fall back to bisection.
        const double log_fpx = lbab + (a - 1.0) * std::log(x)
                                     + (bv - 1.0) * std::log(1.0 - x);
        const double fpx = (log_fpx > -700.0) ? std::exp(log_fpx) : 0.0;

        double x_new;
        if (fpx > 0.0) {
            x_new = x - fx / fpx;
            if (x_new <= xlo || x_new >= xhi)
                x_new = 0.5 * (xlo + xhi);
        } else {
            x_new = 0.5 * (xlo + xhi);
        }

        if (std::abs(x_new - x) < 1e-13 * x) break;
        x = x_new;
    }

    const double t_val = std::sqrt(df * (1.0 - x) / x);
    return flip ? -t_val : t_val;
}

// ---------------------------------------------------------------------------
// IQR helper — matches R's quantile(type = 7) (Hyndman–Fan #7), which is
// identical to numpy's np.percentile(method="linear").
//
// For a sorted n-element vector and probability p:
//   h    = (n - 1) * p
//   lo   = floor(h),  frac = h - lo
//   Q(p) = sorted[lo] * (1 - frac) + sorted[lo+1] * frac
// ---------------------------------------------------------------------------

inline double quantile_type7(const arma::vec& sorted, double p) {
    const int n = static_cast<int>(sorted.n_elem);
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();
    if (n == 1) return sorted[0];
    const double h    = (n - 1) * p;
    const int    lo   = static_cast<int>(std::floor(h));
    const double frac = h - lo;
    if (lo + 1 >= n) return sorted[n - 1];          // guard: p == 1.0
    return sorted[lo] * (1.0 - frac) + sorted[lo + 1] * frac;
}

// Interquartile range for a column vector — uses the same type-7 quantile as R.
inline double iqr(const arma::vec& col) {
    const arma::vec s = arma::sort(col);
    return quantile_type7(s, 0.75) - quantile_type7(s, 0.25);
}

} // namespace detail

/// @endcond

// ---------------------------------------------------------------------------
/// @name Correlation
/// @{
// ---------------------------------------------------------------------------

/**
 * @brief Pearson correlation with Fisher-z confidence interval between ENA
 *        unit points and their centroids.
 *
 * All unique pairs of units are formed; for each pair the per-dimension
 * difference vectors are computed, then Pearson r is obtained between the
 * @p points differences and the @p centroids differences.  The confidence
 * interval is derived via Fisher's z-transform using the pure-C++
 * `normal_quantile` function so the function works outside R without
 * `Rcpp::qnorm`.
 *
 * @param[in] points      Rotated ENA unit points, n_units × n_dims.
 * @param[in] centroids   Corresponding centroid coordinates, n_units × n_dims.
 * @param[in] conf_level  Confidence level for the interval (default 0.95).
 * @returns An n_dims × 3 matrix whose columns are [r, ci_lower, ci_upper].
 *
 * @note Equivalent to `ena_correlation()` in rENA/ena.cpp.
 */
inline arma::mat ena_correlation(arma::mat points, arma::mat centroids,
                                  double conf_level = 0.95) {
    int n = points.n_rows;
    int n_pairs = (n * (n - 1)) / 2;

    arma::umat idx1(1, n_pairs), idx2(1, n_pairs);
    int col = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) { idx1[col] = i; idx2[col] = j; col++; }

    arma::mat pts_diff = points.rows(idx1.row(0))    - points.rows(idx2.row(0));
    arma::mat cts_diff = centroids.rows(idx1.row(0)) - centroids.rows(idx2.row(0));
    arma::mat cr       = arma::cor(pts_diff, cts_diff);

    double qq = normal_quantile((1.0 + conf_level) / 2.0);
    arma::mat out(points.n_cols, 3);
    for (arma::uword i = 0; i < points.n_cols; i++) {
        double r     = cr(i, i);
        double z     = std::atanh(r);
        double sigma = 1.0 / std::sqrt(static_cast<double>(n_pairs) - 3.0);
        out(i, 0) = r;
        out(i, 1) = std::tanh(z - sigma * qq);
        out(i, 2) = std::tanh(z + sigma * qq);
    }
    return out;
}

/// @}

// ---------------------------------------------------------------------------
/// @name Group confidence intervals
/// @{
// ---------------------------------------------------------------------------

/**
 * @brief Student-t confidence interval for the mean of a group of ENA unit points.
 *
 * For each dimension d:
 * @code
 *   mean ± t_{α/2, n-1} × (sample_sd / sqrt(n))
 * @endcode
 * where α = 1 − @p conf_level and degrees-of-freedom = n − 1.
 *
 * @param[in] points      ENA unit points for a single group, n_units × n_dims.
 * @param[in] conf_level  Confidence level for the interval (default 0.95).
 * @returns An n_dims × 3 matrix whose columns are [mean, ci_lower, ci_upper].
 *          When n == 0 all entries are NaN.
 *          When n == 1 the CI bounds are ±Inf.
 */
inline arma::mat mean_ci(const arma::mat& points, double conf_level = 0.95) {
    const int n      = static_cast<int>(points.n_rows);
    const int n_dims = static_cast<int>(points.n_cols);

    arma::mat out(n_dims, 3);
    out.fill(std::numeric_limits<double>::quiet_NaN());

    if (n == 0) return out;

    const double df     = static_cast<double>(n - 1);
    const double t_crit = detail::t_quantile(1.0 - (1.0 - conf_level) / 2.0, df);
    const double INF    = std::numeric_limits<double>::infinity();

    for (int d = 0; d < n_dims; ++d) {
        const arma::vec col = points.col(d);
        const double mu = arma::mean(col);
        out(d, 0) = mu;
        if (n == 1) {
            // t_crit = Inf and se = 0 would give NaN; explicitly set ±Inf
            out(d, 1) = -INF;
            out(d, 2) =  INF;
        } else {
            const double se = arma::stddev(col) / std::sqrt(static_cast<double>(n));
            out(d, 1) = mu - t_crit * se;
            out(d, 2) = mu + t_crit * se;
        }
    }
    return out;
}

/**
 * @brief Tukey-fence outlier interval for a group of ENA unit points.
 *
 * For each dimension d the half-width is computed as:
 * @code
 *   half_width[d] = iqr_factor * IQR(points[:, d])
 * @endcode
 * and the interval [−half_width, +half_width] is symmetric around zero,
 * matching rENA's formula:
 * @code
 *   outlier.interval.values = matrix(...) * c(-1, 1)
 * @endcode
 * where the matrix is built from `c(IQR(dim1), IQR(dim2), ...) * iqr_factor`.
 *
 * IQR uses R's default type-7 / Hyndman-Fan #7 quantile, identical to
 * `numpy.percentile(method="linear")`.
 *
 * @param[in] points      ENA unit points for a single group, n_units × n_dims.
 * @param[in] iqr_factor  Multiplier applied to the IQR (default 1.5, the
 *                        standard Tukey fence threshold).
 * @returns An n_dims × 2 matrix whose columns are [lower, upper].
 *          When n == 0 all entries are NaN.
 *          When n == 1, IQR == 0, so lower == upper == 0.
 */
inline arma::mat outlier_ci(const arma::mat& points, double iqr_factor = 1.5) {
    const int n_dims = static_cast<int>(points.n_cols);
    arma::mat out(n_dims, 2, arma::fill::zeros);

    if (points.n_rows == 0) {
        out.fill(std::numeric_limits<double>::quiet_NaN());
        return out;
    }

    for (int d = 0; d < n_dims; ++d) {
        const double half_width = iqr_factor * detail::iqr(points.col(d));
        out(d, 0) = -half_width;
        out(d, 1) =  half_width;
    }
    return out;
}

/// @}

// ---------------------------------------------------------------------------
/// @name Node-position solvers
/// @{
// ---------------------------------------------------------------------------

/**
 * @brief Multiobjective least-squares node positions for undirected ENA.
 *
 * Half of each adjacency-vector entry (line weight) is distributed to each of
 * its two endpoint nodes, building a per-unit weight matrix.  Each row is then
 * L1-normalised.  An overdetermined system is solved per dimension:
 * @code
 *   (W^T W) X = W^T T
 * @endcode
 * where W is the normalised weight matrix and T contains the rotated unit
 * points.
 *
 * @param[in] adj_mats  Upper-triangular adjacency vectors stacked row-wise,
 *                      n_units × tri_size.
 * @param[in] t         Rotated unit points, n_units × n_dims.
 * @param[in] num_dims  Number of dimensions to solve for.
 * @returns A @ref NodePositions struct containing `nodes`, `centroids`,
 *          `weights`, and `points`.
 *
 * @note Equivalent to `lws_lsq_positions()` in rENA/ena.cpp.
 */
inline NodePositions node_positions(arma::mat adj_mats, arma::mat t,
                                        int num_dims) {
    if (!adj_mats.is_finite() || !t.is_finite())
        throw std::invalid_argument("node_positions: input matrices must not contain NaN or Inf");
    int tri_size  = adj_mats.n_cols;
    int num_nodes = static_cast<int>(
        std::pow(std::ceil(std::sqrt(static_cast<double>(2 * tri_size))), 2.0)
    ) - (2 * tri_size);
    int row_count = adj_mats.n_rows;

    arma::mat weights(row_count, num_nodes, arma::fill::zeros);
    for (int k = 0; k < row_count; k++) {
        arma::rowvec curr = adj_mats.row(k);
        int z = 0;
        for (int x = 0; x < num_nodes - 1; x++) {
            for (int y = 0; y <= x; y++) {
                weights(k, x + 1) += 0.5 * curr[z];
                weights(k, y)     += 0.5 * curr[z];
                z++;
            }
        }
    }
    for (int k = 0; k < row_count; k++) {
        double len = arma::accu(arma::abs(weights.row(k)));
        if (len < 0.0001) len = 0.0001;
        weights.row(k) /= len;
    }

    arma::mat ssX(num_dims, num_nodes, arma::fill::zeros);
    arma::mat ssA = weights.t() * weights;
    for (int i = 0; i < num_dims; i++) {
        arma::mat ssb = weights.t() * t.col(i);
        ssX.row(i) = qe::linalg::solve_spd(ssA, ssb).t();
    }

    NodePositions r;
    r.nodes     = ssX.t();
    r.centroids = (ssX * weights.t()).t();
    r.weights   = weights;
    r.points    = t;
    return r;
}

/**
 * @brief Least-squares node positions for directed (ordered) ENA.
 *
 * Each row of @p line_weights is an n_nodes × n_nodes directed weight matrix
 * stored in row-major order.  The per-unit node weight for node x accumulates
 * the full row weight plus all column weights directed at x from other nodes.
 * Each row is L1-normalised before solving.
 *
 * When @p combine_pairs is `false` (the standard directed case) the system:
 * @code
 *   (W^T W) X = W^T P
 * @endcode
 * is solved directly with W = normalised weight matrix and P = @p points.
 *
 * When @p combine_pairs is `true`, adjacent row pairs (ground row k and
 * response row k+1) are summed before solving:
 * @code
 *   W_combined[k/2] = W[k] + W[k+1]
 *   P_combined[k/2] = P[k] + P[k+1]
 * @endcode
 * The system is then solved on the combined matrices, but centroids are
 * projected back using the original (un-combined) weight matrix so that
 * every unit retains its own centroid.
 *
 * @param[in] line_weights  Directed adjacency vectors, n_units × (n_nodes²).
 * @param[in] points        Rotated unit points, n_units × n_dims.
 * @param[in] num_dims      Number of dimensions to solve for.
 * @param[in] combine_pairs If `true`, sum adjacent row pairs before solving
 *                          (ground + response model).  Default `false`.
 * @returns A @ref NodePositions struct containing `nodes`, `centroids`,
 *          `weights`, and `points`.
 *
 * @note Equivalent to `directed_node_positions()` in rENA/ena.cpp when
 *       @p combine_pairs is `false`.
 * @note Equivalent to `directed_node_positions_with_ground_response_added()`
 *       in rENA/ena.cpp when @p combine_pairs is `true`.
 *
 * @warning When @p combine_pairs is `true`, @p line_weights must have an even
 *          number of rows (row_count must be even); odd row counts result in
 *          an out-of-bounds access when forming the combined matrices.
 */
inline NodePositions directed_node_positions(arma::mat line_weights,
                                              arma::mat points, int num_dims,
                                              bool combine_pairs = false) {
    if (!line_weights.is_finite() || !points.is_finite())
        throw std::invalid_argument("directed_node_positions: input matrices must not contain NaN or Inf");
    int num_nodes = static_cast<int>(
        std::ceil(std::sqrt(static_cast<double>(line_weights.n_cols)))
    );
    int row_count = line_weights.n_rows;

    arma::mat nw(row_count, num_nodes, arma::fill::zeros);
    for (int k = 0; k < row_count; k++) {
        arma::mat curr = line_weights.row(k);
        int z = 0;
        for (int x = 0; x < num_nodes; x++)
            for (int y = 0; y < num_nodes; y++) {
                nw(k, x) += curr[z];
                if (x != y) nw(k, y) += curr[z];
                z++;
            }
    }
    for (int k = 0; k < row_count; k++) {
        double len = arma::accu(arma::abs(nw.row(k)));
        if (len < 0.0001) len = 0.0001;
        nw.row(k) /= len;
    }

    if (combine_pairs) {
        // Combine paired rows (ground at k, response at k+1)
        arma::mat nw_added(row_count / 2, num_nodes, arma::fill::zeros);
        arma::mat pts_added(row_count / 2, num_dims, arma::fill::zeros);
        for (int k = 0; k < row_count; k += 2) {
            nw_added.row(k / 2)  = nw.row(k)     + nw.row(k + 1);
            pts_added.row(k / 2) = points.row(k) + points.row(k + 1);
        }

        arma::mat ssX(num_dims, num_nodes, arma::fill::zeros);
        arma::mat ssA = nw_added.t() * nw_added;
        for (int i = 0; i < num_dims; i++) {
            arma::mat ssb = nw_added.t() * pts_added.col(i);
            ssX.row(i) = qe::linalg::solve_spd(ssA, ssb).t();
        }

        NodePositions r;
        r.nodes     = ssX.t();
        r.centroids = (ssX * nw.t()).t();
        r.weights   = nw;
        r.points    = points;
        return r;
    }

    arma::mat ssX(num_dims, num_nodes, arma::fill::zeros);
    arma::mat ssA = nw.t() * nw;
    for (int i = 0; i < num_dims; i++) {
        arma::mat ssb = nw.t() * points.col(i);
        ssX.row(i) = qe::linalg::solve_spd(ssA, ssb).t();
    }

    NodePositions r;
    r.nodes     = ssX.t();
    r.centroids = (ssX * nw.t()).t();
    r.weights   = nw;
    r.points    = points;
    return r;
}

/// @}

// ---------------------------------------------------------------------------
/// @name Two-group comparison statistics
/// @{
// ---------------------------------------------------------------------------

/**
 * @brief Per-dimension statistics from a two-group comparison.
 *
 * Matches the output of rENA-api's `group.stats()`:
 *
 * - **Parametric** (Welch two-sample t-test): `t`, `df`, `pvalue_t`,
 *   `cohens_d` (pooled-SD formula), `means`, `sds`.
 * - **Non-parametric** (Wilcoxon rank-sum / Mann-Whitney U):
 *   `U` (= R's W statistic for group 1), `pvalue_u` (normal approximation
 *   with tie correction and continuity correction), `effect_r`
 *   (rank-biserial: `1 − 2*U / (n1*n2)`), `medians`.
 *
 * All per-dimension vectors have length `n_dims`; `means`, `sds`, and
 * `medians` are `2 × n_dims` matrices with row 0 = group 1, row 1 = group 2.
 * Entries are NaN when a statistic is undefined (e.g. n < 2 for parametric
 * tests, or n = 0 for either group).
 */
struct GroupStats {
    int n1;       ///< Sample size of group 1.
    int n2;       ///< Sample size of group 2.

    // Parametric
    arma::vec t;          ///< Welch two-sample t-statistics.
    arma::vec df;         ///< Welch–Satterthwaite degrees of freedom.
    arma::vec pvalue_t;   ///< Two-tailed p-values from the Welch t-test.
    arma::vec cohens_d;   ///< Cohen's d (pooled-SD formula, group1 − group2).
    arma::mat means;      ///< 2 × n_dims: means; row 0 = group1, row 1 = group2.
    arma::mat sds;        ///< 2 × n_dims: sample standard deviations.

    // Non-parametric
    arma::vec U;          ///< Wilcoxon rank-sum U for group 1 (= R's W statistic).
    arma::vec pvalue_u;   ///< Two-tailed p-values (normal approx, tie + continuity correction).
    arma::vec effect_r;   ///< Rank-biserial effect: 1 − 2·U / (n1·n2).
    arma::mat medians;    ///< 2 × n_dims: medians; row 0 = group1, row 1 = group2.
};

/// @cond INTERNAL
namespace detail {

// Two-tailed p-value for a Welch t-statistic.
// Uses the relation P(T≤t|df) = 1 − I_x(df/2, 1/2)/2, x = df/(df+t²).
inline double t_test_pvalue(double t_stat, double df) {
    if (!std::isfinite(t_stat) || df <= 0.0)
        return std::numeric_limits<double>::quiet_NaN();
    const double x = df / (df + t_stat * t_stat);
    return betai(df / 2.0, 0.5, x);   // == 2 * P(T_df > |t_stat|)
}

// Wilcoxon rank-sum U (= R's W for group 1) and two-tailed p-value.
// Normal approximation with tie correction and continuity correction,
// matching R's wilcox.test(..., exact=FALSE, correct=TRUE).
struct WilcoxResult { double U; double p; };

inline WilcoxResult wilcox_ranksum(const arma::vec& g1, const arma::vec& g2) {
    const int    n1  = static_cast<int>(g1.n_elem);
    const int    n2  = static_cast<int>(g2.n_elem);
    const int    n   = n1 + n2;
    const double NaN = std::numeric_limits<double>::quiet_NaN();

    if (n1 == 0 || n2 == 0) return {NaN, NaN};

    // Merge both groups into a single array, tagging each element's origin.
    arma::vec      vals(n);
    std::vector<int> grp(n);
    for (int i = 0; i < n1; ++i) { vals[i]      = g1[i]; grp[i]      = 0; }
    for (int i = 0; i < n2; ++i) { vals[n1 + i] = g2[i]; grp[n1 + i] = 1; }

    arma::uvec ord = arma::sort_index(vals);

    // Assign average ranks for tied values; accumulate tie correction T = Σ(t³−t).
    arma::vec rnk(n);
    double    T_ties = 0.0;
    arma::uword i = 0;
    while (i < static_cast<arma::uword>(n)) {
        arma::uword j = i;
        while (j + 1 < static_cast<arma::uword>(n) &&
               vals[ord[j + 1]] == vals[ord[j]])
            ++j;
        const double avg = 0.5 * (static_cast<double>(i) + static_cast<double>(j)) + 1.0;
        for (arma::uword k = i; k <= j; ++k) rnk[ord[k]] = avg;
        const double t = static_cast<double>(j - i + 1);
        T_ties += t * (t * t - 1.0);   // t³ − t
        i = j + 1;
    }

    // W = rank-sum for group 1; U = W − n1*(n1+1)/2 (R's wilcox.test statistic).
    double W = 0.0;
    for (int k = 0; k < n; ++k)
        if (grp[k] == 0) W += rnk[k];
    const double U = W - static_cast<double>(n1) * (n1 + 1) / 2.0;

    // Normal approximation: Var[U] = n1·n2/12 · ((n+1) − T_ties/(n·(n−1)))
    const double mu_U  = static_cast<double>(n1) * n2 / 2.0;
    const double tie_adj = (n > 1) ? T_ties / (static_cast<double>(n) * (n - 1)) : 0.0;
    double var_U = static_cast<double>(n1) * n2 / 12.0
                   * (static_cast<double>(n + 1) - tie_adj);
    if (var_U < 0.0) var_U = 0.0;

    double p;
    if (var_U == 0.0) {
        p = (U == mu_U) ? 1.0 : 0.0;
    } else {
        // Continuity correction: |U − mu| − 0.5
        double z = (std::abs(U - mu_U) - 0.5) / std::sqrt(var_U);
        if (z < 0.0) z = 0.0;
        p = std::erfc(z / std::sqrt(2.0));   // = 2*(1−Φ(z))
        p = std::min(p, 1.0);
    }

    return {U, p};
}

} // namespace detail (group stats helpers)
/// @endcond

/**
 * @brief Compute per-dimension two-group comparison statistics.
 *
 * For each dimension @p d of the input matrices the function computes:
 *
 * **Parametric (Welch two-sample t-test)**
 *  - t-statistic and Welch–Satterthwaite degrees of freedom
 *  - Two-tailed p-value via the regularised incomplete beta function
 *  - Cohen's d using the pooled-SD formula
 *    (`(μ1−μ2) / √(((n1−1)s1² + (n2−1)s2²)/(n1+n2−2))`)
 *  - Per-group means and sample standard deviations
 *
 * **Non-parametric (Wilcoxon rank-sum / Mann-Whitney U)**
 *  - U statistic for group 1 (identical to R's `wilcox.test` W statistic)
 *  - Two-tailed p-value via the normal approximation with tie correction
 *    and continuity correction (matching R's `wilcox.test(..., correct=TRUE)`)
 *  - Rank-biserial effect size: `1 − 2·U / (n1·n2)`
 *    (rENA-api's `nonparam.effect` formula)
 *  - Per-group medians
 *
 * @param[in] g1  Unit points for group 1, n1 × n_dims.
 * @param[in] g2  Unit points for group 2, n2 × n_dims.
 * @returns A @ref GroupStats struct.
 *
 * @note Parametric statistics are NaN when either group has fewer than 2 units
 *       (no sample variance).  All statistics are NaN when either group is empty.
 * @note The Wilcoxon p-value uses the normal approximation regardless of sample
 *       size; R switches to an exact test for small (n ≤ ~50) tie-free samples,
 *       so p-values may differ slightly in those cases.
 * @note Equivalent to `group.stats()` in rENA-api, moved to the C++ layer so
 *       all language bindings share a single implementation.
 */
inline GroupStats group_stats(const arma::mat& g1, const arma::mat& g2) {
    const int    n1     = static_cast<int>(g1.n_rows);
    const int    n2     = static_cast<int>(g2.n_rows);
    const int    n_dims = static_cast<int>(g1.n_cols);
    const double NaN    = std::numeric_limits<double>::quiet_NaN();

    GroupStats out;
    out.n1 = n1;
    out.n2 = n2;

    out.t        = arma::vec(n_dims); out.t.fill(NaN);
    out.df       = arma::vec(n_dims); out.df.fill(NaN);
    out.pvalue_t = arma::vec(n_dims); out.pvalue_t.fill(NaN);
    out.cohens_d = arma::vec(n_dims); out.cohens_d.fill(NaN);
    out.means    = arma::mat(2, n_dims); out.means.fill(NaN);
    out.sds      = arma::mat(2, n_dims); out.sds.fill(NaN);
    out.U        = arma::vec(n_dims); out.U.fill(NaN);
    out.pvalue_u = arma::vec(n_dims); out.pvalue_u.fill(NaN);
    out.effect_r = arma::vec(n_dims); out.effect_r.fill(NaN);
    out.medians  = arma::mat(2, n_dims); out.medians.fill(NaN);

    if (n1 == 0 || n2 == 0) return out;

    for (int d = 0; d < n_dims; ++d) {
        const arma::vec c1 = g1.col(d);
        const arma::vec c2 = g2.col(d);

        // ── Means and medians ───────────────────────────────────────────────
        const double m1 = arma::mean(c1);
        const double m2 = arma::mean(c2);
        out.means(0, d) = m1;
        out.means(1, d) = m2;
        out.medians(0, d) = arma::median(c1);
        out.medians(1, d) = arma::median(c2);

        // ── Standard deviations (ddof = 1, matching R's sd()) ───────────────
        const double s1 = (n1 > 1) ? arma::stddev(c1) : NaN;
        const double s2 = (n2 > 1) ? arma::stddev(c2) : NaN;
        out.sds(0, d) = s1;
        out.sds(1, d) = s2;

        // ── Welch two-sample t-test ──────────────────────────────────────────
        if (n1 > 1 && n2 > 1 && std::isfinite(s1) && std::isfinite(s2)) {
            const double v1n = s1 * s1 / n1;
            const double v2n = s2 * s2 / n2;
            const double se  = std::sqrt(v1n + v2n);
            if (se > 0.0) {
                const double t_stat = (m1 - m2) / se;
                const double df_val = (v1n + v2n) * (v1n + v2n)
                                    / (v1n * v1n / (n1 - 1) + v2n * v2n / (n2 - 1));
                out.t(d)        = t_stat;
                out.df(d)       = df_val;
                out.pvalue_t(d) = detail::t_test_pvalue(t_stat, df_val);
            }
        }

        // ── Cohen's d (pooled SD) ────────────────────────────────────────────
        if (n1 + n2 > 2 && std::isfinite(s1) && std::isfinite(s2)) {
            const double pooled_var = ((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2)
                                    / (n1 + n2 - 2);
            const double pooled_sd  = std::sqrt(pooled_var);
            out.cohens_d(d) = (pooled_sd > 0.0) ? (m1 - m2) / pooled_sd : 0.0;
        }

        // ── Wilcoxon rank-sum ────────────────────────────────────────────────
        const detail::WilcoxResult wr = detail::wilcox_ranksum(c1, c2);
        out.U(d)        = wr.U;
        out.pvalue_u(d) = wr.p;
        if (std::isfinite(wr.U))
            out.effect_r(d) = 1.0 - 2.0 * wr.U / (static_cast<double>(n1) * n2);
    }

    return out;
}

/// @}

} // namespace qe

#endif // LIBQE_MODELING_HPP
