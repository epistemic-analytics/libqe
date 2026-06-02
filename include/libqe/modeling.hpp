#ifndef LIBQE_MODELING_HPP
#define LIBQE_MODELING_HPP

#include <armadillo>
#include <cmath>

namespace qe {

// ---------------------------------------------------------------------------
// Return type for node-position solvers
// ---------------------------------------------------------------------------

struct NodePositions {
    arma::mat nodes;      // n_codes x n_dims
    arma::mat centroids;  // n_units x n_dims
    arma::mat weights;    // n_units x n_codes (half-edge weight per node)
    arma::mat points;     // n_units x n_dims (input rotated points, echoed back)
};

// ---------------------------------------------------------------------------
// Centering
// ---------------------------------------------------------------------------

// Subtract column means (center-to-origin).
// Equivalent to center_data_c() in rENA/ena.cpp.
inline arma::mat center_points(arma::mat values) {
    return values.each_row() - arma::mean(values);
}

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

// ---------------------------------------------------------------------------
// Correlation
// ---------------------------------------------------------------------------

// Pearson correlation + 95% (or custom) CI between ENA points and centroids.
// Returns n_dims x 3 matrix: [r, ci_lower, ci_upper].
// Equivalent to ena_correlation() in rENA/ena.cpp, but uses pure-C++ qnorm
// instead of Rcpp::qnorm so it works outside R.
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

// ---------------------------------------------------------------------------
// Group confidence interval
// ---------------------------------------------------------------------------

// Confidence interval for the mean of a group of ENA unit points.
//
// For each dimension, computes:
//   mean  ± t_{α/2, n-1}  ×  (sample SD / sqrt(n))
//
// where α = 1 - conf_level and degrees-of-freedom = n - 1.
//
// Returns an n_dims × 3 matrix: columns are [mean, ci_lower, ci_upper].
// When n == 1 the CI bounds are ±Inf; when n == 0 all entries are NaN.
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

// ---------------------------------------------------------------------------
// Outlier interval
// ---------------------------------------------------------------------------

// Outlier interval for a group of ENA unit points using the Tukey fence
// threshold (1.5 × IQR by default).
//
// For each dimension d:
//   half_width[d] = iqr_factor * IQR(points[:, d])
//   lower[d]      = -half_width[d]
//   upper[d]      = +half_width[d]
//
// The interval is symmetric around 0 — matching rENA's formula:
//   outlier.interval.values = matrix(...) * c(-1, 1)
// where the matrix is built from c(IQR(dim1), IQR(dim2), ...) * iqr_factor.
//
// IQR uses R's default type-7 / Hyndman–Fan #7 quantile, which is identical
// to numpy's np.percentile(method="linear").
//
// Returns an n_dims × 2 matrix: columns are [lower, upper].
// When n == 0, all entries are NaN.
// When n == 1, IQR == 0, so lower == upper == 0.
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

// ---------------------------------------------------------------------------
// Node-position solvers
// ---------------------------------------------------------------------------

// Multiobjective least-squares node positions for undirected ENA.
// Half of each line weight is distributed to each of its two endpoint nodes,
// then an overdetermined system is solved per dimension.
// Equivalent to lws_lsq_positions() in rENA/ena.cpp.
inline NodePositions node_positions(arma::mat adj_mats, arma::mat t,
                                        int num_dims) {
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
        ssX.row(i) = arma::solve(ssA, ssb).t();
    }

    NodePositions r;
    r.nodes     = ssX.t();
    r.centroids = (ssX * weights.t()).t();
    r.weights   = weights;
    r.points    = t;
    return r;
}

// Least-squares node positions for directed (ordered) ENA.
// When combine_pairs == false (default): standard directed node positions.
// When combine_pairs == true: paired ground+response rows are combined before
//   solving — used for directed ENA where each unit contributes a ground row
//   and a response row that should be averaged together.
// Equivalent to directed_node_positions() and
//   directed_node_positions_with_ground_response_added() in rENA/ena.cpp.
inline NodePositions directed_node_positions(arma::mat line_weights,
                                              arma::mat points, int num_dims,
                                              bool combine_pairs = false) {
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
            ssX.row(i) = arma::solve(ssA, ssb).t();
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
        ssX.row(i) = arma::solve(ssA, ssb).t();
    }

    NodePositions r;
    r.nodes     = ssX.t();
    r.centroids = (ssX * nw.t()).t();
    r.weights   = nw;
    r.points    = points;
    return r;
}

} // namespace qe

#endif // LIBQE_MODELING_HPP
