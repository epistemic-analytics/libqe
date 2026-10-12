// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <libqe/libqe.hpp>

using namespace Rcpp;

// =============================================================================
// Adjacency utilities
// =============================================================================

//' Number of upper-triangle pairs for n codes
//' @param n Number of codes
//' @return Integer: n*(n-1)/2
//' @export
// [[Rcpp::export]]
int choose_two(int n) {
    return qe::choose_two(n);
}

//' Upper-triangle index pairs
//' @param len Number of codes (side length of square matrix)
//' @param row -1 = both rows, 0 = row indices only, 1 = col indices only
//' @export
// [[Rcpp::export]]
arma::umat connection_indices(int len, int row = -1) {
    return qe::connection_indices(len, row);
}

//' Pairwise products → upper-triangle vector
//' @param v Numeric vector of code values
//' @export
// [[Rcpp::export]]
arma::rowvec code_connections(arma::mat v) {
    return qe::code_connections(v);
}

//' Fold a directed (n*n) vector into an undirected upper-triangle vector
//' @param v Numeric vector of length n*n
//' @export
// [[Rcpp::export]]
arma::rowvec fold_directed_network(arma::vec v) {
    return qe::fold_directed_network(v);
}

//' Flatten an adjacency matrix to a connection vector
//' @param x Numeric matrix
//' @param full TRUE = full n*n (directed); FALSE = upper triangle (undirected)
//' @export
// [[Rcpp::export]]
arma::rowvec network_to_vector(arma::mat x, bool full = true) {
    return qe::network_to_vector(x, full);
}

//' Code-name pairs for upper-triangle positions ("A & B")
//' @param v Character vector of code names
//' @export
// [[Rcpp::export]]
std::vector<std::string> connection_names(std::vector<std::string> v) {
    return qe::connection_names(v);
}

// =============================================================================
// Normalization
// =============================================================================

//' Row-wise L2 (sphere) normalization
//' @param m Numeric matrix
//' @export
// [[Rcpp::export]]
arma::mat normalize_networks(arma::mat m) {
    return qe::normalize_networks(m);
}

//' Max-norm scaling (divide all rows by the largest row L2 norm)
//' @param m Numeric matrix
//' @export
// [[Rcpp::export]]
arma::mat scale_networks(arma::mat m) {
    return qe::scale_networks(m);
}

// =============================================================================
// Modeling
// =============================================================================

//' Center points (subtract column means)
//' @param values Numeric matrix
//' @export
// [[Rcpp::export]]
arma::mat center_points(arma::mat values) {
    return qe::center_points(values);
}

//' Confidence interval for the mean of a group of ENA unit points
//'
//' For each dimension computes a t-based CI:
//' \code{mean ± t(α/2, n-1) × (SD / sqrt(n))} where \code{α = 1 - conf_level}.
//'
//' @param points     Numeric matrix (units x dims) — one row per unit in the group
//' @param conf_level Confidence level (default 0.95)
//' @return Numeric matrix (n_dims x 3): columns are [mean, ci_lower, ci_upper].
//'   When \code{nrow(points) == 1} the CI bounds are \code{±Inf}.
//' @export
// [[Rcpp::export]]
arma::mat mean_ci(arma::mat points, double conf_level = 0.95) {
    return qe::mean_ci(points, conf_level);
}

//' Outlier interval based on IQR (Tukey fence) for a group of ENA unit points
//'
//' For each dimension: \code{lower = -iqr_factor * IQR}, \code{upper = +iqr_factor * IQR}.
//' Equivalent to rENA's:
//' \preformatted{
//'   oi <- c(IQR(pts[,1]), ...) * iqr_factor
//'   matrix(rep(oi, 2), ncol = n_dims, byrow = TRUE) * c(-1, 1)
//' }
//' IQR uses R's default \code{type = 7} quantile.
//'
//' @param points     Numeric matrix (units x dims) — one row per unit in the group
//' @param iqr_factor Multiplier applied to the IQR (default 1.5, the Tukey fence)
//' @return Numeric matrix (n_dims x 2): columns are [lower, upper].
//' @export
// [[Rcpp::export]]
arma::mat outlier_ci(arma::mat points, double iqr_factor = 1.5) {
    return qe::outlier_ci(points, iqr_factor);
}

//' Two-group comparison statistics
//'
//' Computes per-dimension parametric and non-parametric statistics for two
//' groups of ENA unit points.  Equivalent to \code{group.stats()} in rENA-api,
//' moved to the C++ layer so all language bindings share a single implementation.
//'
//' \strong{Parametric (Welch two-sample t-test):}
//' \itemize{
//'   \item \code{t}        — Welch t-statistics
//'   \item \code{parameter}— Welch–Satterthwaite degrees of freedom
//'   \item \code{pvalue}   — two-tailed p-values
//'   \item \code{effect}   — Cohen's d (pooled-SD: \code{(μ1−μ2)/pooled_sd})
//'   \item \code{mean}     — 2×n_dims matrix, row 1 = group1, row 2 = group2
//'   \item \code{std.dev}  — 2×n_dims matrix of sample standard deviations
//' }
//'
//' \strong{Non-parametric (Wilcoxon rank-sum):}
//' \itemize{
//'   \item \code{U}        — rank-sum U for group 1 (= R's \code{wilcox.test} W)
//'   \item \code{pvalue}   — two-tailed p-values (normal approx, tie + continuity correction)
//'   \item \code{effect}   — rank-biserial: \code{1 − 2·U / (n1·n2)}
//'   \item \code{median}   — 2×n_dims matrix of medians
//' }
//'
//' Entries are \code{NA} when a statistic is undefined (e.g. n < 2 for
//' parametric tests).
//'
//' @param g1 Numeric matrix (n1 × n_dims) — unit points for group 1
//' @param g2 Numeric matrix (n2 × n_dims) — unit points for group 2
//' @return Named list with \code{N}, \code{parametric}, and \code{nonparametric}
//'   sub-lists, matching the structure of rENA-api's \code{group.stats()}.
//' @export
// [[Rcpp::export]]
List group_stats(arma::mat g1, arma::mat g2) {
    qe::GroupStats s = qe::group_stats(g1, g2);
    return List::create(
        _("N") = IntegerVector::create(s.n1, s.n2),
        _("parametric") = List::create(
            _("t")         = s.t,
            _("parameter") = s.df,
            _("pvalue")    = s.pvalue_t,
            _("effect")    = s.cohens_d,
            _("mean")      = s.means,
            _("std.dev")   = s.sds
        ),
        _("nonparametric") = List::create(
            _("U")      = s.U,
            _("pvalue") = s.pvalue_u,
            _("effect") = s.effect_r,
            _("median") = s.medians
        )
    );
}

// =============================================================================
// Door Temporal Pooling & Smoothing
// =============================================================================

//' Door lookback window pooling for a single block
//' @param block Numeric matrix of connection counts
//' @param lookback_size Window lookback size
//' @param aggregate_mean Logical: TRUE = mean, FALSE = sum
//' @param weighting_linear Logical: TRUE = linear weights, FALSE = equal
//' @param segment_ids Integer vector of segment IDs (0-indexed or 1-indexed)
//' @export
// [[Rcpp::export]]
arma::mat door_lookback_block(
    arma::mat block,
    int lookback_size = 20,
    bool aggregate_mean = false,
    bool weighting_linear = false,
    IntegerVector segment_ids = IntegerVector()
) {
    std::vector<int> segs(segment_ids.begin(), segment_ids.end());
    return qe::lookback_block(block, lookback_size, aggregate_mean, weighting_linear, segs);
}

//' Door EMA smoothing for a single block
//' @param block Numeric matrix of connection counts
//' @param alpha Smoothing factor in (0, 1]
//' @param segment_ids Integer vector of segment IDs
//' @export
// [[Rcpp::export]]
arma::mat door_ema_block(
    arma::mat block,
    double alpha = 0.1,
    IntegerVector segment_ids = IntegerVector()
) {
    std::vector<int> segs(segment_ids.begin(), segment_ids.end());
    return qe::ema_block(block, alpha, segs);
}

//' Door lookback window pooling across units
//' @param conn_counts Numeric matrix of connection counts
//' @param unit_row_indices List of integer vectors of row indices (0-indexed or 1-indexed R converted)
//' @param lookback_sizes Integer vector of lookback sizes per unit
//' @param aggregate_mean Logical: TRUE = mean, FALSE = sum
//' @param weighting_linear Logical: TRUE = linear weights, FALSE = equal
//' @param segment_ids Optional integer vector of segment IDs
//' @export
// [[Rcpp::export]]
arma::mat door_lookback(
    arma::mat conn_counts,
    List unit_row_indices,
    IntegerVector lookback_sizes,
    bool aggregate_mean = false,
    bool weighting_linear = false,
    IntegerVector segment_ids = IntegerVector()
) {
    std::vector<std::vector<int>> unit_rows;
    unit_rows.reserve(unit_row_indices.size());
    for (int i = 0; i < unit_row_indices.size(); ++i) {
        IntegerVector iv = unit_row_indices[i];
        unit_rows.push_back(std::vector<int>(iv.begin(), iv.end()));
    }
    std::vector<int> lbs(lookback_sizes.begin(), lookback_sizes.end());
    std::vector<int> segs(segment_ids.begin(), segment_ids.end());

    return qe::apply_door_lookback(conn_counts, unit_rows, lbs, aggregate_mean, weighting_linear, segs);
}

//' Door EMA smoothing across units
//' @param conn_counts Numeric matrix of connection counts
//' @param unit_row_indices List of integer vectors of row indices
//' @param alpha Smoothing factor
//' @param segment_ids Optional integer vector of segment IDs
//' @export
// [[Rcpp::export]]
arma::mat door_ema(
    arma::mat conn_counts,
    List unit_row_indices,
    double alpha = 0.1,
    IntegerVector segment_ids = IntegerVector()
) {
    std::vector<std::vector<int>> unit_rows;
    unit_rows.reserve(unit_row_indices.size());
    for (int i = 0; i < unit_row_indices.size(); ++i) {
        IntegerVector iv = unit_row_indices[i];
        unit_rows.push_back(std::vector<int>(iv.begin(), iv.end()));
    }
    std::vector<int> segs(segment_ids.begin(), segment_ids.end());

    return qe::apply_door_ema(conn_counts, unit_rows, alpha, segs);
}

// =============================================================================
// Trajectory Curve Fitting & Differential Geometry
// =============================================================================

//' Fit 2D parametric polynomial curve with LOOCV-2D or AIC degree selection
//' @param points Numeric matrix (n x 2) of trajectory coordinates
//' @param t Numeric vector of time values (length n)
//' @param max_degree Maximum polynomial degree (default 3)
//' @param fixed_degree Exact degree to use when >= 1
//' @param criterion Criterion for degree selection: "loocv" or "aic"
//' @param basis Polynomial basis used for fitting: "orthogonal" or "raw"
//' @export
// [[Rcpp::export]]
List fit_trajectory_poly(
    arma::mat points,
    NumericVector t = NumericVector(),
    int max_degree = 3,
    int fixed_degree = 0,
    std::string criterion = "loocv",
    std::string basis = "orthogonal"
) {
    arma::vec t_vec;
    if (t.size() > 0) {
        t_vec = arma::vec(t.begin(), t.size());
    }
    qe::PolyCurveFit fit = qe::fit_poly_loocv2d(points, t_vec, max_degree, fixed_degree, criterion, basis);

    return List::create(
        Named("degree")        = fit.degree,
        Named("coeffs_x")      = fit.coeffs_x,
        Named("coeffs_y")      = fit.coeffs_y,
        Named("basis")         = fit.basis,
        Named("basis_alpha")   = fit.basis_alpha,
        Named("basis_norm2")   = fit.basis_norm2,
        Named("cv_error")      = fit.cv_error,
        Named("aic")           = fit.aic,
        Named("t")             = fit.t,
        Named("fitted_points") = fit.fitted
    );
}

//' Evaluate 2D parametric polynomial curve at specified time points
//' @param coeffs_x Polynomial coefficients for X
//' @param coeffs_y Polynomial coefficients for Y
//' @param t_eval Numeric vector of time evaluation points
//' @export
// [[Rcpp::export]]
arma::mat eval_trajectory_curve(
    arma::vec coeffs_x,
    arma::vec coeffs_y,
    arma::vec t_eval
) {
    return qe::eval_poly_curve(coeffs_x, coeffs_y, t_eval);
}

//' Evaluate trajectory differential geometry (velocities, speed, curvature, turns)
//' @param coeffs_x Polynomial coefficients for X
//' @param coeffs_y Polynomial coefficients for Y
//' @param t_eval Numeric vector of time evaluation points
//' @export
// [[Rcpp::export]]
List eval_trajectory_derivatives(
    arma::vec coeffs_x,
    arma::vec coeffs_y,
    arma::vec t_eval
) {
    qe::TrajectoryDerivatives d = qe::eval_trajectory_derivatives(coeffs_x, coeffs_y, t_eval);
    return List::create(
        Named("t")            = d.t,
        Named("vx")           = d.vx,
        Named("vy")           = d.vy,
        Named("speed")        = d.speed,
        Named("ax")           = d.ax,
        Named("ay")           = d.ay,
        Named("heading_rate") = d.heading_rate,
        Named("curvature")    = d.curvature
    );
}

// =============================================================================
// Trajectory Distance & Following Dynamics
// =============================================================================

//' Integrated Euclidean distance between two trajectory polynomial curves
//' @param coeffs_ax Polynomial coeffs for Curve A X
//' @param coeffs_ay Polynomial coeffs for Curve A Y
//' @param coeffs_bx Polynomial coeffs for Curve B X
//' @param coeffs_by Polynomial coeffs for Curve B Y
//' @param t_start Integration start (default 0.0)
//' @param t_end Integration end (default 1.0)
//' @export
// [[Rcpp::export]]
double integrated_trajectory_distance(
    arma::vec coeffs_ax,
    arma::vec coeffs_ay,
    arma::vec coeffs_bx,
    arma::vec coeffs_by,
    double t_start = 0.0,
    double t_end = 1.0
) {
    return qe::integrated_curve_distance(coeffs_ax, coeffs_ay, coeffs_bx, coeffs_by, t_start, t_end);
}

//' Lagged curve distance between follower and leader
//' @param coeffs_fol_x Polynomial coeffs for Follower X
//' @param coeffs_fol_y Polynomial coeffs for Follower Y
//' @param coeffs_ldr_x Polynomial coeffs for Leader X
//' @param coeffs_ldr_y Polynomial coeffs for Leader Y
//' @param lag Time lag in [0, 1)
//' @export
// [[Rcpp::export]]
double lagged_trajectory_distance(
    arma::vec coeffs_fol_x,
    arma::vec coeffs_fol_y,
    arma::vec coeffs_ldr_x,
    arma::vec coeffs_ldr_y,
    double lag = 0.0
) {
    return qe::lagged_curve_distance(coeffs_fol_x, coeffs_fol_y, coeffs_ldr_x, coeffs_ldr_y, lag);
}

//' Pairwise integrated trajectory distance matrix
//' @param all_coeffs_x List of X coefficient vectors
//' @param all_coeffs_y List of Y coefficient vectors
//' @export
// [[Rcpp::export]]
arma::mat pairwise_trajectory_distance(
    List all_coeffs_x,
    List all_coeffs_y
) {
    std::vector<arma::vec> cx;
    std::vector<arma::vec> cy;
    cx.reserve(all_coeffs_x.size());
    cy.reserve(all_coeffs_y.size());
    for (int i = 0; i < all_coeffs_x.size(); ++i) {
        NumericVector nx = all_coeffs_x[i];
        NumericVector ny = all_coeffs_y[i];
        cx.push_back(arma::vec(nx.begin(), nx.size()));
        cy.push_back(arma::vec(ny.begin(), ny.size()));
    }
    return qe::pairwise_trajectory_distance_matrix(cx, cy);
}

//' Signed turn-lag distance between two agents at specific lag delta
//' @param pts_a Points matrix for agent A (n_a x 2)
//' @param pts_b Points matrix for agent B (n_b x 2)
//' @param times_a Turn times for agent A
//' @param times_b Turn times for agent B
//' @param delta Signed lag in turns
//' @export
// [[Rcpp::export]]
List signed_turn_lag(
    arma::mat pts_a,
    arma::mat pts_b,
    arma::vec times_a,
    arma::vec times_b,
    int delta
) {
    auto [mean_d, count] = qe::signed_turn_lag_distance(pts_a, pts_b, times_a, times_b, delta);
    return List::create(
        Named("mean_distance") = mean_d,
        Named("valid_count")   = count
    );
}

//' Sweep signed turn lags to find optimal leader-follower lag
//' @param pts_a Points matrix for agent A (n_a x 2)
//' @param pts_b Points matrix for agent B (n_b x 2)
//' @param times_a Turn times for agent A
//' @param times_b Turn times for agent B
//' @param max_lag Maximum turn lag to evaluate (default 15)
//' @export
// [[Rcpp::export]]
List sweep_signed_turn_lags(
    arma::mat pts_a,
    arma::mat pts_b,
    arma::vec times_a,
    arma::vec times_b,
    int max_lag = 15
) {
    qe::SignedTurnLagResult res = qe::best_signed_turn_lag(pts_a, pts_b, times_a, times_b, max_lag);
    return List::create(
        Named("best_lag")           = res.best_lag,
        Named("min_mean_distance")  = res.min_mean_distance,
        Named("lags")               = res.lags,
        Named("mean_distances")     = res.mean_distances,
        Named("valid_counts")       = res.valid_counts
    );
}

//' Distance-distance matrix correlation for stability analysis
//' @param X Numeric matrix
//' @param Y Numeric matrix
//' @export
// [[Rcpp::export]]
double dist_dist_correlation(
    arma::mat X,
    arma::mat Y
) {
    return qe::dist_dist_correlation(X, Y);
}
