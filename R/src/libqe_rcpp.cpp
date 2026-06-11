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

//' Pearson correlation with CI between ENA points and centroids
//' @param points  Numeric matrix (units x dims)
//' @param centroids Numeric matrix (units x dims)
//' @param conf_level Confidence level (default 0.95)
//' @export
// [[Rcpp::export]]
arma::mat ena_correlation(arma::mat points, arma::mat centroids,
                              double conf_level = 0.95) {
    return qe::ena_correlation(points, centroids, conf_level);
}

//' Least-squares node positions for undirected ENA
//' @param adj_mats Numeric matrix of line weights (units x connections)
//' @param t        Numeric matrix of rotated points (units x dims)
//' @param num_dims Number of dimensions
//' @return List with nodes, centroids, weights, points
//' @export
// [[Rcpp::export]]
List node_positions(arma::mat adj_mats, arma::mat t, int num_dims) {
    if (!adj_mats.is_finite() || !t.is_finite())
        stop("node_positions: input matrices must not contain NaN or Inf - "
             "filter or impute rows with non-finite values before calling");
    qe::NodePositions r = qe::node_positions(adj_mats, t, num_dims);
    return List::create(
        _("nodes")     = r.nodes,
        _("centroids") = r.centroids,
        _("weights")   = r.weights,
        _("points")    = r.points
    );
}

//' Least-squares node positions for directed ENA
//' @param line_weights Numeric matrix (units x connections)
//' @param points       Numeric matrix of rotated points (units x dims)
//' @param num_dims     Number of dimensions
//' @return List with nodes, centroids, weights, points
//' @export
// [[Rcpp::export]]
List directed_node_positions(arma::mat line_weights, arma::mat points,
                                 int num_dims) {
    if (!line_weights.is_finite() || !points.is_finite())
        stop("directed_node_positions: input matrices must not contain NaN or Inf - "
             "filter or impute rows with non-finite values before calling");
    qe::NodePositions r = qe::directed_node_positions(line_weights, points, num_dims);
    return List::create(
        _("nodes")     = r.nodes,
        _("centroids") = r.centroids,
        _("weights")   = r.weights,
        _("points")    = r.points
    );
}

//' Directed node positions with paired ground+response rows combined
//' @param line_weights Numeric matrix (units x connections)
//' @param points       Numeric matrix of rotated points (units x dims)
//' @param num_dims     Number of dimensions
//' @return List with nodes, centroids, weights, points
//' @export
// [[Rcpp::export]]
List directed_node_positions_combine_pairs(arma::mat line_weights,
                                               arma::mat points,
                                               int num_dims) {
    if (!line_weights.is_finite() || !points.is_finite())
        stop("directed_node_positions_combine_pairs: input matrices must not contain NaN or Inf - "
             "filter or impute rows with non-finite values before calling");
    qe::NodePositions r = qe::directed_node_positions(
        line_weights, points, num_dims, true);
    return List::create(
        _("nodes")     = r.nodes,
        _("centroids") = r.centroids,
        _("weights")   = r.weights,
        _("points")    = r.points
    );
}

// =============================================================================
// Accumulation
// =============================================================================

//' Core connection matrix for one ground+response pair
//' @param ground          Numeric row vector of ground (context) code values
//' @param response        Numeric row vector of response code values
//' @param response_weight Scalar weight applied to the response self-connection
//' @param ordered         TRUE = directed; FALSE = undirected
//' @export
// [[Rcpp::export]]
arma::mat connection_matrix(arma::rowvec ground, arma::rowvec response,
                                         double response_weight = 1.0,
                                         bool ordered = true) {
    return qe::connection_matrix(ground, response, response_weight, ordered);
}

//' Traditional stanza-window accumulation (rENA model)
//'
//' For each row k in a single conversation's code matrix, accumulates
//' connections over a back/forward window.
//'
//' \strong{Undirected} (\code{ordered = FALSE}, default): upper-triangle
//' co-occurrences with back/forward-reference corrections.
//' Returns \code{n_rows × choose_two(n_codes)}.
//'
//' \strong{Directed} (\code{ordered = TRUE}): focal row k as response,
//' sum of prior window rows as ground, using the directed connection formula.
//' \code{window_forward} is ignored. Returns \code{n_rows × n_codes²}.
//'
//' @param codes          Numeric matrix (rows = lines, cols = codes) for ONE conversation
//' @param window_back    Number of prior lines in window (default 1); use .Machine$integer.max for Inf
//' @param window_forward Number of subsequent lines (default 0; ignored when ordered = TRUE)
//' @param binary         If TRUE, binarise non-zero connection counts
//' @param ordered        If TRUE, return directed n² output; FALSE (default) returns upper-tri
//' @return Undirected: (n_rows × choose_two(n_codes)); directed: (n_rows × n_codes²)
//' @export
// [[Rcpp::export]]
arma::mat accumulate_stanza(arma::mat codes,
                            int window_back    = 1,
                            int window_forward = 0,
                            bool binary        = true,
                            bool ordered       = false) {
    return qe::accumulate_stanza(codes, window_back, window_forward, binary, ordered);
}

//' Compute a column-major linear index into a multi-dimensional array
//'
//' Equivalent to tma's flat_index(). Throws if lengths of `indices`
//' and `dims` differ.
//'
//' @param indices 0-based integer vector of per-dimension indices
//' @param dims    Integer vector of array dimensions
//' @return Scalar integer linear index
//' @export
// [[Rcpp::export]]
int flat_index(std::vector<int> indices, std::vector<int> dims) {
    return qe::flat_index(indices, dims);
}

//' Per-row upper-triangle co-occurrence matrix
//'
//' For each row in \code{codes}, computes the pairwise code products
//' (upper-triangle) and optionally binarizes. Output has
//' \code{choose(n_codes, 2)} columns.
//'
//' @param codes  Numeric matrix (rows = observations, cols = codes)
//' @param binary If TRUE, binarise non-zero co-occurrences (default TRUE)
//' @return Numeric matrix (n_rows x choose_two(n_codes))
//' @export
// [[Rcpp::export]]
arma::mat row_connections(arma::mat codes, bool binary = true) {
    return qe::row_connections(codes, binary);
}

//' Rolling backward window sum of a code matrix
//'
//' For each row k, sums the raw code values over rows
//' \code{[max(0, k - window_size + 1), k]}. Returns a matrix of the same
//' shape as \code{codes}. \code{window_size <= 0} is treated as 1
//' (current row only).
//'
//' @param codes       Numeric matrix (rows = observations, cols = codes)
//' @param window_size Number of rows to look back (default 1)
//' @return Numeric matrix (same dimensions as \code{codes})
//' @export
// [[Rcpp::export]]
arma::mat rolling_window_sum(arma::mat codes, int window_size = 1) {
    return qe::rolling_window_sum(codes, window_size);
}

//' Ground/response accumulation for one unit (tma model)
//'
//' @param codes      Numeric matrix for the full context (n_rows x n_codes)
//' @param unit_rows  0-based integer vector of rows belonging to this unit
//' @param decay_fn   R function mapping a numeric distance vector to weights
//' @param ordered    TRUE = directed; FALSE = undirected (upper-tri)
//' @return Numeric vector of connection counts
//' @export
// [[Rcpp::export]]
arma::rowvec accumulate_unit(arma::mat codes, std::vector<int> unit_rows,
                                  Function decay_fn, bool ordered = false) {
    auto cpp_decay = [&](arma::vec distances) -> arma::vec {
        NumericVector d = wrap(distances);
        NumericVector w = decay_fn(d);
        return as<arma::vec>(w);
    };
    return qe::accumulate_unit(codes, unit_rows, cpp_decay, ordered);
}

//' Ground/response accumulation for one unit — returns unit vector and per-row matrix
//'
//' Equivalent to tma's accumulate_network() but without R-env-var setup.
//' The decay function receives a distance vector (response=0, older rows
//' have larger values) and returns a weight vector of the same length.
//'
//' @param codes      Numeric matrix (n_rows x n_codes)
//' @param unit_rows  0-based integer vector of rows belonging to this unit
//' @param decay_fn   R function(distances) -> weights
//' @param ordered    TRUE = directed (full p^2); FALSE = undirected (upper-tri)
//' @return List with `networks` (vector) and `row_networks` (matrix)
//' @export
// [[Rcpp::export]]
List accumulate_unit_with_rows(arma::mat codes, std::vector<int> unit_rows,
                                   Function decay_fn, bool ordered = false) {
    auto cpp_decay = [&](int unit_row, arma::uvec ground_indices) -> arma::vec {
        arma::vec dists(ground_indices.n_elem);
        for (arma::uword k = 0; k < ground_indices.n_elem; ++k)
            dists[k] = static_cast<double>(unit_row - ground_indices[k]);
        return as<arma::vec>(wrap(decay_fn(wrap(dists))));
    };
    qe::UnitNetworks r = qe::accumulate_unit_with_rows(codes, unit_rows, cpp_decay, ordered);
    return List::create(
        _("networks")     = r.networks,
        _("row_networks") = r.row_networks
    );
}

//' Tensor-based multi-modal accumulation for one unit (tma model)
//'
//' Pure-C++ port of tma's apply_tensor().  Accepts 0-based index vectors
//' for sender/receiver/mode dimensions and a pre-converted integer context
//' lookup matrix.
//'
//' @param tensor         Numeric vector (column-major flat tensor)
//' @param dims           Integer vector of tensor dimensions
//' @param dims_sender    0-based sender axis indices
//' @param dims_receiver  0-based receiver axis indices
//' @param dims_mode      0-based mode axis indices
//' @param context_lookup Integer matrix (n_context_rows x n_factors), 0-based
//' @param unit_rows      0-based response-row indices for this unit
//' @param codes          Numeric matrix (n_context_rows x n_codes)
//' @param times          Numeric vector of timestamps per context row
//' @param ordered        TRUE = directed; FALSE = undirected
//' @return List with `connection_counts` (vector) and `row_connection_counts` (matrix)
//' @export
// [[Rcpp::export]]
List apply_tensor(arma::vec tensor,
                     std::vector<int> dims,
                     std::vector<int> dims_sender,
                     std::vector<int> dims_receiver,
                     std::vector<int> dims_mode,
                     arma::imat context_lookup,
                     std::vector<int> unit_rows,
                     arma::mat codes,
                     arma::vec times,
                     bool ordered = true) {
    qe::TensorNetworks r = qe::apply_tensor_unit(
        tensor, dims, dims_sender, dims_receiver, dims_mode,
        context_lookup, unit_rows, codes, times, ordered);
    return List::create(
        _("connection_counts")     = r.connection_counts,
        _("row_connection_counts") = r.row_connection_counts
    );
}

// =============================================================================
// Rotation
// =============================================================================

// Pack a RotationResult into the list shape that matches rENA's ENARotationSet
// payload (rotation + eigenvalues + column names). Column names are attached
// to the rotation matrix as dimnames so downstream R code can index by them.
// Eigenvalues are returned as a plain numeric vector (not an Nx1 matrix) to
// match what rENA's `pcaResults$sdev^2` produces.
static List pack_rotation_result(const qe::RotationResult& r) {
    NumericMatrix rotation = wrap(r.rotation);
    CharacterVector col_names(r.column_names.begin(), r.column_names.end());
    rotation.attr("dimnames") = List::create(R_NilValue, col_names);
    NumericVector eigenvalues(r.eigenvalues.begin(), r.eigenvalues.end());
    return List::create(
        _("rotation")     = rotation,
        _("eigenvalues")  = eigenvalues,
        _("column_names") = col_names
    );
}

//' SVD rotation (matches prcomp(retx=F, scale=F, center=F, tol=0))
//'
//' Caller is responsible for centering upstream. Eigenvalues are stored as
//' \code{sdev^2} (variance) to match rENA's \code{ena.svd}.
//'
//' @param points Numeric matrix (n_units x n_dims)
//' @return List with \code{rotation} (n_dims x n_dims), \code{eigenvalues}
//'   (length n_dims, = sdev^2), and \code{column_names} ("SVD1", "SVD2", ...)
//' @export
// [[Rcpp::export]]
List ena_svd(arma::mat points) {
    if (!points.is_finite())
        stop("ena_svd: input matrix must not contain NaN or Inf - "
             "filter or impute rows with non-finite values before calling");
    return pack_rotation_result(qe::ena_svd(points));
}

//' Project a matrix onto the hyperplane orthogonal to a unit-norm axis
//'
//' Computes \code{data - (data \%*\% axis) \%*\% t(axis)}. The caller is
//' responsible for ensuring \code{axis} is unit-norm.
//'
//' @param data Numeric matrix (n_units x n_dims)
//' @param axis Numeric vector of length n_dims, unit-norm
//' @return Numeric matrix of the same shape as \code{data}
//' @export
// [[Rcpp::export]]
arma::mat deflate(arma::mat data, arma::vec axis) {
    return qe::deflate(data, axis);
}

//' Orthogonal SVD — orthonormalize named axes via QR, fill the rest from SVD
//'
//' Mirrors rENA's \code{orthogonal_svd()} in \code{ena.rotate.by.mean.R}:
//' the named axes in the output are the orthonormalized Q columns, not the
//' original \code{weights} columns. Use \code{complete_rotation} to keep
//' the named axes verbatim.
//'
//' @param data         Numeric matrix (n_units x n_dims)
//' @param weights      Numeric matrix (n_dims x k); columns are the named axes
//' @param named_labels Character vector of length k
//' @return List with \code{rotation}, \code{eigenvalues}, \code{column_names}
//' @export
// [[Rcpp::export]]
List orthogonal_svd(arma::mat data,
                        arma::mat weights,
                        std::vector<std::string> named_labels) {
    return pack_rotation_result(qe::orthogonal_svd(data, weights, named_labels));
}

//' Complete a rotation — keep named axes verbatim, fill remainder from SVD
//'
//' Mirrors the tail of \code{ena.rotate.by.generalized}: the named axes
//' appear in the output exactly as provided, and the trailing columns come
//' from an SVD of the data deflated by all named axes.
//'
//' @param data         Numeric matrix (n_units x n_dims)
//' @param named_axes   Numeric matrix (n_dims x k); columns must be unit-norm
//' @param named_labels Character vector of length k
//' @return List with \code{rotation}, \code{eigenvalues}, \code{column_names}
//' @export
// [[Rcpp::export]]
List complete_rotation(arma::mat data,
                           arma::mat named_axes,
                           std::vector<std::string> named_labels) {
    return pack_rotation_result(qe::complete_rotation(data, named_axes, named_labels));
}

//' Means rotation
//'
//' For each group pair, computes a normalized mean-difference axis on the
//' progressively-deflated data and finishes with \code{orthogonal_svd}.
//' The input is column-centered first, matching rENA's
//' \code{scale(data, scale=F, center=T)} at the top of \code{ena.rotate.by.mean}.
//'
//' Each element of \code{group_pairs} is a length-2 list \code{list(a, b)}
//' of 0-based row indices into \code{points}.
//'
//' @param points      Numeric matrix (n_units x n_dims)
//' @param group_pairs List of length k; each element is \code{list(a, b)}
//'   where \code{a} and \code{b} are 0-based integer index vectors
//' @return List with \code{rotation}, \code{eigenvalues}, \code{column_names}
//' @export
// [[Rcpp::export]]
List means_rotation(arma::mat points, List group_pairs) {
    std::vector<qe::GroupPair> pairs;
    pairs.reserve(group_pairs.size());
    for (R_xlen_t i = 0; i < group_pairs.size(); ++i) {
        List pair = group_pairs[i];
        if (pair.size() != 2) {
            stop("group_pairs[[%d]] must be a length-2 list(a, b)",
                 static_cast<int>(i + 1));
        }
        IntegerVector ra = pair[0];
        IntegerVector rb = pair[1];
        arma::uvec a(ra.size());
        arma::uvec b(rb.size());
        for (R_xlen_t j = 0; j < ra.size(); ++j) {
            a(j) = static_cast<arma::uword>(ra[j]);
        }
        for (R_xlen_t j = 0; j < rb.size(); ++j) {
            b(j) = static_cast<arma::uword>(rb[j]);
        }
        pairs.push_back({a, b});
    }
    return pack_rotation_result(qe::means_rotation(points, pairs));
}

// =============================================================================
// Generalized Means Rotation (GMR)
// =============================================================================

//' Generalized Means Rotation
//'
//' Computes a rotation axis that best represents a target variable's
//' contribution to the ENA point space, after controlling for covariates via
//' Lasso (coordinate-descent, k-fold CV).  Mirrors
//' \code{rENA::ena.rotate.by.generalized()}.
//'
//' The caller is responsible for building \code{x_model_matrix} (e.g. via
//' \code{model.matrix()}) and identifying \code{x1_cols} (0-based indices of
//' the target variable's columns, which receive \code{penalty_factor = 0}).
//' Categorical targets should be encoded as 0-based integers.
//'
//' @param V              Numeric matrix (n_units x n_connections) — ENA points.
//' @param x_model_matrix Numeric matrix (n_units x p) — model matrix for x axis.
//' @param x_target       Numeric vector length n — target variable (raw float
//'   for numeric; 0-based integer codes for categorical).
//' @param x1_cols        Integer vector of 0-based column indices in
//'   \code{x_model_matrix} that belong to the target variable (unpenalized).
//' @param x_categorical  Logical — TRUE if target is categorical.
//' @param x_n_groups     Integer — number of distinct groups (categorical only).
//' @param x_subset       Integer vector of 0-based row indices to subset for
//'   the x-axis GMR step (pass \code{integer(0)} to use all rows).
//' @param has_y          Logical — TRUE to compute a second GMR axis.
//' @param y_model_matrix Numeric matrix (n_units x p) — model matrix for y axis
//'   (ignored when \code{has_y = FALSE}).
//' @param y_target       Numeric vector — y-axis target (ignored when
//'   \code{has_y = FALSE}).
//' @param y1_cols        0-based column indices for y target (ignored when
//'   \code{has_y = FALSE}).
//' @param y_categorical  Logical (ignored when \code{has_y = FALSE}).
//' @param y_n_groups     Integer (ignored when \code{has_y = FALSE}).
//' @param n_lambda       Length of the Lasso lambda path (default 50).
//' @param k_folds        Cross-validation folds for lambda selection (default 5).
//' @param lasso_eps      \code{lambda_min = lasso_eps * lambda_max} (default 0.01).
//' @return List with \code{rotation}, \code{eigenvalues}, \code{column_names}.
//' @export
// [[Rcpp::export]]
List generalized_means_rotation(
    arma::mat V,
    arma::mat x_model_matrix,
    arma::vec x_target,
    IntegerVector x1_cols,
    bool x_categorical,
    int  x_n_groups,
    IntegerVector x_subset,
    bool has_y,
    arma::mat y_model_matrix,
    arma::vec y_target,
    IntegerVector y1_cols,
    bool y_categorical,
    int  y_n_groups,
    int  n_lambda  = 50,
    int  k_folds   = 5,
    double lasso_eps = 0.01
) {
    // ── Unpack x1_cols ───────────────────────────────────────────────────────
    arma::uvec x1(x1_cols.size());
    for (R_xlen_t i = 0; i < x1_cols.size(); ++i)
        x1(i) = static_cast<arma::uword>(x1_cols[i]);

    // ── Unpack x_subset (empty IntegerVector → use all rows) ─────────────────
    arma::uvec x_sub;
    if (x_subset.size() > 0) {
        x_sub.set_size(x_subset.size());
        for (R_xlen_t i = 0; i < x_subset.size(); ++i)
            x_sub(i) = static_cast<arma::uword>(x_subset[i]);
    }

    // ── Unpack y1_cols ───────────────────────────────────────────────────────
    arma::uvec y1(y1_cols.size());
    for (R_xlen_t i = 0; i < y1_cols.size(); ++i)
        y1(i) = static_cast<arma::uword>(y1_cols[i]);

    // ── Build params ─────────────────────────────────────────────────────────
    qe::GeneralizedRotationParams p;
    p.x_model_matrix = x_model_matrix;
    p.x_target       = x_target;
    p.x1_cols        = x1;
    p.x_categorical  = x_categorical;
    p.x_n_groups     = static_cast<arma::uword>(x_n_groups);
    p.x_subset       = x_sub;
    p.has_y          = has_y;
    p.y_model_matrix = y_model_matrix;
    p.y_target       = y_target;
    p.y1_cols        = y1;
    p.y_categorical  = y_categorical;
    p.y_n_groups     = static_cast<arma::uword>(y_n_groups);
    p.n_lambda       = n_lambda;
    p.k_folds        = k_folds;
    p.lasso_eps      = lasso_eps;

    return pack_rotation_result(qe::generalized_means_rotation(V, p));
}
