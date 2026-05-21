// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <libqe/libqe.hpp>

using namespace Rcpp;

// =============================================================================
// Adjacency utilities
// =============================================================================

//' Upper-triangle index pairs
//' @param len Number of codes (side length of square matrix)
//' @param row -1 = both rows, 0 = row indices only, 1 = col indices only
//' @export
// [[Rcpp::export]]
arma::umat lq_tri_indices(int len, int row = -1) {
    return qe::tri_indices(len, row);
}

//' Pairwise products → upper-triangle vector
//' @param v Numeric vector of code values
//' @export
// [[Rcpp::export]]
arma::rowvec lq_vector_to_upper_tri(arma::mat v) {
    return qe::vector_to_upper_tri(v);
}

//' Fold a directed (n*n) vector into an undirected upper-triangle vector
//' @param v Numeric vector of length n*n
//' @export
// [[Rcpp::export]]
arma::rowvec lq_directed_to_upper_tri(arma::vec v) {
    return qe::directed_to_upper_tri(v);
}

//' Flatten an adjacency matrix to a connection vector
//' @param x Numeric matrix
//' @param full TRUE = full n*n (directed); FALSE = upper triangle (undirected)
//' @export
// [[Rcpp::export]]
arma::rowvec lq_adjacency_matrix_to_vector(arma::mat x, bool full = true) {
    return qe::adjacency_matrix_to_vector(x, full);
}

//' Code-name pairs for upper-triangle positions ("A & B")
//' @param v Character vector of code names
//' @export
// [[Rcpp::export]]
std::vector<std::string> lq_svector_to_upper_tri(std::vector<std::string> v) {
    return qe::svector_to_upper_tri(v);
}

// =============================================================================
// Normalization
// =============================================================================

//' Row-wise L2 (sphere) normalization
//' @param m Numeric matrix
//' @export
// [[Rcpp::export]]
arma::mat lq_sphere_norm(arma::mat m) {
    return qe::sphere_norm(m);
}

//' Max-norm scaling (divide all rows by the largest row L2 norm)
//' @param m Numeric matrix
//' @export
// [[Rcpp::export]]
arma::mat lq_skip_sphere_norm(arma::mat m) {
    return qe::skip_sphere_norm(m);
}

// =============================================================================
// Modeling
// =============================================================================

//' Center data (subtract column means)
//' @param values Numeric matrix
//' @export
// [[Rcpp::export]]
arma::mat lq_center_data(arma::mat values) {
    return qe::center_data(values);
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
arma::mat lq_group_ci(arma::mat points, double conf_level = 0.95) {
    return qe::group_ci(points, conf_level);
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
arma::mat lq_outlier_ci(arma::mat points, double iqr_factor = 1.5) {
    return qe::outlier_ci(points, iqr_factor);
}

//' Pearson correlation with CI between ENA points and centroids
//' @param points  Numeric matrix (units x dims)
//' @param centroids Numeric matrix (units x dims)
//' @param conf_level Confidence level (default 0.95)
//' @export
// [[Rcpp::export]]
arma::mat lq_ena_correlation(arma::mat points, arma::mat centroids,
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
List lq_lws_lsq_positions(arma::mat adj_mats, arma::mat t, int num_dims) {
    qe::NodePositions r = qe::lws_lsq_positions(adj_mats, t, num_dims);
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
List lq_directed_node_positions(arma::mat line_weights, arma::mat points,
                                 int num_dims) {
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
List lq_directed_node_positions_ground_response(arma::mat line_weights,
                                                 arma::mat points,
                                                 int num_dims) {
    qe::NodePositions r = qe::directed_node_positions_ground_response(
        line_weights, points, num_dims);
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

//' Core adjacency matrix for one ground+response pair
//' @param ground          Numeric row vector of ground (context) code values
//' @param response        Numeric row vector of response code values
//' @param response_weight Scalar weight applied to the response self-connection
//' @param ordered         TRUE = directed; FALSE = undirected
//' @export
// [[Rcpp::export]]
arma::mat lq_calculate_adjacency_matrix(arma::rowvec ground, arma::rowvec response,
                                         double response_weight = 1.0,
                                         bool ordered = true) {
    return qe::calculate_adjacency_matrix(ground, response, response_weight, ordered);
}

//' Traditional stanza-window accumulation (rENA model)
//'
//' For each row k in a single conversation's code matrix, accumulates
//' co-occurrences over a back/forward window and returns the upper-triangle
//' connection vector.
//'
//' @param codes          Numeric matrix (rows = lines, cols = codes) for ONE conversation
//' @param window_back    Number of prior lines in window (default 1); use .Machine$integer.max for Inf
//' @param window_forward Number of subsequent lines (default 0)
//' @param binary         If TRUE, binarise non-zero connection counts
//' @return Numeric matrix (same n_rows, choose_two(n_codes) columns)
//' @export
// [[Rcpp::export]]
arma::mat lq_stanza_window(arma::mat codes,
                            int window_back    = 1,
                            int window_forward = 0,
                            bool binary        = true) {
    return qe::stanza_window(codes, window_back, window_forward, binary);
}

//' Compute a column-major linear index into a multi-dimensional array
//'
//' Equivalent to tma's calculate_1d_index(). Throws if lengths of `indices`
//' and `dims` differ.
//'
//' @param indices 0-based integer vector of per-dimension indices
//' @param dims    Integer vector of array dimensions
//' @return Scalar integer linear index
//' @export
// [[Rcpp::export]]
int lq_calculate_1d_index(std::vector<int> indices, std::vector<int> dims) {
    return qe::calculate_1d_index(indices, dims);
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
arma::mat lq_rows_to_co_occurrences(arma::mat codes, bool binary = true) {
    return qe::rows_to_co_occurrences(codes, binary);
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
arma::mat lq_rolling_window_sum(arma::mat codes, int window_size = 1) {
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
arma::rowvec lq_accumulate_unit(arma::mat codes, std::vector<int> unit_rows,
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
List lq_accumulate_unit_with_rows(arma::mat codes, std::vector<int> unit_rows,
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
List lq_apply_tensor(arma::vec tensor,
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
