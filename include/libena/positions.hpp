/**
 * @file positions.hpp
 * @brief ENA node-position solvers for undirected and directed networks, and
 *        the points-to-centroids correlation (ena_correlation).
 *
 * Split out of the former libqe/modeling.hpp.
 */
#ifndef LIBENA_POSITIONS_HPP
#define LIBENA_POSITIONS_HPP

#include <armadillo>
#include <cmath>
#include <stdexcept>
#include <libqe/linalg_fallback.hpp>  // qe::linalg::solve_spd
#include <libqe/stats.hpp>            // normal_quantile

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

} // namespace qe

#endif // LIBENA_POSITIONS_HPP
