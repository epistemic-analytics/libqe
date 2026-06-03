/** @file accumulation.hpp
 *  @brief Accumulation primitives shared between the rENA stanza-window model
 *         and the tma ground/response/tensor model.
 *
 *  Both models reduce to the same core connection_matrix() operation; they
 *  differ in how ground and response vectors are assembled from the raw data.
 */
#ifndef LIBQE_ACCUMULATION_HPP
#define LIBQE_ACCUMULATION_HPP

#include <armadillo>
#include <functional>
#include <vector>
#include "adjacency.hpp"

namespace qe {

/// @name Core adjacency math
/// @{

/** @brief Compute the connection matrix for one ground + response event pair.
 *
 *  @param[in] ground          Row vector of accumulated ground codes (length p).
 *  @param[in] response        Row vector of the focal (response) codes (length p).
 *  @param[in] response_weight Scalar weight applied to the response self-connection
 *                             term.  Defaults to 1.0.
 *  @param[in] ordered         When @c true, produces a directed (asymmetric) matrix:
 *                             @c ground→response cross-product plus
 *                             @c 0.5 * response_weight * response self-connection
 *                             (diagonal zeroed).
 *                             When @c false, produces a symmetric undirected matrix:
 *                             @c (response_weight * r⊗r) + g⊗r + r⊗g.
 *
 *  @returns A p × p connection matrix.
 *
 *  @note Equivalent to @c calculate_adjacency_matrix() in @c tma/code.cpp.
 */
inline arma::mat connection_matrix(
    arma::rowvec ground, arma::rowvec response,
    double response_weight = 1.0, bool ordered = true
) {
    arma::mat resp_self = response.t() * response;
    arma::mat g_by_r   = ground.t() * response;

    if (ordered) {
        resp_self.diag().zeros();
        return g_by_r + (0.5 * response_weight * resp_self);
    }
    return (response_weight * resp_self) + g_by_r + (response.t() * ground);
}

/// @}

/// @name Traditional stanza-window accumulation (rENA model)
/// @{

/** @brief Accumulate connection vectors for every row in one conversation using
 *         a stanza window.
 *
 *  For each focal row @c k in @p codes the function assembles a ground context
 *  from the surrounding window and calls connection_matrix().
 *
 *  @par Ordering semantics
 *  - @b Undirected (@p ordered = @c false, default — rENA stanza model): the
 *    window spans [@c k - window_back, @c k + window_forward]; the upper-triangle
 *    outer-product (code_connections) is computed and back/forward-reference
 *    corrections are subtracted.  Returns a matrix of shape
 *    @c n_rows × choose_two(n_codes).
 *  - @b Directed (@p ordered = @c true): focal row @c k is the response; the
 *    sum of prior rows [@c earliest, @c k−1] is the ground.
 *    connection_matrix() is called with @p ordered = @c true and the result is
 *    vectorised.  @p window_forward is ignored (future rows cannot be causal
 *    ground context).  Returns a matrix of shape @c n_rows × n_codes².
 *
 *  @param[in] codes          Code matrix for one conversation (n_rows × n_codes).
 *  @param[in] window_back    Number of prior rows included in the window
 *                            (1 = current row only; INT_MAX = all prior rows).
 *  @param[in] window_forward Number of future rows included in the window
 *                            (ignored when @p ordered is @c true).
 *  @param[in] binary         When @c true, clamp all positive entries to 1.
 *  @param[in] ordered        When @c true, use directed accumulation; when
 *                            @c false, use undirected stanza-window accumulation.
 *
 *  @returns Connection matrix (n_rows × connection_vector_length) where
 *           connection_vector_length is choose_two(n_codes) for undirected or
 *           n_codes² for directed.
 *
 *  @note Equivalent to @c ref_window_df() in @c rENA/ena.cpp (undirected case).
 */
inline arma::mat accumulate_stanza(
    arma::mat codes,
    int window_back    = 1,
    int window_forward = 0,
    bool binary        = true,
    bool ordered       = false
) {
    int n_rows  = codes.n_rows;
    int n_codes = codes.n_cols;
    const int INT_MAX_VAL = std::numeric_limits<int>::max();

    // Shared helper: earliest row index for focal row k
    auto get_earliest = [&](int row) -> int {
        if (window_back == INT_MAX_VAL || window_back == 0)
            return (window_back == 0) ? row : 0;
        return std::max(0, row - (window_back - 1));
    };

    if (ordered) {
        arma::mat out(n_rows, n_codes * n_codes, arma::fill::zeros);
        for (int row = 0; row < n_rows; row++) {
            int earliest = get_earliest(row);
            arma::rowvec response = codes.row(row);
            arma::rowvec ground(n_codes, arma::fill::zeros);
            if (row > earliest)
                ground = arma::sum(codes.rows(earliest, row - 1));
            out.row(row) = arma::vectorise(
                connection_matrix(ground, response, 1.0, true)).t();
        }
        if (binary) out.elem(arma::find(out > 0)).ones();
        return out;
    }

    // Undirected: existing rENA stanza-window logic
    int n_tri = choose_two(n_codes);
    arma::mat out(n_rows, n_tri, arma::fill::zeros);

    for (int row = 0; row < n_rows; row++) {
        int earliest = get_earliest(row);

        int latest = row;
        if (window_forward == INT_MAX_VAL || row + window_forward >= n_rows) {
            latest = n_rows - 1;
        } else if (window_forward > 0) {
            latest = std::min(n_rows - 1, row + window_forward);
        }

        arma::mat window_rows = codes.rows(earliest, latest);
        arma::mat summed      = arma::sum(window_rows);
        arma::rowvec to_ut    = code_connections(summed);

        // Back-reference correction
        int win_rows  = latest - earliest + 1;
        if (win_rows > 0 && window_back > 1 && row - 1 >= 0) {
            int head_rows = win_rows - 1 - window_forward;
            if (head_rows > 0) {
                arma::mat refs    = window_rows.head_rows(head_rows);
                arma::mat ref_sum = arma::sum(refs);
                to_ut -= code_connections(ref_sum);
            }
        }

        // Forward-reference correction
        if (window_forward > 0 && latest <= n_rows - 1) {
            int tail_rows = latest - row;
            if (tail_rows > 0) {
                arma::mat refs    = window_rows.tail_rows(tail_rows);
                arma::mat ref_sum = arma::sum(refs);
                to_ut -= code_connections(ref_sum);
            }
        }

        out.row(row) = to_ut;
    }

    if (binary) out.elem(arma::find(out > 0)).ones();
    return out;
}

/// @}

/// @name Ground/response accumulation (tma model)
/// @{

/** @brief Accumulate connections for a single unit from its context rows.
 *
 *  For every response row belonging to this unit, the function computes a
 *  weighted ground vector from all preceding context rows and calls
 *  connection_matrix().
 *
 *  @par Ordering semantics
 *  When @p ordered is @c false (default), the result is folded into the
 *  upper-triangle representation (length choose_two(p)).  When @p ordered is
 *  @c true, the full directed p² vector is returned.
 *
 *  @param[in] codes      Full context matrix visible to this unit (n × p).
 *  @param[in] unit_rows  0-based indices of the rows that belong to this unit.
 *  @param[in] decay_fn   Callback with signature
 *                        @c arma::vec(arma::vec distances) that maps a vector
 *                        of distances (0 = current row, 1 = one step back, …)
 *                        to a vector of scalar weights; default behaviour is a
 *                        simple rectangular window.
 *  @param[in] ordered    When @c true, return a directed flat vector (length p²);
 *                        when @c false, return an undirected upper-triangle flat
 *                        vector (length choose_two(p)).
 *
 *  @returns Flat connection vector of length choose_two(p) (undirected) or p²
 *           (directed).
 *
 *  @note Equivalent to @c accumulate_network() in @c tma/code.cpp, with the R
 *        function callback replaced by @c std::function.
 */
inline arma::rowvec accumulate_unit(
    const arma::mat& codes,
    const std::vector<int>& unit_rows,
    std::function<arma::vec(arma::vec)> decay_fn,
    bool ordered = false
) {
    int code_cnt = codes.n_cols;
    arma::mat g_w_vec(code_cnt, code_cnt, arma::fill::zeros);

    for (int unit_row : unit_rows) {
        if (unit_row == 0) continue;

        arma::rowvec response = codes.row(unit_row);

        // Ground rows: everything from 0 up to (and including) unit_row
        arma::uvec ground_indices = arma::regspace<arma::uvec>(0, 1, unit_row);
        arma::vec  distances(ground_indices.n_elem);
        for (arma::uword k = 0; k < ground_indices.n_elem; k++)
            distances[k] = static_cast<double>(unit_row - ground_indices[k]);

        arma::vec decay_weights = decay_fn(distances);

        arma::mat ground_codes   = codes.rows(ground_indices);
        arma::mat weighted       = ground_codes.each_col() % decay_weights;
        arma::rowvec g_summed    = arma::sum(weighted);

        // Exclude the response row's own contribution from the ground
        arma::rowvec g_no_resp   = g_summed - weighted.tail_rows(1).row(0);

        g_w_vec += connection_matrix(g_no_resp, response, 1.0, ordered);
    }

    if (!ordered) return fold_directed_network(arma::vectorise(g_w_vec));
    return arma::vectorise(g_w_vec).t();
}

/// @}

/// @name Extended ground/response accumulation — returns per-row connection data
/// @{

/** @brief Result of accumulate_unit_with_rows().
 *
 *  Bundles the unit-level flat connection vector together with the per-response-row
 *  full p² connection matrices needed by tma's @c accumulate_network().
 */
struct UnitNetworks {
    arma::rowvec networks;      ///< Flat connection vector (choose_two(p) or p²).
    arma::mat    row_networks;  ///< Per-response-row full p² matrix (n_unit_rows × p²).
};

/** @brief Like accumulate_unit() but also returns the per-response-row connection
 *         matrix needed by tma's @c accumulate_network().
 *
 *  @param[in] codes      Full context matrix visible to this unit (n × p).
 *  @param[in] unit_rows  0-based indices of the rows that belong to this unit.
 *  @param[in] decay_fn   Callback with signature
 *                        @c arma::vec(int unit_row, arma::uvec ground_indices)
 *                        that returns a weight vector of length
 *                        @c ground_indices.n_elem.  The two-argument form lets
 *                        callers (e.g. the tma Rcpp wrapper) set R environment
 *                        variables before invoking the actual R decay function,
 *                        without any R-specific code leaking into libqe.
 *  @param[in] ordered    When @c true, return directed flat vectors (length p²);
 *                        when @c false, return undirected upper-triangle flat
 *                        vectors (length choose_two(p)).
 *
 *  @returns A UnitNetworks struct containing the aggregated connection vector
 *           and the per-response-row connection matrices.
 *
 *  @note Equivalent to @c accumulate_network() in @c tma/code.cpp (extended form).
 */
inline UnitNetworks accumulate_unit_with_rows(
    const arma::mat&                              codes,
    const std::vector<int>&                       unit_rows,
    std::function<arma::vec(int, arma::uvec)>     decay_fn,
    bool ordered = false
) {
    int code_cnt    = codes.n_cols;
    int n_unit_rows = unit_rows.size();

    arma::mat g_w_mat(code_cnt, code_cnt, arma::fill::zeros);
    arma::mat row_networks(n_unit_rows, code_cnt * code_cnt, arma::fill::zeros);

    for (int i = 0; i < n_unit_rows; ++i) {
        int unit_row = unit_rows[i];

        arma::uvec ground_indices = arma::regspace<arma::uvec>(0, 1, unit_row);
        arma::vec  decay_weights  = decay_fn(unit_row, ground_indices);

        arma::mat    ground_codes = codes.rows(ground_indices);
        arma::mat    weighted     = ground_codes.each_col() % decay_weights;
        arma::rowvec g_summed     = arma::sum(weighted);
        arma::rowvec g_no_resp    = g_summed - weighted.tail_rows(1).row(0);

        arma::rowvec response = codes.row(unit_row);
        arma::mat    conn     = connection_matrix(g_no_resp, response, 1.0, ordered);
        g_w_mat += conn;
        row_networks.row(i) = arma::vectorise(conn).t();
    }

    UnitNetworks result;
    if (!ordered)
        result.networks = fold_directed_network(arma::vectorise(g_w_mat));
    else
        result.networks = arma::vectorise(g_w_mat).t();
    result.row_networks = row_networks;
    return result;
}

/// @}

/// @name Tensor-based multi-modal accumulation (tma model)
/// @{

/** @brief Compute the linear index into a column-major multi-dimensional array.
 *
 *  @param[in] indices  Per-dimension index values (0-based).
 *  @param[in] dims     Size of each dimension.
 *
 *  @returns The scalar column-major flat index.
 *
 *  @note Equivalent to @c flat_index() in @c tma/code.cpp.
 */
inline int flat_index(const std::vector<int>& indices,
                               const std::vector<int>& dims) {
    if (indices.size() != dims.size())
        throw std::invalid_argument("Number of indices must match number of dimensions.");
    size_t linear = 0, stride = 1;
    for (size_t v = 0; v < indices.size(); ++v) {
        linear += static_cast<size_t>(indices[v]) * stride;
        stride *= static_cast<size_t>(dims[v]);
    }
    return static_cast<int>(linear);
}

/** @brief Result of apply_tensor_unit().
 *
 *  Bundles the unit-level aggregated connection vector together with the
 *  per-response-row connection matrices used by the tma tensor accumulation.
 */
struct TensorNetworks {
    arma::rowvec connection_counts;      ///< Unit-level flat vector (p²).
    arma::mat    row_connection_counts;  ///< Per-response-row connection matrix (n_unit_rows × p²).
};

/** @brief Pure-C++ port of the tma @c apply_tensor() inner logic.
 *
 *  For each response row in @p unit_rows, the function looks up window and
 *  weight values from the tensor, collects all ground rows that fall within the
 *  response row's window, and calls connection_matrix() to accumulate the
 *  connection counts.
 *
 *  @param[in] tensor          Flat column-major array containing window and
 *                             weight values indexed by the factor dimensions.
 *  @param[in] dims            Sizes of each dimension of @p tensor.
 *  @param[in] dims_sender     Tensor axis indices corresponding to sender factors.
 *  @param[in] dims_receiver   Tensor axis indices corresponding to receiver
 *                             factors; these axes are overridden to the response
 *                             row's values when looking up window sizes for
 *                             ground rows.
 *  @param[in] dims_mode       Tensor axis indices corresponding to mode factors.
 *  @param[in] context_lookup  Integer matrix (n_context_rows × n_factors, 0-based)
 *                             mapping each context row to its factor level indices.
 *  @param[in] unit_rows       0-based response-row indices for this unit.
 *  @param[in] codes           Full context code matrix (n_context_rows × n_codes).
 *  @param[in] times           Timestamp associated with each context row
 *                             (length n_context_rows).
 *  @param[in] ordered         When @c true, produce a directed full p² result;
 *                             when @c false, produce an undirected upper-triangle
 *                             result.
 *
 *  @returns A TensorNetworks struct containing the unit-level connection counts
 *           and the per-response-row connection counts.
 *
 *  @note Equivalent to the inner loop of @c apply_tensor() in @c tma/code.cpp.
 */
inline TensorNetworks apply_tensor_unit(
    const arma::vec&        tensor,
    const std::vector<int>& dims,
    const std::vector<int>& dims_sender,
    const std::vector<int>& dims_receiver,
    const std::vector<int>& dims_mode,
    const arma::imat&       context_lookup,
    const std::vector<int>& unit_rows,
    const arma::mat&        codes,
    const arma::vec&        times,
    bool ordered = true
) {
    const int  WINDOW_DIM  = 1;
    const int  WEIGHT_DIM  = 0;
    const bool IS_DEFAULT  = (dims.size() == 1 && dims[0] == 2);

    int code_cnt    = codes.n_cols;
    int n_unit_rows = static_cast<int>(unit_rows.size());
    int ctx_cols    = static_cast<int>(context_lookup.n_cols);

    arma::mat g_w_mat(code_cnt, code_cnt, arma::fill::zeros);
    arma::mat row_conn(n_unit_rows, code_cnt * code_cnt, arma::fill::zeros);

    int response_win    = 0;
    int response_weight = 0;
    if (IS_DEFAULT) {
        response_win    = static_cast<int>(tensor[1]);
        response_weight = static_cast<int>(tensor[0]);
    }

    for (int i = 0; i < n_unit_rows; ++i) {
        int    ri            = unit_rows[i];
        double response_time = times[ri];

        std::vector<int> resp_ctx(ctx_cols + 1);
        for (int j = 0; j < ctx_cols; ++j) resp_ctx[j] = context_lookup(ri, j);
        resp_ctx[ctx_cols] = WINDOW_DIM;

        if (!IS_DEFAULT)
            response_win = static_cast<int>(tensor[flat_index(resp_ctx, dims)]);

        std::vector<int>    gri_v;
        std::vector<double> grw_v;
        arma::rowvec g_ws(code_cnt, arma::fill::zeros);

        if (ri > 0) {
            for (int gr = ri - 1; gr >= 0; --gr) {
                std::vector<int> row_v(ctx_cols + 1);
                for (int j = 0; j < ctx_cols; ++j) row_v[j] = context_lookup(gr, j);
                row_v[ctx_cols] = WINDOW_DIM;
                for (int dim : dims_receiver) row_v[dim] = resp_ctx[dim];

                double row_win = static_cast<double>(response_win);
                if (!IS_DEFAULT)
                    row_win = tensor[flat_index(row_v, dims)];

                if (times[gr] + row_win > response_time) {
                    gri_v.push_back(gr);
                    row_v[ctx_cols] = WEIGHT_DIM;
                    double row_wgt = static_cast<double>(response_weight);
                    if (!IS_DEFAULT)
                        row_wgt = tensor[flat_index(row_v, dims)];
                    grw_v.push_back(row_wgt);
                }
            }

            if (!gri_v.empty()) {
                arma::uvec gri_u(gri_v.size());
                for (size_t k = 0; k < gri_v.size(); ++k) gri_u[k] = gri_v[k];
                arma::mat    gc  = codes.rows(gri_u);
                arma::colvec wts = arma::conv_to<arma::colvec>::from(grw_v);
                g_ws = arma::sum(gc.each_col() % wts, 0);
            }
        }

        if (!IS_DEFAULT) {
            resp_ctx[ctx_cols] = WEIGHT_DIM;
            response_weight = static_cast<int>(tensor[flat_index(resp_ctx, dims)]);
        }

        arma::rowvec row_vec = codes.row(ri);
        arma::mat resp = connection_matrix(
            g_ws, row_vec, static_cast<double>(response_weight), ordered);
        g_w_mat += resp;
        row_conn.row(i) = arma::vectorise(resp).t();
    }

    TensorNetworks result;
    result.connection_counts     = arma::vectorise(g_w_mat).t();
    result.row_connection_counts = row_conn;
    return result;
}

/// @}

/// @name Per-row co-occurrence and rolling window (rENA accumulation primitives)
/// @{

/** @brief Compute the upper-triangle co-occurrence vector for each row.
 *
 *  For each row, calls code_connections() and optionally binarizes the result.
 *
 *  @param[in] codes   Code matrix (n_rows × n_codes).
 *  @param[in] binary  When @c true, clamp all positive entries to 1.
 *
 *  @returns Matrix of shape n_rows × choose_two(n_codes).
 *
 *  @note Equivalent to @c rows_to_co_occurrences() in @c rENA/ena.cpp.
 */
inline arma::mat row_connections(
    const arma::mat& codes,
    bool binary = true
) {
    int n_rows = codes.n_rows;
    int n_tri  = choose_two(codes.n_cols);
    arma::mat out(n_rows, n_tri, arma::fill::zeros);
    for (int row = 0; row < n_rows; ++row)
        out.row(row) = code_connections(codes.row(row));
    if (binary) out.elem(arma::find(out > 0)).ones();
    return out;
}

/** @brief Rolling backward window sum of the raw code matrix.
 *
 *  For each focal row @c k, sums rows [@c max(0, k - window_size + 1), @c k].
 *  Returns a matrix of the same shape as @p codes — no upper-triangle transform
 *  is applied.  A @p window_size of 0 or less is treated as 1 (current row only).
 *
 *  @param[in] codes        Code matrix (n_rows × n_codes).
 *  @param[in] window_size  Number of rows to include in the backward window
 *                          (including the focal row).  Values <= 0 are clamped
 *                          to 1.
 *
 *  @returns Matrix of the same shape as @p codes containing the windowed sums.
 *
 *  @note Equivalent to @c ref_window_lag() in @c rENA/ena.cpp.
 */
inline arma::mat rolling_window_sum(
    const arma::mat& codes,
    int window_size = 1
) {
    int n_rows = codes.n_rows;
    arma::mat out(n_rows, codes.n_cols, arma::fill::zeros);
    for (int row = 0; row < n_rows; ++row) {
        int earliest = (window_size > 0)
            ? std::max(0, row - (window_size - 1))
            : row;  // guard: window_size <= 0 -> single row
        out.row(row) = arma::sum(codes.rows(earliest, row));
    }
    return out;
}

/// @}

} // namespace qe

#endif // LIBQE_ACCUMULATION_HPP
