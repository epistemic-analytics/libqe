#ifndef LIBQE_ACCUMULATION_HPP
#define LIBQE_ACCUMULATION_HPP

// Accumulation primitives shared between rENA (stanza-window model) and tma
// (ground/response/tensor model).  Both models reduce to the same core
// calculate_adjacency_matrix() operation; they differ in how ground and
// response vectors are assembled from the raw data.

#include <armadillo>
#include <functional>
#include <vector>
#include "adjacency.hpp"

namespace qe {

// ---------------------------------------------------------------------------
// Core adjacency math
// ---------------------------------------------------------------------------

// Compute the connection matrix for one ground+response event pair.
//
// ordered == true  (directed ENA):
//   ground→response cross-product + 0.5 * response self-connection (diagonal zeroed)
// ordered == false (undirected ENA):
//   symmetric: (weight * r⊗r) + g⊗r + r⊗g
//
// Equivalent to calculate_adjacency_matrix() in tma/code.cpp.
inline arma::mat calculate_adjacency_matrix(
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

// ---------------------------------------------------------------------------
// Traditional stanza-window accumulation (rENA model)
// ---------------------------------------------------------------------------

// For each row k in `codes` (one conversation), compute the co-occurrence
// vector as:
//
//   sum of codes in window [k - window_back, k + window_forward]
//   → upper-triangle outer-product (vector_to_upper_tri)
//   minus back-reference and forward-reference corrections
//
// Returns a matrix with the same number of rows as `codes` and
// choose_two(n_codes) columns.
//
// Equivalent to ref_window_df() in rENA/ena.cpp.
inline arma::mat stanza_window(
    arma::mat codes,
    int window_back    = 1,
    int window_forward = 0,
    bool binary        = true
) {
    int n_rows  = codes.n_rows;
    int n_codes = codes.n_cols;
    int n_tri   = choose_two(n_codes);
    const int INT_MAX_VAL = std::numeric_limits<int>::max();

    arma::mat out(n_rows, n_tri, arma::fill::zeros);

    for (int row = 0; row < n_rows; row++) {
        int earliest = 0;
        if (window_back == INT_MAX_VAL || window_back == 0) {
            earliest = (window_back == 0) ? row : 0;
        } else {
            earliest = std::max(0, row - (window_back - 1));
        }

        int latest = row;
        if (window_forward == INT_MAX_VAL || row + window_forward >= n_rows) {
            latest = n_rows - 1;
        } else if (window_forward > 0) {
            latest = std::min(n_rows - 1, row + window_forward);
        }

        arma::mat window_rows = codes.rows(earliest, latest);
        arma::mat summed      = arma::sum(window_rows);
        arma::rowvec to_ut    = vector_to_upper_tri(summed);

        // Back-reference correction: subtract contribution of rows that are
        // not the focal row and not within window_back of it
        int head_rows = 0;
        int win_rows  = latest - earliest + 1;
        if (win_rows > 0 && window_back > 1 && row - 1 >= 0) {
            head_rows = win_rows - 1 - window_forward;
            if (head_rows > 0) {
                arma::mat refs     = window_rows.head_rows(head_rows);
                arma::mat ref_sum  = arma::sum(refs);
                to_ut -= vector_to_upper_tri(ref_sum);
            }
        }

        // Forward-reference correction
        if (window_forward > 0 && latest <= n_rows - 1) {
            int tail_rows = latest - row;
            if (tail_rows > 0) {
                arma::mat refs    = window_rows.tail_rows(tail_rows);
                arma::mat ref_sum = arma::sum(refs);
                to_ut -= vector_to_upper_tri(ref_sum);
            }
        }

        out.row(row) = to_ut;
    }

    if (binary) out.elem(arma::find(out > 0)).ones();
    return out;
}

// ---------------------------------------------------------------------------
// Ground/response accumulation (tma model)
// ---------------------------------------------------------------------------

// Accumulate connections for a single unit from its context rows.
//
// `codes`      — full context matrix (all rows visible to this unit), n x p
// `unit_rows`  — 0-based indices of the rows that *belong* to this unit
// `decay_fn`   — maps a vector of distances (0 = current, 1 = one step back, …)
//                to a vector of scalar weights; default = simple window
// `ordered`    — true → directed/full matrix; false → undirected upper-tri
//
// Returns a flat connection vector (length choose_two(p) or p*p).
// This is the pure-C++ equivalent of accumulate_network() in tma/code.cpp,
// with the R function callback replaced by std::function.
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

        g_w_vec += calculate_adjacency_matrix(g_no_resp, response, 1.0, ordered);
    }

    if (!ordered) return directed_to_upper_tri(arma::vectorise(g_w_vec));
    return arma::vectorise(g_w_vec).t();
}

// ---------------------------------------------------------------------------
// Extended ground/response accumulation — returns per-row connection data
// ---------------------------------------------------------------------------

struct UnitNetworks {
    arma::rowvec networks;      // flat connection vector (p^2 or choose_two(p))
    arma::mat    row_networks;  // per-response-row full p^2 matrix (n_unit_rows x p^2)
};

// Like accumulate_unit() but also returns the per-response-row connection
// matrix needed by tma's accumulate_network().
//
// `decay_fn(unit_row, ground_indices)` → weight vector of length
// ground_indices.n_elem.  The two-argument form lets callers (e.g. the tma
// Rcpp wrapper) set R environment variables before calling the actual R
// decay function, without any R-specific code leaking into libqe.
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
        arma::mat    conn     = calculate_adjacency_matrix(g_no_resp, response, 1.0, ordered);
        g_w_mat += conn;
        row_networks.row(i) = arma::vectorise(conn).t();
    }

    UnitNetworks result;
    if (!ordered)
        result.networks = directed_to_upper_tri(arma::vectorise(g_w_mat));
    else
        result.networks = arma::vectorise(g_w_mat).t();
    result.row_networks = row_networks;
    return result;
}

// ---------------------------------------------------------------------------
// Tensor-based multi-modal accumulation (tma model)
// ---------------------------------------------------------------------------

// Compute the linear index into a column-major multi-dimensional array.
// Equivalent to calculate_1d_index() in tma/code.cpp.
inline int calculate_1d_index(const std::vector<int>& indices,
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

struct TensorNetworks {
    arma::rowvec connection_counts;      // unit-level flat vector (p^2)
    arma::mat    row_connection_counts;  // per-response-row (n_unit_rows x p^2)
};

// Pure-C++ port of tma's apply_tensor() inner logic.
//
// `tensor`         — flat column-major array (window and weight values)
// `dims`           — dimensions of the tensor
// `dims_sender`    — tensor axis indices for sender factors
// `dims_receiver`  — tensor axis indices for receiver factors (overridden to
//                    response values when looking up ground-row windows)
// `dims_mode`      — tensor axis indices for mode factors
// `context_lookup` — integer matrix (n_context_rows x n_factors), 0-based
// `unit_rows`      — 0-based response-row indices for this unit
// `codes`          — full context code matrix (n_context_rows x n_codes)
// `times`          — timestamp per context row
// `ordered`        — true → directed full matrix; false → undirected upper-tri
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
            response_win = static_cast<int>(tensor[calculate_1d_index(resp_ctx, dims)]);

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
                    row_win = tensor[calculate_1d_index(row_v, dims)];

                if (times[gr] + row_win > response_time) {
                    gri_v.push_back(gr);
                    row_v[ctx_cols] = WEIGHT_DIM;
                    double row_wgt = static_cast<double>(response_weight);
                    if (!IS_DEFAULT)
                        row_wgt = tensor[calculate_1d_index(row_v, dims)];
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
            response_weight = static_cast<int>(tensor[calculate_1d_index(resp_ctx, dims)]);
        }

        arma::rowvec row_vec = codes.row(ri);
        arma::mat resp = calculate_adjacency_matrix(
            g_ws, row_vec, static_cast<double>(response_weight), ordered);
        g_w_mat += resp;
        row_conn.row(i) = arma::vectorise(resp).t();
    }

    TensorNetworks result;
    result.connection_counts     = arma::vectorise(g_w_mat).t();
    result.row_connection_counts = row_conn;
    return result;
}

// ---------------------------------------------------------------------------
// Per-row co-occurrence and rolling window (rENA accumulation primitives)
// ---------------------------------------------------------------------------

// Per-row upper-triangle co-occurrence.
// For each row, computes vector_to_upper_tri(row) and optionally binarizes.
// Output: n_rows x choose_two(n_codes).
// Equivalent to rows_to_co_occurrences() in rENA/ena.cpp.
inline arma::mat rows_to_co_occurrences(
    const arma::mat& codes,
    bool binary = true
) {
    int n_rows = codes.n_rows;
    int n_tri  = choose_two(codes.n_cols);
    arma::mat out(n_rows, n_tri, arma::fill::zeros);
    for (int row = 0; row < n_rows; ++row)
        out.row(row) = vector_to_upper_tri(codes.row(row));
    if (binary) out.elem(arma::find(out > 0)).ones();
    return out;
}

// Rolling backward window sum of raw code matrix.
// For each row k, sums rows [max(0, k - window_size + 1), k].
// Returns a matrix of the same shape as `codes` (no upper-tri transform).
// window_size <= 0 is treated as 1 (current row only).
// Equivalent to ref_window_lag() in rENA/ena.cpp.
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

} // namespace qe

#endif // LIBQE_ACCUMULATION_HPP
