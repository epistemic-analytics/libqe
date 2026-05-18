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

} // namespace qe

#endif // LIBQE_ACCUMULATION_HPP
