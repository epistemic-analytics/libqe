/**
 * @file adjacency.hpp
 * @brief Combinatorics and vector/matrix utilities for ENA connection networks.
 *
 * Pure C++ / Armadillo — no Rcpp dependency.
 * Include @c <RcppArmadillo.h> before this header when building inside an R package.
 */
#ifndef LIBQE_ADJACENCY_HPP
#define LIBQE_ADJACENCY_HPP

#include <armadillo>
#include <string>
#include <vector>
#include <cmath>

namespace qe {

// ---------------------------------------------------------------------------
/// @name Combinatorics
/// @{
// ---------------------------------------------------------------------------

/**
 * @brief Number of unordered pairs from @p n items (n choose 2).
 *
 * @param n Number of codes/nodes.
 * @returns @c (n * (n - 1)) / 2.
 */
inline int choose_two(int n) {
    return (n * (n - 1)) / 2;
}

/**
 * @brief Upper-triangle index pairs for a symmetric matrix of side @p len.
 *
 * @param len Side length of the square matrix (number of codes).
 * @param row Controls what is returned:
 *   - @c -1 (default): 2 × k matrix of [row_idx; col_idx].
 *   - @c  0: row indices only (1 × k).
 *   - @c  1: column indices only (1 × k).
 * @returns An @c arma::umat of shape 2×k or 1×k depending on @p row.
 * @note Equivalent to @c triIndices() in rENA/ena.cpp and tma/code.cpp.
 */
inline arma::umat connection_indices(int len, int row = -1) {
    int vS = choose_two(len);
    int s  = 0;

    arma::umat vR(2, vS, arma::fill::zeros);
    arma::umat vRone(1, vS, arma::fill::zeros);

    for (int i = 2; i <= len; i++) {
        for (int j = 0; j < i - 1; j++) {
            vR(0, s) = j;
            vR(1, s) = i - 1;
            if (row == 0)      vRone[s] = j;
            else if (row == 1) vRone[s] = i - 1;
            s++;
        }
    }
    return (row == -1) ? vR : vRone;
}

/// @}

// ---------------------------------------------------------------------------
/// @name Vector ↔ upper-triangle conversions
/// @{
// ---------------------------------------------------------------------------

/**
 * @brief Compute pairwise products and return as a flat upper-triangle vector.
 *
 * For each pair @c (j, i) with @c j < i, outputs @c v[j] * v[i].
 * The result has length @c choose_two(v.size()).
 *
 * @param v A 1 × p row vector (or p-element matrix).
 * @returns A row vector of length @c choose_two(p).
 * @note Equivalent to @c vector_to_ut() in rENA/ena.cpp.
 */
inline arma::rowvec code_connections(arma::mat v) {
    int vL = v.size();
    int vS = choose_two(vL);
    arma::rowvec out(vS, arma::fill::zeros);
    int s = 0;
    for (int i = 2; i <= vL; i++) {
        for (int j = 0; j < i - 1; j++) {
            out[s] = v[j] * v[i - 1];
            s++;
        }
    }
    return out;
}

/**
 * @brief Fold a directed n² vector into an undirected upper-triangle vector.
 *
 * Symmetric elements are summed: A→B + B→A for every pair.
 *
 * @param v Flat directed vector of length n², column-major (n = sqrt(v.size())).
 * @returns Upper-triangle row vector of length @c choose_two(n).
 * @note Equivalent to @c vector_to_summed_uppertri() in tma/code.cpp.
 */
inline arma::rowvec fold_directed_network(arma::vec v) {
    int n    = static_cast<int>(std::round(std::sqrt(static_cast<double>(v.size()))));
    int tris = choose_two(n);

    arma::mat m(v);
    m.reshape(n, n);
    m = m + m.t();

    arma::colvec flat     = m.as_col();
    arma::uvec   up_inds  = arma::trimatu_ind(arma::size(m), 1);
    return arma::conv_to<arma::rowvec>::from(flat.elem(up_inds));
}

/**
 * @brief Flatten an adjacency matrix to a vector.
 *
 * @param x Square adjacency matrix (n × n).
 * @param full If @c true (default), returns the full n² vector (directed).
 *             If @c false, returns the upper-triangle only (undirected).
 * @returns Row vector of length n² or @c choose_two(n).
 * @note Equivalent to @c adjacency_matrix_to_vector() in tma/code.cpp.
 */
inline arma::rowvec network_to_vector(arma::mat x, bool full = true) {
    if (full) return arma::vectorise(x).t();
    arma::mat combined  = arma::trimatu(x, 1) + arma::trimatl(x, -1);
    arma::uvec up_inds  = arma::trimatu_ind(arma::size(combined), 1);
    return arma::vectorise(combined(up_inds)).t();
}

/// @}

// ---------------------------------------------------------------------------
/// @name String utilities
/// @{
// ---------------------------------------------------------------------------

/**
 * @brief Generate "A & B" pair names for every upper-triangle position.
 *
 * For code names @c {"A", "B", "C"} returns @c {"A & B", "A & C", "B & C"}.
 *
 * @param v Ordered list of code names (length p).
 * @returns Vector of @c choose_two(p) label strings.
 * @note Equivalent to @c svector_to_ut() in rENA/ena.cpp.
 */
inline std::vector<std::string> connection_names(std::vector<std::string> v) {
    int vL = v.size();
    int vS = choose_two(vL);
    std::vector<std::string> out(vS);
    int s = 0;
    for (int i = 2; i <= vL; i++) {
        for (int j = 0; j < i - 1; j++) {
            out[s] = v[j] + " & " + v[i - 1];
            s++;
        }
    }
    return out;
}

/// @}

} // namespace qe

#endif // LIBQE_ADJACENCY_HPP
