#ifndef LIBQE_ADJACENCY_HPP
#define LIBQE_ADJACENCY_HPP

// Pure C++ / Armadillo — no Rcpp dependency.
// Include <RcppArmadillo.h> before this header when building inside an R package.

#include <armadillo>
#include <string>
#include <vector>
#include <cmath>

namespace qe {

// ---------------------------------------------------------------------------
// Combinatorics
// ---------------------------------------------------------------------------

// n choose 2
inline int choose_two(int n) {
    return (n * (n - 1)) / 2;
}

// Upper-triangle index pairs for a square matrix of side `len`.
// row == -1 (default): return 2 x k matrix of [row_idx; col_idx]
// row ==  0: return only the row indices
// row ==  1: return only the column indices
// Equivalent to triIndices() in both rENA/ena.cpp and tma/code.cpp.
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

// ---------------------------------------------------------------------------
// Vector ↔ upper-triangle conversions
// ---------------------------------------------------------------------------

// Compute pairwise products of elements and return as a flat upper-triangle
// vector.  v[j] * v[i] for all j < i.
// Equivalent to vector_to_ut() in rENA/ena.cpp.
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

// Fold a directed (n*n) vector into an undirected upper-triangle vector by
// summing symmetric elements (A→B + B→A).
// Equivalent to vector_to_summed_uppertri() in tma/code.cpp.
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

// Flatten an adjacency matrix to a vector.
// full == true  → full n*n vector (directed)
// full == false → upper-triangle only (undirected)
// Equivalent to adjacency_matrix_to_vector() in tma/code.cpp.
inline arma::rowvec network_to_vector(arma::mat x, bool full = true) {
    if (full) return arma::vectorise(x).t();
    arma::mat combined  = arma::trimatu(x, 1) + arma::trimatl(x, -1);
    arma::uvec up_inds  = arma::trimatu_ind(arma::size(combined), 1);
    return arma::vectorise(combined(up_inds)).t();
}

// ---------------------------------------------------------------------------
// String utilities
// ---------------------------------------------------------------------------

// Return "A & B" pair names for every upper-triangle position.
// Equivalent to svector_to_ut() in rENA/ena.cpp.
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

} // namespace qe

#endif // LIBQE_ADJACENCY_HPP
