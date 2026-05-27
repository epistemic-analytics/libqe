#ifndef LIBQE_NORMALIZATION_HPP
#define LIBQE_NORMALIZATION_HPP

#include <armadillo>
#include <cmath>

namespace qe {

// ---------------------------------------------------------------------------
// Row-wise normalization
// ---------------------------------------------------------------------------

// Divide each row by its own L2 norm (unit hypersphere projection).
// Zero rows are left unchanged.
// Equivalent to fun_sphere_norm() in rENA/ena.cpp.
inline arma::mat normalize_networks(arma::mat m) {
    arma::mat out(m.n_rows, m.n_cols, arma::fill::zeros);
    for (arma::uword r = 0; r < m.n_rows; r++) {
        double len = arma::norm(m.row(r), 2);
        if (len > 0.0) out.row(r) = m.row(r) / len;
    }
    return out;
}

// Divide every entry by the largest row L2 norm found in the matrix.
// Preserves relative magnitudes; does not normalise each row individually.
// Equivalent to fun_skip_sphere_norm() in rENA/ena.cpp.
inline arma::mat scale_networks(arma::mat m) {
    double largest = 0.0;
    for (arma::uword r = 0; r < m.n_rows; r++) {
        largest = std::max(largest, arma::norm(m.row(r), 2));
    }
    if (largest > 0.0) m /= largest;
    return m;
}

} // namespace qe

#endif // LIBQE_NORMALIZATION_HPP
