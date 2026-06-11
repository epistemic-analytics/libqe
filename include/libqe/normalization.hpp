/**
 * @file normalization.hpp
 * @brief Row-wise normalization utilities for ENA connection networks.
 */
#ifndef LIBQE_NORMALIZATION_HPP
#define LIBQE_NORMALIZATION_HPP

#include <armadillo>
#include <cmath>

namespace qe {

/**
 * @brief Normalize each row to unit L2 norm (project onto the unit hypersphere).
 *
 * Rows whose L2 norm is zero are left unchanged.
 *
 * @param m Input matrix (n_units × n_connections), row-major.
 * @returns Matrix of the same shape with each row divided by its L2 norm.
 * @note Equivalent to @c fun_sphere_norm() in rENA/ena.cpp.
 */
inline arma::mat normalize_networks(arma::mat m) {
    arma::mat out(m.n_rows, m.n_cols, arma::fill::zeros);
    for (arma::uword r = 0; r < m.n_rows; r++) {
        double len = arma::norm(m.row(r), 2);
        if (std::isfinite(len) && len > 0.0) out.row(r) = m.row(r) / len;
    }
    return out;
}

/**
 * @brief Scale every entry by the reciprocal of the largest row L2 norm.
 *
 * Preserves relative magnitudes across rows; does not normalise each row
 * individually.  If the largest row norm is zero, the matrix is returned
 * unchanged.
 *
 * @param m Input matrix (n_units × n_connections), modified in place.
 * @returns Scaled matrix (same shape as @p m).
 * @note Equivalent to @c fun_skip_sphere_norm() in rENA/ena.cpp.
 */
inline arma::mat scale_networks(arma::mat m) {
    double largest = 0.0;
    for (arma::uword r = 0; r < m.n_rows; r++) {
        double len = arma::norm(m.row(r), 2);
        if (std::isfinite(len)) largest = std::max(largest, len);
    }
    if (largest > 0.0) m /= largest;
    return m;
}

} // namespace qe

#endif // LIBQE_NORMALIZATION_HPP
