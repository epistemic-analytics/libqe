/** @file stability.hpp
 *  @brief Stability metrics and distance-distance matrix correlation for parameter tuning.
 */
#ifndef LIBQE_STABILITY_HPP
#define LIBQE_STABILITY_HPP

#include <armadillo>
#include <cmath>
#include <algorithm>

namespace qe {

/** @brief Compute Pearson correlation between pairwise Euclidean distance structures of two matrices.
 *
 *  Evaluates dist-dist correlation: r(dist(X), dist(Y)), measuring whether relative
 *  neighborhoods and distances are preserved across parameter changes (e.g. window vs lookback).
 *
 *  @param[in] X Numeric matrix (n_rows × n_cols).
 *  @param[in] Y Numeric matrix (n_rows × n_cols).
 *
 *  @returns Pearson correlation coefficient in [-1, 1].
 */
inline double dist_dist_correlation(const arma::mat& X, const arma::mat& Y) {
    size_t n = X.n_rows;
    if (n != Y.n_rows || n < 3) return 0.0;

    size_t n_pairs = n * (n - 1) / 2;
    arma::vec dx(n_pairs);
    arma::vec dy(n_pairs);

    size_t idx = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            arma::rowvec diff_x = X.row(i) - X.row(j);
            arma::rowvec diff_y = Y.row(i) - Y.row(j);
            dx[idx] = std::sqrt(arma::dot(diff_x, diff_x));
            dy[idx] = std::sqrt(arma::dot(diff_y, diff_y));
            idx++;
        }
    }

    double mean_x = arma::mean(dx);
    double mean_y = arma::mean(dy);

    arma::vec zx = dx - mean_x;
    arma::vec zy = dy - mean_y;

    double sxx = arma::dot(zx, zx);
    double syy = arma::dot(zy, zy);
    double sxy = arma::dot(zx, zy);

    if (sxx <= 1e-12 || syy <= 1e-12) return 0.0;
    return sxy / std::sqrt(sxx * syy);
}

} // namespace qe

#endif // LIBQE_STABILITY_HPP
