/** @file trajectory_distance.hpp
 *  @brief Integrated Euclidean distance between trajectory curves and lagged distance metrics.
 */
#ifndef LIBQE_TRAJECTORY_DISTANCE_HPP
#define LIBQE_TRAJECTORY_DISTANCE_HPP

#include <armadillo>
#include <vector>
#include <cmath>
#include <algorithm>
#include "trajectory.hpp"

namespace qe {

struct GaussLegendreRule {
    arma::vec nodes;
    arma::vec weights;
};

/** @brief Obtain exact 32-point Gauss-Legendre quadrature nodes and weights on [-1, 1] via Golub-Welsch. */
inline const GaussLegendreRule& get_gl_rule_32() {
    static const GaussLegendreRule rule = []() {
        int N = 32;
        arma::mat J(N, N, arma::fill::zeros);
        for (int k = 1; k < N; ++k) {
            double b = static_cast<double>(k) / std::sqrt(4.0 * k * k - 1.0);
            J(k, k - 1) = b;
            J(k - 1, k) = b;
        }
        arma::vec evals;
        arma::mat evecs;
        arma::eig_sym(evals, evecs, J);
        arma::vec w = 2.0 * arma::square(evecs.row(0).t());
        return GaussLegendreRule{evals, w};
    }();
    return rule;
}

/** @brief Compute integrated Euclidean distance between two 2D parametric polynomial curves.
 *
 *  Computes integral_{t_start}^{t_end} ||F_a(t) - F_b(t)|| dt
 *  using 32-point Gauss-Legendre quadrature.
 *
 *  @param[in] coeffs_ax  Coefficients of x_a(t).
 *  @param[in] coeffs_ay  Coefficients of y_a(t).
 *  @param[in] coeffs_bx  Coefficients of x_b(t).
 *  @param[in] coeffs_by  Coefficients of y_b(t).
 *  @param[in] t_start    Start of integration interval (default 0.0).
 *  @param[in] t_end      End of integration interval (default 1.0).
 *
 *  @returns Integrated distance scalar.
 */
inline double integrated_curve_distance(
    const arma::vec& coeffs_ax,
    const arma::vec& coeffs_ay,
    const arma::vec& coeffs_bx,
    const arma::vec& coeffs_by,
    double t_start = 0.0,
    double t_end = 1.0
) {
    if (t_end <= t_start) return 0.0;

    const auto& rule = get_gl_rule_32();
    double mid = 0.5 * (t_start + t_end);
    double half_len = 0.5 * (t_end - t_start);
    double total = 0.0;

    for (size_t i = 0; i < rule.nodes.n_elem; ++i) {
        double t_eval = mid + half_len * rule.nodes[i];
        double xa = eval_poly_scalar(coeffs_ax, t_eval);
        double ya = eval_poly_scalar(coeffs_ay, t_eval);
        double xb = eval_poly_scalar(coeffs_bx, t_eval);
        double yb = eval_poly_scalar(coeffs_by, t_eval);
        double dx = xa - xb;
        double dy = ya - yb;
        double dist = std::sqrt(dx * dx + dy * dy);

        total += rule.weights[i] * dist;
    }

    return half_len * total;
}

/** @brief Compute lagged curve distance: 1/(1-lag) * integral_{lag}^1 ||F_follower(t) - F_leader(t - lag)|| dt. */
inline double lagged_curve_distance(
    const arma::vec& coeffs_fol_x,
    const arma::vec& coeffs_fol_y,
    const arma::vec& coeffs_ldr_x,
    const arma::vec& coeffs_ldr_y,
    double lag = 0.0
) {
    if (lag < 0.0 || lag >= 0.999) return 0.0;
    double t_start = lag;
    double t_end = 1.0;
    const auto& rule = get_gl_rule_32();
    double mid = 0.5 * (t_start + t_end);
    double half_len = 0.5 * (t_end - t_start);
    double total = 0.0;

    for (size_t i = 0; i < rule.nodes.n_elem; ++i) {
        double t_fol = mid + half_len * rule.nodes[i];
        double t_ldr = t_fol - lag;

        double x_fol = eval_poly_scalar(coeffs_fol_x, t_fol);
        double y_fol = eval_poly_scalar(coeffs_fol_y, t_fol);
        double x_ldr = eval_poly_scalar(coeffs_ldr_x, t_ldr);
        double y_ldr = eval_poly_scalar(coeffs_ldr_y, t_ldr);

        double dx = x_fol - x_ldr;
        double dy = y_fol - y_ldr;
        double dist = std::sqrt(dx * dx + dy * dy);

        total += rule.weights[i] * dist;
    }

    double integral = half_len * total;
    double interval = 1.0 - lag;
    return (interval > 1e-6) ? (integral / interval) : 0.0;
}

/** @brief Compute pairwise integrated distance matrix across K polynomial trajectory curves. */
inline arma::mat pairwise_trajectory_distance_matrix(
    const std::vector<arma::vec>& all_coeffs_x,
    const std::vector<arma::vec>& all_coeffs_y
) {
    size_t K = all_coeffs_x.size();
    arma::mat D(K, K, arma::fill::zeros);

    for (size_t i = 0; i < K; ++i) {
        for (size_t j = i + 1; j < K; ++j) {
            double dist = integrated_curve_distance(
                all_coeffs_x[i], all_coeffs_y[i],
                all_coeffs_x[j], all_coeffs_y[j]
            );
            D(i, j) = dist;
            D(j, i) = dist;
        }
    }

    return D;
}

} // namespace qe

#endif // LIBQE_TRAJECTORY_DISTANCE_HPP
