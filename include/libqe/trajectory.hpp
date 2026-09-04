/** @file trajectory.hpp
 *  @brief Parametric trajectory curve fitting, LOOCV-2D degree selection,
 *         and differential geometry (velocity, acceleration, curvature, turns).
 */
#ifndef LIBQE_TRAJECTORY_HPP
#define LIBQE_TRAJECTORY_HPP

#include <armadillo>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <limits>

namespace qe {

/** @brief Polynomial curve fit result in 2D epistemic space. */
struct PolyCurveFit {
    int degree = 1;
    arma::vec coeffs_x;     ///< Raw coefficients [c0, ..., cd] for x(t) = sum(ck * t^k)
    arma::vec coeffs_y;     ///< Raw coefficients [c0, ..., cd] for y(t) = sum(ck * t^k)
    std::string basis = "orthogonal"; ///< Basis used during fitting: "orthogonal" (R stats::poly-like) or "raw".
    arma::vec basis_alpha;  ///< Orthogonal-basis recurrence alpha values (empty for raw basis).
    arma::vec basis_norm2;  ///< Orthogonal-basis recurrence squared norms, matching stats::poly coefs$norm2.
    double cv_error = 0.0;  ///< 2D LOOCV error
    double aic = 0.0;       ///< Joint bivariate AIC
    arma::vec t;            ///< Evaluated or input time points
    arma::mat fitted;       ///< n × 2 fitted coordinates
};

inline arma::mat build_raw_poly_design(const arma::vec& t_vec, int degree) {
    int n = static_cast<int>(t_vec.n_elem);
    arma::mat X(n, degree + 1);
    for (int i = 0; i < n; ++i) {
        double cur_t = t_vec[i];
        double p = 1.0;
        for (int k = 0; k <= degree; ++k) {
            X(i, k) = p;
            p *= cur_t;
        }
    }
    return X;
}

inline arma::vec poly_multiply_by_linear(const arma::vec& coeffs, double shift) {
    arma::vec out(coeffs.n_elem + 1, arma::fill::zeros);
    for (size_t k = 0; k < coeffs.n_elem; ++k) {
        out[k] += -shift * coeffs[k];
        out[k + 1] += coeffs[k];
    }
    return out;
}

inline arma::mat orthogonal_poly_basis_from_coefs(
    const arma::vec& t_vec,
    int degree,
    const arma::vec& alpha,
    const arma::vec& norm2
) {
    int n = static_cast<int>(t_vec.n_elem);
    arma::mat Z(n, degree + 1, arma::fill::ones);
    if (degree >= 1) {
        Z.col(1) = t_vec - alpha[0];
    }
    for (int i = 2; i <= degree; ++i) {
        Z.col(i) = (t_vec - alpha[i - 1]) % Z.col(i - 1) -
                   (norm2[i] / norm2[i - 1]) * Z.col(i - 2);
    }
    for (int i = 0; i <= degree; ++i) {
        Z.col(i) /= std::sqrt(norm2[i + 1]);
    }
    return Z.cols(1, degree);
}

inline arma::mat orthogonal_basis_raw_coefficients(
    int degree,
    const arma::vec& alpha,
    const arma::vec& norm2
) {
    arma::mat raw(degree + 1, degree + 1, arma::fill::zeros);
    raw(0, 0) = 1.0;
    if (degree >= 1) {
        arma::vec p1(2);
        p1[0] = -alpha[0];
        p1[1] = 1.0;
        raw.submat(0, 1, 1, 1) = p1 / std::sqrt(norm2[2]);
    }
    arma::vec prev2(1);
    prev2[0] = 1.0;
    arma::vec prev1;
    if (degree >= 1) {
        prev1.set_size(2);
        prev1[0] = -alpha[0];
        prev1[1] = 1.0;
    }
    for (int i = 2; i <= degree; ++i) {
        arma::vec term = poly_multiply_by_linear(prev1, alpha[i - 1]);
        arma::vec padded_prev2(term.n_elem, arma::fill::zeros);
        padded_prev2.subvec(0, prev2.n_elem - 1) = prev2;
        arma::vec current = term - (norm2[i] / norm2[i - 1]) * padded_prev2;
        raw.submat(0, i, i, i) = current / std::sqrt(norm2[i + 1]);
        prev2 = prev1;
        prev1 = current;
    }
    return raw;
}

inline bool modified_gram_schmidt_qr(
    const arma::mat& X,
    arma::mat& Q,
    arma::mat& R
) {
    const int n = static_cast<int>(X.n_rows);
    const int p = static_cast<int>(X.n_cols);
    Q.zeros(n, p);
    R.zeros(p, p);

    for (int k = 0; k < p; ++k) {
        arma::vec v = X.col(k);
        for (int j = 0; j < k; ++j) {
            R(j, k) = arma::dot(Q.col(j), v);
            v -= R(j, k) * Q.col(j);
        }

        double norm = std::sqrt(arma::dot(v, v));
        if (norm <= 1e-12) {
            return false;
        }

        R(k, k) = norm;
        Q.col(k) = v / norm;
    }

    return true;
}

inline arma::mat build_orthogonal_poly_design(
    const arma::vec& t_vec,
    int degree,
    arma::vec& alpha,
    arma::vec& norm2
) {
    int n = static_cast<int>(t_vec.n_elem);
    if (degree < 1) {
        return arma::mat(n, 0);
    }
    arma::vec unique_t = arma::unique(t_vec);
    if (static_cast<size_t>(degree) >= unique_t.n_elem) {
        throw std::invalid_argument("orthogonal polynomial degree must be less than number of unique time points.");
    }

    double xbar = arma::mean(t_vec);
    arma::vec centered = t_vec - xbar;
    arma::mat X = build_raw_poly_design(centered, degree);

    arma::mat Q;
    arma::mat R;
    bool ok = modified_gram_schmidt_qr(X, Q, R);
    if (!ok) {
        throw std::invalid_argument("orthogonal polynomial degree must be less than number of unique time points.");
    }

    arma::mat Z = Q * arma::diagmat(R.diag());
    arma::vec z_norm2 = arma::sum(arma::square(Z), 0).t();

    alpha.set_size(degree);
    for (int i = 0; i < degree; ++i) {
        alpha[i] = arma::dot(centered, arma::square(Z.col(i))) / z_norm2[i] + xbar;
    }

    norm2.set_size(degree + 2);
    norm2[0] = 1.0;
    for (int i = 0; i <= degree; ++i) {
        norm2[i + 1] = z_norm2[i];
    }

    return orthogonal_poly_basis_from_coefs(t_vec, degree, alpha, norm2);
}

inline arma::mat build_poly_design(
    const arma::vec& t_vec,
    int degree,
    const std::string& basis,
    arma::vec& alpha,
    arma::vec& norm2,
    arma::mat& basis_raw_coeffs
) {
    int n = static_cast<int>(t_vec.n_elem);
    arma::mat X(n, degree + 1);
    X.col(0).ones();
    if (basis == "orthogonal") {
        arma::mat Z = build_orthogonal_poly_design(t_vec, degree, alpha, norm2);
        X.cols(1, degree) = Z;
        basis_raw_coeffs = orthogonal_basis_raw_coefficients(degree, alpha, norm2);
    } else if (basis == "raw") {
        X = build_raw_poly_design(t_vec, degree);
        alpha.reset();
        norm2.reset();
        basis_raw_coeffs.eye(degree + 1, degree + 1);
    } else {
        throw std::invalid_argument("basis must be 'orthogonal' or 'raw'.");
    }
    return X;
}

/** @brief Differential geometry profile along a trajectory curve. */
struct TrajectoryDerivatives {
    arma::vec t;
    arma::vec vx;           ///< dx/dt
    arma::vec vy;           ///< dy/dt
    arma::vec speed;        ///< sqrt(vx^2 + vy^2)
    arma::vec ax;           ///< d^2x/dt^2
    arma::vec ay;           ///< d^2y/dt^2
    arma::vec heading_rate; ///< omega = vx*ay - vy*ax (angular turning velocity)
    arma::vec curvature;    ///< kappa = omega / max(speed^3, eps)
};

/** @brief Evaluate a 1D polynomial sum(c_k * t^k) at a single time t. */
inline double eval_poly_scalar(const arma::vec& coeffs, double t) {
    double val = 0.0;
    double t_pow = 1.0;
    for (size_t k = 0; k < coeffs.n_elem; ++k) {
        val += coeffs[k] * t_pow;
        t_pow *= t;
    }
    return val;
}

/** @brief Evaluate polynomial 1st derivative sum(k * c_k * t^(k-1)) at a single time t. */
inline double eval_poly_deriv1_scalar(const arma::vec& coeffs, double t) {
    double val = 0.0;
    double t_pow = 1.0;
    for (size_t k = 1; k < coeffs.n_elem; ++k) {
        val += static_cast<double>(k) * coeffs[k] * t_pow;
        t_pow *= t;
    }
    return val;
}

/** @brief Evaluate polynomial 2nd derivative sum(k*(k-1) * c_k * t^(k-2)) at a single time t. */
inline double eval_poly_deriv2_scalar(const arma::vec& coeffs, double t) {
    double val = 0.0;
    double t_pow = 1.0;
    for (size_t k = 2; k < coeffs.n_elem; ++k) {
        val += static_cast<double>(k * (k - 1)) * coeffs[k] * t_pow;
        t_pow *= t;
    }
    return val;
}

/** @brief Evaluate 2D parametric curve (x(t), y(t)) at multiple time points. */
inline arma::mat eval_poly_curve(
    const arma::vec& coeffs_x,
    const arma::vec& coeffs_y,
    const arma::vec& t_eval
) {
    size_t n = t_eval.n_elem;
    arma::mat out(n, 2);
    for (size_t i = 0; i < n; ++i) {
        double t = t_eval[i];
        out(i, 0) = eval_poly_scalar(coeffs_x, t);
        out(i, 1) = eval_poly_scalar(coeffs_y, t);
    }
    return out;
}

/** @brief Fit a 2D polynomial curve with automatic LOOCV-2D or AIC degree selection.
 *
 *  Given coordinates (x_i, y_i) and times t_i in [0, 1], constructs polynomial models
 *  for degrees d in [1, max_degree]. LOOCV error is computed via the closed-form hat-matrix formula:
 *  CV(d) = sum( (r_{x,i} / (1 - h_{ii}))^2 + (r_{y,i} / (1 - h_{ii}))^2 )
 *
 *  @param[in] points      n × 2 matrix of trajectory coordinates.
 *  @param[in] t           Vector of length n of strictly increasing times (default [0, 1] linearly spaced).
 *  @param[in] max_degree  Maximum polynomial degree to evaluate (default 3).
 *  @param[in] fixed_degree When >= 1, uses this exact degree without search.
 *  @param[in] criterion   "loocv" (default) or "aic".
 *
 *  @returns PolyCurveFit containing optimal degree, coefficients, errors, and fitted points.
 */
inline PolyCurveFit fit_poly_loocv2d(
    const arma::mat& points,
    const arma::vec& t = arma::vec(),
    int max_degree = 3,
    int fixed_degree = 0,
    const std::string& criterion = "loocv",
    const std::string& basis = "orthogonal"
) {
    int n = static_cast<int>(points.n_rows);
    if (n < 2 || points.n_cols < 2) {
        throw std::invalid_argument("fit_poly_loocv2d requires at least 2 points and 2 columns.");
    }

    // Default normalized time t in [0, 1]
    arma::vec t_vec;
    if (t.n_elem == static_cast<size_t>(n)) {
        t_vec = t;
    } else {
        t_vec = arma::linspace<arma::vec>(0.0, 1.0, n);
    }

    arma::vec x = points.col(0);
    arma::vec y = points.col(1);

    int deg_upper = std::min(max_degree, std::max(1, n - 2));
    if (fixed_degree >= 1) {
        deg_upper = std::min(fixed_degree, n - 1);
    }

    int best_deg = 1;
    double best_score = std::numeric_limits<double>::infinity();
    PolyCurveFit best_fit;

    int deg_start = (fixed_degree >= 1) ? fixed_degree : 1;

    for (int d = deg_start; d <= deg_upper; ++d) {
        int n_params = d + 1;
        if (n_params > n) break;

        arma::vec basis_alpha;
        arma::vec basis_norm2;
        arma::mat basis_raw_coeffs;
        arma::mat X = build_poly_design(t_vec, d, basis, basis_alpha, basis_norm2, basis_raw_coeffs);

        // Solve least squares: beta = (X^T X)^(-1) X^T y
        arma::mat XtX = X.t() * X;
        arma::mat XtX_inv;
        bool inv_ok = arma::inv(XtX_inv, XtX);
        if (!inv_ok) {
            // Pseudo-inverse fallback
            XtX_inv = arma::pinv(XtX);
        }

        arma::vec beta_x = XtX_inv * (X.t() * x);
        arma::vec beta_y = XtX_inv * (X.t() * y);

        arma::vec pred_x = X * beta_x;
        arma::vec pred_y = X * beta_y;

        arma::vec rx = x - pred_x;
        arma::vec ry = y - pred_y;

        // Hat matrix diagonal: h_ii = x_i^T (X^T X)^(-1) x_i
        arma::vec h(n);
        for (int i = 0; i < n; ++i) {
            arma::rowvec xi = X.row(i);
            h[i] = arma::as_scalar(xi * XtX_inv * xi.t());
            h[i] = std::min(0.9999, std::max(0.0, h[i]));
        }

        // 2D LOOCV error
        double cv_err = 0.0;
        for (int i = 0; i < n; ++i) {
            double denom = 1.0 - h[i];
            double rxi = rx[i] / denom;
            double ryi = ry[i] / denom;
            cv_err += (rxi * rxi + ryi * ryi);
        }

        // Bivariate AIC
        double rss = arma::dot(rx, rx) + arma::dot(ry, ry);
        double aic = static_cast<double>(n) * std::log(std::max(1e-12, rss / (2.0 * n))) + 2.0 * (2.0 * n_params);

        double score = (criterion == "aic") ? aic : cv_err;

        if (score < best_score || d == deg_start) {
            best_score = score;
            best_deg = d;
            best_fit.degree = d;
            best_fit.coeffs_x = basis_raw_coeffs * beta_x;
            best_fit.coeffs_y = basis_raw_coeffs * beta_y;
            best_fit.basis = basis;
            best_fit.basis_alpha = basis_alpha;
            best_fit.basis_norm2 = basis_norm2;
            best_fit.cv_error = cv_err;
            best_fit.aic = aic;
            best_fit.t = t_vec;
            best_fit.fitted = arma::join_rows(pred_x, pred_y);
        }
    }

    return best_fit;
}

/** @brief Compute trajectory velocities, speeds, accelerations, heading turn rates, and curvatures. */
inline TrajectoryDerivatives eval_trajectory_derivatives(
    const arma::vec& coeffs_x,
    const arma::vec& coeffs_y,
    const arma::vec& t_eval
) {
    size_t n = t_eval.n_elem;
    TrajectoryDerivatives d;
    d.t = t_eval;
    d.vx.set_size(n);
    d.vy.set_size(n);
    d.speed.set_size(n);
    d.ax.set_size(n);
    d.ay.set_size(n);
    d.heading_rate.set_size(n);
    d.curvature.set_size(n);

    for (size_t i = 0; i < n; ++i) {
        double t = t_eval[i];
        double vx = eval_poly_deriv1_scalar(coeffs_x, t);
        double vy = eval_poly_deriv1_scalar(coeffs_y, t);
        double ax = eval_poly_deriv2_scalar(coeffs_x, t);
        double ay = eval_poly_deriv2_scalar(coeffs_y, t);

        double spd = std::sqrt(vx * vx + vy * vy);
        double omega = vx * ay - vy * ax; // angular velocity / heading rate
        double spd3 = spd * spd * spd;
        double kappa = (spd3 > 1e-12) ? (omega / spd3) : 0.0;

        d.vx[i] = vx;
        d.vy[i] = vy;
        d.speed[i] = spd;
        d.ax[i] = ax;
        d.ay[i] = ay;
        d.heading_rate[i] = omega;
        d.curvature[i] = kappa;
    }

    return d;
}

} // namespace qe

#endif // LIBQE_TRAJECTORY_HPP
