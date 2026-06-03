/**
 * @file lasso.hpp
 * @brief Coordinate-descent Lasso for multivariate response with k-fold CV.
 *
 * Mirrors `glmnet(x, y, family = "mgaussian", penalty.factor = ...)` with
 * k-fold cross-validation to select lambda, and warm-starts along the lambda
 * path for efficiency.
 *
 * The public entry point is qe::lasso_x1_contribution(), which returns an
 * n×q matrix of fitted values attributable to the "target" predictors at the
 * CV-selected lambda.  It is used by `generalized_means_rotation` to isolate
 * the target variable's contribution to the ENA point space after controlling
 * for covariates.
 *
 * **Penalty factors (`pf`):**
 * - `0.0` — predictor is always included (forced in; no shrinkage)
 * - `1.0` — standard L1 penalty
 *
 * The caller is responsible for building the model matrix (main effects and/or
 * interactions); this function accepts a numeric matrix, not a formula.
 */

#ifndef LIBQE_LASSO_HPP
#define LIBQE_LASSO_HPP

#include <armadillo>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace qe {

/**
 * @brief Soft-threshold (shrinkage) operator used by coordinate descent.
 *
 * @details Computes the scalar soft-threshold function:
 *   \f$ S(z,\gamma) = \operatorname{sign}(z)\,\max(|z|-\gamma,\,0) \f$
 *
 * @param[in] z     Input value.
 * @param[in] gamma Threshold amount (must be non-negative for meaningful use).
 * @returns Soft-thresholded value; exactly zero when `|z| <= gamma`.
 */
inline double soft_threshold(double z, double gamma) noexcept {
    return z > gamma ? z - gamma : z < -gamma ? z + gamma : 0.0;
}

/**
 * @brief Single-response Lasso via cyclic coordinate descent at a fixed lambda.
 *
 * @details Fits a Lasso regression for a single response vector using cyclic
 * coordinate descent (naive updates).  The partial-residual update for
 * predictor \f$j\f$ is:
 * \f[
 *   z_j = \frac{X_j^\top r}{n} + \frac{X_j^\top X_j}{n}\,\beta_j
 * \f]
 * \f[
 *   \beta_j \leftarrow \frac{S\!\left(z_j,\;\lambda\,\mathit{pf}_j\right)}{X_j^\top X_j / n}
 * \f]
 * where \f$r = y - X\beta\f$ is the residual maintained incrementally.
 * Predictors with \f$\mathit{pf}_j = 0\f$ skip the soft-threshold and receive
 * the unconstrained OLS update instead.
 *
 * @param[in] X        n×p column-standardized predictor matrix.
 * @param[in] y        n×1 response vector (centering recommended but not required).
 * @param[in] lambda   Regularization strength, on the standardized predictor scale.
 * @param[in] pf       p×1 penalty factors (0 = unpenalized / forced in, 1 = standard L1).
 * @param[in] beta0    Warm-start coefficient vector; pass an empty `arma::vec{}`
 *                     for a cold start from zero.
 * @param[in] max_iter Maximum number of coordinate-descent passes (default 1000).
 * @param[in] tol      Convergence tolerance on the maximum coefficient change
 *                     per pass (default 1e-7).
 * @returns p×1 coefficient vector on the standardized predictor scale.
 */
inline arma::vec lasso_cd(
    const arma::mat& X,
    const arma::vec& y,
    double           lambda,
    const arma::vec& pf,
    arma::vec        beta0   = {},
    int              max_iter = 1000,
    double           tol      = 1e-7
) {
    const arma::uword p = X.n_cols;
    const double      n = static_cast<double>(X.n_rows);

    // Precompute X_j'X_j / n for each predictor
    arma::vec xTx(p);
    for (arma::uword j = 0; j < p; ++j)
        xTx(j) = arma::dot(X.col(j), X.col(j)) / n;

    arma::vec beta = beta0.is_empty()
                   ? arma::vec(p, arma::fill::zeros)
                   : beta0;
    arma::vec r = y - X * beta;   // residual, maintained incrementally

    for (int it = 0; it < max_iter; ++it) {
        double max_delta = 0.0;

        for (arma::uword j = 0; j < p; ++j) {
            if (xTx(j) < 1e-14) continue;   // skip zero-variance predictor

            // Partial correlation statistic
            double z = arma::dot(X.col(j), r) / n + xTx(j) * beta(j);

            double b_new = (pf(j) < 1e-12)
                ? z / xTx(j)                                   // forced in: OLS
                : soft_threshold(z, lambda * pf(j)) / xTx(j); // L1

            double delta = b_new - beta(j);
            if (std::abs(delta) > max_delta) max_delta = std::abs(delta);
            r     -= X.col(j) * delta;
            beta(j) = b_new;
        }

        if (max_delta < tol) break;
    }
    return beta;
}

/**
 * @brief Compute the smallest lambda that drives all penalized coefficients to zero.
 *
 * @details For column-standardized `X` and column-centered `Yc`, lambda_max is:
 * \f[
 *   \lambda_{\max} = \max_{\substack{j:\,\mathit{pf}_j > 0 \\ k}} \frac{|X_j^\top Y_{c,k}|}{n}
 * \f]
 * Above this value every penalized coefficient is exactly zero (unpenalized
 * predictors are unaffected and remain in via OLS regardless of lambda).
 *
 * @param[in] X   n×p column-standardized predictor matrix.
 * @param[in] Yc  n×q column-centered response matrix.
 * @param[in] pf  p×1 penalty factors; predictors with `pf(j) < 1e-12` are skipped.
 * @returns Scalar lambda_max value.
 */
inline double lasso_lambda_max(
    const arma::mat& X,   // column-standardized
    const arma::mat& Yc,  // column-centered responses
    const arma::vec& pf
) {
    const double n = static_cast<double>(X.n_rows);
    double lmax = 0.0;
    for (arma::uword j = 0; j < X.n_cols; ++j) {
        if (pf(j) < 1e-12) continue;   // skip unpenalized predictors
        for (arma::uword k = 0; k < Yc.n_cols; ++k) {
            double v = std::abs(arma::dot(X.col(j), Yc.col(k))) / n;
            if (v > lmax) lmax = v;
        }
    }
    return lmax;
}

/**
 * @brief Cross-validated multivariate Lasso; returns fitted values for target columns only.
 *
 * @details Fits a multivariate Lasso (one independent single-response problem per
 * response column) along a log-spaced lambda path from lambda_max down to
 * `eps * lambda_max`.  Lambda is selected by k-fold cross-validation
 * (mean squared error, averaged over folds and response columns), and the
 * model is refit on the full data at the selected lambda using warm-starts
 * along the path.
 *
 * **Preprocessing performed internally:**
 * - `X_raw` is column-standardized to zero mean and unit variance (zero-variance
 *   columns are left as-is to avoid division by zero).
 * - `Y` is column-centered to absorb the intercept.
 * - Coefficients are unstandardized before the contribution is computed.
 *
 * **Fold assignment:** row \f$i\f$ is assigned to fold \f$i \bmod k\f$
 * (deterministic, order-dependent).
 *
 * **Return value:** The n×q matrix \f$X_{\text{raw},x_1} \hat{B}_{x_1}\f$
 * represents the portion of `Y` (mean-removed) that is linearly explained by
 * the target predictors after penalizing covariate coefficients.  A zero
 * matrix is returned when the target predictors carry no signal at any lambda
 * (e.g., the target is constant after subsetting).  Constant shifts cancel
 * in the downstream between-group scatter (categorical) and slope regression
 * (numeric), so no `ymu` adjustment is applied to the return value.
 *
 * @param[in] X_raw    n×p raw model matrix built by the caller (main effects
 *                     and/or interactions); no formula parsing is done here.
 * @param[in] Y        n×q ENA point matrix (multivariate response).
 * @param[in] x1_cols  0-based column indices in `X_raw` that belong to the
 *                     target variable; these receive `pf = 0` (always included).
 * @param[in] pf       p×1 penalty factors: 0 for `x1_cols`, 1 elsewhere.
 *                     The caller constructs this; non-standard weightings are
 *                     supported.
 * @param[in] n_lambda Length of the lambda path (default 50).
 * @param[in] k_folds  Number of cross-validation folds (default 5; clamped to
 *                     `[2, n/2]`).
 * @param[in] eps      Ratio `lambda_min / lambda_max`; controls how far down
 *                     the path to search (default 0.01).
 * @returns n×q matrix of fitted values attributable to `x1_cols` at the
 *          CV-selected lambda (lambda.min), on the original `Y` scale.
 */
inline arma::mat lasso_x1_contribution(
    const arma::mat& X_raw,
    const arma::mat& Y,
    const arma::uvec& x1_cols,
    const arma::vec&  pf,
    int    n_lambda = 50,
    int    k_folds  = 5,
    double eps      = 0.01
) {
    const arma::uword n = X_raw.n_rows;
    const arma::uword p = X_raw.n_cols;
    const arma::uword q = Y.n_cols;

    // Guard: need at least 2 folds and 2 observations per fold
    k_folds = std::max(2, std::min(k_folds, static_cast<int>(n) / 2));

    // ── Standardize X (μ=0, σ=1); guard zero-variance predictors ────────────
    arma::rowvec xmu  = arma::mean(X_raw, 0);
    arma::rowvec xstd = arma::stddev(X_raw, 0, 0);
    xstd.elem(arma::find(xstd < 1e-12)).ones();   // avoid divide-by-zero
    arma::mat X = (X_raw.each_row() - xmu).each_row() / xstd;

    // ── Center Y (absorbs intercept into the standardized-scale fit) ─────────
    arma::rowvec ymu = arma::mean(Y, 0);
    arma::mat    Yc  = Y.each_row() - ymu;

    // ── Build lambda path: lambda_max → lambda_min on log scale ─────────────
    double lmax = lasso_lambda_max(X, Yc, pf);
    if (lmax < 1e-12)   // target uncorrelated with Y after standardization
        return arma::mat(n, q, arma::fill::zeros);

    arma::vec lambdas = arma::exp(
        arma::linspace(std::log(lmax),
                       std::log(eps * lmax),
                       n_lambda));

    // ── Stratified k-fold CV (fold assignment: row i → fold i % k_folds) ────
    arma::uvec fold_id(n);
    for (arma::uword i = 0; i < n; ++i)
        fold_id(i) = i % static_cast<arma::uword>(k_folds);

    arma::vec cv_err(n_lambda, arma::fill::zeros);

    for (int fold = 0; fold < k_folds; ++fold) {
        arma::uvec tr = arma::find(fold_id != static_cast<arma::uword>(fold));
        arma::uvec va = arma::find(fold_id == static_cast<arma::uword>(fold));

        const arma::mat X_tr = X.rows(tr), X_va = X.rows(va);
        const arma::mat Y_tr = Yc.rows(tr), Y_va = Yc.rows(va);
        const double n_va_q  = static_cast<double>(va.n_elem * q);

        // Warm-start along path for this fold (start cold at lambda_max)
        arma::mat bw(p, q, arma::fill::zeros);
        for (int l = 0; l < n_lambda; ++l) {
            arma::mat b(p, q);
            for (arma::uword k = 0; k < q; ++k)
                b.col(k) = lasso_cd(X_tr, Y_tr.col(k), lambdas(l), pf, bw.col(k));
            bw = b;

            arma::mat err = Y_va - X_va * b;
            cv_err(l) += arma::accu(err % err) / n_va_q;
        }
    }
    cv_err /= static_cast<double>(k_folds);

    // ── Refit on full data at lambda_min, warming along the path ─────────────
    arma::uword best = cv_err.index_min();
    arma::mat bfull(p, q, arma::fill::zeros);
    for (arma::uword l = 0; l <= best; ++l) {
        arma::mat b(p, q);
        for (arma::uword k = 0; k < q; ++k)
            b.col(k) = lasso_cd(X, Yc.col(k), lambdas(l), pf, bfull.col(k));
        bfull = b;
    }

    // ── Unstandardize: beta_orig[j] = beta_std[j] / xstd[j] ─────────────────
    for (arma::uword j = 0; j < p; ++j)
        bfull.row(j) /= xstd(j);

    // ── Return fitted contribution from x1 columns only ──────────────────────
    // This is the portion of Y (mean-removed) linearly explained by the target
    // predictors after covariate adjustment.  Constant shifts cancel in both
    // the between-group scatter (categorical) and slope regression (numeric)
    // that follow, so no ymu adjustment is needed here.
    return X_raw.cols(x1_cols) * bfull.rows(x1_cols);
}

}  // namespace qe
#endif  // LIBQE_LASSO_HPP
