// lasso.hpp — Coordinate-descent Lasso for multivariate response.
//
// Mirrors glmnet(x, y, family = "mgaussian", penalty.factor = ...) with
// k-fold cross-validation to select lambda, and warm-starts along the
// lambda path for efficiency.
//
// Public entry point:
//
//   qe::lasso_x1_contribution(X_raw, Y, x1_cols, pf, n_lambda, k_folds, eps)
//
//   Returns an n×q matrix of fitted values attributable to the x1 columns
//   (the "target" predictors) at the CV-selected lambda.  Used by
//   generalized_means_rotation to isolate the target variable's contribution
//   to the ENA point space after controlling for covariates.
//
// Penalty factors (pf):
//   0.0 → predictor is always included (forced in; no shrinkage)
//   1.0 → standard L1 penalty
//
// The caller is responsible for building the model matrix (main effects and/or
// interactions); this function accepts a numeric matrix, not a formula.

#ifndef LIBQE_LASSO_HPP
#define LIBQE_LASSO_HPP

#include <armadillo>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace qe {

// ── Soft-threshold operator ────────────────────────────────────────────────
// S(z, γ) = sign(z) * max(|z| - γ, 0)
inline double soft_threshold(double z, double gamma) noexcept {
    return z > gamma ? z - gamma : z < -gamma ? z + gamma : 0.0;
}

// ── Single-response coordinate descent at a fixed lambda ──────────────────
//
// X        n×p  column-standardized predictor matrix
// y        n×1  response (centering recommended but not required)
// lambda   regularization strength, on the standardized predictor scale
// pf       p×1  penalty factors  (0 = unpenalized, 1 = standard L1)
// beta0    warm-start (pass empty arma::vec{} for a cold start from zero)
//
// Returns: p×1 coefficient vector on the standardized scale.
//
// Algorithm: cyclic coordinate descent (naive updates).
// The partial residual update is:
//   z_j = X_j' r / n  +  (X_j'X_j / n) * beta_j   [add back j's contribution]
//   beta_j ← S(z_j, lambda * pf_j) / (X_j'X_j / n)
// where r = y - X*beta is maintained incrementally.
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

// ── Lambda_max — smallest lambda that zeros all penalized coefficients ─────
//
// For standardized X and centered Yc, this is:
//   max over penalized j, over response columns k  of  |X_j' Yc_k| / n
//
// Above lambda_max all penalized coefficients are exactly 0 (but unpenalized
// ones are not, because they remain in via OLS regardless of lambda).
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

// ── Cross-validated Lasso → fitted contribution from target columns ────────
//
// X_raw    n×p   raw model matrix (main effects and/or interactions).
//                The caller builds this from the data frame; no formula
//                parsing is done here.
// Y        n×q   ENA point matrix (multivariate response).
// x1_cols  0-based column indices in X_raw that belong to the target variable
//          (these receive penalty_factor = 0, i.e. they are always included).
// pf       p×1   penalty factors: 0 for x1_cols, 1 elsewhere.
//          (Caller constructs this; allows non-standard weightings if needed.)
// n_lambda length of the lambda path                          (default 50)
// k_folds  number of cross-validation folds                  (default 5)
// eps      lambda_min = eps * lambda_max                      (default 0.01)
//
// Returns: n×q matrix of fitted values attributable to x1_cols at the
//          CV-selected lambda (lambda.min).
//
// The returned matrix is on the original (unstandardized) Y scale and
// represents the portion of Y that is linearly explained by the target
// predictors after penalizing (and thus shrinking/zeroing) covariate
// coefficients.  A zero matrix is returned if the target predictors carry
// no signal at any lambda (e.g., target is constant after subsetting).
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
