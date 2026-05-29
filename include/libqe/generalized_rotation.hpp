// generalized_rotation.hpp — Generalized Means Rotation (GMR) with
// Lasso-based covariate adjustment.
//
// Mirrors rENA's ena.rotate.by.generalized() + gmr() + get_x1_main_effect().
// Authoritative reference: commit 46776a1981a90b3a3b2861ed1010e9dbb7acf901
// ("Fixed bugs in rotation functions") by Zhiqiang Cai.
//
// Algorithm overview
// ------------------
//  1. Compute x_vector: the rotation axis that best represents the target
//     variable's contribution to the ENA point space, after controlling for
//     covariates via Lasso (lasso_x1_contribution).
//  2. Deflate V by x_vector → defA = V - V*x*x'.
//  3. Optionally compute secondary axis x1: the normalised group-mean
//     difference in the *original* V space (not defA) for groups 0 vs 1.
//     x1 stays empty when the target is not categorical or there are not
//     exactly 2 groups — there is no SVD fallback.
//  4. Compute y_vector from a GMR call on defA, or from SVD of defA.
//     Explicitly orthogonalise y_vector against x_vector, and against x1
//     when x1 is available.
//  5. Double-deflate V by x_vector and y_vector, then call complete_rotation()
//     to fill remaining dimensions from the SVD of the doubly-deflated space.
//
// Caller responsibilities
// -----------------------
//  - Build the model matrix (X_raw) from the data frame, including dummy
//    variables for categorical predictors and any desired interaction terms.
//    No formula parsing is done here.
//  - Identify x1_cols: the 0-based column indices in X_raw that correspond to
//    the target variable (these are unpenalized in the Lasso).
//  - Encode categorical targets as 0-based integer codes in x_target.
//  - Set x_n_groups to the number of distinct groups when x_categorical=true.
//
// The R wrapper (rENA::ena.rotate.by.generalized) handles these steps and
// calls libqe::generalized_means_rotation() with the prepared matrices.

#ifndef LIBQE_GENERALIZED_ROTATION_HPP
#define LIBQE_GENERALIZED_ROTATION_HPP

#include "linalg_fallback.hpp"  // qe::linalg::eig_sym (works w/o LAPACK)
#include "rotation.hpp"         // ena_svd, complete_rotation, RotationResult
#include "lasso.hpp"            // lasso_x1_contribution

#include <armadillo>
#include <stdexcept>
#include <string>
#include <vector>

namespace qe {

// ── Between-group scatter matrix ──────────────────────────────────────────
//
// Mirrors rENA's compute_SB().
// SB = Σ_g  n_g * (μ_g - μ)(μ_g - μ)'    where μ is the grand mean.
//
// V       n×p  matrix of ENA points (or covariate-adjusted contribution)
// labels  n×1  0-based integer group labels (arma::uvec)
// n_groups     number of distinct groups
inline arma::mat between_group_scatter(
    const arma::mat&  V,
    const arma::uvec& labels,
    arma::uword       n_groups
) {
    const arma::uword p = V.n_cols;
    arma::rowvec mu = arma::mean(V, 0);
    arma::mat SB(p, p, arma::fill::zeros);

    for (arma::uword g = 0; g < n_groups; ++g) {
        arma::uvec idx = arma::find(labels == g);
        if (idx.is_empty()) continue;
        arma::vec diff = (arma::mean(V.rows(idx), 0) - mu).t();
        SB += static_cast<double>(idx.n_elem) * diff * diff.t();
    }
    return SB;
}

// ── Rotation direction from covariate-adjusted contribution ───────────────
//
// Numeric target:
//   Centres target, then β = (t_c' t_c)^{-1} t_c' Vx  (slope from lm(Vx ~ t)).
//   Mirrors R: model = lm(Vx ~ target); beta = coefficients[2,]
//
// Categorical target:
//   Computes the between-group scatter matrix SB and returns its leading
//   eigenvector (the direction of maximum between-group variance).
//   Mirrors R: sb = compute_SB(Vx, target); r = svd(sb)$v[,1]
//
// Returns a unit vector (p×1).
inline arma::vec gmr_direction(
    const arma::mat& Vx,         // n×q covariate-adjusted contribution
    const arma::vec& target,     // n×1 (numeric float, or 0-based int codes)
    bool             categorical,
    arma::uword      n_groups = 0
) {
    arma::vec r;

    if (categorical) {
        if (n_groups < 2)
            throw std::runtime_error(
                "gmr_direction: x_n_groups must be >= 2 for categorical target");
        arma::uvec labels = arma::conv_to<arma::uvec>::from(target);
        arma::mat  SB     = between_group_scatter(Vx, labels, n_groups);

        arma::vec eigval; arma::mat eigvec;
        qe::linalg::eig_sym(eigval, eigvec, SB);  // ascending order
        r = eigvec.col(eigvec.n_cols - 1);         // largest eigenvalue → last col
    } else {
        // Centre target to get the OLS slope (mirrors R's lm() intercept handling)
        arma::vec tc   = target - arma::mean(target);
        double    tTt  = arma::dot(tc, tc);
        if (tTt < 1e-12)
            throw std::runtime_error(
                "gmr_direction: target has zero variance after centering");
        arma::rowvec beta = (tc.t() * Vx) / tTt;  // 1×q slope vector
        r = beta.t();
    }

    double norm = arma::norm(r);
    if (norm < 1e-12)
        throw std::runtime_error("gmr_direction: resulting direction has zero norm");
    return r / norm;
}

// ── Result of one GMR axis step ───────────────────────────────────────────
struct GMRStepResult {
    arma::vec direction;   // unit vector in connection space (p×1)
};

// ── One GMR axis step ─────────────────────────────────────────────────────
//
// Computes the rotation direction for one axis of the generalized means
// rotation, including the Lasso covariate adjustment.
//
// V            n×q ENA points (full data; subset applied internally)
// X_model      n×p model matrix (caller-built; target cols + covariate cols)
// target        n×1 target variable (raw float or 0-based integer codes)
// x1_cols      0-based indices of target columns in X_model (unpenalized)
// categorical  true → categorical target (uses between-group scatter)
// n_groups     number of distinct groups (only used when categorical=true)
// subset       optional 0-based row indices to use (empty = use all rows)
// n_lambda, k_folds, lasso_eps — forwarded to lasso_x1_contribution
inline GMRStepResult gmr_step(
    const arma::mat&  V,
    const arma::mat&  X_model,
    const arma::vec&  target,
    const arma::uvec& x1_cols,
    bool              categorical,
    arma::uword       n_groups,
    const arma::uvec& subset,
    int               n_lambda  = 50,
    int               k_folds   = 5,
    double            lasso_eps = 0.01
) {
    const arma::uword n = V.n_rows;
    const arma::uword p = X_model.n_cols;

    // ── Select rows ──────────────────────────────────────────────────────────
    arma::uvec rows = subset.is_empty()
                    ? arma::regspace<arma::uvec>(0, n - 1)
                    : subset;

    const arma::mat V_sub  = V.rows(rows);
    const arma::mat X_sub  = X_model.rows(rows);
    const arma::vec t_sub  = target.rows(rows);

    // ── Lasso-adjusted contribution Vx ───────────────────────────────────────
    arma::mat Vx_sub;
    if (x1_cols.n_elem >= p) {
        // All columns are target columns (no penalized covariates):
        // fall back to simple OLS projection (Vx = fitted values from lm(V~t))
        arma::vec tc  = t_sub - arma::mean(t_sub);
        double    tTt = arma::dot(tc, tc);
        if (tTt > 1e-12) {
            arma::rowvec b1 = (tc.t() * V_sub) / tTt;
            Vx_sub = tc * b1;
        } else {
            Vx_sub.zeros(V_sub.n_rows, V_sub.n_cols);
        }
    } else {
        arma::vec pf(p, arma::fill::ones);
        for (arma::uword j : x1_cols) pf(j) = 0.0;

        Vx_sub = lasso_x1_contribution(
            X_sub, V_sub, x1_cols, pf, n_lambda, k_folds, lasso_eps);
    }

    // ── Rotation direction ───────────────────────────────────────────────────
    arma::vec direction = gmr_direction(Vx_sub, t_sub, categorical, n_groups);

    return {direction};
}

// ── Parameters for generalized_means_rotation ─────────────────────────────
struct GeneralizedRotationParams {
    // ── X axis (required) ─────────────────────────────────────────────────
    arma::mat   x_model_matrix;     // n×p  model matrix for x axis
    arma::vec   x_target;           // n×1  target variable (raw or 0-based int)
    arma::uvec  x1_cols;            // 0-based target column indices in x_model_matrix
    bool        x_categorical = false;
    arma::uword x_n_groups    = 0;
    arma::uvec  x_subset;           // optional row subset (0-based); empty = all

    // ── Y axis (optional) ─────────────────────────────────────────────────
    // If has_y is false, the y axis is the leading SVD of the deflated space.
    bool        has_y          = false;
    arma::mat   y_model_matrix; // n×p  model matrix for y axis
    arma::vec   y_target;       // n×1
    arma::uvec  y1_cols;
    bool        y_categorical  = false;
    arma::uword y_n_groups     = 0;
    // y axis always uses all rows (no subset)

    // ── Lasso tuning ───────────────────────────────────────────────────────
    int    n_lambda  = 50;    // length of lambda path
    int    k_folds   = 5;     // cross-validation folds
    double lasso_eps = 0.01;  // lambda_min = lasso_eps * lambda_max
};

// ── Generalized Means Rotation ────────────────────────────────────────────
//
// Full implementation of ena.rotate.by.generalized() in C++/Armadillo.
// Matches the authoritative R implementation in commit
// 46776a1981a90b3a3b2861ed1010e9dbb7acf901 by Zhiqiang Cai.
//
// V       n×q   ENA points (line weights; caller normalizes/centers upstream)
// params  GeneralizedRotationParams
//
// Returns a RotationResult with:
//   rotation      q×q  full rotation matrix (column j = axis j)
//   eigenvalues   q×1  variance per SVD-filled axis (named axes have 0)
//   column_names       "GMR1", "GMR2"|"SVD2", "SVD3", ..., "SVD{q}"
inline RotationResult generalized_means_rotation(
    const arma::mat&                  V,
    const GeneralizedRotationParams&  params
) {
    // ── Step 1: X axis via GMR ────────────────────────────────────────────
    GMRStepResult x_gmr = gmr_step(
        V,
        params.x_model_matrix,
        params.x_target,
        params.x1_cols,
        params.x_categorical,
        params.x_n_groups,
        params.x_subset,
        params.n_lambda,
        params.k_folds,
        params.lasso_eps);

    const arma::vec x_vector = x_gmr.direction;

    // ── Step 2: Deflate V by x_vector ─────────────────────────────────────
    // defA is used only for y_vector computation; x1 uses original V.
    const arma::mat defA = V - V * x_vector * x_vector.t();

    // ── Step 3: Secondary axis x1 ─────────────────────────────────────────
    // Mirrors R: x1 = normalize(mean(V[group0]) - mean(V[group1])).
    //
    // Key points matching the authoritative R:
    //   - Uses original V (not defA)
    //   - Uses the full-length target (all n rows), not just the subset
    //   - Groups are always 0 (first) and 1 (second) in 0-based encoding
    //   - NO fallback: x1 stays empty when categorical/2-group condition fails
    //   - x1 does NOT deflate defA; it is only used to orthogonalise y_vector
    arma::vec x1;

    if (params.x_categorical && params.x_n_groups >= 2) {
        arma::uvec labels = arma::conv_to<arma::uvec>::from(params.x_target);
        arma::uvec g0 = arma::find(labels == 0);
        arma::uvec g1 = arma::find(labels == 1);

        if (!g0.is_empty() && !g1.is_empty()) {
            arma::rowvec m0 = arma::mean(V.rows(g0), 0);   // original V, not defA
            arma::rowvec m1 = arma::mean(V.rows(g1), 0);
            arma::vec diff  = (m0 - m1).t();
            double   dlen   = arma::norm(diff);
            if (dlen > 1e-10) x1 = diff / dlen;
        }
    }

    // ── Step 4: Y axis ────────────────────────────────────────────────────
    // GMR on defA (deflated by x_vector only), or SVD of defA.
    arma::vec y_vector;
    std::string y_name = "SVD2";

    if (params.has_y) {
        GMRStepResult y_gmr = gmr_step(
            defA,
            params.y_model_matrix,
            params.y_target,
            params.y1_cols,
            params.y_categorical,
            params.y_n_groups,
            {},          // no row subset for y axis
            params.n_lambda,
            params.k_folds,
            params.lasso_eps);
        y_vector = y_gmr.direction;
        y_name   = "GMR2";
    } else {
        RotationResult svd_r = ena_svd(defA);
        y_vector = svd_r.rotation.col(0);
    }

    // ── Step 5: Orthogonalise y_vector ────────────────────────────────────
    // Mirrors R (authoritative commit):
    //   y_vector <- y_vector - (t(y_vector) %*% x_vector) * x_vector
    //   if (!is.null(x1)) y_vector <- y_vector - (t(y_vector) %*% x1) * x1
    //   y_vector <- safe_normalize(y_vector)
    y_vector -= arma::dot(y_vector, x_vector) * x_vector;
    if (!x1.is_empty())
        y_vector -= arma::dot(y_vector, x1) * x1;

    double yn = arma::norm(y_vector);
    if (yn < 1e-12)
        throw std::runtime_error(
            "generalized_means_rotation: y_vector has zero norm after orthogonalization");
    y_vector /= yn;

    // ── Step 6: Combine and complete with SVD ─────────────────────────────
    // Double-deflate the original data (mirrors R):
    //   defA <- A - A %*% x %*% t(x) - A %*% y %*% t(y)
    // then pass to prcomp (equivalent: SVD of defA_final).
    arma::mat defA_final = V
        - V * x_vector * x_vector.t()
        - V * y_vector * y_vector.t();

    arma::mat named_axes = arma::join_rows(x_vector, y_vector);
    return complete_rotation(defA_final, named_axes, {"GMR1", y_name});
}

}  // namespace qe
#endif  // LIBQE_GENERALIZED_ROTATION_HPP
