/** @file linalg_fallback.hpp
 *  @brief Thin LAPACK-free implementations of SVD, QR, and symmetric
 *         eigendecomposition for use in WASM / no-LAPACK builds.
 *
 *  When @c ARMA_USE_LAPACK is defined these wrappers delegate directly to the
 *  standard Armadillo routines (@c arma::svd, @c arma::qr, @c arma::eig_sym),
 *  so there is zero overhead in normal builds.  When LAPACK is not available
 *  (e.g. Emscripten WASM builds) the wrappers fall back to pure C++
 *  implementations that work for the small matrices typical in ENA
 *  (p ≤ ~100 connection dimensions).
 *
 *  All routines live in namespace @c qe::linalg and operate on @c arma::mat
 *  (double precision).
 *
 *  Fallback algorithms:
 *  - **SVD**      — one-sided Jacobi iterations on \f$A^T\f$ (Demmel & Veselić 1992)
 *  - **QR**       — modified Gram-Schmidt
 *  - **eig_sym**  — cyclic Jacobi (classical Jacobi sweep)
 */

#ifndef LIBQE_LINALG_FALLBACK_HPP
#define LIBQE_LINALG_FALLBACK_HPP

#include <armadillo>
#include <cmath>
#include <limits>

namespace qe {
namespace linalg {

/// @cond INTERNAL
// ── helpers ───────────────────────────────────────────────────────────────────

namespace detail {

/// @brief Modified Gram-Schmidt orthonormalization applied in-place.
///
/// Columns of @p Q are successively orthonormalized against all previously
/// processed columns.  A column whose norm falls below 1e-14 after projection
/// is left unchanged (not zeroed) to avoid introducing NaNs.
///
/// @param Q  Matrix whose columns are orthonormalized in-place (modified).
inline void mgs(arma::mat& Q) {
    const arma::uword p = Q.n_cols;
    for (arma::uword j = 0; j < p; ++j) {
        for (arma::uword i = 0; i < j; ++i)
            Q.col(j) -= arma::dot(Q.col(i), Q.col(j)) * Q.col(i);
        double n = arma::norm(Q.col(j));
        if (n > 1e-14) Q.col(j) /= n;
    }
}

/// @brief One cyclic Jacobi sweep on a p×p symmetric matrix.
///
/// Iterates over all super-diagonal index pairs @c (q,r) and applies a
/// two-sided Jacobi rotation that annihilates @c A(q,r).  The same rotation
/// is accumulated in @p V so that, starting with @c V = I, the full sequence
/// of sweeps yields @c V = eigenvector matrix.
///
/// @param A  Symmetric p×p matrix, updated in-place toward diagonal form.
/// @param V  Rotation accumulator (p×p); initialize to identity to recover
///           eigenvectors.
///
/// @note Algorithm follows the classical cyclic-by-rows Jacobi scheme.
///       Off-diagonal entries no larger than 1e-14 × (|a_qq| + |a_rr|) are
///       skipped for numerical stability.  The `<=` (not `<`) guard correctly
///       handles the all-zero case (a_qq = a_rr = a_qr = 0) that would
///       otherwise produce NaN via 0/0 in the theta computation.
inline void jacobi_sweep(arma::mat& A, arma::mat& V) {
    const arma::uword p = A.n_rows;
    for (arma::uword q = 0; q < p - 1; ++q) {
        for (arma::uword r = q + 1; r < p; ++r) {
            double aqq = A(q, q), arr = A(r, r), aqr = A(q, r);
            if (std::abs(aqr) <= 1e-14 * (std::abs(aqq) + std::abs(arr)))
                continue;
            double theta = (arr - aqq) / (2.0 * aqr);
            double t = (theta >= 0.0)
                ? 1.0 / (theta + std::sqrt(1.0 + theta * theta))
                : 1.0 / (theta - std::sqrt(1.0 + theta * theta));
            double c = 1.0 / std::sqrt(1.0 + t * t); ///< cosine of Jacobi rotation angle
            double s = t * c;                          ///< sine of Jacobi rotation angle
            // Apply Jacobi rotation to A (symmetric update)
            double new_qq = aqq - t * aqr;
            double new_rr = arr + t * aqr;
            A(q, q) = new_qq;  A(r, r) = new_rr;  A(q, r) = A(r, q) = 0.0;
            for (arma::uword k = 0; k < p; ++k) {
                if (k == q || k == r) continue;
                double akq = A(k, q), akr = A(k, r);
                A(k, q) = A(q, k) = c * akq - s * akr;
                A(k, r) = A(r, k) = s * akq + c * akr;
            }
            // Accumulate rotation in V
            for (arma::uword k = 0; k < p; ++k) {
                double vkq = V(k, q), vkr = V(k, r);
                V(k, q) = c * vkq - s * vkr;
                V(k, r) = s * vkq + c * vkr;
            }
        }
    }
}

} // namespace detail

/// @endcond

/// @name Eigendecomposition
/// @{

// ── eig_sym ───────────────────────────────────────────────────────────────────

/// @brief Eigendecomposition of a real symmetric matrix.
///
/// Computes eigenvalues and eigenvectors of the symmetric p×p matrix @p S and
/// returns them sorted in **ascending** eigenvalue order.
///
/// @param[out] eigval  p×1 vector of eigenvalues, ascending.
/// @param[out] eigvec  p×p matrix whose columns are the corresponding
///                     orthonormal eigenvectors.
/// @param[in]  S       Real symmetric p×p input matrix (not modified).
///
/// @warning With @c ARMA_USE_LAPACK defined this calls @c arma::eig_sym
///          directly.  Without LAPACK the fallback runs up to @c 50*p cyclic
///          Jacobi sweeps; convergence is guaranteed for well-conditioned
///          matrices but may be slow for p > 200.
///
/// @note Fallback algorithm: classical cyclic-by-rows Jacobi iteration.
///       Convergence is declared when the Frobenius norm of the strictly
///       off-diagonal part drops below 1e-13.
inline void eig_sym(arma::vec& eigval, arma::mat& eigvec, const arma::mat& S) {
#if defined(ARMA_USE_LAPACK)
    arma::eig_sym(eigval, eigvec, S);
#else
    const arma::uword p = S.n_rows;
    arma::mat A = S;                              ///< working copy of S
    eigvec.eye(p, p);                             ///< accumulate rotations; starts as identity

    for (int sweep = 0; sweep < 50 * static_cast<int>(p); ++sweep) {
        // Check off-diagonal convergence
        double off = 0.0;
        for (arma::uword q = 0; q < p - 1; ++q)
            for (arma::uword r = q + 1; r < p; ++r)
                off += A(q, r) * A(q, r);
        if (std::sqrt(off) < 1e-13) break;
        detail::jacobi_sweep(A, eigvec);
    }
    eigval = A.diag();

    // Sort ascending
    arma::uvec idx = arma::sort_index(eigval);
    eigval  = eigval(idx);
    eigvec  = eigvec.cols(idx);
#endif
}

/// @}

/// @name QR Factorization
/// @{

// ── qr_full ───────────────────────────────────────────────────────────────────

/// @brief Full (square) QR factorization of an n×p matrix.
///
/// Produces a square n×n orthogonal factor @p Q and an n×p upper-trapezoidal
/// factor @p R such that @c A = Q*R.  This matches the convention of
/// @c arma::qr(), which returns the full (non-economy) @p Q.
///
/// @param[out] Q  n×n orthogonal matrix.
/// @param[out] R  n×p upper-trapezoidal matrix.
/// @param[in]  A  n×p input matrix (not modified).
///
/// @warning With @c ARMA_USE_LAPACK defined this calls @c arma::qr directly.
///          Without LAPACK the fallback constructs the full ONB by seeding
///          columns beyond @c A.n_cols with standard basis vectors and then
///          applying modified Gram-Schmidt; numerical quality degrades for
///          nearly rank-deficient inputs.
///
/// @note The caller (@c orthogonal_svd) depends on @p Q being n×n so that
///       @c Q.cols(k, n-1) spans the null space of @c A^T.
inline void qr_full(arma::mat& Q, arma::mat& R, const arma::mat& A) {
#if defined(ARMA_USE_LAPACK)
    arma::qr(Q, R, A);
#else
    // Build a full n×n orthogonal matrix.
    // Step 1: thin QR via modified Gram-Schmidt (gives first k=A.n_cols columns).
    // Step 2: extend to a full ONB using random vectors + re-orthogonalisation.
    const arma::uword n = A.n_rows, p = A.n_cols;

    // Start with all n columns of A padded with random vectors if n > p.
    arma::mat Qfull(n, n, arma::fill::zeros);
    // Copy A's columns first
    for (arma::uword j = 0; j < p && j < n; ++j)
        Qfull.col(j) = A.col(j);
    // Fill remaining columns with standard basis vectors (or random fallback)
    for (arma::uword j = p; j < n; ++j) {
        Qfull(j, j) = 1.0;   ///< standard basis vector e_j used as seed
    }
    // Full MGS orthonormalization
    detail::mgs(Qfull);
    Q = Qfull;

    // R is n×p (upper trapezoidal); only first p rows are non-zero
    R = Q.t() * A;
    // Zero the strictly lower-triangular part of R[0:p, 0:p]
    for (arma::uword j = 0; j < p; ++j)
        for (arma::uword i = j + 1; i < n; ++i)
            R(i, j) = 0.0;
#endif
}

/// @}

/// @name Singular Value Decomposition
/// @{

// ── svd ───────────────────────────────────────────────────────────────────────

/// @brief Full SVD: @c A ≈ U * diag(s) * V', singular values descending.
///
/// Decomposes the n×p matrix @p A into left singular vectors @p U, singular
/// values @p s, and right singular vectors @p V.
///
/// @param[out] U  n×min(n,p) matrix of left singular vectors.
/// @param[out] s  min(n,p)×1 vector of singular values, descending.
/// @param[out] V  p×p matrix of right singular vectors (all p columns).
/// @param[in]  A  n×p input matrix (not modified).
///
/// @warning With @c ARMA_USE_LAPACK defined this calls @c arma::svd directly.
///          Without LAPACK the fallback computes @c A^T*A, calls @c eig_sym,
///          and derives @p U via @c U[:,j] = A*V[:,j]/s_j.  Columns with
///          @c s_j < 1e-14 are zeroed.  Accuracy is governed by the
///          conditioning of @c A^T*A (squares the condition number of @p A).
///
/// @note Fallback algorithm: eigendecomposition of \f$A^T A\f$ via cyclic
///       Jacobi (Demmel & Veselić 1992).  Suitable for the small, moderately
///       conditioned matrices typical in ENA (p ≤ ~100).
inline void svd(arma::mat& U, arma::vec& s, arma::mat& V, const arma::mat& A) {
#if defined(ARMA_USE_LAPACK)
    arma::svd(U, s, V, A);
#else
    // One-sided Jacobi on A^T:
    // We work with B = A^T (p×n). The right singular vectors of A are the
    // left singular vectors of B (= columns of U_B after Jacobi on B^T B).
    // We use the eigendecomposition of A^T * A to get V and s.
    const arma::uword n = A.n_rows, p = A.n_cols;
    const arma::uword k = std::min(n, p);

    arma::mat AtA = A.t() * A;   ///< p×p symmetric Gram matrix
    arma::vec eigval;
    arma::mat eigvec;
    eig_sym(eigval, eigvec, AtA);

    // Eigenvalues from eig_sym are ascending; we want descending.
    arma::uvec idx = arma::sort_index(eigval, "descend");
    eigval = eigval(idx);
    eigvec = eigvec.cols(idx);

    // Singular values (non-negative square roots)
    s.set_size(k);
    for (arma::uword j = 0; j < k; ++j)
        s(j) = std::sqrt(std::max(0.0, eigval(j)));

    // V = eigvec (right singular vectors of A are eigvectors of A^T*A)
    V = eigvec;

    // U = A * V / s (left singular vectors, only first k columns needed)
    U.set_size(n, k);
    for (arma::uword j = 0; j < k; ++j) {
        if (s(j) > 1e-14)
            U.col(j) = A * V.col(j) / s(j);
        else
            U.col(j).zeros();
    }
#endif
}

// ── svd_right ─────────────────────────────────────────────────────────────────

/// @brief Singular values and right singular vectors only (left vectors skipped).
///
/// A lighter-weight variant of @c svd() for callers that only need @p s and
/// @p V.  Under LAPACK the full @c arma::svd is still called internally; the
/// savings come in WASM builds where computing @p U is avoided.
///
/// @param[out] s  min(n,p)×1 vector of singular values, descending.
/// @param[out] V  p×p matrix of right singular vectors.
/// @param[in]  A  n×p input matrix (not modified).
///
/// @warning With @c ARMA_USE_LAPACK defined this still invokes the full
///          @c arma::svd; the left-vector output is discarded.
inline void svd_right(arma::vec& s, arma::mat& V, const arma::mat& A) {
#if defined(ARMA_USE_LAPACK)
    arma::mat U;
    arma::svd(U, s, V, A);
#else
    arma::mat U_unused;
    svd(U_unused, s, V, A);
#endif
}

/// @}

/// @name Linear System Solve
/// @{

// ── solve_spd ─────────────────────────────────────────────────────────────────

/// @brief Solve @c A*x = b for symmetric positive (semi-)definite @p A.
///
/// Returns the n×k solution matrix @p x given the n×n system matrix @p A and
/// the n×k right-hand side @p b.  Near-zero eigenvalues are handled gracefully,
/// making this a pseudo-inverse solve safe for rank-deficient Gram matrices.
///
/// @param[in] A    Real symmetric positive (semi-)definite n×n matrix.
/// @param[in] b    n×k right-hand side matrix.
/// @param[in] tol  Relative threshold below which eigenvalues are treated as
///                 zero.  Eigenvalues with @c |λ| < tol * max(|λ|) are
///                 inverted as zero (i.e. those directions are projected out).
///                 Default: 1e-12.
/// @returns  n×k solution matrix @p x satisfying @c A*x ≈ b.
///
/// @warning With @c ARMA_USE_LAPACK defined this delegates to
///          @c arma::solve(A, b, arma::solve_opts::fast), which uses a
///          Cholesky factorization and does **not** handle rank deficiency.
///          Without LAPACK the eigendecomposition-based pseudo-inverse is used,
///          which is safe for rank-deficient inputs but slower.
///
/// @note Without LAPACK: @c A = V * diag(λ) * V^T is computed via @c eig_sym;
///       the solution is @c x = V * diag(1/λ) * V^T * b with small λ skipped.
inline arma::mat solve_spd(const arma::mat& A, const arma::mat& b,
                            double tol = 1e-12) {
#if defined(ARMA_USE_LAPACK)
    return arma::solve(A, b, arma::solve_opts::fast);
#else
    arma::vec  eigval;
    arma::mat  eigvec;
    eig_sym(eigval, eigvec, A);                     ///< A = eigvec * diag(eigval) * eigvec^T

    const double thresh = tol * std::abs(eigval.max());
    arma::vec inv_eigval(eigval.n_elem, arma::fill::zeros);
    for (arma::uword i = 0; i < eigval.n_elem; ++i)
        if (std::abs(eigval(i)) > thresh)
            inv_eigval(i) = 1.0 / eigval(i);

    // x = eigvec * diag(inv_eigval) * eigvec^T * b
    return eigvec * arma::diagmat(inv_eigval) * eigvec.t() * b;
#endif
}

/// @}

} // namespace linalg
} // namespace qe

#endif // LIBQE_LINALG_FALLBACK_HPP
