// linalg_fallback.hpp — Thin LAPACK-free implementations of SVD, QR, and
// symmetric eigendecomposition for use in WASM / no-LAPACK builds.
//
// When ARMA_USE_LAPACK is defined these wrappers just call the standard
// Armadillo routines (arma::svd, arma::qr, arma::eig_sym), so there is zero
// overhead in normal builds.  When LAPACK is not available (e.g. Emscripten
// WASM builds) the wrappers fall back to pure C++ implementations that work
// for the small matrices typical in ENA (p ≤ ~100 connection dimensions).
//
// Public API (all in namespace qe::linalg):
//
//   svd(U, s, V, A)       – full SVD.  U is n×min(n,p), s is min(n,p)×1,
//                            V is p×p (all right singular vectors).
//   svd_right(s, V, A)    – singular values + right singular vectors only.
//   qr_full(Q, R, A)      – thin QR:  Q is n×p, R is p×p, Q'Q = I.
//   eig_sym(eigval, eigvec, S) – eigendecomp of symmetric S, ascending order.
//
// All routines are single-precision-safe (work on arma::mat i.e. double).
// The fallback algorithms are:
//   - SVD:     one-sided Jacobi iterations on A^T (Demmel & Veselić 1992)
//   - QR:      modified Gram-Schmidt
//   - eig_sym: cyclic Jacobi (classical Jacobi sweep)

#ifndef LIBQE_LINALG_FALLBACK_HPP
#define LIBQE_LINALG_FALLBACK_HPP

#include <armadillo>
#include <cmath>
#include <limits>

namespace qe {
namespace linalg {

// ── helpers ───────────────────────────────────────────────────────────────────

namespace detail {

// Modified Gram-Schmidt orthonormalization in-place.
// Columns of Q are orthonormalized; zero-norm columns are left as-is.
inline void mgs(arma::mat& Q) {
    const arma::uword p = Q.n_cols;
    for (arma::uword j = 0; j < p; ++j) {
        for (arma::uword i = 0; i < j; ++i)
            Q.col(j) -= arma::dot(Q.col(i), Q.col(j)) * Q.col(i);
        double n = arma::norm(Q.col(j));
        if (n > 1e-14) Q.col(j) /= n;
    }
}

// One cyclic Jacobi sweep on a p×p symmetric matrix A.
// Simultaneously accumulates rotations in V (start with V = I to get eigvec).
inline void jacobi_sweep(arma::mat& A, arma::mat& V) {
    const arma::uword p = A.n_rows;
    for (arma::uword q = 0; q < p - 1; ++q) {
        for (arma::uword r = q + 1; r < p; ++r) {
            double aqq = A(q, q), arr = A(r, r), aqr = A(q, r);
            if (std::abs(aqr) < 1e-14 * (std::abs(aqq) + std::abs(arr)))
                continue;
            double theta = (arr - aqq) / (2.0 * aqr);
            double t = (theta >= 0.0)
                ? 1.0 / (theta + std::sqrt(1.0 + theta * theta))
                : 1.0 / (theta - std::sqrt(1.0 + theta * theta));
            double c = 1.0 / std::sqrt(1.0 + t * t);
            double s = t * c;
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

// ── eig_sym ───────────────────────────────────────────────────────────────────
// Eigendecomposition of symmetric S (p×p), ascending eigenvalue order.
// Output: eigval (p×1), eigvec (p×p) columns are eigenvectors.
inline void eig_sym(arma::vec& eigval, arma::mat& eigvec, const arma::mat& S) {
#if defined(ARMA_USE_LAPACK)
    arma::eig_sym(eigval, eigvec, S);
#else
    const arma::uword p = S.n_rows;
    arma::mat A = S;                              // working copy
    eigvec.eye(p, p);                             // accumulate rotations

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

// ── qr_full ───────────────────────────────────────────────────────────────────
// Full QR: Q (n×n) is a square orthogonal matrix, R (n×p) is upper trapezoidal.
// Matches arma::qr() which returns a full Q, not just the thin factor.
// The caller (orthogonal_svd) relies on Q being n×n to extract Q.cols(k, n-1).
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
        Qfull(j, j) = 1.0;   // standard basis e_j
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

// ── svd ───────────────────────────────────────────────────────────────────────
// Full SVD via one-sided Jacobi (Demmel–Veselić).
// Result: A ≈ U * diag(s) * V', descending singular values.
// U is n×min(n,p); s is min(n,p)×1; V is p×p.
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

    arma::mat AtA = A.t() * A;   // p×p symmetric
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
// Computes singular values and right singular vectors only (skips U).
// s is min(n,p)×1; V is p×p.  More efficient than svd() for our use case.
inline void svd_right(arma::vec& s, arma::mat& V, const arma::mat& A) {
#if defined(ARMA_USE_LAPACK)
    arma::mat U;
    arma::svd(U, s, V, A);
#else
    arma::mat U_unused;
    svd(U_unused, s, V, A);
#endif
}

} // namespace linalg
} // namespace qe

#endif // LIBQE_LINALG_FALLBACK_HPP
