/**
 * @file rotation.hpp
 * @brief Rotation routines for Epistemic Network Analysis (ENA):
 *        SVD, deflation, orthogonal-SVD, complete (generalized), and
 *        means rotation.  All routines are designed to match the
 *        numerical output of the reference R implementation in rENA.
 */

#ifndef LIBQE_ROTATION_HPP
#define LIBQE_ROTATION_HPP

#include "linalg_fallback.hpp"
#include <armadillo>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qe {

/// @name Return types
/// @{

/**
 * @brief Aggregated result returned by every rotation routine.
 *
 * The layout mirrors the output of R's `prcomp()`: `rotation` holds the
 * right-singular vectors (principal axes), `eigenvalues` holds the
 * explained-variance values (sdev^2 in R parlance), and `column_names`
 * carries the human-readable axis labels used downstream for network plots
 * and loadings tables.
 */
struct RotationResult {
    arma::mat                rotation;      ///< p x p orthogonal matrix; column j is rotation axis j.
    arma::vec                eigenvalues;   ///< Length-p vector of sdev^2 values, matching rENA's convention.
    std::vector<std::string> column_names;  ///< Axis labels, e.g. "MR1", "SVD2", "GMR1".
};

/**
 * @brief A pair of 0-based row-index vectors identifying two groups within a
 *        points matrix.
 *
 * Used by means_rotation() to specify which rows belong to group A and which
 * belong to group B for each successive mean-difference axis.
 */
struct GroupPair {
    arma::uvec a;  ///< Row indices of group A (0-based).
    arma::uvec b;  ///< Row indices of group B (0-based).
};

/// @}

/// @name SVD rotation
/// @{

/**
 * @brief Compute a full SVD rotation of a (pre-centered) points matrix.
 *
 * Produces the right-singular vectors V as the rotation matrix and scales
 * each singular value to an eigenvalue (sdev^2) compatible with rENA's
 * `prcomp()` output.
 *
 * @param points  n x p data matrix.  The caller is responsible for
 *                centering upstream; this function applies no centering.
 *
 * @returns A RotationResult where:
 *   - `rotation`    = V  (p x p full SVD; trailing columns span the null
 *                         space when the matrix is rank-deficient).
 *   - `eigenvalues[j]` = s[j]^2 / max(1, n - 1)  (== sdev^2 in R).
 *   - `column_names`   = {"SVD1", "SVD2", ..., "SVDp"}.
 *
 * @note Equivalent to `prcomp(points, retx=FALSE, scale=FALSE,
 *       center=FALSE, tol=0)` in rENA/R.
 *
 * @note Sign convention: none.  Signs follow the underlying LAPACK SVD,
 *       matching rENA's long-standing behavior.  A deterministic sign rule
 *       (e.g. svd_flip) may be added later as an opt-in flag.
 */
inline RotationResult ena_svd(const arma::mat& points) {
    if (!points.is_finite())
        throw std::invalid_argument("ena_svd: input matrix must not contain NaN or Inf");
    arma::mat U, V;
    arma::vec s;
    qe::linalg::svd(U, s, V, points);

    const arma::uword p     = points.n_cols;
    const double      denom = points.n_rows > 1
                                  ? static_cast<double>(points.n_rows - 1)
                                  : 1.0;

    arma::vec eigenvalues(p, arma::fill::zeros);
    const arma::uword k = std::min<arma::uword>(s.n_elem, p);
    for (arma::uword j = 0; j < k; ++j) {
        eigenvalues(j) = (s(j) * s(j)) / denom;
    }

    std::vector<std::string> labels(p);
    for (arma::uword j = 0; j < p; ++j) {
        labels[j] = "SVD" + std::to_string(j + 1);
    }
    return {V, eigenvalues, std::move(labels)};
}

/// @}

/// @name Deflation
/// @{

/**
 * @brief Project a data matrix onto the hyperplane orthogonal to a given axis.
 *
 * Computes `data - (data * axis) * axis^T`, effectively removing the
 * component of every row of `data` that lies along `axis`.
 *
 * @param data  n x p data matrix.
 * @param axis  Unit-norm column vector of length p.  The caller is
 *              responsible for normalizing `axis` before calling this
 *              function; no normalization is performed internally.
 *
 * @returns n x p matrix with the projection onto `axis` subtracted out.
 */
inline arma::mat deflate(const arma::mat& data, const arma::vec& axis) {
    return data - (data * axis) * axis.t();
}

/// @}

/// @name Orthogonal SVD rotation
/// @{

/**
 * @brief Orthonormalize named axes via QR, then complete the rotation with
 *        an SVD of the data projected onto the orthogonal complement.
 *
 * The named axes in `weights` are orthonormalized via a full QR
 * decomposition.  The first k columns of the resulting Q become the
 * "fixed" part of the rotation; the remaining p - k columns define a
 * complementary subspace onto which `data` is projected, and a further SVD
 * fills those trailing rotation columns.
 *
 * @param data          n x p data matrix (caller-centered if required).
 * @param weights       p x k matrix of named axis directions.  Column norms
 *                      need not be 1; QR handles normalization internally.
 * @param named_labels  Length-k vector of labels for the first k rotation
 *                      axes.  Trailing axes receive labels "SVD{k+1}" ..
 *                      "SVDp".
 *
 * @returns A RotationResult containing the combined p x p rotation,
 *          eigenvalues for the SVD-filled trailing axes (indices k..p-1;
 *          the first k eigenvalues are zero), and axis labels.
 *
 * @note Equivalent to `orthogonal_svd()` in rENA/R/ena.rotate.by.mean.R:
 * @code
 *   Q     = qr.Q(qr(weights), complete = TRUE)   # p x p
 *   X_bar = data %*% Q[, (k+1):p]                # n x (p - k)
 *   V     = prcomp(X_bar)$rotation
 *   out   = cbind(Q[, 1:k], Q[, (k+1):p] %*% V)
 * @endcode
 *
 * @note The named axes in the OUTPUT are the orthonormalized Q columns, NOT
 *       the original `weights` columns.  Use complete_rotation() if you need
 *       to preserve input axes verbatim.
 *
 * @throws std::runtime_error if `data.n_cols != weights.n_rows` or
 *         `named_labels.size() != weights.n_cols`.
 */
inline RotationResult orthogonal_svd(
    const arma::mat&                data,
    const arma::mat&                weights,
    const std::vector<std::string>& named_labels) {
    const arma::uword p = weights.n_rows;
    const arma::uword k = weights.n_cols;
    if (data.n_cols != p) {
        throw std::runtime_error(
            "orthogonal_svd: data.n_cols must equal weights.n_rows");
    }
    if (named_labels.size() != k) {
        throw std::runtime_error(
            "orthogonal_svd: named_labels.size() must equal weights.n_cols");
    }
    if (k == 0) {
        return ena_svd(data);
    }

    arma::mat Q, R;
    qe::linalg::qr_full(Q, R, weights);                 // Q is p x p

    arma::mat rotation(p, p);
    arma::vec eigenvalues(p, arma::fill::zeros);
    rotation.cols(0, k - 1) = Q.cols(0, k - 1);

    if (k < p) {
        arma::mat Q_comp = Q.cols(k, p - 1);            // p x (p - k)
        arma::mat X_bar  = data * Q_comp;               // n x (p - k)
        RotationResult inner = ena_svd(X_bar);
        rotation.cols(k, p - 1) = Q_comp * inner.rotation;
        for (arma::uword j = 0; j < (p - k); ++j) {
            eigenvalues(k + j) = inner.eigenvalues(j);
        }
    }

    std::vector<std::string> labels(p);
    for (arma::uword j = 0; j < k; ++j) labels[j] = named_labels[j];
    for (arma::uword j = k; j < p; ++j) {
        labels[j] = "SVD" + std::to_string(j + 1);
    }
    return {rotation, eigenvalues, std::move(labels)};
}

/// @}

/// @name Complete (generalized) rotation
/// @{

/**
 * @brief Keep named axes verbatim and fill remaining axes from an SVD of the
 *        data after parallel deflation by all named axes.
 *
 * Unlike orthogonal_svd(), this routine preserves the caller-supplied
 * `named_axes` exactly (no QR re-orthonormalization).  The complementary
 * axes are obtained by deflating `data` simultaneously by all named axes and
 * then running ena_svd() on the result.
 *
 * The deflation is **parallel**: every projection is subtracted from the
 * original `data`, not from a progressively deflated copy:
 * @code
 *   defA = data - data * (named_axes * named_axes^T)
 * @endcode
 * This matches `defA <- A - A %*% v1 %*% t(v1) - A %*% v2 %*% t(v2)` in
 * rENA line-for-line.  For mutually orthogonal axes the result is identical
 * to sequential deflation; for non-orthogonal axes it differs.
 *
 * @param data          n x p data matrix.
 * @param named_axes    p x k matrix of named rotation axes.  Each column
 *                      must be unit-norm; orthonormality between columns is
 *                      NOT required.
 * @param named_labels  Length-k vector of labels for the first k axes.
 *                      Trailing axes receive labels "SVD{k+1}" .. "SVDp".
 *                      Conventional labels for generalized rotation are
 *                      "GMR1", "GMR2", then "SVD{k+1}" onward, but labels
 *                      are entirely caller-supplied.
 *
 * @returns A RotationResult where columns 0..k-1 of `rotation` equal
 *          `named_axes` verbatim and columns k..p-1 come from the SVD of
 *          the deflated data. Columns from the deflated data's null space
 *          (rank-deficient data) are orthogonalised against the named axes
 *          and each other, so the result is orthonormal whenever the named
 *          axes are.
 *
 * @note Equivalent to the tail of `ena.rotate.by.generalized` in rENA
 *       (canonical version: commit 2c079126 on rENA `origin/main`).
 *
 * @throws std::runtime_error if `data.n_cols != named_axes.n_rows` or
 *         `named_labels.size() != named_axes.n_cols`.
 */
inline RotationResult complete_rotation(
    const arma::mat&                data,
    const arma::mat&                named_axes,
    const std::vector<std::string>& named_labels) {
    const arma::uword p = named_axes.n_rows;
    const arma::uword k = named_axes.n_cols;
    if (data.n_cols != p) {
        throw std::runtime_error(
            "complete_rotation: data.n_cols must equal named_axes.n_rows");
    }
    if (named_labels.size() != k) {
        throw std::runtime_error(
            "complete_rotation: named_labels.size() must equal named_axes.n_cols");
    }
    if (k == 0) {
        return ena_svd(data);
    }

    arma::mat defA = data - data * named_axes * named_axes.t();
    RotationResult inner = ena_svd(defA);

    arma::mat rotation(p, p);
    arma::vec eigenvalues(p, arma::fill::zeros);
    rotation.cols(0, k - 1) = named_axes;
    if (k < p) {
        // The SVD columns with non-zero singular values are taken as they are
        // (rENA's pattern; with orthonormal named axes they lie in defA's row
        // space, orthogonal to the axes). When the data are rank-deficient --
        // e.g. an all-zero connection column from a code mask, or fewer units
        // than connections -- the trailing columns come from defA's null
        // space, which contains the named axes themselves, so the SVD can
        // return one that nearly duplicates a named axis (|cos| ~ 0.99): the
        // rotation is no longer orthonormal and that axis's variance is
        // counted twice. Null-space columns are therefore orthogonalised
        // against the named axes and the axes already chosen (Gram-Schmidt),
        // skipping any with nothing left. Full-rank data are unaffected.
        arma::mat basis(p, p, arma::fill::zeros);   // orthonormal span so far
        arma::uword nb = 0;
        auto add_to_basis = [&](arma::vec v) {
            for (int pass = 0; pass < 2; ++pass)
                if (nb > 0) v -= basis.cols(0, nb - 1) * (basis.cols(0, nb - 1).t() * v);
            const double len = arma::norm(v);
            if (len > 1e-10 && nb < p) basis.col(nb++) = v / len;
        };
        for (arma::uword j = 0; j < k; ++j) add_to_basis(named_axes.col(j));

        // A singular value below 1e-7 of the largest is numerically zero.
        const double max_eig  = inner.eigenvalues.is_empty() ? 0.0 : inner.eigenvalues.max();
        const double null_eig = 1e-14 * max_eig;
        arma::uword filled = 0;
        auto take = [&](arma::vec v, double eigenvalue, bool in_null_space) {
            if (filled >= p - k) return;
            if (in_null_space) {
                for (int pass = 0; pass < 2; ++pass)
                    if (nb > 0) v -= basis.cols(0, nb - 1) * (basis.cols(0, nb - 1).t() * v);
                const double len = arma::norm(v);
                if (len < 1e-8) return;        // nothing left: skip the candidate
                v /= len;
            }
            rotation.col(k + filled) = v;
            eigenvalues(k + filled) = eigenvalue;
            ++filled;
            add_to_basis(v);
        };
        for (arma::uword j = 0; j < inner.rotation.n_cols && filled < p - k; ++j)
            take(inner.rotation.col(j), inner.eigenvalues(j), inner.eigenvalues(j) <= null_eig);
        // The SVD columns span R^p, so this only runs on degenerate input.
        for (arma::uword i = 0; i < p && filled < p - k; ++i) {
            arma::vec e(p, arma::fill::zeros);
            e(i) = 1.0;
            take(e, 0.0, true);
        }
    }

    std::vector<std::string> labels(p);
    for (arma::uword j = 0; j < k; ++j) labels[j] = named_labels[j];
    for (arma::uword j = k; j < p; ++j) {
        labels[j] = "SVD" + std::to_string(j + 1);
    }
    return {rotation, eigenvalues, std::move(labels)};
}

/// @}

/// @name Means rotation
/// @{

/**
 * @brief Compute a means rotation from one or more group pairs.
 *
 * For each GroupPair, computes the normalized mean-difference vector on the
 * progressively deflated data and stacks the resulting unit-norm axes into a
 * weights matrix.  The final rotation is completed by orthogonal_svd().
 * The input `points` is mean-centered before any axes are computed, matching
 * rENA's `scale(data, scale=FALSE, center=TRUE)`.
 *
 * @param points  n x p matrix of network positions (raw, not pre-centered).
 * @param pairs   Ordered list of GroupPair objects.  Each pair contributes
 *                one "MR" axis in sequence.  Must be non-empty.
 *
 * @returns A RotationResult with `column_names` = {"MR1", ..., "MRm",
 *          "SVD{m+1}", ..., "SVDp"} where m = pairs.size().
 *
 * @note Equivalent to `ena.rotate.by.mean()` in rENA/R/ena.rotate.by.mean.R.
 *       The progressive deflation (each axis is computed on the data after
 *       removing all previous axes) is the key numerical step.
 *
 * @warning There is NO guard against a zero-norm mean-difference vector.
 *          If the two group means are identical on the current deflated data,
 *          the axis will contain NaN values (division by zero).  This mirrors
 *          a latent bug in rENA and is preserved verbatim for numerical
 *          fidelity; a safe opt-in guard will be added in a future release.
 *
 * @throws std::runtime_error if `pairs` is empty.
 */
inline RotationResult means_rotation(const arma::mat&              points,
                                      const std::vector<GroupPair>& pairs) {
    if (pairs.empty()) {
        throw std::runtime_error("Unable to rotate without 2 groups.");
    }

    arma::mat data     = points.each_row() - arma::mean(points, 0);
    arma::mat deflated = data;
    arma::mat weights(data.n_cols, pairs.size(), arma::fill::zeros);
    std::vector<std::string> labels;
    labels.reserve(pairs.size());

    for (std::size_t i = 0; i < pairs.size(); ++i) {
        const arma::rowvec mean_a = arma::mean(deflated.rows(pairs[i].a), 0);
        const arma::rowvec mean_b = arma::mean(deflated.rows(pairs[i].b), 0);
        const arma::vec    diff   = (mean_a - mean_b).t();
        const double       norm   = std::sqrt(arma::accu(diff % diff));
        const arma::vec    axis   = diff / norm;        // no zero-norm guard

        deflated        = deflate(deflated, axis);
        weights.col(i)  = axis;
        labels.push_back("MR" + std::to_string(i + 1));
    }

    return orthogonal_svd(deflated, weights, labels);
}

/// @}

}  // namespace qe

#endif  // LIBQE_ROTATION_HPP
