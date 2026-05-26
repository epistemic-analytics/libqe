#ifndef LIBQE_ROTATION_HPP
#define LIBQE_ROTATION_HPP

#include <armadillo>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qe {

// ---------------------------------------------------------------------------
// Return type for rotation routines
// ---------------------------------------------------------------------------

struct RotationResult {
    arma::mat                rotation;       // p x p, column j = rotation axis j
    arma::vec                eigenvalues;    // length p; matches rENA's sdev^2
    std::vector<std::string> column_names;   // "MR1", "SVD2", ...
};

// Pair of 0-based row-index vectors identifying two groups in a points matrix.
struct GroupPair {
    arma::uvec a;
    arma::uvec b;
};

// ---------------------------------------------------------------------------
// SVD rotation
// ---------------------------------------------------------------------------

// Equivalent to rENA's prcomp(points, retx=FALSE, scale=FALSE, center=FALSE, tol=0):
//   rotation       = V (p x p, full SVD; trailing cols span null space if rank-deficient)
//   eigenvalues[j] = singular_value[j]^2 / max(1, n - 1)   == sdev^2
// Caller is responsible for centering upstream.
//
// Sign convention: none. Signs come from the underlying LAPACK SVD, matching
// rENA's long-standing behavior. A deterministic sign rule (e.g. svd_flip)
// may be added later as an opt-in flag.
inline RotationResult ena_svd(const arma::mat& points) {
    arma::mat U, V;
    arma::vec s;
    arma::svd(U, s, V, points);

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

// ---------------------------------------------------------------------------
// Deflation
// ---------------------------------------------------------------------------

// Project `data` onto the hyperplane orthogonal to `axis` (unit-norm column):
//   data - (data * axis) * axis^T
// Caller is responsible for normalizing `axis`.
inline arma::mat deflate(const arma::mat& data, const arma::vec& axis) {
    return data - (data * axis) * axis.t();
}

// ---------------------------------------------------------------------------
// Orthogonal SVD — orthonormalizes named axes via QR, then completes the
// rotation from an SVD of the data projected onto the orthogonal complement.
// ---------------------------------------------------------------------------
//
// Mirrors orthogonal_svd() in rENA/R/ena.rotate.by.mean.R:
//   Q     = qr.Q(qr(weights), complete = TRUE)        // p x p
//   X_bar = data %*% Q[, k+1:p]                       // n x (p - k)
//   V     = prcomp(X_bar)$rotation
//   out   = cbind(Q[, 1:k], Q[, k+1:p] %*% V)
//
// IMPORTANT: the named axes in the OUTPUT are the orthonormalized Q columns,
// not the original `weights` columns. Use complete_rotation() if you need to
// preserve the input axes verbatim.
//
// `weights`       : p x k       (column-norm not required; QR handles it)
// `named_labels`  : length k    (trailing labels become "SVD{k+1}".."SVDp")
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
    arma::qr(Q, R, weights);                            // Q is p x p

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

// ---------------------------------------------------------------------------
// Complete rotation — keep named axes verbatim, fill remaining axes from an
// SVD of the data deflated by all named axes.
// ---------------------------------------------------------------------------
//
// Mirrors the tail of rENA's ena.rotate.by.generalized (canonical version:
// commit 2c079126 on rENA `origin/main`). The deflation is *parallel* — each
// projection comes off the original `data`, not from a progressively
// deflated copy:
//
//   defA     = data - data * (sum_j a_j * a_j^T)        (parallel)
//   svd_v    = prcomp(defA)$rotation
//   combined = cbind(named_axes, svd_v[, 1:(p - k)])
//
// This matches `defA <- A - A %*% v1 %*% t(v1) - A %*% v2 %*% t(v2)` line-
// for-line. For mutually orthogonal axes the result equals sequential
// deflation; for non-orthogonal axes it differs.
//
// Caller is responsible for ensuring each column of `named_axes` is unit-norm.
// Orthonormality between columns is NOT assumed.
//
// Conventional column labels for generalized rotation are "GMR1", "GMR2",
// then "SVD{k+1}".."SVDp". The labels are caller-supplied; libqe doesn't bake
// them in.
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
        rotation.cols(k, p - 1) = inner.rotation.cols(0, p - k - 1);
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

// ---------------------------------------------------------------------------
// Means rotation
// ---------------------------------------------------------------------------
//
// Mirrors rENA/R/ena.rotate.by.mean.R. For each group pair, computes a
// normalized mean-difference axis on the (progressively deflated) data and
// stacks the axes into a weights matrix; finishes with orthogonal_svd().
// Centers `points` first to match rENA's `scale(data, scale=F, center=T)`.
//
// MATCH-RENA NOTE: there is NO guard against a zero-norm mean-difference
// vector — this is a latent bug carried forward from rENA verbatim. It will
// be addressed as a separate, opt-in change later.
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

}  // namespace qe

#endif  // LIBQE_ROTATION_HPP
