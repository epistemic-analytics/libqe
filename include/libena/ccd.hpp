/** @file ccd.hpp
 *  @brief Cross-Covariance Decay (CCD) window-size estimation.
 *
 *  Estimates the ENA moving-window size from the "discourse coherence length":
 *  the half-life decay lag of the noise-corrected Frobenius norm of pooled
 *  conversation cross-covariance matrices.
 *
 *  Reference: Shaffer, D. W. & Cai, Z. (2026). Discourse Coherence Length:
 *  Noise-Corrected Covariance Estimation of Window-Size in Epistemic Network
 *  Analysis. International Conference on Quantitative Ethnography (ICQE26).
 *
 *  This kernel is pure-numeric: it consumes conversation subsets that have
 *  already been extracted from the raw data (one code matrix per conversation,
 *  rows in sequence).  Column selection and conversation splitting belong in
 *  each language wrapper (cf. accumulation.hpp), so that the C++ core never
 *  touches data-frame semantics.
 */
#ifndef LIBQE_CCD_HPP
#define LIBQE_CCD_HPP

#include <armadillo>
#include <cmath>
#include <vector>

namespace qe {

/** @brief Result of a CCD window-size estimation.
 *
 *  The `*` vectors are all length `max_window + 1`, indexed by lag `0..max_window`.
 *  Entries for lags with no contributing conversation are NaN (weight 0).
 */
struct CCDResult {
    int       window_size;            ///< Estimated half-life lag window size (>= 1).
    int       peak_lag;               ///< Lag of the peak corrected cross-covariance norm.
    arma::vec lag;                    ///< Evaluated lags, 0..max_window.
    arma::vec frob;                   ///< Uncorrected pooled Frobenius norm per lag.
    arma::vec frob_sq_unbiased;       ///< Noise-corrected squared Frobenius norm per lag.
    arma::vec frob_unbiased_signed;   ///< sign * sqrt(|corrected sq|) per lag.
    arma::vec total_weight;           ///< Pooled overlap weight per lag.
};

/** @brief Estimate the ENA moving-window size via cross-covariance decay.
 *
 *  For each lag @c l in @c 0..max_window, pools window-matched cross-covariance
 *  matrices across all conversation subsets, subtracts a trace-variance noise
 *  floor, and locates the first lag strictly after the peak where the corrected
 *  norm falls to half of its peak value.
 *
 *  @param[in] conversations One code matrix per conversation (N_k rows × C codes,
 *                           rows in sequence).  All matrices must share the same
 *                           number of columns.
 *  @param[in] max_window    Maximum lag to evaluate.  Default 20.
 *  @param[in] min_overlap   Minimum overlapping rows (N_k - l) required for a
 *                           conversation to contribute at lag l.  Default 10.
 *
 *  @returns A @ref CCDResult.  When no conversation contributes at any positive
 *           lag (all-NaN or non-positive corrected covariance), the window size
 *           defaults to 1 and @c peak_lag to 0.
 */
inline CCDResult ccd_window(
    const std::vector<arma::mat>& conversations,
    int max_window = 20,
    int min_overlap = 10
) {
    if (max_window < 0) max_window = 0;
    const int n_lags = max_window + 1;

    CCDResult out;
    out.lag                  = arma::regspace<arma::vec>(0, max_window);
    out.frob                 = arma::vec(n_lags).fill(arma::datum::nan);
    out.frob_sq_unbiased     = arma::vec(n_lags).fill(arma::datum::nan);
    out.frob_unbiased_signed = arma::vec(n_lags).fill(arma::datum::nan);
    out.total_weight         = arma::zeros<arma::vec>(n_lags);
    out.window_size          = 1;
    out.peak_lag             = 0;

    // Determine code count from the first non-empty conversation.
    arma::uword C = 0;
    for (const arma::mat& X : conversations) {
        if (X.n_cols > 0) { C = X.n_cols; break; }
    }
    if (C == 0) return out;  // nothing to work with -> window 1

    // 1. Cross-covariance curves & noise floor across lags.
    for (int i = 0; i < n_lags; ++i) {
        const int l = i;
        arma::mat Cov_sum = arma::zeros<arma::mat>(C, C);
        double total_weight = 0.0;
        double noise_floor_sq_weighted_sum = 0.0;

        for (const arma::mat& X_k : conversations) {
            const int N_k = static_cast<int>(X_k.n_rows);
            const int weight = N_k - l;
            if (weight < min_overlap) continue;

            // A = rows [l, N_k-1], B = rows [0, N_k-1-l]; both have `weight` rows.
            const arma::mat A = X_k.rows(l, N_k - 1);
            const arma::mat B = X_k.rows(0, N_k - 1 - l);

            const arma::rowvec p_A = arma::mean(A, 0);
            const arma::rowvec p_B = arma::mean(B, 0);

            // Window-matched cross-covariance: crossprod(B, A)/w - outer(p_B, p_A).
            const arma::mat Cov_k = (B.t() * A) / weight - p_B.t() * p_A;

            // Trace-variance noise floor (n-1 denominator, matching R's var()).
            const double tr_var_A =
                arma::accu(arma::square(A.each_row() - p_A)) / (weight - 1);
            const double tr_var_B =
                arma::accu(arma::square(B.each_row() - p_B)) / (weight - 1);
            const double var_k_sq = tr_var_A * tr_var_B;

            Cov_sum += weight * Cov_k;
            total_weight += weight;
            noise_floor_sq_weighted_sum += weight * var_k_sq;
        }

        if (total_weight > 0.0) {
            const arma::mat Cov_pooled = Cov_sum / total_weight;
            const double frob_sq_pooled = arma::accu(arma::square(Cov_pooled));
            const double noise_floor_sq_pooled =
                noise_floor_sq_weighted_sum / (total_weight * total_weight);
            const double frob_sq_unbiased_pooled =
                frob_sq_pooled - noise_floor_sq_pooled;

            const double sgn = (frob_sq_unbiased_pooled > 0.0) ? 1.0
                             : (frob_sq_unbiased_pooled < 0.0) ? -1.0 : 0.0;

            out.frob[i]                 = std::sqrt(frob_sq_pooled);
            out.frob_sq_unbiased[i]     = frob_sq_unbiased_pooled;
            out.frob_unbiased_signed[i] = sgn * std::sqrt(std::abs(frob_sq_unbiased_pooled));
            out.total_weight[i]         = total_weight;
        }
    }

    // 2. Half-life decay lag detection over the signed corrected norm.
    const arma::vec& f = out.frob_unbiased_signed;

    // Peak among lags > 0 with finite, positive corrected norm.
    int    peak_lag = -1;
    double max_f    = -arma::datum::inf;
    for (int i = 1; i < n_lags; ++i) {
        if (std::isfinite(f[i]) && f[i] > 0.0 && f[i] > max_f) {
            max_f    = f[i];
            peak_lag = i;   // lag == index
        }
    }

    if (peak_lag < 0) {
        // No positive corrected covariance anywhere -> default window 1.
        out.window_size = 1;
        out.peak_lag    = 0;
        return out;
    }
    out.peak_lag = peak_lag;

    // First lag at or after the peak whose corrected norm has decayed to <= 50%.
    // NaN comparisons are false, so NaN lags are skipped (matching R's which()).
    int decay_lag = -1;
    for (int i = peak_lag; i < n_lags; ++i) {
        if (f[i] / max_f <= 0.5) { decay_lag = i; break; }
    }

    if (decay_lag >= 0) {
        out.window_size = decay_lag;
    } else {
        // No decay below half-life within max_window: fall back to the last lag
        // (at or after the peak) with a finite corrected norm.
        int last_valid = -1;
        for (int i = peak_lag; i < n_lags; ++i) {
            if (std::isfinite(f[i])) last_valid = i;
        }
        out.window_size = (last_valid >= 0) ? last_valid : 1;
    }

    return out;
}

} // namespace qe

#endif // LIBQE_CCD_HPP
