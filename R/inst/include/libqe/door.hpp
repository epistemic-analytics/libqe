/** @file door.hpp
 *  @brief Door temporal pooling and smoothing functions for trajectory modeling.
 *
 *  Provides sliding lookback window pooling (with uniform or linear weighting,
 *  sum or mean aggregation, and optional segment boundary resets) as well as
 *  Exponential Moving Average (EMA) smoothing over consecutive time steps.
 */
#ifndef LIBQE_DOOR_HPP
#define LIBQE_DOOR_HPP

#include <armadillo>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

namespace qe {

/** @brief Apply a sliding lookback window pool over a single unit's connection matrix.
 *
 *  For each row @c i, pools rows in the window [@c max(0, i - lookback_size + 1), @c i].
 *  If @p segment_ids is non-empty, the window does not cross rows with a different segment ID.
 *
 *  @param[in] block            Matrix of connection counts for one unit (n_rows × n_features).
 *  @param[in] lookback_size    Number of rows in the lookback window (>= 1).
 *  @param[in] aggregate_mean   When @c true, averages rows; when @c false, sums rows.
 *  @param[in] weighting_linear When @c true, applies linearly increasing weights
 *                              (oldest row in window has weight 1, current row has weight k).
 *                              When @c false, all rows in window have equal weight 1.
 *  @param[in] segment_ids      Optional vector of segment/activity IDs (length n_rows).
 *                              Window pooling resets at segment transitions.
 *
 *  @returns A matrix of shape n_rows × n_features containing door-smoothed values.
 */
inline arma::mat lookback_block(
    const arma::mat& block,
    int lookback_size = 20,
    bool aggregate_mean = false,
    bool weighting_linear = false,
    const std::vector<int>& segment_ids = {}
) {
    int n_rows = static_cast<int>(block.n_rows);
    int n_cols = static_cast<int>(block.n_cols);
    arma::mat out(n_rows, n_cols, arma::fill::zeros);

    if (n_rows == 0 || n_cols == 0) return out;
    int k_max = std::max(1, lookback_size);
    bool has_segments = (!segment_ids.empty() && static_cast<int>(segment_ids.size()) >= n_rows);

    for (int i = 0; i < n_rows; ++i) {
        int start_idx = std::max(0, i - k_max + 1);

        if (has_segments) {
            int cur_seg = segment_ids[i];
            for (int s = i; s >= start_idx; --s) {
                if (segment_ids[s] != cur_seg) {
                    start_idx = s + 1;
                    break;
                }
            }
        }

        int win_len = i - start_idx + 1;
        if (win_len <= 0) continue;

        arma::rowvec acc(n_cols, arma::fill::zeros);
        arma::rowvec denom(n_cols, arma::fill::zeros);

        for (int j = start_idx; j <= i; ++j) {
            double w = 1.0;
            if (weighting_linear) {
                // Linear weight: 1 for earliest in window, win_len for current row i
                w = static_cast<double>(j - start_idx + 1);
            }
            for (int col = 0; col < n_cols; ++col) {
                double value = block(j, col);
                if (!std::isnan(value)) {
                    acc[col] += w * value;
                    denom[col] += w;
                }
            }
        }

        if (aggregate_mean) {
            for (int col = 0; col < n_cols; ++col) {
                acc[col] = (denom[col] > 0.0) ? (acc[col] / denom[col]) : arma::datum::nan;
            }
        }

        out.row(i) = acc;
    }

    return out;
}

/** @brief Apply Exponential Moving Average (EMA) smoothing over a single unit's connection matrix.
 *
 *  Computes S_0 = Y_0, S_t = alpha * Y_t + (1 - alpha) * S_{t-1}.
 *  If @p segment_ids is non-empty, S_t resets to Y_t when transitioning to a new segment.
 *
 *  @param[in] block       Matrix of connection counts for one unit (n_rows × n_features).
 *  @param[in] alpha       Smoothing parameter alpha in (0, 1].
 *  @param[in] segment_ids Optional vector of segment/activity IDs (length n_rows).
 *
 *  @returns A matrix of shape n_rows × n_features containing EMA-smoothed values.
 */
inline arma::mat ema_block(
    const arma::mat& block,
    double alpha = 0.1,
    const std::vector<int>& segment_ids = {}
) {
    int n_rows = static_cast<int>(block.n_rows);
    int n_cols = static_cast<int>(block.n_cols);
    arma::mat out(n_rows, n_cols, arma::fill::zeros);

    if (n_rows == 0 || n_cols == 0) return out;
    double a = std::max(1e-6, std::min(1.0, alpha));
    bool has_segments = (!segment_ids.empty() && static_cast<int>(segment_ids.size()) >= n_rows);

    out.row(0) = block.row(0);

    for (int i = 1; i < n_rows; ++i) {
        bool reset = has_segments && (segment_ids[i] != segment_ids[i - 1]);
        if (reset) {
            out.row(i) = block.row(i);
        } else {
            for (int col = 0; col < n_cols; ++col) {
                double cur = block(i, col);
                if (std::isnan(cur)) {
                    cur = out(i - 1, col);
                }
                out(i, col) = a * cur + (1.0 - a) * out(i - 1, col);
            }
        }
    }

    return out;
}

/** @brief Apply lookback door pooling to an entire connection matrix across multiple units.
 *
 *  @param[in] conn_counts        Complete connection matrix (all units).
 *  @param[in] unit_row_indices   Vector of row-index vectors, one per unit (0-indexed).
 *  @param[in] lookback_sizes     Vector of lookback sizes, one per unit.
 *  @param[in] aggregate_mean     When @c true, compute mean; when @c false, compute sum.
 *  @param[in] weighting_linear   When @c true, use linear weighting.
 *  @param[in] segment_ids        Optional vector of segment IDs for every row in conn_counts.
 *
 *  @returns Smoothed connection matrix with identical dimensions to @p conn_counts.
 */
inline arma::mat apply_door_lookback(
    const arma::mat& conn_counts,
    const std::vector<std::vector<int>>& unit_row_indices,
    const std::vector<int>& lookback_sizes,
    bool aggregate_mean = false,
    bool weighting_linear = false,
    const std::vector<int>& segment_ids = {}
) {
    arma::mat out = conn_counts;
    size_t n_units = unit_row_indices.size();
    bool has_segments = (!segment_ids.empty() && segment_ids.size() >= conn_counts.n_rows);

    for (size_t u = 0; u < n_units; ++u) {
        const auto& rows = unit_row_indices[u];
        if (rows.empty()) continue;

        int k = (u < lookback_sizes.size()) ? lookback_sizes[u] : 20;
        int n_u = static_cast<int>(rows.size());

        arma::mat block(n_u, conn_counts.n_cols);
        std::vector<int> block_segs(n_u, 0);

        for (int i = 0; i < n_u; ++i) {
            int r = rows[i];
            block.row(i) = conn_counts.row(r);
            if (has_segments) block_segs[i] = segment_ids[r];
        }

        arma::mat smoothed = lookback_block(block, k, aggregate_mean, weighting_linear,
                                            has_segments ? block_segs : std::vector<int>{});

        for (int i = 0; i < n_u; ++i) {
            out.row(rows[i]) = smoothed.row(i);
        }
    }

    return out;
}

/** @brief Apply EMA door smoothing to an entire connection matrix across multiple units.
 */
inline arma::mat apply_door_ema(
    const arma::mat& conn_counts,
    const std::vector<std::vector<int>>& unit_row_indices,
    double alpha = 0.1,
    const std::vector<int>& segment_ids = {}
) {
    arma::mat out = conn_counts;
    size_t n_units = unit_row_indices.size();
    bool has_segments = (!segment_ids.empty() && segment_ids.size() >= conn_counts.n_rows);

    for (size_t u = 0; u < n_units; ++u) {
        const auto& rows = unit_row_indices[u];
        if (rows.empty()) continue;

        int n_u = static_cast<int>(rows.size());
        arma::mat block(n_u, conn_counts.n_cols);
        std::vector<int> block_segs(n_u, 0);

        for (int i = 0; i < n_u; ++i) {
            int r = rows[i];
            block.row(i) = conn_counts.row(r);
            if (has_segments) block_segs[i] = segment_ids[r];
        }

        arma::mat smoothed = ema_block(block, alpha, has_segments ? block_segs : std::vector<int>{});

        for (int i = 0; i < n_u; ++i) {
            out.row(rows[i]) = smoothed.row(i);
        }
    }

    return out;
}

} // namespace qe

#endif // LIBQE_DOOR_HPP
