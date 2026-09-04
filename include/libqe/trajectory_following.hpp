/** @file trajectory_following.hpp
 *  @brief Who-follows-whom leader-follower turn lag analysis along shared timeline.
 */
#ifndef LIBQE_TRAJECTORY_FOLLOWING_HPP
#define LIBQE_TRAJECTORY_FOLLOWING_HPP

#include <armadillo>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace qe {

/** @brief Result of signed turn-lag sweep across delta in [-max_lag, max_lag]. */
struct SignedTurnLagResult {
    int best_lag = 0;             ///< delta* that minimizes mean distance
    double min_mean_distance = 0.0;
    arma::ivec lags;              ///< Range of evaluated lags
    arma::vec mean_distances;     ///< Mean distance for each lag
    arma::ivec valid_counts;      ///< Number of paired turns for each lag
};

/** @brief Compute mean distance between agent A at turn T and agent B at turn T - delta.
 *
 *  @param[in] pts_a    n_a × 2 coordinates for agent A.
 *  @param[in] pts_b    n_b × 2 coordinates for agent B.
 *  @param[in] times_a  Length n_a vector of calendar/turn indices for A.
 *  @param[in] times_b  Length n_b vector of calendar/turn indices for B.
 *  @param[in] delta    Signed lag in turns (delta > 0: A compares to B's past => A follows B;
 *                      delta < 0: B compares to A's past => B follows A; delta = 0: sync).
 *
 *  @returns Pair of (mean_distance, valid_pairs_count).
 */
inline std::pair<double, int> signed_turn_lag_distance(
    const arma::mat& pts_a,
    const arma::mat& pts_b,
    const arma::vec& times_a,
    const arma::vec& times_b,
    int delta
) {
    int n_a = static_cast<int>(pts_a.n_rows);
    int n_b = static_cast<int>(pts_b.n_rows);
    if (n_a == 0 || n_b == 0) return {0.0, 0};

    double sum_dist = 0.0;
    int count = 0;

    for (int i = 0; i < n_a; ++i) {
        double t_a = times_a[i];
        double target_tb = t_a - static_cast<double>(delta);

        // Find the latest turn for agent B that occurred at or before target_tb
        int best_b_idx = -1;
        for (int j = 0; j < n_b; ++j) {
            if (times_b[j] <= target_tb) {
                best_b_idx = j;
            } else {
                break;
            }
        }

        if (best_b_idx >= 0) {
            double dx = pts_a(i, 0) - pts_b(best_b_idx, 0);
            double dy = pts_a(i, 1) - pts_b(best_b_idx, 1);
            sum_dist += std::sqrt(dx * dx + dy * dy);
            count++;
        }
    }

    double mean_d = (count > 0) ? (sum_dist / count) : 0.0;
    return {mean_d, count};
}

/** @brief Sweep signed turn lags in [-max_lag, max_lag] to determine leader-follower dynamics. */
inline SignedTurnLagResult best_signed_turn_lag(
    const arma::mat& pts_a,
    const arma::mat& pts_b,
    const arma::vec& times_a,
    const arma::vec& times_b,
    int max_lag = 15
) {
    int k_max = std::max(1, max_lag);
    int n_lags = 2 * k_max + 1;

    SignedTurnLagResult res;
    res.lags.set_size(n_lags);
    res.mean_distances.set_size(n_lags);
    res.valid_counts.set_size(n_lags);

    double min_dist = std::numeric_limits<double>::infinity();
    int best_delta = 0;

    for (int idx = 0; idx < n_lags; ++idx) {
        int delta = -k_max + idx;
        res.lags[idx] = delta;

        auto [mean_d, count] = signed_turn_lag_distance(pts_a, pts_b, times_a, times_b, delta);
        res.mean_distances[idx] = mean_d;
        res.valid_counts[idx] = count;

        if (count > 0 && mean_d < min_dist) {
            min_dist = mean_d;
            best_delta = delta;
        }
    }

    res.best_lag = best_delta;
    res.min_mean_distance = (min_dist == std::numeric_limits<double>::infinity()) ? 0.0 : min_dist;
    return res;
}

} // namespace qe

#endif // LIBQE_TRAJECTORY_FOLLOWING_HPP
