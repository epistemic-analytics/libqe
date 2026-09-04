"""Tests for pylibqe.door."""

import numpy as np

from pylibqe import door


class TestLookbackBlock:
    def test_sum_and_mean(self):
        mat = np.array(
            [
                [1.0, 0.0],
                [2.0, 1.0],
                [3.0, 0.0],
            ]
        )

        summed = door.lookback_block(mat, lookback_size=2, aggregate_mean=False)
        np.testing.assert_allclose(summed, [[1, 0], [3, 1], [5, 1]])

        meaned = door.lookback_block(mat, lookback_size=2, aggregate_mean=True)
        np.testing.assert_allclose(meaned, [[1, 0], [1.5, 0.5], [2.5, 0.5]])

    def test_na_values_match_etm_reference_behavior(self):
        mat = np.array(
            [
                [1.0, 1.0],
                [np.nan, 3.0],
                [5.0, np.nan],
            ]
        )

        summed = door.lookback_block(mat, lookback_size=3, aggregate_mean=False)
        np.testing.assert_allclose(summed, [[1, 1], [1, 4], [6, 4]])

        meaned = door.lookback_block(mat, lookback_size=3, aggregate_mean=True)
        np.testing.assert_allclose(meaned, [[1, 1], [1, 2], [3, 2]])


class TestEmaBlock:
    def test_na_values_carry_previous_smoothed_value(self):
        mat = np.array(
            [
                [10.0, 0.0],
                [np.nan, 10.0],
                [0.0, np.nan],
            ]
        )

        out = door.ema_block(mat, alpha=0.5)
        np.testing.assert_allclose(out, [[10, 0], [10, 5], [5, 5]])
