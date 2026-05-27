"""Tests for pylibqe.modeling."""
import numpy as np
import pytest
from pylibqe import modeling, NodePositions


class TestMeanCI:
    def test_output_shape(self):
        rng = np.random.default_rng(10)
        pts = rng.standard_normal((20, 2))
        out = modeling.mean_ci(pts, 0.95)
        assert out.shape == (2, 3)

    def test_lower_le_mean_le_upper(self):
        rng = np.random.default_rng(11)
        pts = rng.standard_normal((15, 3))
        out = modeling.mean_ci(pts, 0.95)
        assert np.all(out[:, 1] <= out[:, 0] + 1e-12)   # lower <= mean
        assert np.all(out[:, 0] <= out[:, 2] + 1e-12)   # mean  <= upper

    def test_mean_column_matches_numpy(self):
        rng = np.random.default_rng(12)
        pts = rng.random((30, 2))
        out = modeling.mean_ci(pts, 0.95)
        np.testing.assert_allclose(out[:, 0], np.mean(pts, axis=0), atol=1e-10)

    def test_wider_conf_level_gives_wider_ci(self):
        rng = np.random.default_rng(13)
        pts = rng.standard_normal((20, 2))
        lo = modeling.mean_ci(pts, 0.80)
        hi = modeling.mean_ci(pts, 0.99)
        assert np.all((hi[:, 2] - hi[:, 1]) > (lo[:, 2] - lo[:, 1]))

    def test_n1_gives_inf_bounds(self):
        pts = np.array([[1.5, 2.5]], dtype=np.float64)
        out = modeling.mean_ci(pts, 0.95)
        assert np.isinf(out[0, 1])   # lower = -inf
        assert np.isinf(out[0, 2])   # upper = +inf

    def test_matches_scipy_ttest(self):
        """CI bounds should match scipy.stats.t.interval for each dimension."""
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(14)
        pts = rng.standard_normal((12, 2))
        out = modeling.mean_ci(pts, 0.95)
        for d in range(2):
            col  = pts[:, d]
            n    = len(col)
            lo, hi = stats.t.interval(0.95, df=n - 1,
                                       loc=np.mean(col),
                                       scale=stats.sem(col))
            np.testing.assert_allclose(out[d, 1], lo, rtol=1e-7)
            np.testing.assert_allclose(out[d, 2], hi, rtol=1e-7)

    def test_ci_tightens_with_more_data(self):
        rng = np.random.default_rng(15)
        small = rng.standard_normal((5,  2))
        large = rng.standard_normal((50, 2))
        w_small = np.mean(modeling.mean_ci(small, 0.95)[:, 2] - modeling.mean_ci(small, 0.95)[:, 1])
        w_large = np.mean(modeling.mean_ci(large, 0.95)[:, 2] - modeling.mean_ci(large, 0.95)[:, 1])
        assert w_large < w_small


class TestOutlierCI:
    def test_output_shape(self):
        rng = np.random.default_rng(20)
        pts = rng.standard_normal((20, 3))
        out = modeling.outlier_ci(pts, 1.5)
        assert out.shape == (3, 2)

    def test_symmetric_around_zero(self):
        rng = np.random.default_rng(21)
        pts = rng.standard_normal((30, 2))
        out = modeling.outlier_ci(pts)
        np.testing.assert_allclose(out[:, 0], -out[:, 1], atol=1e-14)

    def test_matches_rena_iqr_formula(self):
        """outlier_ci must exactly reproduce rENA's IQR * 1.5 calculation."""
        rng = np.random.default_rng(42)
        pts = rng.standard_normal((20, 2))
        out = modeling.outlier_ci(pts, 1.5)
        # numpy's linear percentile == R's type-7 quantile
        q1 = np.percentile(pts, 25, axis=0, method="linear")
        q3 = np.percentile(pts, 75, axis=0, method="linear")
        half = (q3 - q1) * 1.5
        np.testing.assert_allclose(out[:, 0], -half, rtol=1e-10)
        np.testing.assert_allclose(out[:, 1],  half, rtol=1e-10)

    def test_iqr_factor_scales_proportionally(self):
        rng = np.random.default_rng(22)
        pts = rng.standard_normal((25, 2))
        out15 = modeling.outlier_ci(pts, 1.5)
        out30 = modeling.outlier_ci(pts, 3.0)
        np.testing.assert_allclose(out30, out15 * 2, atol=1e-14)

    def test_n0_returns_nan(self):
        pts = np.zeros((0, 2), dtype=np.float64)
        out = modeling.outlier_ci(pts)
        assert out.shape == (2, 2)
        assert np.all(np.isnan(out))


class TestCenterPoints:
    def test_column_means_become_zero(self):
        m = np.array([
            [1.0, 4.0],
            [3.0, 2.0],
            [5.0, 6.0],
        ], dtype=np.float64)
        out = modeling.center_points(m)
        np.testing.assert_allclose(np.mean(out, axis=0), [0.0, 0.0], atol=1e-12)

    def test_shape_preserved(self):
        m = np.random.rand(6, 4)
        out = modeling.center_points(m)
        assert out.shape == m.shape

    def test_already_centered_unchanged(self):
        m = np.array([[-1.0, 2.0], [0.0, -2.0], [1.0, 0.0]], dtype=np.float64)
        out = modeling.center_points(m)
        np.testing.assert_allclose(out, m, atol=1e-12)


class TestEnaCorrelation:
    def test_output_shape(self):
        rng = np.random.default_rng(42)
        n = 10
        points    = rng.standard_normal((n, 2))
        centroids = rng.standard_normal((n, 2))
        out = modeling.ena_correlation(points, centroids)
        assert out.shape == (2, 3)

    def test_perfect_correlation(self):
        pts = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
        # centroids identical to points → r should be 1.0
        out = modeling.ena_correlation(pts, pts)
        assert np.isclose(out[0, 0], 1.0, atol=1e-6)

    def test_columns_are_r_lo_hi(self):
        rng = np.random.default_rng(7)
        pts = rng.standard_normal((20, 2))
        cts = rng.standard_normal((20, 2))
        out = modeling.ena_correlation(pts, cts)
        # ci_lower <= r <= ci_upper for each dim
        assert np.all(out[:, 1] <= out[:, 0] + 1e-9)
        assert np.all(out[:, 0] <= out[:, 2] + 1e-9)


class TestNodePositions:
    def _make_data(self, n_units=8, n_tri=3, n_dims=2, seed=0):
        rng = np.random.default_rng(seed)
        adj  = np.abs(rng.standard_normal((n_units, n_tri)))
        pts  = rng.standard_normal((n_units, n_dims))
        return adj, pts

    def test_returns_node_positions(self):
        adj, pts = self._make_data()
        result = modeling.node_positions(adj, pts, 2)
        assert isinstance(result, NodePositions)

    def test_nodes_shape(self):
        adj, pts = self._make_data(n_tri=3, n_dims=2)
        result = modeling.node_positions(adj, pts, 2)
        # n_tri=3 → choose_two(n) → n_codes=3 (since choose_two(3)=3)
        assert result.nodes.ndim == 2
        assert result.nodes.shape[1] == 2

    def test_centroids_n_rows_match_units(self):
        n = 8
        adj, pts = self._make_data(n_units=n)
        result = modeling.node_positions(adj, pts, 2)
        assert result.centroids.shape[0] == n

    def test_points_echoed_back(self):
        adj, pts = self._make_data()
        result = modeling.node_positions(adj, pts, 2)
        np.testing.assert_allclose(result.points, pts, atol=1e-10)

    def test_repr_contains_shape(self):
        adj, pts = self._make_data()
        result = modeling.node_positions(adj, pts, 2)
        assert "NodePositions" in repr(result)


class TestDirectedNodePositions:
    def test_returns_node_positions(self):
        rng = np.random.default_rng(1)
        lw  = np.abs(rng.standard_normal((6, 4)))   # 4 = 2*2 directed pairs
        pts = rng.standard_normal((6, 2))
        result = modeling.directed_node_positions(lw, pts, 2)
        assert isinstance(result, NodePositions)

    def test_centroids_row_count(self):
        rng = np.random.default_rng(2)
        n = 10
        lw  = np.abs(rng.standard_normal((n, 4)))
        pts = rng.standard_normal((n, 2))
        result = modeling.directed_node_positions(lw, pts, 2)
        assert result.centroids.shape[0] == n


class TestDirectedNodePositionsCombinePairs:
    def test_centroids_row_count_halved(self):
        # n_rows must be even (paired ground/response)
        rng = np.random.default_rng(3)
        n = 12
        lw  = np.abs(rng.standard_normal((n, 4)))
        pts = rng.standard_normal((n, 2))
        result = modeling.directed_node_positions_combine_pairs(lw, pts, 2)
        # row pairing halves centroids
        assert result.centroids.shape[0] == n
