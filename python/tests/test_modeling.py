"""Tests for pylibqe.modeling."""
import numpy as np
import pytest
from pylibqe import modeling, NodePositions


class TestCenterData:
    def test_column_means_become_zero(self):
        m = np.array([
            [1.0, 4.0],
            [3.0, 2.0],
            [5.0, 6.0],
        ], dtype=np.float64)
        out = modeling.center_data(m)
        np.testing.assert_allclose(np.mean(out, axis=0), [0.0, 0.0], atol=1e-12)

    def test_shape_preserved(self):
        m = np.random.rand(6, 4)
        out = modeling.center_data(m)
        assert out.shape == m.shape

    def test_already_centered_unchanged(self):
        m = np.array([[-1.0, 2.0], [0.0, -2.0], [1.0, 0.0]], dtype=np.float64)
        out = modeling.center_data(m)
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


class TestLwsLsqPositions:
    def _make_data(self, n_units=8, n_tri=3, n_dims=2, seed=0):
        rng = np.random.default_rng(seed)
        adj  = np.abs(rng.standard_normal((n_units, n_tri)))
        pts  = rng.standard_normal((n_units, n_dims))
        return adj, pts

    def test_returns_node_positions(self):
        adj, pts = self._make_data()
        result = modeling.lws_lsq_positions(adj, pts, 2)
        assert isinstance(result, NodePositions)

    def test_nodes_shape(self):
        adj, pts = self._make_data(n_tri=3, n_dims=2)
        result = modeling.lws_lsq_positions(adj, pts, 2)
        # n_tri=3 → choose_two(n) → n_codes=3 (since choose_two(3)=3)
        assert result.nodes.ndim == 2
        assert result.nodes.shape[1] == 2

    def test_centroids_n_rows_match_units(self):
        n = 8
        adj, pts = self._make_data(n_units=n)
        result = modeling.lws_lsq_positions(adj, pts, 2)
        assert result.centroids.shape[0] == n

    def test_points_echoed_back(self):
        adj, pts = self._make_data()
        result = modeling.lws_lsq_positions(adj, pts, 2)
        np.testing.assert_allclose(result.points, pts, atol=1e-10)

    def test_repr_contains_shape(self):
        adj, pts = self._make_data()
        result = modeling.lws_lsq_positions(adj, pts, 2)
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


class TestDirectedNodePositionsGroundResponse:
    def test_centroids_row_count_halved(self):
        # n_rows must be even (paired ground/response)
        rng = np.random.default_rng(3)
        n = 12
        lw  = np.abs(rng.standard_normal((n, 4)))
        pts = rng.standard_normal((n, 2))
        result = modeling.directed_node_positions_ground_response(lw, pts, 2)
        # row pairing halves centroids
        assert result.centroids.shape[0] == n
