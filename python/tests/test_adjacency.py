"""Tests for pylibqe.adjacency."""
import numpy as np
import pytest
from pylibqe import adjacency


class TestChooseTwo:
    def test_zero(self):
        assert adjacency.choose_two(0) == 0

    def test_two(self):
        assert adjacency.choose_two(2) == 1

    def test_three(self):
        assert adjacency.choose_two(3) == 3

    def test_four(self):
        assert adjacency.choose_two(4) == 6

    def test_matches_formula(self):
        for n in range(2, 10):
            assert adjacency.choose_two(n) == n * (n - 1) // 2


class TestConnectionIndices:
    def test_both_rows_shape(self):
        idx = adjacency.connection_indices(3, -1)
        assert idx.shape == (2, 3)   # choose_two(3)=3 pairs

    def test_row_indices_only(self):
        idx = adjacency.connection_indices(3, 0)
        assert idx.shape == (1, 3)

    def test_col_indices_only(self):
        idx = adjacency.connection_indices(3, 1)
        assert idx.shape == (1, 3)

    def test_values_are_valid_indices(self):
        idx = adjacency.connection_indices(4, -1)
        assert np.all(idx < 4)
        assert np.all(idx >= 0)

    def test_row_lt_col(self):
        """Upper triangle means row index < col index for every pair."""
        idx = adjacency.connection_indices(5, -1)
        rows = idx[0]
        cols = idx[1]
        assert np.all(rows < cols)


class TestCodeConnections:
    def test_two_elements(self):
        v = np.array([2.0, 3.0])
        out = adjacency.code_connections(v)
        assert out.shape == (1,)
        np.testing.assert_allclose(out, [6.0])

    def test_three_elements(self):
        v = np.array([1.0, 2.0, 3.0])
        out = adjacency.code_connections(v)
        # pairs: 1*2, 1*3, 2*3
        assert out.shape == (3,)
        np.testing.assert_allclose(out, [2.0, 3.0, 6.0])

    def test_zero_vector(self):
        v = np.zeros(4)
        out = adjacency.code_connections(v)
        assert np.all(out == 0)

    def test_output_length(self):
        for n in range(2, 7):
            v = np.ones(n)
            out = adjacency.code_connections(v)
            assert len(out) == adjacency.choose_two(n)


class TestFoldDirectedNetwork:
    def test_symmetric_matrix(self):
        # 2x2 identity → flat [1,0,0,1] → upper tri = [0+0] = sum off-diag
        v = np.array([1.0, 0.0, 0.0, 1.0])
        out = adjacency.fold_directed_network(v)
        assert out.shape == (1,)
        np.testing.assert_allclose(out, [0.0])

    def test_asymmetric_summing(self):
        # [[0,3],[2,0]] → upper tri should be 3+2=5
        v = np.array([0.0, 2.0, 3.0, 0.0])
        out = adjacency.fold_directed_network(v)
        np.testing.assert_allclose(out, [5.0])

    def test_output_length_3x3(self):
        v = np.zeros(9)
        out = adjacency.fold_directed_network(v)
        assert len(out) == 3   # choose_two(3)


class TestNetworkToVector:
    def test_full_3x3(self):
        m = np.eye(3, dtype=np.float64)
        out = adjacency.network_to_vector(m, full=True)
        assert out.shape == (9,)

    def test_upper_tri_3x3(self):
        m = np.eye(3, dtype=np.float64)
        out = adjacency.network_to_vector(m, full=False)
        assert len(out) == 3   # choose_two(3)

    def test_full_default_is_true(self):
        m = np.eye(3, dtype=np.float64)
        assert len(adjacency.network_to_vector(m)) == 9


class TestConnectionNames:
    def test_two_codes(self):
        result = adjacency.connection_names(["A", "B"])
        assert result == ["A & B"]

    def test_three_codes(self):
        result = adjacency.connection_names(["X", "Y", "Z"])
        assert result == ["X & Y", "X & Z", "Y & Z"]

    def test_output_length(self):
        codes = ["c1", "c2", "c3", "c4"]
        result = adjacency.connection_names(codes)
        assert len(result) == adjacency.choose_two(len(codes))
