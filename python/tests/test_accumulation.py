"""Tests for qe.accumulation."""
import numpy as np
import pytest
from qe import accumulation


class TestConnectionMatrix:
    def test_directed_shape(self):
        g = np.array([1.0, 0.0, 1.0])
        r = np.array([0.0, 1.0, 1.0])
        out = accumulation.connection_matrix(g, r, ordered=True)
        assert out.shape == (3, 3)

    def test_zero_ground_gives_response_self(self):
        g = np.zeros(3)
        r = np.array([0.0, 0.0, 1.0])
        out = accumulation.connection_matrix(g, r, response_weight=1.0, ordered=True)
        # g→r cross-product is zero; only 0.5 * r⊗r (diagonal zeroed)
        assert out[2, 2] == 0.0   # diagonal zeroed in directed mode

    def test_undirected_is_symmetric(self):
        g = np.array([1.0, 0.5, 0.0])
        r = np.array([0.0, 1.0, 0.5])
        out = accumulation.connection_matrix(g, r, ordered=False)
        np.testing.assert_allclose(out, out.T, atol=1e-12)


class TestAccumulateStanza:
    def test_output_shape(self, codes_binary):
        from qe import adjacency
        out = accumulation.accumulate_stanza(codes_binary, window_back=1)
        assert out.shape == (codes_binary.shape[0],
                             adjacency.choose_two(codes_binary.shape[1]))

    def test_window_1_matches_row_connections(self, codes_binary):
        # window_back=1, window_forward=0, binary=True should equal
        # row_connections because there's no context to accumulate.
        sw  = accumulation.accumulate_stanza(codes_binary, window_back=1, binary=True)
        rco = accumulation.row_connections(codes_binary, binary=True)
        np.testing.assert_allclose(sw, rco)

    def test_all_zero_codes(self):
        codes = np.zeros((5, 4), dtype=np.float64)
        out = accumulation.accumulate_stanza(codes, window_back=2)
        assert np.all(out == 0)

    def test_binary_output_is_01(self, codes_binary):
        out = accumulation.accumulate_stanza(codes_binary, window_back=2, binary=True)
        assert set(np.unique(out)).issubset({0.0, 1.0})

    def test_larger_window_increases_nonzero(self, codes_binary):
        out1 = accumulation.accumulate_stanza(codes_binary, window_back=1, binary=True)
        out2 = accumulation.accumulate_stanza(codes_binary, window_back=4, binary=True)
        # More context → at least as many non-zero entries
        assert np.sum(out2 > 0) >= np.sum(out1 > 0)


class TestRowConnections:
    def test_output_dimensions(self, codes_binary):
        from qe import adjacency
        out = accumulation.row_connections(codes_binary)
        assert out.shape == (4, adjacency.choose_two(3))

    def test_binary_binarises_nonzero(self, codes_continuous):
        out = accumulation.row_connections(codes_continuous, binary=True)
        assert set(np.unique(out)).issubset({0.0, 1.0})

    def test_non_binary_preserves_products(self):
        codes = np.array([[2.0, 3.0, 0.0]], dtype=np.float64)
        out = accumulation.row_connections(codes, binary=False)
        # 2*3=6, 2*0=0, 3*0=0
        np.testing.assert_allclose(out[0], [6.0, 0.0, 0.0])

    def test_all_zero_gives_all_zero(self):
        codes = np.zeros((4, 3), dtype=np.float64)
        out = accumulation.row_connections(codes)
        assert np.all(out == 0)

    def test_rows_are_independent(self):
        # Row 1 has c1; row 2 has c2 → neither should show c1&c2 co-occurrence
        codes = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]], dtype=np.float64)
        out = accumulation.row_connections(codes, binary=False)
        assert np.all(out == 0)


class TestRollingWindowSum:
    def test_window_1_returns_input(self, codes_binary):
        out = accumulation.rolling_window_sum(codes_binary, window_size=1)
        np.testing.assert_allclose(out, codes_binary)

    def test_window_2_sums_adjacent_rows(self):
        codes = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ], dtype=np.float64)
        out = accumulation.rolling_window_sum(codes, window_size=2)
        np.testing.assert_allclose(out[0], [1.0, 0.0])   # row 0 only
        np.testing.assert_allclose(out[1], [1.0, 1.0])   # rows 0+1
        np.testing.assert_allclose(out[2], [1.0, 1.0])   # rows 1+2

    def test_large_window_clamps_to_start(self):
        codes = np.array([[1.0, 0.0],
                          [1.0, 1.0]], dtype=np.float64)
        out = accumulation.rolling_window_sum(codes, window_size=100)
        np.testing.assert_allclose(out[1], [2.0, 1.0])

    def test_shape_preserved(self, codes_binary):
        out = accumulation.rolling_window_sum(codes_binary, window_size=3)
        assert out.shape == codes_binary.shape


class TestFlatIndex:
    def test_simple_1d(self):
        # Single dimension: index 2 in a size-5 array → 2
        assert accumulation.flat_index([2], [5]) == 2

    def test_2d_column_major(self):
        # 2D column-major: [row, col] in a 3×4 matrix
        # col-major: linear = row + col * n_rows
        assert accumulation.flat_index([1, 2], [3, 4]) == 1 + 2 * 3  # == 7

    def test_first_element_is_zero(self):
        assert accumulation.flat_index([0, 0], [3, 3]) == 0

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(Exception):
            accumulation.flat_index([0, 1], [3])
