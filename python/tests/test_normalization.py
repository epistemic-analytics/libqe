"""Tests for pylibqe.normalization."""
import numpy as np
import pytest
from pylibqe import normalization


class TestSphereNorm:
    def test_unit_vector_unchanged(self):
        m = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        out = normalization.sphere_norm(m)
        np.testing.assert_allclose(out, m)

    def test_each_row_has_unit_norm(self):
        m = np.array([
            [3.0, 4.0],
            [1.0, 0.0],
            [0.0, 5.0],
        ], dtype=np.float64)
        out = normalization.sphere_norm(m)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0, 1.0], atol=1e-12)

    def test_zero_row_unchanged(self):
        m = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
        ], dtype=np.float64)
        out = normalization.sphere_norm(m)
        np.testing.assert_allclose(out[0], [0.0, 0.0])

    def test_shape_preserved(self):
        m = np.random.rand(5, 4)
        out = normalization.sphere_norm(m)
        assert out.shape == m.shape

    def test_known_values(self):
        m = np.array([[3.0, 4.0]], dtype=np.float64)  # norm = 5
        out = normalization.sphere_norm(m)
        np.testing.assert_allclose(out, [[0.6, 0.8]])


class TestSkipSphereNorm:
    def test_shape_preserved(self):
        m = np.random.rand(4, 3)
        out = normalization.skip_sphere_norm(m)
        assert out.shape == m.shape

    def test_largest_row_norm_becomes_one(self):
        m = np.array([
            [3.0, 4.0],   # norm = 5  (largest)
            [1.0, 0.0],   # norm = 1
        ], dtype=np.float64)
        out = normalization.skip_sphere_norm(m)
        # All entries divided by 5
        np.testing.assert_allclose(out, m / 5.0)

    def test_largest_norm_row_has_norm_one(self):
        m = np.array([
            [0.0, 3.0],
            [4.0, 0.0],
            [3.0, 4.0],  # norm = 5 (largest)
        ], dtype=np.float64)
        out = normalization.skip_sphere_norm(m)
        norms = np.linalg.norm(out, axis=1)
        assert np.isclose(norms[2], 1.0)

    def test_all_zero_unchanged(self):
        m = np.zeros((3, 3), dtype=np.float64)
        out = normalization.skip_sphere_norm(m)
        np.testing.assert_allclose(out, m)
