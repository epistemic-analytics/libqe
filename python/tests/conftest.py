"""Shared fixtures for qe tests."""
import numpy as np
import pytest


@pytest.fixture
def square_mat():
    """A simple 3×3 matrix."""
    return np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)
