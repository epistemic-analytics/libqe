"""Shared fixtures for qe tests."""
import numpy as np
import pytest


@pytest.fixture
def codes_binary():
    """Small 4×3 binary code matrix used across accumulation tests."""
    return np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 0, 0],
    ], dtype=np.float64)


@pytest.fixture
def codes_continuous():
    """4×3 continuous (non-binary) code matrix."""
    return np.array([
        [2.0, 3.0, 0.0],
        [1.0, 0.0, 2.0],
        [0.0, 1.5, 1.0],
        [3.0, 0.0, 0.5],
    ], dtype=np.float64)


@pytest.fixture
def square_mat():
    """A simple 3×3 matrix."""
    return np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)
