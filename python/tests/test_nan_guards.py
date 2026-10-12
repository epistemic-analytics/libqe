"""Tests for NaN input handling in normalization."""
import numpy as np
import pytest
from qe import normalization


# ── normalize_networks / scale_networks: silent skip, no raise ────────────────

def test_normalize_networks_nan_row_silent():
    m = np.array([[float("nan"), float("nan")], [1.0, 0.0]], dtype=np.float64)
    result = normalization.normalize_networks(m)
    assert not np.any(np.isnan(result)), "NaN row should become zeros, not propagate"
    assert result[1, 0] == pytest.approx(1.0)

def test_scale_networks_nan_row_skips_norm():
    # scale_networks divides m in-place; NaN values in the NaN row remain NaN.
    # The key guarantee is that the NaN row's norm does NOT corrupt the scale factor.
    m = np.array([[float("nan"), float("nan")], [3.0, 4.0]], dtype=np.float64)
    result = normalization.scale_networks(m)
    assert np.all(np.isnan(result[0])), "NaN row stays NaN after in-place divide"
    assert result[1, 0] == pytest.approx(3.0 / 5.0)
    assert result[1, 1] == pytest.approx(4.0 / 5.0)
