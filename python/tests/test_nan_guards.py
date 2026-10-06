"""Tests for NaN/Inf input guards across normalization, modeling, and rotation."""
import math
import numpy as np
import pytest
from qe import modeling, normalization, rotation


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_adj():
    rng = np.random.default_rng(0)
    return rng.random((5, 6)).astype(np.float64)

@pytest.fixture
def clean_points():
    rng = np.random.default_rng(1)
    return rng.random((5, 2)).astype(np.float64)

def _with_nan(m):
    out = m.copy()
    out[0, 0] = float("nan")
    return out

def _with_inf(m):
    out = m.copy()
    out[1, 1] = float("inf")
    return out


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


# ── node_positions ────────────────────────────────────────────────────────────

def test_node_positions_nan_adj_raises(clean_adj, clean_points):
    with pytest.raises((ValueError, Exception), match="NaN or Inf"):
        modeling.node_positions(_with_nan(clean_adj), clean_points, 2)

def test_node_positions_inf_adj_raises(clean_adj, clean_points):
    with pytest.raises((ValueError, Exception), match="NaN or Inf"):
        modeling.node_positions(_with_inf(clean_adj), clean_points, 2)

def test_node_positions_nan_t_raises(clean_adj, clean_points):
    with pytest.raises((ValueError, Exception), match="NaN or Inf"):
        modeling.node_positions(clean_adj, _with_nan(clean_points), 2)

def test_node_positions_clean_succeeds(clean_adj, clean_points):
    result = modeling.node_positions(clean_adj, clean_points, 2)
    assert result.nodes.shape[1] == 2


# ── directed_node_positions ───────────────────────────────────────────────────

def test_directed_node_positions_nan_raises(clean_points):
    rng = np.random.default_rng(2)
    lw = rng.random((5, 4)).astype(np.float64)
    with pytest.raises((ValueError, Exception), match="NaN or Inf"):
        modeling.directed_node_positions(_with_nan(lw), clean_points, 2)

def test_directed_node_positions_inf_points_raises(clean_points):
    rng = np.random.default_rng(3)
    lw = rng.random((5, 4)).astype(np.float64)
    with pytest.raises((ValueError, Exception), match="NaN or Inf"):
        modeling.directed_node_positions(lw, _with_inf(clean_points), 2)


# ── ena_svd ───────────────────────────────────────────────────────────────────

def test_ena_svd_nan_raises(clean_points):
    with pytest.raises((ValueError, Exception), match="NaN or Inf"):
        rotation.ena_svd(_with_nan(clean_points))

def test_ena_svd_inf_raises(clean_points):
    with pytest.raises((ValueError, Exception), match="NaN or Inf"):
        rotation.ena_svd(_with_inf(clean_points))

def test_ena_svd_clean_succeeds(clean_points):
    result = rotation.ena_svd(clean_points)
    assert result.rotation.shape == (2, 2)
    assert all(math.isfinite(v) for v in result.eigenvalues)
