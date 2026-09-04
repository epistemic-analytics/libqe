"""Tests for pylibqe.trajectory."""

import numpy as np

from pylibqe import trajectory


def test_fit_poly_defaults_to_orthogonal_basis_and_predicts_curve():
    t = np.linspace(0.0, 1.0, 8)
    points = np.column_stack(
        [
            0.4 + 1.7 * t - 0.9 * t**2 + 0.3 * t**3,
            -0.2 + 0.8 * t + 0.5 * t**2 - 0.4 * t**3,
        ]
    )

    fit = trajectory.fit_poly(points, t=t, fixed_degree=3)
    assert fit["degree"] == 3
    assert fit["basis"] == "orthogonal"
    assert fit["basis_alpha"].shape == (3,)
    assert fit["basis_norm2"].shape == (5,)

    got = trajectory.eval_curve(fit["coeffs_x"], fit["coeffs_y"], t)
    np.testing.assert_allclose(got, points, atol=1e-10)


def test_fit_poly_raw_basis_remains_available():
    t = np.linspace(0.0, 1.0, 5)
    points = np.column_stack([1.0 + 2.0 * t, 4.0 - 3.0 * t])

    fit = trajectory.fit_poly(points, t=t, fixed_degree=1, basis="raw")
    assert fit["basis"] == "raw"
    np.testing.assert_allclose(fit["coeffs_x"], [1, 2], atol=1e-12)
    np.testing.assert_allclose(fit["coeffs_y"], [4, -3], atol=1e-12)


def test_distances_and_signed_turn_lag():
    cx = np.array([0.0, 1.0])
    cy = np.array([0.0, 0.0])
    shifted_y = np.array([1.0, 0.0])

    assert trajectory.integrated_distance(cx, cy, cx, cy) == 0.0
    assert abs(trajectory.integrated_distance(cx, cy, cx, shifted_y) - 1.0) < 1e-12
    assert abs(trajectory.lagged_distance(cx, cy, cx, cy, lag=0.25) - 0.25) < 1e-12

    pts_a = np.array([[0.0, 0.0], [1.0, 0.2], [1.5, 0.7], [2.0, 1.0]])
    pts_b = np.array([[0.1, 0.0], [0.7, 0.1], [1.2, 0.4], [1.8, 0.9]])
    times_a = np.array([1.0, 3.0, 5.0, 8.0])
    times_b = np.array([1.0, 2.0, 4.0, 7.0])

    lag = trajectory.signed_turn_lag(pts_a, pts_b, times_a, times_b, delta=1)
    assert lag["valid_count"] == 3
    assert abs(lag["mean_distance"] - 0.3213662108262485) < 1e-12
