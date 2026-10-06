"""Parametric trajectory fitting, derivatives, distances, and lag analysis."""

from collections.abc import Sequence
from typing import Annotated, Any

import numpy
from numpy.typing import NDArray

FloatVector = Annotated[NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu")]
FloatMatrix = Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")]


def fit_poly(
    points: FloatMatrix,
    t: FloatVector | None = None,
    max_degree: int = 3,
    fixed_degree: int = 0,
    criterion: str = "loocv",
    basis: str = "orthogonal",
) -> dict[str, Any]:
    """Fit a 2D parametric polynomial curve."""


def eval_curve(coeffs_x: FloatVector, coeffs_y: FloatVector, t_eval: FloatVector) -> FloatMatrix:
    """Evaluate a fitted trajectory curve."""


def eval_derivatives(coeffs_x: FloatVector, coeffs_y: FloatVector, t_eval: FloatVector) -> dict[str, FloatVector]:
    """Evaluate trajectory velocities, accelerations, speed, heading rate, and curvature."""


def integrated_distance(
    coeffs_ax: FloatVector,
    coeffs_ay: FloatVector,
    coeffs_bx: FloatVector,
    coeffs_by: FloatVector,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> float:
    """Compute integrated Euclidean distance between two polynomial curves."""


def lagged_distance(
    coeffs_fol_x: FloatVector,
    coeffs_fol_y: FloatVector,
    coeffs_ldr_x: FloatVector,
    coeffs_ldr_y: FloatVector,
    lag: float = 0.0,
) -> float:
    """Compute normalized lagged Euclidean distance between follower and leader curves."""


def signed_turn_lag(
    pts_a: FloatMatrix,
    pts_b: FloatMatrix,
    times_a: FloatVector,
    times_b: FloatVector,
    delta: int,
) -> dict[str, float | int]:
    """Compute mean point distance for a signed integer turn lag."""


def sweep_signed_turn_lags(
    pts_a: FloatMatrix,
    pts_b: FloatMatrix,
    times_a: FloatVector,
    times_b: FloatVector,
    max_lag: int = 15,
) -> dict[str, Any]:
    """Sweep signed integer turn lags and return the best lag profile."""


def dist_dist_correlation(X: FloatMatrix, Y: FloatMatrix) -> float:
    """Compute distance-distance Pearson correlation for two coordinate matrices."""
