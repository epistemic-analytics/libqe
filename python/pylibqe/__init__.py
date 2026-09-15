"""
pylibqe — Python bindings for libqe ENA math primitives.

Submodules
----------
pylibqe.adjacency
    Upper-triangle conversions, index helpers, string pair names.

pylibqe.normalization
    Row-wise L2 sphere normalization.

pylibqe.modeling
    Column centering, ENA correlation, least-squares node positions,
    and two-group comparison statistics.
    Also exports :class:`NodePositions` and :class:`GroupStatsResult`.

pylibqe.accumulation
    Stanza-window accumulation, rolling window sum,
    per-row co-occurrence, adjacency matrix construction.

pylibqe.rotation
    SVD rotation, deflation, orthogonal SVD, means rotation, and the
    generalized-rotation tail. Also exports :class:`RotationResult`.

pylibqe.door
    Door lookback pooling and EMA smoothing.

pylibqe.trajectory
    Parametric trajectory fitting, derivatives, distances, and lag analysis.

pylibqe.ccd
    Cross-covariance decay (CCD) window-size estimation.

All matrix inputs/outputs use numpy float64 arrays.
"""

from ._pylibqe import (
    adjacency,
    normalization,
    modeling,
    accumulation,
    rotation,
    door,
    trajectory,
    ccd,
)
from ._pylibqe.modeling import NodePositions, GroupStatsResult
from ._pylibqe import UnitNetworks, TensorNetworks, RotationResult

__all__ = [
    "adjacency",
    "normalization",
    "modeling",
    "accumulation",
    "rotation",
    "door",
    "trajectory",
    "ccd",
    "NodePositions",
    "GroupStatsResult",
    "UnitNetworks",
    "TensorNetworks",
    "RotationResult",
]
