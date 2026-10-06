"""
qe — Python bindings for libqe ENA math primitives.

Submodules
----------
qe.adjacency
    Upper-triangle conversions, index helpers, string pair names.

qe.normalization
    Row-wise L2 sphere normalization.

qe.modeling
    Column centering, ENA correlation, least-squares node positions,
    and two-group comparison statistics.
    Also exports :class:`NodePositions` and :class:`GroupStatsResult`.

qe.accumulation
    Stanza-window accumulation, rolling window sum,
    per-row co-occurrence, adjacency matrix construction.

qe.rotation
    SVD rotation, deflation, orthogonal SVD, means rotation, and the
    generalized-rotation tail. Also exports :class:`RotationResult`.

qe.door
    Door lookback pooling and EMA smoothing.

qe.trajectory
    Parametric trajectory fitting, derivatives, distances, and lag analysis.

qe.ccd
    Cross-covariance decay (CCD) window-size estimation.

All matrix inputs/outputs use numpy float64 arrays.
"""

from ._qe import (
    adjacency,
    normalization,
    modeling,
    accumulation,
    rotation,
    door,
    trajectory,
    ccd,
)
from ._qe.modeling import NodePositions, GroupStatsResult
from ._qe import UnitNetworks, TensorNetworks, RotationResult

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
