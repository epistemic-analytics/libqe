"""
qe — Python bindings for libqe numerics.

Submodules
----------
qe.adjacency
    Upper-triangle conversions, index helpers, string pair names.

qe.normalization
    Row-wise L2 sphere normalization.

qe.modeling
    Column centering, confidence intervals and two-group comparison
    statistics. Also exports :class:`GroupStatsResult`.

qe.door
    Door lookback pooling and EMA smoothing.

qe.trajectory
    Parametric trajectory fitting, derivatives, distances, and lag analysis.

The ENA model code (rotations, node positions, CCD) is in qe-ena
(``import ena``) and all accumulation in qe-tma (``import tma``).

All matrix inputs/outputs use numpy float64 arrays.
"""

from ._qe import (
    adjacency,
    normalization,
    modeling,
    door,
    trajectory,
)
from ._qe.modeling import GroupStatsResult

__all__ = [
    "adjacency",
    "normalization",
    "modeling",
    "door",
    "trajectory",
    "GroupStatsResult",
]
