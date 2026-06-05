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

All matrix inputs/outputs use numpy float64 arrays.
"""

from ._pylibqe import (
    adjacency,
    normalization,
    modeling,
    accumulation,
    rotation,
)
from ._pylibqe.modeling import NodePositions, GroupStatsResult
from ._pylibqe import UnitNetworks, TensorNetworks, RotationResult

__all__ = [
    "adjacency",
    "normalization",
    "modeling",
    "accumulation",
    "rotation",
    "NodePositions",
    "GroupStatsResult",
    "UnitNetworks",
    "TensorNetworks",
    "RotationResult",
]
