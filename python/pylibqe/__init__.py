"""
pylibqe — Python bindings for libqe ENA math primitives.

Submodules
----------
pylibqe.adjacency
    Upper-triangle conversions, index helpers, string pair names.

pylibqe.normalization
    Row-wise L2 sphere normalization.

pylibqe.modeling
    Column centering, ENA correlation, least-squares node positions.
    Also exports :class:`NodePositions`.

pylibqe.accumulation
    Stanza-window accumulation, rolling window sum,
    per-row co-occurrence, adjacency matrix construction.

All matrix inputs/outputs use numpy float64 arrays.
"""

from ._pylibqe import adjacency, normalization, modeling, accumulation
from ._pylibqe.modeling import NodePositions

__all__ = [
    "adjacency",
    "normalization",
    "modeling",
    "accumulation",
    "NodePositions",
]
