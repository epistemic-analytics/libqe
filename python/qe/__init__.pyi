"""
qe — Python bindings for libqe numerics.

Submodules
----------
qe.adjacency
    Upper-triangle conversions, index helpers, string pair names.

qe.normalization
    Row-wise L2 sphere normalization and max-norm scaling.

qe.modeling
    Column centering, confidence intervals and two-group comparison
    statistics.

qe.door
    Door lookback pooling and EMA smoothing.

qe.trajectory
    Parametric trajectory fitting, derivatives, distances, and lag analysis.

All matrix inputs/outputs use numpy float64 arrays.
"""

from . import adjacency as adjacency
from . import normalization as normalization
from . import modeling as modeling
from . import door as door
from . import trajectory as trajectory
from .modeling import GroupStatsResult as GroupStatsResult

__all__ = [
    "adjacency",
    "normalization",
    "modeling",
    "door",
    "trajectory",
    "GroupStatsResult",
]
