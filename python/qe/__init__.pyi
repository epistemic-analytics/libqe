"""
qe — Python bindings for libqe ENA math primitives.

Submodules
----------
qe.adjacency
    Upper-triangle conversions, index helpers, string pair names.

qe.normalization
    Row-wise L2 sphere normalization and max-norm scaling.

qe.modeling
    Column centering, ENA correlation, least-squares node positions,
    and two-group comparison statistics.

qe.accumulation
    Stanza-window accumulation, rolling window sum,
    per-row co-occurrence, adjacency matrix construction.

qe.rotation
    SVD rotation, deflation, orthogonal SVD, means rotation, and
    generalized means rotation.

qe.door
    Door lookback pooling and EMA smoothing.

qe.trajectory
    Parametric trajectory fitting, derivatives, distances, and lag analysis.

All matrix inputs/outputs use numpy float64 arrays.
"""

from . import adjacency as adjacency
from . import normalization as normalization
from . import modeling as modeling
from . import accumulation as accumulation
from . import rotation as rotation
from . import door as door
from . import trajectory as trajectory
from .modeling import NodePositions as NodePositions
from .modeling import GroupStatsResult as GroupStatsResult
from .accumulation import UnitNetworks as UnitNetworks
from .accumulation import TensorNetworks as TensorNetworks
from .rotation import RotationResult as RotationResult

__all__ = [
    "adjacency",
    "normalization",
    "modeling",
    "accumulation",
    "rotation",
    "door",
    "trajectory",
    "NodePositions",
    "GroupStatsResult",
    "UnitNetworks",
    "TensorNetworks",
    "RotationResult",
]
