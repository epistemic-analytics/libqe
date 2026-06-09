"""
pylibqe — Python bindings for libqe ENA math primitives.

Submodules
----------
pylibqe.adjacency
    Upper-triangle conversions, index helpers, string pair names.

pylibqe.normalization
    Row-wise L2 sphere normalization and max-norm scaling.

pylibqe.modeling
    Column centering, ENA correlation, least-squares node positions,
    and two-group comparison statistics.

pylibqe.accumulation
    Stanza-window accumulation, rolling window sum,
    per-row co-occurrence, adjacency matrix construction.

pylibqe.rotation
    SVD rotation, deflation, orthogonal SVD, means rotation, and
    generalized means rotation.

All matrix inputs/outputs use numpy float64 arrays.
"""

from . import adjacency as adjacency
from . import normalization as normalization
from . import modeling as modeling
from . import accumulation as accumulation
from . import rotation as rotation
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
    "NodePositions",
    "GroupStatsResult",
    "UnitNetworks",
    "TensorNetworks",
    "RotationResult",
]
