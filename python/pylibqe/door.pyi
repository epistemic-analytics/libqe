"""Door temporal pooling and smoothing helpers."""

from collections.abc import Sequence
from typing import Annotated

import numpy
from numpy.typing import NDArray


def lookback_block(
    block: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")],
    lookback_size: int = 20,
    aggregate_mean: bool = False,
    weighting_linear: bool = False,
    segment_ids: Sequence[int] = ...,
) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
    """Apply sliding lookback pooling over a single unit block."""


def ema_block(
    block: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")],
    alpha: float = 0.1,
    segment_ids: Sequence[int] = ...,
) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
    """Apply exponential moving average smoothing over a single unit block."""
