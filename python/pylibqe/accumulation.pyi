"""Network accumulation primitives"""

from collections.abc import Sequence
from typing import Annotated

import numpy
from numpy.typing import NDArray

import _pylibqe


def connection_matrix(ground: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu')], response: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu')], response_weight: float = 1.0, ordered: bool = True) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
    """
    Connection matrix for one ground+response event pair.
    ordered=True (directed):  ground→response cross-product
    ordered=False (undirected): symmetric outer-product
    """

def accumulate_stanza(codes: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], window_back: int = 1, window_forward: int = 0, binary: bool = True, ordered: bool = False) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
    """
    Stanza-window accumulation.
    ordered=False (default): upper-tri co-occurrences, returns (n_rows × choose_two(n_codes)).
    ordered=True: directed — focal row as response, prior window rows as ground,
    returns (n_rows × n_codes²). window_forward is ignored when ordered=True.
    """

def row_connections(codes: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], binary: bool = True) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
    """
    Per-row upper-triangle co-occurrence.
    Each row is processed independently (no cross-row accumulation).
    Returns (n_rows × choose_two(n_codes)) matrix.
    """

def rolling_window_sum(codes: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], window_size: int = 1) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
    """
    Rolling backward window sum of a raw code matrix.
    Row k = sum of rows [max(0, k - window_size + 1), k].
    Returns matrix of same shape as input.
    """

def flat_index(indices: Sequence[int], dims: Sequence[int]) -> int:
    """
    Linear index into a column-major multi-dimensional array (equivalent to sub2ind with Fortran/column-major ordering).
    """

def accumulate_unit(codes: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], unit_rows: Sequence[int], decay_fn: object, ordered: bool = False) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
    """
    Ground/response accumulation for one unit (tma model).

    Parameters
    ----------
    codes     : ndarray (n_context_rows x n_codes)
    unit_rows : list[int]  0-based row indices belonging to this unit
    decay_fn  : callable(distances: ndarray 1-D) -> ndarray 1-D
                Maps distance vector to weight vector.
    ordered   : bool  True = directed full matrix, False = undirected upper-tri

    Returns ndarray 1-D — flat connection vector.
    """

def accumulate_unit_with_rows(codes: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], unit_rows: Sequence[int], decay_fn: object, ordered: bool = False) -> _pylibqe.UnitNetworks:
    """
    Ground/response accumulation returning per-row connection data (tma model).

    decay_fn : callable(unit_row: int, ground_indices: ndarray int64 1-D) -> ndarray float64 1-D
    Returns UnitNetworks with .networks (1-D) and .row_networks (2-D).
    """

def apply_tensor_unit(tensor: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu')], dims: Sequence[int], dims_sender: Sequence[int], dims_receiver: Sequence[int], dims_mode: Sequence[int], context_lookup: Annotated[NDArray[numpy.int32], dict(shape=(None, None), order='C', device='cpu')], unit_rows: Sequence[int], codes: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], times: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu')], ordered: bool = True) -> _pylibqe.TensorNetworks:
    """
    Tensor-based multi-modal accumulation for one unit (tma model).

    tensor         : ndarray 1-D  flat column-major tensor of weights/windows
    dims           : list[int]    shape of the tensor
    dims_sender    : list[int]    tensor axis indices for sender factors
    dims_receiver  : list[int]    tensor axis indices for receiver factors
    dims_mode      : list[int]    tensor axis indices for mode factors
    context_lookup : ndarray int32 2-D  (n_rows x n_factors) factor indices
    unit_rows      : list[int]    0-based response-row indices for this unit
    codes          : ndarray 2-D  (n_rows x n_codes) code matrix
    times          : ndarray 1-D  timestamp per context row
    ordered        : bool         True = directed, False = undirected upper-tri

    Returns TensorNetworks with .connection_counts (1-D) and .row_connection_counts (2-D).

    Default mode: when dims=[2] and tensor has 2 elements [weight, window], uses
    a simplified single-weight/window path (equivalent to tma's default tensor).
    """
