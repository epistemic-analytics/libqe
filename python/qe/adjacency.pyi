"""Adjacency utilities: upper-triangle conversions, index helpers"""

from collections.abc import Sequence
from typing import Annotated

import numpy
from numpy.typing import NDArray


def choose_two(n: int) -> int:
    """Return n*(n-1)/2 (number of unique pairs in a set of n elements)."""

def n_connections(n: int, diagonal: bool = False) -> int:
    """Number of undirected connection columns: n*(n-1)/2, or n*(n+1)/2 with diagonal."""

def connection_indices(len: int, row: int = -1, diagonal: bool = False) -> Annotated[NDArray[numpy.int64], dict(shape=(None, None))]:
    """
    Upper-triangle index pairs for an len×len matrix.
    row=-1: 2×k array of [row_idx; col_idx]  row=0: row indices only  row=1: col indices only
    diagonal=True includes (i, i): (0,0), (0,1), (1,1), (0,2), ...
    """

def code_connections(v: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
    """
    Compute pairwise products v[j]*v[i] for all j < i (upper-triangle vector).
    """

def fold_directed_network(v: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu')], diagonal: bool = False) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
    """
    Fold a directed n*n flat vector to upper-triangle by summing A→B + B→A.
    diagonal=True keeps each self-connection (A→A) once: A & A, A & B, B & B, ...
    """

def network_to_vector(x: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], full: bool = True, diagonal: bool = False) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
    """
    Flatten an adjacency matrix to a vector.
    full=True: full n*n vector (directed)  full=False: upper-triangle only (undirected); diagonal=True includes A & A
    """

def connection_names(v: Sequence[str], diagonal: bool = False) -> list[str]:
    """
    Return 'A & B' pair names for every upper-triangle position.
    Example: ['X','Y','Z'] → ['X & Y', 'X & Z', 'Y & Z']; with diagonal=True
    ['X & X', 'X & Y', 'Y & Y', 'X & Z', 'Y & Z', 'Z & Z']
    """
