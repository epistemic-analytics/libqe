"""
Rotation primitives: SVD, deflation, orthogonal SVD, means rotation, generalized-rotation tail.
"""

from collections.abc import Sequence
from typing import Annotated

import numpy
from numpy.typing import NDArray

import _pylibqe


def ena_svd(points: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')]) -> _pylibqe.RotationResult:
    """
    SVD rotation matching prcomp(retx=F, scale=F, center=F, tol=0).

    Caller is responsible for centering upstream. Eigenvalues are stored
    as sdev^2 to match rENA's ena.svd.

    Parameters
    ----------
    points : ndarray (n_units × n_dims)

    Returns RotationResult with column_names = ['SVD1', ..., 'SVDp'].

    Sign convention: none. Signs come from LAPACK's SVD, matching rENA's
    long-standing behavior. A deterministic sign rule may be added later.
    """

def deflate(data: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], axis: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
    """
    Project `data` onto the hyperplane orthogonal to a unit-norm axis:
      data - (data @ axis) @ axis.T

    Caller is responsible for normalizing `axis`.

    Returns a matrix of the same shape as `data`.
    """

def orthogonal_svd(data: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], weights: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], named_labels: Sequence[str]) -> _pylibqe.RotationResult:
    """
    Orthonormalize named axes via QR, fill the rest from SVD.

    Mirrors rENA's orthogonal_svd() in ena.rotate.by.mean.R. The named
    axes in the OUTPUT are the orthonormalized Q columns, not the
    original `weights` columns — use complete_rotation() to keep the
    named axes verbatim.

    Parameters
    ----------
    data         : ndarray (n_units × n_dims)
    weights      : ndarray (n_dims × k)   — columns are the named axes
    named_labels : list[str] of length k  — labels for the named axes

    Returns RotationResult with column_names = named_labels + ['SVD{k+1}'..'SVDp'].
    """

def complete_rotation(data: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], named_axes: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], named_labels: Sequence[str]) -> _pylibqe.RotationResult:
    """
    Keep named axes verbatim, fill remaining axes from an SVD of the
    data deflated by all named axes in parallel:
      defA = data - data @ named_axes @ named_axes.T

    Mirrors the tail of ena.rotate.by.generalized (canonical version:
    commit 2c079126 on rENA origin/main). The deflation is *parallel*
    (each projection comes off the original data), matching rENA's
    literal expression `defA <- A - A %*% v1 %*% t(v1) - A %*% v2 %*% t(v2)`.
    For mutually orthogonal axes this equals sequential deflation.
    On rank-deficient data (e.g. an all-zero connection column) the
    trailing axes that come from the deflated data's null space are
    orthogonalised against the named axes, so the rotation is orthonormal
    whenever the named axes are.

    Caller is responsible for ensuring each column of `named_axes` is
    unit-norm. Orthonormality between columns is NOT assumed.

    Conventional labels for generalized rotation are 'GMR1', 'GMR2',
    then 'SVD{k+1}'..'SVDp'.
    """

def means_rotation(points: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], group_pairs: list) -> _pylibqe.RotationResult:
    """
    Means rotation matching ena.rotate.by.mean.

    For each group pair, computes a normalized mean-difference axis on
    the progressively-deflated data and finishes with orthogonal_svd.
    The input is column-centered first, matching rENA's
      scale(data, scale=F, center=T)
    at the top of ena.rotate.by.mean.

    Parameters
    ----------
    points      : ndarray (n_units × n_dims)
    group_pairs : list of length k; each element is (a, b) where a and
                  b are 0-based integer index sequences into `points`.

    Returns RotationResult with column_names = ['MR1', ..., 'MRk',
    'SVD{k+1}', ..., 'SVDp'].

    MATCH-RENA NOTE: no guard against zero-norm mean-difference vectors
    (latent bug carried forward from rENA verbatim).
    """

def generalized_means_rotation(V: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], x_model: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], x_target: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu')], x1_cols: Sequence[int], x_categorical: bool, x_n_groups: int, x_subset: Sequence[int], has_y: bool, y_model: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], y_target: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu')], y1_cols: Sequence[int], y_categorical: bool, y_n_groups: int, n_lambda: int = 50, k_folds: int = 5, lasso_eps: float = 0.01) -> _pylibqe.RotationResult:
    """
    Generalized Means Rotation (GMR) with Lasso-based covariate adjustment.

    Mirrors rENA's ``ena.rotate.by.generalized()``. The x axis is the direction
    in ENA space most explained by ``x_target`` after controlling for covariates
    via Lasso (coordinate-descent, k-fold CV). The y axis is either a second GMR
    axis (``has_y=True``) or the leading SVD of the x-deflated space.

    All index lists (``x1_cols``, ``x_subset``, ``y1_cols``) are **0-based int**.
    Pass an empty list ``[]`` for ``x_subset`` to use all rows.
    Pass empty arrays/lists for all ``y_*`` arguments when ``has_y=False``.

    Parameters
    ----------
    V            : ndarray (n_units × n_dims)   ENA point matrix
    x_model      : ndarray (n_units × p)        model matrix for x axis
    x_target     : ndarray (n_units,)            target variable
    x1_cols      : list[int]  0-based target column indices in x_model
    x_categorical: bool
    x_n_groups   : int   number of groups (only used when x_categorical=True)
    x_subset     : list[int]  0-based row indices; [] = use all rows
    has_y        : bool  True → compute second GMR axis; False → SVD fallback
    y_model      : ndarray (n_units × p)  (ignored when has_y=False)
    y_target     : ndarray (n_units,)     (ignored when has_y=False)
    y1_cols      : list[int]              (ignored when has_y=False)
    y_categorical: bool                   (ignored when has_y=False)
    y_n_groups   : int                    (ignored when has_y=False)
    n_lambda     : int    lambda path length (default 50)
    k_folds      : int    CV folds for lambda selection (default 5)
    lasso_eps    : float  lambda_min = lasso_eps * lambda_max (default 0.01)

    Returns RotationResult with column_names = ['GMR1', 'GMR2'|'SVD2',
    'SVD3', ..., 'SVDp'].

    Reference: Zhiqiang Cai, commit 46776a1981a90b3a3b2861ed1010e9dbb7acf901.
    """
