"""ENA modeling: centering, correlation, node-position solvers"""

from typing import Annotated

import numpy
from numpy.typing import NDArray


class NodePositions:
    """
    Result struct returned by node-position solvers.

    Attributes
    ----------
    nodes     : ndarray (n_codes × n_dims)   — solved node coordinates
    centroids : ndarray (n_units × n_dims)   — unit centroid positions
    weights   : ndarray (n_units × n_codes)  — half-edge weight per node
    points    : ndarray (n_units × n_dims)   — input rotated points (echo)
    """

    @property
    def nodes(self) -> object: ...

    @property
    def centroids(self) -> object: ...

    @property
    def weights(self) -> object: ...

    @property
    def points(self) -> object: ...

    def __repr__(self) -> str: ...

def center_points(values: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
    """Subtract column means (center each column to zero)."""

def ena_correlation(points: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], centroids: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], conf_level: float = 0.95) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
    """
    Pearson correlation + CI between unit points and centroids.
    Returns (n_dims × 3) array: columns are [r, ci_lower, ci_upper].
    """

def mean_ci(points: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], conf_level: float = 0.95) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
    """
    t-based confidence interval for the mean of a group of ENA unit points.

    For each dimension computes: mean ± t(α/2, n-1) × (SD / sqrt(n))
    where α = 1 - conf_level.

    Parameters
    ----------
    points     : ndarray (n_units × n_dims)  — one row per unit in the group
    conf_level : float  confidence level, e.g. 0.95 (default)

    Returns (n_dims × 3) array: columns are [mean, ci_lower, ci_upper].
    When n_units == 1 the CI bounds are ±inf.
    """

def outlier_ci(points: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], iqr_factor: float = 1.5) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
    """
    Outlier interval based on IQR (Tukey fence) for a group of ENA unit points.

    For each dimension d:
      lower[d] = -iqr_factor * IQR(points[:, d])
      upper[d] = +iqr_factor * IQR(points[:, d])

    Symmetric around 0, matching rENA's formula:
      oi = IQR(each dim) * iqr_factor
      result = matrix([[−oi], [+oi]])

    IQR uses type-7 / Hyndman-Fan #7 quantile (R default,
    identical to numpy's percentile(method='linear')).

    Parameters
    ----------
    points     : ndarray (n_units × n_dims)  — one row per unit
    iqr_factor : float  multiplier applied to IQR (default 1.5)

    Returns (n_dims × 2) array: columns are [lower, upper].
    All entries are NaN when n_units == 0.
    """

class GroupStatsResult:
    """
    Two-group comparison statistics returned by group_stats().

    Attributes
    ----------
    n1, n2       : int  — sample sizes
    Parametric (Welch t-test):
      t          : ndarray (n_dims,)  — t-statistics
      df         : ndarray (n_dims,)  — Welch–Satterthwaite degrees of freedom
      pvalue_t   : ndarray (n_dims,)  — two-tailed p-values
      cohens_d   : ndarray (n_dims,)  — Cohen's d (pooled SD)
      means      : ndarray (2 × n_dims)  — row 0 = group1, row 1 = group2
      sds        : ndarray (2 × n_dims)  — sample standard deviations
    Non-parametric (Wilcoxon rank-sum):
      U          : ndarray (n_dims,)  — U for group 1 (= R's W statistic)
      pvalue_u   : ndarray (n_dims,)  — two-tailed p-values (normal approx)
      effect_r   : ndarray (n_dims,)  — rank-biserial: 1 − 2·U / (n1·n2)
      medians    : ndarray (2 × n_dims)  — row 0 = group1, row 1 = group2
    """

    @property
    def n1(self) -> int: ...

    @property
    def n2(self) -> int: ...

    @property
    def t(self) -> object: ...

    @property
    def df(self) -> object: ...

    @property
    def pvalue_t(self) -> object: ...

    @property
    def cohens_d(self) -> object: ...

    @property
    def means(self) -> object: ...

    @property
    def sds(self) -> object: ...

    @property
    def U(self) -> object: ...

    @property
    def pvalue_u(self) -> object: ...

    @property
    def effect_r(self) -> object: ...

    @property
    def medians(self) -> object: ...

    def __repr__(self) -> str: ...

def group_stats(g1: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], g2: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')]) -> GroupStatsResult:
    """
    Per-dimension two-group comparison statistics.

    Computes Welch t-test (t, df, p-value, Cohen's d, means, SDs) and
    Wilcoxon rank-sum test (U, p-value, rank-biserial effect, medians)
    for each dimension independently.

    Parameters
    ----------
    g1 : ndarray (n1 × n_dims)  — unit points for group 1
    g2 : ndarray (n2 × n_dims)  — unit points for group 2

    Returns GroupStatsResult.

    Notes
    -----
    Parametric entries are NaN when n < 2 for either group.
    Wilcoxon p-values use the normal approximation (tie + continuity
    correction), matching R's wilcox.test(..., exact=FALSE, correct=TRUE).
    Equivalent to rENA-api's group.stats().
    """

def node_positions(adj_mats: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], t: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], num_dims: int) -> NodePositions:
    """Multiobjective least-squares node positions for undirected ENA."""

def directed_node_positions(line_weights: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], points: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], num_dims: int) -> NodePositions:
    """Least-squares node positions for directed (ordered) ENA."""

def directed_node_positions_combine_pairs(line_weights: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], points: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')], num_dims: int) -> NodePositions:
    """
    Directed node positions with paired ground+response rows combined before solving.
    """
