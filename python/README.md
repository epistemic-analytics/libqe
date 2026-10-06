# qe-lib — Python bindings for libqe

nanobind bindings that expose the libqe modules as the `qe` Python package
(install `qe-lib`, `import qe`),
built with CMake + scikit-build-core.

## API reference

All matrix arguments are 2-D `numpy.ndarray` with `dtype=float64`, C-contiguous.
All vector arguments are 1-D `numpy.ndarray` with `dtype=float64`.
Return values are always freshly allocated numpy arrays owned by Python.

### `qe.adjacency`

| Function | Returns | Notes |
|----------|---------|-------|
| `choose_two(n)` | `int` | n × (n-1) / 2 |
| `connection_indices(len, row=-1)` | `2 × n_pairs ndarray[int64]` | Upper-tri (i, j) pairs. `row=-1` both rows, `0` row-only, `1` col-only |
| `code_connections(v)` | `1-D ndarray` | Pairwise products of a code vector → connection vector |
| `fold_directed_network(v)` | `1-D ndarray` | n² directed vector → upper-tri (symmetric pairs summed) |
| `network_to_vector(x, full=True)` | `1-D ndarray` | Matrix → flat vector. `full=True` → n², `False` → upper-tri |
| `connection_names(names)` | `list[str]` | `"A & B"` pair labels for every upper-tri position |

### `qe.normalization`

| Function | Returns | Notes |
|----------|---------|-------|
| `normalize_networks(m)` | `2-D ndarray` | Row-wise L2 normalization. Zero rows left unchanged. |
| `scale_networks(m)` | `2-D ndarray` | Max-norm scaling: divide all rows by the largest row L2 norm. |

### `qe.modeling`

| Function | Returns | Notes |
|----------|---------|-------|
| `center_points(values)` | `2-D ndarray` | Subtract column means. |
| `mean_ci(points, conf_level=0.95)` | `n_dims × 3 ndarray` | `[mean, ci_lower, ci_upper]` — matches rENA `t.test` exactly. |
| `outlier_ci(points, iqr_factor=1.5)` | `n_dims × 2 ndarray` | `[lower, upper]` symmetric around 0 — matches rENA IQR formula. |
| `ena_correlation(points, centroids, conf_level=0.95)` | `n_units × 3 ndarray` | Pearson r with CI: `[r, ci_lower, ci_upper]`. |
| `node_positions(adj_mats, t, num_dims)` | `NodePositions` | Undirected ENA node positions. |
| `directed_node_positions(line_weights, points, num_dims)` | `NodePositions` | Directed ENA node positions. |
| `directed_node_positions_combine_pairs(line_weights, points, num_dims)` | `NodePositions` | Directed positions — paired ground+response rows combined before solve. |
| `group_stats(g1, g2)` | `GroupStatsResult` | Per-dimension Welch t-test + Wilcoxon rank-sum. See fields below. |

`NodePositions` fields: `.nodes`, `.centroids`, `.weights`, `.points` (all 2-D ndarray).

#### `GroupStatsResult` fields

| Field | Type | Notes |
|-------|------|-------|
| `n1`, `n2` | `int` | Sample sizes |
| `t` | `ndarray (n_dims,)` | Welch t-statistic per dimension |
| `df` | `ndarray (n_dims,)` | Welch–Satterthwaite degrees of freedom |
| `pvalue_t` | `ndarray (n_dims,)` | Two-tailed p-values (t-test) |
| `cohens_d` | `ndarray (n_dims,)` | Cohen's d (pooled SD, group1 − group2) |
| `means` | `2 × n_dims ndarray` | Row 0 = group1, row 1 = group2 |
| `sds` | `2 × n_dims ndarray` | Sample standard deviations |
| `U` | `ndarray (n_dims,)` | Wilcoxon U for group 1 (= R's W) |
| `pvalue_u` | `ndarray (n_dims,)` | Two-tailed p-values (normal approx, tie + continuity correction) |
| `effect_r` | `ndarray (n_dims,)` | Rank-biserial: 1 − 2·U / (n1·n2) |
| `medians` | `2 × n_dims ndarray` | Row 0 = group1, row 1 = group2 |

### `qe.accumulation`

| Function | Returns | Notes |
|----------|---------|-------|
| `connection_matrix(ground, response, response_weight=1.0, ordered=True)` | `2-D ndarray` | Core adjacency math for one ground+response event pair. |
| `accumulate_stanza(codes, window_back=1, window_forward=0, binary=True, ordered=False)` | `2-D ndarray` | Stanza-window accumulation. `ordered=False`: undirected upper-tri → `n_rows × choose_two(n_codes)`. `ordered=True`: directed → `n_rows × n_codes²`. |
| `row_connections(codes, binary=True)` | `2-D ndarray` | Per-row upper-tri co-occurrence without windowing. |
| `rolling_window_sum(codes, window_size=1)` | `2-D ndarray` | Rolling backward sum of raw code values. |
| `flat_index(indices, dims)` | `int` | Column-major linear index. `indices` and `dims` are **0-based**. |
| `accumulate_unit(codes, unit_rows, decay_fn, ordered=False)` | `UnitNetworks` | tma ground/response accumulation. `unit_rows` 0-based int list. `decay_fn(dists) -> weights`. |
| `accumulate_unit_with_rows(codes, unit_rows, decay_fn, ordered=False)` | `UnitNetworks` | Like `accumulate_unit` + per-response-row networks. |
| `apply_tensor_unit(tensor, dims, ...)` | `TensorNetworks` | Tensor-based multi-modal accumulation (tma). |

`UnitNetworks` fields: `.networks`, `.row_networks`.
`TensorNetworks` fields: `.connection_counts`, `.row_connection_counts`.

### `qe.rotation`

| Function | Returns | Notes |
|----------|---------|-------|
| `ena_svd(points)` | `RotationResult` | SVD-based ENA rotation. |
| `deflate(data, axis)` | `2-D ndarray` | Project out a single unit axis from `data`. |
| `orthogonal_svd(points, fixed_axes)` | `RotationResult` | SVD constrained to be orthogonal to fixed axes. |
| `complete_rotation(points, axes)` | `RotationResult` | Fix named axes first, fill remaining dims with SVD. |
| `means_rotation(points, groups)` | `RotationResult` | Means rotation (MR) — rotate toward group mean difference. |
| `generalized_means_rotation(V, x_model, x_target, ...)` | `RotationResult` | GMR with Lasso covariate adjustment — mirrors rENA's `ena.rotate.by.generalized()`. |

`RotationResult` fields: `.rotation`, `.eigenvalues`, `.column_names`.

### `qe.door`

| Function | Returns | Notes |
|----------|---------|-------|
| `lookback_block(block, lookback_size=20, aggregate_mean=False, weighting_linear=False, segment_ids=[])` | `2-D ndarray` | Lookback pooling over one unit block; matches ETM missing-value behavior. |
| `ema_block(block, alpha=0.1, segment_ids=[])` | `2-D ndarray` | EMA smoothing over one unit block; missing current values carry the prior smoothed value. |

### `qe.trajectory`

| Function | Returns | Notes |
|----------|---------|-------|
| `fit_poly(points, t=None, max_degree=3, fixed_degree=0, criterion="loocv", basis="orthogonal")` | `dict` | Fit a 2D polynomial trajectory; default basis matches R `stats::poly` predictions. |
| `eval_curve(coeffs_x, coeffs_y, t_eval)` | `2-D ndarray` | Evaluate fitted trajectory coordinates. |
| `eval_derivatives(coeffs_x, coeffs_y, t_eval)` | `dict` | Velocity, acceleration, speed, heading rate, and curvature. |
| `integrated_distance(...)` | `float` | Integrated Euclidean distance between curves. |
| `lagged_distance(...)` | `float` | Normalized follower/leader lagged curve distance. |
| `signed_turn_lag(...)` / `sweep_signed_turn_lags(...)` | `dict` | Discrete signed turn-lag distance metrics. |
| `dist_dist_correlation(X, Y)` | `float` | Correlation between pairwise distance structures. |

## Build dependencies

- Python ≥ 3.9
- numpy
- armadillo (system or conda-forge / homebrew)
- nanobind ≥ 1.9 (`pip install nanobind` or `brew install nanobind`)
- scikit-build-core ≥ 0.4 (`pip install scikit-build-core`)
- cmake ≥ 3.18

## Install (development)

```bash
cd libqe/python
pip install -e ".[dev]"
```

Or build a wheel:

```bash
pip wheel .
```

## Run tests

```bash
cd libqe/python
pytest tests/
```

## Usage example

```python
import numpy as np
from qe import adjacency, normalization, modeling, accumulation

# Pair names for 3 codes
print(adjacency.connection_names(["X", "Y", "Z"]))
# ['X & Y', 'X & Z', 'Y & Z']

# Stanza-window accumulation
codes = np.array([[1,1,0],[1,0,1],[0,1,1]], dtype=float)
co = accumulation.accumulate_stanza(codes, window_back=2, binary=True)
print(co)

# Sphere normalization
normed = normalization.normalize_networks(codes)

# Two-group comparison
g1 = np.random.randn(20, 2)
g2 = np.random.randn(18, 2) + 0.5
r = modeling.group_stats(g1, g2)
print(r.pvalue_t)   # per-dimension two-tailed p-values
```
