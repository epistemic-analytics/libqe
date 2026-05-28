# pylibqe — Python bindings for libqe

nanobind bindings that expose the five libqe modules as a `pylibqe` Python
package, built with CMake + scikit-build-core.

## API reference

All matrix arguments are 2-D `numpy.ndarray` with `dtype=float64`, C-contiguous.
All vector arguments are 1-D `numpy.ndarray` with `dtype=float64`.
Return values are always freshly allocated numpy arrays owned by Python.

### `pylibqe.adjacency`

| Function | Returns | Notes |
|----------|---------|-------|
| `choose_two(n)` | `int` | n × (n-1) / 2 |
| `connection_indices(len, row=-1)` | `2 × n_pairs ndarray[int64]` | Upper-tri (i, j) pairs. `row=-1` both rows, `0` row-only, `1` col-only |
| `code_connections(v)` | `1-D ndarray` | Pairwise products → connection vector |
| `fold_directed_network(v)` | `1-D ndarray` | n² directed vector → upper-tri |
| `network_to_vector(x, full=True)` | `1-D ndarray` | Matrix → flat vector. `full=True` → n², `False` → upper-tri |
| `connection_names(names)` | `list[str]` | `"A & B"` pair labels |

### `pylibqe.normalization`

| Function | Returns | Notes |
|----------|---------|-------|
| `normalize_networks(m)` |`accumulate_stanza(codes, window_back=1, window_forward=0, binary=True, ordered=False)` | `2-D ndarray` | Stanza-window accumulation. `ordered=False` (default): undirected upper-tri, returns `n_rows × choose_two(n_codes)`. `ordered=True`: directed, returns `n_rows × n_codes²`.| Row-wise L2 normalization. Zero rows left unchanged. |
| `scale_networks(m)` |`accumulate_stanza(codes, window_back=1, window_forward=0, binary=True, ordered=False)` | `2-D ndarray` | Stanza-window accumulation. `ordered=False` (default): undirected upper-tri, returns `n_rows × choose_two(n_codes)`. `ordered=True`: directed, returns `n_rows × n_codes²`.| Max-norm scaling: divide all rows by the largest row L2 norm |

### `pylibqe.modeling`

| Function | Returns | Notes |
|----------|---------|-------|
| `center_points(values)` |`accumulate_stanza(codes, window_back=1, window_forward=0, binary=True, ordered=False)` | `2-D ndarray` | Stanza-window accumulation. `ordered=False` (default): undirected upper-tri, returns `n_rows × choose_two(n_codes)`. `ordered=True`: directed, returns `n_rows × n_codes²`.| Subtract column means |
| `mean_ci(points, conf_level=0.95)` | `n_dims × 3 ndarray` | `[mean, ci_lower, ci_upper]` — matches rENA `t.test` exactly |
| `outlier_ci(points, iqr_factor=1.5)` | `n_dims × 2 ndarray` | `[lower, upper]` symmetric around 0 — matches rENA IQR formula |
| `ena_correlation(points, centroids, conf_level=0.95)` | `n_units × 3 ndarray` | Pearson r with CI: `[r, ci_lower, ci_upper]` |
| `node_positions(adj_mats, t, num_dims)` | `NodePositions` | Undirected ENA node positions |
| `directed_node_positions(line_weights, points, num_dims)` | `NodePositions` | Directed ENA node positions |
| `directed_node_positions_combine_pairs(line_weights, points, num_dims)` | `NodePositions` | Directed positions — paired ground+response rows combined before solve |

`NodePositions` fields: `.nodes`, `.centroids`, `.weights`, `.points` (all 2-D ndarray).

### `pylibqe.accumulation`

| Function | Returns | Notes |
|----------|---------|-------|
| `connection_matrix(ground, response, response_weight=1.0, ordered=True)` |`accumulate_stanza(codes, window_back=1, window_forward=0, binary=True, ordered=False)` | `2-D ndarray` | Stanza-window accumulation. `ordered=False` (default): undirected upper-tri, returns `n_rows × choose_two(n_codes)`. `ordered=True`: directed, returns `n_rows × n_codes²`.| Core adjacency math for one ground+response event pair |
| `accumulate_stanza(codes, window_back=1, window_forward=0, binary=True, ordered=False)` | `2-D ndarray` | Stanza-window accumulation. `ordered=False` (default): undirected upper-tri, returns `n_rows × choose_two(n_codes)`. `ordered=True`: directed, returns `n_rows × n_codes²`.| `2-D ndarray` | rENA stanza-window. `codes` = code matrix for one conversation. Returns `n_rows × choose_two(n_codes)` |
| `row_connections(codes, binary=True)` |`accumulate_stanza(codes, window_back=1, window_forward=0, binary=True, ordered=False)` | `2-D ndarray` | Stanza-window accumulation. `ordered=False` (default): undirected upper-tri, returns `n_rows × choose_two(n_codes)`. `ordered=True`: directed, returns `n_rows × n_codes²`.| Per-row upper-tri co-occurrence without windowing |
| `rolling_window_sum(codes, window_size=1)` |`accumulate_stanza(codes, window_back=1, window_forward=0, binary=True, ordered=False)` | `2-D ndarray` | Stanza-window accumulation. `ordered=False` (default): undirected upper-tri, returns `n_rows × choose_two(n_codes)`. `ordered=True`: directed, returns `n_rows × n_codes²`.| Rolling backward sum of raw code values |
| `flat_index(indices, dims)` | `int` | Column-major linear index. `indices` and `dims` are **0-based** |
| `accumulate_unit(codes, unit_rows, decay_fn, ordered=False)` | `UnitNetworks` | tma ground/response accumulation. `unit_rows` 0-based int list. `decay_fn(dists) -> weights` |
| `accumulate_unit_with_rows(codes, unit_rows, decay_fn, ordered=False)` | `UnitNetworks` | Like above + per-row networks |
| `apply_tensor_unit(tensor, dims, ...)` | `TensorNetworks` | Tensor-based multi-modal accumulation (tma) |

`UnitNetworks` fields: `.networks`, `.row_networks`.
`TensorNetworks` fields: `.connection_counts`, `.row_connection_counts`.

### `pylibqe.rotation`

| Function | Returns | Notes |
|----------|---------|-------|
| `ena_svd(points)` | `RotationResult` | SVD-based ENA rotation |
| `deflate(data, axis)` |`accumulate_stanza(codes, window_back=1, window_forward=0, binary=True, ordered=False)` | `2-D ndarray` | Stanza-window accumulation. `ordered=False` (default): undirected upper-tri, returns `n_rows × choose_two(n_codes)`. `ordered=True`: directed, returns `n_rows × n_codes²`.| Project out a single axis |
| `orthogonal_svd(points, fixed_axes)` | `RotationResult` | SVD constrained to be orthogonal to fixed axes |
| `complete_rotation(points, axes)` | `RotationResult` | Complete rotation given pre-specified axes |
| `means_rotation(points, groups)` | `RotationResult` | Means rotation (MR) — rotate toward group mean difference |

`RotationResult` fields: `.rotation`, `.eigenvalues`, `.column_names`.

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
from pylibqe import adjacency, normalization, modeling, accumulation

# Pair names for 3 codes
print(adjacency.connection_names(["X", "Y", "Z"]))
# ['X & Y', 'X & Z', 'Y & Z']

# Stanza-window accumulation
codes = np.array([[1,1,0],[1,0,1],[0,1,1]], dtype=float)
co = accumulation.accumulate_stanza(codes, window_back=2, binary=True)
print(co)

# Sphere normalization
normed = normalization.normalize_networks(codes)
```
