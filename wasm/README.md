# @qe-libs/libqe-wasm

WebAssembly bindings for [libqe](https://gitlab.com/epistemic-analytics/qe-packages/libqe) —
the shared C++ core for Quantitative Ethnography packages.

Exposes the full libqe API as a zero-dependency ES module usable in browsers and Node.js.

## Install

```bash
# one-time: tell npm where to find the @qe-libs scope
echo "@qe-libs:registry=https://gitlab.com/api/v4/projects/22522458/packages/npm/" >> ~/.npmrc

npm install @qe-libs/libqe-wasm
```

## Usage

```js
import loadLibQE from '@qe-libs/libqe-wasm';

const qe = await loadLibQE();

// Code-pair labels
qe.connection_names(['Concept A', 'Concept B', 'Concept C']);
// → ['Concept A & Concept B', 'Concept A & Concept C', 'Concept B & Concept C']

// Stanza-window accumulation
const codes = new Float64Array([1,1,0,  1,0,1,  0,1,1]);  // 3×3 row-major
const out   = qe.accumulate_stanza(codes, 3, 3, /*back=*/2, /*forward=*/0, /*binary=*/true, /*ordered=*/false);
// out → { data: Float64Array, rows: 3, cols: 3 }

// Group confidence interval
const points = new Float64Array([0.1,0.2, 0.3,0.4, 0.5,0.6]);  // 3 units × 2 dims
const ci = qe.mean_ci(points, 3, 2, 0.95);
// ci → { data: Float64Array, rows: 2, cols: 3 }  — [mean, lower, upper] per dim

// Outlier interval (IQR × 1.5)
const oi = qe.outlier_ci(points, 3, 2, 1.5);
// oi → { data: Float64Array, rows: 2, cols: 2 }  — [lower, upper] per dim
```

## Matrix convention

All matrix inputs are **flat `Float64Array` in row-major order** with explicit
`rows` and `cols` arguments. Return values are plain objects:

```js
{ data: Float64Array, rows: number, cols: number }
```

Access a cell: `data[row * cols + col]`

## API reference

### Adjacency

| Function | Description |
|----------|-------------|
| `connection_names(names)` | `string[]` → pair labels `["A & B", ...]` |
| `connection_indices(len)` | Upper-triangle index pairs `{ rows, cols }` |
| `code_connections(data, n_codes)` | Code vector → connection vector |
| `fold_directed_network(data)` | n² directed vector → upper-tri |
| `network_to_vector(data, rows, cols, full)` | Matrix → flat vector |

### Normalization

| Function | Description |
|----------|-------------|
| `normalize_networks(data, rows, cols)` | Row-wise L2 normalization |
| `scale_networks(data, rows, cols)` | Max-norm scaling |

### Modeling

| Function | Description |
|----------|-------------|
| `center_points(data, rows, cols)` | Subtract column means |
| `mean_ci(data, rows, cols, conf_level)` | t-based CI → `n_dims × 3` `[mean, lower, upper]` |
| `outlier_ci(data, rows, cols, iqr_factor)` | IQR-based interval → `n_dims × 2` `[lower, upper]` |
| `ena_correlation(pts, pr, pc, cen, cr, cc, conf_level)` | Pearson r + CI → `n_units × 3` `[r, lower, upper]` |
| `node_positions(adj, ar, ac, t, tr, tc, dims)` | Undirected ENA node positions. `dims` may be omitted or `0` to use all columns of `t` |
| `directed_node_positions(lw, lr, lc, pt, pr, pc, dims)` | Directed ENA node positions. `dims` may be omitted or `0` to use all columns of `pt` |
| `directed_node_positions_combine_pairs(lw, lr, lc, pt, pr, pc, dims)` | Directed ENA — ground+response rows averaged before solve. `dims` may be omitted or `0` |

### Accumulation

| Function | Description |
|----------|-------------|
| `connection_matrix(ground, gn, response, rn, weight, ordered)` | Core adjacency math for one ground+response pair → `n_codes × n_codes` matrix |
| `accumulate_stanza(data, rows, cols, back, forward, binary, ordered)` | Stanza-window. `ordered=false`: upper-tri `choose(n,2)` cols. `ordered=true`: directed `n²` cols |
| `row_connections(data, rows, cols, binary)` | Per-row co-occurrence |
| `rolling_window_sum(data, rows, cols, window_size)` | Rolling backward sum |
| `flat_index(indices, dims)` | Column-major linear index |
| `accumulate_unit(codes, rows, cols, unit_rows, decay_fn, ordered)` | Ground/response accumulation for one unit (tma). `decay_fn(Float64Array) → Float64Array` |
| `accumulate_unit_with_rows(codes, rows, cols, unit_rows, decay_fn, ordered)` | Like `accumulate_unit` but also returns per-response-row networks → `{ networks, row_networks }` |
| `accumulate_tensor_unit(tensor, dims, dims_sender, dims_receiver, dims_mode, context_lookup, cl_rows, cl_cols, unit_rows, codes, rows, cols, times, ordered)` | tma tensor accumulation for one unit → `{ connection_counts: Float64Array, row_connection_counts: matObj }`. All index arrays are **0-based Int32Array**; `tensor` is flat column-major `Float64Array`; `context_lookup` is row-major `Int32Array (n_context_rows × n_factors)` |

### Door

| Function | Description |
|----------|-------------|
| `door_lookback_block(data, rows, cols, lookbackSize, aggregateMean, weightingLinear, segmentIds)` | Lookback pooling over one unit block; JS wrapper supplies default `segmentIds = []`. |
| `door_ema_block(data, rows, cols, alpha, segmentIds)` | EMA smoothing over one unit block; JS wrapper supplies default `segmentIds = []`. |

### Trajectory

| Function | Description |
|----------|-------------|
| `fit_trajectory_poly(data, rows, cols, t, maxDegree, fixedDegree, criterion, basis)` | Fit a 2D polynomial trajectory; JS wrapper defaults to `criterion = "loocv"` and `basis = "orthogonal"`. |
| `eval_trajectory_curve(coeffsX, coeffsY, tEval)` | Evaluate fitted trajectory coordinates. |
| `eval_trajectory_derivatives(coeffsX, coeffsY, tEval)` | Evaluate velocity, acceleration, speed, heading rate, and curvature. |
| `integrated_trajectory_distance(...)` | Integrated Euclidean distance between two curves. |
| `lagged_trajectory_distance(...)` | Normalized lagged follower/leader curve distance. |
| `signed_turn_lag(...)` / `sweep_signed_turn_lags(...)` | Discrete signed turn-lag distance metrics. |
| `dist_dist_correlation(X, xr, xc, Y, yr, yc)` | Correlation between pairwise distance structures. |

### Rotation

| Function | Description |
|----------|-------------|
| `ena_svd(data, rows, cols)` | SVD rotation → `{ rotation, eigenvalues, column_names }` |
| `deflate(data, rows, cols, axis)` | Project out a given axis → `{ data, rows, cols }` |
| `orthogonal_svd(data, rows, cols, weights, wr, wc, labels)` | Weighted SVD with orthogonalization |
| `complete_rotation(data, rows, cols, axes, ax_rows, ax_cols, labels)` | Fix named axes then fill remaining with SVD |
| `means_rotation(data, rows, cols, group_pairs)` | Group-means rotation. `group_pairs`: `Array<{a: Int32Array, b: Int32Array}>` (0-based row indices) |
| `generalized_means_rotation(V, vr, vc, xm, xr, xc, x_target, x1_cols, x_categorical, x_n_groups, x_subset, has_y, ym, yr, yc, y_target, y1_cols, y_categorical, y_n_groups, n_lambda, k_folds, lasso_eps)` | Lasso-based GMR. Index arrays are `Int32Array` (0-based); pass zero-length `Int32Array` for `x_subset` to use all rows. Returns `{ rotation, eigenvalues, column_names }` with labels `GMR1`, `GMR2`\|`SVD2`, … |

## Building from source

### Prerequisites

- [Emscripten](https://emscripten.org/docs/getting_started/downloads.html) (`emcc` on PATH)
- [Conan 2.x](https://docs.conan.io/2/) (`pip install conan`)
- CMake ≥ 3.18, Ninja

### Steps

```bash
# From the libqe repo root:
sh wasm/scripts/build.sh

# Output: wasm/dist/libqe.js  wasm/dist/libqe.wasm
```

The build uses `../include/libqe/` headers directly — no Conan registry fetch needed.
Conan is only used to install Armadillo for the Emscripten cross-compilation.

## Notes on Armadillo and BLAS

libqe uses [Armadillo](https://arma.sourceforge.net/) for linear algebra.
The WASM build compiles with `ARMA_DONT_USE_BLAS` and `ARMA_DONT_USE_LAPACK`
so Armadillo uses its own built-in LU/QR routines instead of calling external
symbols that Emscripten cannot resolve. For ENA's typical data sizes this has
no meaningful performance impact.

## Running tests

```bash
cd wasm
npm ci
npm test   # requires dist/ — run build first
```
