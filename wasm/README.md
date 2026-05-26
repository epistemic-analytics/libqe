# @qe-libs/libqe-wasm

WebAssembly bindings for [libqe](https://gitlab.com/epistemic-analytics/qe-packages/libqe) —
the shared C++ core for Quantitative Ethnography packages.

Exposes the full libqe API as a zero-dependency ES module usable in browsers and Node.js.

## Install

```bash
# one-time: tell npm where to find the @qe-libs scope
echo "@qe-libs:registry=https://gitlab.com/api/v4/projects/epistemic-analytics%2Fqe-packages%2Flibqe/packages/npm/" >> ~/.npmrc

npm install @qe-libs/libqe-wasm
```

## Usage

```js
import loadLibQE from '@qe-libs/libqe-wasm';

const qe = await loadLibQE();

// Code-pair labels
qe.svector_to_upper_tri(['Concept A', 'Concept B', 'Concept C']);
// → ['Concept A & Concept B', 'Concept A & Concept C', 'Concept B & Concept C']

// Stanza-window accumulation
const codes = new Float64Array([1,1,0,  1,0,1,  0,1,1]);  // 3×3 row-major
const out   = qe.stanza_window(codes, 3, 3, /*back=*/2, /*forward=*/0, /*binary=*/true);
// out → { data: Float64Array, rows: 3, cols: 3 }

// Group confidence interval
const points = new Float64Array([0.1,0.2, 0.3,0.4, 0.5,0.6]);  // 3 units × 2 dims
const ci = qe.group_ci(points, 3, 2, 0.95);
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
| `svector_to_upper_tri(names)` | `string[]` → pair labels `["A & B", ...]` |
| `tri_indices(len)` | Upper-triangle index pairs `{ rows, cols }` |
| `vector_to_upper_tri(data, n_codes)` | Code vector → connection vector |
| `directed_to_upper_tri(data)` | n² directed vector → upper-tri |
| `adjacency_matrix_to_vector(data, rows, cols, full)` | Matrix → flat vector |

### Normalization

| Function | Description |
|----------|-------------|
| `sphere_norm(data, rows, cols)` | Row-wise L2 normalization |
| `skip_sphere_norm(data, rows, cols)` | Max-norm scaling |

### Modeling

| Function | Description |
|----------|-------------|
| `center_data(data, rows, cols)` | Subtract column means |
| `group_ci(data, rows, cols, conf_level)` | t-based CI → `n_dims × 3` `[mean, lower, upper]` |
| `outlier_ci(data, rows, cols, iqr_factor)` | IQR-based interval → `n_dims × 2` `[lower, upper]` |
| `lws_lsq_positions(adj, ar, ac, t, tr, tc, dims)` | Undirected ENA node positions |
| `directed_node_positions(lw, lr, lc, pt, pr, pc, dims)` | Directed ENA node positions |

### Accumulation

| Function | Description |
|----------|-------------|
| `stanza_window(data, rows, cols, back, forward, binary)` | rENA stanza-window |
| `rows_to_co_occurrences(data, rows, cols, binary)` | Per-row co-occurrence |
| `rolling_window_sum(data, rows, cols, window_size)` | Rolling backward sum |
| `calculate_1d_index(indices, dims)` | Column-major linear index |

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
