# LibQE.jl — Julia bindings for libqe

Julia bindings for [libqe](https://gitlab.com/epistemic-analytics/qe-packages/libqe),
the shared C++ core for Quantitative Ethnography packages.

Built with [CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl).
All functions accept standard Julia `Matrix{Float64}` / `Vector{Float64}` arguments.
Matrix inputs are **zero-copy** across the C++ boundary (Julia and Armadillo are both
column-major).

## Requirements

- Julia ≥ 1.9
- CxxWrap 0.15
- CMake ≥ 3.18
- Armadillo (system install)

```bash
# macOS — Armadillo via Homebrew (Accelerate provides BLAS automatically)
brew install armadillo

# Linux — system Armadillo + BLAS/LAPACK
apt-get install libarmadillo-dev libblas-dev liblapack-dev
```

## Build and install

```julia
# one-time: install CxxWrap so CMake can find libcxxwrap-julia
using Pkg; Pkg.add("CxxWrap")
```

```bash
cd julia
cmake -B build -DCMAKE_BUILD_TYPE=Release .
cmake --build build
cmake --install build      # copies libqe_julia.{so,dylib} into LibQE/lib/
```

```julia
using Pkg; Pkg.develop(path="julia/LibQE")
using LibQE
```

## Quick start

```julia
using LibQE

# Code-pair labels
svector_to_upper_tri(["A", "B", "C"])
# → ["A & B", "A & C", "B & C"]

# Stanza-window accumulation
codes = Float64[1 1 0; 1 0 1; 0 1 1]   # 3 rows × 3 codes
stanza_window(codes; window_back=2, binary=true)
# → 3×3 Matrix{Float64}  (choose_two(3) = 3 connection columns)

# Group confidence interval (matches rENA's t.test per dimension)
pts = randn(15, 2)
group_ci(pts)
# → 2×3 Matrix  [mean  ci_lower  ci_upper] per row = dimension
```

## API reference

### Adjacency

| Function | Returns | Notes |
|----------|---------|-------|
| `svector_to_upper_tri(names)` | `Vector{String}` | `"A & B"` pair labels for every upper-tri position |
| `tri_indices(len; row=-1)` | `2 × n_pairs Matrix{Int32}` | Upper-tri (i, j) index pairs. `row=-1` both rows, `0` row-indices only, `1` col-indices only |
| `vector_to_upper_tri(v)` | `Vector{Float64}` | Pairwise products of a code vector → connection vector |
| `directed_to_upper_tri(v)` | `Vector{Float64}` | n² directed vector → upper-triangle (symmetric pairs summed) |
| `adjacency_matrix_to_vector(m; full=true)` | `Vector{Float64}` | Flatten adjacency matrix. `full=true` → n² directed; `false` → upper-tri |

### Normalization

| Function | Returns | Notes |
|----------|---------|-------|
| `sphere_norm(m)` | `Matrix{Float64}` | Row-wise L2 normalization. Zero rows left unchanged. |
| `skip_sphere_norm(m)` | `Matrix{Float64}` | Max-norm scaling: divide all rows by the largest row L2 norm |

### Modeling

| Function | Returns | Notes |
|----------|---------|-------|
| `center_data(m)` | `Matrix{Float64}` | Subtract column means |
| `group_ci(pts; conf_level=0.95)` | `n_dims × 3 Matrix` | `[mean, ci_lower, ci_upper]` — matches rENA `t.test` exactly |
| `outlier_ci(pts; iqr_factor=1.5)` | `n_dims × 2 Matrix` | `[lower, upper]` symmetric around 0 — matches rENA IQR formula |
| `ena_correlation(pts, centroids; conf_level=0.95)` | `n_units × 3 Matrix` | Pearson r with CI: `[r, ci_lower, ci_upper]` |
| `lws_lsq_positions(adj, t, dims)` | `NamedTuple` | Undirected ENA node positions → `(nodes, centroids, weights, points)` |
| `directed_node_positions(lw, pts, dims)` | `NamedTuple` | Directed ENA node positions → same structure |

### Accumulation

| Function | Returns | Notes |
|----------|---------|-------|
| `calculate_adjacency_matrix(ground, response; response_weight=1.0, ordered=true)` | `Matrix{Float64}` | Core adjacency math for one ground+response event pair |
| `stanza_window(codes; window_back=1, window_forward=0, binary=true)` | `Matrix{Float64}` | rENA stanza-window. `codes` = code matrix for one conversation. Returns `n_rows × choose_two(n_codes)` |
| `rows_to_co_occurrences(codes; binary=true)` | `Matrix{Float64}` | Per-row upper-tri co-occurrence without windowing |
| `rolling_window_sum(codes; window_size=1)` | `Matrix{Float64}` | Rolling backward sum of raw code values |
| `calculate_1d_index(indices, dims)` | `Int` | Column-major linear index. `indices` and `dims` are **0-based** |
| `accumulate_unit(codes, unit_rows, decay_fn; ordered=false)` | `Vector{Float64}` | tma ground/response accumulation. `unit_rows` are **0-based** `Vector{Int32}` |
| `accumulate_unit_with_rows(codes, unit_rows, decay_fn; ordered=false)` | `NamedTuple` | Like `accumulate_unit` + per-response-row networks → `(networks, row_networks)` |

## Running tests

```bash
cd julia/LibQE
julia --project=. -e 'using Pkg; Pkg.test()'
```
