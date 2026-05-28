# LibQE.jl — Julia bindings for libqe

Julia bindings for [libqe](https://gitlab.com/epistemic-analytics/qe-packages/libqe),
the shared C++ core for Quantitative Ethnography packages.

Built with [CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl).
All functions accept standard Julia`accumulate_stanza(codes; window_back=1, window_forward=0, binary=true, ordered=false)` | `Matrix{Float64}` | Stanza-window accumulation. `ordered=false` (default): undirected upper-tri, returns `n_rows × choose_two(n_codes)`. `ordered=true`: directed, returns `n_rows × n_codes²`./ `Vector{Float64}` arguments.
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
connection_names(["A", "B", "C"])
# → ["A & B", "A & C", "B & C"]

# Stanza-window accumulation
codes = Float64[1 1 0; 1 0 1; 0 1 1]   # 3 rows × 3 codes
accumulate_stanza(codes; window_back=2, binary=true)
# → 3×3 Matrix{Float64}  (choose_two(3) = 3 connection columns)

# Group confidence interval (matches rENA's t.test per dimension)
pts = randn(15, 2)
mean_ci(pts)
# → 2×3 Matrix  [mean  ci_lower  ci_upper] per row = dimension
```

## API reference

### Adjacency

| Function | Returns | Notes |
|----------|---------|-------|
| `connection_names(names)` | `Vector{String}` | `"A & B"` pair labels for every upper-tri position |
| `connection_indices(len; row=-1)` | `2 × n_pairs Matrix{Int32}` | Upper-tri (i, j) index pairs. `row=-1` both rows, `0` row-indices only, `1` col-indices only |
| `code_connections(v)` | `Vector{Float64}` | Pairwise products of a code vector → connection vector |
| `fold_directed_network(v)` | `Vector{Float64}` | n² directed vector → upper-triangle (symmetric pairs summed) |
| `network_to_vector(m; full=true)` | `Vector{Float64}` | Flatten adjacency matrix. `full=true` → n² directed; `false` → upper-tri |

### Normalization

| Function | Returns | Notes |
|----------|---------|-------|
| `normalize_networks(m)` |`accumulate_stanza(codes; window_back=1, window_forward=0, binary=true, ordered=false)` | `Matrix{Float64}` | Stanza-window accumulation. `ordered=false` (default): undirected upper-tri, returns `n_rows × choose_two(n_codes)`. `ordered=true`: directed, returns `n_rows × n_codes²`.| Row-wise L2 normalization. Zero rows left unchanged. |
| `scale_networks(m)` |`accumulate_stanza(codes; window_back=1, window_forward=0, binary=true, ordered=false)` | `Matrix{Float64}` | Stanza-window accumulation. `ordered=false` (default): undirected upper-tri, returns `n_rows × choose_two(n_codes)`. `ordered=true`: directed, returns `n_rows × n_codes²`.| Max-norm scaling: divide all rows by the largest row L2 norm |

### Modeling

| Function | Returns | Notes |
|----------|---------|-------|
| `center_points(m)` |`accumulate_stanza(codes; window_back=1, window_forward=0, binary=true, ordered=false)` | `Matrix{Float64}` | Stanza-window accumulation. `ordered=false` (default): undirected upper-tri, returns `n_rows × choose_two(n_codes)`. `ordered=true`: directed, returns `n_rows × n_codes²`.| Subtract column means |
| `mean_ci(pts; conf_level=0.95)` | `n_dims × 3 Matrix` | `[mean, ci_lower, ci_upper]` — matches rENA `t.test` exactly |
| `outlier_ci(pts; iqr_factor=1.5)` | `n_dims × 2 Matrix` | `[lower, upper]` symmetric around 0 — matches rENA IQR formula |
| `ena_correlation(pts, centroids; conf_level=0.95)` | `n_units × 3 Matrix` | Pearson r with CI: `[r, ci_lower, ci_upper]` |
| `node_positions(adj, t, dims)` | `NamedTuple` | Undirected ENA node positions → `(nodes, centroids, weights, points)` |
| `directed_node_positions(lw, pts, dims)` | `NamedTuple` | Directed ENA node positions → same structure |

### Accumulation

| Function | Returns | Notes |
|----------|---------|-------|
| `connection_matrix(ground, response; response_weight=1.0, ordered=true)` |`accumulate_stanza(codes; window_back=1, window_forward=0, binary=true, ordered=false)` | `Matrix{Float64}` | Stanza-window accumulation. `ordered=false` (default): undirected upper-tri, returns `n_rows × choose_two(n_codes)`. `ordered=true`: directed, returns `n_rows × n_codes²`.| Core adjacency math for one ground+response event pair |
| `accumulate_stanza(codes; window_back=1, window_forward=0, binary=true, ordered=false)` | `Matrix{Float64}` | Stanza-window accumulation. `ordered=false` (default): undirected upper-tri, returns `n_rows × choose_two(n_codes)`. `ordered=true`: directed, returns `n_rows × n_codes²`.| `Matrix{Float64}` | rENA stanza-window. `codes` = code matrix for one conversation. Returns `n_rows × choose_two(n_codes)` |
| `row_connections(codes; binary=true)` |`accumulate_stanza(codes; window_back=1, window_forward=0, binary=true, ordered=false)` | `Matrix{Float64}` | Stanza-window accumulation. `ordered=false` (default): undirected upper-tri, returns `n_rows × choose_two(n_codes)`. `ordered=true`: directed, returns `n_rows × n_codes²`.| Per-row upper-tri co-occurrence without windowing |
| `rolling_window_sum(codes; window_size=1)` |`accumulate_stanza(codes; window_back=1, window_forward=0, binary=true, ordered=false)` | `Matrix{Float64}` | Stanza-window accumulation. `ordered=false` (default): undirected upper-tri, returns `n_rows × choose_two(n_codes)`. `ordered=true`: directed, returns `n_rows × n_codes²`.| Rolling backward sum of raw code values |
| `flat_index(indices, dims)` | `Int` | Column-major linear index. `indices` and `dims` are **0-based** |
| `accumulate_unit(codes, unit_rows, decay_fn; ordered=false)` | `Vector{Float64}` | tma ground/response accumulation. `unit_rows` are **0-based** `Vector{Int32}` |
| `accumulate_unit_with_rows(codes, unit_rows, decay_fn; ordered=false)` | `NamedTuple` | Like `accumulate_unit` + per-response-row networks → `(networks, row_networks)` |

## Running tests

```bash
cd julia/LibQE
julia --project=. -e 'using Pkg; Pkg.test()'
```
