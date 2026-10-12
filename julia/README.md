# LibQE.jl

Julia bindings for [libqe](https://github.com/epistemic-analytics/libqe),
the shared C++ core for Quantitative Ethnography packages.

Built with [CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl).
All functions accept standard Julia `Matrix{Float64}` / `Vector{Float64}` arguments.
Matrix inputs are **zero-copy** across the C++ boundary (Julia and Armadillo are both
column-major).

## Quick start

```julia
using LibQE

# Code-pair labels
connection_names(["A", "B", "C"])
# → ["A & B", "A & C", "B & C"]

# Group confidence interval (matches rENA's t.test per dimension)
pts = randn(15, 2)
mean_ci(pts)
# → 2×3 Matrix  [mean  ci_lower  ci_upper] per row = dimension

# Two-group comparison statistics
g1, g2 = randn(20, 2), randn(18, 2) .+ 0.5
r = group_stats(g1, g2)
# r.t        → Vector{Float64}  Welch t-statistic per dimension
# r.pvalue_t → Vector{Float64}  two-tailed p-value per dimension
# r.cohens_d → Vector{Float64}  Cohen's d per dimension
# r.means    → 2×n_dims Matrix  group means
```

## API reference

The ENA model code (rotations, node positions, `ena_correlation`) is in ENA.jl
(rENA) and all accumulation in TMA.jl (tma); both were removed from LibQE.jl
in 0.2.0.

### Adjacency

| Function | Returns | Notes |
|----------|---------|-------|
| `connection_names(names)` | `Vector{String}` | `"A & B"` pair labels for every upper-tri position |
| `connection_indices(len; row=-1)` | `2 × n_pairs Matrix{Int32}` | Upper-tri (i, j) index pairs. `row=-1` both rows, `0` row-indices, `1` col-indices |
| `code_connections(v)` | `Vector{Float64}` | Pairwise products of a code vector → connection vector |
| `fold_directed_network(v)` | `Vector{Float64}` | n² directed vector → upper-triangle (symmetric pairs summed) |
| `network_to_vector(m; full=true)` | `Vector{Float64}` | Flatten adjacency matrix. `full=true` → n²; `false` → upper-tri |

### Normalization

| Function | Returns | Notes |
|----------|---------|-------|
| `normalize_networks(m)` | `Matrix{Float64}` | Row-wise L2 normalization; zero rows left unchanged |
| `scale_networks(m)` | `Matrix{Float64}` | Divide all rows by the largest row L2 norm |

### Modeling

| Function | Returns | Notes |
|----------|---------|-------|
| `center_points(m)` | `Matrix{Float64}` | Subtract column means |
| `mean_ci(pts; conf_level=0.95)` | `n_dims × 3 Matrix` | `[mean, ci_lower, ci_upper]` — matches rENA `t.test` exactly |
| `outlier_ci(pts; iqr_factor=1.5)` | `n_dims × 2 Matrix` | `[lower, upper]` symmetric around 0 — matches rENA IQR formula |
| `group_stats(g1, g2)` | `GroupStatsJ` | Per-dimension Welch t-test + Wilcoxon rank-sum. See fields below |

#### `GroupStatsJ` fields

| Field | Type | Notes |
|-------|------|-------|
| `n1`, `n2` | `Int32` | Sample sizes |
| `t_stat` | `Vector{Float64}` | Welch t-statistic per dimension |
| `df` | `Vector{Float64}` | Welch–Satterthwaite degrees of freedom |
| `pvalue_t` | `Vector{Float64}` | Two-tailed p-values (t-test) |
| `cohens_d` | `Vector{Float64}` | Cohen's d (pooled SD, group1 − group2) |
| `means` | `2 × n_dims Matrix` | Row 0 = group1, row 1 = group2 |
| `sds` | `2 × n_dims Matrix` | Sample standard deviations |
| `U` | `Vector{Float64}` | Wilcoxon U for group 1 (= R's W) |
| `pvalue_u` | `Vector{Float64}` | Two-tailed p-values (normal approx, tie + continuity correction) |
| `effect_r` | `Vector{Float64}` | Rank-biserial: 1 − 2·U / (n1·n2) |
| `medians` | `2 × n_dims Matrix` | Row 0 = group1, row 1 = group2 |

## Running tests

```julia
cd julia
julia --project=. -e 'using Pkg; Pkg.test()'
```
