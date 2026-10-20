"""
    LibQE

Julia bindings for libqe — the shared C++ core for Quantitative Ethnography.

Wraps three modules via CxxWrap.jl:

- **Adjacency**: upper-triangle index and vector utilities
- **Normalization**: row-wise L2 sphere norm, max-norm scaling
- **Modeling**: centering, confidence intervals, two-group statistics

The ENA model code (rotations, node positions) is in ENA.jl and all
accumulation in TMA.jl.

All functions accept standard Julia `Matrix{Float64}` and `Vector{Float64}`
arguments.  Matrix inputs are **zero-copy** (Julia and Armadillo are both
column-major); return values involve one copy across the C++ boundary but
`reshape` back to matrix form is zero-copy in Julia.

## Build

Build the C++ shared library before loading this package:

```sh
cd julia
cmake -B build -DCMAKE_BUILD_TYPE=Release .
cmake --build build
cmake --install build      # copies libqe_julia.so → LibQE/lib/
```

Then in Julia:
```julia
using LibQE
pairs = connection_names(["A", "B", "C"])
```
"""
# CxxWrap loads a native shared library at module init time; precompilation
# of the method table is not safe across different library builds.
__precompile__(false)

module LibQE

using CxxWrap

# ── Public API ────────────────────────────────────────────────────────────────
export
    # Adjacency
    choose_two, n_connections, connection_indices, code_connections,
    fold_directed_network, network_to_vector, connection_names,
    # Normalization
    normalize_networks, scale_networks,
    # Modeling
    center_points, mean_ci, outlier_ci, group_stats

# ── Load shared library ───────────────────────────────────────────────────────
# The .so/.dylib built by CMake is installed into julia/lib/ (one level up
# from this file's julia/src/ directory).  cranqe runs the build step before
# invoking Pkg.test(); for local testing build it manually first:
#   cd julia && cmake -B build -DCMAKE_BUILD_TYPE=Release . \
#               && cmake --build build && cmake --install build
const _lib_dir  = joinpath(@__DIR__, "..", "lib")
const _lib_name = "libqe_julia"

function _lib_path()
    for ext in ("", ".so", ".dylib", ".dll")
        p = joinpath(_lib_dir, _lib_name * ext)
        isfile(p) && return p
    end
    error("libqe_julia shared library not found in $(_lib_dir). " *
          "Build it first: cd julia && cmake -B build -DCMAKE_BUILD_TYPE=Release . " *
          "&& cmake --build build && cmake --install build")
end

@wrapmodule(_lib_path)

function __init__()
    @initcxx
end

# ── StdVector converters ──────────────────────────────────────────────────────
# CxxWrap 0.15 exposes C++ std::vector<T> params as StdVector{T}, not Vector{T}.
# Julia does not auto-convert between the two, so we do it explicitly.

function _sv_str(v::AbstractVector{<:AbstractString})
    sv = CxxWrap.StdLib.StdVector{CxxWrap.StdLib.StdString}()
    for x in v; push!(sv, CxxWrap.StdLib.StdString(x)); end
    sv
end

# ── Adjacency ─────────────────────────────────────────────────────────────────

"""
    n_connections(n; diagonal=false) -> Int

Number of undirected connection columns: `choose_two(n)`, or `n(n+1)/2` with
`diagonal` (self-connections, A & A).
"""
n_connections(n::Integer; diagonal::Bool = false) = Int(n_connections(Int32(n), diagonal))

"""
    connection_indices(len; row=-1, diagonal=false) -> Matrix{Int32}

Upper-triangle (i, j) index pairs for a matrix of side `len`.
`row = -1` returns both rows, `0` row indices only, `1` col indices only.
Result is a `2 × n_connections(len; diagonal)` matrix; with `diagonal` the
pairs include (i, i), in the order (0,0), (0,1), (1,1), (0,2), ...
"""
function connection_indices(len::Integer; row::Integer = -1, diagonal::Bool = false)
    n_pairs = n_connections(len; diagonal = diagonal)
    flat = connection_indices(Int32(len), Int32(row), diagonal)
    reshape(flat, 2, n_pairs)
end

"""
    code_connections(v) -> Vector{Float64}

Pairwise products of a code vector → upper-triangle connection vector.
"""
function code_connections(v::Vector{Float64})
    code_connections(v, Int32(length(v)))
end

"""
    fold_directed_network(v; diagonal=false) -> Vector{Float64}

Fold a directed n² vector to the upper triangle, summing A→B + B→A. With
`diagonal`, each self-connection (A→A) is kept once: A & A, A & B, B & B, ...
"""
fold_directed_network(v::Vector{Float64}; diagonal::Bool = false) =
    collect(Float64, fold_directed_network(v, diagonal))

"""
    connection_names(names; diagonal=false) -> Vector{String}

`"A & B"` pair labels for every upper-tri position; with `diagonal`,
`"A & A", "A & B", "B & B", ...`.
"""
function connection_names(names::AbstractVector{<:AbstractString}; diagonal::Bool = false)
    # CxxWrap exposes connection_names(StdVector{StdString}[, Bool]); convert both ways.
    raw = connection_names(_sv_str(names), diagonal)   # C++ binding; returns StdVector{StdString}
    String[String(raw[i]) for i in 1:length(raw)]
end

# ── Normalization ─────────────────────────────────────────────────────────────

"""
    normalize_networks(m) -> Matrix{Float64}

Row-wise L2 normalization. Zero rows are left unchanged.
"""
function normalize_networks(m::Matrix{Float64})
    rows, cols = size(m)
    reshape(normalize_networks(vec(m), Int32(rows), Int32(cols)), rows, cols)
end

"""
    scale_networks(m) -> Matrix{Float64}

Max-norm scaling: divide all rows by the largest row L2 norm.
"""
function scale_networks(m::Matrix{Float64})
    rows, cols = size(m)
    reshape(scale_networks(vec(m), Int32(rows), Int32(cols)), rows, cols)
end

# ── Modeling ──────────────────────────────────────────────────────────────────

"""
    center_points(m) -> Matrix{Float64}

Subtract column means (center each dimension).
"""
function center_points(m::Matrix{Float64})
    rows, cols = size(m)
    reshape(center_points(vec(m), Int32(rows), Int32(cols)), rows, cols)
end

"""
    mean_ci(points; conf_level=0.95) -> Matrix{Float64}

t-based confidence interval for the mean of a group of ENA unit points.
Returns an `n_dims × 3` matrix with columns `[mean, ci_lower, ci_upper]`.
Matches rENA's `t.test(points[,d])\$conf.int` exactly.
"""
function mean_ci(points::Matrix{Float64}; conf_level::Float64 = 0.95)
    rows, cols = size(points)
    reshape(mean_ci(vec(points), Int32(rows), Int32(cols), conf_level), cols, 3)
end

"""
    outlier_ci(points; iqr_factor=1.5) -> Matrix{Float64}

Outlier interval using IQR × `iqr_factor` (Tukey fence).
Returns an `n_dims × 2` matrix with columns `[lower, upper]`, symmetric
around 0. Matches rENA's IQR-based formula exactly.
"""
function outlier_ci(points::Matrix{Float64}; iqr_factor::Float64 = 1.5)
    rows, cols = size(points)
    reshape(outlier_ci(vec(points), Int32(rows), Int32(cols), iqr_factor), cols, 2)
end

"""
    group_stats(g1, g2) -> GroupStatsJ

Per-dimension Welch t-test and Wilcoxon rank-sum between two groups.
`g1` and `g2` are `Matrix{Float64}` with `n_units × n_dims` layout.
Access fields on the returned struct:
  `t_stat`, `df`, `pvalue_t`, `cohens_d`,
  `U`, `pvalue_u`, `effect_r`,
  `n1`, `n2`, `means`, `sds`, `medians`
"""
function group_stats(g1::Matrix{Float64}, g2::Matrix{Float64})
    g1r, g1c = size(g1)
    g2r, g2c = size(g2)
    group_stats(vec(g1), Int32(g1r), Int32(g1c),
                vec(g2), Int32(g2r), Int32(g2c))
end

end # module LibQE
