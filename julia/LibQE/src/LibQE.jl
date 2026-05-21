"""
    LibQE

Julia bindings for libqe — the shared C++ core for Quantitative Ethnography.

Wraps four modules via CxxWrap.jl:

- **Adjacency**: upper-triangle index and vector utilities
- **Normalization**: row-wise L2 sphere norm, max-norm scaling
- **Modeling**: centering, ENA correlation, least-squares node positions
- **Accumulation**: stanza-window (rENA), ground/response (tma), rolling sum

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
pairs = svector_to_upper_tri(["A", "B", "C"])
```
"""
module LibQE

using CxxWrap

# ── Load shared library ───────────────────────────────────────────────────────
# The .so/.dylib built by CMake is installed into LibQE/lib/.
const _lib_dir  = joinpath(@__DIR__, "..", "lib")
const _lib_name = "libqe_julia"

function _lib_path()
    for ext in ("", ".so", ".dylib", ".dll")
        p = joinpath(_lib_dir, _lib_name * ext)
        isfile(p) && return p
    end
    error("libqe_julia shared library not found in $(_lib_dir). " *
          "Build it first: cd julia && cmake -B build . && cmake --build build --target install")
end

@wrapmodule(_lib_path)

function __init__()
    @initcxx
end

# ── Internal helpers ──────────────────────────────────────────────────────────

# Unpack a NodePositionsResult into a NamedTuple.
function _unpack_positions(r)
    (
        nodes     = reshape(nodes(r),     nodes_rows(r),     nodes_cols(r)),
        centroids = reshape(centroids(r), centroids_rows(r), centroids_cols(r)),
        weights   = reshape(weights(r),   weights_rows(r),   weights_cols(r)),
        points    = reshape(points(r),    points_rows(r),    points_cols(r)),
    )
end

# ── Adjacency ─────────────────────────────────────────────────────────────────

"""
    tri_indices(len; row=-1) -> Matrix{Int32}

Upper-triangle (i, j) index pairs for a matrix of side `len`.
`row = -1` returns both rows, `0` row indices only, `1` col indices only.
Result is a `2 × choose_two(len)` matrix.
"""
function tri_indices(len::Integer; row::Integer = -1)
    n_pairs = len * (len - 1) ÷ 2
    flat = lq_tri_indices(Int32(len), Int32(row))
    reshape(flat, 2, n_pairs)
end

"""
    vector_to_upper_tri(v) -> Vector{Float64}

Pairwise products of a code vector → upper-triangle connection vector.
"""
function vector_to_upper_tri(v::Vector{Float64})
    lq_vector_to_upper_tri(v, Int32(length(v)))
end

"""
    directed_to_upper_tri(v) -> Vector{Float64}

Fold a directed (n²) flat vector into an undirected upper-triangle vector.
"""
directed_to_upper_tri(v::Vector{Float64}) = lq_directed_to_upper_tri(v)

"""
    adjacency_matrix_to_vector(m; full=true) -> Vector{Float64}

Flatten an adjacency matrix. `full=true` returns the full n² vector (directed);
`full=false` returns the upper triangle (undirected).
"""
function adjacency_matrix_to_vector(m::Matrix{Float64}; full::Bool = true)
    rows, cols = size(m)
    lq_adjacency_matrix_to_vector(m, Int32(rows), Int32(cols), full)
end

"""
    svector_to_upper_tri(names) -> Vector{String}

Generate `"A & B"` pair labels for every upper-triangle position.
"""
svector_to_upper_tri(names::Vector{String}) = lq_svector_to_upper_tri(names)

# ── Normalization ─────────────────────────────────────────────────────────────

"""
    sphere_norm(m) -> Matrix{Float64}

Row-wise L2 normalization. Zero rows are left unchanged.
"""
function sphere_norm(m::Matrix{Float64})
    rows, cols = size(m)
    reshape(lq_sphere_norm(m, Int32(rows), Int32(cols)), rows, cols)
end

"""
    skip_sphere_norm(m) -> Matrix{Float64}

Max-norm scaling: divide all rows by the largest row L2 norm.
"""
function skip_sphere_norm(m::Matrix{Float64})
    rows, cols = size(m)
    reshape(lq_skip_sphere_norm(m, Int32(rows), Int32(cols)), rows, cols)
end

# ── Modeling ──────────────────────────────────────────────────────────────────

"""
    center_data(m) -> Matrix{Float64}

Subtract column means (center each dimension).
"""
function center_data(m::Matrix{Float64})
    rows, cols = size(m)
    reshape(lq_center_data(m, Int32(rows), Int32(cols)), rows, cols)
end

"""
    group_ci(points; conf_level=0.95) -> Matrix{Float64}

t-based confidence interval for the mean of a group of ENA unit points.
Returns an `n_dims × 3` matrix with columns `[mean, ci_lower, ci_upper]`.
Matches rENA's `t.test(points[,d])\$conf.int` exactly.
"""
function group_ci(points::Matrix{Float64}; conf_level::Float64 = 0.95)
    rows, cols = size(points)
    reshape(lq_group_ci(points, Int32(rows), Int32(cols), conf_level), cols, 3)
end

"""
    outlier_ci(points; iqr_factor=1.5) -> Matrix{Float64}

Outlier interval using IQR × `iqr_factor` (Tukey fence).
Returns an `n_dims × 2` matrix with columns `[lower, upper]`, symmetric
around 0. Matches rENA's IQR-based formula exactly.
"""
function outlier_ci(points::Matrix{Float64}; iqr_factor::Float64 = 1.5)
    rows, cols = size(points)
    reshape(lq_outlier_ci(points, Int32(rows), Int32(cols), iqr_factor), cols, 2)
end

"""
    ena_correlation(points, centroids; conf_level=0.95) -> Matrix{Float64}

Pearson correlation with CI between ENA unit points and group centroids.
Returns an `n_units × 3` matrix with columns `[r, ci_lower, ci_upper]`.
"""
function ena_correlation(points::Matrix{Float64}, centroids::Matrix{Float64};
                          conf_level::Float64 = 0.95)
    pr, pc = size(points)
    cr, cc = size(centroids)
    n_units = pr
    reshape(lq_ena_correlation(points, Int32(pr), Int32(pc),
                                centroids, Int32(cr), Int32(cc),
                                conf_level), n_units, 3)
end

"""
    lws_lsq_positions(adj_mats, t, num_dims) -> NamedTuple

Least-squares node positions for undirected ENA.
Returns `(nodes, centroids, weights, points)` — each a `Matrix{Float64}`.
"""
function lws_lsq_positions(adj_mats::Matrix{Float64}, t::Matrix{Float64},
                             num_dims::Integer)
    ar, ac = size(adj_mats)
    tr, tc = size(t)
    r = lq_lws_lsq_positions(adj_mats, Int32(ar), Int32(ac),
                               t,        Int32(tr), Int32(tc), Int32(num_dims))
    _unpack_positions(r)
end

"""
    directed_node_positions(line_weights, points, num_dims) -> NamedTuple

Least-squares node positions for directed ENA.
Returns `(nodes, centroids, weights, points)`.
"""
function directed_node_positions(line_weights::Matrix{Float64},
                                  points::Matrix{Float64}, num_dims::Integer)
    lr, lc = size(line_weights)
    pr, pc = size(points)
    r = lq_directed_node_positions(line_weights, Int32(lr), Int32(lc),
                                    points,       Int32(pr), Int32(pc),
                                    Int32(num_dims))
    _unpack_positions(r)
end

# ── Accumulation ──────────────────────────────────────────────────────────────

"""
    calculate_adjacency_matrix(ground, response; response_weight=1.0, ordered=true)
    -> Matrix{Float64}

Core adjacency math for one ground+response event pair.
"""
function calculate_adjacency_matrix(ground::Vector{Float64}, response::Vector{Float64};
                                     response_weight::Float64 = 1.0, ordered::Bool = true)
    n = length(ground)
    reshape(lq_calculate_adjacency_matrix(ground, Int32(n), response, Int32(n),
                                           response_weight, ordered), n, n)
end

"""
    stanza_window(codes; window_back=1, window_forward=0, binary=true)
    -> Matrix{Float64}

Traditional stanza-window accumulation (rENA model).  `codes` is the code
matrix for **one conversation**.  Returns a matrix with `choose(n_codes, 2)`
columns.  Pass `typemax(Int32)` for an infinite back window.
"""
function stanza_window(codes::Matrix{Float64};
                        window_back::Integer    = 1,
                        window_forward::Integer = 0,
                        binary::Bool            = true)
    rows, cols = size(codes)
    n_tri = cols * (cols - 1) ÷ 2
    result = lq_stanza_window(codes, Int32(rows), Int32(cols),
                               Int32(window_back), Int32(window_forward), binary)
    reshape(result, rows, n_tri)
end

"""
    rows_to_co_occurrences(codes; binary=true) -> Matrix{Float64}

Per-row upper-triangle co-occurrence matrix without windowing.
"""
function rows_to_co_occurrences(codes::Matrix{Float64}; binary::Bool = true)
    rows, cols = size(codes)
    n_tri = cols * (cols - 1) ÷ 2
    reshape(lq_rows_to_co_occurrences(codes, Int32(rows), Int32(cols), binary),
            rows, n_tri)
end

"""
    rolling_window_sum(codes; window_size=1) -> Matrix{Float64}

Rolling backward sum of raw code values (no upper-tri transform).
"""
function rolling_window_sum(codes::Matrix{Float64}; window_size::Integer = 1)
    rows, cols = size(codes)
    reshape(lq_rolling_window_sum(codes, Int32(rows), Int32(cols), Int32(window_size)),
            rows, cols)
end

"""
    calculate_1d_index(indices, dims) -> Int

Column-major linear index into a multi-dimensional array. `indices` and `dims`
are **0-based** integer vectors.
"""
function calculate_1d_index(indices::Vector{<:Integer}, dims::Vector{<:Integer})
    lq_calculate_1d_index(Int32.(indices), Int32.(dims))
end

"""
    accumulate_unit(codes, unit_rows, decay_fn; ordered=false) -> Vector{Float64}

Ground/response accumulation for one unit (tma model).
`unit_rows` is a **0-based** `Vector{Int32}`.
`decay_fn(distances::Vector{Float64}) -> Vector{Float64}` maps distances to weights.
"""
function accumulate_unit(codes::Matrix{Float64}, unit_rows::Vector{Int32},
                          decay_fn::Function; ordered::Bool = false)
    rows, cols = size(codes)
    lq_accumulate_unit(codes, Int32(rows), Int32(cols),
                        unit_rows, decay_fn, ordered)
end

"""
    accumulate_unit_with_rows(codes, unit_rows, decay_fn; ordered=false)
    -> NamedTuple{(:networks, :row_networks)}

Like `accumulate_unit` but also returns the per-response-row connection matrix.
"""
function accumulate_unit_with_rows(codes::Matrix{Float64}, unit_rows::Vector{Int32},
                                    decay_fn::Function; ordered::Bool = false)
    rows, cols = size(codes)
    r = lq_accumulate_unit_with_rows(codes, Int32(rows), Int32(cols),
                                      unit_rows, decay_fn, ordered)
    n_unit = length(unit_rows)
    (
        networks     = nodes(r),                              # flat Vector{Float64}
        row_networks = reshape(weights(r), n_unit, cols^2),   # n_unit × p²
    )
end

end # module LibQE
