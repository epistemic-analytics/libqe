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
    choose_two, connection_indices, code_connections,
    fold_directed_network, network_to_vector, connection_names,
    # Normalization
    normalize_networks, scale_networks,
    # Modeling
    center_points, mean_ci, outlier_ci, ena_correlation,
    node_positions, directed_node_positions,
    directed_node_positions_combine_pairs, group_stats,
    # Accumulation
    connection_matrix, accumulate_stanza, row_connections,
    rolling_window_sum, flat_index, accumulate_unit,
    accumulate_unit_with_rows, accumulate_tensor_unit,
    # Rotation
    ena_svd, deflate, orthogonal_svd, complete_rotation,
    means_rotation, generalized_means_rotation

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

function _sv_i32(v::AbstractVector{<:Integer})
    sv = CxxWrap.StdLib.StdVector{Int32}()
    for x in v; push!(sv, Int32(x)); end
    sv
end

function _sv_str(v::AbstractVector{<:AbstractString})
    sv = CxxWrap.StdLib.StdVector{CxxWrap.StdLib.StdString}()
    for x in v; push!(sv, CxxWrap.StdLib.StdString(x)); end
    sv
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

# Unpack a RotationResultJ into a NamedTuple.
function _unpack_rotation(r)
    raw_names = column_names(r)   # StdVector{StdString} from C++
    (
        rotation     = reshape(rot_matrix(r), rot_rows(r), rot_cols(r)),
        eigenvalues  = eigenvalues(r),
        column_names = String[String(raw_names[i]) for i in 1:length(raw_names)],
    )
end

# Unpack a TensorNetworksJ into a NamedTuple.
function _unpack_tensor_networks(r)
    (
        connection_counts    = connection_counts(r),
        row_connection_counts = reshape(row_networks(r),
                                        row_networks_rows(r),
                                        row_networks_cols(r)),
    )
end

# ── Adjacency ─────────────────────────────────────────────────────────────────

"""
    connection_indices(len; row=-1) -> Matrix{Int32}

Upper-triangle (i, j) index pairs for a matrix of side `len`.
`row = -1` returns both rows, `0` row indices only, `1` col indices only.
Result is a `2 × choose_two(len)` matrix.
"""
function connection_indices(len::Integer; row::Integer = -1)
    n_pairs = len * (len - 1) ÷ 2
    flat = connection_indices(Int32(len), Int32(row))
    reshape(flat, 2, n_pairs)
end

"""
    code_connections(v) -> Vector{Float64}

Pairwise products of a code vector → upper-triangle connection vector.
"""
function code_connections(v::Vector{Float64})
    code_connections(v, Int32(length(v)))
end

# fold_directed_network(v::Vector{Float64}) -> Vector{Float64}
# CxxWrap maps this directly (C++ binding takes only the vector; length is
# derived internally). No Julia wrapper needed — call it directly.
#   fold_directed_network(v) → upper-triangle Vector{Float64}

"""
    connection_names(names) -> Vector{String}

`"A & B"` pair labels for every upper-tri position.
"""
function connection_names(names::AbstractVector{<:AbstractString})
    # CxxWrap exposes connection_names(StdVector{StdString}); convert both ways.
    raw = connection_names(_sv_str(names))   # C++ binding; returns StdVector{StdString}
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
    ena_correlation(points, centroids; conf_level=0.95) -> Matrix{Float64}

Pearson correlation with CI between ENA unit points and group centroids.
Returns an `n_units × 3` matrix with columns `[r, ci_lower, ci_upper]`.
"""
function ena_correlation(points::Matrix{Float64}, centroids::Matrix{Float64};
                          conf_level::Float64 = 0.95)
    pr, pc = size(points)
    cr, cc = size(centroids)
    n_units = pr
    reshape(ena_correlation(vec(points), Int32(pr), Int32(pc),
                                vec(centroids), Int32(cr), Int32(cc),
                                conf_level), n_units, 3)
end

"""
    node_positions(adj_mats, t, num_dims) -> NamedTuple

Least-squares node positions for undirected ENA.
Returns `(nodes, centroids, weights, points)` — each a `Matrix{Float64}`.
"""
function node_positions(adj_mats::Matrix{Float64}, t::Matrix{Float64},
                             num_dims::Integer)
    (all(isfinite, adj_mats) && all(isfinite, t)) ||
        throw(ArgumentError("node_positions: input matrices must not contain NaN or Inf — " *
                            "filter or impute rows with non-finite values before calling"))
    ar, ac = size(adj_mats)
    tr, tc = size(t)
    r = node_positions(vec(adj_mats), Int32(ar), Int32(ac),
                       vec(t),        Int32(tr), Int32(tc), Int32(num_dims))
    _unpack_positions(r)
end

"""
    directed_node_positions(line_weights, points, num_dims) -> NamedTuple

Least-squares node positions for directed ENA.
Returns `(nodes, centroids, weights, points)`.
"""
function directed_node_positions(line_weights::Matrix{Float64},
                                  points::Matrix{Float64}, num_dims::Integer)
    (all(isfinite, line_weights) && all(isfinite, points)) ||
        throw(ArgumentError("directed_node_positions: input matrices must not contain NaN or Inf — " *
                            "filter or impute rows with non-finite values before calling"))
    lr, lc = size(line_weights)
    pr, pc = size(points)
    r = directed_node_positions(vec(line_weights), Int32(lr), Int32(lc),
                                vec(points),       Int32(pr), Int32(pc),
                                Int32(num_dims))
    _unpack_positions(r)
end

"""
    directed_node_positions_combine_pairs(line_weights, points, num_dims) -> NamedTuple

Directed ENA node positions — ground and response rows averaged before the
least-squares solve (`combine_pairs = true`).
Returns `(nodes, centroids, weights, points)`.
"""
function directed_node_positions_combine_pairs(line_weights::Matrix{Float64},
                                                points::Matrix{Float64},
                                                num_dims::Integer)
    (all(isfinite, line_weights) && all(isfinite, points)) ||
        throw(ArgumentError("directed_node_positions_combine_pairs: input matrices must not contain NaN or Inf — " *
                            "filter or impute rows with non-finite values before calling"))
    lr, lc = size(line_weights)
    r = directed_node_positions_combine_pairs(
            vec(line_weights), Int32(lr), Int32(lc),
            vec(points),       Int32(pr), Int32(pc),
            Int32(num_dims))
    _unpack_positions(r)
end

# ── Accumulation ──────────────────────────────────────────────────────────────

"""
    connection_matrix(ground, response; response_weight=1.0, ordered=true)
    -> Matrix{Float64}

Core adjacency math for one ground+response event pair.
"""
function connection_matrix(ground::Vector{Float64}, response::Vector{Float64};
                                     response_weight::Float64 = 1.0, ordered::Bool = true)
    n = length(ground)
    reshape(connection_matrix(ground, Int32(n), response, Int32(n),
                                           response_weight, ordered), n, n)
end

"""
    accumulate_stanza(codes; window_back=1, window_forward=0, binary=true, ordered=false)
    -> Matrix{Float64}

Stanza-window accumulation for one conversation.

- `ordered=false` (default): undirected upper-tri co-occurrences.
  Returns `n_rows × choose(n_codes, 2)`.
  Pass `typemax(Int32)` for an infinite back window.
- `ordered=true`: directed — focal row as response, prior window rows as
  ground. Returns `n_rows × n_codes²`. `window_forward` is ignored.
"""
function accumulate_stanza(codes::Matrix{Float64};
                        window_back::Integer    = 1,
                        window_forward::Integer = 0,
                        binary::Bool            = true,
                        ordered::Bool           = false)
    rows, cols = size(codes)
    n_out = ordered ? cols * cols : cols * (cols - 1) ÷ 2
    result = accumulate_stanza(vec(codes), Int32(rows), Int32(cols),
                               Int32(window_back), Int32(window_forward),
                               binary, ordered)
    reshape(result, rows, n_out)
end

"""
    row_connections(codes; binary=true) -> Matrix{Float64}

Per-row upper-triangle co-occurrence matrix without windowing.
"""
function row_connections(codes::Matrix{Float64}; binary::Bool = true)
    rows, cols = size(codes)
    n_tri = cols * (cols - 1) ÷ 2
    reshape(row_connections(vec(codes), Int32(rows), Int32(cols), binary),
            rows, n_tri)
end

"""
    rolling_window_sum(codes; window_size=1) -> Matrix{Float64}

Rolling backward sum of raw code values (no upper-tri transform).
"""
function rolling_window_sum(codes::Matrix{Float64}; window_size::Integer = 1)
    rows, cols = size(codes)
    reshape(rolling_window_sum(vec(codes), Int32(rows), Int32(cols), Int32(window_size)),
            rows, cols)
end

"""
    flat_index(indices, dims) -> Int

Column-major linear index into a multi-dimensional array. `indices` and `dims`
are **0-based** integer vectors.
"""
function flat_index(indices::AbstractVector{<:Integer}, dims::AbstractVector{<:Integer})
    # CxxWrap 0.15 exposes flat_index as (StdVector{Int32}, StdVector{Int32}).
    # Vector{Int32} does not auto-convert, so we build StdVectors explicitly to
    # break the dispatch loop that would occur from calling flat_index(Int32.(v),...).
    sv_idx = CxxWrap.StdLib.StdVector{Int32}()
    sv_dim = CxxWrap.StdLib.StdVector{Int32}()
    for x in indices; push!(sv_idx, Int32(x)); end
    for x in dims;    push!(sv_dim, Int32(x)); end
    flat_index(sv_idx, sv_dim)
end

"""
    accumulate_unit(codes, unit_rows, decay_fn; ordered=false) -> Vector{Float64}

Ground/response accumulation for one unit (tma model).
`unit_rows` is a **0-based** `Vector{Int32}`.
`decay_fn(distances::Vector{Float64}) -> Vector{Float64}` maps distances to weights.
"""
function accumulate_unit(codes::Matrix{Float64}, unit_rows::AbstractVector{<:Integer},
                          decay_fn::Function; ordered::Bool = false)
    rows, cols = size(codes)
    accumulate_unit(vec(codes), Int32(rows), Int32(cols),
                    _sv_i32(unit_rows), decay_fn, ordered)
end

"""
    accumulate_unit_with_rows(codes, unit_rows, decay_fn; ordered=false)
    -> NamedTuple{(:networks, :row_networks)}

Like `accumulate_unit` but also returns the per-response-row connection matrix.
"""
function accumulate_unit_with_rows(codes::Matrix{Float64}, unit_rows::AbstractVector{<:Integer},
                                    decay_fn::Function; ordered::Bool = false)
    rows, cols = size(codes)
    r = accumulate_unit_with_rows(vec(codes), Int32(rows), Int32(cols),
                                  _sv_i32(unit_rows), decay_fn, ordered)
    n_unit = length(unit_rows)
    (
        networks     = nodes(r),                              # flat Vector{Float64}
        row_networks = reshape(weights(r), n_unit, cols^2),   # n_unit × p²
    )
end

"""
    accumulate_tensor_unit(codes, dims, dims_sender, dims_receiver, dims_mode,
                           context_lookup, unit_rows, times; ordered=true)
    -> NamedTuple{(:connection_counts, :row_connection_counts)}

tma tensor-based accumulation for one unit.

- `codes`          — `Matrix{Float64}` (n_context_rows × n_codes)
- `tensor`         — `Vector{Float64}` flat column-major, shaped by `dims`
- `dims`           — `Vector{Int32}` tensor axis sizes
- `dims_sender`    — `Vector{Int32}` axis indices for sender factors
- `dims_receiver`  — `Vector{Int32}` axis indices for receiver factors
- `dims_mode`      — `Vector{Int32}` axis indices for mode factors
- `context_lookup` — `Matrix{Int32}` (n_context_rows × n_factors), **0-based**
- `unit_rows`      — `Vector{Int32}` **0-based** response-row indices for this unit
- `times`          — `Vector{Float64}` one timestamp per context row
- `ordered`        — `true` → directed n²; `false` → undirected upper-tri

Returns `(connection_counts, row_connection_counts)`.
"""
function accumulate_tensor_unit(codes::Matrix{Float64},
                                 tensor::Vector{Float64},
                                 dims::AbstractVector{<:Integer},
                                 dims_sender::AbstractVector{<:Integer},
                                 dims_receiver::AbstractVector{<:Integer},
                                 dims_mode::AbstractVector{<:Integer},
                                 context_lookup::Matrix{<:Integer},
                                 unit_rows::AbstractVector{<:Integer},
                                 times::Vector{Float64};
                                 ordered::Bool = true)
    rows, cols = size(codes)
    cl_rows, cl_cols = size(context_lookup)
    r = apply_tensor_unit(
            tensor,
            _sv_i32(dims), _sv_i32(dims_sender),
            _sv_i32(dims_receiver), _sv_i32(dims_mode),
            vec(Int32.(context_lookup)), Int32(cl_rows), Int32(cl_cols),
            _sv_i32(unit_rows),
            vec(codes), Int32(rows), Int32(cols),
            times,
            ordered)
    _unpack_tensor_networks(r)
end

# ── Rotation ──────────────────────────────────────────────────────────────────

"""
    ena_svd(points) -> NamedTuple{(:rotation, :eigenvalues, :column_names)}

SVD rotation of ENA point space.
Returns `(rotation, eigenvalues, column_names)`.
"""
function ena_svd(points::Matrix{Float64})
    all(isfinite, points) ||
        throw(ArgumentError("ena_svd: input matrix must not contain NaN or Inf — " *
                            "filter or impute rows with non-finite values before calling"))
    rows, cols = size(points)
    _unpack_rotation(ena_svd(vec(points), Int32(rows), Int32(cols)))
end

"""
    deflate(data, axis) -> Matrix{Float64}

Project out the given unit `axis` from `data` (remove its variance).
"""
function deflate(data::Matrix{Float64}, axis::Vector{Float64})
    rows, cols = size(data)
    reshape(deflate(vec(data), Int32(rows), Int32(cols), axis), rows, cols)
end

"""
    orthogonal_svd(data, weights, labels) -> NamedTuple

Weighted SVD with orthogonalization against previously fixed axes.
`labels` names the resulting axes.
Returns `(rotation, eigenvalues, column_names)`.
"""
function orthogonal_svd(data::Matrix{Float64}, weights::Matrix{Float64},
                         labels::Vector{String})
    dr, dc = size(data)
    wr, wc = size(weights)
    _unpack_rotation(orthogonal_svd(vec(data),    Int32(dr), Int32(dc),
                                    vec(weights), Int32(wr), Int32(wc),
                                    _sv_str(labels)))
end

"""
    complete_rotation(data, named_axes, labels) -> NamedTuple

Fix the columns of `named_axes` as the first rotation axes, then fill the
remaining dimensions with SVD of the doubly-deflated space.
`labels` names the fixed axes (length must equal `size(named_axes, 2)`).
Returns `(rotation, eigenvalues, column_names)`.
"""
function complete_rotation(data::Matrix{Float64}, named_axes::Matrix{Float64},
                            labels::Vector{String})
    dr, dc = size(data)
    ar, ac = size(named_axes)
    _unpack_rotation(complete_rotation(vec(data),       Int32(dr), Int32(dc),
                                       vec(named_axes), Int32(ar), Int32(ac),
                                       _sv_str(labels)))
end

"""
    means_rotation(data, group_pairs) -> NamedTuple

Group-means rotation.

`group_pairs` is a `Vector` of `Tuple{Vector{Int32}, Vector{Int32}}` where
each tuple holds **0-based** row indices for group A and group B.

Returns `(rotation, eigenvalues, column_names)`.
"""
function means_rotation(data::Matrix{Float64},
                         group_pairs::Vector{<:Tuple{Vector{Int32},Vector{Int32}}})
    rows, cols = size(data)
    a_flat  = vcat([p[1] for p in group_pairs]...)
    b_flat  = vcat([p[2] for p in group_pairs]...)
    a_sizes = Int32[length(p[1]) for p in group_pairs]
    b_sizes = Int32[length(p[2]) for p in group_pairs]
    _unpack_rotation(means_rotation(vec(data), Int32(rows), Int32(cols),
                                    _sv_i32(a_flat), _sv_i32(a_sizes),
                                    _sv_i32(b_flat), _sv_i32(b_sizes)))
end

"""
    generalized_means_rotation(V, x_model, x_target, x1_cols, x_categorical,
                                x_n_groups, x_subset, has_y, y_model, y_target,
                                y1_cols, y_categorical, y_n_groups;
                                n_lambda=50, k_folds=5, lasso_eps=0.01) -> NamedTuple

Generalized Means Rotation (GMR) with Lasso-based covariate adjustment.

Mirrors rENA's `ena.rotate.by.generalized()`. The x axis is the direction in
ENA space most explained by `x_target` after controlling for covariates via
Lasso (coordinate-descent, k-fold CV). The y axis is either a second GMR axis
(`has_y=true`) or the leading SVD of the x-deflated space.

All index vectors (`x1_cols`, `x_subset`, `y1_cols`) are **0-based `Int32`**.
Pass `Int32[]` for `x_subset` to use all rows.
Pass empty arrays for all `y_*` arguments when `has_y=false`.

Returns `(rotation, eigenvalues, column_names)` with labels
`GMR1`, `GMR2`|`SVD2`, `SVD3`, …
"""
function generalized_means_rotation(
    V::Matrix{Float64},
    x_model::Matrix{Float64},
    x_target::Vector{Float64},
    x1_cols::Vector{Int32},
    x_categorical::Bool,
    x_n_groups::Int32,
    x_subset::Vector{Int32},
    has_y::Bool,
    y_model::Matrix{Float64},
    y_target::Vector{Float64},
    y1_cols::Vector{Int32},
    y_categorical::Bool,
    y_n_groups::Int32;
    n_lambda::Int=50, k_folds::Int=5, lasso_eps::Float64=0.01
)
    vr, vc = size(V)
    xr, xc = size(x_model)
    yr, yc = size(y_model)
    _unpack_rotation(generalized_means_rotation(
        vec(V),        Int32(vr), Int32(vc),
        vec(x_model),  Int32(xr), Int32(xc),
        x_target,
        _sv_i32(x1_cols),
        x_categorical, x_n_groups,
        _sv_i32(x_subset),
        has_y,
        vec(y_model),  Int32(yr), Int32(yc),
        y_target,
        _sv_i32(y1_cols),
        y_categorical, y_n_groups,
        Int32(n_lambda), Int32(k_folds), lasso_eps))
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
