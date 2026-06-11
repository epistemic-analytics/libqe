using Test
using LinearAlgebra
using Random
using Statistics
using LibQE

# ── Adjacency ─────────────────────────────────────────────────────────────────

@testset "connection_names" begin
    pairs = connection_names(["A", "B", "C"])
    @test pairs == ["A & B", "A & C", "B & C"]
    @test length(pairs) == 3
end

@testset "connection_indices shape" begin
    idx = connection_indices(4)          # choose_two(4) = 6
    @test size(idx) == (2, 6)
    @test all(idx .>= 0)
end

@testset "code_connections" begin
    v   = Float64[1, 2, 3]
    ut  = code_connections(v)
    @test length(ut) == 3         # choose_two(3)
    @test ut[1] ≈ 1*2             # v[1]*v[2]
    @test ut[2] ≈ 1*3             # v[1]*v[3]
    @test ut[3] ≈ 2*3             # v[2]*v[3]
end

# ── Normalization ─────────────────────────────────────────────────────────────

@testset "normalize_networks" begin
    m = Matrix{Float64}(I, 3, 3)
    n = normalize_networks(m)
    @test size(n) == (3, 3)
    for r in eachrow(n)
        @test norm(r) ≈ 1.0 atol=1e-10
    end
end

@testset "scale_networks output shape" begin
    m = rand(5, 3)
    @test size(scale_networks(m)) == (5, 3)
end

# ── Modeling ──────────────────────────────────────────────────────────────────

@testset "center_points" begin
    m = Float64[1 2; 3 4; 5 6]   # 3×2
    c = center_points(m)
    @test size(c) == (3, 2)
    @test all(abs.(mean(c, dims=1)) .< 1e-10)
end

@testset "mean_ci shape and rENA alignment" begin
    rng = MersenneTwister(42)
    pts = randn(rng, 15, 2)
    ci  = mean_ci(pts)
    @test size(ci) == (2, 3)      # n_dims × [mean, lower, upper]
    @test all(ci[:, 2] .< ci[:, 1])
    @test all(ci[:, 1] .< ci[:, 3])
end

@testset "outlier_ci shape and symmetry" begin
    rng = MersenneTwister(42)
    pts = randn(rng, 20, 2)
    oi  = outlier_ci(pts)
    @test size(oi) == (2, 2)      # n_dims × [lower, upper]
    @test all(oi[:, 1] .≈ -oi[:, 2])
end

# ── Accumulation ─────────────────────────────────────────────────────────────

@testset "accumulate_stanza shape" begin
    codes = Float64[1 1 0; 1 0 1; 0 1 1]   # 3 rows × 3 codes
    out   = accumulate_stanza(codes; window_back=2, binary=true)
    @test size(out) == (3, 3)              # 3 rows × choose_two(3)
    @test all(out .∈ Ref([0.0, 1.0]))      # binary output
end

@testset "row_connections" begin
    codes = Float64[1 0 1; 0 1 1]
    out   = row_connections(codes)
    @test size(out) == (2, 3)             # 2 rows × choose_two(3)
end

@testset "rolling_window_sum identity at window=1" begin
    codes = Float64[1 2 3; 4 5 6; 7 8 9]
    out   = rolling_window_sum(codes; window_size=1)
    @test out ≈ codes
end

@testset "flat_index" begin
    # 3D array dims [2,3,4], 0-based index [1,2,3] → 1 + 2*2 + 3*6 = 23
    @test flat_index(Int32[1, 2, 3], Int32[2, 3, 4]) == 23
end

@testset "accumulate_unit with uniform decay" begin
    codes     = Matrix{Float64}(I, 3, 3)
    unit_rows = Int32[1, 2]
    decay     = dists -> ones(Float64, length(dists))
    result    = accumulate_unit(codes, unit_rows, decay; ordered=false)
    @test length(result) == 3     # choose_two(3)
end

# ── NaN / Inf input guards ────────────────────────────────────────────────────

@testset "normalize_networks NaN row becomes zeros" begin
    m      = [NaN NaN; 1.0 0.0]
    result = normalize_networks(m)
    @test !any(isnan, result)
    @test result[2, 1] ≈ 1.0
end

@testset "scale_networks NaN row skipped gracefully" begin
    m      = [NaN NaN; 3.0 4.0]
    result = scale_networks(m)
    @test !any(isnan, result)
end

@testset "node_positions rejects NaN in adj_mats" begin
    adj = [0.1 0.2 NaN; 0.4 0.5 0.6; 0.7 0.8 0.9; 0.1 0.2 0.3; 0.4 0.5 0.6]
    t   = rand(5, 2)
    @test_throws ArgumentError node_positions(adj, t, 2)
end

@testset "node_positions rejects Inf in points" begin
    adj = rand(5, 6)
    t   = [1.0 2.0; Inf 0.0; 0.0 1.0; 1.0 0.0; 0.5 0.5]
    @test_throws ArgumentError node_positions(adj, t, 2)
end

@testset "node_positions succeeds on clean inputs" begin
    Random.seed!(7)
    adj = rand(5, 6)
    t   = rand(5, 2)
    r   = node_positions(adj, t, 2)
    @test size(r.nodes, 2) == 2
    @test all(isfinite, r.nodes)
end

@testset "directed_node_positions rejects NaN" begin
    lw  = [0.1 0.2 NaN 0.4; 0.5 0.6 0.7 0.8; 0.9 0.1 0.2 0.3; 0.4 0.5 0.6 0.7; 0.8 0.9 0.1 0.2]
    pts = rand(5, 2)
    @test_throws ArgumentError directed_node_positions(lw, pts, 2)
end

@testset "ena_svd rejects NaN input" begin
    pts = [1.0 2.0; NaN 0.0; 0.0 1.0; 1.0 0.0; 0.5 0.5]
    @test_throws ArgumentError ena_svd(pts)
end

@testset "ena_svd rejects Inf input" begin
    pts = [1.0 2.0; Inf 0.0; 0.0 1.0; 1.0 0.0; 0.5 0.5]
    @test_throws ArgumentError ena_svd(pts)
end

@testset "ena_svd succeeds on clean inputs" begin
    Random.seed!(8)
    pts = rand(5, 2)
    r   = ena_svd(pts)
    @test size(r.rotation) == (2, 2)
    @test all(isfinite, r.rotation)
end
