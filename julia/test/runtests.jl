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

# ── NaN / Inf input guards ────────────────────────────────────────────────────

@testset "normalize_networks NaN row becomes zeros" begin
    m      = [NaN NaN; 1.0 0.0]
    result = normalize_networks(m)
    @test !any(isnan, result)
    @test result[2, 1] ≈ 1.0
end

@testset "scale_networks NaN row: norm skipped, finite rows still scaled" begin
    m      = [NaN NaN; 3.0 4.0]
    result = scale_networks(m)
    # NaN values remain in the NaN row (in-place divide); finite rows are scaled
    @test all(isnan, result[1, :])
    @test result[2, 1] ≈ 3.0 / 5.0
    @test result[2, 2] ≈ 4.0 / 5.0
end
