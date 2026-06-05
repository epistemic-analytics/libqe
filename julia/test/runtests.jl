using Test
using LibQE

# ── Adjacency ─────────────────────────────────────────────────────────────────

@testset "svector_to_upper_tri" begin
    pairs = svector_to_upper_tri(["A", "B", "C"])
    @test pairs == ["A & B", "A & C", "B & C"]
    @test length(pairs) == 3
end

@testset "tri_indices shape" begin
    idx = tri_indices(4)          # choose_two(4) = 6
    @test size(idx) == (2, 6)
    @test all(idx .>= 0)
end

@testset "vector_to_upper_tri" begin
    v   = Float64[1, 2, 3]
    ut  = vector_to_upper_tri(v)
    @test length(ut) == 3         # choose_two(3)
    @test ut[1] ≈ 1*2             # v[1]*v[2]
    @test ut[2] ≈ 1*3             # v[1]*v[3]
    @test ut[3] ≈ 2*3             # v[2]*v[3]
end

# ── Normalization ─────────────────────────────────────────────────────────────

@testset "sphere_norm" begin
    m = Matrix{Float64}(I, 3, 3)  # using LinearAlgebra
    n = sphere_norm(m)
    @test size(n) == (3, 3)
    for r in eachrow(n)
        @test norm(r) ≈ 1.0 atol=1e-10
    end
end

@testset "skip_sphere_norm output shape" begin
    m = rand(5, 3)
    @test size(skip_sphere_norm(m)) == (5, 3)
end

# ── Modeling ──────────────────────────────────────────────────────────────────

@testset "center_data" begin
    m = Float64[1 2; 3 4; 5 6]   # 3×2, column-major in Julia
    c = center_data(m)
    @test size(c) == (3, 2)
    @test all(abs.(mean(c, dims=1)) .< 1e-10)
end

@testset "group_ci shape and rENA alignment" begin
    rng = MersenneTwister(42)
    pts = randn(rng, 15, 2)
    ci  = group_ci(pts)
    @test size(ci) == (2, 3)      # n_dims × [mean, lower, upper]
    # lower < mean < upper for each dimension
    @test all(ci[:, 2] .< ci[:, 1])
    @test all(ci[:, 1] .< ci[:, 3])
end

@testset "outlier_ci shape and symmetry" begin
    rng = MersenneTwister(42)
    pts = randn(rng, 20, 2)
    oi  = outlier_ci(pts)
    @test size(oi) == (2, 2)      # n_dims × [lower, upper]
    @test all(oi[:, 1] .≈ -oi[:, 2])   # symmetric around 0
end

# ── Accumulation ─────────────────────────────────────────────────────────────

@testset "stanza_window shape" begin
    codes = Float64[1 1 0; 1 0 1; 0 1 1]   # 3 rows × 3 codes
    out   = stanza_window(codes; window_back=2, binary=true)
    @test size(out) == (3, 3)              # 3 rows × choose_two(3)
    @test all(out .∈ Ref([0.0, 1.0]))      # binary output
end

@testset "rows_to_co_occurrences" begin
    codes = Float64[1 0 1; 0 1 1]
    out   = rows_to_co_occurrences(codes)
    @test size(out) == (2, 3)             # 2 rows × choose_two(3)
end

@testset "rolling_window_sum identity at window=1" begin
    codes = Float64[1 2 3; 4 5 6; 7 8 9]
    out   = rolling_window_sum(codes; window_size=1)
    @test out ≈ codes
end

@testset "calculate_1d_index" begin
    # 3D array dims [2,3,4], 0-based index [1,2,3] → 1 + 2*2 + 3*6 = 23
    @test calculate_1d_index([1, 2, 3], [2, 3, 4]) == 23
end

@testset "accumulate_unit with uniform decay" begin
    codes    = Matrix{Float64}(I, 3, 3)
    unit_rows = Int32[1, 2]
    decay    = dists -> ones(Float64, length(dists))
    result   = accumulate_unit(codes, unit_rows, decay; ordered=false)
    @test length(result) == 3     # choose_two(3)
end
