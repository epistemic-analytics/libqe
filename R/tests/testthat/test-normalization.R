test_that("sphere_norm: each non-zero row has L2 norm == 1", {
    set.seed(10)
    m   <- matrix(runif(20), nrow = 4)
    out <- normalize_networks(m)
    norms <- apply(out, 1, function(r) sqrt(sum(r^2)))
    expect_equal(norms, rep(1, 4), tolerance = 1e-10)
})

test_that("sphere_norm: zero rows remain zero", {
    m      <- matrix(c(0, 0, 0, 1, 2, 3), nrow = 2, byrow = TRUE)
    out    <- normalize_networks(m)
    expect_equal(out[1, ], c(0, 0, 0))
})

test_that("sphere_norm: direction is preserved (output proportional to input)", {
    set.seed(11)
    m   <- matrix(runif(6), nrow = 2)
    out <- normalize_networks(m)
    # Each row of out should be a positive scalar multiple of input row
    for (i in 1:2) {
        ratio <- m[i, ] / out[i, ]
        expect_equal(diff(ratio), rep(0, ncol(m) - 1), tolerance = 1e-10)
        expect_gt(ratio[1], 0)
    }
})

test_that("skip_sphere_norm: largest row has L2 norm == 1", {
    set.seed(12)
    m     <- matrix(runif(12), nrow = 3)
    out   <- scale_networks(m)
    norms <- apply(out, 1, function(r) sqrt(sum(r^2)))
    expect_equal(max(norms), 1, tolerance = 1e-10)
})

test_that("skip_sphere_norm: relative magnitudes between rows are preserved", {
    m   <- matrix(c(3, 4,   # norm 5
                    6, 8),  # norm 10  → largest; after scaling: norm 1
                  nrow = 2, byrow = TRUE)
    out <- scale_networks(m)
    expect_equal(sqrt(sum(out[2, ]^2)), 1,   tolerance = 1e-10)
    expect_equal(sqrt(sum(out[1, ]^2)), 0.5, tolerance = 1e-10)
})
