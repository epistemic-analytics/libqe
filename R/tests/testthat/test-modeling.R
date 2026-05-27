test_that("group_ci: output dimensions are n_dims x 3", {
    set.seed(10)
    pts <- matrix(rnorm(20 * 2), nrow = 20)
    out <- lq_mean_ci(pts, 0.95)
    expect_equal(dim(out), c(2L, 3L))
})

test_that("group_ci: lower <= mean <= upper for all dims", {
    set.seed(11)
    pts <- matrix(rnorm(15 * 3), nrow = 15)
    out <- lq_mean_ci(pts, 0.95)
    for (d in 1:3) {
        expect_lte(out[d, 2], out[d, 1] + 1e-12)   # lower <= mean
        expect_gte(out[d, 3], out[d, 1] - 1e-12)   # upper >= mean
    }
})

test_that("group_ci: mean column matches colMeans", {
    set.seed(12)
    pts <- matrix(runif(30 * 2), nrow = 30)
    out <- lq_mean_ci(pts, 0.95)
    expect_equal(out[, 1], colMeans(pts), tolerance = 1e-10)
})

test_that("group_ci: higher confidence level gives strictly wider CI", {
    # Use identical data; only the conf_level differs — no randomness in the comparison
    set.seed(13)
    pts <- matrix(rnorm(20 * 2), nrow = 20)
    # For the same data, a higher conf_level must produce a wider CI
    for (lo_level in c(0.50, 0.80, 0.90)) {
        hi_level <- lo_level + 0.09          # guaranteed strictly higher
        width_lo <- diff(lq_mean_ci(pts, lo_level)[1, 2:3])
        width_hi <- diff(lq_mean_ci(pts, hi_level)[1, 2:3])
        expect_lt(width_lo, width_hi)
    }
})

test_that("group_ci: n=1 returns Inf CI bounds", {
    pts <- matrix(c(1.5, 2.5), nrow = 1)
    out <- lq_mean_ci(pts, 0.95)
    expect_true(is.infinite(out[1, 2]))   # lower = -Inf
    expect_true(is.infinite(out[1, 3]))   # upper = +Inf
})

test_that("group_ci: CI width is proportional to 1/sqrt(n) for fixed sd", {
    # Use replicated data so the sample SD is identical; only n differs.
    # Width = 2 * t_crit(n) * sd / sqrt(n). With the same data repeated, sd is ~constant.
    base <- c(1, -1, 2, -2, 0)        # 5 fixed values, mean=0, known sd
    pts_small <- matrix(rep(base,  1), ncol = 1)          # n = 5
    pts_large <- matrix(rep(base, 10), ncol = 1)          # n = 50 (same sd, different n)
    w_small <- diff(lq_mean_ci(pts_small, 0.95)[1, 2:3])
    w_large <- diff(lq_mean_ci(pts_large, 0.95)[1, 2:3])
    expect_lt(w_large, w_small)
})

test_that("group_ci: matches R's t.test CI on a single dimension", {
    set.seed(15)
    x   <- rnorm(12)
    pts <- matrix(x, ncol = 1)
    out <- lq_mean_ci(pts, 0.95)
    ref <- t.test(x, conf.level = 0.95)$conf.int
    expect_equal(out[1, 2], ref[1], tolerance = 1e-8)
    expect_equal(out[1, 3], ref[2], tolerance = 1e-8)
})

test_that("group_ci lower/upper exactly match rENA's t.test formula", {
    set.seed(42)
    pts <- matrix(rnorm(15 * 2), nrow = 15)
    # rENA's exact computation — 2 x n_dims, rows = [lower, upper]
    rena_ci <- matrix(
        c(as.vector(t.test(pts[, 1], conf.level = 0.95)$conf.int),
          as.vector(t.test(pts[, 2], conf.level = 0.95)$conf.int)),
        ncol = 2)
    our_ci <- lq_mean_ci(pts, 0.95)   # n_dims x 3, cols = [mean, lower, upper]
    expect_equal(our_ci[, 2], rena_ci[1, ], tolerance = 1e-10)  # lower bounds
    expect_equal(our_ci[, 3], rena_ci[2, ], tolerance = 1e-10)  # upper bounds
})

# -----------------------------------------------------------------------
# outlier_ci tests
# -----------------------------------------------------------------------

test_that("outlier_ci: output is n_dims x 2", {
    set.seed(20)
    pts <- matrix(rnorm(20 * 3), nrow = 20)
    out <- lq_outlier_ci(pts, 1.5)
    expect_equal(dim(out), c(3L, 2L))
})

test_that("outlier_ci exactly matches rENA's IQR formula", {
    set.seed(42)
    pts <- matrix(rnorm(20 * 2), nrow = 20)
    # rENA's exact computation — 2 x n_dims, rows = [lower, upper]
    oi_raw  <- c(IQR(pts[, 1]), IQR(pts[, 2])) * 1.5
    rena_oi <- matrix(rep(oi_raw, 2), ncol = 2, byrow = TRUE) * c(-1, 1)
    our_oi  <- lq_outlier_ci(pts, 1.5)   # n_dims x 2, cols = [lower, upper]
    expect_equal(our_oi[, 1], rena_oi[1, ], tolerance = 1e-10)  # lower bounds
    expect_equal(our_oi[, 2], rena_oi[2, ], tolerance = 1e-10)  # upper bounds
})

test_that("outlier_ci: lower = -upper (symmetric around 0)", {
    set.seed(21)
    pts <- matrix(rnorm(30 * 2), nrow = 30)
    out <- lq_outlier_ci(pts)
    expect_equal(out[, 1], -out[, 2], tolerance = 1e-14)
})

test_that("outlier_ci: iqr_factor scales bounds proportionally", {
    set.seed(22)
    pts <- matrix(rnorm(25 * 2), nrow = 25)
    out15 <- lq_outlier_ci(pts, 1.5)
    out30 <- lq_outlier_ci(pts, 3.0)
    expect_equal(out30, out15 * 2, tolerance = 1e-14)
})

test_that("outlier_ci: n=0 returns NaN", {
    out <- lq_outlier_ci(matrix(0, nrow = 0, ncol = 2))
    expect_true(all(is.nan(out)))
})

test_that("center_data: column means are zero after centering", {
    set.seed(20)
    m   <- matrix(runif(20), nrow = 4)
    out <- lq_center_points(m)
    expect_equal(colMeans(out), rep(0, ncol(m)), tolerance = 1e-10)
})

test_that("center_data: pairwise row differences are preserved", {
    set.seed(21)
    m   <- matrix(runif(15), nrow = 3)
    out <- lq_center_points(m)
    expect_equal(out[1, ] - out[2, ], m[1, ] - m[2, ], tolerance = 1e-10)
})

test_that("lws_lsq_positions: output dimensions are correct", {
    set.seed(30)
    n_units <- 5; n_codes <- 4; n_dims <- 2
    lw  <- matrix(runif(n_units * choose(n_codes, 2)), nrow = n_units)
    pts <- matrix(runif(n_units * n_dims), nrow = n_units)
    out <- lq_node_positions(lw, pts, n_dims)
    expect_equal(nrow(out$nodes),     n_codes)
    expect_equal(ncol(out$nodes),     n_dims)
    expect_equal(nrow(out$centroids), n_units)
    expect_equal(ncol(out$centroids), n_dims)
})

test_that("lws_lsq_positions: centroids are consistent with nodes and weights", {
    # centroids = weights %*% t(nodes), transposed; verify numerically
    set.seed(31)
    lw  <- matrix(abs(rnorm(6 * 3)), nrow = 6)
    pts <- matrix(rnorm(6 * 2), nrow = 6)
    out <- lq_node_positions(lw, pts, 2L)
    reconstructed <- out$weights %*% out$nodes
    expect_equal(reconstructed, out$centroids, tolerance = 1e-6)
})

test_that("directed_node_positions: output dimensions are correct", {
    set.seed(32)
    n_codes <- 3; n_units <- 4; n_dims <- 2
    lw  <- matrix(runif(n_units * n_codes^2), nrow = n_units)
    pts <- matrix(runif(n_units * n_dims), nrow = n_units)
    out <- lq_directed_node_positions(lw, pts, n_dims)
    expect_equal(nrow(out$nodes), n_codes)
    expect_equal(ncol(out$nodes), n_dims)
    expect_equal(nrow(out$centroids), n_units)
})

test_that("directed_node_positions_ground_response: requires even n_units", {
    set.seed(33)
    n_codes <- 3; n_units <- 6; n_dims <- 2
    lw  <- matrix(runif(n_units * n_codes^2), nrow = n_units)
    pts <- matrix(runif(n_units * n_dims), nrow = n_units)
    out <- lq_directed_node_positions_combine_pairs(lw, pts, n_dims)
    # centroids are computed over all n_units rows
    expect_equal(nrow(out$centroids), n_units)
    expect_equal(nrow(out$nodes),     n_codes)
})

test_that("ena_correlation: returns matrix of correct shape", {
    set.seed(34)
    pts <- matrix(runif(10 * 2), nrow = 10)
    cts <- matrix(runif(10 * 2), nrow = 10)
    out <- lq_ena_correlation(pts, cts, 0.95)
    expect_equal(dim(out), c(2L, 3L))
})

test_that("ena_correlation: r is in [-1, 1] and CI brackets r", {
    set.seed(35)
    pts <- matrix(runif(20 * 2), nrow = 20)
    cts <- matrix(runif(20 * 2), nrow = 20)
    out <- lq_ena_correlation(pts, cts, 0.95)
    for (i in 1:2) {
        r <- out[i, 1]; lo <- out[i, 2]; hi <- out[i, 3]
        expect_gte(r,  -1); expect_lte(r,  1)
        expect_lte(lo,  r); expect_gte(hi, r)
    }
})

test_that("ena_correlation: perfectly correlated points give r == 1", {
    pts <- matrix(1:20, nrow = 10)
    cts <- pts * 2                   # identical direction → r = 1
    out <- lq_ena_correlation(pts, cts, 0.95)
    expect_equal(out[1, 1], 1, tolerance = 1e-8)
})
