test_that("lq_center_data matches rENA center_data_c", {
    skip_if_not_installed("rENA")
    set.seed(20)
    m <- matrix(runif(15), nrow = 3)
    expect_equal(lq_center_data(m), rENA:::center_data_c(m), tolerance = 1e-10)
})

test_that("lq_center_data: column means are zero after centering", {
    set.seed(21)
    m   <- matrix(runif(20), nrow = 4)
    out <- lq_center_data(m)
    expect_equal(colMeans(out), rep(0, ncol(m)), tolerance = 1e-10)
})

test_that("lq_lws_lsq_positions matches rENA lws_lsq_positions", {
    skip_if_not_installed("rENA")
    inp <- make_lws_inputs()
    lq  <- lq_lws_lsq_positions(inp$line_weights, inp$points, 2L)
    ref <- rENA:::lws_lsq_positions(inp$line_weights, inp$points, 2L)
    expect_equal(lq$nodes,     ref$nodes,     tolerance = 1e-8)
    expect_equal(lq$centroids, ref$centroids, tolerance = 1e-8)
    expect_equal(lq$weights,   ref$weights,   tolerance = 1e-8)
})

test_that("lq_lws_lsq_positions: nodes and centroids have correct dimensions", {
    set.seed(30)
    n_units <- 5; n_codes <- 4; n_dims <- 2
    lw  <- matrix(runif(n_units * choose(n_codes, 2)), nrow = n_units)
    pts <- matrix(runif(n_units * n_dims), nrow = n_units)
    out <- lq_lws_lsq_positions(lw, pts, n_dims)
    expect_equal(nrow(out$nodes),     n_codes)
    expect_equal(ncol(out$nodes),     n_dims)
    expect_equal(nrow(out$centroids), n_units)
    expect_equal(ncol(out$centroids), n_dims)
})

test_that("lq_directed_node_positions matches rENA directed_node_positions", {
    skip_if_not_installed("rENA")
    set.seed(31)
    n_codes <- 3; n_units <- 4; n_dims <- 2
    lw  <- matrix(runif(n_units * n_codes^2), nrow = n_units)
    pts <- matrix(runif(n_units * n_dims), nrow = n_units)
    lq  <- lq_directed_node_positions(lw, pts, n_dims)
    ref <- rENA:::directed_node_positions(lw, pts, n_dims)
    expect_equal(lq$nodes,     ref$nodes,     tolerance = 1e-8)
    expect_equal(lq$centroids, ref$centroids, tolerance = 1e-8)
})

test_that("lq_directed_node_positions_ground_response matches rENA", {
    skip_if_not_installed("rENA")
    set.seed(32)
    n_codes <- 3; n_units <- 6; n_dims <- 2  # n_units must be even
    lw  <- matrix(runif(n_units * n_codes^2), nrow = n_units)
    pts <- matrix(runif(n_units * n_dims), nrow = n_units)
    lq  <- lq_directed_node_positions_ground_response(lw, pts, n_dims)
    ref <- rENA:::directed_node_positions_with_ground_response_added(lw, pts, n_dims)
    expect_equal(lq$nodes,     ref$nodes,     tolerance = 1e-8)
    expect_equal(lq$centroids, ref$centroids, tolerance = 1e-8)
})

test_that("lq_ena_correlation matches rENA ena_correlation", {
    skip_if_not_installed("rENA")
    set.seed(33)
    pts <- matrix(runif(10 * 2), nrow = 10)
    cts <- matrix(runif(10 * 2), nrow = 10)
    lq  <- lq_ena_correlation(pts, cts, 0.95)
    ref <- rENA:::ena_correlation(pts, cts, 0.95)
    expect_equal(lq, ref, tolerance = 1e-6)
})
