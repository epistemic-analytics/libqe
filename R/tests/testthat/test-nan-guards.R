library(libqe)

set.seed(42)
clean_adj    <- matrix(runif(30), nrow = 5, ncol = 6)
clean_points <- matrix(runif(10), nrow = 5, ncol = 2)

nan_adj    <- clean_adj;    nan_adj[1, 1]    <- NaN
nan_points <- clean_points; nan_points[2, 1] <- NaN
inf_points <- clean_points; inf_points[3, 2] <- Inf

# ── normalize_networks / scale_networks: NaN rows should become zeros, no error

test_that("normalize_networks with NaN row produces zeros, not NaN", {
  m      <- matrix(c(NaN, NaN, 1, 0), nrow = 2, byrow = TRUE)
  result <- normalize_networks(m)
  expect_false(any(is.nan(result)))
  expect_equal(result[2, 1], 1.0)
})

test_that("scale_networks NaN row stays NaN, finite rows correctly scaled", {
  # scale_networks divides m in-place; NaN values in the NaN row remain NaN.
  # The guarantee is that the NaN row norm does NOT corrupt the scale factor.
  m      <- matrix(c(NaN, NaN, 3, 4), nrow = 2, byrow = TRUE)
  result <- scale_networks(m)
  expect_true(all(is.nan(result[1, ])))
  expect_equal(result[2, 1], 3 / 5, tolerance = 1e-10)
  expect_equal(result[2, 2], 4 / 5, tolerance = 1e-10)
})

# ── node_positions

test_that("node_positions errors on NaN in adj_mats", {
  expect_error(node_positions(nan_adj, clean_points, 2L),
               regexp = "NaN or Inf")
})

test_that("node_positions errors on Inf in points", {
  expect_error(node_positions(clean_adj, inf_points, 2L),
               regexp = "NaN or Inf")
})

test_that("node_positions succeeds on clean inputs", {
  r <- node_positions(clean_adj, clean_points, 2L)
  expect_equal(ncol(r$nodes), 2L)
  expect_true(all(is.finite(r$nodes)))
})

# ── directed_node_positions

test_that("directed_node_positions errors on NaN in line_weights", {
  lw <- matrix(runif(20), nrow = 5, ncol = 4)
  lw[1, 1] <- NaN
  expect_error(directed_node_positions(lw, clean_points, 2L),
               regexp = "NaN or Inf")
})

test_that("directed_node_positions errors on Inf in points", {
  lw <- matrix(runif(20), nrow = 5, ncol = 4)
  expect_error(directed_node_positions(lw, inf_points, 2L),
               regexp = "NaN or Inf")
})

# ── ena_svd

test_that("ena_svd errors on NaN input", {
  expect_error(ena_svd(nan_points), regexp = "NaN or Inf")
})

test_that("ena_svd errors on Inf input", {
  expect_error(ena_svd(inf_points), regexp = "NaN or Inf")
})

test_that("ena_svd succeeds on clean input", {
  r <- ena_svd(clean_points)
  expect_equal(ncol(r$rotation), 2L)
  expect_true(all(is.finite(r$rotation)))
})
