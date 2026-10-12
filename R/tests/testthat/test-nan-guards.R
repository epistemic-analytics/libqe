library(libqe)

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
