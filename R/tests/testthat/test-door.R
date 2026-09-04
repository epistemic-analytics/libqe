test_that("door_lookback_block computes sliding window sums and means", {
  mat <- matrix(c(
    1, 0,
    2, 1,
    3, 0,
    4, 2,
    5, 1
  ), nrow = 5, byrow = TRUE)

  # Lookback size 2, sum, equal weight
  res_sum2 <- door_lookback_block(mat, lookback_size = 2L, aggregate_mean = FALSE, weighting_linear = FALSE)
  expect_equal(nrow(res_sum2), 5)
  expect_equal(ncol(res_sum2), 2)
  expect_equal(res_sum2[1, ], c(1, 0))
  expect_equal(res_sum2[2, ], c(3, 1))  # row 1 + row 2
  expect_equal(res_sum2[3, ], c(5, 1))  # row 2 + row 3
  expect_equal(res_sum2[4, ], c(7, 2))  # row 3 + row 4
  expect_equal(res_sum2[5, ], c(9, 3))  # row 4 + row 5

  # Lookback size 2, mean, equal weight
  res_mean2 <- door_lookback_block(mat, lookback_size = 2L, aggregate_mean = TRUE, weighting_linear = FALSE)
  expect_equal(res_mean2[1, ], c(1, 0))
  expect_equal(res_mean2[2, ], c(1.5, 0.5))
  expect_equal(res_mean2[3, ], c(2.5, 0.5))
})

test_that("door_lookback_block respects segment boundaries", {
  mat <- matrix(c(
    1, 0,
    2, 0,
    10, 5,
    20, 5
  ), nrow = 4, byrow = TRUE)
  segs <- c(1L, 1L, 2L, 2L)

  res <- door_lookback_block(mat, lookback_size = 3L, aggregate_mean = FALSE, segment_ids = segs)
  # Row 3 should NOT pool from row 2 because segment changed from 1 to 2
  expect_equal(res[1, ], c(1, 0))
  expect_equal(res[2, ], c(3, 0))
  expect_equal(res[3, ], c(10, 5))
  expect_equal(res[4, ], c(30, 10))
})

test_that("door_lookback_block ignores NA values like ETM reference", {
  mat <- matrix(c(
    1, 1,
    NA, 3,
    5, NA
  ), nrow = 3, byrow = TRUE)

  res_sum <- door_lookback_block(mat, lookback_size = 3L, aggregate_mean = FALSE)
  expect_equal(res_sum[1, ], c(1, 1))
  expect_equal(res_sum[2, ], c(1, 4))
  expect_equal(res_sum[3, ], c(6, 4))

  res_mean <- door_lookback_block(mat, lookback_size = 3L, aggregate_mean = TRUE)
  expect_equal(res_mean[1, ], c(1, 1))
  expect_equal(res_mean[2, ], c(1, 2))
  expect_equal(res_mean[3, ], c(3, 2))
})

test_that("door_ema_block smooths values with alpha", {
  mat <- matrix(c(
    10, 0,
    0,  10,
    0,  0
  ), nrow = 3, byrow = TRUE)

  res <- door_ema_block(mat, alpha = 0.5)
  expect_equal(res[1, ], c(10, 0))
  expect_equal(res[2, ], c(5, 5))     # 0.5 * 0 + 0.5 * 10 = 5 for col 1; 0.5 * 10 + 0.5 * 0 = 5 for col 2
  expect_equal(res[3, ], c(2.5, 2.5))
})

test_that("door_ema_block carries previous values when current values are NA", {
  mat <- matrix(c(
    10, 0,
    NA, 10,
    0, NA
  ), nrow = 3, byrow = TRUE)

  res <- door_ema_block(mat, alpha = 0.5)
  expect_equal(res[1, ], c(10, 0))
  expect_equal(res[2, ], c(10, 5))
  expect_equal(res[3, ], c(5, 5))
})
