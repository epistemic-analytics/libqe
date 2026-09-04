test_that("fit_trajectory_poly fits linear and quadratic trajectories", {
  # Linear trajectory: x(t) = 2*t + 1, y(t) = -3*t + 4
  t <- seq(0, 1, length.out = 10)
  x <- 2 * t + 1
  y <- -3 * t + 4
  pts <- cbind(x, y)

  fit <- fit_trajectory_poly(pts, t = t, max_degree = 3L)
  expect_true(fit$degree >= 1L)
  expect_equal(fit$coeffs_x[1], 1.0, tolerance = 1e-6)
  expect_equal(fit$coeffs_x[2], 2.0, tolerance = 1e-6)
  expect_equal(fit$coeffs_y[1], 4.0, tolerance = 1e-6)
  expect_equal(fit$coeffs_y[2], -3.0, tolerance = 1e-6)

  # Curve evaluation
  eval_pts <- eval_trajectory_curve(fit$coeffs_x, fit$coeffs_y, c(0.0, 0.5, 1.0))
  expect_equal(eval_pts[1, ], c(1.0, 4.0), tolerance = 1e-6)
  expect_equal(eval_pts[2, ], c(2.0, 2.5), tolerance = 1e-6)
  expect_equal(eval_pts[3, ], c(3.0, 1.0), tolerance = 1e-6)
})

test_that("fit_trajectory_poly default orthogonal basis matches stats::poly predictions", {
  t <- seq(0, 1, length.out = 9)
  x <- c(-0.32, -0.18, -0.05, 0.11, 0.24, 0.28, 0.18, 0.02, -0.11)
  y <- c(0.10, 0.03, -0.06, -0.12, -0.05, 0.08, 0.18, 0.16, 0.05)
  pts <- cbind(x, y)

  fit <- fit_trajectory_poly(
    pts,
    t = t,
    max_degree = 3L,
    fixed_degree = 3L,
    criterion = "loocv"
  )

  rx <- stats::lm(x ~ stats::poly(t, 3))
  ry <- stats::lm(y ~ stats::poly(t, 3))
  expected_fit <- unname(cbind(stats::fitted(rx), stats::fitted(ry)))

  expect_equal(fit$basis, "orthogonal")
  expect_equal(unname(fit$fitted_points), expected_fit, tolerance = 1e-10)

  t_eval <- seq(0, 1, length.out = 17)
  got <- eval_trajectory_curve(fit$coeffs_x, fit$coeffs_y, t_eval)
  expected <- unname(cbind(
    stats::predict(rx, newdata = data.frame(t = t_eval)),
    stats::predict(ry, newdata = data.frame(t = t_eval))
  ))
  expect_equal(unname(got), expected, tolerance = 1e-10)
})

test_that("fit_trajectory_poly can still use raw polynomial basis", {
  t <- seq(0, 1, length.out = 6)
  pts <- cbind(1 + 2 * t, 4 - 3 * t)

  fit <- fit_trajectory_poly(pts, t = t, fixed_degree = 1L, basis = "raw")

  expect_equal(fit$basis, "raw")
  expect_length(fit$basis_alpha, 0)
  expect_length(fit$basis_norm2, 0)
  expect_equal(as.numeric(fit$coeffs_x), c(1, 2), tolerance = 1e-10)
  expect_equal(as.numeric(fit$coeffs_y), c(4, -3), tolerance = 1e-10)
})

test_that("eval_trajectory_derivatives computes speed and curvature", {
  # Circular trajectory x(t) = cos(2*pi*t), y(t) = sin(2*pi*t)
  # Approx with polynomial or simple parabolic arc x(t) = t, y(t) = t^2
  coeffs_x <- c(0, 1)    # x(t) = t
  coeffs_y <- c(0, 0, 1) # y(t) = t^2

  t_eval <- c(0.0, 0.5, 1.0)
  derivs <- eval_trajectory_derivatives(coeffs_x, coeffs_y, t_eval)

  expect_equal(length(derivs$speed), 3)
  # At t=0: vx = 1, vy = 0 => speed = 1, ax = 0, ay = 2 => omega = 1*2 - 0*0 = 2 => curvature = 2 / 1^3 = 2
  expect_equal(derivs$speed[1], 1.0, tolerance = 1e-6)
  expect_equal(derivs$heading_rate[1], 2.0, tolerance = 1e-6)
  expect_equal(derivs$curvature[1], 2.0, tolerance = 1e-6)
})

test_that("integrated_trajectory_distance computes distance between curves", {
  # Identical curves => distance 0
  cx <- c(0, 1)
  cy <- c(0, 2)
  d_same <- integrated_trajectory_distance(cx, cy, cx, cy, 0.0, 1.0)
  expect_equal(d_same, 0.0, tolerance = 1e-6)

  # Parallel curves offset by dx = 3, dy = 4 => constant distance = 5
  cx_b <- c(3, 1)
  cy_b <- c(4, 2)
  d_parallel <- integrated_trajectory_distance(cx, cy, cx_b, cy_b, 0.0, 1.0)
  expect_equal(d_parallel, 5.0, tolerance = 1e-6)
})

test_that("sweep_signed_turn_lags detects true follower lag", {
  # Agent B speaks at times 1, 2, 3, 4, 5 with positions (1,1), (2,2), (3,3), (4,4), (5,5)
  # Agent A echoes B's position with a 2-turn delay: at times 3, 4, 5, 6, 7
  times_b <- c(1, 2, 3, 4, 5)
  pts_b <- cbind(c(1, 2, 3, 4, 5), c(1, 2, 3, 4, 5))

  times_a <- c(3, 4, 5, 6, 7)
  pts_a <- cbind(c(1, 2, 3, 4, 5), c(1, 2, 3, 4, 5)) # identical positions shifted by 2 turns

  res <- sweep_signed_turn_lags(pts_a, pts_b, times_a, times_b, max_lag = 4L)
  expect_equal(res$best_lag, 2L)
  expect_equal(res$min_mean_distance, 0.0, tolerance = 1e-6)
})

test_that("dist_dist_correlation calculates matrix neighborhood stability", {
  set.seed(42)
  X <- matrix(rnorm(20), ncol = 2)
  # Y is rotated and scaled version of X => distance correlation should be 1.0
  theta <- pi / 4
  R <- matrix(c(cos(theta), -sin(theta), sin(theta), cos(theta)), 2, 2)
  Y <- X %*% R * 2.5

  r_val <- dist_dist_correlation(X, Y)
  expect_equal(r_val, 1.0, tolerance = 1e-6)
})
