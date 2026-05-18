test_that("lq_sphere_norm matches rENA fun_sphere_norm", {
    skip_if_not_installed("rENA")
    set.seed(10)
    m <- as.data.frame(matrix(runif(20), nrow = 4))
    expect_equal(lq_sphere_norm(as.matrix(m)), rENA:::fun_sphere_norm(m),
                 tolerance = 1e-10)
})

test_that("lq_sphere_norm: zero rows stay zero", {
    m <- matrix(c(0, 0, 0, 1, 2, 3), nrow = 2, byrow = TRUE)
    out <- lq_sphere_norm(m)
    expect_equal(out[1, ], c(0, 0, 0))
    expect_equal(sum(out[2, ]^2), 1, tolerance = 1e-10)
})

test_that("lq_skip_sphere_norm matches rENA fun_skip_sphere_norm", {
    skip_if_not_installed("rENA")
    set.seed(11)
    m <- as.data.frame(matrix(runif(20), nrow = 4))
    expect_equal(lq_skip_sphere_norm(as.matrix(m)),
                 rENA:::fun_skip_sphere_norm(m),
                 tolerance = 1e-10)
})

test_that("lq_skip_sphere_norm: largest row has L2 norm == 1", {
    set.seed(12)
    m <- matrix(runif(12), nrow = 3)
    out <- lq_skip_sphere_norm(m)
    norms <- apply(out, 1, function(r) sqrt(sum(r^2)))
    expect_lte(max(norms), 1 + 1e-10)
    expect_equal(max(norms), 1, tolerance = 1e-10)
})
