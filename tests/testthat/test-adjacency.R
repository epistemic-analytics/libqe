test_that("lq_tri_indices matches rENA triIndices", {
    skip_if_not_installed("rENA")
    for (n in c(3, 5, 6, 10)) {
        expect_equal(lq_tri_indices(n),      rENA:::triIndices(n))
        expect_equal(lq_tri_indices(n, 0L),  rENA:::triIndices(n, 0L))
        expect_equal(lq_tri_indices(n, 1L),  rENA:::triIndices(n, 1L))
    }
})

test_that("lq_vector_to_upper_tri matches rENA vector_to_ut", {
    skip_if_not_installed("rENA")
    set.seed(1)
    for (n in c(3, 6)) {
        v <- matrix(runif(n), nrow = 1)
        expect_equal(lq_vector_to_upper_tri(v), rENA:::vector_to_ut(v))
    }
})

test_that("lq_directed_to_upper_tri: folding A+A^T upper-tri is correct", {
    # vector_to_summed_uppertri is an internal C++ function in tma, not
    # callable from R.  Verify the math directly: for a known directed matrix,
    # folding should equal trimatu(M + M^T) read off by column.
    set.seed(2)
    for (n in c(3, 4)) {
        m  <- matrix(runif(n * n), n, n)
        v  <- as.vector(m)                          # column-major, matches arma
        lq <- as.vector(lq_directed_to_upper_tri(v))

        sym      <- m + t(m)
        expected <- sym[upper.tri(sym)]             # row-major upper tri
        expect_equal(lq, expected, tolerance = 1e-10)
    }
})

test_that("lq_adjacency_matrix_to_vector matches tma adjacency_matrix_to_vector", {
    skip_if_not_installed("tma")
    set.seed(3)
    m <- matrix(runif(16), 4, 4)
    expect_equal(lq_adjacency_matrix_to_vector(m, TRUE),
                 tma:::adjacency_matrix_to_vector(m, TRUE))
    expect_equal(lq_adjacency_matrix_to_vector(m, FALSE),
                 tma:::adjacency_matrix_to_vector(m, FALSE))
})

test_that("lq_svector_to_upper_tri matches rENA svector_to_ut", {
    skip_if_not_installed("rENA")
    codes <- c("Data", "Collaboration", "Design")
    expect_equal(lq_svector_to_upper_tri(codes), rENA:::svector_to_ut(codes))
})

test_that("choose_two: output dimensions are correct", {
    for (n in c(3, 6, 10)) {
        idx <- lq_tri_indices(n)
        expect_equal(ncol(idx), n * (n - 1) / 2)
    }
})
