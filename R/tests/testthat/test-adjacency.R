test_that("tri_indices: correct number of pairs", {
    for (n in c(3, 5, 6, 10)) {
        idx <- connection_indices(n)
        expect_equal(ncol(idx), n * (n - 1L) / 2L)
        expect_equal(nrow(idx), 2L)
    }
})

test_that("tri_indices: all pairs are strict upper-triangle (row < col)", {
    for (n in c(3, 6)) {
        idx <- connection_indices(n)
        expect_true(all(idx[1, ] < idx[2, ]))
    }
})

test_that("tri_indices: row=0 and row=1 return correct single rows", {
    idx  <- connection_indices(4L)
    expect_equal(connection_indices(4L, 0L), idx[1L, , drop = FALSE])
    expect_equal(connection_indices(4L, 1L), idx[2L, , drop = FALSE])
})

test_that("vector_to_upper_tri: pairwise products match manual calculation", {
    v   <- c(2, 3, 5)   # pairs: (2,3)=6, (2,5)=10, (3,5)=15
    out <- as.vector(code_connections(matrix(v, nrow = 1)))
    expect_equal(out, c(6, 10, 15))
})

test_that("vector_to_upper_tri: zero vector produces zero output", {
    v   <- c(0, 0, 0, 0)
    out <- code_connections(matrix(v, nrow = 1))
    expect_true(all(out == 0))
})

test_that("directed_to_upper_tri: symmetric matrix folds correctly", {
    # For a symmetric 3x3 matrix M, M + M^T = 2M, so upper-tri should be
    # twice the upper-triangle values of M.
    m   <- matrix(c(1,2,3, 2,4,5, 3,5,6), 3, 3)
    v   <- as.vector(m)
    out <- as.vector(fold_directed_network(v))
    # upper-tri of (m + t(m)): positions (1,2),(1,3),(2,3) → 4,6,10
    expect_equal(out, c(4, 6, 10))
})

test_that("directed_to_upper_tri: length is choose_two(n)", {
    for (n in c(3, 4, 6)) {
        v   <- runif(n * n)
        out <- fold_directed_network(v)
        expect_length(out, n * (n - 1L) / 2L)
    }
})

test_that("adjacency_matrix_to_vector: full=TRUE returns column-major vector", {
    m   <- matrix(1:9, 3, 3)
    out <- as.vector(network_to_vector(m, TRUE))
    expect_equal(out, as.vector(m))
})

test_that("adjacency_matrix_to_vector: full=FALSE returns upper-triangle", {
    # Values must be in the upper triangle (row < col) for full=FALSE to pick them up
    m   <- matrix(c(0, 1, 2,
                    0, 0, 3,
                    0, 0, 0), 3, 3, byrow = TRUE)
    out <- as.vector(network_to_vector(m, FALSE))
    expect_equal(out, c(1, 2, 3))
})

test_that("svector_to_upper_tri: produces correct 'A & B' pair names", {
    codes <- c("X", "Y", "Z")
    out   <- connection_names(codes)
    expect_equal(out, c("X & Y", "X & Z", "Y & Z"))
})

test_that("svector_to_upper_tri: length matches choose_two(n)", {
    codes <- c("A", "B", "C", "D")
    expect_length(connection_names(codes), 6L)
})
