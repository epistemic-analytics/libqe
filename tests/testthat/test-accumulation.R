test_that("stanza_window: window_back=1 produces only same-row self-products", {
    # With window_back=1 the stanza for each row k is just row k itself.
    # co-occurrence of a row with itself: v[j]*v[i] for all j<i.
    codes <- matrix(c(1, 1, 0,   # pairs: c1&c2=1, c1&c3=0, c2&c3=0
                      1, 0, 1,   # pairs: 1,0,0  ... but binary → same
                      0, 1, 1,
                      1, 1, 1),
                    nrow = 4, byrow = TRUE)
    out <- lq_stanza_window(codes, 1L, 0L, TRUE)
    expect_equal(dim(out), c(4L, 3L))
    # Row 1: codes (1,1,0) → pairs (1,0,0)
    expect_equal(as.vector(out[1, ]), c(1, 0, 0))
    # Row 4: codes (1,1,1) → all pairs present
    expect_equal(as.vector(out[4, ]), c(1, 1, 1))
})

test_that("stanza_window: binary=FALSE preserves continuous co-occurrence counts", {
    codes <- matrix(c(2, 3, 0,
                      1, 1, 1), nrow = 2, byrow = TRUE)
    out <- lq_stanza_window(codes, 1L, 0L, FALSE)
    # Row 1: 2*3=6, 2*0=0, 3*0=0
    expect_equal(as.vector(out[1, ]), c(6, 0, 0))
})

test_that("stanza_window: wider window accumulates across prior rows", {
    codes <- matrix(c(1, 0, 0,
                      0, 1, 0,
                      0, 0, 1), nrow = 3, byrow = TRUE)
    # window_back=3: all prior rows included for row 3
    out <- lq_stanza_window(codes, 3L, 0L, FALSE)
    # Row 3 (window covers all 3 rows; back-reference correction removes rows 1+2
    # from the reference side): co-occurrence between (0,0,1) response and
    # remaining context (0,1,0) gives c2&c3=1; c1 pairs are zero.
    expect_equal(as.vector(out[3, ]), c(0, 1, 1))
    # Row 1 stanza = just row 1 = (1,0,0), all pairs zero (only one code active)
    expect_equal(as.vector(out[1, ]), c(0, 0, 0))
})

test_that("stanza_window: all-zero codes produce all-zero output", {
    codes <- matrix(0, nrow = 5, ncol = 4)
    out   <- lq_stanza_window(codes, 2L, 0L, TRUE)
    expect_true(all(out == 0))
})

test_that("calculate_adjacency_matrix: ordered=TRUE has zero diagonal contribution", {
    g  <- matrix(c(1, 0, 1, 0), nrow = 1)
    r  <- matrix(c(0, 1, 0, 1), nrow = 1)
    m  <- lq_calculate_adjacency_matrix(g, r, 1.0, TRUE)
    # r⊗r diagonal should be zeroed; off-diagonal self-connections allowed
    expect_equal(diag(m), rep(0, 4), tolerance = 1e-10)
})

test_that("calculate_adjacency_matrix: zero ground gives only self-connection", {
    g  <- matrix(rep(0, 4), nrow = 1)
    r  <- matrix(c(1, 0, 0, 0), nrow = 1)
    m  <- lq_calculate_adjacency_matrix(g, r, 1.0, TRUE)
    # g⊗r = 0; self-connection matrix has diagonal zeroed → all zero for unit r
    expect_equal(sum(abs(m)), 0, tolerance = 1e-10)
})

test_that("accumulate_unit: all-zero codes produce all-zero output", {
    codes     <- matrix(0, nrow = 5, ncol = 3)
    unit_rows <- c(1L, 3L)   # 0-based
    decay_fn  <- function(d) rep(1, length(d))
    out <- lq_accumulate_unit(codes, unit_rows, decay_fn, FALSE)
    expect_true(all(out == 0))
})

test_that("accumulate_unit: output length is choose_two(n_codes) when unordered", {
    set.seed(40)
    codes     <- matrix(sample(0:1, 20, replace = TRUE), nrow = 5, ncol = 4)
    unit_rows <- c(1L, 3L, 4L)
    decay_fn  <- function(d) as.numeric(d < 2)
    out <- lq_accumulate_unit(codes, unit_rows, decay_fn, FALSE)
    expect_length(out, choose(4, 2))
})

test_that("accumulate_unit: output length is n_codes^2 when ordered", {
    set.seed(41)
    codes     <- matrix(sample(0:1, 15, replace = TRUE), nrow = 5, ncol = 3)
    unit_rows <- c(0L, 2L, 4L)
    decay_fn  <- function(d) as.numeric(d < 3)
    out <- lq_accumulate_unit(codes, unit_rows, decay_fn, TRUE)
    expect_length(out, 3^2)
})
