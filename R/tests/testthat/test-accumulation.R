test_that("stanza_window: window_back=1 produces only same-row self-products", {
    # With window_back=1 the stanza for each row k is just row k itself.
    # co-occurrence of a row with itself: v[j]*v[i] for all j<i.
    codes <- matrix(c(1, 1, 0,   # pairs: c1&c2=1, c1&c3=0, c2&c3=0
                      1, 0, 1,   # pairs: 1,0,0  ... but binary → same
                      0, 1, 1,
                      1, 1, 1),
                    nrow = 4, byrow = TRUE)
    out <- accumulate_stanza(codes, 1L, 0L, TRUE)
    expect_equal(dim(out), c(4L, 3L))
    # Row 1: codes (1,1,0) → pairs (1,0,0)
    expect_equal(as.vector(out[1, ]), c(1, 0, 0))
    # Row 4: codes (1,1,1) → all pairs present
    expect_equal(as.vector(out[4, ]), c(1, 1, 1))
})

test_that("stanza_window: binary=FALSE preserves continuous co-occurrence counts", {
    codes <- matrix(c(2, 3, 0,
                      1, 1, 1), nrow = 2, byrow = TRUE)
    out <- accumulate_stanza(codes, 1L, 0L, FALSE)
    # Row 1: 2*3=6, 2*0=0, 3*0=0
    expect_equal(as.vector(out[1, ]), c(6, 0, 0))
})

test_that("stanza_window: wider window accumulates across prior rows", {
    codes <- matrix(c(1, 0, 0,
                      0, 1, 0,
                      0, 0, 1), nrow = 3, byrow = TRUE)
    # window_back=3: all prior rows included for row 3
    out <- accumulate_stanza(codes, 3L, 0L, FALSE)
    # Row 3 (window covers all 3 rows; back-reference correction removes rows 1+2
    # from the reference side): co-occurrence between (0,0,1) response and
    # remaining context (0,1,0) gives c2&c3=1; c1 pairs are zero.
    expect_equal(as.vector(out[3, ]), c(0, 1, 1))
    # Row 1 stanza = just row 1 = (1,0,0), all pairs zero (only one code active)
    expect_equal(as.vector(out[1, ]), c(0, 0, 0))
})

test_that("stanza_window: all-zero codes produce all-zero output", {
    codes <- matrix(0, nrow = 5, ncol = 4)
    out   <- accumulate_stanza(codes, 2L, 0L, TRUE)
    expect_true(all(out == 0))
})

test_that("calculate_adjacency_matrix: ordered=TRUE has zero diagonal contribution", {
    g  <- matrix(c(1, 0, 1, 0), nrow = 1)
    r  <- matrix(c(0, 1, 0, 1), nrow = 1)
    m  <- connection_matrix(g, r, 1.0, TRUE)
    # r⊗r diagonal should be zeroed; off-diagonal self-connections allowed
    expect_equal(diag(m), rep(0, 4), tolerance = 1e-10)
})

test_that("calculate_adjacency_matrix: zero ground gives only self-connection", {
    g  <- matrix(rep(0, 4), nrow = 1)
    r  <- matrix(c(1, 0, 0, 0), nrow = 1)
    m  <- connection_matrix(g, r, 1.0, TRUE)
    # g⊗r = 0; self-connection matrix has diagonal zeroed → all zero for unit r
    expect_equal(sum(abs(m)), 0, tolerance = 1e-10)
})

test_that("accumulate_unit: all-zero codes produce all-zero output", {
    codes     <- matrix(0, nrow = 5, ncol = 3)
    unit_rows <- c(1L, 3L)   # 0-based
    decay_fn  <- function(d) rep(1, length(d))
    out <- accumulate_unit(codes, unit_rows, decay_fn, FALSE)
    expect_true(all(out == 0))
})

test_that("accumulate_unit: output length is choose_two(n_codes) when unordered", {
    set.seed(40)
    codes     <- matrix(sample(0:1, 20, replace = TRUE), nrow = 5, ncol = 4)
    unit_rows <- c(1L, 3L, 4L)
    decay_fn  <- function(d) as.numeric(d < 2)
    out <- accumulate_unit(codes, unit_rows, decay_fn, FALSE)
    expect_length(out, choose(4, 2))
})

test_that("accumulate_unit: output length is n_codes^2 when ordered", {
    set.seed(41)
    codes     <- matrix(sample(0:1, 15, replace = TRUE), nrow = 5, ncol = 3)
    unit_rows <- c(0L, 2L, 4L)
    decay_fn  <- function(d) as.numeric(d < 3)
    out <- accumulate_unit(codes, unit_rows, decay_fn, TRUE)
    expect_length(out, 3^2)
})

# --- rows_to_co_occurrences ---

test_that("rows_to_co_occurrences: output dimensions are n_rows x choose_two(n_codes)", {
    codes <- matrix(c(1, 1, 0,
                      1, 0, 1), nrow = 2, byrow = TRUE)
    out <- row_connections(codes, TRUE)
    expect_equal(dim(out), c(2L, 3L))  # choose(3,2) = 3
})

test_that("rows_to_co_occurrences: binary=TRUE binarises non-zero products", {
    codes <- matrix(c(2, 3, 0,
                      1, 0, 1), nrow = 2, byrow = TRUE)
    out <- row_connections(codes, TRUE)
    # Row 1: 2*3=6>0 -> 1; 2*0=0; 3*0=0
    expect_equal(as.vector(out[1, ]), c(1, 0, 0))
})

test_that("rows_to_co_occurrences: binary=FALSE preserves product magnitudes", {
    codes <- matrix(c(2, 3, 0), nrow = 1)
    out <- row_connections(codes, FALSE)
    # pairs: c1&c2=6, c1&c3=0, c2&c3=0
    expect_equal(as.vector(out[1, ]), c(6, 0, 0))
})

test_that("rows_to_co_occurrences: all-zero codes produce all-zero output", {
    codes <- matrix(0, nrow = 4, ncol = 3)
    out   <- row_connections(codes, TRUE)
    expect_true(all(out == 0))
})

test_that("rows_to_co_occurrences: each row is independent (no cross-row accumulation)", {
    # Row 1 has only c1 active; row 2 has only c2 active.
    # Neither row should show a c1&c2 co-occurrence.
    codes <- matrix(c(1, 0, 0,
                      0, 1, 0), nrow = 2, byrow = TRUE)
    out <- row_connections(codes, FALSE)
    expect_true(all(out == 0))
})

# --- rolling_window_sum ---

test_that("rolling_window_sum: window_size=1 returns the codes matrix unchanged", {
    codes <- matrix(c(1, 0,
                      0, 1,
                      1, 1), nrow = 3, byrow = TRUE)
    out <- rolling_window_sum(codes, 1L)
    expect_equal(out, codes)
})

test_that("rolling_window_sum: window_size=2 sums current and prior row", {
    codes <- matrix(c(1, 0,
                      0, 1,
                      1, 0), nrow = 3, byrow = TRUE)
    out <- rolling_window_sum(codes, 2L)
    expect_equal(as.vector(out[1, ]), c(1, 0))  # row 1: just itself
    expect_equal(as.vector(out[2, ]), c(1, 1))  # rows 1+2
    expect_equal(as.vector(out[3, ]), c(1, 1))  # rows 2+3
})

test_that("rolling_window_sum: window larger than available rows clamps to row 0", {
    codes <- matrix(c(1, 0,
                      1, 1), nrow = 2, byrow = TRUE)
    out <- rolling_window_sum(codes, 10L)
    # Row 2: all rows summed = (2,1)
    expect_equal(as.vector(out[2, ]), c(2, 1))
})

test_that("rolling_window_sum: output dimensions match input", {
    set.seed(99)
    codes <- matrix(runif(12), nrow = 4, ncol = 3)
    out   <- rolling_window_sum(codes, 2L)
    expect_equal(dim(out), dim(codes))
})

# --- accumulate_stanza ordered=TRUE (directed) ---

test_that("accumulate_stanza: ordered=TRUE returns n_rows x n_codes^2", {
    codes <- matrix(c(1, 1, 0,
                      1, 0, 1,
                      0, 1, 1), nrow = 3, byrow = TRUE)
    out <- accumulate_stanza(codes, window_back = 2L, binary = TRUE, ordered = TRUE)
    expect_equal(dim(out), c(3L, 9L))  # 3 rows, 3^2 = 9 columns
})

test_that("accumulate_stanza: ordered=TRUE row 1 has zero ground (no prior rows)", {
    codes <- matrix(c(1, 1, 0,
                      1, 0, 1), nrow = 2, byrow = TRUE)
    out <- accumulate_stanza(codes, window_back = 2L, binary = FALSE, ordered = TRUE)
    # Row 1: no prior rows → ground = zeros → connection_matrix(0,response) = 0.5*resp⊗resp (diag zeroed)
    # response = (1,1,0): resp⊗resp off-diagonal: [1,2]=1, [2,1]=1; all others 0; scaled by 0.5
    expect_equal(out[1, ], as.vector(0.5 * (matrix(c(1,1,0), ncol=1) %*% matrix(c(1,1,0), nrow=1) - diag(c(1,1,0)))))
})

test_that("accumulate_stanza: ordered=TRUE vs FALSE give different column counts", {
    codes <- matrix(runif(12), nrow = 4, ncol = 3)
    out_undirected <- accumulate_stanza(codes, window_back = 2L, ordered = FALSE)
    out_directed   <- accumulate_stanza(codes, window_back = 2L, ordered = TRUE)
    expect_equal(ncol(out_undirected), 3L)   # choose(3,2)
    expect_equal(ncol(out_directed),   9L)   # 3^2
    expect_equal(nrow(out_undirected), nrow(out_directed))
})

test_that("accumulate_stanza: ordered=TRUE binary binarises non-zero entries", {
    codes <- matrix(c(2, 3,
                      1, 1,
                      1, 2), nrow = 3, byrow = TRUE)
    out_bin  <- accumulate_stanza(codes, window_back = 3L, binary = TRUE,  ordered = TRUE)
    out_cont <- accumulate_stanza(codes, window_back = 3L, binary = FALSE, ordered = TRUE)
    # binary output has only 0s and 1s
    expect_true(all(out_bin %in% c(0, 1)))
    # continuous output can have values > 1
    expect_true(any(out_cont > 1))
})

# --- finalize_row_connections / aggregate_row_connections (weight models) ---

# Two response rows, 3 codes, directed 3x3 per row (column-major), with counts > 1
# and asymmetry so the fold and each weight model are distinguishable.
weighted_rc <- function() {
    rc <- matrix(0, nrow = 2, ncol = 9)
    rc[1, 4] <- 2    # m(1,2)
    rc[1, 2] <- 1    # m(2,1)  -> row 1 folds to A&B = 3
    rc[1, 8] <- 4    # m(2,3)  -> row 1 B&C = 4
    rc[2, 7] <- 9    # m(1,3)  -> row 2 A&C = 9
    rc[2, 4] <- 0.5  # m(1,2)  -> row 2 A&B = 0.5
    rc
}

test_that("aggregate_row_connections: unordered weight models apply per row before the sum", {
    rc <- weighted_rc()
    expect_equal(aggregate_row_connections(rc, 3L, FALSE, "binary"),  c(2, 1, 1))
    expect_equal(aggregate_row_connections(rc, 3L, FALSE, "product"), c(3.5, 9, 4))
    expect_equal(aggregate_row_connections(rc, 3L, FALSE, "sqrt"),    c(sqrt(3) + sqrt(0.5), 3, 2))
    expect_equal(aggregate_row_connections(rc, 3L, FALSE, "log1p"),   c(log1p(3) + log1p(0.5), log1p(9), log1p(4)))
    # per-row, not on the unit total: sqrt(3 + 0.5) != sqrt(3) + sqrt(0.5)
    expect_false(isTRUE(all.equal(aggregate_row_connections(rc, 3L, FALSE, "sqrt")[1], sqrt(3.5))))
})

test_that("aggregate_row_connections: legacy TRUE/FALSE and NULL map to binary/product", {
    rc <- weighted_rc()
    expect_equal(aggregate_row_connections(rc, 3L, FALSE, TRUE),  aggregate_row_connections(rc, 3L, FALSE, "binary"))
    expect_equal(aggregate_row_connections(rc, 3L, FALSE, FALSE), aggregate_row_connections(rc, 3L, FALSE, "product"))
    expect_equal(aggregate_row_connections(rc, 3L),               aggregate_row_connections(rc, 3L, FALSE, "binary"))
    expect_equal(aggregate_row_connections(rc, 3L, FALSE, "LOG"), aggregate_row_connections(rc, 3L, FALSE, "log1p"))
    expect_error(aggregate_row_connections(rc, 3L, FALSE, "cube"), "Unknown weight model")
    expect_error(aggregate_row_connections(rc, 3L, FALSE, 1), "must be TRUE/FALSE")
})

test_that("aggregate_row_connections: ordered binary keeps raw directed counts; other weights per cell", {
    rc <- weighted_rc()
    expect_equal(aggregate_row_connections(rc, 3L, TRUE, "binary"),  colSums(rc))
    expect_equal(aggregate_row_connections(rc, 3L, TRUE, "product"), colSums(rc))
    expect_equal(aggregate_row_connections(rc, 3L, TRUE, "sqrt"),    colSums(sqrt(rc)))
    expect_equal(aggregate_row_connections(rc, 3L, TRUE, "log1p"),   colSums(log1p(rc)))
})

test_that("finalize_row_connections: rows are finalised individually and sum to the aggregate", {
    rc <- weighted_rc()
    for (w in c("binary", "product", "sqrt", "log1p")) for (ord in c(FALSE, TRUE)) {
        fin <- finalize_row_connections(rc, 3L, ord, w)
        expect_equal(dim(fin), c(2L, if (ord) 9L else 3L))
        expect_equal(colSums(fin), aggregate_row_connections(rc, 3L, ord, w))
    }
    expect_equal(finalize_row_connections(rc, 3L, FALSE, "sqrt")[1, ], c(sqrt(3), 0, 2))
})

test_that("aggregate_row_connections: apply_tensor rows match rENA ena.accumulate.data weight.by", {
    # One unit/conversation, window 3; rENA's legacy accumulator gives
    # binary 2,3,2 | "product" 12,9,12 | sqrt 4.7321,4.8637,4.7321 | log1p 3.6889,3.7377,3.6889
    codes <- matrix(c(2, 0, 1,
                      1, 3, 0,
                      0, 1, 2), nrow = 3, byrow = TRUE)
    r <- apply_tensor(tensor = c(1, 3), dims = 2L, dims_sender = integer(0),
                      dims_receiver = integer(0), dims_mode = integer(0),
                      context_lookup = matrix(0L, 3, 0), unit_rows = 0:2,
                      codes = codes, times = c(0, 1, 2), ordered = TRUE)
    rc <- r$row_connection_counts
    expect_equal(aggregate_row_connections(rc, 3L, FALSE, "binary"),  c(2, 3, 2))
    expect_equal(aggregate_row_connections(rc, 3L, FALSE, "product"), c(12, 9, 12))
    expect_equal(aggregate_row_connections(rc, 3L, FALSE, "sqrt"),
                 c(4.73205080756888, 4.86370330515627, 4.73205080756888), tolerance = 1e-12)
    expect_equal(aggregate_row_connections(rc, 3L, FALSE, "log1p"),
                 c(3.68887945411394, 3.73766961828337, 3.68887945411394), tolerance = 1e-12)
})
