# Rotation parity tests against fixtures generated from rENA's own math.
# Fixtures live in tests/testthat/fixtures/rotation/ and are produced by
# fixtures/rotation/_generate.R.
#
# Sign convention: rENA inherits whatever signs LAPACK returns for the SVD,
# and downstream code is sign-tolerant. So these tests compare rotations
# up to per-column sign flips. A deterministic sign convention may be added
# to libqe later; the comparator would then tighten to exact equality.

# ── helpers ───────────────────────────────────────────────────────────────────

load_fixture <- function(name) {
    readRDS(test_path("fixtures", "rotation", paste0(name, ".rds")))
}

# Up-to-sign matrix equality: per-column, check that the column matches the
# expected column either as-is or negated. Allows libqe and rENA to disagree
# on signs while still verifying axis directions.
expect_equal_up_to_sign <- function(actual, expected, tol = 1e-9) {
    expect_equal(dim(actual), dim(expected))
    for (j in seq_len(ncol(expected))) {
        a <- actual[, j]
        e <- expected[, j]
        delta_pos <- max(abs(a - e))
        delta_neg <- max(abs(a + e))
        if (min(delta_pos, delta_neg) > tol) {
            fail(sprintf(
                "column %d differs from expected beyond sign flip (min |Δ| = %g)",
                j, min(delta_pos, delta_neg)))
        }
    }
    invisible(TRUE)
}

# Convert a list-of-pairs from a fixture (1-based indices) to libqe's expected
# input format (0-based integer vectors inside list(a, b)).
pairs_to_libqe <- function(group_pairs) {
    lapply(group_pairs, function(p) {
        list(as.integer(p$a - 1L), as.integer(p$b - 1L))
    })
}

# ── ena_svd ───────────────────────────────────────────────────────────────────

test_that("ena_svd: matches prcomp on small dataset (up to sign)", {
    fx  <- load_fixture("svd_small")
    out <- ena_svd(fx$points)
    expect_equal_up_to_sign(unname(out$rotation), unname(fx$rotation))
    expect_equal(unname(out$eigenvalues), unname(fx$eigenvalues), tolerance = 1e-10)
})

test_that("ena_svd: matches prcomp on medium dataset (up to sign)", {
    fx  <- load_fixture("svd_med")
    out <- ena_svd(fx$points)
    expect_equal_up_to_sign(unname(out$rotation), unname(fx$rotation))
    expect_equal(unname(out$eigenvalues), unname(fx$eigenvalues), tolerance = 1e-10)
})

test_that("ena_svd: column names are SVD1..SVDp", {
    fx  <- load_fixture("svd_small")
    out <- ena_svd(fx$points)
    expect_equal(out$column_names,
                 paste0("SVD", seq_len(ncol(fx$points))))
})

test_that("ena_svd: rotation is orthonormal", {
    fx  <- load_fixture("svd_med")
    out <- ena_svd(fx$points)
    p   <- ncol(fx$points)
    expect_equal(unname(t(out$rotation) %*% out$rotation),
                 diag(p), tolerance = 1e-10)
})

# ── deflate ───────────────────────────────────────────────────────────────────

test_that("deflate: matches manual data - data %*% axis %*% t(axis)", {
    set.seed(7)
    d    <- matrix(rnorm(20 * 4), nrow = 20)
    axis <- rnorm(4); axis <- axis / sqrt(sum(axis ^ 2))
    expect_equal(
        deflate(d, axis),
        d - (d %*% axis) %*% t(axis),
        tolerance = 1e-12
    )
})

test_that("deflate: result is orthogonal to the deflation axis", {
    set.seed(8)
    d    <- matrix(rnorm(15 * 5), nrow = 15)
    axis <- rnorm(5); axis <- axis / sqrt(sum(axis ^ 2))
    out  <- deflate(d, axis)
    expect_true(all(abs(out %*% axis) < 1e-10))
})

# ── means_rotation ────────────────────────────────────────────────────────────

for (name in c("means_small_1pair", "means_small_2pair",
                "means_med_1pair",   "means_med_2pair")) {
    local({
        nm <- name
        test_that(sprintf("means_rotation: %s matches rENA orthogonal_svd (up to sign)", nm), {
            fx  <- load_fixture(nm)
            out <- means_rotation(fx$points, pairs_to_libqe(fx$group_pairs))
            expect_equal_up_to_sign(unname(out$rotation), unname(fx$rotation))
        })
    })
}

test_that("means_rotation: column labels are MR1..MRk, SVD(k+1)..SVDp", {
    fx  <- load_fixture("means_small_2pair")
    out <- means_rotation(fx$points, pairs_to_libqe(fx$group_pairs))
    p   <- ncol(fx$points)
    expect_equal(out$column_names, c("MR1", "MR2", "SVD3", "SVD4"))
    expect_equal(length(out$column_names), p)
})

test_that("means_rotation: rotation is orthonormal", {
    fx  <- load_fixture("means_med_2pair")
    out <- means_rotation(fx$points, pairs_to_libqe(fx$group_pairs))
    p   <- ncol(fx$points)
    expect_equal(unname(t(out$rotation) %*% out$rotation),
                 diag(p), tolerance = 1e-10)
})

test_that("means_rotation: errors on empty group_pairs", {
    expect_error(
        means_rotation(matrix(rnorm(20), nrow = 5), list()),
        "without 2 groups"
    )
})

# ── complete_rotation ─────────────────────────────────────────────────────────

for (name in c("complete_small_1axis", "complete_med_2axis")) {
    local({
        nm <- name
        test_that(sprintf("complete_rotation: %s matches rENA pattern (up to sign)", nm), {
            fx  <- load_fixture(nm)
            k   <- ncol(fx$named_axes)
            labels <- if (k == 1) "GMR1" else paste0("GMR", seq_len(k))
            out <- complete_rotation(fx$points, fx$named_axes, labels)

            # First k columns must match the input named_axes exactly (no flip).
            expect_equal(unname(out$rotation[, 1:k, drop = FALSE]),
                          unname(fx$named_axes),
                          tolerance = 1e-10)

            # Trailing SVD columns match up to sign.
            p <- ncol(fx$points)
            if (k < p) {
                expect_equal_up_to_sign(
                    unname(out$rotation[, (k + 1):p, drop = FALSE]),
                    unname(fx$rotation[, (k + 1):p, drop = FALSE])
                )
            }
        })
    })
}

test_that("complete_rotation: column labels are user-provided then SVD", {
    fx  <- load_fixture("complete_med_2axis")
    out <- complete_rotation(fx$points, fx$named_axes, c("GMR1", "GMR2"))
    expect_equal(out$column_names, c("GMR1", "GMR2", "SVD3", "SVD4", "SVD5", "SVD6"))
})

# ── orthogonal_svd ────────────────────────────────────────────────────────────

test_that("orthogonal_svd: equals means_rotation for the same inputs", {
    # means_rotation is just orthogonal_svd called on the centered+deflated data
    # with the stacked mean-diff weights — so the two outputs should agree.
    fx <- load_fixture("means_small_2pair")
    centered <- scale(fx$points, scale = FALSE, center = TRUE)
    deflated <- centered
    weights  <- matrix(0, ncol(centered), length(fx$group_pairs))
    for (i in seq_along(fx$group_pairs)) {
        a  <- fx$group_pairs[[i]]$a
        b  <- fx$group_pairs[[i]]$b
        d  <- colMeans(deflated[a, , drop = FALSE]) -
              colMeans(deflated[b, , drop = FALSE])
        ax <- d / sqrt(sum(d ^ 2))
        deflated     <- deflate(deflated, ax)
        weights[, i] <- ax
    }
    via_orth  <- orthogonal_svd(deflated, weights, c("MR1", "MR2"))
    via_means <- means_rotation(fx$points, pairs_to_libqe(fx$group_pairs))
    expect_equal(via_orth$rotation, via_means$rotation, tolerance = 1e-12)
})

test_that("orthogonal_svd: errors on label/weights size mismatch", {
    expect_error(
        orthogonal_svd(matrix(rnorm(20), 5), matrix(0, 4, 2), c("A")),
        "named_labels.size"
    )
})

test_that("orthogonal_svd: errors on data/weights dim mismatch", {
    expect_error(
        orthogonal_svd(matrix(rnorm(20), 5), matrix(0, 3, 1), c("A")),
        "data.n_cols"
    )
})
