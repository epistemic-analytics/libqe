# Generate rotation fixtures by running the math of rENA's rotation routines on
# synthetic inputs. Run from the repo root after a `R CMD INSTALL R/`:
#
#   Rscript R/tests/testthat/fixtures/rotation/_generate.R
#
# Outputs:
#   svd_<name>.rds       -- list(points, rotation, eigenvalues)
#   means_<name>.rds     -- list(points, group_pairs, rotation, eigenvalues)
#   complete_<name>.rds  -- list(points, named_axes, rotation, eigenvalues)
#
# Fixtures are stored as-is (no sign alignment) — the test suite compares
# up-to-sign to match rENA's long-standing convention.

stopifnot(requireNamespace("rENA", quietly = TRUE))

# Resolve fixture directory from this script's path
fixture_dir <- "R/tests/testthat/fixtures/rotation"
dir.create(fixture_dir, showWarnings = FALSE, recursive = TRUE)

# ── helpers ───────────────────────────────────────────────────────────────────

# Replicate rENA's prcomp call used in every rotation routine.
prcomp_ena <- function(x) {
    prcomp(x, retx = FALSE, scale. = FALSE, center = FALSE, tol = 0)
}

# Replicate ena.svd's payload (just rotation + eigenvalues).
gen_svd_fixture <- function(name, points) {
    pc <- prcomp_ena(points)
    saveRDS(list(
        points      = points,
        rotation    = unname(pc$rotation),
        eigenvalues = pc$sdev ^ 2
    ), file.path(fixture_dir, paste0("svd_", name, ".rds")))
}

# Replicate ena.rotate.by.mean's pipeline, calling rENA's own orthogonal_svd()
# so we are testing against the *exact* math libqe must reproduce.
gen_means_fixture <- function(name, points, group_pairs) {
    centered <- scale(points, scale = FALSE, center = TRUE)
    deflated <- centered
    weights  <- matrix(0, nrow = ncol(centered), ncol = length(group_pairs))
    for (i in seq_along(group_pairs)) {
        a <- group_pairs[[i]]$a
        b <- group_pairs[[i]]$b
        m1 <- colMeans(deflated[a, , drop = FALSE])
        m2 <- colMeans(deflated[b, , drop = FALSE])
        d  <- m1 - m2
        ax <- d / sqrt(sum(d ^ 2))
        deflated  <- deflated - (deflated %*% ax) %*% t(ax)
        weights[, i] <- ax
    }
    rot <- rENA:::orthogonal_svd(deflated, weights)
    saveRDS(list(
        points      = points,
        group_pairs = group_pairs,
        rotation    = unname(as.matrix(rot)),
        eigenvalues = NULL                       # rENA doesn't save eigenvalues here
    ), file.path(fixture_dir, paste0("means_", name, ".rds")))
}

# Replicate the tail of ena.rotate.by.generalized (canonical: commit 2c07912
# on rENA origin/main). The deflation is *parallel*:
#   defA = A - A %*% v1 %*% t(v1) - A %*% v2 %*% t(v2)
# matching the literal R expression. Sequential deflation is mathematically
# different for non-orthogonal axes.
gen_complete_fixture <- function(name, points, named_axes) {
    A    <- as.matrix(points)
    defA <- A - A %*% named_axes %*% t(named_axes)
    pc   <- prcomp_ena(defA)
    k    <- ncol(named_axes)
    p    <- ncol(A)
    combined <- cbind(named_axes, pc$rotation[, seq_len(p - k), drop = FALSE])
    saveRDS(list(
        points      = points,
        named_axes  = named_axes,
        rotation    = unname(combined),
        eigenvalues = pc$sdev ^ 2                # padded layout: first k are unused
    ), file.path(fixture_dir, paste0("complete_", name, ".rds")))
}

# ── datasets ──────────────────────────────────────────────────────────────────

set.seed(42)

# small_4d: 30 units x 4 dims, two natural groups
pts_small <- matrix(rnorm(30 * 4), nrow = 30, ncol = 4)
pts_small[1:15, ] <- pts_small[1:15, ] + matrix(c(2, -1, 0.5, 0),
                                                  nrow = 15, ncol = 4,
                                                  byrow = TRUE)
small_pairs_1 <- list(list(a = 1:15, b = 16:30))
small_pairs_2 <- list(
    list(a = 1:15,  b = 16:30),
    list(a = 1:10,  b = 21:30)
)

# medium_6d: 50 units x 6 dims (closer to a realistic ENA upper-triangle size
# for a 4-code dataset: choose(4,2) = 6)
pts_med <- matrix(rnorm(50 * 6), nrow = 50, ncol = 6)
pts_med[1:25, 1] <- pts_med[1:25, 1] + 1.5
pts_med[1:25, 3] <- pts_med[1:25, 3] - 0.8
med_pairs_1 <- list(list(a = 1:25, b = 26:50))
med_pairs_2 <- list(
    list(a = 1:25, b = 26:50),
    list(a = c(1:10, 26:35), b = c(11:25, 36:50))
)

# Named axes for complete_rotation fixtures: just use centered group-mean
# differences (unit-normed). This isn't gmr, but it's a perfectly valid set
# of unit-norm "named axes" for testing the math.
make_named <- function(pts, group_pairs) {
    centered <- scale(pts, scale = FALSE, center = TRUE)
    out <- matrix(0, nrow = ncol(centered), ncol = length(group_pairs))
    for (i in seq_along(group_pairs)) {
        a <- group_pairs[[i]]$a
        b <- group_pairs[[i]]$b
        d <- colMeans(centered[a, , drop = FALSE]) -
             colMeans(centered[b, , drop = FALSE])
        out[, i] <- d / sqrt(sum(d ^ 2))
    }
    out
}

# ── generate ──────────────────────────────────────────────────────────────────

gen_svd_fixture("small", scale(pts_small, scale = FALSE, center = TRUE))
gen_svd_fixture("med",   scale(pts_med,   scale = FALSE, center = TRUE))

gen_means_fixture("small_1pair", pts_small, small_pairs_1)
gen_means_fixture("small_2pair", pts_small, small_pairs_2)
gen_means_fixture("med_1pair",   pts_med,   med_pairs_1)
gen_means_fixture("med_2pair",   pts_med,   med_pairs_2)

gen_complete_fixture("small_1axis",
                      scale(pts_small, scale = FALSE, center = TRUE),
                      make_named(pts_small, small_pairs_1))
gen_complete_fixture("med_2axis",
                      scale(pts_med, scale = FALSE, center = TRUE),
                      make_named(pts_med, med_pairs_2))

cat("Generated", length(list.files(fixture_dir, pattern = "\\.rds$")),
    "fixtures in", fixture_dir, "\n")
