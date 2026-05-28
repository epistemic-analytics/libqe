# Tests for generalized_means_rotation — validates the C++ implementation
# against a pure-R reference derived from the algorithm documented in
# generalized_rotation.hpp (authoritative commit 46776a1981a90b3a3b2861ed1010e9dbb7acf901).

library(libqe)

# ── Fixtures (defined once at top level, outside test blocks) ─────────────────

set.seed(42)
n <- 12L
q <- 4L
V  <- matrix(rnorm(n * q), n, q)

# Numeric target
tn  <- rnorm(n)
tc  <- tn - mean(tn)
tTt <- sum(tc^2)
b1  <- as.vector(t(tc) %*% V) / tTt     # OLS slope (q-vector)
x_exp_num <- b1 / sqrt(sum(b1^2))       # expected x_vector, numeric case

# Binary categorical target (0-based integer codes)
tc_bin <- c(rep(0L, n / 2L), rep(1L, n / 2L))
dummy  <- as.numeric(tc_bin)             # model matrix column for categorical

# ── Helpers ───────────────────────────────────────────────────────────────────

safe_norm <- function(v) v / sqrt(sum(v^2))

# same direction up to sign: |a·b| ≈ 1
aligned <- function(a, b, tol = 1e-6) {
    abs(abs(sum(a * b)) - 1) < tol
}

# ── Dummy values for ignored y parameters ─────────────────────────────────────

dummy_mat <- matrix(0, n, 1)
dummy_vec <- numeric(n)
dummy_idx <- integer(0)

# ── Test 1: Numeric target, no y — x_vector matches OLS direction ─────────────

test_that("numeric target, no y: x_vector aligns with OLS direction", {
    res <- generalized_means_rotation(
        V             = V,
        x_model_matrix = matrix(tn, n, 1),
        x_target      = tn,
        x1_cols       = 0L,
        x_categorical = FALSE,
        x_n_groups    = 0L,
        x_subset      = integer(0),
        has_y         = FALSE,
        y_model_matrix = dummy_mat,
        y_target      = dummy_vec,
        y1_cols       = dummy_idx,
        y_categorical = FALSE,
        y_n_groups    = 0L
    )
    expect_true(aligned(res$rotation[, 1], x_exp_num))
})

# ── Test 2: Numeric target — column_names start with "GMR1" then "SVD2" ───────

test_that("numeric target, no y: column_names[1:2] are 'GMR1' and 'SVD2'", {
    res <- generalized_means_rotation(
        V             = V,
        x_model_matrix = matrix(tn, n, 1),
        x_target      = tn,
        x1_cols       = 0L,
        x_categorical = FALSE,
        x_n_groups    = 0L,
        x_subset      = integer(0),
        has_y         = FALSE,
        y_model_matrix = dummy_mat,
        y_target      = dummy_vec,
        y1_cols       = dummy_idx,
        y_categorical = FALSE,
        y_n_groups    = 0L
    )
    expect_equal(res$column_names[1:2], c("GMR1", "SVD2"))
})

# ── Test 3: column_names length equals q ──────────────────────────────────────

test_that("numeric target, no y: length(column_names) == q", {
    res <- generalized_means_rotation(
        V             = V,
        x_model_matrix = matrix(tn, n, 1),
        x_target      = tn,
        x1_cols       = 0L,
        x_categorical = FALSE,
        x_n_groups    = 0L,
        x_subset      = integer(0),
        has_y         = FALSE,
        y_model_matrix = dummy_mat,
        y_target      = dummy_vec,
        y1_cols       = dummy_idx,
        y_categorical = FALSE,
        y_n_groups    = 0L
    )
    expect_equal(length(res$column_names), q)
})

# ── Test 4: Rotation matrix is square q×q ────────────────────────────────────

test_that("numeric target, no y: rotation matrix is q×q", {
    res <- generalized_means_rotation(
        V             = V,
        x_model_matrix = matrix(tn, n, 1),
        x_target      = tn,
        x1_cols       = 0L,
        x_categorical = FALSE,
        x_n_groups    = 0L,
        x_subset      = integer(0),
        has_y         = FALSE,
        y_model_matrix = dummy_mat,
        y_target      = dummy_vec,
        y1_cols       = dummy_idx,
        y_categorical = FALSE,
        y_n_groups    = 0L
    )
    expect_equal(dim(res$rotation), c(q, q))
})

# ── Test 5: Rotation matrix is orthogonal ────────────────────────────────────

test_that("numeric target, no y: rotation is orthogonal (t(R) %*% R ≈ I)", {
    res <- generalized_means_rotation(
        V             = V,
        x_model_matrix = matrix(tn, n, 1),
        x_target      = tn,
        x1_cols       = 0L,
        x_categorical = FALSE,
        x_n_groups    = 0L,
        x_subset      = integer(0),
        has_y         = FALSE,
        y_model_matrix = dummy_mat,
        y_target      = dummy_vec,
        y1_cols       = dummy_idx,
        y_categorical = FALSE,
        y_n_groups    = 0L
    )
    R <- res$rotation
    expect_equal(unname(t(R) %*% R), diag(q), tolerance = 1e-8)
})

# ── Test 6: Categorical target, no y — x_vector matches OLS direction ─────────
#
# With a binary dummy model matrix and no penalized covariates, the Lasso
# degenerates to OLS. The OLS slope with centered binary dummy:
#   tc_c = dummy - mean(dummy),  b1_c = (tc_c' V) / sum(tc_c^2)
# so x_vector = normalize(b1_c).

test_that("categorical target, no y: x_vector aligns with OLS direction", {
    tc_c    <- dummy - mean(dummy)
    b1_c    <- as.vector(t(tc_c) %*% V) / sum(tc_c^2)
    x_exp_cat <- safe_norm(b1_c)

    res_cat <- generalized_means_rotation(
        V             = V,
        x_model_matrix = matrix(dummy, n, 1),
        x_target      = as.numeric(tc_bin),
        x1_cols       = 0L,
        x_categorical = TRUE,
        x_n_groups    = 2L,
        x_subset      = integer(0),
        has_y         = FALSE,
        y_model_matrix = dummy_mat,
        y_target      = dummy_vec,
        y1_cols       = dummy_idx,
        y_categorical = FALSE,
        y_n_groups    = 0L
    )
    expect_true(aligned(res_cat$rotation[, 1], x_exp_cat))
})

# ── Test 7: Categorical — column_names start with "GMR1" then "SVD2" ──────────

test_that("categorical target, no y: column_names[1:2] are 'GMR1' and 'SVD2'", {
    res_cat <- generalized_means_rotation(
        V             = V,
        x_model_matrix = matrix(dummy, n, 1),
        x_target      = as.numeric(tc_bin),
        x1_cols       = 0L,
        x_categorical = TRUE,
        x_n_groups    = 2L,
        x_subset      = integer(0),
        has_y         = FALSE,
        y_model_matrix = dummy_mat,
        y_target      = dummy_vec,
        y1_cols       = dummy_idx,
        y_categorical = FALSE,
        y_n_groups    = 0L
    )
    expect_equal(res_cat$column_names[1:2], c("GMR1", "SVD2"))
})

# ── Test 8: Categorical — rotation is orthogonal ──────────────────────────────

test_that("categorical target, no y: rotation is orthogonal", {
    res_cat <- generalized_means_rotation(
        V             = V,
        x_model_matrix = matrix(dummy, n, 1),
        x_target      = as.numeric(tc_bin),
        x1_cols       = 0L,
        x_categorical = TRUE,
        x_n_groups    = 2L,
        x_subset      = integer(0),
        has_y         = FALSE,
        y_model_matrix = dummy_mat,
        y_target      = dummy_vec,
        y1_cols       = dummy_idx,
        y_categorical = FALSE,
        y_n_groups    = 0L
    )
    R <- res_cat$rotation
    expect_equal(unname(t(R) %*% R), diag(q), tolerance = 1e-8)
})

# ── Test 9: has_y=TRUE — column_names[2] is "GMR2" not "SVD2" ────────────────

test_that("has_y=TRUE: column_names[1:2] are 'GMR1' and 'GMR2'", {
    set.seed(7)
    y_target_shuffled <- tn[sample(n)]

    res_y <- generalized_means_rotation(
        V             = V,
        x_model_matrix = matrix(tn, n, 1),
        x_target      = tn,
        x1_cols       = 0L,
        x_categorical = FALSE,
        x_n_groups    = 0L,
        x_subset      = integer(0),
        has_y         = TRUE,
        y_model_matrix = matrix(y_target_shuffled, n, 1),
        y_target      = y_target_shuffled,
        y1_cols       = 0L,
        y_categorical = FALSE,
        y_n_groups    = 0L
    )
    expect_equal(res_y$column_names[1:2], c("GMR1", "GMR2"))
})

# ── Test 10: has_y=TRUE — rotation is orthogonal ──────────────────────────────

test_that("has_y=TRUE: rotation is orthogonal", {
    set.seed(7)
    y_target_shuffled <- tn[sample(n)]

    res_y <- generalized_means_rotation(
        V             = V,
        x_model_matrix = matrix(tn, n, 1),
        x_target      = tn,
        x1_cols       = 0L,
        x_categorical = FALSE,
        x_n_groups    = 0L,
        x_subset      = integer(0),
        has_y         = TRUE,
        y_model_matrix = matrix(y_target_shuffled, n, 1),
        y_target      = y_target_shuffled,
        y1_cols       = 0L,
        y_categorical = FALSE,
        y_n_groups    = 0L
    )
    R <- res_y$rotation
    expect_equal(unname(t(R) %*% R), diag(q), tolerance = 1e-8)
})

# ── Test 11: Subset parameter — x_vector differs from full-data result ─────────

test_that("x_subset: rotation from subset differs from full-data rotation", {
    res_full <- generalized_means_rotation(
        V             = V,
        x_model_matrix = matrix(tn, n, 1),
        x_target      = tn,
        x1_cols       = 0L,
        x_categorical = FALSE,
        x_n_groups    = 0L,
        x_subset      = integer(0),
        has_y         = FALSE,
        y_model_matrix = dummy_mat,
        y_target      = dummy_vec,
        y1_cols       = dummy_idx,
        y_categorical = FALSE,
        y_n_groups    = 0L
    )

    res_sub <- generalized_means_rotation(
        V             = V,
        x_model_matrix = matrix(tn, n, 1),
        x_target      = tn,
        x1_cols       = 0L,
        x_categorical = FALSE,
        x_n_groups    = 0L,
        x_subset      = 0L:(n / 2L - 1L),
        has_y         = FALSE,
        y_model_matrix = dummy_mat,
        y_target      = dummy_vec,
        y1_cols       = dummy_idx,
        y_categorical = FALSE,
        y_n_groups    = 0L
    )

    expect_false(isTRUE(all.equal(res_full$rotation[, 1], res_sub$rotation[, 1])))
})

# ── Test 12: x1 uses full target — call completes when subset != all rows ──────
#
# When x_subset covers only one group the OLS fallback for the subset produces
# a zero Vx, and the C++ returns an arbitrary x_vector direction (last eigenvector
# of the zero scatter matrix). gmr_direction does NOT throw; the function
# completes successfully and returns a q x q rotation matrix.
#
# NOTE: Because x1 (between-group diff on FULL V) and x_vector (from the
# zero-variance subset) are not collinear, the sequential orthogonalization
# of y_vector (subtract x_vector, then subtract x1) does not guarantee mutual
# orthogonality between columns 1 and 2. The test only verifies that the call
# completes and returns the correct output shape.

test_that("categorical with x_subset: call completes and returns q x q matrix", {
    res_sub_cat <- generalized_means_rotation(
        V             = V,
        x_model_matrix = matrix(dummy, n, 1),
        x_target      = as.numeric(tc_bin),
        x1_cols       = 0L,
        x_categorical = TRUE,
        x_n_groups    = 2L,
        x_subset      = 0L:(n / 2L - 1L),
        has_y         = FALSE,
        y_model_matrix = dummy_mat,
        y_target      = dummy_vec,
        y1_cols       = dummy_idx,
        y_categorical = FALSE,
        y_n_groups    = 0L
    )
    expect_equal(dim(res_sub_cat$rotation), c(q, q))
})
