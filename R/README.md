# libqe R package

R package that ships the `libqe` C++ headers so downstream packages can
consume them via `LinkingTo`, and exports Rcpp-wrapped versions of every
function for direct use and testing in R.

## Installation

```r
install.packages("libqe", repos = c("https://cran.qe-libs.org", "https://cran.rstudio.com"))
```

From source (within the repo root):

```r
R CMD INSTALL R/
```

## Using libqe headers in your R package

Add to your `DESCRIPTION`:

```
LinkingTo: Rcpp, RcppArmadillo, libqe
Imports: Rcpp
```

Then in your `.cpp` files:

```cpp
// [[Rcpp::depends(RcppArmadillo, libqe)]]
#include <RcppArmadillo.h>
#include <libqe/libqe.hpp>

// All qe:: functions are now available
```

## R function reference

All exported R functions are prefixed ``.

### Adjacency utilities

| Function | Description |
|----------|-------------|
| `connection_indices(len, row)` | Upper-triangle (i, j) index pairs for a matrix of side `len`. `row = -1` returns both rows, `0` row indices only, `1` col indices only. |
| `code_connections(v)` | Pairwise products of a code vector → upper-triangle connection vector. |
| `fold_directed_network(v)` | Fold a directed (n²) flat vector into an undirected upper-triangle vector by summing symmetric pairs. |
| `network_to_vector(x, full)` | Flatten an adjacency matrix. `full = TRUE` returns the full n² vector (directed); `full = FALSE` returns the upper triangle. |
| `connection_names(v)` | Generate code-pair labels (`"A & B"`) for every upper-triangle position from a character vector of code names. |

### Normalization

| Function | Description |
|----------|-------------|
| `normalize_networks(m)` | Row-wise L2 normalization. Each row is divided by its Euclidean norm (zero rows are left as-is). |
| `scale_networks(m)` | Max-norm scaling: divide all rows by the largest row L2 norm in the matrix. |

### Modeling

| Function | Description |
|----------|-------------|
| `mean_ci(points, conf_level)` | t-based confidence interval for the mean of a group of ENA unit points. Returns an `n_dims × 3` matrix with columns `[mean, ci_lower, ci_upper]`. Uses `n-1` degrees of freedom; default `conf_level = 0.95`. Bounds match rENA's `t.test(points[,d])$conf.int` exactly — see shape note below. When `n == 1` the CI bounds are `±Inf`. |
| `outlier_ci(points, iqr_factor)` | Outlier interval using IQR × `iqr_factor` (Tukey fence; default `iqr_factor = 1.5`). Returns an `n_dims × 2` matrix with columns `[lower, upper]` where `lower = -iqr_factor × IQR` and `upper = +iqr_factor × IQR`. Symmetric around 0. Matches rENA's IQR-based formula exactly. |
| `center_points(values)` | Subtract column means (center each dimension). |
| `ena_correlation(points, centroids, conf_level)` | Pearson correlation with confidence interval between ENA unit points and group centroids. Returns a matrix with columns `r`, `lower`, `upper`. Default `conf_level = 0.95`. |
| `node_positions(adj_mats, t, num_dims)` | Least-squares node positions for **undirected** ENA. Returns a list with `nodes`, `centroids`, `weights`, `points`. |
| `directed_node_positions(line_weights, points, num_dims)` | Least-squares node positions for **directed** ENA. Same return structure. |
| `directed_node_positions_combine_pairs(line_weights, points, num_dims)` | Directed node positions where paired ground+response rows are combined before solving. Same return structure. |

### Accumulation

| Function | Description |
|----------|-------------|
| `connection_matrix(ground, response, response_weight, ordered)` | Core adjacency math for one ground+response event pair. `ordered = TRUE` → directed; `FALSE` → undirected. |
| `accumulate_stanza(codes, window_back, window_forward, binary, ordered=FALSE)` | Stanza-window accumulation. `ordered=FALSE` (default): undirected upper-tri, returns `n_rows × choose(n_codes, 2)`. `ordered=TRUE`: directed — focal row as response, prior window rows as ground, returns `n_rows × n_codes²`. Use `.Machine$integer.max` for an infinite back window.| Traditional stanza-window accumulation (rENA model). `codes` is the code matrix for **one conversation**. Returns a matrix with `choose(n_codes, 2)` columns. Use `.Machine$integer.max` for an infinite window. |
| `row_connections(codes, binary)` | Per-row upper-triangle co-occurrence matrix without windowing. |
| `rolling_window_sum(codes, window_size)` | Rolling backward sum of raw code values (no upper-tri transform). |
| `accumulate_unit(codes, unit_rows, decay_fn, ordered)` | Ground/response accumulation for one unit (tma model). `decay_fn` is an R function mapping a distance vector to weights. `unit_rows` are **0-based**. |
| `accumulate_unit_with_rows(codes, unit_rows, decay_fn, ordered)` | Like `accumulate_unit` but also returns per-response-row connection data. Returns a list with `networks` and `row_networks`. |
| `flat_index(indices, dims)` | Column-major linear index into a multi-dimensional array. `indices` and `dims` are **0-based** integer vectors. |
| `apply_tensor(tensor, dims, dims_sender, dims_receiver, dims_mode, context_lookup, unit_rows, codes, times, ordered)` | Tensor-based multi-modal accumulation (tma model). Returns a list with `connection_counts` and `row_connection_counts`. |

### Door, trajectory, and stability

| Function | Description |
|----------|-------------|
| `door_lookback_block(block, lookback_size, aggregate_mean, weighting_linear, segment_ids)` | Lookback pooling over one unit block, matching ETM missing-value behavior. |
| `door_ema_block(block, alpha, segment_ids)` | EMA smoothing over one unit block, carrying prior smoothed values for missing observations. |
| `fit_trajectory_poly(points, t, max_degree, fixed_degree, criterion, basis)` | Fit a 2D polynomial trajectory; defaults to R-compatible orthogonal polynomials. |
| `eval_trajectory_curve(coeffs_x, coeffs_y, t_eval)` | Evaluate fitted polynomial trajectory coordinates. |
| `eval_trajectory_derivatives(coeffs_x, coeffs_y, t_eval)` | Evaluate velocity, acceleration, speed, heading rate, and curvature. |
| `integrated_trajectory_distance(...)` | Integrated Euclidean distance between two fitted trajectory curves. |
| `lagged_trajectory_distance(...)` | Normalized lagged distance for follower/leader curve comparisons. |
| `signed_turn_lag(...)` / `sweep_signed_turn_lags(...)` | Discrete turn-lag follower/leader distance metrics. |
| `dist_dist_correlation(X, Y)` | Pearson correlation between pairwise distance structures. |

### Shape note — `mean_ci` vs rENA

rENA returns a `2 × n_dims` matrix (rows = lower/upper, columns = dimensions):

```r
# rENA
matrix(c(t.test(pts[,1])$conf.int, t.test(pts[,2])$conf.int), ncol = 2)
#       [,1]   [,2]
# [1,]  lower  lower   ← row 1 = lower bounds
# [2,]  upper  upper   ← row 2 = upper bounds
```

`mean_ci` returns `n_dims × 3` (rows = dimensions, columns = mean/lower/upper):

```r
mean_ci(pts)
#       [,1]  [,2]  [,3]
# [1,]  mean  lower upper   ← dim 1
# [2,]  mean  lower upper   ← dim 2
```

The lower/upper **values are identical** — the shapes are transposed. Convert with:
```r
t(mean_ci(pts)[, 2:3])   # → 2 × n_dims matching rENA's format
```

## Quick examples

```r
library(libqe)

# Code-pair labels for 4 codes
connection_names(c("A", "B", "C", "D"))
# [1] "A & B" "A & C" "A & D" "B & C" "B & D" "C & D"

# Stanza-window accumulation (infinite back window)
codes <- matrix(c(1,1,0, 1,0,1, 0,1,1), nrow = 3, byrow = TRUE)
accumulate_stanza(codes, window_back = .Machine$integer.max, binary = TRUE)

# Sphere normalization
normalize_networks(codes)

# Group confidence interval (matches R's t.test per dimension)
group_pts <- matrix(rnorm(20), nrow = 10, ncol = 2)
mean_ci(group_pts, conf_level = 0.95)
# n_dims × 3:  cols = [mean, ci_lower, ci_upper]

# Outlier interval (matches rENA's IQR * 1.5 formula)
outlier_ci(group_pts, iqr_factor = 1.5)
# n_dims × 2:  cols = [lower, upper]  (symmetric: lower = -1.5*IQR, upper = +1.5*IQR)

# tma-style accumulation with a simple decay function
decay <- function(distances) rep(1, length(distances))
accumulate_unit(codes, unit_rows = c(0L, 1L, 2L), decay_fn = decay, ordered = FALSE)
```

## Running tests

```r
testthat::test_local()
# or
R CMD check .
```
