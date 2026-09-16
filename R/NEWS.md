# libqe 0.1.4

- Added `aggregate_row_connections`, the fold + per-row-binarize + sum aggregation of `apply_tensor_unit`'s per-response-row counts into a unit vector — the step tma performs in R (`as.unordered` + `colSums.ena.matrix(binary)`), now shared in the C++ kernel. WASM and Python bindings added. `apply_tensor_unit` itself is unchanged, so the R tma pipeline (which already does this aggregation in R) is unaffected; no R wrapper is exported for the new function. Note: the JS (rena-wasm) and Python (pytma) consumers previously mis-aggregated the tensor path by reading the raw `connection_counts` — both now call `aggregate_row_connections`.

# libqe 0.1.3

- Added the cross-covariance decay (CCD) window-size kernel (`ccd_window`) shared across the R, WASM, and Python surfaces; ports rENA's `ena.ccd` numeric core to C++.

# libqe 0.1.2

Released: 2026-09-03

- Added door temporal pooling kernels for lookback and EMA smoothing, including ETM-compatible missing-value handling.
- Added trajectory kernels for R-compatible polynomial fitting, curve evaluation, derivatives, integrated distances, lagged distances, signed turn-lag analysis, and distance-distance correlation.
- Added R bindings for the new door and trajectory surfaces.
- Made orthogonal polynomial fitting portable to no-LAPACK WASM builds.
- Added parity and regression coverage for door and trajectory behavior.
