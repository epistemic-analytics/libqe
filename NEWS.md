# libqe 0.1.2

Released: 2026-09-03

- Added door temporal pooling kernels for lookback and EMA smoothing, including ETM-compatible missing-value handling.
- Added trajectory kernels for R-compatible polynomial fitting, curve evaluation, derivatives, integrated distances, lagged distances, signed turn-lag analysis, and distance-distance correlation.
- Added R, Python, and WASM bindings for the new door and trajectory surfaces.
- Made orthogonal polynomial fitting portable to no-LAPACK WASM builds.
- Added parity and regression coverage across R, Python, and WASM.
