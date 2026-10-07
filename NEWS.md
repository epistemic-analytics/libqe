# libqe (development version)

- Python: the `qe-lib` sdist is self-contained. It used to omit the libqe headers (CMake read them from `../include`, outside the sdist), so every source install failed — Windows, Python 3.14, macOS before 26. `scripts/sync-headers.sh` now also copies them into `python/include/`.
- Python: Windows wheels are built by the GitHub `windows-builds.yml` workflow from the published sdist (dispatched by cranqe's nightly), and the build uses a fixed `build/{wheel_tag}` directory instead of a temp dir whose cleanup failed on Windows (WinError 32).

# libqe 0.1.7

- Python: the package is now published as **`qe-lib`** and imported as **`qe`** (was `pylibqe`; extension module `_pylibqe` → `_qe`). Already-published `pylibqe` wheels stay installable from qe-libs.org. The C++ API is unchanged.
- Python: Linux wheels link exactly one BLAS/LAPACK (OpenBLAS) and use Armadillo header-only. They used to link the distro's `libarmadillo` wrapper plus every BLAS CMake found — three OpenBLAS builds, ATLAS, reference BLAS/LAPACK, SuperLU and ARPACK — all of which `auditwheel` bundled, making each manylinux wheel ~43 MB (the qe-libs.org site passed GitLab Pages' 1 GB limit). A manylinux_2_28 wheel is now ~5 MB (aarch64: 23.1 → 5.2 MB) with identical results. macOS (Accelerate) and Windows (vcpkg) builds are unchanged.

# libqe 0.1.6

- Fixed `complete_rotation` — and so `generalized_means_rotation` (GMR, rENA's `ena.rotate.by.generalized`) — returning a non-orthonormal rotation on rank-deficient data, e.g. a model with a masked (all-zero) connection column or fewer units than connections. The trailing SVD axes then came from the deflated data's null space, which contains the named axes, so one could nearly duplicate GMR1 (|cos| 0.99 on a masked RS.data model): the SVD axes after the named ones were skewed and the variance shares wrong (GMR1 23.7% instead of 29.9%). Null-space axes are now orthogonalised against the named axes and each other. Full-rank results are unchanged (bit-identical on RS.data), as is the rENA pattern for non-orthogonal named axes.
- R: corrected the OpenMP flags in `Makevars` / `Makevars.win` to `SHLIB_OPENMP_CXXFLAGS` for both compiling and linking (was the C/Fortran variables), as R requires for a C++ package; fixes linking on Windows.
- Package metadata (R `URL`/`BugReports`, Python project URLs, npm `repository`) now points at https://github.com/epistemic-analytics/libqe.

# libqe 0.1.5

- Added weight models to `aggregate_row_connections` (= rENA's `weight.by`): `"binary"`, `"product"`, `"sqrt"` and `"log1p"` (alias `"log"`). The weight is applied per response row — after the fold (unordered) or per directed cell (ordered) — and before rows are summed into the unit network, the same stage legacy rENA applies `weight.by`. `"binary"` on ordered networks keeps the raw directed counts, as before.
- Added `finalize_row_connections`, the per-row step (fold + weight) without the sum, so callers can expose line-level connection counts.
- The weight argument accepts either a weight-model name or the previous boolean binary flag (`TRUE` = `"binary"`, `FALSE` = `"product"`); existing callers are unaffected. The C++ `bool binary` overload is retained with identical results.
- Both functions are now exported from the R package (previously WASM and Python only). WASM and Python bindings updated; the Python keyword is now `weight`.

# libqe 0.1.4

- Added `aggregate_row_connections`, the fold + per-row-binarize + sum aggregation of `apply_tensor_unit`'s per-response-row counts into a unit vector — the step tma performs in R (`as.unordered` + `colSums.ena.matrix(binary)`), now shared in the C++ kernel. WASM and Python bindings added. `apply_tensor_unit` itself is unchanged, so the R tma pipeline (which already does this aggregation in R) is unaffected; no R wrapper is exported for the new function. Note: the JS (rena-wasm) and Python (pytma) consumers previously mis-aggregated the tensor path by reading the raw `connection_counts` — both now call `aggregate_row_connections`.

# libqe 0.1.3

- Added the cross-covariance decay (CCD) window-size kernel (`ccd_window`) shared across the R, WASM, and Python surfaces; ports rENA's `ena.ccd` numeric core to C++.

# libqe 0.1.2

Released: 2026-09-03

- Added door temporal pooling kernels for lookback and EMA smoothing, including ETM-compatible missing-value handling.
- Added trajectory kernels for R-compatible polynomial fitting, curve evaluation, derivatives, integrated distances, lagged distances, signed turn-lag analysis, and distance-distance correlation.
- Added R, Python, and WASM bindings for the new door and trajectory surfaces.
- Made orthogonal polynomial fitting portable to no-LAPACK WASM builds.
- Added parity and regression coverage across R, Python, and WASM.
