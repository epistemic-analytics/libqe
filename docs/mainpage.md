# libqe

Header-only C++ numerics shared by the Quantitative Ethnography packages. The
ENA model code is in [libena](https://qe-libs.org/conan/docs/libena/) and
network accumulation in [libtma](https://qe-libs.org/conan/docs/libtma/), both
built on this library. libqe also ships as an R package, a Python extension, a
Julia package and an npm/WebAssembly package.

Everything is in namespace `qe`. Matrices are Armadillo types.

## Install

Add the qe-libs Conan remote and require `libqe`. Available versions are
listed on the [registry page](https://qe-libs.org/conan/).

```
conan remote add qe-libs https://gitlab.com/api/v4/projects/22522458/packages/conan
```

```cmake
find_package(libqe REQUIRED)
target_link_libraries(app PRIVATE libqe::libqe)
```

```cpp
#include <armadillo>
#include <libqe/libqe.hpp>   // every libqe module
```

## Modules

| Header | Contents |
|--------|----------|
| adjacency.hpp | Upper-triangle index pairs, network vector/matrix conversions, connection names |
| normalization.hpp | Row-wise sphere normalization and max-norm scaling of networks |
| stats.hpp | Centering, normal and t quantiles, group confidence and outlier intervals, Welch t-test and Wilcoxon rank-sum |
| linalg_fallback.hpp | LAPACK-free SVD, QR, symmetric eigen and SPD solve |
| lasso.hpp | Coordinate-descent Lasso with k-fold cross-validation |
| door.hpp | Lookback and EMA temporal pooling |
| trajectory.hpp | Polynomial trajectory fitting with LOOCV degree selection, evaluation and derivatives |
| trajectory_distance.hpp | Integrated and lagged distances between trajectory curves |
| trajectory_following.hpp | Who-follows-whom turn-lag analysis |
| stability.hpp | Distance-distance correlation for stability comparisons |
| validate.hpp | Input checks shared by the language bindings |
| bind/ | Armadillo conversions for the nanobind, Emscripten and CxxWrap bindings; not included by `libqe.hpp` |

## Source

[libqe on GitLab](https://gitlab.com/epistemic-analytics/qe-packages/libqe) —
the headers are in `include/libqe/`.
