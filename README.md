# libqe

Header-only C++ library providing shared computational primitives for
[Quantitative Ethnography](https://www.quantitativeethnography.org/) packages.
`libqe` is the common core consumed by [rENA](https://gitlab.com/epistemic-analytics/qe-packages/rENA)
and [tma](https://gitlab.com/epistemic-analytics/qe-packages/tma); it ships as
an R package (via `LinkingTo`) and as a Python extension (nanobind).

## Modules

| Module | Header | Contents |
|--------|--------|----------|
| **adjacency** | `adjacency.hpp` | Upper-triangle index pairs, vector/matrix ↔ upper-tri conversions, code-name pair strings |
| **normalization** | `normalization.hpp` | Row-wise L2 sphere norm, max-norm (skip-sphere) scaling |
| **modeling** | `modeling.hpp` | Column-mean centering, group confidence interval (t-based, matches rENA), outlier interval (IQR-based, matches rENA), Pearson correlation with CI, least-squares node positions (undirected, directed, ground/response) |
| **accumulation** | `accumulation.hpp` | Core adjacency math, stanza-window accumulation (rENA), ground/response accumulation with decay (tma), tensor-based multi-modal accumulation (tma), rolling window sum, per-row co-occurrence |

All four modules are pulled in by `#include <libqe/libqe.hpp>`.

## Repository layout

```
libqe/
├── include/libqe/      ← canonical C++ headers (single source of truth)
│   ├── libqe.hpp
│   ├── adjacency.hpp
│   ├── normalization.hpp
│   ├── modeling.hpp
│   └── accumulation.hpp
├── R/                  ← R package (Rcpp wrappers + LinkingTo mechanism)
│   ├── DESCRIPTION
│   ├── configure       ← copies headers into inst/include/ at install time
│   ├── src/
│   │   └── libqe_rcpp.cpp
│   └── tests/testthat/
├── python/             ← pylibqe Python extension (nanobind + CMake)
│   ├── src/pylibqe.cpp
│   ├── tests/
│   └── pyproject.toml
└── scripts/
    ├── sync-headers.sh          ← copy include/ → R/inst/include/
    └── check-headers-in-sync.sh ← verify the two trees match
```

Edit headers in `include/libqe/` only. The R `configure` script and
`scripts/sync-headers.sh` propagate them.

## Using libqe in an R package (`LinkingTo`)

This is the standard pattern used by RcppArmadillo and BH.

**`DESCRIPTION`**
```
LinkingTo: Rcpp, RcppArmadillo, libqe
Imports: Rcpp
```

**`src/your_code.cpp`**
```cpp
// [[Rcpp::depends(RcppArmadillo, libqe)]]
#include <RcppArmadillo.h>
#include <libqe/libqe.hpp>

// use qe::stanza_window(), qe::lws_lsq_positions(), etc.
```

Install libqe from the repository before installing the downstream package:
```r
install.packages("libqe", repos = "https://rena.qe-libs.org")
```

## Installing the R package

```r
# From the QE package repository
install.packages("libqe", repos = "https://rena.qe-libs.org")

# From source (within the repo)
R CMD INSTALL R/
```

## Installing the Python package

```bash
cd python
pip install -e ".[dev]"
```

See [python/README.md](python/README.md) for full build requirements and usage.

## Running tests

**R**
```r
cd R && Rscript -e "testthat::test_local()"
# or: R CMD check R/
```

**Python**
```bash
cd python && pytest tests/
```

## Header sync

The R package's `configure` script copies `include/libqe/*.hpp` into
`R/inst/include/libqe/` at install time.  To sync manually (e.g. before
`R CMD build`):

```bash
bash scripts/sync-headers.sh
bash scripts/check-headers-in-sync.sh  # verify
```

## License

GPL-3 — see [R/DESCRIPTION](R/DESCRIPTION).
