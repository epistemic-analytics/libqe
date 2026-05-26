# libqe

Header-only C++ library providing shared computational primitives for
[Quantitative Ethnography](https://www.quantitativeethnography.org/) packages.
`libqe` is the common core consumed by [rENA](https://gitlab.com/epistemic-analytics/qe-packages/rENA)
and [tma](https://gitlab.com/epistemic-analytics/qe-packages/tma); it ships as
an R package (via `LinkingTo`), a Python extension (nanobind), a Julia package
(CxxWrap.jl), and a Conan recipe for downstream C++ consumers.

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
├── julia/              ← LibQE.jl Julia bindings (CxxWrap.jl + CMake)
│   ├── CMakeLists.txt
│   ├── src/libqe_julia.cpp
│   └── LibQE/          ← the Julia package itself
│       ├── Project.toml
│       ├── src/LibQE.jl
│       └── test/runtests.jl
├── conanfile.py        ← Conan recipe (header-only, exports include/libqe/)
├── conan-test/         ← Conan test_package consumer
└── scripts/
    ├── sync-headers.sh          ← copy include/ → R/inst/include/
    └── check-headers-in-sync.sh ← verify the two trees match
```

Edit headers in `include/libqe/` only. The R `configure` script and
`scripts/sync-headers.sh` propagate them.

## Installing the R package

```r
# From the QE package repository
install.packages("libqe", repos = "https://rena.qe-libs.org")

# From source (within the repo)
R CMD INSTALL R/
```

The R package also ships the `libqe` headers for downstream packages via
`LinkingTo`. See [R/README.md](R/README.md) for the full function reference
and `LinkingTo` usage.

## Installing the Python package

```bash
cd python
pip install -e ".[dev]"
```

See [python/README.md](python/README.md) for build requirements and the full
API reference.

## Installing the Julia package

```julia
# one-time: install CxxWrap so CMake can find libcxxwrap-julia
using Pkg; Pkg.add("CxxWrap")
```

```bash
cd julia
cmake -B build -DCMAKE_BUILD_TYPE=Release .
cmake --build build
cmake --install build      # copies libqe_julia.{so,dylib} into LibQE/lib/
```

```julia
using Pkg; Pkg.develop(path="julia/LibQE")
using LibQE
```

The CMake build picks up the canonical headers from `include/` when run from
the repo; in CI / out-of-tree builds it falls back to a `find_package(libqe)`
provided by Conan. See [julia/README.md](julia/README.md) for build
prerequisites, platform notes, and the full API reference.

## Using libqe via Conan (C++ consumers)

`libqe` ships a Conan recipe at the repo root. Header-only, with `armadillo`
as a transitive `requires`.

```bash
# Build & cache locally (from the repo root)
conan create . --build=missing

# Run the bundled test_package
conan create . --test-folder=conan-test --build=missing
```

In a downstream CMake project:

```python
# conanfile.py / conanfile.txt
[requires]
libqe/<version>
```

```cmake
find_package(libqe REQUIRED)
target_link_libraries(my_app PRIVATE libqe::libqe)
```

The version is sourced from `R/DESCRIPTION` (or `CI_COMMIT_TAG` on tagged
releases) — there is a single place to bump it.

## WebAssembly bindings

The WASM / npm distribution lives in a separate repository:
[**libqe-wasm**](https://gitlab.com/epistemic-analytics/qe-packages/libqe-wasm).
It exposes the full libqe API as a zero-dependency ES module usable in browsers
and Node.js, built with Emscripten and published to npm as
`@qe-libs/libqe-wasm`.

```js
import loadLibQE from '@qe-libs/libqe-wasm';
const qe = await loadLibQE();
qe.svector_to_upper_tri(['A', 'B', 'C']);  // → ['A & B', 'A & C', 'B & C']
```

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

**Julia**
```bash
cd julia/LibQE && julia --project=. -e 'using Pkg; Pkg.test()'
```

**Conan**
```bash
conan create . --test-folder=conan-test --build=missing
```

## Header sync

The R package's `configure` script copies `include/libqe/*.hpp` into
`R/inst/include/libqe/` at install time.  To sync manually (e.g. before
`R CMD build`):

```bash
bash scripts/sync-headers.sh
bash scripts/check-headers-in-sync.sh  # verify
```

## Related projects

| Package | Language | Description |
|---------|----------|-------------|
| [rENA](https://gitlab.com/epistemic-analytics/qe-packages/rENA) | R | Epistemic Network Analysis |
| [tma](https://gitlab.com/epistemic-analytics/qe-packages/tma) | R | Temporal / multi-modal accumulation |
| [libqe-wasm](https://gitlab.com/epistemic-analytics/qe-packages/libqe-wasm) | JS/WASM | Browser and Node.js bindings (`@qe-libs/libqe-wasm`) |

## License

GPL-3 — see [R/DESCRIPTION](R/DESCRIPTION).
