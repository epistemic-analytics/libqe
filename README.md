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

## Installing the Julia package

**Requirements:** Julia ≥ 1.9, CMake ≥ 3.18, CxxWrap 0.15, Armadillo.

```bash
# macOS — Armadillo via Homebrew (Accelerate provides BLAS automatically)
brew install armadillo

# Linux — system Armadillo + BLAS/LAPACK
apt-get install libarmadillo-dev libblas-dev liblapack-dev
```

```julia
# one-time: install CxxWrap so CMake can find libcxxwrap-julia
using Pkg; Pkg.add("CxxWrap")
```

```bash
cd julia
cmake -B build -DCMAKE_BUILD_TYPE=Release .
cmake --build build
cmake --install build      # installs libqe_julia.{so,dylib} into LibQE/lib/
```

Then load the package in Julia:

```julia
using Pkg; Pkg.develop(path="julia/LibQE")
using LibQE

LibQE.svector_to_upper_tri(["A", "B", "C"])
# → ["A & B", "A & C", "B & C"]

codes = Float64[1 1 0; 1 0 1; 0 1 1]   # 3 rows × 3 codes
LibQE.stanza_window(codes; window_back=2)
# → 3×3 Matrix{Float64}  (choose_two(3) = 3 connection columns)
```

The CMake build picks up the canonical headers from `include/` when run from
the repo; in CI / out-of-tree builds it falls back to a `find_package(libqe)`
provided by Conan.

### Julia API summary

| Function | Returns | Notes |
|----------|---------|-------|
| `svector_to_upper_tri(names)` | `Vector{String}` | `"A & B"` pair labels |
| `tri_indices(len; row=-1)` | `2 × n_pairs Matrix{Int32}` | Upper-tri (i,j) pairs |
| `vector_to_upper_tri(v)` | `Vector{Float64}` | Code vector → connection vector |
| `directed_to_upper_tri(v)` | `Vector{Float64}` | n² directed → upper-tri |
| `adjacency_matrix_to_vector(m; full=true)` | `Vector{Float64}` | Matrix → flat vector |
| `sphere_norm(m)` | `Matrix{Float64}` | Row-wise L2 normalization |
| `skip_sphere_norm(m)` | `Matrix{Float64}` | Max-norm scaling |
| `center_data(m)` | `Matrix{Float64}` | Subtract column means |
| `group_ci(pts; conf_level=0.95)` | `n_dims×3 Matrix` | `[mean, lower, upper]` — matches rENA `t.test` |
| `outlier_ci(pts; iqr_factor=1.5)` | `n_dims×2 Matrix` | `[lower, upper]` — matches rENA IQR formula |
| `ena_correlation(pts, centroids; conf_level=0.95)` | `n_units×3 Matrix` | `[r, lower, upper]` |
| `lws_lsq_positions(adj, t, dims)` | `NamedTuple` | Undirected ENA node positions |
| `directed_node_positions(lw, pts, dims)` | `NamedTuple` | Directed ENA node positions |
| `calculate_adjacency_matrix(ground, response; ...)` | `Matrix{Float64}` | Core adjacency math |
| `stanza_window(codes; window_back, window_forward, binary)` | `Matrix{Float64}` | rENA accumulation |
| `rows_to_co_occurrences(codes; binary=true)` | `Matrix{Float64}` | Per-row co-occurrence |
| `rolling_window_sum(codes; window_size=1)` | `Matrix{Float64}` | Rolling backward sum |
| `calculate_1d_index(indices, dims)` | `Int` | Column-major linear index (0-based) |
| `accumulate_unit(codes, unit_rows, decay_fn; ordered)` | `Vector{Float64}` | tma ground/response accumulation |
| `accumulate_unit_with_rows(codes, unit_rows, decay_fn; ordered)` | `NamedTuple` | As above + per-row networks |

All matrix inputs are `Matrix{Float64}` (Julia column-major = Armadillo column-major,
so inputs are zero-copy across the C++ boundary).

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
