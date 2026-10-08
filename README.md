# libqe

Header-only C++ library providing shared computational primitives for
[Quantitative Ethnography](https://www.quantitativeethnography.org/) packages.
`libqe` is the common core consumed by [rENA](https://gitlab.com/epistemic-analytics/qe-packages/rENA)
and [tma](https://gitlab.com/epistemic-analytics/qe-packages/tma); it ships as
an R package (via `LinkingTo`), a Python extension (nanobind), a Julia package
(CxxWrap.jl), a WebAssembly/npm package (Emscripten), and a Conan recipe for
downstream C++ consumers.

## Modules

The headers are being split into three layers. `libqe` keeps the generic
numerics; the ENA model code (`libena`) is moving to rENA and all accumulation
(`libtma`) to tma. Until then all three trees ship from this repo.

**libqe** — `include/libqe/`

| Module | Header | Contents |
|--------|--------|----------|
| **adjacency** | `adjacency.hpp` | Upper-triangle index pairs, vector/matrix ↔ upper-tri conversions, code-name pair strings |
| **normalization** | `normalization.hpp` | Row-wise L2 sphere norm, max-norm (skip-sphere) scaling |
| **stats** | `stats.hpp` | Column-mean centering, normal/t quantiles, group confidence interval (t-based, matches rENA), outlier interval (IQR-based, matches rENA), two-group statistics (Welch t-test, Wilcoxon rank-sum) |
| **linear algebra** | `linalg_fallback.hpp`, `lasso.hpp` | LAPACK-free eigen/QR/SVD/SPD-solve fallbacks; coordinate-descent lasso |
| **door** | `door.hpp` | Lookback and EMA temporal pooling kernels for trajectory/model workflows |
| **trajectory** | `trajectory.hpp`, `trajectory_distance.hpp`, `trajectory_following.hpp` | R-compatible polynomial trajectory fitting, curve evaluation, derivatives, integrated distances, and lag/following metrics |
| **stability** | `stability.hpp` | Distance-distance correlation for stability comparisons |
| **validation** | `validate.hpp` | Shared input checks (`require_finite`, `require_dims`) |
| **binding helpers** | `bind/nanobind.hpp`, `bind/emscripten.hpp`, `bind/cxxwrap.hpp` | Armadillo ↔ numpy / JavaScript / Julia array conversion for the language bindings; need the respective binding library, so not included by `libqe.hpp` |

**libena** — `include/libena/` (moving to rENA)

| Module | Header | Contents |
|--------|--------|----------|
| **rotation** | `rotation.hpp`, `generalized_rotation.hpp` | ENA SVD, means rotation, generalized means rotation (GMR) |
| **positions** | `positions.hpp` | Least-squares node positions (undirected, directed), points-to-centroids correlation with CI |
| **ccd** | `ccd.hpp` | Moving-window size estimate from cross-covariance decay |

**libtma** — `include/libtma/` (moving to tma)

| Module | Header | Contents |
|--------|--------|----------|
| **accumulation** | `accumulation.hpp` | Connection-matrix kernel, stanza-window accumulation (rENA), ground/response accumulation with decay, tensor-based multi-modal accumulation, weight models, rolling window sum, per-row co-occurrence |

Each layer has an umbrella header: `<libena/libena.hpp>` and
`<libtma/libtma.hpp>` include `<libqe/libqe.hpp>` plus their own modules.
For now `<libqe/libqe.hpp>` still includes the libena and libtma headers too,
so existing code keeps compiling; that goes away in libqe 0.2.0.

## Repository layout

```
libqe/
├── include/            ← canonical C++ headers (single source of truth)
│   ├── libqe/          ← generic numerics
│   │   ├── libqe.hpp
│   │   ├── adjacency.hpp, normalization.hpp, stats.hpp
│   │   ├── linalg_fallback.hpp, lasso.hpp
│   │   ├── door.hpp, stability.hpp
│   │   ├── trajectory.hpp, trajectory_distance.hpp, trajectory_following.hpp
│   │   ├── validate.hpp
│   │   └── bind/       ← nanobind / Emscripten / CxxWrap array helpers
│   ├── libena/         ← ENA model code (moving to rENA)
│   │   ├── libena.hpp
│   │   ├── rotation.hpp, generalized_rotation.hpp
│   │   └── positions.hpp, ccd.hpp
│   └── libtma/         ← accumulation (moving to tma)
│       ├── libtma.hpp
│       └── accumulation.hpp
├── R/                  ← R package (Rcpp wrappers + LinkingTo mechanism)
│   ├── DESCRIPTION
│   ├── configure       ← copies headers into inst/include/ at install time
│   ├── src/
│   │   └── libqe_rcpp.cpp
│   └── tests/testthat/
├── python/             ← qe-lib Python extension, `import qe` (nanobind + CMake)
│   ├── src/qe.cpp
│   ├── tests/
│   └── pyproject.toml
├── julia/              ← LibQE.jl Julia bindings (CxxWrap.jl + CMake)
│   ├── CMakeLists.txt
│   ├── src/libqe_julia.cpp
│   └── LibQE/          ← the Julia package itself
│       ├── Project.toml
│       ├── src/LibQE.jl
│       └── test/runtests.jl
├── wasm/               ← @qe-libs/libqe-wasm npm package (Emscripten + Embind)
│   ├── CMakeLists.txt
│   ├── conanfile.py
│   ├── src/libqe_wasm.cpp
│   ├── profiles/wasm   ← Conan cross-compilation profile for Emscripten
│   ├── js/index.js
│   └── test/basic.test.js
├── conanfile.py        ← Conan recipe (header-only, exports include/{libqe,libena,libtma}/)
├── conan-test/         ← Conan test_package consumer
└── scripts/
    ├── sync-headers.sh          ← copy include/ → R/inst/include/ and python/include/
    └── check-headers-in-sync.sh ← verify the two trees match
```

Edit headers under `include/` only. The R `configure` script and
`scripts/sync-headers.sh` propagate them.

## Installing the R package

```r
# From the QE package repository
install.packages("libqe", repos = c("https://cran.qe-libs.org", "https://cran.rstudio.com"))

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

## Installing the WASM package

```bash
# one-time: tell npm where to find the @qe-libs scope
echo "@qe-libs:registry=https://gitlab.com/api/v4/projects/22522458/packages/npm/" >> ~/.npmrc

npm install @qe-libs/libqe-wasm
```

```js
import loadLibQE from '@qe-libs/libqe-wasm';
const qe = await loadLibQE();
qe.connection_names(['A', 'B', 'C']);  // → ['A & B', 'A & C', 'B & C']
```

See [wasm/README.md](wasm/README.md) for the full API reference and build-from-source instructions.

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

**WASM**
```bash
cd wasm && npm ci && npm test   # requires dist/ — run build first
```

**Conan**
```bash
conan create . --test-folder=conan-test --build=missing
```

## Header sync

The R package's `configure` script copies `include/<lib>/*.hpp` into
`R/inst/include/<lib>/` for `libqe`, `libena` and `libtma` at install time,
clearing previously copied headers first. Only top-level headers are copied —
`bind/` is for the language bindings and never reaches R. To sync manually
(e.g. before `R CMD build`, or before building a Python sdist, which also
gets `bind/`):

```bash
bash scripts/sync-headers.sh
bash scripts/check-headers-in-sync.sh  # verify
```

## Related projects

| Package | Language | Description |
|---------|----------|-------------|
| [rENA](https://gitlab.com/epistemic-analytics/qe-packages/rENA) | R | Epistemic Network Analysis |
| [tma](https://gitlab.com/epistemic-analytics/qe-packages/tma) | R | Temporal / multi-modal accumulation |
| [@qe-libs/libqe-wasm](wasm/) | JS/WASM | Browser and Node.js bindings (lives in `wasm/`) |

## License

GPL-3 — see [R/DESCRIPTION](R/DESCRIPTION).
