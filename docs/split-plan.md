# Splitting libqe into libqe, libena and libtma

Status: **phases 1–3 done; 4a, 4b and 5 on the `develop` branches of rENA, tma and ona — pushed, all CI green against each other; held from `main` until released in the order below; then phase 6** (updated 2026-10-08).

## Goal

Return libqe to its original scope: shared, generic numerics. ENA model code
moves to **libena** in the rENA repo; **all** accumulation moves to **libtma**
in the tma repo. Every binding (R, Python, JS/WASM, Julia) shares code through
the C++ headers.

ENA and tma code crept into libqe to keep compilation out of rENA (rENA is
`NeedsCompilation: no` and calls libqe's compiled R exports). The split undoes
that: rENA and tma become compiled packages with `LinkingTo: libqe`.

## End state

```
libqe   (libqe repo)   generic numerics + binding helpers
  ▲  ▲
  │  └── libtma (tma repo)   ALL accumulation
  └───── libena (rENA repo)  rotation, node positions, ccd — no accumulation

rENA ──R-level Imports──► tma
```

- All three are header-only and published to Conan through cranqe, for
  symmetry, even though each repo's own bindings find its headers in-tree.
  libena's bindings take libqe and libtma from Conan.
- libena has no separate R package: rENA is its R face. Likewise tma for libtma.
- rENA uses tma at the **R level only** (`Imports: tma`), not C++ `LinkingTo`.
- `namespace qe` is kept throughout the move. Renaming to `qe::ena` /
  `qe::tma` is an optional, separate step (phase 7), so moving and renaming
  never happen in the same change.

## Function placement

Where an R function has a different name, the C++ name is shown with the R
name in brackets. R exports follow their function. † marks dead-code
candidates for phase 7.

### libqe (this repo)

| Header | Public functions | Types and internal helpers |
|---|---|---|
| `adjacency.hpp` | `choose_two`, `connection_indices`, `code_connections`, `fold_directed_network`, `connection_names`, `network_to_vector`† | — |
| `normalization.hpp` | `normalize_networks`, `scale_networks` | — |
| `stats.hpp` (generic half of the old `modeling.hpp`) | `center_points`, `normal_quantile`, `mean_ci`, `outlier_ci`, `group_stats` | `GroupStats`; `detail::` `betacf`, `betai`, `t_quantile`, `quantile_type7`, `iqr`, `t_test_pvalue`, `WilcoxResult`, `wilcox_ranksum` |
| `linalg_fallback.hpp` | `linalg::eig_sym`, `qr_full`, `svd`, `svd_right`, `solve_spd` | `linalg::detail::mgs`, `jacobi_sweep` |
| `lasso.hpp` | `soft_threshold`, `lasso_cd`, `lasso_lambda_max`, `lasso_x1_contribution` | — |
| `door.hpp` | `lookback_block` (`door_lookback_block`), `ema_block` (`door_ema_block`), `apply_door_lookback`† (`door_lookback`), `apply_door_ema`† (`door_ema`) | — |
| `trajectory.hpp` | `fit_poly_loocv2d` (`fit_trajectory_poly`), `eval_poly_curve` (`eval_trajectory_curve`), `eval_trajectory_derivatives` | `PolyCurveFit`, `TrajectoryDerivatives`, polynomial design/basis helpers |
| `trajectory_distance.hpp` | `integrated_curve_distance` (`integrated_trajectory_distance`), `lagged_curve_distance` (`lagged_trajectory_distance`), `pairwise_trajectory_distance_matrix`† (`pairwise_trajectory_distance`) | `GaussLegendreRule`, `get_gl_rule_32` |
| `trajectory_following.hpp` | `signed_turn_lag_distance` (`signed_turn_lag`), `best_signed_turn_lag` (`sweep_signed_turn_lags`) | `SignedTurnLagResult` |
| `stability.hpp` | `dist_dist_correlation` | — |
| `validate.hpp` | `require_finite`, `require_dims` | — |
| `bind/{nanobind,emscripten,cxxwrap}.hpp` | array conversion (`to_mat`/`from_mat`, `js_to_mat`/`mat_to_js`, `view_mat`/`pack`, …); `GroupStats` packers | — |

### libena (rENA repo)

| Header | Public functions | Types and internal helpers |
|---|---|---|
| `rotation.hpp` | `ena_svd`, `deflate`, `orthogonal_svd`, `complete_rotation`, `means_rotation` | `RotationResult`, `GroupPair` |
| `generalized_rotation.hpp` | `generalized_means_rotation` | `GeneralizedRotationParams`, `GMRStepResult`, `gmr_step`, `gmr_direction`, `between_group_scatter` |
| `positions.hpp` (ENA half of the old `modeling.hpp`) | `node_positions`, `directed_node_positions` (`directed_node_positions`, `directed_node_positions_combine_pairs`), `ena_correlation` | `NodePositions` |
| `ccd.hpp` | `ccd_window` | `CCDResult` |
| `bind/*.hpp` | packers for `RotationResult`, `NodePositions`, `CCDResult` (written in phase 4) | — |
| `libena.hpp` | umbrella; includes `libqe.hpp` | — |

### libtma (tma repo)

| Header | Public functions | Types and internal helpers |
|---|---|---|
| `accumulation.hpp` (whole file) | `connection_matrix`, `accumulate_stanza`, `rolling_window_sum`, `row_connections`, `apply_tensor_unit` (`apply_tensor`, `apply_tensor_members`), `flat_index`, `finalize_row_connections`, `aggregate_row_connections` (2 overloads), `accumulate_unit`†, `accumulate_unit_with_rows`† | `TensorNetworks`, `UnitNetworks`, `WeightModel`, `weight_model_from_string`, `weight_model_from_bool`, `apply_weight_model` |
| `bind/*.hpp` | packers for `TensorNetworks`, `UnitNetworks` (written in phase 4) | — |
| `libtma.hpp` | umbrella; includes `libqe.hpp` | — |

### Cross-layer dependencies

- **libena → libqe:** `linalg::svd`, `qr_full`, `eig_sym`, `solve_spd`;
  `lasso_x1_contribution`; `normal_quantile` (in `ena_correlation`).
- **libtma → libqe:** `choose_two`, `code_connections`, `fold_directed_network`.
- **libena ↔ libtma:** none.

## Decisions

| Decision | Why |
|---|---|
| **All accumulation → libtma**, including the legacy stanza model (`accumulate_stanza`, `rolling_window_sum`, `row_connections`) | One owner for accumulation; libena keeps only model code. Accumulation questions resolve to tma. |
| **`ccd_window` → libena** | It needs full ENA models to run, so it is not an accumulation primitive. |
| **`trajectory*`, `door`, `stability` → libqe** | Generic maths, even though tma is their only library consumer. |
| **`lasso_x1_contribution` → libqe** | Generic maths, even though its only caller is libena's `gmr_step`. |
| **`modeling.hpp` splits** into libqe `stats.hpp` + libena `positions.hpp` | `ena_correlation` goes with positions. |
| **One self-contained module per product per language** | rENA compiles libqe + libtma + libena; tma compiles libqe + libtma. |
| **Bindings return plain data** (dicts / NamedTuples / JS objects), not registered classes | Modules can be loaded together without type-registration clashes. Julia's CxxWrap `add_type` structs become NamedTuples built in the Julia wrapper. |
| **Python keeps attribute access** via small Python-side result classes wrapping the dicts | `rENA/python/ena/ena.py` reads `node_positions.centroids` / `.nodes`. |
| **Shared binding helpers live in libqe** (`validate.hpp`, `bind/*.hpp`); struct packers live with the owning layer | Bindings contain only their function lists. Treat `bind/` changes as API changes: they reach every product at its next libqe bump. |
| **Struct packers are written in phase 4, not phase 2** | Python's result structs live inside `NB_MODULE` with `nb::object` fields, and plain-data returns rewrite every packer anyway. |
| **R `configure` copy and `sync-headers.sh` skip `bind/`** | R doesn't use the non-R binding helpers. |
| **NaN/Inf validation is unified later** (phase 7) | Extraction is behavior-neutral; WASM and Julia currently have no NaN checks, so adding them is a behavior change. |
| **Additive release is 0.1.9**, not 0.1.10 as first planned | `R/DESCRIPTION` was already at 0.1.9 for that dev cycle. rENA and tma pin `libqe (>= 0.1.9)`. |

## Phases

| # | Phase | Status |
|---|---|---|
| 1 | Baseline: all suites green, rotation parity fixtures recorded | ✅ Done |
| 2 | Split inside libqe (no repo moves) | ✅ Done — `6c4a20b`, `391fd0b`, `9c84a54`, `9ed8cf6` (+ fixes `c0ad788`, `7125a16`) |
| 3 | Release libqe 0.1.9 (additive) | ✅ Done — tag `v0.1.9` at `740dba2`; R, Python, Conan, npm, Julia published |
| 4a | libena → rENA | ✅ Released 2026-10-09: rENA `v0.4.13`, qe-ena 0.1.6, rena-wasm 0.1.12, ENA.jl 0.1.0, libena 0.4.13 |
| 4b | libtma → tma | ✅ Released 2026-10-09: tma `v0.3.6`, qe-tma 0.1.3, tma-wasm 0.1.0, TMA.jl 0.1.0, libtma 0.3.6 |
| 5 | Switch rENA and ona callers to `tma::` | ✅ Released with rENA 0.4.13 and ona 0.1.3 (2026-10-12) |
| 6 | Release libqe 0.2.0 (drop ENA and accumulation code) | 🟡 Built on libqe `main`, not pushed; see "Phase 6 as built" |
| 7 | Cleanup | ⬜ |

### 1. Baseline
R, Python, Julia, WASM and Conan suites green on `main`; the rotation fixtures
in `R/tests/testthat/fixtures/rotation` are the parity reference.

### 2. Split inside the libqe repo
- `include/libena/` and `include/libtma/` with umbrella headers; whole-file
  moves via `git mv` so history follows; `modeling.hpp` split in its own commit.
- Shared conversion/validation helpers extracted into `libqe/validate.hpp` and
  `libqe/bind/*.hpp` with no behavior change.
- `libqe.hpp` still includes all three layers as a compatibility shim, so the
  bindings compile unchanged.
- Exit criterion: every suite passes with no test edits.

### 3. Release libqe 0.1.9
Additive only; existing consumers keep working.

### 4. Extract (4a and 4b are independent)
- **4a, rENA:** take `include/libena/`, its Rcpp wrappers from
  `R/src/libqe_rcpp.cpp`, the rotation fixtures and parity tests. Add `src/`,
  `LinkingTo: libqe, Rcpp, RcppArmadillo`, `NeedsCompilation: yes` and a
  Makevars. Replace `libqe::` calls in rENA's R code with local Rcpp exports,
  keeping R function names. Python, WASM and Julia modules build on libqe's
  `bind/` helpers; write libena's packers once, in plain-data form.
- **4b, tma:** take `include/libtma/`, its wrappers and tests. Add `src/` and
  `LinkingTo`; export the stanza, `rolling_window_sum` and `row_connections`
  functions in R. Python, WASM and Julia modules use the `bind/` helpers; write
  libtma's packers. Move `tma/python` off the stale `pylibqe` import.
- **Both:** a cranqe `packages.yml` Conan entry for each; per-platform R binary
  builds in cranqe (both packages become compiled).
- Exit criterion: rENA's and tma's existing test suites pass unchanged.

### 5. Switch the callers
rENA's `ref_window_df`, `rows_to_co_occurrences` and `ref_window_lag`, and
ona's `accumulate_stanza(ordered = TRUE)`, move to `tma::`. ona adds
`Imports: tma`. Needs the tma release from 4b.

### 6. Release libqe 0.2.0
Remove the ENA and accumulation headers, wrappers, bindings and the
`libqe.hpp` shim entries. Needs 4a, 4b and 5 shipped. rENA and tma keep
`libqe (>= 0.1.9)`: R's DESCRIPTION can't express an upper bound alongside
it; qe-ena and qe-tma allow `qe-lib<0.3`.

### 7. Cleanup (separate, later)
- Unify NaN/Inf validation across languages through `validate.hpp`
  (behavior change for WASM and Julia).
- Fold `row_connections` into the stanza function.
- Decide the † dead-code candidates. They are exported API, so removal is a
  breaking change; check every repo first (the consumer sweep was grep-based).
- Retire the stale `libqe-wasm` repo and the `pylibqe` references in ena-mcp.
- Optionally rename to `qe::ena` / `qe::tma`.

## Phase 4a as built

- **R:** `inst/include/libena/` (canonical), wrappers in `src/libena_rcpp.cpp`
  (internal; `ena_correlation_c` / `directed_node_positions_c` because rENA
  exports R functions with the plain names). `-I../inst/include` is in
  `PKG_CPPFLAGS` so rENA's libena wins over libqe's transitional copy.
- **Headers for the other bindings:** `scripts/sync-headers.sh` vendors libena
  and libqe (from Conan, pinned `LIBQE_VERSION`, default 0.1.9) into
  `python/include`, `wasm/include`, `julia/include`. Bump it with
  DESCRIPTION's `LinkingTo: libqe` floor.
- **Python:** qe-ena compiles `ena._libena` (nanobind); `ena.libena` wraps the
  dicts in NamedTuples. Still depends on qe-lib for generic + accumulation.
- **WASM:** rena-wasm builds `dist/libena.{js,wasm}`; `src/index.js` proxies the
  11 libena names to it, the rest to libqe-wasm.
- **Julia:** new package **ENA.jl** (`julia/`, `using ENA`); results are plain
  tuples (no `add_type`), so it loads alongside LibQE.jl. Adds `ccd_window`.
- **Conan:** `conan/conanfile.py` — recipe `libena`, version from DESCRIPTION,
  requires `libqe/[>=0.1.9 <0.3]`.
- **cranqe:** macOS Python builds run `sync-headers.sh`; Julia builds too;
  Julia packages register under `Project.toml`'s name (libqe's entry moves
  to `LibQE` on its next release); Conan uploads under the recipe name.
- **Bugs found in LibQE.jl (fixed in ENA.jl, also fixed in LibQE.jl after 0.1.9):**
  `ena_correlation` reshapes to `n_units × 3` instead of `n_dims × 3`;
  `directed_node_positions_combine_pairs` used undefined `pr`/`pc`. Fixed in
  LibQE.jl too (`f8d9db5`).

## Phase 4b as built

Decided with the user: full parity in tma (Python, WASM, Julia, Conan), and
pyTMA becomes **qe-tma** (`import tma`). Julia package name **TMA**.

- **R:** `inst/include/libtma/` (canonical; `accumulation.hpp` gained the
  `<algorithm>` include it relied on implicitly), wrappers in
  `src/libtma_rcpp.cpp`, **exported** (tma is where rENA/ona get accumulation in
  phase 5). tma 0.3.6.
- **Python:** qe-tma 0.1.3 compiles `tma._libtma`; `tma.libtma` wraps dicts in
  `UnitNetworks` / `TensorNetworks`. Node positions from qe-ena (`ena.libena`,
  >= 0.1.6 — on rENA `develop` only, so tma's `test:python` builds qe-ena from
  rENA `develop` via `QE_ENA_REF` until it is published).
- **WASM:** new `@qe-libs/tma-wasm` 0.1.0 (`dist/libtma`), `loadTMA()`.
- **Julia:** new **TMA.jl** 0.1.0 (`using TMA`); adds the weight models;
  `accumulate_unit_with_rows` no longer reuses the ENA struct.
- **Conan:** `conan/conanfile.py` — recipe `libtma`, version from DESCRIPTION.
- **cranqe:** tma uses per-target release tags; Julia builds skip a missing
  subdir (as the other languages do).
- **Parity:** every binding has tests checking identical results against
  libqe's copy (skipped once libqe 0.2.0 removes it).

## Phase 5 as built

Decided with the user: rENA's non-R products **compile the libtma they use**
(qe-ena does not depend on qe-tma, which depends on qe-ena — no cycle).

- **R:** rENA's stanza wrappers call `tma::accumulate_stanza` /
  `row_connections` / `rolling_window_sum` (rENA Imports `tma (>= 0.3.6)`);
  ona's `accum_stanza_window` calls `tma::accumulate_stanza` (ona 0.1.3,
  Imports tma, no longer libqe).
- **Shared bindings:** libtma's nanobind and Embind bindings moved into tma's
  `bind/` headers as `def_<name>()` / `register_<name>()`; each module picks
  what it needs (qe-ena: `accumulate_stanza`; rena-wasm: all ten).
- **Headers:** rENA's `sync-headers.sh` vendors libtma: `LIBTMA_INCLUDE`, else
  Conan `libtma/$LIBTMA_VERSION` (0.3.6), else a `../tma` checkout, else a
  clone of `$TMA_REF` from `$TMA_GIT_URL`.
- **CI (temporary):** rENA's and ona's pipelines install tma (and ona: rENA)
  from `develop` via the job token (tma is private; its allowlist admits the
  epistemic-analytics group). Drop once released.
- **Verified locally:** rENA R 568, qe-ena 159, rena-wasm 249, ona 6 tests
  pass against tma `develop`; parity tests confirm identical results to
  libqe's copies.

### Release order
1. **tma**: ✅ merged `develop` → `main` and pushed with cranqe `fd7a5240`
   (2026-10-09; `main` had to be protected first — see open items); first
   cranqe dev build published every target. ✅ Released 2026-10-09: `v0.3.6`
   (R), `conan/v0.3.6` (libtma; needed cranqe's verify-tag-version to accept
   conanfile.py, `96ab513a`), `python/v0.1.3` (qe-tma), `wasm/v0.1.0`
   (tma-wasm), `julia/v0.1.0` (TMA.jl, registered as `TMA`). Originally:
   release `v0.3.6` **and `conan/v0.3.6`** (rENA's cranqe builds take libtma
   from Conan at exactly that version), plus qe-tma / tma-wasm / TMA.jl tags
   as wanted. qe-tma needs qe-ena >= 0.1.6, so its install works once step 2
   publishes it.
2. **rENA**: ✅ merged `develop` → `main` (2026-10-09; two CI fixes on the
   way: `sync-headers.sh` uses one Conan cache per package — libqe's package
   ships a stale `include/libtma`; libena's Conan test package uses libqe's
   fallback linear algebra — ConanCenter's prebuilt OpenBLAS hit SIGILL).
   ✅ Released `v0.4.13` (R), `python/v0.1.6` (qe-ena), `wasm/v0.1.12`
   (rena-wasm), `julia/v0.1.0` (ENA.jl, registered as `ENA`),
   `conan/v0.4.13` (libena). Windows R binaries and wheels for rENA and tma
   built on GitHub; the nightly publishes them.
3. **ona**: ✅ merged `develop` → `main` (2026-10-12; with CI installing tma
   and rENA from qe-libs.org instead of `develop`) and released `v0.1.3`
   (Imports tma, no longer libqe).
4. ✅ Removed the temporary `develop` installs from rENA's (`afa2a63`, also
   drops `sync-headers.sh`'s sibling/clone fallbacks for libtma), tma's
   (`991cbbf`) and ona's CI. The same commits lift qe-ena / qe-tma to
   `qe-lib<0.3` (qe-ena 0.1.7, qe-tma 0.1.4) and bump the other targets past
   their releases (rENA 0.4.14, rena-wasm 0.1.13, ENA.jl 0.1.1; tma 0.3.7,
   tma-wasm 0.1.1, TMA.jl 0.1.1).
5. Then phase 6 (libqe 0.2.0).

## Phase 6 as built

- **Removed:** `include/libena/`, `include/libtma/` and their includes in
  `libqe.hpp`; from every binding the ENA functions (rotations, node
  positions, `ena_correlation`, `ccd_window`) and accumulation functions,
  with their tests, R man pages, Python stubs (`qe.accumulation`,
  `qe.rotation`, `qe.ccd`; `NodePositions`, `UnitNetworks`,
  `TensorNetworks`, `RotationResult`) and README/demo entries. R exports
  drop from 47 to 25. The Conan package ships `include/libqe/` only.
- **Kept:** `qe.modeling` (center_points, mean_ci, outlier_ci, group_stats —
  qe-ena uses it), `stats.hpp`'s `normal_quantile` and the linalg/lasso
  helpers libena needs.
- **Header sync:** `sync-headers.sh`, `check-headers-in-sync.sh` and
  `R/configure` handle `libqe` only (the sync script deletes leftover
  `libena`/`libtma` copies from older checkouts).
- **Versions:** 0.2.0 in R, Python, wasm and Julia manifests.
- **Verified locally** against libqe 0.2.0: libqe R 113, Python 72, wasm 7,
  Julia, Conan test package; rENA R 568, tma R 199, ona R 1921 (its libqe
  parity test now skips — its guard needed `inherits = FALSE`, ona
  `develop`); libena and libtma Conan test packages; qe-ena 155 and qe-tma 86
  (installed `--no-deps`: both pin `qe-lib<0.2`).
- **Pins:** qe-ena 0.1.7 and qe-tma 0.1.4 allow `qe-lib<0.3` (rENA
  `afa2a63`, tma `991cbbf`); verified with a normal (dependency-resolving)
  install next to qe-lib 0.2.0. Release them before or with libqe 0.2.0 so
  pip can pair them.
- **Push order:** rENA, then tma, then libqe: tma's CI installs the newest
  qe-ena from the dev index (it needs the `<0.3` one once qe-lib 0.2.0 dev
  builds land), and both CIs install the newest qe-lib dev build.
- **After release:** bump `LIBQE_VERSION` in rENA's and tma's
  `sync-headers.sh` from 0.1.9 to 0.2.0 (its Conan package has no stale
  libena/libtma copies).

## Open items

- **tma's old perf branch and stashes — not blocking (decided 2026-10-08):**
  `feature/global-encoding-perf`'s two bug fixes are already on tma `main`;
  its perf work (global `match()` encoding, C++ fold, `apply_tensor_batch`,
  benchmarks) targets the pre-libqe `src/code.cpp` and is a reference for a
  later perf pass on libtma (phase 7). The 9 stashes (2024-07 to 2025-08) all
  predate the move to libqe. Both left as they are.
- **Before 5:** ona's unpushed commit `fd04d92` ("docs: state
  accum_stanza_window's column order correctly").
- **libqe 0.1.10 dev bump** is committed (`f6dfc5d`) and goes out with the next
  substantive push.
- **Before 4a/4b ship:** rENA and tma stay on `develop` until rENA is verified
  against the converted tma and libqe (phase 5 switches rENA/ona/rena-wasm to
  tma's accumulation); then merge both to `main` and watch the first compiled
  builds (R binaries, qe-ena wheels incl. Windows on GitHub,
  rena-wasm, ENA.jl, libena Conan); then release rENA and its bindings.

- **Protected refs:** the cranqe trigger token is a protected group variable,
  so a package's `main` and `v*` (and `*/v*`) tags must be protected or its
  trigger 404s. Done 2026-10-09: `main` in tma, ncodeR, rhoR, PRIA; tags in
  tma and ona (libqe and rENA already had both).

- **Julia packages from private repos:** the Julia registry points Pkg at the
  package's git repo, so TMA.jl (tma: GitLab and GitHub private) and LibQE.jl
  (libqe GitLab private; its GitHub mirror is public) install only for people
  with repo access. ENA.jl is fine (rENA public). Decide: make the repos (or
  a mirror) public and point Package.toml's `repo` there, or accept it.

## Issues carried into tma

- **Forward stanza windows** exist only in the stanza model; the tensor path
  is backward-only.
- **2× hazard:** `connection_matrix` / `apply_tensor_unit` with
  `ordered = false` followed by `fold_directed_network` doubles non-binary
  counts. tma's R code avoids it by always calling `apply_tensor` with
  `ordered = TRUE` and folding once in `aggregate_row_connections`; any binding
  calling the unordered C++ path directly would double.
- **ONA's ordered branch** binarizes when `binary = TRUE`, contradicting the
  `WeightModel` doc.

## Risks

- **rENA and tma become compiled.** cranqe needs working per-platform binaries
  (Linux, macOS, and Windows via GitHub) before release, or installs break for
  users without toolchains. This is the largest risk.
- **Parity drift during the move.** Mitigated by moving fixtures and tests with
  the code and keeping tests unchanged through each move.
- **Coordinated release.** Phase 6 is the one point where libqe, tma and rENA
  must all have released, in order.
- **`bind/` convention coupling.** A change to a `bind/` convention reaches every
  product at its next libqe bump.
