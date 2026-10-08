# Splitting libqe into libqe, libena and libtma

Status: **phases 1–3 done; phase 4 next** (updated 2026-10-08).

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
| 4a | libena → rENA | ⬜ Next |
| 4b | libtma → tma | ⬜ Blocked (see open items) |
| 5 | Switch rENA and ona callers to `tma::` | ⬜ |
| 6 | Release libqe 0.2.0 (drop ENA and accumulation code) | ⬜ |
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
`libqe.hpp` shim entries. Needs 4a, 4b and 5 shipped. rENA and tma pin
`libqe (>= 0.1.9, < 0.3)`.

### 7. Cleanup (separate, later)
- Unify NaN/Inf validation across languages through `validate.hpp`
  (behavior change for WASM and Julia).
- Fold `row_connections` into the stanza function.
- Decide the † dead-code candidates. They are exported API, so removal is a
  breaking change; check every repo first (the consumer sweep was grep-based).
- Retire the stale `libqe-wasm` repo and the `pylibqe` references in ena-mcp.
- Optionally rename to `qe::ena` / `qe::tma`.

## Open items

- **Before 4b:** tma's unmerged `feature/global-encoding-perf` branch (encoding
  changes, a C++ fold, `apply_tensor_batch`) rewrites `accum_multidim_c.R`,
  which 4b also touches — merge, rebase or abandon it first. tma also has 9 old
  stashes (2024-07 to 2025-08), five on `accumulation_testing`; review or drop.
- **Before 5:** ona's unpushed commit `fd04d92` ("docs: state
  accum_stanza_window's column order correctly").
- **libqe 0.1.10 dev bump** is committed (`f6dfc5d`) and goes out with the next
  substantive push.

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
