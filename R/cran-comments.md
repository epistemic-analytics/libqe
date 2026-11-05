## libqe 0.2.5

First CRAN submission.

'libqe' is the shared C++ header library used (through LinkingTo) by the
quantitative ethnography packages 'rENA' and 'tma', which will be submitted
once 'libqe' is on CRAN. Every function is also exported to R through 'Rcpp'
wrappers so that it can be tested and used directly.

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.

The local macOS run also reports a compiler warning from R's own
R_ext/Boolean.h (clang: unknown warning group '-Wfixed-enum-extension');
it comes from the toolchain, not the package, and does not appear on
win-builder.

## Test environments

* local macOS (arm64), R 4.5.2 -- `R CMD check --as-cran`
* win-builder, R-devel (2026-11-04 r90836): Status: 1 NOTE (new submission)
* win-builder, R-release (4.6.2): Status: 1 NOTE (new submission)
* GitLab CI, rocker/r-ver:4.5.0 (Ubuntu) -- install and testthat suite

## Downstream dependencies

'rENA' and 'tma' (not yet on CRAN) were installed against this version
without changes.
