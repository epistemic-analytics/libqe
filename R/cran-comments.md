## libqe 0.2.5

First CRAN submission.

'libqe' is the shared C++ header library used (through LinkingTo) by the
quantitative ethnography packages 'rENA' and 'tma', which will be submitted
once 'libqe' is on CRAN. Every function is also exported to R through 'Rcpp'
wrappers so that it can be tested and used directly.

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.

## Test environments

* local macOS (arm64), R 4.5.2 — `R CMD check --as-cran`
* GitLab CI, rocker/r-ver:4.5.0 (Ubuntu) — install and testthat suite

Still to run before submitting (not done locally):

* win-builder (R-devel and R-release)
* R-hub (Linux, Windows, macOS)

## Downstream dependencies

'rENA' and 'tma' (not yet on CRAN) were installed against this version
without changes.
