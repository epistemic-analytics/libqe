# pylibqe — Python bindings for libqe

pybind11 bindings are planned here once the C++ headers are validated via the
R package test suite.  The binding layer will expose the same four modules
(adjacency, normalization, modeling, accumulation) as a `pylibqe` Python
package built with CMake + scikit-build-core.

## Planned build dependencies
- armadillo (system or conda-forge)
- pybind11
- cmake >= 3.18
