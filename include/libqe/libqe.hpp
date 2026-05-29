#ifndef LIBQE_HPP
#define LIBQE_HPP

// Master include — bring in all libqe modules.
//
// Usage in an Rcpp package:
//   // [[Rcpp::depends(RcppArmadillo, libqe)]]
//   #include <RcppArmadillo.h>
//   #include <libqe/libqe.hpp>
//
// Usage in a pybind11 module (CMake):
//   #include <armadillo>
//   #include <libqe/libqe.hpp>

#include "adjacency.hpp"
#include "normalization.hpp"
#include "modeling.hpp"
#include "accumulation.hpp"
#include "linalg_fallback.hpp"
#include "rotation.hpp"
#include "lasso.hpp"
#include "generalized_rotation.hpp"

#endif // LIBQE_HPP
