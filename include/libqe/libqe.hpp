/**
 * @file libqe.hpp
 * @brief Master include — brings in all libqe modules.
 *
 * Usage in an Rcpp package:
 * @code
 * // [[Rcpp::depends(RcppArmadillo, libqe)]]
 * #include <RcppArmadillo.h>
 * #include <libqe/libqe.hpp>
 * @endcode
 *
 * Usage in a plain CMake project (pybind11, Emscripten, etc.):
 * @code
 * #include <armadillo>
 * #include <libqe/libqe.hpp>
 * @endcode
 */
#ifndef LIBQE_HPP
#define LIBQE_HPP

#include "adjacency.hpp"
#include "normalization.hpp"
#include "modeling.hpp"
#include "accumulation.hpp"
#include "linalg_fallback.hpp"
#include "rotation.hpp"
#include "lasso.hpp"
#include "generalized_rotation.hpp"

#endif // LIBQE_HPP
