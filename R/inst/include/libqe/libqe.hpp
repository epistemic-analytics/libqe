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

// The ENA model code (libena) lives in rENA and accumulation (libtma) in tma;
// include <libena/libena.hpp> or <libtma/libtma.hpp> for those.
#include "adjacency.hpp"
#include "normalization.hpp"
#include "stats.hpp"
#include "linalg_fallback.hpp"
#include "lasso.hpp"
#include "door.hpp"
#include "trajectory.hpp"
#include "trajectory_distance.hpp"
#include "trajectory_following.hpp"
#include "stability.hpp"
#include "validate.hpp"
// bind/*.hpp (nanobind, Emscripten, CxxWrap helpers) are deliberately not
// included: they need third-party headers and are for binding code only.

#endif // LIBQE_HPP
