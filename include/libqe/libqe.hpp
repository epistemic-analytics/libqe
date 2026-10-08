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

// TRANSITIONAL: the <libena/...> and <libtma/...> includes below keep every
// existing consumer of <libqe/libqe.hpp> compiling while the ENA and
// accumulation headers move out to rENA (libena) and tma (libtma).  They are
// removed in libqe 0.2.0; new code should include <libena/libena.hpp> or
// <libtma/libtma.hpp> directly.  Include order is unchanged from before the
// split.
#include "adjacency.hpp"
#include "normalization.hpp"
#include "stats.hpp"
#include <libena/positions.hpp>
#include <libtma/accumulation.hpp>
#include "linalg_fallback.hpp"
#include <libena/rotation.hpp>
#include "lasso.hpp"
#include <libena/generalized_rotation.hpp>
#include "door.hpp"
#include "trajectory.hpp"
#include "trajectory_distance.hpp"
#include "trajectory_following.hpp"
#include "stability.hpp"
#include <libena/ccd.hpp>

#endif // LIBQE_HPP
