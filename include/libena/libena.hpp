/**
 * @file libena.hpp
 * @brief Master include — brings in libqe and all libena (ENA) modules.
 *
 * libena holds the Epistemic Network Analysis model code built on libqe's
 * generic numerics: rotations, node positions, and CCD window estimation.
 *
 * @code
 * #include <armadillo>
 * #include <libena/libena.hpp>
 * @endcode
 */
#ifndef LIBENA_HPP
#define LIBENA_HPP

#include <libqe/libqe.hpp>

#include "rotation.hpp"
#include "generalized_rotation.hpp"
#include "positions.hpp"
#include "ccd.hpp"

#endif // LIBENA_HPP
