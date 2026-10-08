/**
 * @file libtma.hpp
 * @brief Master include — brings in libqe and all libtma (accumulation) modules.
 *
 * libtma holds all network accumulation built on libqe's generic numerics:
 * the stanza-window model, ground/response and tensor accumulation, and the
 * per-line weight models.
 *
 * @code
 * #include <armadillo>
 * #include <libtma/libtma.hpp>
 * @endcode
 */
#ifndef LIBTMA_HPP
#define LIBTMA_HPP

#include <libqe/libqe.hpp>

#include "accumulation.hpp"

#endif // LIBTMA_HPP
