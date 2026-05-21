#' libqe: Shared C++ Core for Quantitative Ethnography Packages
#'
#' Header-only C++ library providing shared computational primitives for
#' rENA, tma, and other Quantitative Ethnography packages. Exposes four
#' modules via Rcpp-wrapped \code{lq_*} functions:
#'
#' \describe{
#'   \item{Adjacency}{Upper-triangle index pairs and vector/matrix conversions.
#'     See \code{\link{lq_tri_indices}}, \code{\link{lq_vector_to_upper_tri}},
#'     \code{\link{lq_svector_to_upper_tri}}.}
#'   \item{Normalization}{Row-wise L2 sphere norm and max-norm scaling.
#'     See \code{\link{lq_sphere_norm}}, \code{\link{lq_skip_sphere_norm}}.}
#'   \item{Modeling}{Column centering, ENA correlation, and least-squares node
#'     positions (undirected and directed).
#'     See \code{\link{lq_center_data}}, \code{\link{lq_ena_correlation}},
#'     \code{\link{lq_lws_lsq_positions}}, \code{\link{lq_directed_node_positions}}.}
#'   \item{Accumulation}{Stanza-window (rENA) and ground/response/tensor (tma)
#'     accumulation primitives.
#'     See \code{\link{lq_stanza_window}}, \code{\link{lq_accumulate_unit}},
#'     \code{\link{lq_apply_tensor}}.}
#' }
#'
#' \strong{Using libqe headers in your own R package:}
#'
#' Add to \file{DESCRIPTION}:
#' \preformatted{
#'   LinkingTo: Rcpp, RcppArmadillo, libqe
#' }
#'
#' Then in your \file{.cpp} source:
#' \preformatted{
#'   // [[Rcpp::depends(RcppArmadillo, libqe)]]
#'   #include <RcppArmadillo.h>
#'   #include <libqe/libqe.hpp>
#' }
#'
#' @docType package
#' @name libqe-package
#' @aliases libqe
"_PACKAGE"
