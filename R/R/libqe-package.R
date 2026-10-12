#' libqe: Shared C++ Core for Quantitative Ethnography Packages
#'
#' Header-only C++ library providing shared computational primitives for
#' rENA, tma, and other Quantitative Ethnography packages. Exposes shared
#' computational modules via Rcpp-wrapped functions:
#'
#' \describe{
#'   \item{Adjacency}{Upper-triangle index pairs and vector/matrix conversions.
#'     See \code{\link{connection_indices}}, \code{\link{network_to_vector}},
#'     \code{\link{code_connections}}, and \code{\link{fold_directed_network}}.}
#'   \item{Normalization}{Row-wise L2 sphere norm and max-norm scaling.
#'     See \code{\link{normalize_networks}} and \code{\link{scale_networks}}.}
#'   \item{Statistics}{Column centering, confidence intervals and two-group
#'     tests. See \code{\link{center_points}}, \code{\link{mean_ci}},
#'     \code{\link{outlier_ci}}, and \code{\link{group_stats}}.}
#'   \item{Door}{Temporal pooling transforms for lookback and EMA-smoothed
#'     event streams. See \code{\link{door_lookback}} and
#'     \code{\link{door_ema}}.}
#'   \item{Trajectory}{Polynomial trajectory fitting, evaluation, derivative,
#'     and distance utilities. See \code{\link{fit_trajectory_poly}},
#'     \code{\link{eval_trajectory_curve}}, and
#'     \code{\link{integrated_trajectory_distance}}.}
#' }
#'
#' The ENA model code (rotations, node positions, CCD) is in rENA and all
#' accumulation is in tma.
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
