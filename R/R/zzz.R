#' @useDynLib libqe, .registration = TRUE
#' @importFrom Rcpp evalCpp
NULL

.onLoad <- function(libname, pkgname) {
    # Nothing to initialise — libqe is a header + Rcpp library.
    invisible(NULL)
}
