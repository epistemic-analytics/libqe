# Shared fixtures used across all parity tests.
# Each test file calls these helpers rather than duplicating setup.

make_simple_codes <- function() {
    matrix(c(
        1, 1, 0,
        1, 0, 1,
        0, 1, 1,
        1, 1, 1
    ), nrow = 4, byrow = TRUE)
}

# Two-unit, two-conversation dataset matching the toy data embedded in
# rENA/ena.cpp and used in its internal examples.
make_toy_data <- function() {
    list(
        units        = data.frame(Name = rep(c("J", "Z"), 3)),
        conversation = data.frame(Day = c(1, 1, 1, 2, 2, 2)),
        codes        = data.frame(
            c1 = c(1, 1, 0, 0, 1, 1),
            c2 = c(1, 0, 1, 1, 0, 0),
            c3 = c(0, 1, 0, 1, 0, 1)
        )
    )
}

# Minimal line-weight matrix + rotated points for node-position tests.
# Three codes → 3 upper-tri connections; two units.
make_lws_inputs <- function() {
    set.seed(42)
    list(
        line_weights = matrix(runif(2 * 3), nrow = 2, ncol = 3),
        points       = matrix(runif(2 * 2), nrow = 2, ncol = 2)
    )
}
