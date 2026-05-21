// Conan install-verification test.
// Exercises one function from each libqe module to confirm that headers are
// reachable and compile cleanly after installation.
#include <armadillo>
#include <libqe/libqe.hpp>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>

int main() {
    // ── adjacency ────────────────────────────────────────────────────────────
    std::vector<std::string> codes = {"A", "B", "C"};
    auto pairs = qe::svector_to_upper_tri(codes);
    assert(pairs.size() == 3);
    assert(pairs[0] == "A & B");
    assert(pairs[1] == "A & C");
    assert(pairs[2] == "B & C");

    // ── normalization ────────────────────────────────────────────────────────
    arma::mat eye3 = arma::eye(3, 3);
    arma::mat normed = qe::sphere_norm(eye3);
    for (arma::uword r = 0; r < normed.n_rows; ++r)
        assert(std::abs(arma::norm(normed.row(r)) - 1.0) < 1e-10);

    // ── accumulation ─────────────────────────────────────────────────────────
    arma::mat c = {{1, 1, 0}, {1, 0, 1}, {0, 1, 1}};
    arma::mat out = qe::stanza_window(c, /*back=*/2, /*forward=*/0, /*binary=*/true);
    assert(out.n_rows == 3);
    assert(out.n_cols == 3);  // choose_two(3)

    // ── modeling ─────────────────────────────────────────────────────────────
    arma::mat data = {{1, 2}, {3, 4}, {5, 6}};
    arma::mat centered = qe::center_data(data);
    arma::rowvec col_means = arma::mean(centered, 0);
    assert(std::abs(col_means(0)) < 1e-10);
    assert(std::abs(col_means(1)) < 1e-10);

    return 0;
}
