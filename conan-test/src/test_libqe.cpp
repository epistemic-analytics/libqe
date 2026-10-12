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
    auto pairs = qe::connection_names(codes);
    assert(pairs.size() == 3);
    assert(pairs[0] == "A & B");
    assert(pairs[1] == "A & C");
    assert(pairs[2] == "B & C");

    // ── normalization ────────────────────────────────────────────────────────
    arma::mat eye3 = arma::eye(3, 3);
    arma::mat normed = qe::normalize_networks(eye3);
    for (arma::uword r = 0; r < normed.n_rows; ++r)
        assert(std::abs(arma::norm(normed.row(r)) - 1.0) < 1e-10);

    // ── modeling ─────────────────────────────────────────────────────────────
    arma::mat data = {{1, 2}, {3, 4}, {5, 6}};
    arma::mat centered = qe::center_points(data);
    arma::rowvec col_means = arma::mean(centered, 0);
    assert(std::abs(col_means(0)) < 1e-10);
    assert(std::abs(col_means(1)) < 1e-10);

    arma::mat ci = qe::mean_ci(data, 0.95);
    assert(ci.n_rows == 2 && ci.n_cols == 3);   // per dim: mean, lower, upper
    assert(std::abs(ci(0, 0) - 3.0) < 1e-10);

    return 0;
}
