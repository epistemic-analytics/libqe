#ifndef LIBQE_MODELING_HPP
#define LIBQE_MODELING_HPP

#include <armadillo>
#include <cmath>

namespace qe {

// ---------------------------------------------------------------------------
// Return type for node-position solvers
// ---------------------------------------------------------------------------

struct NodePositions {
    arma::mat nodes;      // n_codes x n_dims
    arma::mat centroids;  // n_units x n_dims
    arma::mat weights;    // n_units x n_codes (half-edge weight per node)
    arma::mat points;     // n_units x n_dims (input rotated points, echoed back)
};

// ---------------------------------------------------------------------------
// Centering
// ---------------------------------------------------------------------------

// Subtract column means (center-to-origin).
// Equivalent to center_data_c() in rENA/ena.cpp.
inline arma::mat center_data(arma::mat values) {
    return values.each_row() - arma::mean(values);
}

// ---------------------------------------------------------------------------
// Normal quantile (probit) — pure C++, no R dependency
// Rational approximation by Peter Acklam; max abs error < 1.15e-9.
// Used internally by ena_correlation() so we don't need Rcpp::qnorm.
// ---------------------------------------------------------------------------

inline double normal_quantile(double p) {
    static const double a[] = {
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00
    };
    static const double b[] = {
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01
    };
    static const double c[] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00
    };
    static const double d[] = {
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00
    };
    const double p_low  = 0.02425;
    const double p_high = 1.0 - p_low;
    double q, r;
    if (p < p_low) {
        q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= p_high) {
        q = p - 0.5; r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
}

// ---------------------------------------------------------------------------
// Correlation
// ---------------------------------------------------------------------------

// Pearson correlation + 95% (or custom) CI between ENA points and centroids.
// Returns n_dims x 3 matrix: [r, ci_lower, ci_upper].
// Equivalent to ena_correlation() in rENA/ena.cpp, but uses pure-C++ qnorm
// instead of Rcpp::qnorm so it works outside R.
inline arma::mat ena_correlation(arma::mat points, arma::mat centroids,
                                  double conf_level = 0.95) {
    int n = points.n_rows;
    int n_pairs = (n * (n - 1)) / 2;

    arma::umat idx1(1, n_pairs), idx2(1, n_pairs);
    int col = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) { idx1[col] = i; idx2[col] = j; col++; }

    arma::mat pts_diff = points.rows(idx1.row(0))    - points.rows(idx2.row(0));
    arma::mat cts_diff = centroids.rows(idx1.row(0)) - centroids.rows(idx2.row(0));
    arma::mat cr       = arma::cor(pts_diff, cts_diff);

    double qq = normal_quantile((1.0 + conf_level) / 2.0);
    arma::mat out(points.n_cols, 3);
    for (arma::uword i = 0; i < points.n_cols; i++) {
        double r     = cr(i, i);
        double z     = std::atanh(r);
        double sigma = 1.0 / std::sqrt(static_cast<double>(n_pairs) - 3.0);
        out(i, 0) = r;
        out(i, 1) = std::tanh(z - sigma * qq);
        out(i, 2) = std::tanh(z + sigma * qq);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Node-position solvers
// ---------------------------------------------------------------------------

// Multiobjective least-squares node positions for undirected ENA.
// Half of each line weight is distributed to each of its two endpoint nodes,
// then an overdetermined system is solved per dimension.
// Equivalent to lws_lsq_positions() in rENA/ena.cpp.
inline NodePositions lws_lsq_positions(arma::mat adj_mats, arma::mat t,
                                        int num_dims) {
    int tri_size  = adj_mats.n_cols;
    int num_nodes = static_cast<int>(
        std::pow(std::ceil(std::sqrt(static_cast<double>(2 * tri_size))), 2.0)
    ) - (2 * tri_size);
    int row_count = adj_mats.n_rows;

    arma::mat weights(row_count, num_nodes, arma::fill::zeros);
    for (int k = 0; k < row_count; k++) {
        arma::rowvec curr = adj_mats.row(k);
        int z = 0;
        for (int x = 0; x < num_nodes - 1; x++) {
            for (int y = 0; y <= x; y++) {
                weights(k, x + 1) += 0.5 * curr[z];
                weights(k, y)     += 0.5 * curr[z];
                z++;
            }
        }
    }
    for (int k = 0; k < row_count; k++) {
        double len = arma::accu(arma::abs(weights.row(k)));
        if (len < 0.0001) len = 0.0001;
        weights.row(k) /= len;
    }

    arma::mat ssX(num_dims, num_nodes, arma::fill::zeros);
    arma::mat ssA = weights.t() * weights;
    for (int i = 0; i < num_dims; i++) {
        arma::mat ssb = weights.t() * t.col(i);
        ssX.row(i) = arma::solve(ssA, ssb, arma::solve_opts::equilibrate).t();
    }

    NodePositions r;
    r.nodes     = ssX.t();
    r.centroids = (ssX * weights.t()).t();
    r.weights   = weights;
    r.points    = t;
    return r;
}

// Least-squares node positions for directed (ordered) ENA.
// Equivalent to directed_node_positions() in rENA/ena.cpp.
inline NodePositions directed_node_positions(arma::mat line_weights,
                                              arma::mat points, int num_dims) {
    int num_nodes = static_cast<int>(
        std::ceil(std::sqrt(static_cast<double>(line_weights.n_cols)))
    );
    int row_count = line_weights.n_rows;

    arma::mat nw(row_count, num_nodes, arma::fill::zeros);
    for (int k = 0; k < row_count; k++) {
        arma::mat curr = line_weights.row(k);
        int z = 0;
        for (int x = 0; x < num_nodes; x++)
            for (int y = 0; y < num_nodes; y++) {
                nw(k, x) += curr[z]; nw(k, y) += curr[z]; z++;
            }
    }
    for (int k = 0; k < row_count; k++) {
        double len = arma::accu(arma::abs(nw.row(k)));
        if (len < 0.0001) len = 0.0001;
        nw.row(k) /= len;
    }

    arma::mat ssX(num_dims, num_nodes, arma::fill::zeros);
    arma::mat ssA = nw.t() * nw;
    for (int i = 0; i < num_dims; i++) {
        arma::mat ssb = nw.t() * points.col(i);
        ssX.row(i) = arma::solve(ssA, ssb, arma::solve_opts::equilibrate).t();
    }

    NodePositions r;
    r.nodes     = ssX.t();
    r.centroids = (ssX * nw.t()).t();
    r.weights   = nw;
    r.points    = points;
    return r;
}

// Directed node positions with paired ground+response rows combined before
// solving — used for directed ENA where each unit contributes a ground row
// and a response row that should be averaged together.
// Equivalent to directed_node_positions_with_ground_response_added() in rENA/ena.cpp.
inline NodePositions directed_node_positions_ground_response(
    arma::mat line_weights, arma::mat points, int num_dims) {

    int num_nodes = static_cast<int>(
        std::ceil(std::sqrt(static_cast<double>(line_weights.n_cols)))
    );
    int row_count = line_weights.n_rows;

    arma::mat nw(row_count, num_nodes, arma::fill::zeros);
    for (int k = 0; k < row_count; k++) {
        arma::mat curr = line_weights.row(k);
        int z = 0;
        for (int x = 0; x < num_nodes; x++)
            for (int y = 0; y < num_nodes; y++) {
                nw(k, x) += curr[z]; nw(k, y) += curr[z]; z++;
            }
    }
    for (int k = 0; k < row_count; k++) {
        double len = arma::accu(arma::abs(nw.row(k)));
        if (len < 0.0001) len = 0.0001;
        nw.row(k) /= len;
    }

    // Combine paired rows (ground at k, response at k+1)
    arma::mat nw_added(row_count / 2, num_nodes, arma::fill::zeros);
    arma::mat pts_added(row_count / 2, num_dims, arma::fill::zeros);
    for (int k = 0; k < row_count; k += 2) {
        nw_added.row(k / 2)  = nw.row(k)     + nw.row(k + 1);
        pts_added.row(k / 2) = points.row(k) + points.row(k + 1);
    }

    arma::mat ssX(num_dims, num_nodes, arma::fill::zeros);
    arma::mat ssA = nw_added.t() * nw_added;
    for (int i = 0; i < num_dims; i++) {
        arma::mat ssb = nw_added.t() * pts_added.col(i);
        ssX.row(i) = arma::solve(ssA, ssb, arma::solve_opts::equilibrate).t();
    }

    NodePositions r;
    r.nodes     = ssX.t();
    r.centroids = (ssX * nw.t()).t();
    r.weights   = nw;
    r.points    = points;
    return r;
}

} // namespace qe

#endif // LIBQE_MODELING_HPP
