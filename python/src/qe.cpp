// qe — nanobind bindings for libqe
//
// Exposes these submodules (C++ header in parentheses):
//   qe.adjacency     — vector/matrix upper-triangle utilities (libqe/adjacency.hpp)
//   qe.normalization — row-wise normalization (libqe/normalization.hpp)
//   qe.modeling      — centering, confidence intervals, group statistics
//                      (libqe/stats.hpp)
//   qe.door          — lookback / EMA temporal pooling (libqe/door.hpp)
//   qe.trajectory    — curve fitting, derivatives, distance, following
//                      (libqe/trajectory*.hpp)
//
// The ENA model code (rotations, node positions, CCD) is in qe-ena and all
// accumulation in qe-tma (libqe 0.2.0).
//
// All matrix arguments are accepted as 2-D numpy float64 arrays (C-contiguous).
// All vector arguments are accepted as 1-D numpy float64 arrays.
// Return values are always freshly allocated numpy arrays (owned by Python).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <optional>
#include <variant>

#include <armadillo>
#include <libqe/libqe.hpp>
#include <libqe/validate.hpp>
#include <libqe/bind/nanobind.hpp>

namespace nb = nanobind;
using namespace nb::literals;

// Armadillo ↔ numpy conversion (to_mat, from_mat, …) comes from
// libqe/bind/nanobind.hpp; input validation from libqe/validate.hpp.
using namespace qe::bind::py;

// ── Module definition ─────────────────────────────────────────────────────────

NB_MODULE(_qe, m) {
    m.doc() = "qe — Python bindings for libqe numerics";

    // ── adjacency ─────────────────────────────────────────────────────────────
    auto adj = m.def_submodule("adjacency",
        "Adjacency utilities: upper-triangle conversions, index helpers");

    adj.def("choose_two", &qe::choose_two, "n"_a,
        "Return n*(n-1)/2 (number of unique pairs in a set of n elements).");

    adj.def("connection_indices", [](int len, int row) {
        return from_umat(qe::connection_indices(len, row));
    }, "len"_a, "row"_a = -1,
        "Upper-triangle index pairs for an len×len matrix.\n"
        "row=-1: 2×k array of [row_idx; col_idx]  "
        "row=0: row indices only  "
        "row=1: col indices only");

    adj.def("code_connections", [](NpVec v) {
        // wrap 1-D input as a 1×n row vector for code_connections
        arma::mat vm(1, v.shape(0));
        for (size_t j = 0; j < v.shape(0); ++j) vm(0, j) = v(j);
        return from_rowvec(qe::code_connections(vm));
    }, "v"_a,
        "Compute pairwise products v[j]*v[i] for all j < i (upper-triangle vector).");

    adj.def("fold_directed_network", [](NpVec v) {
        return from_rowvec(qe::fold_directed_network(to_vec(v)));
    }, "v"_a,
        "Fold a directed n*n flat vector to upper-triangle by summing A→B + B→A.");

    adj.def("network_to_vector", [](NpMat x, bool full) {
        return from_rowvec(qe::network_to_vector(to_mat(x), full));
    }, "x"_a, "full"_a = true,
        "Flatten an adjacency matrix to a vector.\n"
        "full=True: full n*n vector (directed)  "
        "full=False: upper-triangle only (undirected)");

    adj.def("connection_names", [](std::vector<std::string> v) {
        return qe::connection_names(v);
    }, "v"_a,
        "Return 'A & B' pair names for every upper-triangle position.\n"
        "Example: ['X','Y','Z'] → ['X & Y', 'X & Z', 'Y & Z']");

    // ── normalization ─────────────────────────────────────────────────────────
    auto nrm = m.def_submodule("normalization",
        "Row-wise L2 normalization");

    nrm.def("normalize_networks", [](NpMat mat) {
        return from_mat(qe::normalize_networks(to_mat(mat)));
    }, "m"_a,
        "Divide each row by its own L2 norm (project onto unit hypersphere). "
        "Zero rows are left unchanged.");

    nrm.def("scale_networks", [](NpMat mat) {
        return from_mat(qe::scale_networks(to_mat(mat)));
    }, "m"_a,
        "Divide all entries by the *largest* row L2 norm. "
        "Preserves relative magnitudes across rows.");

    // ── modeling ──────────────────────────────────────────────────────────────
    auto mod = m.def_submodule("modeling",
        "Centering, confidence intervals and two-group statistics");

    mod.def("center_points", [](NpMat values) {
        return from_mat(qe::center_points(to_mat(values)));
    }, "values"_a,
        "Subtract column means (center each column to zero).");

    mod.def("mean_ci", [](NpMat points, double conf_level) {
        return from_mat(qe::mean_ci(to_mat(points), conf_level));
    }, "points"_a, "conf_level"_a = 0.95,
        "t-based confidence interval for the mean of a group of ENA unit points.\n\n"
        "For each dimension computes: mean ± t(α/2, n-1) × (SD / sqrt(n))\n"
        "where α = 1 - conf_level.\n\n"
        "Parameters\n----------\n"
        "points     : ndarray (n_units × n_dims)  — one row per unit in the group\n"
        "conf_level : float  confidence level, e.g. 0.95 (default)\n\n"
        "Returns (n_dims × 3) array: columns are [mean, ci_lower, ci_upper].\n"
        "When n_units == 1 the CI bounds are ±inf.");

    mod.def("outlier_ci", [](NpMat points, double iqr_factor) {
        return from_mat(qe::outlier_ci(to_mat(points), iqr_factor));
    }, "points"_a, "iqr_factor"_a = 1.5,
        "Outlier interval based on IQR (Tukey fence) for a group of ENA unit points.\n\n"
        "For each dimension d:\n"
        "  lower[d] = -iqr_factor * IQR(points[:, d])\n"
        "  upper[d] = +iqr_factor * IQR(points[:, d])\n\n"
        "Symmetric around 0, matching rENA's formula:\n"
        "  oi = IQR(each dim) * iqr_factor\n"
        "  result = matrix([[−oi], [+oi]])\n\n"
        "IQR uses type-7 / Hyndman-Fan #7 quantile (R default,\n"
        "identical to numpy's percentile(method='linear')).\n\n"
        "Parameters\n----------\n"
        "points     : ndarray (n_units × n_dims)  — one row per unit\n"
        "iqr_factor : float  multiplier applied to IQR (default 1.5)\n\n"
        "Returns (n_dims × 2) array: columns are [lower, upper].\n"
        "All entries are NaN when n_units == 0.");

    // Python-side GroupStatsResult — mirrors qe::GroupStats
    struct PyGroupStats {
        int n1, n2;
        // parametric
        nb::object t;
        nb::object df;
        nb::object pvalue_t;
        nb::object cohens_d;
        nb::object means;
        nb::object sds;
        // non-parametric
        nb::object U;
        nb::object pvalue_u;
        nb::object effect_r;
        nb::object medians;
    };
    auto make_py_gs = [](const qe::GroupStats& s) {
        PyGroupStats p;
        p.n1       = s.n1;
        p.n2       = s.n2;
        p.t        = nb::cast(from_vec(s.t));
        p.df       = nb::cast(from_vec(s.df));
        p.pvalue_t = nb::cast(from_vec(s.pvalue_t));
        p.cohens_d = nb::cast(from_vec(s.cohens_d));
        p.means    = nb::cast(from_mat(s.means));
        p.sds      = nb::cast(from_mat(s.sds));
        p.U        = nb::cast(from_vec(s.U));
        p.pvalue_u = nb::cast(from_vec(s.pvalue_u));
        p.effect_r = nb::cast(from_vec(s.effect_r));
        p.medians  = nb::cast(from_mat(s.medians));
        return p;
    };

    nb::class_<PyGroupStats>(mod, "GroupStatsResult",
        "Two-group comparison statistics returned by group_stats().\n\n"
        "Attributes\n"
        "----------\n"
        "n1, n2       : int  — sample sizes\n"
        "Parametric (Welch t-test):\n"
        "  t          : ndarray (n_dims,)  — t-statistics\n"
        "  df         : ndarray (n_dims,)  — Welch–Satterthwaite degrees of freedom\n"
        "  pvalue_t   : ndarray (n_dims,)  — two-tailed p-values\n"
        "  cohens_d   : ndarray (n_dims,)  — Cohen's d (pooled SD)\n"
        "  means      : ndarray (2 × n_dims)  — row 0 = group1, row 1 = group2\n"
        "  sds        : ndarray (2 × n_dims)  — sample standard deviations\n"
        "Non-parametric (Wilcoxon rank-sum):\n"
        "  U          : ndarray (n_dims,)  — U for group 1 (= R's W statistic)\n"
        "  pvalue_u   : ndarray (n_dims,)  — two-tailed p-values (normal approx)\n"
        "  effect_r   : ndarray (n_dims,)  — rank-biserial: 1 − 2·U / (n1·n2)\n"
        "  medians    : ndarray (2 × n_dims)  — row 0 = group1, row 1 = group2")
        .def_ro("n1",       &PyGroupStats::n1)
        .def_ro("n2",       &PyGroupStats::n2)
        .def_ro("t",        &PyGroupStats::t)
        .def_ro("df",       &PyGroupStats::df)
        .def_ro("pvalue_t", &PyGroupStats::pvalue_t)
        .def_ro("cohens_d", &PyGroupStats::cohens_d)
        .def_ro("means",    &PyGroupStats::means)
        .def_ro("sds",      &PyGroupStats::sds)
        .def_ro("U",        &PyGroupStats::U)
        .def_ro("pvalue_u", &PyGroupStats::pvalue_u)
        .def_ro("effect_r", &PyGroupStats::effect_r)
        .def_ro("medians",  &PyGroupStats::medians)
        .def("__repr__", [](const PyGroupStats& p) {
            return std::string("<GroupStatsResult n1=") + std::to_string(p.n1)
                 + " n2=" + std::to_string(p.n2) + ">";
        });

    mod.def("group_stats", [make_py_gs](NpMat g1, NpMat g2) {
        return make_py_gs(qe::group_stats(to_mat(g1), to_mat(g2)));
    }, "g1"_a, "g2"_a,
        "Per-dimension two-group comparison statistics.\n\n"
        "Computes Welch t-test (t, df, p-value, Cohen's d, means, SDs) and\n"
        "Wilcoxon rank-sum test (U, p-value, rank-biserial effect, medians)\n"
        "for each dimension independently.\n\n"
        "Parameters\n----------\n"
        "g1 : ndarray (n1 × n_dims)  — unit points for group 1\n"
        "g2 : ndarray (n2 × n_dims)  — unit points for group 2\n\n"
        "Returns GroupStatsResult.\n\n"
        "Notes\n-----\n"
        "Parametric entries are NaN when n < 2 for either group.\n"
        "Wilcoxon p-values use the normal approximation (tie + continuity\n"
        "correction), matching R's wilcox.test(..., exact=FALSE, correct=TRUE).\n"
        "Equivalent to rENA-api's group.stats().");

    // ── door ─────────────────────────────────────────────────────────────────
    auto door_mod = m.def_submodule("door", "Door temporal pooling and EMA smoothing");

    door_mod.def("lookback_block", [](NpMat block, int lookback_size, bool aggregate_mean, bool weighting_linear, const std::vector<int>& segment_ids) {
        arma::mat b = to_mat(block);
        return from_mat(qe::lookback_block(b, lookback_size, aggregate_mean, weighting_linear, segment_ids));
    }, "block"_a, "lookback_size"_a = 20, "aggregate_mean"_a = false, "weighting_linear"_a = false, "segment_ids"_a = std::vector<int>{},
    "Apply sliding lookback window pool over a block of connection counts.");

    door_mod.def("ema_block", [](NpMat block, double alpha, const std::vector<int>& segment_ids) {
        arma::mat b = to_mat(block);
        return from_mat(qe::ema_block(b, alpha, segment_ids));
    }, "block"_a, "alpha"_a = 0.1, "segment_ids"_a = std::vector<int>{},
    "Apply Exponential Moving Average (EMA) smoothing over a block of connection counts.");

    // ── trajectory ───────────────────────────────────────────────────────────
    auto traj_mod = m.def_submodule("trajectory", "Parametric curve fitting, derivatives, distance, and following");

    traj_mod.def("fit_poly", [](NpMat points, std::optional<NpVec> t, int max_degree, int fixed_degree, const std::string& criterion, const std::string& basis) {
        arma::mat pts = to_mat(points);
        arma::vec t_vec;
        if (t.has_value()) {
            t_vec = to_vec(t.value());
        }
        qe::PolyCurveFit fit = qe::fit_poly_loocv2d(pts, t_vec, max_degree, fixed_degree, criterion, basis);
        nb::dict out;
        out["degree"] = fit.degree;
        out["coeffs_x"] = from_vec(fit.coeffs_x);
        out["coeffs_y"] = from_vec(fit.coeffs_y);
        out["basis"] = fit.basis;
        out["basis_alpha"] = from_vec(fit.basis_alpha);
        out["basis_norm2"] = from_vec(fit.basis_norm2);
        out["cv_error"] = fit.cv_error;
        out["aic"] = fit.aic;
        out["t"] = from_vec(fit.t);
        out["fitted_points"] = from_mat(fit.fitted);
        return out;
    }, "points"_a, "t"_a = nb::none(), "max_degree"_a = 3, "fixed_degree"_a = 0, "criterion"_a = "loocv", "basis"_a = "orthogonal",
    "Fit 2D parametric polynomial curve with R-compatible orthogonal or raw basis.");

    traj_mod.def("eval_curve", [](NpVec coeffs_x, NpVec coeffs_y, NpVec t_eval) {
        return from_mat(qe::eval_poly_curve(to_vec(coeffs_x), to_vec(coeffs_y), to_vec(t_eval)));
    }, "coeffs_x"_a, "coeffs_y"_a, "t_eval"_a);

    traj_mod.def("eval_derivatives", [](NpVec coeffs_x, NpVec coeffs_y, NpVec t_eval) {
        qe::TrajectoryDerivatives d = qe::eval_trajectory_derivatives(to_vec(coeffs_x), to_vec(coeffs_y), to_vec(t_eval));
        nb::dict out;
        out["t"] = from_vec(d.t);
        out["vx"] = from_vec(d.vx);
        out["vy"] = from_vec(d.vy);
        out["speed"] = from_vec(d.speed);
        out["ax"] = from_vec(d.ax);
        out["ay"] = from_vec(d.ay);
        out["heading_rate"] = from_vec(d.heading_rate);
        out["curvature"] = from_vec(d.curvature);
        return out;
    }, "coeffs_x"_a, "coeffs_y"_a, "t_eval"_a);

    traj_mod.def("integrated_distance", [](NpVec coeffs_ax, NpVec coeffs_ay, NpVec coeffs_bx, NpVec coeffs_by, double t_start, double t_end) {
        return qe::integrated_curve_distance(to_vec(coeffs_ax), to_vec(coeffs_ay), to_vec(coeffs_bx), to_vec(coeffs_by), t_start, t_end);
    }, "coeffs_ax"_a, "coeffs_ay"_a, "coeffs_bx"_a, "coeffs_by"_a, "t_start"_a = 0.0, "t_end"_a = 1.0);

    traj_mod.def("lagged_distance", [](NpVec coeffs_fol_x, NpVec coeffs_fol_y, NpVec coeffs_ldr_x, NpVec coeffs_ldr_y, double lag) {
        return qe::lagged_curve_distance(to_vec(coeffs_fol_x), to_vec(coeffs_fol_y), to_vec(coeffs_ldr_x), to_vec(coeffs_ldr_y), lag);
    }, "coeffs_fol_x"_a, "coeffs_fol_y"_a, "coeffs_ldr_x"_a, "coeffs_ldr_y"_a, "lag"_a = 0.0);

    traj_mod.def("signed_turn_lag", [](NpMat pts_a, NpMat pts_b, NpVec times_a, NpVec times_b, int delta) {
        auto [mean_d, count] = qe::signed_turn_lag_distance(to_mat(pts_a), to_mat(pts_b), to_vec(times_a), to_vec(times_b), delta);
        nb::dict out;
        out["mean_distance"] = mean_d;
        out["valid_count"] = count;
        return out;
    }, "pts_a"_a, "pts_b"_a, "times_a"_a, "times_b"_a, "delta"_a);

    traj_mod.def("sweep_signed_turn_lags", [](NpMat pts_a, NpMat pts_b, NpVec times_a, NpVec times_b, int max_lag) {
        qe::SignedTurnLagResult res = qe::best_signed_turn_lag(to_mat(pts_a), to_mat(pts_b), to_vec(times_a), to_vec(times_b), max_lag);
        nb::dict out;
        out["best_lag"] = res.best_lag;
        out["min_mean_distance"] = res.min_mean_distance;
        arma::vec lags_d = arma::conv_to<arma::vec>::from(res.lags);
        arma::vec counts_d = arma::conv_to<arma::vec>::from(res.valid_counts);
        out["lags"] = from_vec(lags_d);
        out["mean_distances"] = from_vec(res.mean_distances);
        out["valid_counts"] = from_vec(counts_d);
        return out;
    }, "pts_a"_a, "pts_b"_a, "times_a"_a, "times_b"_a, "max_lag"_a = 15);

    traj_mod.def("dist_dist_correlation", [](NpMat X, NpMat Y) {
        return qe::dist_dist_correlation(to_mat(X), to_mat(Y));
    }, "X"_a, "Y"_a);
}
