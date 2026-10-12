// libqe WebAssembly bindings via Emscripten Embind.
//
// Matrix convention
// -----------------
// Armadillo stores matrices column-major.  JavaScript typed arrays are
// row-major by convention.  Every function that accepts a matrix takes a
// flat Float64Array (row-major) plus explicit rows/cols, and converts
// internally.  Return values are plain JS objects:
//
//   { data: Float64Array, rows: number, cols: number }
//
// This keeps the JS API free of Armadillo-specific types while avoiding
// an extra copy on the way in (vecFromJSArray copies once; that is
// unavoidable when crossing the WASM heap boundary).
//
// Disable BLAS/LAPACK — Armadillo falls back to its own built-in routines
// (LU, QR, etc.) which compile cleanly with Emscripten.

#ifndef ARMA_DONT_USE_BLAS
#  define ARMA_DONT_USE_BLAS
#endif
#ifndef ARMA_DONT_USE_LAPACK
#  define ARMA_DONT_USE_LAPACK
#endif

#include <armadillo>
#include <libqe/libqe.hpp>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <libqe/bind/emscripten.hpp>
#include <vector>
#include <string>

using namespace emscripten;

// ── Helpers ───────────────────────────────────────────────────────────────────

// Array conversion (js_to_mat, mat_to_js, …) and the rows × cols length
// check come from libqe/bind/emscripten.hpp and libqe/validate.hpp.
using namespace qe::bind::js;

// ── Adjacency ─────────────────────────────────────────────────────────────────

// connection_indices(len) → { rows: Int32Array, cols: Int32Array }
static val connection_indices(int len) {
    arma::umat idx = qe::connection_indices(len, -1);
    std::vector<int> rows_v(idx.n_cols), cols_v(idx.n_cols);
    for (arma::uword i = 0; i < idx.n_cols; ++i) {
        rows_v[i] = static_cast<int>(idx(0, i));
        cols_v[i] = static_cast<int>(idx(1, i));
    }
    val result = val::object();
    result.set("rows", val::array(rows_v.begin(), rows_v.end()));
    result.set("cols", val::array(cols_v.begin(), cols_v.end()));
    return result;
}

// code_connections(data, n_codes) → Float64Array
static val code_connections(const val& data, int n_codes) {
    arma::mat v = js_to_mat(data, 1, n_codes);
    return rowvec_to_js(qe::code_connections(v));
}

// fold_directed_network(data) → Float64Array  (data.length == n*n)
static val fold_directed_network(const val& data) {
    std::vector<double> v = vecFromJSArray<double>(data);
    arma::vec av(v.data(), v.size());
    return rowvec_to_js(qe::fold_directed_network(av));
}

// network_to_vector(data, rows, cols, full) → Float64Array
static val network_to_vector(const val& data, int rows, int cols, bool full) {
    arma::mat m = js_to_mat(data, rows, cols);
    return rowvec_to_js(qe::network_to_vector(m, full));
}

// connection_names(names) → string[]
static val connection_names(const val& names) {
    std::vector<std::string> v = vecFromJSArray<std::string>(names);
    auto pairs = qe::connection_names(v);
    val result = val::array();
    for (size_t i = 0; i < pairs.size(); ++i)
        result.call<void>("push", val(pairs[i]));
    return result;
}

// ── Normalization ─────────────────────────────────────────────────────────────

// normalize_networks(data, rows, cols) → { data, rows, cols }
static val normalize_networks(const val& data, int rows, int cols) {
    return mat_to_js(qe::normalize_networks(js_to_mat(data, rows, cols)));
}

// scale_networks(data, rows, cols) → { data, rows, cols }
static val scale_networks(const val& data, int rows, int cols) {
    return mat_to_js(qe::scale_networks(js_to_mat(data, rows, cols)));
}

// ── Modeling ──────────────────────────────────────────────────────────────────

// center_points(data, rows, cols) → { data, rows, cols }
static val center_points(const val& data, int rows, int cols) {
    return mat_to_js(qe::center_points(js_to_mat(data, rows, cols)));
}

// mean_ci(data, rows, cols, conf_level) → { data, rows:n_dims, cols:3 }
// columns: [mean, ci_lower, ci_upper]
static val mean_ci(const val& data, int rows, int cols, double conf_level) {
    return mat_to_js(qe::mean_ci(js_to_mat(data, rows, cols), conf_level));
}

// outlier_ci(data, rows, cols, iqr_factor) → { data, rows:n_dims, cols:2 }
// columns: [lower, upper]
static val outlier_ci(const val& data, int rows, int cols, double iqr_factor) {
    return mat_to_js(qe::outlier_ci(js_to_mat(data, rows, cols), iqr_factor));
}

// group_stats(g1, g1_rows, g1_cols, g2, g2_rows, g2_cols)
// → {
//     n1: int, n2: int,
//     t, df, pvalue_t, cohens_d : Float64Array (length n_dims),
//     means, sds                : matObj (rows=2, cols=n_dims),
//     U, pvalue_u, effect_r     : Float64Array (length n_dims),
//     medians                   : matObj (rows=2, cols=n_dims)
//   }
//
// Matches rENA-api's group.stats() (now computed in C++ for all bindings).
static val group_stats(const val& g1_data, int g1_rows, int g1_cols,
                       const val& g2_data, int g2_rows, int g2_cols) {
    const qe::GroupStats s = qe::group_stats(
        js_to_mat(g1_data, g1_rows, g1_cols),
        js_to_mat(g2_data, g2_rows, g2_cols));

    auto to_js_vec = [](const arma::vec& v) {
        std::vector<double> d(v.memptr(), v.memptr() + v.n_elem);
        return val::array(d.begin(), d.end());
    };

    val result = val::object();
    result.set("n1",       s.n1);
    result.set("n2",       s.n2);
    result.set("t",        to_js_vec(s.t));
    result.set("df",       to_js_vec(s.df));
    result.set("pvalue_t", to_js_vec(s.pvalue_t));
    result.set("cohens_d", to_js_vec(s.cohens_d));
    result.set("means",    mat_to_js(s.means));
    result.set("sds",      mat_to_js(s.sds));
    result.set("U",        to_js_vec(s.U));
    result.set("pvalue_u", to_js_vec(s.pvalue_u));
    result.set("effect_r", to_js_vec(s.effect_r));
    result.set("medians",  mat_to_js(s.medians));
    return result;
}

// choose_two(n) → int
static int choose_two(int n) {
    return qe::choose_two(n);
}

// ── Door ──────────────────────────────────────────────────────────────────────

static val door_lookback_block(
    const val& data, int rows, int cols,
    int lookback_size, bool aggregate_mean, bool weighting_linear,
    const val& segment_ids_val
) {
    std::vector<int> segs = vecFromJSArray<int>(segment_ids_val);
    return mat_to_js(qe::lookback_block(
        js_to_mat(data, rows, cols),
        lookback_size, aggregate_mean, weighting_linear, segs));
}

static val door_ema_block(
    const val& data, int rows, int cols,
    double alpha, const val& segment_ids_val
) {
    std::vector<int> segs = vecFromJSArray<int>(segment_ids_val);
    return mat_to_js(qe::ema_block(
        js_to_mat(data, rows, cols),
        alpha, segs));
}

// ── Trajectory ────────────────────────────────────────────────────────────────

static val fit_trajectory_poly(
    const val& data, int rows, int cols,
    const val& t_val, int max_degree, int fixed_degree, const std::string& criterion,
    const std::string& basis
) {
    arma::mat pts = js_to_mat(data, rows, cols);
    std::vector<double> tv = vecFromJSArray<double>(t_val);
    arma::vec t_vec;
    if (!tv.empty()) {
        t_vec = arma::vec(tv.data(), tv.size());
    }

    qe::PolyCurveFit fit = qe::fit_poly_loocv2d(pts, t_vec, max_degree, fixed_degree, criterion, basis);

    auto to_js_vec = [](const arma::vec& v) {
        std::vector<double> d(v.memptr(), v.memptr() + v.n_elem);
        return val::array(d.begin(), d.end());
    };

    val result = val::object();
    result.set("degree",        fit.degree);
    result.set("coeffs_x",      to_js_vec(fit.coeffs_x));
    result.set("coeffs_y",      to_js_vec(fit.coeffs_y));
    result.set("basis",         fit.basis);
    result.set("basis_alpha",   to_js_vec(fit.basis_alpha));
    result.set("basis_norm2",   to_js_vec(fit.basis_norm2));
    result.set("cv_error",      fit.cv_error);
    result.set("aic",           fit.aic);
    result.set("t",             to_js_vec(fit.t));
    result.set("fitted_points", mat_to_js(fit.fitted));
    return result;
}

static val eval_trajectory_curve(
    const val& cx_val, const val& cy_val, const val& t_eval_val
) {
    std::vector<double> cx = vecFromJSArray<double>(cx_val);
    std::vector<double> cy = vecFromJSArray<double>(cy_val);
    std::vector<double> te = vecFromJSArray<double>(t_eval_val);

    arma::vec v_cx(cx.data(), cx.size());
    arma::vec v_cy(cy.data(), cy.size());
    arma::vec v_te(te.data(), te.size());

    return mat_to_js(qe::eval_poly_curve(v_cx, v_cy, v_te));
}

static val eval_trajectory_derivatives(
    const val& cx_val, const val& cy_val, const val& t_eval_val
) {
    std::vector<double> cx = vecFromJSArray<double>(cx_val);
    std::vector<double> cy = vecFromJSArray<double>(cy_val);
    std::vector<double> te = vecFromJSArray<double>(t_eval_val);

    arma::vec v_cx(cx.data(), cx.size());
    arma::vec v_cy(cy.data(), cy.size());
    arma::vec v_te(te.data(), te.size());

    qe::TrajectoryDerivatives d = qe::eval_trajectory_derivatives(v_cx, v_cy, v_te);

    auto to_js_vec = [](const arma::vec& v) {
        std::vector<double> dt(v.memptr(), v.memptr() + v.n_elem);
        return val::array(dt.begin(), dt.end());
    };

    val result = val::object();
    result.set("t",            to_js_vec(d.t));
    result.set("vx",           to_js_vec(d.vx));
    result.set("vy",           to_js_vec(d.vy));
    result.set("speed",        to_js_vec(d.speed));
    result.set("ax",           to_js_vec(d.ax));
    result.set("ay",           to_js_vec(d.ay));
    result.set("heading_rate", to_js_vec(d.heading_rate));
    result.set("curvature",    to_js_vec(d.curvature));
    return result;
}

static double integrated_trajectory_distance(
    const val& cax_val, const val& cay_val,
    const val& cbx_val, const val& cby_val,
    double t_start, double t_end
) {
    std::vector<double> ax = vecFromJSArray<double>(cax_val);
    std::vector<double> ay = vecFromJSArray<double>(cay_val);
    std::vector<double> bx = vecFromJSArray<double>(cbx_val);
    std::vector<double> by = vecFromJSArray<double>(cby_val);

    return qe::integrated_curve_distance(
        arma::vec(ax.data(), ax.size()),
        arma::vec(ay.data(), ay.size()),
        arma::vec(bx.data(), bx.size()),
        arma::vec(by.data(), by.size()),
        t_start, t_end);
}

static double lagged_trajectory_distance(
    const val& cfx_val, const val& cfy_val,
    const val& clx_val, const val& cly_val,
    double lag
) {
    std::vector<double> fx = vecFromJSArray<double>(cfx_val);
    std::vector<double> fy = vecFromJSArray<double>(cfy_val);
    std::vector<double> lx = vecFromJSArray<double>(clx_val);
    std::vector<double> ly = vecFromJSArray<double>(cly_val);

    return qe::lagged_curve_distance(
        arma::vec(fx.data(), fx.size()),
        arma::vec(fy.data(), fy.size()),
        arma::vec(lx.data(), lx.size()),
        arma::vec(ly.data(), ly.size()),
        lag);
}

static val signed_turn_lag(
    const val& pa_val, int ra, int ca,
    const val& pb_val, int rb, int cb,
    const val& ta_val, const val& tb_val,
    int delta
) {
    std::vector<double> ta = vecFromJSArray<double>(ta_val);
    std::vector<double> tb = vecFromJSArray<double>(tb_val);

    auto [mean_d, count] = qe::signed_turn_lag_distance(
        js_to_mat(pa_val, ra, ca),
        js_to_mat(pb_val, rb, cb),
        arma::vec(ta.data(), ta.size()),
        arma::vec(tb.data(), tb.size()),
        delta);

    val result = val::object();
    result.set("mean_distance", mean_d);
    result.set("valid_count",   count);
    return result;
}

static val sweep_signed_turn_lags(
    const val& pa_val, int ra, int ca,
    const val& pb_val, int rb, int cb,
    const val& ta_val, const val& tb_val,
    int max_lag
) {
    std::vector<double> ta = vecFromJSArray<double>(ta_val);
    std::vector<double> tb = vecFromJSArray<double>(tb_val);

    qe::SignedTurnLagResult res = qe::best_signed_turn_lag(
        js_to_mat(pa_val, ra, ca),
        js_to_mat(pb_val, rb, cb),
        arma::vec(ta.data(), ta.size()),
        arma::vec(tb.data(), tb.size()),
        max_lag);

    val result = val::object();
    result.set("best_lag",          res.best_lag);
    result.set("min_mean_distance", res.min_mean_distance);

    std::vector<int> lags_v(res.lags.memptr(), res.lags.memptr() + res.lags.n_elem);
    std::vector<double> dists_v(res.mean_distances.memptr(), res.mean_distances.memptr() + res.mean_distances.n_elem);
    std::vector<int> counts_v(res.valid_counts.memptr(), res.valid_counts.memptr() + res.valid_counts.n_elem);

    result.set("lags",           val::array(lags_v.begin(), lags_v.end()));
    result.set("mean_distances", val::array(dists_v.begin(), dists_v.end()));
    result.set("valid_counts",   val::array(counts_v.begin(), counts_v.end()));
    return result;
}

static double dist_dist_correlation(
    const val& x_val, int rx, int cx,
    const val& y_val, int ry, int cy
) {
    return qe::dist_dist_correlation(
        js_to_mat(x_val, rx, cx),
        js_to_mat(y_val, ry, cy));
}

// ── Embind registrations ──────────────────────────────────────────────────────

EMSCRIPTEN_BINDINGS(libqe) {
    // Adjacency
    function("choose_two",                            &choose_two);
    function("connection_indices",                    &connection_indices);
    function("code_connections",                      &code_connections);
    function("fold_directed_network",                 &fold_directed_network);
    function("network_to_vector",                     &network_to_vector);
    function("connection_names",                      &connection_names);

    // Normalization
    function("normalize_networks",                    &normalize_networks);
    function("scale_networks",                        &scale_networks);

    // Modeling
    function("center_points",                         &center_points);
    function("mean_ci",                               &mean_ci);
    function("outlier_ci",                            &outlier_ci);
    function("group_stats",                           &group_stats);

    // Door
    function("door_lookback_block",                   &door_lookback_block);
    function("door_ema_block",                        &door_ema_block);

    // Trajectory
    function("fit_trajectory_poly",                   &fit_trajectory_poly);
    function("eval_trajectory_curve",                 &eval_trajectory_curve);
    function("eval_trajectory_derivatives",           &eval_trajectory_derivatives);
    function("integrated_trajectory_distance",        &integrated_trajectory_distance);
    function("lagged_trajectory_distance",            &lagged_trajectory_distance);
    function("signed_turn_lag",                       &signed_turn_lag);
    function("sweep_signed_turn_lags",                &sweep_signed_turn_lags);
    function("dist_dist_correlation",                 &dist_dist_correlation);
}
