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
#include <vector>
#include <string>

using namespace emscripten;

// ── Helpers ───────────────────────────────────────────────────────────────────

// JS Float64Array (row-major) → arma::mat (column-major)
static arma::mat js_to_mat(const val& data, int rows, int cols) {
    std::vector<double> v = vecFromJSArray<double>(data);
    // arma::mat(ptr, rows, cols) reads column-major; transpose to convert
    // from the row-major JS layout.
    arma::mat m(v.data(), cols, rows);   // read as (cols × rows) col-major
    return m.t();                         // transpose → (rows × cols)
}

// JS Int32Array (row-major) → arma::imat (column-major integers)
static arma::imat js_to_imat(const val& data, int rows, int cols) {
    std::vector<int> v = vecFromJSArray<int>(data);
    arma::imat m(v.data(), cols, rows);  // read as (cols × rows) col-major
    return m.t();                         // transpose → (rows × cols)
}

// arma::mat (column-major) → JS { data: Float64Array, rows, cols }
static val mat_to_js(const arma::mat& m) {
    // Transpose to row-major for JS consumers.
    arma::mat row_major = m.t();
    std::vector<double> v(row_major.memptr(),
                          row_major.memptr() + row_major.n_elem);
    val result = val::object();
    result.set("data", val::array(v.begin(), v.end()));
    result.set("rows", static_cast<int>(m.n_rows));
    result.set("cols", static_cast<int>(m.n_cols));
    return result;
}

// arma::rowvec → JS Float64Array
static val rowvec_to_js(const arma::rowvec& v) {
    std::vector<double> vec(v.memptr(), v.memptr() + v.n_elem);
    return val::array(vec.begin(), vec.end());
}

// RotationResult → JS { rotation: matObj, eigenvalues: Float64Array, column_names: string[] }
static val rotation_to_js(const qe::RotationResult& r) {
    val result = val::object();
    result.set("rotation",     mat_to_js(r.rotation));
    std::vector<double> ev(r.eigenvalues.memptr(),
                           r.eigenvalues.memptr() + r.eigenvalues.n_elem);
    result.set("eigenvalues",  val::array(ev.begin(), ev.end()));
    val names = val::array();
    for (const auto& s : r.column_names)
        names.call<void>("push", val(s));
    result.set("column_names", names);
    return result;
}

// NodePositions → JS { nodes, centroids, weights, points }
static val node_positions_to_js(const qe::NodePositions& r) {
    val result = val::object();
    result.set("nodes",     mat_to_js(r.nodes));
    result.set("centroids", mat_to_js(r.centroids));
    result.set("weights",   mat_to_js(r.weights));
    result.set("points",    mat_to_js(r.points));
    return result;
}

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

// ena_correlation(pts, pt_rows, pt_cols, cen, cen_rows, cen_cols, conf_level)
// → { data, rows:n_units, cols:3 }  columns: [r, ci_lower, ci_upper]
static val ena_correlation(const val& pts, int pt_rows, int pt_cols,
                            const val& cen, int cen_rows, int cen_cols,
                            double conf_level) {
    return mat_to_js(qe::ena_correlation(
        js_to_mat(pts, pt_rows, pt_cols),
        js_to_mat(cen, cen_rows, cen_cols),
        conf_level));
}

// node_positions(adj_data, adj_rows, adj_cols,
//                t_data,   t_rows,   t_cols [, num_dims])
// → { nodes, centroids, weights, points }  (each a matrix object)
//
// num_dims: how many columns of t to project onto.  If omitted or <= 0,
// defaults to t_cols (use all dimensions).  Embind passes 0 for missing
// integer arguments, so this default covers the "caller forgot num_dims" case.
static val node_positions(const val& adj_data, int adj_rows, int adj_cols,
                           const val& t_data,   int t_rows,   int t_cols,
                           int num_dims) {
    arma::mat t = js_to_mat(t_data, t_rows, t_cols);
    if (num_dims <= 0) num_dims = static_cast<int>(t.n_cols);
    return node_positions_to_js(qe::node_positions(
        js_to_mat(adj_data, adj_rows, adj_cols),
        t,
        num_dims));
}

// directed_node_positions — same signature as node_positions
static val directed_node_positions(const val& lw_data, int lw_rows, int lw_cols,
                                    const val& pt_data,  int pt_rows, int pt_cols,
                                    int num_dims) {
    arma::mat pt = js_to_mat(pt_data, pt_rows, pt_cols);
    if (num_dims <= 0) num_dims = static_cast<int>(pt.n_cols);
    return node_positions_to_js(qe::directed_node_positions(
        js_to_mat(lw_data, lw_rows, lw_cols),
        pt,
        num_dims));
}

// directed_node_positions_combine_pairs — ground+response rows combined before solve
static val directed_node_positions_combine_pairs(
        const val& lw_data, int lw_rows, int lw_cols,
        const val& pt_data, int pt_rows, int pt_cols,
        int num_dims) {
    arma::mat pt = js_to_mat(pt_data, pt_rows, pt_cols);
    if (num_dims <= 0) num_dims = static_cast<int>(pt.n_cols);
    return node_positions_to_js(qe::directed_node_positions(
        js_to_mat(lw_data, lw_rows, lw_cols),
        pt,
        num_dims, /*combine_pairs=*/true));
}

// ── Accumulation ─────────────────────────────────────────────────────────────

// connection_matrix(ground, gn, response, rn, response_weight, ordered)
// → { data, rows:n_codes, cols:n_codes }
static val connection_matrix(const val& ground_data, int gn,
                              const val& response_data, int rn,
                              double response_weight, bool ordered) {
    std::vector<double> gv = vecFromJSArray<double>(ground_data);
    std::vector<double> rv = vecFromJSArray<double>(response_data);
    arma::rowvec g(gv.data(), gn);
    arma::rowvec r(rv.data(), rn);
    return mat_to_js(qe::connection_matrix(g, r, response_weight, ordered));
}

// accumulate_stanza(data, rows, cols, window_back, window_forward, binary, ordered)
// ordered=false → { data, rows, choose_two(cols) }
// ordered=true  → { data, rows, cols² }
static val accumulate_stanza(const val& data, int rows, int cols,
                              int window_back, int window_forward,
                              bool binary, bool ordered) {
    return mat_to_js(qe::accumulate_stanza(
        js_to_mat(data, rows, cols), window_back, window_forward, binary, ordered));
}

// row_connections(data, rows, cols, binary) → { data, rows, cols }
static val row_connections(const val& data, int rows, int cols, bool binary) {
    return mat_to_js(qe::row_connections(js_to_mat(data, rows, cols), binary));
}

// rolling_window_sum(data, rows, cols, window_size) → { data, rows, cols }
static val rolling_window_sum(const val& data, int rows, int cols, int window_size) {
    return mat_to_js(qe::rolling_window_sum(js_to_mat(data, rows, cols), window_size));
}

// flat_index(indices_array, dims_array) → int
static int flat_index(const val& indices_val, const val& dims_val) {
    return qe::flat_index(
        vecFromJSArray<int>(indices_val),
        vecFromJSArray<int>(dims_val));
}

// accumulate_unit(codes, rows, cols, unit_rows, decay_fn, ordered)
// decay_fn: JS function(Float64Array distances) → Float64Array weights
// → Float64Array  (length choose_two(n_codes) or n_codes²)
static val accumulate_unit(const val& codes_data, int rows, int cols,
                            const val& unit_rows_val,
                            const val& decay_fn_js,
                            bool ordered) {
    std::vector<int> unit_rows = vecFromJSArray<int>(unit_rows_val);

    auto cpp_decay = [&decay_fn_js](arma::vec distances) -> arma::vec {
        std::vector<double> dv(distances.memptr(), distances.memptr() + distances.n_elem);
        val js_result = decay_fn_js(val::array(dv.begin(), dv.end()));
        std::vector<double> wv = vecFromJSArray<double>(js_result);
        return arma::vec(wv.data(), wv.size());
    };

    return rowvec_to_js(qe::accumulate_unit(
        js_to_mat(codes_data, rows, cols), unit_rows, cpp_decay, ordered));
}

// accumulate_unit_with_rows(codes, rows, cols, unit_rows, decay_fn, ordered)
// → { networks: Float64Array, row_networks: matObj }
static val accumulate_unit_with_rows(const val& codes_data, int rows, int cols,
                                      const val& unit_rows_val,
                                      const val& decay_fn_js,
                                      bool ordered) {
    std::vector<int> unit_rows = vecFromJSArray<int>(unit_rows_val);

    auto cpp_decay = [&decay_fn_js](int unit_row, arma::uvec ground_indices) -> arma::vec {
        arma::vec distances(ground_indices.n_elem);
        for (arma::uword k = 0; k < ground_indices.n_elem; ++k)
            distances[k] = static_cast<double>(unit_row - ground_indices[k]);
        std::vector<double> dv(distances.memptr(), distances.memptr() + distances.n_elem);
        val js_result = decay_fn_js(val::array(dv.begin(), dv.end()));
        std::vector<double> wv = vecFromJSArray<double>(js_result);
        return arma::vec(wv.data(), wv.size());
    };

    qe::UnitNetworks r = qe::accumulate_unit_with_rows(
        js_to_mat(codes_data, rows, cols), unit_rows, cpp_decay, ordered);

    val result = val::object();
    result.set("networks",     rowvec_to_js(r.networks));
    result.set("row_networks", mat_to_js(r.row_networks));
    return result;
}

// accumulate_tensor_unit(tensor, dims, dims_sender, dims_receiver, dims_mode,
//                        context_lookup, cl_rows, cl_cols,
//                        unit_rows, codes, rows, cols, times, ordered)
// → { connection_counts: Float64Array,
//     row_connection_counts: { data: Float64Array, rows, cols } }
//
// tensor           — Float64Array, flat column-major, shape described by dims
// dims             — Int32Array  — sizes of each tensor axis
// dims_sender      — Int32Array  — which tensor axes correspond to sender factors
// dims_receiver    — Int32Array  — which tensor axes correspond to receiver factors
// dims_mode        — Int32Array  — which tensor axes correspond to mode factors
// context_lookup   — Int32Array, row-major (n_context_rows × n_factors), 0-based
// cl_rows/cl_cols  — shape of context_lookup
// unit_rows        — Int32Array, 0-based response-row indices for this unit
// codes            — Float64Array, row-major (rows × cols)
// times            — Float64Array, one timestamp per context row
// ordered          — bool (true → directed n²; false → undirected upper-tri)
static val accumulate_tensor_unit(
        const val& tensor_val,
        const val& dims_val,
        const val& dims_sender_val,
        const val& dims_receiver_val,
        const val& dims_mode_val,
        const val& context_lookup_val, int cl_rows, int cl_cols,
        const val& unit_rows_val,
        const val& codes_val, int rows, int cols,
        const val& times_val,
        bool ordered) {

    std::vector<double> tv = vecFromJSArray<double>(tensor_val);
    arma::vec tensor(tv.data(), tv.size());

    std::vector<int> dims          = vecFromJSArray<int>(dims_val);
    std::vector<int> dims_sender   = vecFromJSArray<int>(dims_sender_val);
    std::vector<int> dims_receiver = vecFromJSArray<int>(dims_receiver_val);
    std::vector<int> dims_mode     = vecFromJSArray<int>(dims_mode_val);
    std::vector<int> unit_rows     = vecFromJSArray<int>(unit_rows_val);

    arma::imat context_lookup = js_to_imat(context_lookup_val, cl_rows, cl_cols);
    arma::mat  codes_mat      = js_to_mat(codes_val, rows, cols);

    std::vector<double> timev = vecFromJSArray<double>(times_val);
    arma::vec times_vec(timev.data(), timev.size());

    qe::TensorNetworks r = qe::apply_tensor_unit(
        tensor, dims, dims_sender, dims_receiver, dims_mode,
        context_lookup, unit_rows, codes_mat, times_vec, ordered);

    std::vector<double> cc(r.connection_counts.memptr(),
                           r.connection_counts.memptr() + r.connection_counts.n_elem);

    val result = val::object();
    result.set("connection_counts",     val::array(cc.begin(), cc.end()));
    result.set("row_connection_counts", mat_to_js(r.row_connection_counts));
    return result;
}

// aggregate_row_connections(row_conn, rows, cols, n_codes, ordered, binary)
//   row_conn — Float64Array, row-major (rows × cols), cols == n_codes²
//              (TensorNetworks.row_connection_counts from accumulate_tensor_unit)
// → Float64Array — unit vector, length n_codes² (ordered) or choose_two (unordered)
static val aggregate_row_connections(
        const val& row_conn_val, int rows, int cols,
        int n_codes, bool ordered, bool binary) {
    arma::mat row_conn = js_to_mat(row_conn_val, rows, cols);
    arma::rowvec out = qe::aggregate_row_connections(row_conn, n_codes, ordered, binary);
    std::vector<double> v(out.memptr(), out.memptr() + out.n_elem);
    return val::array(v.begin(), v.end());
}

// ── Rotation ──────────────────────────────────────────────────────────────────

// ena_svd(data, rows, cols)
// → { rotation: matObj, eigenvalues: Float64Array, column_names: string[] }
static val ena_svd(const val& data, int rows, int cols) {
    return rotation_to_js(qe::ena_svd(js_to_mat(data, rows, cols)));
}

// deflate(data, rows, cols, axis_data)
// → { data, rows, cols }
static val deflate(const val& data, int rows, int cols, const val& axis_data) {
    std::vector<double> av = vecFromJSArray<double>(axis_data);
    arma::vec axis(av.data(), av.size());
    return mat_to_js(qe::deflate(js_to_mat(data, rows, cols), axis));
}

// orthogonal_svd(data, rows, cols, weights_data, w_rows, w_cols, named_labels)
// → { rotation, eigenvalues, column_names }
static val orthogonal_svd(const val& data, int rows, int cols,
                           const val& weights_data, int w_rows, int w_cols,
                           const val& named_labels_val) {
    std::vector<std::string> labels = vecFromJSArray<std::string>(named_labels_val);
    return rotation_to_js(qe::orthogonal_svd(
        js_to_mat(data, rows, cols),
        js_to_mat(weights_data, w_rows, w_cols),
        labels));
}

// complete_rotation(data, rows, cols, named_axes_data, ax_rows, ax_cols, named_labels)
// → { rotation, eigenvalues, column_names }
static val complete_rotation(const val& data, int rows, int cols,
                              const val& axes_data, int ax_rows, int ax_cols,
                              const val& named_labels_val) {
    std::vector<std::string> labels = vecFromJSArray<std::string>(named_labels_val);
    return rotation_to_js(qe::complete_rotation(
        js_to_mat(data, rows, cols),
        js_to_mat(axes_data, ax_rows, ax_cols),
        labels));
}

// means_rotation(data, rows, cols, group_pairs)
// group_pairs: Array of { a: Int32Array, b: Int32Array }  (0-based row indices)
// → { rotation, eigenvalues, column_names }
static val means_rotation(const val& data, int rows, int cols,
                           const val& group_pairs_js) {
    int n_pairs = group_pairs_js["length"].as<int>();
    std::vector<qe::GroupPair> pairs;
    pairs.reserve(n_pairs);
    for (int i = 0; i < n_pairs; ++i) {
        val pair = group_pairs_js[i];
        std::vector<int> av = vecFromJSArray<int>(pair["a"]);
        std::vector<int> bv = vecFromJSArray<int>(pair["b"]);
        arma::uvec a(av.size()), b(bv.size());
        for (size_t j = 0; j < av.size(); ++j) a(j) = static_cast<arma::uword>(av[j]);
        for (size_t j = 0; j < bv.size(); ++j) b(j) = static_cast<arma::uword>(bv[j]);
        pairs.push_back({a, b});
    }
    return rotation_to_js(qe::means_rotation(js_to_mat(data, rows, cols), pairs));
}

// generalized_means_rotation(...)
// All index arrays (x1_cols, x_subset, y1_cols) are Int32Array of 0-based indices.
// Pass a zero-length Int32Array for x_subset to use all rows.
// Pass empty Float64Array / Int32Array for ignored y params when has_y=false.
// → { rotation, eigenvalues, column_names }
static val generalized_means_rotation(
    const val& V_data,  int V_rows,  int V_cols,
    const val& xm_data, int xm_rows, int xm_cols,
    const val& x_target_data,
    const val& x1_cols_data,
    bool x_categorical, int x_n_groups,
    const val& x_subset_data,
    bool has_y,
    const val& ym_data, int ym_rows, int ym_cols,
    const val& y_target_data,
    const val& y1_cols_data,
    bool y_categorical, int y_n_groups,
    int n_lambda, int k_folds, double lasso_eps
) {
    auto js_to_uvec = [](const val& v) {
        std::vector<int> iv = vecFromJSArray<int>(v);
        arma::uvec out(iv.size());
        for (size_t i = 0; i < iv.size(); ++i)
            out(i) = static_cast<arma::uword>(iv[i]);
        return out;
    };
    auto js_to_arma_vec = [](const val& v) {
        std::vector<double> dv = vecFromJSArray<double>(v);
        return arma::vec(dv.data(), dv.size());
    };

    qe::GeneralizedRotationParams p;
    p.x_model_matrix = js_to_mat(xm_data, xm_rows, xm_cols);
    p.x_target       = js_to_arma_vec(x_target_data);
    p.x1_cols        = js_to_uvec(x1_cols_data);
    p.x_categorical  = x_categorical;
    p.x_n_groups     = static_cast<arma::uword>(x_n_groups);
    p.x_subset       = js_to_uvec(x_subset_data);  // empty → use all rows
    p.has_y          = has_y;
    p.y_model_matrix = js_to_mat(ym_data, ym_rows, ym_cols);
    p.y_target       = js_to_arma_vec(y_target_data);
    p.y1_cols        = js_to_uvec(y1_cols_data);
    p.y_categorical  = y_categorical;
    p.y_n_groups     = static_cast<arma::uword>(y_n_groups);
    p.n_lambda       = n_lambda;
    p.k_folds        = k_folds;
    p.lasso_eps      = lasso_eps;

    return rotation_to_js(
        qe::generalized_means_rotation(js_to_mat(V_data, V_rows, V_cols), p));
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

// CCD window-size estimation.
// codes_data: full code matrix (n_rows × n_codes, row-major, flat).
// group_sizes: number of rows in each conversation subset.
// row_indices: 0-based row indices into the code matrix, concatenated in
//              conversation order (length == sum(group_sizes)); this lets the
//              caller pass non-contiguous conversations (cf. parseData's
//              convoGroups) without pre-copying subsets.
// Returns { window_size, peak_lag, lag, frob, frob_sq_unbiased,
//           frob_unbiased_signed, total_weight }.
static val ccd_window(
    const val& codes_data, int n_rows, int n_codes,
    const val& group_sizes_val, const val& row_indices_val,
    int max_window, int min_overlap
) {
    arma::mat codes = js_to_mat(codes_data, n_rows, n_codes);
    std::vector<int> group_sizes = vecFromJSArray<int>(group_sizes_val);
    std::vector<int> row_indices = vecFromJSArray<int>(row_indices_val);

    std::vector<arma::mat> conversations;
    conversations.reserve(group_sizes.size());
    std::size_t cursor = 0;
    for (int sz : group_sizes) {
        arma::uvec rows(sz);
        for (int i = 0; i < sz; ++i) rows[i] = static_cast<arma::uword>(row_indices[cursor++]);
        conversations.push_back(codes.rows(rows));
    }

    qe::CCDResult res = qe::ccd_window(conversations, max_window, min_overlap);

    auto vec_to_js = [](const arma::vec& v) {
        std::vector<double> tmp(v.memptr(), v.memptr() + v.n_elem);
        return val::array(tmp.begin(), tmp.end());
    };

    val result = val::object();
    result.set("window_size",          res.window_size);
    result.set("peak_lag",             res.peak_lag);
    result.set("lag",                  vec_to_js(res.lag));
    result.set("frob",                 vec_to_js(res.frob));
    result.set("frob_sq_unbiased",     vec_to_js(res.frob_sq_unbiased));
    result.set("frob_unbiased_signed", vec_to_js(res.frob_unbiased_signed));
    result.set("total_weight",         vec_to_js(res.total_weight));
    return result;
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
    function("ena_correlation",                       &ena_correlation);
    function("group_stats",                           &group_stats);
    function("node_positions",                        &node_positions);
    function("directed_node_positions",               &directed_node_positions);
    function("directed_node_positions_combine_pairs", &directed_node_positions_combine_pairs);

    // Accumulation
    function("connection_matrix",                     &connection_matrix);
    function("accumulate_stanza",                     &accumulate_stanza);
    function("row_connections",                       &row_connections);
    function("rolling_window_sum",                    &rolling_window_sum);
    function("flat_index",                            &flat_index);
    function("accumulate_unit",                       &accumulate_unit);
    function("accumulate_unit_with_rows",             &accumulate_unit_with_rows);
    function("accumulate_tensor_unit",                &accumulate_tensor_unit);
    function("aggregate_row_connections",             &aggregate_row_connections);

    // Rotation
    function("ena_svd",                               &ena_svd);
    function("deflate",                               &deflate);
    function("orthogonal_svd",                        &orthogonal_svd);
    function("complete_rotation",                     &complete_rotation);
    function("means_rotation",                        &means_rotation);
    function("generalized_means_rotation",            &generalized_means_rotation);

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

    // CCD window-size estimation
    function("ccd_window",                            &ccd_window);
}
