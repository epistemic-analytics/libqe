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
static val network_to_vector(const val& data, int rows, int cols,
                                       bool full) {
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

// node_positions(adj_data, adj_rows, adj_cols,
//                t_data,   t_rows,   t_cols, num_dims)
// → { nodes, centroids, weights, points }  (each a matrix object)
static val node_positions(const val& adj_data, int adj_rows, int adj_cols,
                              const val& t_data,   int t_rows,   int t_cols,
                              int num_dims) {
    qe::NodePositions r = qe::node_positions(
        js_to_mat(adj_data, adj_rows, adj_cols),
        js_to_mat(t_data,   t_rows,   t_cols),
        num_dims);
    val result = val::object();
    result.set("nodes",     mat_to_js(r.nodes));
    result.set("centroids", mat_to_js(r.centroids));
    result.set("weights",   mat_to_js(r.weights));
    result.set("points",    mat_to_js(r.points));
    return result;
}

// directed_node_positions — same signature as node_positions
static val directed_node_positions(const val& lw_data, int lw_rows, int lw_cols,
                                   const val& pt_data, int pt_rows, int pt_cols,
                                   int num_dims) {
    qe::NodePositions r = qe::directed_node_positions(
        js_to_mat(lw_data, lw_rows, lw_cols),
        js_to_mat(pt_data, pt_rows, pt_cols),
        num_dims);
    val result = val::object();
    result.set("nodes",     mat_to_js(r.nodes));
    result.set("centroids", mat_to_js(r.centroids));
    result.set("weights",   mat_to_js(r.weights));
    result.set("points",    mat_to_js(r.points));
    return result;
}

// ── Accumulation ─────────────────────────────────────────────────────────────

// accumulate_stanza(data, rows, cols, window_back, window_forward, binary, ordered)
// ordered=false → { data, rows, choose_two(cols) }
// ordered=true  → { data, rows, cols² }
static val accumulate_stanza(const val& data, int rows, int cols,
                         int window_back, int window_forward, bool binary,
                         bool ordered) {
    return mat_to_js(qe::accumulate_stanza(
        js_to_mat(data, rows, cols), window_back, window_forward, binary, ordered));
}

// row_connections(data, rows, cols, binary) → { data, rows, cols }
static val row_connections(const val& data, int rows, int cols,
                                   bool binary) {
    return mat_to_js(qe::row_connections(
        js_to_mat(data, rows, cols), binary));
}

// rolling_window_sum(data, rows, cols, window_size) → { data, rows, cols }
static val rolling_window_sum(const val& data, int rows, int cols,
                               int window_size) {
    return mat_to_js(qe::rolling_window_sum(
        js_to_mat(data, rows, cols), window_size));
}

// flat_index(indices_array, dims_array) → int
static int flat_index(const val& indices_val, const val& dims_val) {
    return qe::flat_index(
        vecFromJSArray<int>(indices_val),
        vecFromJSArray<int>(dims_val));
}

// ── Embind registrations ──────────────────────────────────────────────────────

EMSCRIPTEN_BINDINGS(libqe) {
    // Adjacency
    function("connection_indices",         &connection_indices);
    function("code_connections",           &code_connections);
    function("fold_directed_network",      &fold_directed_network);
    function("network_to_vector",          &network_to_vector);
    function("connection_names",           &connection_names);

    // Normalization
    function("normalize_networks",         &normalize_networks);
    function("scale_networks",             &scale_networks);

    // Modeling
    function("center_points",              &center_points);
    function("mean_ci",                    &mean_ci);
    function("outlier_ci",                 &outlier_ci);
    function("node_positions",             &node_positions);
    function("directed_node_positions",    &directed_node_positions);

    // Accumulation
    function("accumulate_stanza",          &accumulate_stanza);
    function("row_connections",            &row_connections);
    function("rolling_window_sum",         &rolling_window_sum);
    function("flat_index",                 &flat_index);
}
