// pylibqe — nanobind bindings for libqe
//
// Exposes four submodules that mirror the four C++ headers:
//   pylibqe.adjacency     — vector/matrix upper-triangle utilities
//   pylibqe.normalization — row-wise normalization
//   pylibqe.modeling      — centering, correlation, node-position solvers
//   pylibqe.accumulation  — stanza window, rolling sum, co-occurrence
//
// All matrix arguments are accepted as 2-D numpy float64 arrays (C-contiguous).
// All vector arguments are accepted as 1-D numpy float64 arrays.
// Return values are always freshly allocated numpy arrays (owned by Python).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <armadillo>
#include <libqe/libqe.hpp>

namespace nb = nanobind;
using namespace nb::literals;

// ── Armadillo ↔ numpy conversion helpers ──────────────────────────────────────
//
// Armadillo stores matrices in column-major order; numpy defaults to row-major.
// We always copy on the boundary so callers never see stale memory.

using NpMat  = nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using NpVec  = nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// numpy 2-D (rows × cols, C-order) → arma::mat (col-major)
static arma::mat to_mat(NpMat arr) {
    arma::mat m(arr.shape(0), arr.shape(1));
    for (size_t i = 0; i < arr.shape(0); ++i)
        for (size_t j = 0; j < arr.shape(1); ++j)
            m(i, j) = arr(i, j);
    return m;
}

// numpy 1-D → arma::rowvec (copy)
static arma::rowvec to_rowvec(NpVec arr) {
    return arma::rowvec(const_cast<double*>(arr.data()), arr.shape(0), /*copy=*/true);
}

// numpy 1-D → arma::vec (copy)
static arma::vec to_vec(NpVec arr) {
    return arma::vec(const_cast<double*>(arr.data()), arr.shape(0), /*copy=*/true);
}

// arma::mat → numpy 2-D (rows × cols, C-order, Python owns the copy)
static nb::ndarray<nb::numpy, double, nb::ndim<2>> from_mat(const arma::mat& m) {
    size_t shape[2] = {m.n_rows, m.n_cols};
    double* data = new double[m.n_rows * m.n_cols];
    for (size_t i = 0; i < m.n_rows; ++i)
        for (size_t j = 0; j < m.n_cols; ++j)
            data[i * m.n_cols + j] = m(i, j);
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return nb::ndarray<nb::numpy, double, nb::ndim<2>>(data, 2, shape, owner);
}

// arma::rowvec → numpy 1-D (Python owns the copy)
static nb::ndarray<nb::numpy, double, nb::ndim<1>> from_rowvec(const arma::rowvec& v) {
    size_t shape[1] = {v.n_elem};
    double* data = new double[v.n_elem];
    std::copy(v.begin(), v.end(), data);
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return nb::ndarray<nb::numpy, double, nb::ndim<1>>(data, 1, shape, owner);
}

// arma::umat → numpy 2-D int64 (Python owns the copy)
static nb::ndarray<nb::numpy, int64_t, nb::ndim<2>> from_umat(const arma::umat& m) {
    size_t shape[2] = {m.n_rows, m.n_cols};
    int64_t* data = new int64_t[m.n_rows * m.n_cols];
    for (size_t i = 0; i < m.n_rows; ++i)
        for (size_t j = 0; j < m.n_cols; ++j)
            data[i * m.n_cols + j] = static_cast<int64_t>(m(i, j));
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<int64_t*>(p); });
    return nb::ndarray<nb::numpy, int64_t, nb::ndim<2>>(data, 2, shape, owner);
}

// ── Module definition ─────────────────────────────────────────────────────────

NB_MODULE(_pylibqe, m) {
    m.doc() = "pylibqe — Python bindings for libqe ENA math primitives";

    // ── adjacency ─────────────────────────────────────────────────────────────
    auto adj = m.def_submodule("adjacency",
        "Adjacency utilities: upper-triangle conversions, index helpers");

    adj.def("choose_two", &qe::choose_two, "n"_a,
        "Return n*(n-1)/2 (number of unique pairs in a set of n elements).");

    adj.def("tri_indices", [](int len, int row) {
        return from_umat(qe::tri_indices(len, row));
    }, "len"_a, "row"_a = -1,
        "Upper-triangle index pairs for an len×len matrix.\n"
        "row=-1: 2×k array of [row_idx; col_idx]  "
        "row=0: row indices only  "
        "row=1: col indices only");

    adj.def("vector_to_upper_tri", [](NpVec v) {
        // wrap 1-D input as a 1×n row vector for vector_to_upper_tri
        arma::mat vm(1, v.shape(0));
        for (size_t j = 0; j < v.shape(0); ++j) vm(0, j) = v(j);
        return from_rowvec(qe::vector_to_upper_tri(vm));
    }, "v"_a,
        "Compute pairwise products v[j]*v[i] for all j < i (upper-triangle vector).");

    adj.def("directed_to_upper_tri", [](NpVec v) {
        return from_rowvec(qe::directed_to_upper_tri(to_vec(v)));
    }, "v"_a,
        "Fold a directed n*n flat vector to upper-triangle by summing A→B + B→A.");

    adj.def("adjacency_matrix_to_vector", [](NpMat x, bool full) {
        return from_rowvec(qe::adjacency_matrix_to_vector(to_mat(x), full));
    }, "x"_a, "full"_a = true,
        "Flatten an adjacency matrix to a vector.\n"
        "full=True: full n*n vector (directed)  "
        "full=False: upper-triangle only (undirected)");

    adj.def("svector_to_upper_tri", [](std::vector<std::string> v) {
        return qe::svector_to_upper_tri(v);
    }, "v"_a,
        "Return 'A & B' pair names for every upper-triangle position.\n"
        "Example: ['X','Y','Z'] → ['X & Y', 'X & Z', 'Y & Z']");

    // ── normalization ─────────────────────────────────────────────────────────
    auto nrm = m.def_submodule("normalization",
        "Row-wise L2 normalization");

    nrm.def("sphere_norm", [](NpMat mat) {
        return from_mat(qe::sphere_norm(to_mat(mat)));
    }, "m"_a,
        "Divide each row by its own L2 norm (project onto unit hypersphere). "
        "Zero rows are left unchanged.");

    nrm.def("skip_sphere_norm", [](NpMat mat) {
        return from_mat(qe::skip_sphere_norm(to_mat(mat)));
    }, "m"_a,
        "Divide all entries by the *largest* row L2 norm. "
        "Preserves relative magnitudes across rows.");

    // ── modeling ──────────────────────────────────────────────────────────────
    auto mod = m.def_submodule("modeling",
        "ENA modeling: centering, correlation, node-position solvers");

    // Python-side NodePositions: stores arrays eagerly as nb::object so that
    // def_ro can return them without lifetime-policy conflicts with the capsule
    // owners created by from_mat().
    struct PyNodePositions {
        nb::object nodes, centroids, weights, points;
    };
    auto make_py_np = [](const qe::NodePositions& r) {
        PyNodePositions p;
        p.nodes     = nb::cast(from_mat(r.nodes));
        p.centroids = nb::cast(from_mat(r.centroids));
        p.weights   = nb::cast(from_mat(r.weights));
        p.points    = nb::cast(from_mat(r.points));
        return p;
    };

    nb::class_<PyNodePositions>(mod, "NodePositions",
        "Result struct returned by node-position solvers.\n\n"
        "Attributes\n"
        "----------\n"
        "nodes     : ndarray (n_codes × n_dims)   — solved node coordinates\n"
        "centroids : ndarray (n_units × n_dims)   — unit centroid positions\n"
        "weights   : ndarray (n_units × n_codes)  — half-edge weight per node\n"
        "points    : ndarray (n_units × n_dims)   — input rotated points (echo)")
        .def_ro("nodes",     &PyNodePositions::nodes)
        .def_ro("centroids", &PyNodePositions::centroids)
        .def_ro("weights",   &PyNodePositions::weights)
        .def_ro("points",    &PyNodePositions::points)
        .def("__repr__", [](const PyNodePositions& p) {
            auto nodes_arr = nb::cast<nb::ndarray<double, nb::ndim<2>>>(p.nodes);
            auto cents_arr = nb::cast<nb::ndarray<double, nb::ndim<2>>>(p.centroids);
            return std::string("<NodePositions nodes=")
                + std::to_string(nodes_arr.shape(0)) + "x"
                + std::to_string(nodes_arr.shape(1)) + " centroids="
                + std::to_string(cents_arr.shape(0)) + "x"
                + std::to_string(cents_arr.shape(1)) + ">";
        });

    mod.def("center_data", [](NpMat values) {
        return from_mat(qe::center_data(to_mat(values)));
    }, "values"_a,
        "Subtract column means (center each column to zero).");

    mod.def("ena_correlation", [](NpMat points, NpMat centroids, double conf_level) {
        return from_mat(qe::ena_correlation(to_mat(points), to_mat(centroids), conf_level));
    }, "points"_a, "centroids"_a, "conf_level"_a = 0.95,
        "Pearson correlation + CI between unit points and centroids.\n"
        "Returns (n_dims × 3) array: columns are [r, ci_lower, ci_upper].");

    mod.def("lws_lsq_positions", [&make_py_np](NpMat adj_mats, NpMat t, int num_dims) {
        return make_py_np(qe::lws_lsq_positions(to_mat(adj_mats), to_mat(t), num_dims));
    }, "adj_mats"_a, "t"_a, "num_dims"_a,
        "Multiobjective least-squares node positions for undirected ENA.");

    mod.def("directed_node_positions", [&make_py_np](NpMat line_weights, NpMat points, int num_dims) {
        return make_py_np(qe::directed_node_positions(
            to_mat(line_weights), to_mat(points), num_dims));
    }, "line_weights"_a, "points"_a, "num_dims"_a,
        "Least-squares node positions for directed (ordered) ENA.");

    mod.def("directed_node_positions_ground_response",
        [&make_py_np](NpMat line_weights, NpMat points, int num_dims) {
            return make_py_np(qe::directed_node_positions_ground_response(
                to_mat(line_weights), to_mat(points), num_dims));
        }, "line_weights"_a, "points"_a, "num_dims"_a,
        "Directed node positions with paired ground+response rows combined before solving.");

    // ── accumulation ──────────────────────────────────────────────────────────
    auto acc = m.def_submodule("accumulation",
        "Network accumulation primitives");

    acc.def("calculate_adjacency_matrix", [](NpVec ground, NpVec response,
                                              double response_weight, bool ordered) {
        return from_mat(qe::calculate_adjacency_matrix(
            to_rowvec(ground), to_rowvec(response), response_weight, ordered));
    }, "ground"_a, "response"_a, "response_weight"_a = 1.0, "ordered"_a = true,
        "Connection matrix for one ground+response event pair.\n"
        "ordered=True (directed):  ground→response cross-product\n"
        "ordered=False (undirected): symmetric outer-product");

    acc.def("stanza_window", [](NpMat codes, int window_back,
                                 int window_forward, bool binary) {
        return from_mat(qe::stanza_window(
            to_mat(codes), window_back, window_forward, binary));
    }, "codes"_a, "window_back"_a = 1, "window_forward"_a = 0, "binary"_a = true,
        "Traditional stanza-window accumulation (rENA model).\n"
        "For each row, accumulates co-occurrences over the surrounding window "
        "and applies back/forward-reference corrections.\n"
        "Returns (n_rows × choose_two(n_codes)) matrix.");

    acc.def("rows_to_co_occurrences", [](NpMat codes, bool binary) {
        return from_mat(qe::rows_to_co_occurrences(to_mat(codes), binary));
    }, "codes"_a, "binary"_a = true,
        "Per-row upper-triangle co-occurrence.\n"
        "Each row is processed independently (no cross-row accumulation).\n"
        "Returns (n_rows × choose_two(n_codes)) matrix.");

    acc.def("rolling_window_sum", [](NpMat codes, int window_size) {
        return from_mat(qe::rolling_window_sum(to_mat(codes), window_size));
    }, "codes"_a, "window_size"_a = 1,
        "Rolling backward window sum of a raw code matrix.\n"
        "Row k = sum of rows [max(0, k - window_size + 1), k].\n"
        "Returns matrix of same shape as input.");

    acc.def("calculate_1d_index", [](std::vector<int> indices, std::vector<int> dims) {
        return qe::calculate_1d_index(indices, dims);
    }, "indices"_a, "dims"_a,
        "Linear index into a column-major multi-dimensional array "
        "(equivalent to sub2ind with Fortran/column-major ordering).");
}
