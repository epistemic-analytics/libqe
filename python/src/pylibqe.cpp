// pylibqe — nanobind bindings for libqe
//
// Exposes five submodules that mirror the five C++ headers:
//   pylibqe.adjacency     — vector/matrix upper-triangle utilities
//   pylibqe.normalization — row-wise normalization
//   pylibqe.modeling      — centering, correlation, node-position solvers
//   pylibqe.accumulation  — stanza window, rolling sum, co-occurrence
//   pylibqe.rotation      — SVD, means rotation, generalized-rotation tail
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

// int32 matrix type for context_lookup (arma::imat)
using NpIMat = nb::ndarray<int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// arma::imat → NpIMat input converter
static arma::imat to_imat(NpIMat arr) {
    arma::imat m(arr.shape(0), arr.shape(1));
    for (size_t i = 0; i < arr.shape(0); ++i)
        for (size_t j = 0; j < arr.shape(1); ++j)
            m(i, j) = arr(i, j);
    return m;
}

// arma::vec (column vector) → numpy 1-D
static nb::ndarray<nb::numpy, double, nb::ndim<1>> from_vec(const arma::vec& v) {
    size_t shape[1] = {v.n_elem};
    double* data = new double[v.n_elem];
    std::copy(v.begin(), v.end(), data);
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return nb::ndarray<nb::numpy, double, nb::ndim<1>>(data, 1, shape, owner);
}

// arma::uvec → numpy 1-D int64
static nb::ndarray<nb::numpy, int64_t, nb::ndim<1>> from_uvec(const arma::uvec& v) {
    size_t shape[1] = {v.n_elem};
    int64_t* data = new int64_t[v.n_elem];
    for (size_t i = 0; i < v.n_elem; ++i) data[i] = static_cast<int64_t>(v[i]);
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<int64_t*>(p); });
    return nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>(data, 1, shape, owner);
}

// ── Module definition ─────────────────────────────────────────────────────────

NB_MODULE(_pylibqe, m) {
    m.doc() = "pylibqe — Python bindings for libqe ENA math primitives";

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

    mod.def("center_points", [](NpMat values) {
        return from_mat(qe::center_points(to_mat(values)));
    }, "values"_a,
        "Subtract column means (center each column to zero).");

    mod.def("ena_correlation", [](NpMat points, NpMat centroids, double conf_level) {
        return from_mat(qe::ena_correlation(to_mat(points), to_mat(centroids), conf_level));
    }, "points"_a, "centroids"_a, "conf_level"_a = 0.95,
        "Pearson correlation + CI between unit points and centroids.\n"
        "Returns (n_dims × 3) array: columns are [r, ci_lower, ci_upper].");

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

    mod.def("node_positions", [&make_py_np](NpMat adj_mats, NpMat t, int num_dims) {
        return make_py_np(qe::node_positions(to_mat(adj_mats), to_mat(t), num_dims));
    }, "adj_mats"_a, "t"_a, "num_dims"_a,
        "Multiobjective least-squares node positions for undirected ENA.");

    mod.def("directed_node_positions", [&make_py_np](NpMat line_weights, NpMat points, int num_dims) {
        return make_py_np(qe::directed_node_positions(
            to_mat(line_weights), to_mat(points), num_dims));
    }, "line_weights"_a, "points"_a, "num_dims"_a,
        "Least-squares node positions for directed (ordered) ENA.");

    mod.def("directed_node_positions_combine_pairs",
        [&make_py_np](NpMat line_weights, NpMat points, int num_dims) {
            return make_py_np(qe::directed_node_positions(
                to_mat(line_weights), to_mat(points), num_dims, true));
        }, "line_weights"_a, "points"_a, "num_dims"_a,
        "Directed node positions with paired ground+response rows combined before solving.");

    // Python-side UnitNetworks and TensorNetworks
    struct PyUnitNetworks {
        nb::object networks;      // 1-D ndarray — flat connection vector
        nb::object row_networks;  // 2-D ndarray — per-response-row connections
    };
    struct PyTensorNetworks {
        nb::object connection_counts;      // 1-D ndarray
        nb::object row_connection_counts;  // 2-D ndarray
    };

    nb::class_<PyUnitNetworks>(m, "UnitNetworks",
        "Result of accumulate_unit_with_rows.\n\n"
        "Attributes\n----------\n"
        "networks     : ndarray 1-D — flat connection vector (p^2 or choose_two(p))\n"
        "row_networks : ndarray 2-D — per-response-row p^2 connection matrix")
        .def_ro("networks",     &PyUnitNetworks::networks)
        .def_ro("row_networks", &PyUnitNetworks::row_networks)
        .def("__repr__", [](const PyUnitNetworks& u) {
            auto rn = nb::cast<nb::ndarray<double, nb::ndim<2>>>(u.row_networks);
            return std::string("<UnitNetworks networks len=")
                + std::to_string(nb::cast<nb::ndarray<double, nb::ndim<1>>>(u.networks).shape(0))
                + " row_networks=" + std::to_string(rn.shape(0))
                + "x" + std::to_string(rn.shape(1)) + ">";
        });

    nb::class_<PyTensorNetworks>(m, "TensorNetworks",
        "Result of apply_tensor_unit.\n\n"
        "Attributes\n----------\n"
        "connection_counts     : ndarray 1-D — unit-level flat vector (p^2)\n"
        "row_connection_counts : ndarray 2-D — per-response-row p^2 matrix")
        .def_ro("connection_counts",     &PyTensorNetworks::connection_counts)
        .def_ro("row_connection_counts", &PyTensorNetworks::row_connection_counts)
        .def("__repr__", [](const PyTensorNetworks& t) {
            auto rcc = nb::cast<nb::ndarray<double, nb::ndim<2>>>(t.row_connection_counts);
            auto cc  = nb::cast<nb::ndarray<double, nb::ndim<1>>>(t.connection_counts);
            std::string s = "<TensorNetworks connection_counts len=";
            s += std::to_string(cc.shape(0));
            s += " row_connection_counts=";
            s += std::to_string(rcc.shape(0));
            s += "x";
            s += std::to_string(rcc.shape(1));
            s += ">";
            return s;
        });

    // ── accumulation ──────────────────────────────────────────────────────────
    auto acc = m.def_submodule("accumulation",
        "Network accumulation primitives");

    acc.def("connection_matrix", [](NpVec ground, NpVec response,
                                              double response_weight, bool ordered) {
        return from_mat(qe::connection_matrix(
            to_rowvec(ground), to_rowvec(response), response_weight, ordered));
    }, "ground"_a, "response"_a, "response_weight"_a = 1.0, "ordered"_a = true,
        "Connection matrix for one ground+response event pair.\n"
        "ordered=True (directed):  ground→response cross-product\n"
        "ordered=False (undirected): symmetric outer-product");

    acc.def("accumulate_stanza", [](NpMat codes, int window_back,
                                 int window_forward, bool binary) {
        return from_mat(qe::accumulate_stanza(
            to_mat(codes), window_back, window_forward, binary));
    }, "codes"_a, "window_back"_a = 1, "window_forward"_a = 0, "binary"_a = true,
        "Traditional stanza-window accumulation (rENA model).\n"
        "For each row, accumulates co-occurrences over the surrounding window "
        "and applies back/forward-reference corrections.\n"
        "Returns (n_rows × choose_two(n_codes)) matrix.");

    acc.def("row_connections", [](NpMat codes, bool binary) {
        return from_mat(qe::row_connections(to_mat(codes), binary));
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

    acc.def("flat_index", [](std::vector<int> indices, std::vector<int> dims) {
        return qe::flat_index(indices, dims);
    }, "indices"_a, "dims"_a,
        "Linear index into a column-major multi-dimensional array "
        "(equivalent to sub2ind with Fortran/column-major ordering).");

    acc.def("accumulate_unit",
        [](NpMat codes, std::vector<int> unit_rows, nb::object decay_fn, bool ordered) {
            auto cpp_decay = [&decay_fn](arma::vec distances) -> arma::vec {
                auto np_dist = from_vec(distances);
                auto result  = decay_fn(np_dist);
                return to_vec(nb::cast<NpVec>(result));
            };
            return from_rowvec(qe::accumulate_unit(
                to_mat(codes), unit_rows, cpp_decay, ordered));
        },
        "codes"_a, "unit_rows"_a, "decay_fn"_a, "ordered"_a = false,
        "Ground/response accumulation for one unit (tma model).\n\n"
        "Parameters\n----------\n"
        "codes     : ndarray (n_context_rows x n_codes)\n"
        "unit_rows : list[int]  0-based row indices belonging to this unit\n"
        "decay_fn  : callable(distances: ndarray 1-D) -> ndarray 1-D\n"
        "            Maps distance vector to weight vector.\n"
        "ordered   : bool  True = directed full matrix, False = undirected upper-tri\n\n"
        "Returns ndarray 1-D — flat connection vector.");

    acc.def("accumulate_unit_with_rows",
        [](NpMat codes, std::vector<int> unit_rows, nb::object decay_fn, bool ordered) {
            auto cpp_decay = [&decay_fn](int unit_row, arma::uvec ground_indices) -> arma::vec {
                auto np_gi  = from_uvec(ground_indices);
                auto result = decay_fn(unit_row, np_gi);
                return to_vec(nb::cast<NpVec>(result));
            };
            qe::UnitNetworks r = qe::accumulate_unit_with_rows(
                to_mat(codes), unit_rows, cpp_decay, ordered);
            PyUnitNetworks p;
            p.networks     = nb::cast(from_rowvec(r.networks));
            p.row_networks = nb::cast(from_mat(r.row_networks));
            return p;
        },
        "codes"_a, "unit_rows"_a, "decay_fn"_a, "ordered"_a = false,
        "Ground/response accumulation returning per-row connection data (tma model).\n\n"
        "decay_fn : callable(unit_row: int, ground_indices: ndarray int64 1-D) -> ndarray float64 1-D\n"
        "Returns UnitNetworks with .networks (1-D) and .row_networks (2-D).");

    acc.def("apply_tensor_unit",
        [](NpVec tensor, std::vector<int> dims,
           std::vector<int> dims_sender, std::vector<int> dims_receiver,
           std::vector<int> dims_mode,
           NpIMat context_lookup, std::vector<int> unit_rows,
           NpMat codes, NpVec times, bool ordered) {
            qe::TensorNetworks r = qe::apply_tensor_unit(
                to_vec(tensor), dims, dims_sender, dims_receiver, dims_mode,
                to_imat(context_lookup), unit_rows,
                to_mat(codes), to_vec(times), ordered);
            PyTensorNetworks p;
            p.connection_counts     = nb::cast(from_rowvec(r.connection_counts));
            p.row_connection_counts = nb::cast(from_mat(r.row_connection_counts));
            return p;
        },
        "tensor"_a, "dims"_a,
        "dims_sender"_a, "dims_receiver"_a, "dims_mode"_a,
        "context_lookup"_a, "unit_rows"_a,
        "codes"_a, "times"_a, "ordered"_a = true,
        "Tensor-based multi-modal accumulation for one unit (tma model).\n\n"
        "tensor         : ndarray 1-D  flat column-major tensor of weights/windows\n"
        "dims           : list[int]    shape of the tensor\n"
        "dims_sender    : list[int]    tensor axis indices for sender factors\n"
        "dims_receiver  : list[int]    tensor axis indices for receiver factors\n"
        "dims_mode      : list[int]    tensor axis indices for mode factors\n"
        "context_lookup : ndarray int32 2-D  (n_rows x n_factors) factor indices\n"
        "unit_rows      : list[int]    0-based response-row indices for this unit\n"
        "codes          : ndarray 2-D  (n_rows x n_codes) code matrix\n"
        "times          : ndarray 1-D  timestamp per context row\n"
        "ordered        : bool         True = directed, False = undirected upper-tri\n\n"
        "Returns TensorNetworks with .connection_counts (1-D) and .row_connection_counts (2-D).\n\n"
        "Default mode: when dims=[2] and tensor has 2 elements [weight, window], uses\n"
        "a simplified single-weight/window path (equivalent to tma's default tensor).");

    // ── rotation ──────────────────────────────────────────────────────────────
    auto rot = m.def_submodule("rotation",
        "Rotation primitives: SVD, deflation, orthogonal SVD, means rotation, "
        "generalized-rotation tail.");

    // Python-side RotationResult — same eager-cast pattern as PyNodePositions.
    struct PyRotationResult {
        nb::object rotation;       // 2-D ndarray (p × p)
        nb::object eigenvalues;    // 1-D ndarray (length p), == sdev^2
        std::vector<std::string> column_names;
    };
    auto make_py_rot = [](const qe::RotationResult& r) {
        PyRotationResult p;
        p.rotation     = nb::cast(from_mat(r.rotation));
        p.eigenvalues  = nb::cast(from_vec(r.eigenvalues));
        p.column_names = r.column_names;
        return p;
    };

    nb::class_<PyRotationResult>(m, "RotationResult",
        "Result struct returned by rotation routines.\n\n"
        "Attributes\n"
        "----------\n"
        "rotation     : ndarray (p × p)   — column j is rotation axis j\n"
        "eigenvalues  : ndarray (p,)      — sdev^2 from the underlying SVD\n"
        "column_names : list[str]         — labels for each column of rotation\n\n"
        "Eigenvalues match rENA's prcomp(...)$sdev^2 convention. For rotations\n"
        "that fix some named axes (means_rotation, complete_rotation), the\n"
        "eigenvalues for those leading columns are 0.")
        .def_ro("rotation",     &PyRotationResult::rotation)
        .def_ro("eigenvalues",  &PyRotationResult::eigenvalues)
        .def_ro("column_names", &PyRotationResult::column_names)
        .def("__repr__", [](const PyRotationResult& r) {
            auto rot_arr = nb::cast<nb::ndarray<double, nb::ndim<2>>>(r.rotation);
            std::string s = "<RotationResult rotation=";
            s += std::to_string(rot_arr.shape(0));
            s += "x";
            s += std::to_string(rot_arr.shape(1));
            s += " labels=[";
            for (std::size_t i = 0; i < r.column_names.size(); ++i) {
                if (i) s += ", ";
                s += r.column_names[i];
            }
            s += "]>";
            return s;
        });

    rot.def("ena_svd", [make_py_rot](NpMat points) {
        return make_py_rot(qe::ena_svd(to_mat(points)));
    }, "points"_a,
        "SVD rotation matching prcomp(retx=F, scale=F, center=F, tol=0).\n\n"
        "Caller is responsible for centering upstream. Eigenvalues are stored\n"
        "as sdev^2 to match rENA's ena.svd.\n\n"
        "Parameters\n----------\n"
        "points : ndarray (n_units × n_dims)\n\n"
        "Returns RotationResult with column_names = ['SVD1', ..., 'SVDp'].\n\n"
        "Sign convention: none. Signs come from LAPACK's SVD, matching rENA's\n"
        "long-standing behavior. A deterministic sign rule may be added later.");

    rot.def("deflate", [](NpMat data, NpVec axis) {
        return from_mat(qe::deflate(to_mat(data), to_vec(axis)));
    }, "data"_a, "axis"_a,
        "Project `data` onto the hyperplane orthogonal to a unit-norm axis:\n"
        "  data - (data @ axis) @ axis.T\n\n"
        "Caller is responsible for normalizing `axis`.\n\n"
        "Returns a matrix of the same shape as `data`.");

    rot.def("orthogonal_svd",
        [make_py_rot](NpMat data, NpMat weights,
                       std::vector<std::string> named_labels) {
            return make_py_rot(qe::orthogonal_svd(
                to_mat(data), to_mat(weights), named_labels));
        },
        "data"_a, "weights"_a, "named_labels"_a,
        "Orthonormalize named axes via QR, fill the rest from SVD.\n\n"
        "Mirrors rENA's orthogonal_svd() in ena.rotate.by.mean.R. The named\n"
        "axes in the OUTPUT are the orthonormalized Q columns, not the\n"
        "original `weights` columns — use complete_rotation() to keep the\n"
        "named axes verbatim.\n\n"
        "Parameters\n----------\n"
        "data         : ndarray (n_units × n_dims)\n"
        "weights      : ndarray (n_dims × k)   — columns are the named axes\n"
        "named_labels : list[str] of length k  — labels for the named axes\n\n"
        "Returns RotationResult with column_names = named_labels + ['SVD{k+1}'..'SVDp'].");

    rot.def("complete_rotation",
        [make_py_rot](NpMat data, NpMat named_axes,
                       std::vector<std::string> named_labels) {
            return make_py_rot(qe::complete_rotation(
                to_mat(data), to_mat(named_axes), named_labels));
        },
        "data"_a, "named_axes"_a, "named_labels"_a,
        "Keep named axes verbatim, fill remaining axes from an SVD of the\n"
        "data deflated by all named axes in parallel:\n"
        "  defA = data - data @ named_axes @ named_axes.T\n\n"
        "Mirrors the tail of ena.rotate.by.generalized (canonical version:\n"
        "commit 2c079126 on rENA origin/main). The deflation is *parallel*\n"
        "(each projection comes off the original data), matching rENA's\n"
        "literal expression `defA <- A - A %*% v1 %*% t(v1) - A %*% v2 %*% t(v2)`.\n"
        "For mutually orthogonal axes this equals sequential deflation.\n\n"
        "Caller is responsible for ensuring each column of `named_axes` is\n"
        "unit-norm. Orthonormality between columns is NOT assumed.\n\n"
        "Conventional labels for generalized rotation are 'GMR1', 'GMR2',\n"
        "then 'SVD{k+1}'..'SVDp'.");

    rot.def("means_rotation",
        [make_py_rot](NpMat points, nb::list group_pairs) {
            std::vector<qe::GroupPair> pairs;
            pairs.reserve(group_pairs.size());
            for (std::size_t i = 0; i < group_pairs.size(); ++i) {
                nb::sequence pair = nb::cast<nb::sequence>(group_pairs[i]);
                if (nb::len(pair) != 2) {
                    throw std::runtime_error(
                        "group_pairs[" + std::to_string(i) +
                        "] must be a length-2 sequence (a, b)");
                }
                auto seq_to_uvec = [](nb::handle h) {
                    auto seq = nb::cast<nb::sequence>(h);
                    arma::uvec out(nb::len(seq));
                    std::size_t k = 0;
                    for (nb::handle item : seq) {
                        out(k++) = nb::cast<arma::uword>(item);
                    }
                    return out;
                };
                pairs.push_back({ seq_to_uvec(pair[0]), seq_to_uvec(pair[1]) });
            }
            return make_py_rot(qe::means_rotation(to_mat(points), pairs));
        },
        "points"_a, "group_pairs"_a,
        "Means rotation matching ena.rotate.by.mean.\n\n"
        "For each group pair, computes a normalized mean-difference axis on\n"
        "the progressively-deflated data and finishes with orthogonal_svd.\n"
        "The input is column-centered first, matching rENA's\n"
        "  scale(data, scale=F, center=T)\n"
        "at the top of ena.rotate.by.mean.\n\n"
        "Parameters\n----------\n"
        "points      : ndarray (n_units × n_dims)\n"
        "group_pairs : list of length k; each element is (a, b) where a and\n"
        "              b are 0-based integer index sequences into `points`.\n\n"
        "Returns RotationResult with column_names = ['MR1', ..., 'MRk',\n"
        "'SVD{k+1}', ..., 'SVDp'].\n\n"
        "MATCH-RENA NOTE: no guard against zero-norm mean-difference vectors\n"
        "(latent bug carried forward from rENA verbatim).");
}
