// qe — nanobind bindings for libqe
//
// Exposes these submodules (C++ header in parentheses):
//   qe.adjacency     — vector/matrix upper-triangle utilities (libqe/adjacency.hpp)
//   qe.normalization — row-wise normalization (libqe/normalization.hpp)
//   qe.modeling      — centering, confidence intervals, group statistics
//                      (libqe/stats.hpp); correlation, node-position solvers
//                      (libena/positions.hpp)
//   qe.accumulation  — stanza window, rolling sum, co-occurrence, tensor
//                      accumulation, weight models (libtma/accumulation.hpp)
//   qe.rotation      — SVD, means rotation, generalized rotation
//                      (libena/rotation.hpp, libena/generalized_rotation.hpp)
//   qe.door          — lookback / EMA temporal pooling (libqe/door.hpp)
//   qe.trajectory    — curve fitting, derivatives, distance, following
//                      (libqe/trajectory*.hpp)
//   qe.ccd           — moving-window size estimate (libena/ccd.hpp)
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
    m.doc() = "qe — Python bindings for libqe ENA math primitives";

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

    mod.def("node_positions", [&make_py_np](NpMat adj_mats, NpMat t, int num_dims) {
        arma::mat am = to_mat(adj_mats);
        arma::mat tv = to_mat(t);
        qe::require_finite(am, "adj_mats");
        qe::require_finite(tv, "t");
        return make_py_np(qe::node_positions(am, tv, num_dims));
    }, "adj_mats"_a, "t"_a, "num_dims"_a,
        "Multiobjective least-squares node positions for undirected ENA.");

    mod.def("directed_node_positions", [&make_py_np](NpMat line_weights, NpMat points, int num_dims) {
        arma::mat lw = to_mat(line_weights);
        arma::mat pt = to_mat(points);
        qe::require_finite(lw, "line_weights");
        qe::require_finite(pt, "points");
        return make_py_np(qe::directed_node_positions(lw, pt, num_dims));
    }, "line_weights"_a, "points"_a, "num_dims"_a,
        "Least-squares node positions for directed (ordered) ENA.");

    mod.def("directed_node_positions_combine_pairs",
        [&make_py_np](NpMat line_weights, NpMat points, int num_dims) {
            arma::mat lw = to_mat(line_weights);
            arma::mat pt = to_mat(points);
            qe::require_finite(lw, "line_weights");
            qe::require_finite(pt, "points");
            return make_py_np(qe::directed_node_positions(lw, pt, num_dims, true));
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
                                 int window_forward, bool binary, bool ordered) {
        return from_mat(qe::accumulate_stanza(
            to_mat(codes), window_back, window_forward, binary, ordered));
    }, "codes"_a, "window_back"_a = 1, "window_forward"_a = 0, "binary"_a = true,
       "ordered"_a = false,
        "Stanza-window accumulation.\n"
        "ordered=False (default): upper-tri co-occurrences, returns (n_rows × choose_two(n_codes)).\n"
        "ordered=True: directed — focal row as response, prior window rows as ground,\n"
        "returns (n_rows × n_codes²). window_forward is ignored when ordered=True.");

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

    // Weight model: a name ("binary" | "product" | "sqrt" | "log1p" | "log") or
    // the legacy bool binary flag (True -> "binary", False -> "product").
    auto weight_model = [](const std::variant<bool, std::string>& w) {
        return std::holds_alternative<bool>(w)
            ? qe::weight_model_from_bool(std::get<bool>(w))
            : qe::weight_model_from_string(std::get<std::string>(w));
    };

    acc.def("aggregate_row_connections",
        [weight_model](NpMat row_conn, int n_codes, bool ordered,
                       std::variant<bool, std::string> weight) {
            return from_rowvec(qe::aggregate_row_connections(
                to_mat(row_conn), n_codes, ordered, weight_model(weight)));
        },
        "row_conn"_a, "n_codes"_a, "ordered"_a = false, "weight"_a = "binary",
        "Aggregate apply_tensor_unit's row_connection_counts into a unit vector,\n"
        "matching tma's R aggregation (as.unordered + colSums.ena.matrix), with the\n"
        "weight model applied per row before the sum (= rENA's weight.by).\n\n"
        "row_conn : ndarray 2-D  (n_response_rows x n_codes^2) per-row directed counts\n"
        "n_codes  : int          number of codes p\n"
        "ordered  : bool         True = directed p^2 rows; False = fold each row to choose(p,2)\n"
        "weight   : str | bool   'binary' (unordered: clamp to 1; ordered: raw counts),\n"
        "                        'product' (raw counts), 'sqrt', 'log1p' (alias 'log');\n"
        "                        a bool is the legacy binary flag (True = 'binary',\n"
        "                        False = 'product')\n\n"
        "Returns a 1-D ndarray of length p^2 (ordered) or choose(p,2) (unordered).");

    acc.def("finalize_row_connections",
        [weight_model](NpMat row_conn, int n_codes, bool ordered,
                       std::variant<bool, std::string> weight) {
            return from_mat(qe::finalize_row_connections(
                to_mat(row_conn), n_codes, ordered, weight_model(weight)));
        },
        "row_conn"_a, "n_codes"_a, "ordered"_a = false, "weight"_a = "binary",
        "Per-row step of aggregate_row_connections (fold + weight model) without\n"
        "the sum; its rows sum to aggregate_row_connections.\n\n"
        "Returns a 2-D ndarray (n_response_rows x p^2 ordered, or x choose(p,2)).");

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
        arma::mat pt = to_mat(points);
        qe::require_finite(pt, "points");
        return make_py_rot(qe::ena_svd(pt));
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
        "For mutually orthogonal axes this equals sequential deflation.\n"
        "On rank-deficient data (e.g. an all-zero connection column) the\n"
        "trailing axes that come from the deflated data's null space are\n"
        "orthogonalised against the named axes, so the rotation is orthonormal\n"
        "whenever the named axes are.\n\n"
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

    rot.def("generalized_means_rotation",
        [make_py_rot](NpMat V,
                      NpMat x_model, NpVec x_target,
                      std::vector<int> x1_cols,
                      bool x_categorical, int x_n_groups,
                      std::vector<int> x_subset,
                      bool has_y,
                      NpMat y_model, NpVec y_target,
                      std::vector<int> y1_cols,
                      bool y_categorical, int y_n_groups,
                      int n_lambda, int k_folds, double lasso_eps) {
            auto to_uvec = [](const std::vector<int>& v) {
                arma::uvec out(v.size());
                for (std::size_t i = 0; i < v.size(); ++i)
                    out(i) = static_cast<arma::uword>(v[i]);
                return out;
            };

            qe::GeneralizedRotationParams p;
            p.x_model_matrix = to_mat(x_model);
            p.x_target       = to_vec(x_target);
            p.x1_cols        = to_uvec(x1_cols);
            p.x_categorical  = x_categorical;
            p.x_n_groups     = static_cast<arma::uword>(x_n_groups);
            p.x_subset       = to_uvec(x_subset);   // empty list → use all rows
            p.has_y          = has_y;
            p.y_model_matrix = to_mat(y_model);
            p.y_target       = to_vec(y_target);
            p.y1_cols        = to_uvec(y1_cols);
            p.y_categorical  = y_categorical;
            p.y_n_groups     = static_cast<arma::uword>(y_n_groups);
            p.n_lambda       = n_lambda;
            p.k_folds        = k_folds;
            p.lasso_eps      = lasso_eps;

            return make_py_rot(qe::generalized_means_rotation(to_mat(V), p));
        },
        "V"_a, "x_model"_a, "x_target"_a, "x1_cols"_a,
        "x_categorical"_a, "x_n_groups"_a, "x_subset"_a,
        "has_y"_a,
        "y_model"_a, "y_target"_a, "y1_cols"_a,
        "y_categorical"_a, "y_n_groups"_a,
        "n_lambda"_a=50, "k_folds"_a=5, "lasso_eps"_a=0.01,
        "Generalized Means Rotation (GMR) with Lasso-based covariate adjustment.\n\n"
        "Mirrors rENA's ``ena.rotate.by.generalized()``. The x axis is the direction\n"
        "in ENA space most explained by ``x_target`` after controlling for covariates\n"
        "via Lasso (coordinate-descent, k-fold CV). The y axis is either a second GMR\n"
        "axis (``has_y=True``) or the leading SVD of the x-deflated space.\n\n"
        "All index lists (``x1_cols``, ``x_subset``, ``y1_cols``) are **0-based int**.\n"
        "Pass an empty list ``[]`` for ``x_subset`` to use all rows.\n"
        "Pass empty arrays/lists for all ``y_*`` arguments when ``has_y=False``.\n\n"
        "Parameters\n----------\n"
        "V            : ndarray (n_units × n_dims)   ENA point matrix\n"
        "x_model      : ndarray (n_units × p)        model matrix for x axis\n"
        "x_target     : ndarray (n_units,)            target variable\n"
        "x1_cols      : list[int]  0-based target column indices in x_model\n"
        "x_categorical: bool\n"
        "x_n_groups   : int   number of groups (only used when x_categorical=True)\n"
        "x_subset     : list[int]  0-based row indices; [] = use all rows\n"
        "has_y        : bool  True → compute second GMR axis; False → SVD fallback\n"
        "y_model      : ndarray (n_units × p)  (ignored when has_y=False)\n"
        "y_target     : ndarray (n_units,)     (ignored when has_y=False)\n"
        "y1_cols      : list[int]              (ignored when has_y=False)\n"
        "y_categorical: bool                   (ignored when has_y=False)\n"
        "y_n_groups   : int                    (ignored when has_y=False)\n"
        "n_lambda     : int    lambda path length (default 50)\n"
        "k_folds      : int    CV folds for lambda selection (default 5)\n"
        "lasso_eps    : float  lambda_min = lasso_eps * lambda_max (default 0.01)\n\n"
        "Returns RotationResult with column_names = ['GMR1', 'GMR2'|'SVD2',\n"
        "'SVD3', ..., 'SVDp'].\n\n"
        "Reference: Zhiqiang Cai, commit 46776a1981a90b3a3b2861ed1010e9dbb7acf901.");

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

    // ── CCD (cross-covariance decay window-size estimation) ──────────────────
    auto ccd_mod = m.def_submodule("ccd",
        "Cross-covariance decay window-size estimation");

    ccd_mod.def("ccd_window", [](nb::list conversations, int max_window, int min_overlap) {
        std::vector<arma::mat> convos;
        convos.reserve(conversations.size());
        for (auto item : conversations) {
            NpMat arr = nb::cast<NpMat>(item);
            convos.push_back(to_mat(arr));
        }
        qe::CCDResult res = qe::ccd_window(convos, max_window, min_overlap);
        nb::dict out;
        out["window_size"]          = res.window_size;
        out["peak_lag"]             = res.peak_lag;
        out["lag"]                  = from_vec(res.lag);
        out["frob"]                 = from_vec(res.frob);
        out["frob_sq_unbiased"]     = from_vec(res.frob_sq_unbiased);
        out["frob_unbiased_signed"] = from_vec(res.frob_unbiased_signed);
        out["total_weight"]         = from_vec(res.total_weight);
        return out;
    }, "conversations"_a, "max_window"_a = 20, "min_overlap"_a = 10);
}
