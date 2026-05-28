// libqe Julia bindings via CxxWrap.jl
//
// Memory layout advantage
// -----------------------
// Julia matrices (Matrix{Float64}) are column-major — identical to Armadillo's
// layout.  Input matrices are therefore zero-copy: we build an arma::mat view
// directly over the Julia array's data pointer without any transposition.
//
// The return path does involve one copy (Armadillo result → std::vector<double>
// → Julia Vector{Float64}) but this is unavoidable when crossing the boundary.
// The Julia wrapper then calls reshape(), which is zero-copy in Julia.
//
// Callback functions (accumulate_unit, accumulate_unit_with_rows, apply_tensor)
// accept a jlcxx::JuliaFunction whose return value is extracted via Julia's C
// API.  The pattern is spelled out in full for accumulate_unit and follows
// directly for the others.

#include <jlcxx/jlcxx.hpp>
#include <jlcxx/functions.hpp>
#include <jlcxx/array.hpp>
#include <jlcxx/stl.hpp>

#include <armadillo>
#include <libqe/libqe.hpp>

#include <vector>
#include <string>
#include <stdexcept>

// ── Helpers ───────────────────────────────────────────────────────────────────

// Julia array (column-major) → zero-copy arma::mat view.
// CAUTION: the arma::mat must not outlive the Julia array.
static arma::mat view_mat(jlcxx::ArrayRef<double> data, int rows, int cols) {
    return arma::mat(data.data(), static_cast<arma::uword>(rows),
                                  static_cast<arma::uword>(cols),
                     /*copy_aux_mem=*/false, /*strict=*/true);
}

// arma::mat → std::vector<double> (column-major; Julia reshape is zero-copy).
static std::vector<double> pack(const arma::mat& m) {
    return std::vector<double>(m.memptr(), m.memptr() + m.n_elem);
}

// arma::rowvec → std::vector<double>
static std::vector<double> pack(const arma::rowvec& v) {
    return std::vector<double>(v.memptr(), v.memptr() + v.n_elem);
}

// Julia Matrix{Int32} (column-major) → arma::imat.
// Cannot be zero-copy: arma::sword is s32 or s64 depending on ARMA_64BIT_WORD,
// so we always copy with an explicit cast.
static arma::imat copy_imat(jlcxx::ArrayRef<int32_t> data, int rows, int cols) {
    arma::imat m(static_cast<arma::uword>(rows),
                 static_cast<arma::uword>(cols));
    // data is column-major (Julia-native), matching Armadillo's layout
    for (arma::uword j = 0; j < static_cast<arma::uword>(cols); ++j)
        for (arma::uword i = 0; i < static_cast<arma::uword>(rows); ++i)
            m(i, j) = static_cast<arma::sword>(data[j * rows + i]);
    return m;
}

// Unpack a jl_value_t* (expected to be Vector{Float64}) into arma::vec.
// Used to interpret the return value of a Julia decay function.
static arma::vec unpack_jl_vec(jl_value_t* val) {
    if (!jl_is_array(val))
        throw std::runtime_error("decay_fn must return a Vector{Float64}");
    auto* arr = reinterpret_cast<jl_array_t*>(val);
    return arma::vec(reinterpret_cast<double*>(jl_array_data(arr)),
                     static_cast<arma::uword>(jl_array_len(arr)),
                     /*copy=*/false);
}

// ── NodePositions result type ─────────────────────────────────────────────────
// Wrapping the four output matrices as a plain struct makes the multi-return
// function signatures manageable.  Julia unwraps it into a NamedTuple.

struct NodePositionsResult {
    std::vector<double> nodes;
    int32_t nodes_rows, nodes_cols;

    std::vector<double> centroids;
    int32_t centroids_rows, centroids_cols;

    std::vector<double> weights;
    int32_t weights_rows, weights_cols;

    std::vector<double> points;
    int32_t points_rows, points_cols;
};

static NodePositionsResult pack_positions(const qe::NodePositions& r) {
    NodePositionsResult out;
    out.nodes         = pack(r.nodes);
    out.nodes_rows    = static_cast<int32_t>(r.nodes.n_rows);
    out.nodes_cols    = static_cast<int32_t>(r.nodes.n_cols);
    out.centroids     = pack(r.centroids);
    out.centroids_rows = static_cast<int32_t>(r.centroids.n_rows);
    out.centroids_cols = static_cast<int32_t>(r.centroids.n_cols);
    out.weights       = pack(r.weights);
    out.weights_rows  = static_cast<int32_t>(r.weights.n_rows);
    out.weights_cols  = static_cast<int32_t>(r.weights.n_cols);
    out.points        = pack(r.points);
    out.points_rows   = static_cast<int32_t>(r.points.n_rows);
    out.points_cols   = static_cast<int32_t>(r.points.n_cols);
    return out;
}

// ── RotationResult ────────────────────────────────────────────────────────────
// rotation.hpp functions return qe::RotationResult.  We pack it into a plain
// struct for safe transport across the CxxWrap boundary.

struct RotationResultJ {
    std::vector<double> rot_matrix;    // rotation matrix, flat column-major
    int32_t rot_rows, rot_cols;

    std::vector<double> eigenvalues;   // flat (length = n_dims)

    std::vector<std::string> column_names;
};

static RotationResultJ pack_rotation(const qe::RotationResult& r) {
    RotationResultJ out;
    out.rot_matrix  = pack(r.rotation);
    out.rot_rows    = static_cast<int32_t>(r.rotation.n_rows);
    out.rot_cols    = static_cast<int32_t>(r.rotation.n_cols);
    out.eigenvalues = std::vector<double>(r.eigenvalues.memptr(),
                                          r.eigenvalues.memptr() + r.eigenvalues.n_elem);
    out.column_names = r.column_names;
    return out;
}

// ── TensorNetworks ────────────────────────────────────────────────────────────
// accumulation.hpp apply_tensor_unit returns qe::TensorNetworks.

struct TensorNetworksJ {
    std::vector<double> connection_counts;   // flat (length = n_codes² or choose_two)

    std::vector<double> row_networks;        // flat column-major
    int32_t row_networks_rows, row_networks_cols;
};

static TensorNetworksJ pack_tensor_networks(const qe::TensorNetworks& r) {
    TensorNetworksJ out;
    out.connection_counts = std::vector<double>(
        r.connection_counts.memptr(),
        r.connection_counts.memptr() + r.connection_counts.n_elem);
    out.row_networks      = pack(r.row_connection_counts);
    out.row_networks_rows = static_cast<int32_t>(r.row_connection_counts.n_rows);
    out.row_networks_cols = static_cast<int32_t>(r.row_connection_counts.n_cols);
    return out;
}

// ── Module definition ─────────────────────────────────────────────────────────

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {

    // ── NodePositionsResult (Julia sees it as a CxxWrap-wrapped struct) ───────
    mod.add_type<NodePositionsResult>("NodePositionsResult")
        .method("nodes",          [](const NodePositionsResult& r){ return r.nodes; })
        .method("nodes_rows",     [](const NodePositionsResult& r){ return r.nodes_rows; })
        .method("nodes_cols",     [](const NodePositionsResult& r){ return r.nodes_cols; })
        .method("centroids",      [](const NodePositionsResult& r){ return r.centroids; })
        .method("centroids_rows", [](const NodePositionsResult& r){ return r.centroids_rows; })
        .method("centroids_cols", [](const NodePositionsResult& r){ return r.centroids_cols; })
        .method("weights",        [](const NodePositionsResult& r){ return r.weights; })
        .method("weights_rows",   [](const NodePositionsResult& r){ return r.weights_rows; })
        .method("weights_cols",   [](const NodePositionsResult& r){ return r.weights_cols; })
        .method("points",         [](const NodePositionsResult& r){ return r.points; })
        .method("points_rows",    [](const NodePositionsResult& r){ return r.points_rows; })
        .method("points_cols",    [](const NodePositionsResult& r){ return r.points_cols; });

    // ── RotationResultJ ───────────────────────────────────────────────────────
    mod.add_type<RotationResultJ>("RotationResultJ")
        .method("rot_matrix",    [](const RotationResultJ& r){ return r.rot_matrix; })
        .method("rot_rows",      [](const RotationResultJ& r){ return r.rot_rows; })
        .method("rot_cols",      [](const RotationResultJ& r){ return r.rot_cols; })
        .method("eigenvalues",   [](const RotationResultJ& r){ return r.eigenvalues; })
        .method("column_names",  [](const RotationResultJ& r){ return r.column_names; });

    // ── TensorNetworksJ ───────────────────────────────────────────────────────
    mod.add_type<TensorNetworksJ>("TensorNetworksJ")
        .method("connection_counts",   [](const TensorNetworksJ& r){ return r.connection_counts; })
        .method("row_networks",        [](const TensorNetworksJ& r){ return r.row_networks; })
        .method("row_networks_rows",   [](const TensorNetworksJ& r){ return r.row_networks_rows; })
        .method("row_networks_cols",   [](const TensorNetworksJ& r){ return r.row_networks_cols; });

    // ── Adjacency ─────────────────────────────────────────────────────────────

    mod.method("choose_two", [](int32_t n) -> int32_t {
        return static_cast<int32_t>(qe::choose_two(n));
    });

    // tri_indices(len, row) → Matrix{Int32}  (2 × choose_two(len))
    mod.method("connection_indices", [](int32_t len, int32_t row) -> std::vector<int32_t> {
        arma::umat idx = qe::connection_indices(len, row);
        std::vector<int32_t> v(idx.n_elem);
        for (arma::uword i = 0; i < idx.n_elem; ++i)
            v[i] = static_cast<int32_t>(idx(i));
        return v;
    });

    // vector_to_upper_tri(v, n) → Vector{Float64}
    mod.method("code_connections", [](jlcxx::ArrayRef<double> v, int32_t n)
                                          -> std::vector<double> {
        arma::mat row(v.data(), 1, n, false, true);
        return pack(qe::code_connections(row));
    });

    // directed_to_upper_tri(v) → Vector{Float64}
    mod.method("fold_directed_network", [](jlcxx::ArrayRef<double> v)
                                            -> std::vector<double> {
        arma::vec av(v.data(), v.size(), false, true);
        return pack(qe::fold_directed_network(av));
    });

    // adjacency_matrix_to_vector(m, rows, cols, full) → Vector{Float64}
    mod.method("network_to_vector",
        [](jlcxx::ArrayRef<double> m, int32_t rows, int32_t cols, bool full)
         -> std::vector<double> {
            return pack(qe::network_to_vector(view_mat(m, rows, cols), full));
        });

    // svector_to_upper_tri(names) → Vector{String}
    mod.method("connection_names",
        [](const std::vector<std::string>& names) -> std::vector<std::string> {
            return qe::connection_names(names);
        });

    // ── Normalization ─────────────────────────────────────────────────────────

    mod.method("normalize_networks",
        [](jlcxx::ArrayRef<double> m, int32_t rows, int32_t cols)
         -> std::vector<double> {
            return pack(qe::normalize_networks(view_mat(m, rows, cols)));
        });

    mod.method("scale_networks",
        [](jlcxx::ArrayRef<double> m, int32_t rows, int32_t cols)
         -> std::vector<double> {
            return pack(qe::scale_networks(view_mat(m, rows, cols)));
        });

    // ── Modeling ──────────────────────────────────────────────────────────────

    mod.method("center_points",
        [](jlcxx::ArrayRef<double> m, int32_t rows, int32_t cols)
         -> std::vector<double> {
            return pack(qe::center_points(view_mat(m, rows, cols)));
        });

    // group_ci → n_dims × 3 [mean, lower, upper]; returned flat (col-major)
    mod.method("mean_ci",
        [](jlcxx::ArrayRef<double> m, int32_t rows, int32_t cols, double conf_level)
         -> std::vector<double> {
            return pack(qe::mean_ci(view_mat(m, rows, cols), conf_level));
        });

    // outlier_ci → n_dims × 2 [lower, upper]; returned flat (col-major)
    mod.method("outlier_ci",
        [](jlcxx::ArrayRef<double> m, int32_t rows, int32_t cols, double iqr_factor)
         -> std::vector<double> {
            return pack(qe::outlier_ci(view_mat(m, rows, cols), iqr_factor));
        });

    // ena_correlation → n_units × 3 [r, lower, upper]
    mod.method("ena_correlation",
        [](jlcxx::ArrayRef<double> pts, int32_t pr, int32_t pc,
           jlcxx::ArrayRef<double> cen, int32_t cr, int32_t cc,
           double conf_level) -> std::vector<double> {
            return pack(qe::ena_correlation(view_mat(pts, pr, pc),
                                            view_mat(cen, cr, cc), conf_level));
        });

    mod.method("node_positions",
        [](jlcxx::ArrayRef<double> adj, int32_t ar, int32_t ac,
           jlcxx::ArrayRef<double> t,   int32_t tr, int32_t tc,
           int32_t num_dims) -> NodePositionsResult {
            return pack_positions(qe::node_positions(
                view_mat(adj, ar, ac), view_mat(t, tr, tc), num_dims));
        });

    mod.method("directed_node_positions",
        [](jlcxx::ArrayRef<double> lw, int32_t lr, int32_t lc,
           jlcxx::ArrayRef<double> pt, int32_t pr, int32_t pc,
           int32_t num_dims) -> NodePositionsResult {
            return pack_positions(qe::directed_node_positions(
                view_mat(lw, lr, lc), view_mat(pt, pr, pc), num_dims));
        });

    mod.method("directed_node_positions_combine_pairs",
        [](jlcxx::ArrayRef<double> lw, int32_t lr, int32_t lc,
           jlcxx::ArrayRef<double> pt, int32_t pr, int32_t pc,
           int32_t num_dims) -> NodePositionsResult {
            return pack_positions(qe::directed_node_positions(
                view_mat(lw, lr, lc), view_mat(pt, pr, pc), num_dims, true));
        });

    // ── Accumulation ──────────────────────────────────────────────────────────

    mod.method("connection_matrix",
        [](jlcxx::ArrayRef<double> ground, int32_t gn,
           jlcxx::ArrayRef<double> resp,   int32_t rn,
           double response_weight, bool ordered) -> std::vector<double> {
            arma::rowvec g(ground.data(), gn, false, true);
            arma::rowvec r(resp.data(),   rn, false, true);
            return pack(qe::connection_matrix(g, r, response_weight, ordered));
        });

    mod.method("accumulate_stanza",
        [](jlcxx::ArrayRef<double> codes, int32_t rows, int32_t cols,
           int32_t window_back, int32_t window_forward, bool binary, bool ordered)
         -> std::vector<double> {
            return pack(qe::accumulate_stanza(view_mat(codes, rows, cols),
                                          window_back, window_forward, binary, ordered));
        });

    mod.method("row_connections",
        [](jlcxx::ArrayRef<double> codes, int32_t rows, int32_t cols, bool binary)
         -> std::vector<double> {
            return pack(qe::row_connections(view_mat(codes, rows, cols), binary));
        });

    mod.method("rolling_window_sum",
        [](jlcxx::ArrayRef<double> codes, int32_t rows, int32_t cols, int32_t window_size)
         -> std::vector<double> {
            return pack(qe::rolling_window_sum(view_mat(codes, rows, cols), window_size));
        });

    mod.method("flat_index",
        [](const std::vector<int32_t>& indices, const std::vector<int32_t>& dims) -> int32_t {
            std::vector<int> idx(indices.begin(), indices.end());
            std::vector<int> d(dims.begin(),    dims.end());
            return static_cast<int32_t>(qe::flat_index(idx, d));
        });

    // ── Accumulation — callback-based ─────────────────────────────────────────
    //
    // accumulate_unit and accumulate_unit_with_rows accept a Julia function as
    // the decay_fn argument.  The Julia function receives a Vector{Float64} of
    // distances and must return a Vector{Float64} of weights.
    //
    // Pattern: jlcxx::JuliaFunction wraps the Julia callable; calling it returns
    // jl_value_t* which is extracted via Julia's C API (jl_array_data /
    // jl_array_len) to construct an arma::vec view without copying.

    mod.method("accumulate_unit",
        [](jlcxx::ArrayRef<double> codes, int32_t rows, int32_t cols,
           const std::vector<int32_t>& unit_rows_i32,
           jlcxx::JuliaFunction decay_fn,
           bool ordered) -> std::vector<double> {

            std::vector<int> unit_rows(unit_rows_i32.begin(), unit_rows_i32.end());

            auto cpp_decay = [&decay_fn](arma::vec distances) -> arma::vec {
                std::vector<double> dv(distances.memptr(),
                                       distances.memptr() + distances.n_elem);
                jl_value_t* result = decay_fn(dv);
                return unpack_jl_vec(result);
            };

            return pack(qe::accumulate_unit(view_mat(codes, rows, cols),
                                            unit_rows, cpp_decay, ordered));
        });

    mod.method("accumulate_unit_with_rows",
        [](jlcxx::ArrayRef<double> codes, int32_t rows, int32_t cols,
           const std::vector<int32_t>& unit_rows_i32,
           jlcxx::JuliaFunction decay_fn,
           bool ordered) -> NodePositionsResult {

            std::vector<int> unit_rows(unit_rows_i32.begin(), unit_rows_i32.end());
            int n_unit = static_cast<int>(unit_rows.size());
            int n_codes = cols;

            auto cpp_decay = [&decay_fn](int unit_row, arma::uvec ground_indices) -> arma::vec {
                arma::vec distances(ground_indices.n_elem);
                for (arma::uword k = 0; k < ground_indices.n_elem; ++k)
                    distances[k] = static_cast<double>(unit_row - ground_indices[k]);
                std::vector<double> dv(distances.memptr(),
                                       distances.memptr() + distances.n_elem);
                jl_value_t* result = decay_fn(dv);
                return unpack_jl_vec(result);
            };

            qe::UnitNetworks r = qe::accumulate_unit_with_rows(
                view_mat(codes, rows, cols), unit_rows, cpp_decay, ordered);

            // Repack into NodePositionsResult (reusing the struct for its flat vectors)
            NodePositionsResult out;
            out.nodes      = pack(r.networks);
            out.nodes_rows = 1;
            out.nodes_cols = static_cast<int32_t>(r.networks.n_elem);
            out.centroids  = {};   // unused
            out.weights    = pack(r.row_networks);
            out.weights_rows = n_unit;
            out.weights_cols = n_codes * n_codes;
            out.points = {};       // unused
            return out;
        });

    // accumulate_tensor_unit — tma tensor-based accumulation for one unit.
    //
    // context_lookup is passed as a flat Int32 vector (column-major, Julia-native)
    // plus cl_rows/cl_cols.  All index vectors are Int32; copied to std::vector<int>
    // because qe::apply_tensor_unit takes std::vector<int>.
    mod.method("apply_tensor_unit",
        [](jlcxx::ArrayRef<double> tensor_ref,
           const std::vector<int32_t>& dims_i32,
           const std::vector<int32_t>& dims_sender_i32,
           const std::vector<int32_t>& dims_receiver_i32,
           const std::vector<int32_t>& dims_mode_i32,
           jlcxx::ArrayRef<int32_t> ctx_ref, int32_t cl_rows, int32_t cl_cols,
           const std::vector<int32_t>& unit_rows_i32,
           jlcxx::ArrayRef<double> codes_ref, int32_t rows, int32_t cols,
           jlcxx::ArrayRef<double> times_ref,
           bool ordered) -> TensorNetworksJ {

            arma::vec tensor(tensor_ref.data(), tensor_ref.size(), false, true);

            std::vector<int> dims(dims_i32.begin(), dims_i32.end());
            std::vector<int> dims_sender(dims_sender_i32.begin(), dims_sender_i32.end());
            std::vector<int> dims_receiver(dims_receiver_i32.begin(), dims_receiver_i32.end());
            std::vector<int> dims_mode(dims_mode_i32.begin(), dims_mode_i32.end());
            std::vector<int> unit_rows(unit_rows_i32.begin(), unit_rows_i32.end());

            arma::imat context_lookup = copy_imat(ctx_ref, cl_rows, cl_cols);
            arma::mat  codes          = view_mat(codes_ref, rows, cols);
            arma::vec  times(times_ref.data(), times_ref.size(), false, true);

            return pack_tensor_networks(qe::apply_tensor_unit(
                tensor, dims, dims_sender, dims_receiver, dims_mode,
                context_lookup, unit_rows, codes, times, ordered));
        });

    // ── Rotation ──────────────────────────────────────────────────────────────

    mod.method("ena_svd",
        [](jlcxx::ArrayRef<double> m, int32_t rows, int32_t cols) -> RotationResultJ {
            return pack_rotation(qe::ena_svd(view_mat(m, rows, cols)));
        });

    mod.method("deflate",
        [](jlcxx::ArrayRef<double> m, int32_t rows, int32_t cols,
           jlcxx::ArrayRef<double> axis_ref) -> std::vector<double> {
            arma::vec axis(axis_ref.data(), axis_ref.size(), false, true);
            return pack(qe::deflate(view_mat(m, rows, cols), axis));
        });

    mod.method("orthogonal_svd",
        [](jlcxx::ArrayRef<double> data,    int32_t dr, int32_t dc,
           jlcxx::ArrayRef<double> weights, int32_t wr, int32_t wc,
           const std::vector<std::string>& labels) -> RotationResultJ {
            return pack_rotation(qe::orthogonal_svd(
                view_mat(data,    dr, dc),
                view_mat(weights, wr, wc),
                labels));
        });

    mod.method("complete_rotation",
        [](jlcxx::ArrayRef<double> data,  int32_t dr, int32_t dc,
           jlcxx::ArrayRef<double> axes,  int32_t ar, int32_t ac,
           const std::vector<std::string>& labels) -> RotationResultJ {
            return pack_rotation(qe::complete_rotation(
                view_mat(data, dr, dc),
                view_mat(axes, ar, ac),
                labels));
        });

    // means_rotation — group pairs passed as four flat vectors:
    //   a_flat / a_sizes: concatenated a-group indices + size of each group
    //   b_flat / b_sizes: same for b-groups
    // Julia wrapper reconstructs Vector{Tuple{Vector{Int32},Vector{Int32}}} → these.
    mod.method("means_rotation",
        [](jlcxx::ArrayRef<double> m, int32_t rows, int32_t cols,
           const std::vector<int32_t>& a_flat, const std::vector<int32_t>& a_sizes,
           const std::vector<int32_t>& b_flat, const std::vector<int32_t>& b_sizes)
           -> RotationResultJ {

            std::vector<qe::GroupPair> pairs;
            pairs.reserve(a_sizes.size());
            size_t ai = 0, bi = 0;
            for (size_t k = 0; k < a_sizes.size(); ++k) {
                arma::uvec a(static_cast<arma::uword>(a_sizes[k]));
                arma::uvec b(static_cast<arma::uword>(b_sizes[k]));
                for (arma::uword j = 0; j < a.n_elem; ++j)
                    a(j) = static_cast<arma::uword>(a_flat[ai++]);
                for (arma::uword j = 0; j < b.n_elem; ++j)
                    b(j) = static_cast<arma::uword>(b_flat[bi++]);
                pairs.push_back({a, b});
            }
            return pack_rotation(qe::means_rotation(view_mat(m, rows, cols), pairs));
        });
}
