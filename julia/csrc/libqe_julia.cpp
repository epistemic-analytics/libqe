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
// The ENA model code (rotations, node positions) is in rENA's ENA.jl and all
// accumulation in tma's TMA.jl (libqe 0.2.0).

#include <jlcxx/jlcxx.hpp>
#include <jlcxx/array.hpp>
#include <jlcxx/stl.hpp>

#include <armadillo>
#include <libqe/libqe.hpp>
#include <libqe/bind/cxxwrap.hpp>

#include <vector>
#include <string>
#include <stdexcept>

// ── Helpers ───────────────────────────────────────────────────────────────────

// Array conversion (view_mat, pack, copy_imat, unpack_jl_vec) comes from
// libqe/bind/cxxwrap.hpp.
using namespace qe::bind::jl;

// ── GroupStats ────────────────────────────────────────────────────────────────
// stats.hpp group_stats returns qe::GroupStats.  All per-dimension vectors
// are flattened to std::vector<double>; matrices are flat column-major (Julia
// native) with accompanying row/col counts so Julia can reshape() them.

struct GroupStatsJ {
    int32_t n1, n2;

    // Parametric
    std::vector<double> t;
    std::vector<double> df;
    std::vector<double> pvalue_t;
    std::vector<double> cohens_d;
    std::vector<double> means;     // flat col-major, 2 × n_dims
    int32_t means_rows, means_cols;
    std::vector<double> sds;       // flat col-major, 2 × n_dims
    int32_t sds_rows, sds_cols;

    // Non-parametric
    std::vector<double> U;
    std::vector<double> pvalue_u;
    std::vector<double> effect_r;
    std::vector<double> medians;   // flat col-major, 2 × n_dims
    int32_t medians_rows, medians_cols;
};

static GroupStatsJ pack_group_stats(const qe::GroupStats& s) {
    GroupStatsJ out;
    out.n1 = static_cast<int32_t>(s.n1);
    out.n2 = static_cast<int32_t>(s.n2);

    auto pack_vec = [](const arma::vec& v) {
        return std::vector<double>(v.memptr(), v.memptr() + v.n_elem);
    };

    out.t        = pack_vec(s.t);
    out.df       = pack_vec(s.df);
    out.pvalue_t = pack_vec(s.pvalue_t);
    out.cohens_d = pack_vec(s.cohens_d);
    out.means       = pack(s.means);
    out.means_rows  = static_cast<int32_t>(s.means.n_rows);
    out.means_cols  = static_cast<int32_t>(s.means.n_cols);
    out.sds         = pack(s.sds);
    out.sds_rows    = static_cast<int32_t>(s.sds.n_rows);
    out.sds_cols    = static_cast<int32_t>(s.sds.n_cols);
    out.U        = pack_vec(s.U);
    out.pvalue_u = pack_vec(s.pvalue_u);
    out.effect_r = pack_vec(s.effect_r);
    out.medians       = pack(s.medians);
    out.medians_rows  = static_cast<int32_t>(s.medians.n_rows);
    out.medians_cols  = static_cast<int32_t>(s.medians.n_cols);
    return out;
}

// ── Module definition ─────────────────────────────────────────────────────────

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {

    // ── GroupStatsJ ───────────────────────────────────────────────────────────
    mod.add_type<GroupStatsJ>("GroupStatsJ")
        .method("n1",           [](const GroupStatsJ& r){ return r.n1; })
        .method("n2",           [](const GroupStatsJ& r){ return r.n2; })
        .method("t_stat",       [](const GroupStatsJ& r){ return r.t; })
        .method("df",           [](const GroupStatsJ& r){ return r.df; })
        .method("pvalue_t",     [](const GroupStatsJ& r){ return r.pvalue_t; })
        .method("cohens_d",     [](const GroupStatsJ& r){ return r.cohens_d; })
        .method("means",        [](const GroupStatsJ& r){ return r.means; })
        .method("means_rows",   [](const GroupStatsJ& r){ return r.means_rows; })
        .method("means_cols",   [](const GroupStatsJ& r){ return r.means_cols; })
        .method("sds",          [](const GroupStatsJ& r){ return r.sds; })
        .method("sds_rows",     [](const GroupStatsJ& r){ return r.sds_rows; })
        .method("sds_cols",     [](const GroupStatsJ& r){ return r.sds_cols; })
        .method("U",            [](const GroupStatsJ& r){ return r.U; })
        .method("pvalue_u",     [](const GroupStatsJ& r){ return r.pvalue_u; })
        .method("effect_r",     [](const GroupStatsJ& r){ return r.effect_r; })
        .method("medians",      [](const GroupStatsJ& r){ return r.medians; })
        .method("medians_rows", [](const GroupStatsJ& r){ return r.medians_rows; })
        .method("medians_cols", [](const GroupStatsJ& r){ return r.medians_cols; });

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

    // group_stats(g1, g1_rows, g1_cols, g2, g2_rows, g2_cols) → GroupStatsJ
    // Julia-side: all matrices are column-major Float64 arrays (Julia-native layout,
    // zero-copy view). Returns a GroupStatsJ struct; call reshape() on the flat
    // matrix vectors using the accompanying _rows/_cols accessors.
    mod.method("group_stats",
        [](jlcxx::ArrayRef<double> g1, int32_t g1r, int32_t g1c,
           jlcxx::ArrayRef<double> g2, int32_t g2r, int32_t g2c) -> GroupStatsJ {
            return pack_group_stats(qe::group_stats(view_mat(g1, g1r, g1c),
                                                    view_mat(g2, g2r, g2c)));
        });
}
