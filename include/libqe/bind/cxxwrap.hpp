/**
 * @file bind/cxxwrap.hpp
 * @brief Armadillo ↔ Julia conversion helpers for CxxWrap.jl bindings.
 *
 * Julia matrices are column-major, the same as Armadillo, so input matrices
 * can be zero-copy views.  Results are copied out to std::vector<double>
 * (column-major), which the Julia side reshape()s without another copy.
 *
 * Requires CxxWrap (jlcxx); never included by libqe.hpp.  Bindings typically
 * `using namespace qe::bind::jl;`.
 */
#ifndef LIBQE_BIND_CXXWRAP_HPP
#define LIBQE_BIND_CXXWRAP_HPP

#include <jlcxx/jlcxx.hpp>
#include <jlcxx/array.hpp>

#include <armadillo>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace qe {
namespace bind {
namespace jl {

// Julia array (column-major) → zero-copy arma::mat view.
// CAUTION: the arma::mat must not outlive the Julia array.
inline arma::mat view_mat(jlcxx::ArrayRef<double> data, int rows, int cols) {
    return arma::mat(data.data(), static_cast<arma::uword>(rows),
                                  static_cast<arma::uword>(cols),
                     /*copy_aux_mem=*/false, /*strict=*/true);
}

// arma::mat → std::vector<double> (column-major; Julia reshape is zero-copy).
inline std::vector<double> pack(const arma::mat& m) {
    return std::vector<double>(m.memptr(), m.memptr() + m.n_elem);
}

// arma::rowvec → std::vector<double>
inline std::vector<double> pack(const arma::rowvec& v) {
    return std::vector<double>(v.memptr(), v.memptr() + v.n_elem);
}

// Julia Matrix{Int32} (column-major) → arma::imat.
// Cannot be zero-copy: arma::sword is s32 or s64 depending on ARMA_64BIT_WORD,
// so we always copy with an explicit cast.
inline arma::imat copy_imat(jlcxx::ArrayRef<int32_t> data, int rows, int cols) {
    arma::imat m(static_cast<arma::uword>(rows),
                 static_cast<arma::uword>(cols));
    // data is column-major (Julia-native), matching Armadillo's layout
    for (arma::uword j = 0; j < static_cast<arma::uword>(cols); ++j)
        for (arma::uword i = 0; i < static_cast<arma::uword>(rows); ++i)
            m(i, j) = static_cast<arma::sword>(data[j * rows + i]);
    return m;
}

// Unpack a jl_value_t* (expected to be Vector{Float64}) into arma::vec.
// Used to interpret the return value of a Julia callback (e.g. a decay
// function).
inline arma::vec unpack_jl_vec(jl_value_t* val) {
    if (!jl_is_array(val))
        throw std::runtime_error("decay_fn must return a Vector{Float64}");
    auto* arr = reinterpret_cast<jl_array_t*>(val);
    return arma::vec(jl_array_data(arr, double),
                     static_cast<arma::uword>(jl_array_len(arr)),
                     /*copy=*/false);
}

} // namespace jl
} // namespace bind
} // namespace qe

#endif // LIBQE_BIND_CXXWRAP_HPP
