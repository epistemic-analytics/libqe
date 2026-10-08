/**
 * @file bind/nanobind.hpp
 * @brief Armadillo ↔ numpy conversion helpers for nanobind (Python) bindings.
 *
 * Armadillo stores matrices in column-major order; numpy defaults to
 * row-major.  Every helper copies on the boundary so callers never see stale
 * memory, and every returned array is freshly allocated and owned by Python.
 *
 * Requires nanobind; never included by libqe.hpp.  Bindings typically
 * `using namespace qe::bind::py;` after including it.
 */
#ifndef LIBQE_BIND_NANOBIND_HPP
#define LIBQE_BIND_NANOBIND_HPP

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <armadillo>
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace qe {
namespace bind {
namespace py {

using NpMat  = ::nanobind::ndarray<double, ::nanobind::ndim<2>, ::nanobind::c_contig, ::nanobind::device::cpu>;
using NpVec  = ::nanobind::ndarray<double, ::nanobind::ndim<1>, ::nanobind::c_contig, ::nanobind::device::cpu>;

// int32 matrix type for context_lookup (arma::imat)
using NpIMat = ::nanobind::ndarray<int32_t, ::nanobind::ndim<2>, ::nanobind::c_contig, ::nanobind::device::cpu>;

// numpy 2-D (rows × cols, C-order) → arma::mat (col-major)
inline arma::mat to_mat(NpMat arr) {
    arma::mat m(arr.shape(0), arr.shape(1));
    for (size_t i = 0; i < arr.shape(0); ++i)
        for (size_t j = 0; j < arr.shape(1); ++j)
            m(i, j) = arr(i, j);
    return m;
}

// numpy 1-D → arma::rowvec (copy)
inline arma::rowvec to_rowvec(NpVec arr) {
    return arma::rowvec(const_cast<double*>(arr.data()), arr.shape(0), /*copy=*/true);
}

// numpy 1-D → arma::vec (copy)
inline arma::vec to_vec(NpVec arr) {
    return arma::vec(const_cast<double*>(arr.data()), arr.shape(0), /*copy=*/true);
}

// numpy 2-D int32 (rows × cols, C-order) → arma::imat
inline arma::imat to_imat(NpIMat arr) {
    arma::imat m(arr.shape(0), arr.shape(1));
    for (size_t i = 0; i < arr.shape(0); ++i)
        for (size_t j = 0; j < arr.shape(1); ++j)
            m(i, j) = arr(i, j);
    return m;
}

// arma::mat → numpy 2-D (rows × cols, C-order, Python owns the copy)
inline ::nanobind::ndarray<::nanobind::numpy, double, ::nanobind::ndim<2>> from_mat(const arma::mat& m) {
    size_t shape[2] = {m.n_rows, m.n_cols};
    double* data = new double[m.n_rows * m.n_cols];
    for (size_t i = 0; i < m.n_rows; ++i)
        for (size_t j = 0; j < m.n_cols; ++j)
            data[i * m.n_cols + j] = m(i, j);
    ::nanobind::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return ::nanobind::ndarray<::nanobind::numpy, double, ::nanobind::ndim<2>>(data, 2, shape, owner);
}

// arma::rowvec → numpy 1-D (Python owns the copy)
inline ::nanobind::ndarray<::nanobind::numpy, double, ::nanobind::ndim<1>> from_rowvec(const arma::rowvec& v) {
    size_t shape[1] = {v.n_elem};
    double* data = new double[v.n_elem];
    std::copy(v.begin(), v.end(), data);
    ::nanobind::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return ::nanobind::ndarray<::nanobind::numpy, double, ::nanobind::ndim<1>>(data, 1, shape, owner);
}

// arma::vec (column vector) → numpy 1-D
inline ::nanobind::ndarray<::nanobind::numpy, double, ::nanobind::ndim<1>> from_vec(const arma::vec& v) {
    size_t shape[1] = {v.n_elem};
    double* data = new double[v.n_elem];
    std::copy(v.begin(), v.end(), data);
    ::nanobind::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return ::nanobind::ndarray<::nanobind::numpy, double, ::nanobind::ndim<1>>(data, 1, shape, owner);
}

// arma::umat → numpy 2-D int64 (Python owns the copy)
inline ::nanobind::ndarray<::nanobind::numpy, int64_t, ::nanobind::ndim<2>> from_umat(const arma::umat& m) {
    size_t shape[2] = {m.n_rows, m.n_cols};
    int64_t* data = new int64_t[m.n_rows * m.n_cols];
    for (size_t i = 0; i < m.n_rows; ++i)
        for (size_t j = 0; j < m.n_cols; ++j)
            data[i * m.n_cols + j] = static_cast<int64_t>(m(i, j));
    ::nanobind::capsule owner(data, [](void* p) noexcept { delete[] static_cast<int64_t*>(p); });
    return ::nanobind::ndarray<::nanobind::numpy, int64_t, ::nanobind::ndim<2>>(data, 2, shape, owner);
}

// arma::uvec → numpy 1-D int64
inline ::nanobind::ndarray<::nanobind::numpy, int64_t, ::nanobind::ndim<1>> from_uvec(const arma::uvec& v) {
    size_t shape[1] = {v.n_elem};
    int64_t* data = new int64_t[v.n_elem];
    for (size_t i = 0; i < v.n_elem; ++i) data[i] = static_cast<int64_t>(v[i]);
    ::nanobind::capsule owner(data, [](void* p) noexcept { delete[] static_cast<int64_t*>(p); });
    return ::nanobind::ndarray<::nanobind::numpy, int64_t, ::nanobind::ndim<1>>(data, 1, shape, owner);
}

} // namespace py
} // namespace bind
} // namespace qe

#endif // LIBQE_BIND_NANOBIND_HPP
