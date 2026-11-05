/**
 * @file validate.hpp
 * @brief Input checks shared by the language bindings.
 *
 * No third-party dependencies beyond Armadillo, so every binding (and any
 * other C++ consumer) can use the same checks and error messages.
 */
#ifndef LIBQE_VALIDATE_HPP
#define LIBQE_VALIDATE_HPP

#include <armadillo>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace qe {

/**
 * @brief Throw if @p m contains NaN or Inf.
 *
 * @param[in] m      Matrix to check.
 * @param[in] param  Name of the caller's argument, used in the message.
 * @throws std::invalid_argument naming @p param.
 */
inline void require_finite(const arma::mat& m, const char* param) {
    if (!m.is_finite())
        throw std::invalid_argument(
            std::string(param) + " contains NaN or Inf — "
            "filter or impute rows with non-finite values before calling");
}

/**
 * @brief Throw unless @p rows × @p cols describes exactly @p n elements.
 *
 * Bindings that receive a flat buffer plus explicit dimensions must call this
 * before building an arma::mat over the buffer: arma::mat(ptr, rows, cols)
 * reads rows*cols elements, so a short buffer would otherwise be padded with
 * whatever memory follows it.
 *
 * @throws std::invalid_argument on a negative dimension or a size mismatch.
 */
inline void require_dims(std::size_t n, int rows, int cols) {
    if (rows < 0 || cols < 0 ||
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) != n)
        throw std::invalid_argument("matrix dimensions do not match the data length");
}

} // namespace qe

#endif // LIBQE_VALIDATE_HPP
