/**
 * @file bind/emscripten.hpp
 * @brief Armadillo ↔ JavaScript conversion helpers for Emscripten (WASM)
 *        bindings.
 *
 * Matrix convention: JavaScript typed arrays are row-major, Armadillo is
 * column-major.  Matrices cross the boundary as a flat row-major array plus
 * explicit rows/cols, and are returned as plain JS objects
 * `{ data: Float64Array, rows: number, cols: number }`.
 *
 * Requires Emscripten; never included by libqe.hpp.  Include it after any
 * ARMA_* configuration macros (e.g. ARMA_DONT_USE_BLAS), since it includes
 * <armadillo>.  Bindings typically `using namespace qe::bind::js;`.
 */
#ifndef LIBQE_BIND_EMSCRIPTEN_HPP
#define LIBQE_BIND_EMSCRIPTEN_HPP

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <armadillo>
#include <vector>

#include "../validate.hpp"

namespace qe {
namespace bind {
namespace js {

// JS Float64Array (row-major) → arma::mat (column-major)
inline arma::mat js_to_mat(const emscripten::val& data, int rows, int cols) {
    std::vector<double> v = emscripten::vecFromJSArray<double>(data);
    qe::require_dims(v.size(), rows, cols);
    // arma::mat(ptr, rows, cols) reads column-major; transpose to convert
    // from the row-major JS layout.
    arma::mat m(v.data(), cols, rows);   // read as (cols × rows) col-major
    return m.t();                         // transpose → (rows × cols)
}

// JS Int32Array (row-major) → arma::imat (column-major integers)
inline arma::imat js_to_imat(const emscripten::val& data, int rows, int cols) {
    std::vector<int> v = emscripten::vecFromJSArray<int>(data);
    qe::require_dims(v.size(), rows, cols);
    arma::imat m(v.data(), cols, rows);  // read as (cols × rows) col-major
    return m.t();                         // transpose → (rows × cols)
}

// arma::mat (column-major) → JS { data: Float64Array, rows, cols }
inline emscripten::val mat_to_js(const arma::mat& m) {
    // Transpose to row-major for JS consumers.
    arma::mat row_major = m.t();
    std::vector<double> v(row_major.memptr(),
                          row_major.memptr() + row_major.n_elem);
    emscripten::val result = emscripten::val::object();
    result.set("data", emscripten::val::array(v.begin(), v.end()));
    result.set("rows", static_cast<int>(m.n_rows));
    result.set("cols", static_cast<int>(m.n_cols));
    return result;
}

// arma::rowvec → JS Float64Array
inline emscripten::val rowvec_to_js(const arma::rowvec& v) {
    std::vector<double> vec(v.memptr(), v.memptr() + v.n_elem);
    return emscripten::val::array(vec.begin(), vec.end());
}

} // namespace js
} // namespace bind
} // namespace qe

#endif // LIBQE_BIND_EMSCRIPTEN_HPP
