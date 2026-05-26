cmake_minimum_required(VERSION 3.18)
project(libqe_wasm VERSION 0.1.0)

# ── libqe headers ─────────────────────────────────────────────────────────────
# When building from inside the repo the canonical headers are one level up.
# After 'conan install .' the headers arrive via CMakeDeps; we try the repo
# path first so developers don't need Conan installed just to compile.
set(LIBQE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")
if(NOT EXISTS "${LIBQE_INCLUDE_DIR}/libqe/libqe.hpp")
    find_package(libqe REQUIRED)
    set(LIBQE_INCLUDE_DIR "")
    message(STATUS "libqe: using Conan-provided package")
else()
    message(STATUS "libqe: using repo headers at ${LIBQE_INCLUDE_DIR}")
endif()

# ── Armadillo (provided by Conan) ─────────────────────────────────────────────
find_package(Armadillo REQUIRED)

# ── WASM module ───────────────────────────────────────────────────────────────
add_executable(libqe src/libqe_wasm.cpp)

target_include_directories(libqe PRIVATE
    ${LIBQE_INCLUDE_DIR}
    ${ARMADILLO_INCLUDE_DIRS}
)

target_link_libraries(libqe PRIVATE ${ARMADILLO_LIBRARIES})

# Disable BLAS/LAPACK — Armadillo falls back to its own built-in routines,
# which is the only path that works without external symbols in WASM.
target_compile_definitions(libqe PRIVATE
    ARMA_DONT_USE_BLAS
    ARMA_DONT_USE_LAPACK
)

# ── Emscripten link flags ─────────────────────────────────────────────────────
# MODULARIZE=1     — output a factory function rather than auto-executing
# EXPORT_NAME      — name of the JS factory function
# --bind           — enable Embind
# ALLOW_MEMORY_GROWTH — needed for dynamic Armadillo allocations
# ENVIRONMENT      — target web + node so the module works in both
set_target_properties(libqe PROPERTIES
    SUFFIX ".js"
    LINK_FLAGS [=[
        -s MODULARIZE=1
        -s EXPORT_NAME=LibQE
        -s WASM=1
        -s ALLOW_MEMORY_GROWTH=1
        -s ENVIRONMENT=web,node
        --bind
        -O3
    ]=]
)

# ── output to dist/ ───────────────────────────────────────────────────────────
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/libqe.js
    ${CMAKE_CURRENT_BINARY_DIR}/libqe.wasm
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/dist
)
