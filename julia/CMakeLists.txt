cmake_minimum_required(VERSION 3.18)
project(libqe_julia VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Julia + CxxWrap ───────────────────────────────────────────────────────────
# Bootstrap order: find_package(Julia) needs FindJulia.cmake which lives inside
# the JlCxx artifact.  So we must locate the CxxWrap prefix *before* calling
# find_package — use find_program to get julia, then ask it for the prefix.

find_program(Julia_EXECUTABLE julia REQUIRED)

execute_process(
    COMMAND "${Julia_EXECUTABLE}" --project=@. --startup-file=no -e
            "using CxxWrap; print(CxxWrap.prefix_path())"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE JLCXX_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE JLCXX_RESULT
)
if(NOT JLCXX_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Could not locate CxxWrap prefix. "
        "Install it in Julia with: using Pkg; Pkg.add(\"CxxWrap\")")
endif()

# Prepend both paths so find_package(Julia) can find FindJulia.cmake
# and find_package(JlCxx) can find JlCxxConfig.cmake.
list(PREPEND CMAKE_MODULE_PATH "${JLCXX_PREFIX}/lib/cmake/JlCxx")
list(PREPEND CMAKE_PREFIX_PATH "${JLCXX_PREFIX}")

find_package(Julia REQUIRED)
find_package(JlCxx REQUIRED)

# ── Armadillo ─────────────────────────────────────────────────────────────────
find_package(Armadillo REQUIRED)

# ── libqe headers ─────────────────────────────────────────────────────────────
# When building from inside the repo the canonical headers are one level up.
# After 'conan install .' the headers arrive via CMakeDeps; we try the repo
# path first so developers don't need Conan installed just to compile.
set(LIBQE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")
if(NOT EXISTS "${LIBQE_INCLUDE_DIR}/libqe/libqe.hpp")
    # Fall back to a Conan-provided libqe target if the repo headers aren't
    # present (e.g. CI builds outside the monorepo).
    find_package(libqe REQUIRED)
    set(LIBQE_INCLUDE_DIR "")     # libqe::libqe target carries the include path
    message(STATUS "libqe: using Conan-provided package")
else()
    message(STATUS "libqe: using repo headers at ${LIBQE_INCLUDE_DIR}")
endif()

# ── shared library ────────────────────────────────────────────────────────────
add_library(libqe_julia SHARED csrc/libqe_julia.cpp)

target_include_directories(libqe_julia PRIVATE
    ${LIBQE_INCLUDE_DIR}
    ${ARMADILLO_INCLUDE_DIRS}
)

# cxxwrap_julia_stl provides the std::vector wrappers used via jlcxx/stl.hpp.
# Linux tolerates its symbols being unresolved at link time (CxxWrap loads the
# library first), but macOS's linker rejects undefined symbols.
target_link_libraries(libqe_julia PRIVATE
    JlCxx::cxxwrap_julia
    JlCxx::cxxwrap_julia_stl
    ${ARMADILLO_LIBRARIES}
)

# On macOS, Accelerate provides BLAS/LAPACK; on Linux use the system LAPACK.
if(APPLE)
    target_link_libraries(libqe_julia PRIVATE "-framework Accelerate")
else()
    find_package(LAPACK REQUIRED)
    find_package(BLAS   REQUIRED)
    target_link_libraries(libqe_julia PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
endif()

# CxxWrap requires the library name to match what @wrapmodule expects.
set_target_properties(libqe_julia PROPERTIES
    PREFIX ""
    OUTPUT_NAME "libqe_julia"
)

install(TARGETS libqe_julia
    LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/lib"
)
