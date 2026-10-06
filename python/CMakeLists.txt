cmake_minimum_required(VERSION 3.18)
project(qe VERSION 0.1.0)

# ── Python ────────────────────────────────────────────────────────────────────
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# ── nanobind (locate via the Python package, works with pip or homebrew) ──────
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NB_CMAKE_DIR
    RESULT_VARIABLE NB_RESULT
)
if(NOT NB_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not locate nanobind cmake directory. "
            "Install it with: pip install nanobind")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${NB_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

# ── Armadillo ─────────────────────────────────────────────────────────────────
find_package(Armadillo REQUIRED)

# ── libqe headers — canonical location at repo root include/ ─────────────────
set(LIBQE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")
if(NOT EXISTS "${LIBQE_INCLUDE_DIR}/libqe/libqe.hpp")
    message(FATAL_ERROR "libqe headers not found at ${LIBQE_INCLUDE_DIR}. "
            "Expected ../include/libqe/libqe.hpp relative to libqe/python/.")
endif()

# ── binding module ────────────────────────────────────────────────────────────
nanobind_add_module(_qe NB_STATIC src/qe.cpp)

target_include_directories(_qe PRIVATE
    ${LIBQE_INCLUDE_DIR}
    ${ARMADILLO_INCLUDE_DIRS}
)

# Link armadillo + platform BLAS/LAPACK
if(APPLE)
    # macOS ships BLAS/LAPACK inside the Accelerate framework
    target_link_libraries(_qe PRIVATE ${ARMADILLO_LIBRARIES} "-framework Accelerate")
elseif(WIN32)
    # vcpkg's armadillo port (pulls in openblas + lapack)
    target_link_libraries(_qe PRIVATE ${ARMADILLO_LIBRARIES})
    find_package(LAPACK REQUIRED)
    find_package(BLAS  REQUIRED)
    target_link_libraries(_qe PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
else()
    # Linux: use Armadillo header-only and link exactly one BLAS/LAPACK.
    # The distro's libarmadillo wrapper, plus an unconstrained
    # find_package(BLAS/LAPACK), linked three OpenBLAS builds (serial,
    # pthreads, OpenMP), ATLAS, reference BLAS/LAPACK, SuperLU and ARPACK on
    # manylinux, all of which auditwheel bundled into every wheel (~43 MB).
    # Prefer OpenBLAS (manylinux); fall back to whatever single BLAS is
    # installed (e.g. reference BLAS on Debian's libarmadillo-dev, as in
    # libqe's own test:python CI job).
    target_compile_definitions(_qe PRIVATE
        ARMA_DONT_USE_WRAPPER ARMA_DONT_USE_SUPERLU ARMA_DONT_USE_ARPACK)
    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS QUIET)
    if(NOT BLAS_FOUND)
        message(STATUS "OpenBLAS not found; using the system BLAS/LAPACK")
        unset(BLA_VENDOR)
        find_package(BLAS REQUIRED)
    endif()
    find_package(LAPACK REQUIRED)
    message(STATUS "qe links BLAS: ${BLAS_LIBRARIES}  LAPACK: ${LAPACK_LIBRARIES}")
    target_link_libraries(_qe PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
endif()

install(TARGETS _qe LIBRARY DESTINATION qe)
