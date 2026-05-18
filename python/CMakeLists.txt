cmake_minimum_required(VERSION 3.18)
project(pylibqe VERSION 0.1.0)

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
nanobind_add_module(_pylibqe NB_STATIC src/pylibqe.cpp)

target_include_directories(_pylibqe PRIVATE
    ${LIBQE_INCLUDE_DIR}
    ${ARMADILLO_INCLUDE_DIRS}
)

# Link armadillo + platform BLAS/LAPACK
target_link_libraries(_pylibqe PRIVATE ${ARMADILLO_LIBRARIES})

if(APPLE)
    # macOS ships BLAS/LAPACK inside the Accelerate framework
    target_link_libraries(_pylibqe PRIVATE "-framework Accelerate")
else()
    find_package(LAPACK REQUIRED)
    find_package(BLAS  REQUIRED)
    target_link_libraries(_pylibqe PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
endif()

install(TARGETS _pylibqe LIBRARY DESTINATION pylibqe)
