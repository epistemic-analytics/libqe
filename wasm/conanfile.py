from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMakeDeps
import os


class LibqeWasmConan(ConanFile):
    name     = "libqe-wasm"
    settings = "os", "arch", "compiler", "build_type"

    def requirements(self):
        # Armadillo headers — libqe headers come from ../include/ directly.
        self.requires("armadillo/15.2.6")

    def generate(self):
        # Propagate ARMA_DONT_USE_BLAS/LAPACK to the CMake build so that
        # Armadillo never emits BLAS/LAPACK symbol references that Emscripten
        # cannot resolve.
        tc = CMakeToolchain(self)
        tc.preprocessor_definitions["ARMA_DONT_USE_BLAS"]   = "1"
        tc.preprocessor_definitions["ARMA_DONT_USE_LAPACK"] = "1"
        tc.generate()
        CMakeDeps(self).generate()
