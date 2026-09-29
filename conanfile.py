from conan import ConanFile
from conan.tools.files import copy
from conan.tools.layout import basic_layout
import os


class LibqeConan(ConanFile):
    name         = "libqe"
    version      = "0.0.0"   # placeholder — set_version() always overrides this
    description  = "Header-only C++ core for Quantitative Ethnography packages (rENA, tma)"
    license      = "GPL-3.0-only"
    url          = "https://github.com/epistemic-analytics/libqe"
    homepage     = url
    topics       = ("header-only", "quantitative-ethnography", "ena", "armadillo")
    package_type = "header-library"

    # Export the canonical headers into the Conan recipe cache so they are
    # available to package() regardless of where conan create is invoked from.
    exports_sources = "include/libqe/*.hpp"

    # No source files need copying to the build folder — headers are read
    # directly from the exported source tree.
    no_copy_source = True

    # ── version: single source of truth is R/DESCRIPTION ─────────────────────
    # On a release tag (CI_COMMIT_TAG=v0.2.0) the tag wins.
    # Everywhere else (local dev, branch CI) the version is read from
    # R/DESCRIPTION so there is exactly one place to bump it.
    def set_version(self):
        tag = os.environ.get("CI_COMMIT_TAG", "")
        if tag.startswith("v") and tag[1:]:
            self.version = tag[1:]          # "v0.2.1" → "0.2.1"
            return
        desc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "R", "DESCRIPTION")
        with open(desc) as f:
            for line in f:
                if line.startswith("Version:"):
                    self.version = line.split(":", 1)[1].strip()
                    return

    # ── dependencies ─────────────────────────────────────────────────────────
    def requirements(self):
        # Armadillo headers are a hard dependency; consumers inherit this.
        # Pin to a specific patch to guarantee reproducible builds.
        # Update here when upgrading across the qe-packages ecosystem.
        self.requires("armadillo/12.6.4")

    # ── layout ───────────────────────────────────────────────────────────────
    def layout(self):
        basic_layout(self, src_folder=".")

    # ── package_id ───────────────────────────────────────────────────────────
    def package_id(self):
        # Header-only: the installed artifact is identical regardless of
        # consumer OS / arch / compiler.  Clearing info ensures a single
        # cached copy is shared by all configurations.
        self.info.clear()

    # ── packaging ────────────────────────────────────────────────────────────
    def package(self):
        copy(self, "*.hpp",
             src=os.path.join(self.source_folder, "include"),
             dst=os.path.join(self.package_folder, "include"))

    # ── consumer interface ───────────────────────────────────────────────────
    def package_info(self):
        # No compiled library — only include paths.
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []

        # CMake integration: find_package(libqe) / target_link_libraries(... libqe::libqe)
        self.cpp_info.set_property("cmake_file_name",   "libqe")
        self.cpp_info.set_property("cmake_target_name", "libqe::libqe")
