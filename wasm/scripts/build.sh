#!/bin/sh
# Build the libqe WASM module.
#
# Prerequisites:
#   - emscripten (emcc on PATH)
#   - conan 2.x
#
# Usage (from repo root or wasm/ directory):
#   sh wasm/scripts/build.sh   # from libqe/
#   sh scripts/build.sh        # from libqe/wasm/

set -e
# REPO = libqe/wasm/ (where CMakeLists.txt and conanfile.py live)
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${REPO}/build/wasm"

echo "==> conan install"
conan install "${REPO}" \
    -pr:b=default \
    -pr:h="${REPO}/profiles/wasm" \
    --build=missing \
    --output-folder="${BUILD}"

echo "==> cmake configure"
# Detect Ninja; fall back to Unix Makefiles if not on PATH.
# emcmake may alter PATH so we probe explicitly rather than relying on
# CMake's generator auto-detection.
if command -v ninja > /dev/null 2>&1; then
    GENERATOR_FLAGS="-G Ninja"
else
    GENERATOR_FLAGS=""
fi
emcmake cmake -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="${BUILD}/conan_toolchain.cmake" \
    ${GENERATOR_FLAGS} \
    "${REPO}"

echo "==> build"
cmake --build "${BUILD}"

echo "==> install to dist/"
cmake --install "${BUILD}"

echo "Done. Output in ${REPO}/dist/"
