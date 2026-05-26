#!/bin/sh
# Build the libqe WASM module.
#
# Prerequisites:
#   - emscripten (emcc on PATH)
#   - conan 2.x
#   - ninja
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
emcmake cmake -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="${BUILD}/conan_toolchain.cmake" \
    -G Ninja \
    "${REPO}"

echo "==> build"
cmake --build "${BUILD}"

echo "==> install to dist/"
cmake --install "${BUILD}"

echo "Done. Output in ${REPO}/dist/"
