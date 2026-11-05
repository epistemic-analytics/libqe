#!/bin/sh
# sync-headers.sh — copy canonical headers into R/inst/include/<lib>/ and
#                   python/include/libqe/
#
# R's LinkingTo mechanism requires headers in inst/include/ of the installed
# package.  R/inst/include/libqe/ is a committed copy of include/libqe/ (the
# R package has no configure script, so it builds the same way from a git
# checkout, a devtools install and a CRAN source tarball).  After editing a
# header under include/, run this script and commit the updated copy;
# scripts/check-headers-in-sync.sh fails CI when the two trees differ.
#
# python/include/ is a git-ignored copy made for a self-contained Python sdist
# (the sdist only contains python/, so it cannot reach ../include/):
#     sh scripts/sync-headers.sh && python -m build --sdist python/
# cranqe runs this script before building, when the repo provides it.
#
# R gets only the top-level *.hpp files: the bind/ helpers need nanobind,
# Emscripten or CxxWrap and never ship to R.  Python also gets bind/.
# (libena and libtma moved to rENA and tma in libqe 0.2.0; copies an older
# checkout left behind are removed.)
#
# Usage (from repo root):  sh scripts/sync-headers.sh

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Clear both destinations before copying so a header that moved or was
# removed upstream is not left behind.
for LIB in libena libtma; do
    rm -rf "${REPO_ROOT}/R/inst/include/${LIB}" "${REPO_ROOT}/python/include/${LIB}"
done

for LIB in libqe; do
    SRC="${REPO_ROOT}/include/${LIB}"

    DST="${REPO_ROOT}/R/inst/include/${LIB}"
    echo "Syncing ${SRC} -> ${DST}"
    mkdir -p "${DST}"
    rm -f "${DST}"/*.hpp
    cp "${SRC}"/*.hpp "${DST}/"

    DST="${REPO_ROOT}/python/include/${LIB}"
    echo "Syncing ${SRC} -> ${DST}"
    rm -rf "${DST}"
    mkdir -p "${DST}"
    cp -R "${SRC}"/. "${DST}/"
done
echo "Done."
