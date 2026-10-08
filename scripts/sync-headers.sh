#!/bin/sh
# sync-headers.sh — copy canonical headers into R/inst/include/<lib>/ and
#                   python/include/<lib>/ for <lib> in libqe, libena, libtma
#
# R's LinkingTo mechanism requires headers in inst/include/ of the installed
# package.  The configure script handles this automatically when installing
# directly from the repo (R CMD INSTALL R/).  This script is needed for:
#
#   1. Building a distributable source tarball:
#        sh scripts/sync-headers.sh && R CMD build R/
#
#   2. Local devtools installs (devtools copies to a temp dir, so the
#      configure script cannot find ../include/):
#        sh scripts/sync-headers.sh && Rscript -e "devtools::install('R')"
#
#   3. Building a self-contained Python sdist (the sdist only contains python/,
#      so it cannot reach ../include/):
#        sh scripts/sync-headers.sh && python -m build --sdist python/
#      cranqe runs this script before building, when the repo provides it.
#
# libena and libtma are synced while libqe.hpp still includes them
# (transitional, until libqe 0.2.0).  R gets only the top-level *.hpp files;
# Python also gets the bind/ helpers its extension compiles against.
#
# Usage (from repo root):  sh scripts/sync-headers.sh

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Both destinations are generated (git-ignored) copies: clear them before
# copying so a header that moved or was removed upstream is not left behind.
for LIB in libqe libena libtma; do
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
