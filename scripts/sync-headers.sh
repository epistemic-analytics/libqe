#!/bin/sh
# sync-headers.sh — copy canonical headers into R/inst/include/libqe/ and
#                   python/include/libqe/
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
# Usage (from repo root):  sh scripts/sync-headers.sh

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${REPO_ROOT}/include/libqe"

for DST in "${REPO_ROOT}/R/inst/include/libqe" "${REPO_ROOT}/python/include/libqe"; do
    echo "Syncing ${SRC} -> ${DST}"
    mkdir -p "${DST}"
    cp "${SRC}"/*.hpp "${DST}/"
done
echo "Done."
