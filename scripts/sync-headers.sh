#!/bin/sh
# sync-headers.sh — copy canonical headers into R/inst/include/libqe/
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
# Usage (from repo root):  sh scripts/sync-headers.sh

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${REPO_ROOT}/include/libqe"
DST="${REPO_ROOT}/R/inst/include/libqe"

echo "Syncing ${SRC} -> ${DST}"
mkdir -p "${DST}"
cp "${SRC}"/*.hpp "${DST}/"
echo "Done."
