#!/bin/sh
# sync-headers.sh — copy canonical headers into R/inst/include/libqe/
#
# Run this manually (or from CI) whenever include/libqe/*.hpp changes.
# R's LinkingTo mechanism requires headers to live in inst/include/ of the
# installed package; this script keeps that copy up to date.
#
# Usage (from repo root):  ./scripts/sync-headers.sh
#           or in CI:      sh scripts/sync-headers.sh

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${REPO_ROOT}/include/libqe"
DST="${REPO_ROOT}/R/inst/include/libqe"

echo "Syncing ${SRC} -> ${DST}"
mkdir -p "${DST}"
cp "${SRC}"/*.hpp "${DST}/"
echo "Done."
