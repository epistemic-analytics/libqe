#!/bin/sh
# check-headers-in-sync.sh — CI gate: fail if include/ and R/inst/include/ differ.
#
# Usage (from repo root):  ./scripts/check-headers-in-sync.sh
# Exit code 0 = in sync, 1 = out of sync (CI should fail the build).

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${REPO_ROOT}/include/libqe"
DST="${REPO_ROOT}/R/inst/include/libqe"

DIFF=$(diff -rq --exclude='.gitkeep' "${SRC}" "${DST}" 2>&1)
if [ -n "${DIFF}" ]; then
    echo "ERROR: include/libqe/ and R/inst/include/libqe/ are out of sync:"
    echo "${DIFF}"
    echo ""
    echo "Run ./scripts/sync-headers.sh and commit the result."
    exit 1
fi

echo "OK: include/libqe/ and R/inst/include/libqe/ are in sync."
