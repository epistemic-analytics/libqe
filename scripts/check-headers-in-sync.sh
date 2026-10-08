#!/bin/sh
# check-headers-in-sync.sh — CI gate: fail if include/<lib>/ and
#                            R/inst/include/<lib>/ differ, for <lib> in
#                            libqe, libena, libtma.
#
# The bind/ helpers are never copied to R (see R/configure), so they are
# excluded from the comparison.
#
# Usage (from repo root):  ./scripts/check-headers-in-sync.sh
# Exit code 0 = in sync, 1 = out of sync (CI should fail the build).

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATUS=0

for LIB in libqe libena libtma; do
    SRC="${REPO_ROOT}/include/${LIB}"
    DST="${REPO_ROOT}/R/inst/include/${LIB}"

    DIFF=$(diff -rq --exclude='.gitkeep' --exclude='bind' "${SRC}" "${DST}" 2>&1) || true
    if [ -n "${DIFF}" ]; then
        echo "ERROR: include/${LIB}/ and R/inst/include/${LIB}/ are out of sync:"
        echo "${DIFF}"
        echo ""
        STATUS=1
    else
        echo "OK: include/${LIB}/ and R/inst/include/${LIB}/ are in sync."
    fi
done

if [ "${STATUS}" -ne 0 ]; then
    echo "Run ./scripts/sync-headers.sh and commit the result."
fi
exit "${STATUS}"
