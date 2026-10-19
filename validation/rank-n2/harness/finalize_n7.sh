#!/usr/bin/env bash
# Write the committed outputs of the n = 7 additions from the runs of
#   run_suite.sh n7-controls,  run_suite.sh n7  (S5b)  and  run_suite.sh identity-n7
# (GENERIC_RUNNER="ninjars case-check"):
#   suite:    the reference files of the n = 7 cases and their twins (reference/, controls/reference/),
#             the summaries summary/n7.json and summary/n7-controls.json and the exact scalar tables
#             scalars/n7.json and scalars/n7-controls.json;
#   identity: the reference files of the n = 7 identity cases and their pieces (identity/reference/),
#             summary/identity-n7.json, scalars/identity-n7.json and identity/residuals.json (every
#             identity of the manifest, with the COLLIER runs of both $WORK/identity and
#             $WORK/identity-n7).
# Unlike finalize.sh it rewrites no file of another set.  Original Ninja's results for the twins
# (controls/ninja/) come from ninja_controls.py.
#   harness/finalize_n7.sh [suite] [identity]      (default: both)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RN2="$(dirname "$HERE")"
REPO="$(cd "$RN2/../.." && pwd)"
WORK=${WORK:-$REPO/target/rank-n2-work}
PY="$HERE/python.sh"
PARTS=" ${*:-suite identity} "
if [[ "$PARTS" == *" suite "* ]]; then
  "$PY" "$HERE/assemble.py" --cases "$RN2"/controls/cases/*-n7-r*.json --work "$WORK/n7-controls" \
    --out "$RN2/controls/reference" --summary "$RN2/summary/n7-controls.json" --case-prefix controls/cases/ \
    --ninja "$RN2/controls/ninja"
  "$PY" "$HERE/assemble.py" --cases "$RN2"/cases/*-n7-r*.json --work "$WORK/n7" --out "$RN2/reference" \
    --summary "$RN2/summary/n7.json" --case-prefix cases/ --twins "$RN2/summary/n7-controls.json"
  for set in n7 n7-controls; do
    "$PY" "$HERE/export_scalars.py" "$WORK/$set/scalars-exact" "$RN2/scalars/$set.json"
  done
fi
if [[ "$PARTS" == *" identity "* ]]; then
  "$PY" "$HERE/assemble.py" --cases "$RN2"/identity/cases/*-n7*.json --work "$WORK/identity-n7" \
    --out "$RN2/identity/reference" --summary "$RN2/summary/identity-n7.json" --case-prefix identity/cases/
  "$PY" "$HERE/export_scalars.py" "$WORK/identity-n7/scalars-exact" "$RN2/scalars/identity-n7.json"
  # verify_identities.py keys its runs by directory name: merge both COLLIER runs into one.
  MERGED="$WORK/identity-merged/coli-quad"
  rm -rf "$MERGED"; mkdir -p "$MERGED"
  cp "$WORK"/identity/coli-quad/*.json "$WORK"/identity-n7/coli-quad/*.json "$MERGED"/
  "$PY" "$HERE/verify_identities.py" --manifest "$RN2/identity/manifest.json" --refs "$RN2/identity/reference" \
    --run "$MERGED" --json "$RN2/identity/residuals.json"
fi
