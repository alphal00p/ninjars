#!/usr/bin/env bash
# Write the committed outputs of the n = 7 additions (S5b) from the runs of
#   run_suite.sh n7-controls  and  run_suite.sh n7
# (GENERIC_RUNNER="ninjars case-check"): the reference files of the n = 7 cases and their twins
# (reference/, controls/reference/), the summaries summary/n7.json and summary/n7-controls.json and
# the exact scalar tables scalars/n7.json and scalars/n7-controls.json.  Unlike finalize.sh it
# rewrites no file of another set.  Original Ninja's results for the twins (controls/ninja/) come
# from ninja_controls.py.
#   harness/finalize_n7.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RN2="$(dirname "$HERE")"
REPO="$(cd "$RN2/../.." && pwd)"
WORK=${WORK:-$REPO/target/rank-n2-work}
PY="$HERE/python.sh"
"$PY" "$HERE/assemble.py" --cases "$RN2"/controls/cases/*-n7-r*.json --work "$WORK/n7-controls" \
  --out "$RN2/controls/reference" --summary "$RN2/summary/n7-controls.json" --case-prefix controls/cases/ \
  --ninja "$RN2/controls/ninja"
"$PY" "$HERE/assemble.py" --cases "$RN2"/cases/*-n7-r*.json --work "$WORK/n7" --out "$RN2/reference" \
  --summary "$RN2/summary/n7.json" --case-prefix cases/ --twins "$RN2/summary/n7-controls.json"
for set in n7 n7-controls; do
  "$PY" "$HERE/export_scalars.py" "$WORK/$set/scalars-exact" "$RN2/scalars/$set.json"
done
