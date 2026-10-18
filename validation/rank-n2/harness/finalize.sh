#!/usr/bin/env bash
# Write the committed outputs from the runs of run_suite.sh (every set): references,
# summaries, exact scalar tables, hpref comparison, identity residuals, UV-pole check.
#   harness/finalize.sh [HPREF_DIR]   (HPREF_DIR default validation/rank-n2/provenance/review-hpref)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RN2="$(dirname "$HERE")"
REPO="$(cd "$RN2/../.." && pwd)"
WORK=${WORK:-$REPO/target/rank-n2-work}
PY="$HERE/python.sh"
HPREF=${1:-$RN2/provenance/review-hpref}
mkdir -p "$RN2/summary" "$RN2/scalars"
asm() {  # set cases out-dir case-prefix [extra assemble.py options]
  local set=$1 cases=$2 out=$3 prefix=$4; shift 4
  "$PY" "$HERE/assemble.py" --cases "$cases" --work "$WORK/$set" --out "$out" \
    --summary "$RN2/summary/$set.json" --case-prefix "$prefix" "$@"
}
# controls (and the synthetic twins) first: the suite's input ambiguity uses their generic-engine runs
asm controls "$RN2/controls/cases" "$RN2/controls/reference" controls/cases/ --ninja "$RN2/controls/ninja"
TW=(); for n in 2 3 4 5 6; do for r in $n $((n + 1)); do TW+=("$REPO/validation/benchmark/synthetic/syn-n$n-r$r.json"); done; done
"$PY" "$HERE/assemble.py" --cases "${TW[@]}" --work "$WORK/synthetic-twins" --out "$RN2/controls/reference" \
  --summary "$RN2/summary/synthetic-twins.json" --case-prefix validation/benchmark/synthetic/ \
  --ninja "$REPO/validation/benchmark/ninja"
asm suite "$RN2/cases" "$RN2/reference" cases/ --twins "$RN2/summary/controls.json"
asm synthetic "$REPO/validation/benchmark/synthetic-rank-n2" "$RN2/reference" validation/benchmark/synthetic-rank-n2/ \
  --twins "$RN2/summary/synthetic-twins.json"
asm identity "$RN2/identity/cases" "$RN2/identity/reference" identity/cases/
asm scan "$RN2/scan/cases" "$RN2/scan/reference" scan/cases/
export_scalars() {  # run-dir out-file
  "$PY" "$HERE/export_scalars.py" "$1" "$2"
}
for set in suite synthetic controls synthetic-twins identity scan; do
  export_scalars "$WORK/$set/scalars-exact" "$RN2/scalars/$set.json"
  if [ -d "$WORK/$set/scalars-exact-shifts" ]; then
    export_scalars "$WORK/$set/scalars-exact-shifts" "$RN2/scalars/shift-invariants/$set.json"
  fi
done
"$PY" "$HERE/compare_hpref.py" --hpref "$HPREF" --refs "$RN2/reference" --work "$WORK/suite" \
  --json "$RN2/provenance/hpref-comparison.json" | tail -1
"$PY" "$HERE/verify_identities.py" --manifest "$RN2/identity/manifest.json" --refs "$RN2/identity/reference" \
  --run "$WORK/identity/coli-quad" --json "$RN2/identity/residuals.json"
"$PY" "$HERE/uv_check.py" "$RN2/cases" "$RN2/reference" | tail -1
"$PY" "$HERE/uv_check.py" "$RN2/controls/cases" "$RN2/controls/reference" | tail -1
