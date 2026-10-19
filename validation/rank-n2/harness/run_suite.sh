#!/usr/bin/env bash
# Every evaluation behind validation/rank-n2/{reference,controls/reference,identity/reference,
# scan/reference} and the synthetic rank n+2 references.
#
#   harness/run_suite.sh SET [JOBS]   SET = suite | controls | identity | scan | synthetic |
#                                           synthetic-twins (syn-n{2..6}-r{n,n+1}, ambiguity calibration) |
#                                           n7 | n7-controls | identity-n7 (the n = 7 suite cases, their twins,
#                                           and the n = 7 identity cases with their pieces;
#                                           finalize them with finalize_n7.sh)
#
# Steps (env STEPS, default "collier hp"; all output in $WORK/<set>/, default WORK =
# <repo>/target/rank-n2-work):
#  collier:  coli-quad         COLLIER COLI, quad build, exact inputs            (all cases)
#            dd-quad           COLLIER DD, quad (N <= 6; pentagon rank <= 5, hexagon rank <= 6; none at n = 7)
#            coli-quad-shifts  COLI quad with MomInv from the exact shifts instead of the s_matrix
#            id-dd-quad        exact rank reduction to rank <= N pieces, DD quad, shift invariants (n = 5, 6)
#            id-coli-quad      the same pieces with COLI quad                                 (n >= 5)
#            mp                independent mpmath tadpoles/bubbles (n <= 2)
#            coli-double, dd-double   binary64 COLLIER (diagnostics only)
#  hp:       scalar-cases      rank-0 sub-cases of every 3- and 4-propagator subset (scalar_cases.py)
#            scalars-exact     their exact values: $GENERIC_RUNNER (see README, "Scalar masters")
#            scalars-coli, scalars-dd   COLLIER quad values of the same scalars (scalar_check)
#            reduction-exact, reduction-exact80   review_reduction/ at 60 and 80 digits, exact scalars
#            reduction         the same with COLLIER-COLI scalars (diagnostic)
#            generic-f256      $GENERIC_RUNNER on the rank <= n+1 cases (NinjaRS generic, Float 256)
#  ambiguity: (lanes whose s_matrix differs from the exact shift invariants)
#            scalars-exact-shifts, reduction-exact-shifts   the reduction with every invariant
#                              from the exact shifts (input ambiguity of reduction-type codes)
# GENERIC_RUNNER is a command "RUNNER --bits 256 --out-dir DIR CASE.json..." writing per case
# DIR/<case>.json with lanes[].laurent_decimal (format ninjars-generic-float-run-v1).
# Then: assemble.py (reference files) and compare_hpref.py.  Requires build.sh first.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RN2="$(dirname "$HERE")"
REPO="$(cd "$RN2/../.." && pwd)"
SET=${1:?set}; JOBS=${2:-16}
WORK=${WORK:-$REPO/target/rank-n2-work}
case $SET in
  suite)     CASES=("$RN2"/cases/*.json) ;;
  controls)  CASES=("$RN2"/controls/cases/*.json) ;;
  identity)  CASES=("$RN2"/identity/cases/*.json) ;;
  scan)      CASES=("$RN2"/scan/cases/*.json) ;;
  synthetic) CASES=(); for n in 1 2 3 4 5 6; do CASES+=("$REPO/validation/benchmark/synthetic-rank-n2/syn-n$n-r$((n + 2)).json"); done ;;
  synthetic-twins) CASES=(); for n in 2 3 4 5 6; do for r in $n $((n + 1)); do CASES+=("$REPO/validation/benchmark/synthetic/syn-n$n-r$r.json"); done; done ;;
  n7)        CASES=("$RN2"/cases/*-n7-r*.json) ;;
  n7-controls) CASES=("$RN2"/controls/cases/*-n7-r*.json) ;;
  identity-n7) CASES=("$RN2"/identity/cases/*-n7*.json) ;;
  *) echo "unknown set $SET" >&2; exit 2 ;;
esac
OUT="$WORK/$SET"; mkdir -p "$OUT"
PY="$HERE/python.sh"
sel() {  # cases whose n is in the given list
  local want=" $* " c n
  for c in "${CASES[@]}"; do
    n=$(grep -m1 -o '"n": [0-9]*' "$c" | grep -o '[0-9]*$')
    case "$want" in *" $n "*) echo "$c" ;; esac
  done
}
run() { local tag=$1; shift; "$PY" "$HERE/collier_reference.py" --jobs "$JOBS" --tag "$tag" --out-dir "$OUT/$tag" "$@" > "$OUT/$tag.log"; }
mapfile -t HIGH < <(sel 5 6 7)
mapfile -t HIGH_DD < <(sel 5 6)
mapfile -t LOW < <(sel 1 2)
STEPS=${STEPS:-collier hp ambiguity}
if [[ " $STEPS " == *" collier "* ]]; then
  run coli-quad "${CASES[@]}"
  run dd-quad --mode 2 "${CASES[@]}"
  run coli-quad-shifts --invariants shifts "${CASES[@]}"
  [ ${#HIGH_DD[@]} -gt 0 ] && run id-dd-quad --mode 2 --reduce 0 --invariants shifts "${HIGH_DD[@]}"
  [ ${#HIGH[@]} -gt 0 ] && run id-coli-quad --reduce 0 --invariants shifts "${HIGH[@]}"
  [ ${#LOW[@]} -gt 0 ] && run mp --build mp "${LOW[@]}"
  run coli-double --build precise "${CASES[@]}"
  run dd-double --build precise --mode 2 "${CASES[@]}"
fi
if [[ " $STEPS " == *" hp "* ]]; then
  : "${GENERIC_RUNNER:?set GENERIC_RUNNER (see README)}"
  rm -rf "$OUT/scalar-cases"
  "$PY" "$HERE/scalar_cases.py" "${CASES[@]}" --out "$OUT/scalar-cases" --chunk 800 > "$OUT/scalar-cases.log"
  shopt -s nullglob
  SC=("$OUT"/scalar-cases/*.json)
  if [ ${#SC[@]} -gt 0 ]; then
    for f in "${SC[@]}"; do $GENERIC_RUNNER --bits 256 --out-dir "$OUT/scalars-exact" "$f" > "$f.log" 2>&1 & done
    wait
    run scalars-coli "${SC[@]}"
    run scalars-dd --mode 2 "${SC[@]}"
  fi
  export E2E_SCALARS="$OUT/scalars-exact"
  RR="$HERE/review_reduction/run_reduction.py"
  E2E_DPS=60 "$PY" "$RR" --jobs "$JOBS" --out-dir "$OUT/reduction-exact" "${CASES[@]}" > "$OUT/reduction-exact.log"
  E2E_DPS=80 "$PY" "$RR" --jobs "$JOBS" --out-dir "$OUT/reduction-exact80" "${CASES[@]}" > "$OUT/reduction-exact80.log"
  if [ "${COLLIER_SCALAR_REDUCTION:-0}" = 1 ]; then
    E2E_SCALARS= "$PY" "$RR" --jobs "$JOBS" --out-dir "$OUT/reduction" "${CASES[@]}" > "$OUT/reduction.log"
  fi
  # rank <= n+1 cases, one runner per (n, rank) group (plans are built once per group)
  declare -A GROUP=()
  for c in "${CASES[@]}"; do
    n=$(grep -m1 -o '"n": [0-9]*' "$c" | grep -o '[0-9]*$'); r=$(grep -m1 -o '"rank": [0-9]*' "$c" | grep -o '[0-9]*$')
    [ "$r" -le $((n + 1)) ] && GROUP["n$n-r$r"]+="$c "
  done
  mkdir -p "$OUT/generic-logs"
  for g in "${!GROUP[@]}"; do
    # shellcheck disable=SC2086
    $GENERIC_RUNNER --bits 256 --out-dir "$OUT/generic-f256" ${GROUP[$g]} > "$OUT/generic-logs/$g.log" 2>&1 &
  done
  wait
fi
if [[ " $STEPS " == *" ambiguity "* ]]; then
  : "${GENERIC_RUNNER:?set GENERIC_RUNNER (see README)}"
  mapfile -t AMB < <("$PY" -c "
import json, sys
sys.path.insert(0, sys.argv[1])
from assemble import inconsistent_smatrix
for f in sys.argv[2:]:
    if any(inconsistent_smatrix(l) for l in json.load(open(f))['lanes']):
        print(f)" "$HERE" "${CASES[@]}")
  if [ ${#AMB[@]} -gt 0 ]; then
    rm -rf "$OUT/scalar-cases-shifts"
    "$PY" "$HERE/scalar_cases.py" "${AMB[@]}" --out "$OUT/scalar-cases-shifts" --chunk 400 --invariants shifts > /dev/null
    for f in "$OUT"/scalar-cases-shifts/*.json; do $GENERIC_RUNNER --bits 256 --out-dir "$OUT/scalars-exact-shifts" "$f" > "$f.log" 2>&1 & done
    wait
    E2E_SCALARS="$OUT/scalars-exact-shifts" E2E_DPS=60 "$PY" "$HERE/review_reduction/run_reduction.py" --jobs "$JOBS" \
      --out-dir "$OUT/reduction-exact-shifts" "${AMB[@]}" > "$OUT/reduction-exact-shifts.log"
  fi
fi
echo "runs done: $OUT"
