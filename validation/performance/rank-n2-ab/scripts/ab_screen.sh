#!/usr/bin/env bash
# ab_screen.sh <cpu> <out-root> <sequence> <binary-A> <binary-B> [case.json ...]
#
# Interleaved A/B (or A/A) of case-bench on a few cases (default syn-n2-r2 and
# syn-n2-r1), widths 1,32, the options of run_one.sh, one run per letter of
# <sequence> into <out-root>/<NN>-<letter>. Each run is wrapped in
# `perf stat -e cycles:u,instructions:u,task-clock` (perf from $PERF), so the
# process's user-space cycles per instruction are recorded next to the
# timings. Inputs are read from $INPUTS (default: the current directory).
# Analysis: screen_analyze.py.
CPU=$1; ROOT=$2; SEQ=$3; BINA=$4; BINB=$5; shift 5
CASES=("$@")
[ ${#CASES[@]} -eq 0 ] && CASES=(validation/benchmark/synthetic/syn-n2-r2.json validation/benchmark/synthetic/syn-n2-r1.json)
PERF=${PERF:-perf}
cd "${INPUTS:-$PWD}" || exit 1
mkdir -p "$ROOT"
for (( k=0; k<${#SEQ}; k++ )); do
  L=${SEQ:$k:1}
  if [ "$L" = A ]; then BIN=$BINA; else BIN=$BINB; fi
  OUT=$ROOT/$(printf %02d $((k+1)))-$L; mkdir -p "$OUT"
  echo "$(date +%T) run $((k+1)) $L cpu $CPU $(cut -d' ' -f1-3 /proc/loadavg)" >> "$ROOT/progress.log"
  taskset -c "$CPU" "$PERF" stat -x, -e cycles:u,instructions:u,task-clock -o "$OUT/perf.csv" \
    "$BIN" case-bench "${CASES[@]}" --widths 1,32 --blocks 9 --min-block-ms 20 \
    --reference-dir validation/benchmark/ninja -o "$OUT/cases.json" > "$OUT/cases.log" 2>&1
  echo "exit $?" >> "$ROOT/progress.log"
done
echo "$(date +%T) DONE" >> "$ROOT/progress.log"
