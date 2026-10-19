#!/usr/bin/env bash
# multi_screen.sh <cpu> <out-root> <rounds> <binary>... : case-bench of syn-n2-r2 and
# syn-n2-r1 (widths 1,32, options of run_one.sh) for several binaries, interleaved
# round-robin with the starting binary rotated each round; runs under perf stat
# (user-space cycles and instructions). Output <out-root>/<round>-<k>-<binary>/.
# Analysis: multi_analyze.py <out-root> <reference binary name>.
CPU=$1; ROOT=$2; ROUNDS=$3; shift 3
BINS=("$@"); N=${#BINS[@]}
PERF=${PERF:-perf}
cd "${INPUTS:-$PWD}" || exit 1
mkdir -p "$ROOT"
for r in $(seq "$ROUNDS"); do
  for k in $(seq 0 $((N - 1))); do
    BIN=${BINS[$(( (k + r) % N ))]}
    OUT=$ROOT/$(printf %02d "$r")-$k-$(basename "$BIN"); mkdir -p "$OUT"
    echo "$(date +%T) round $r $(basename "$BIN") $(cut -d' ' -f1-3 /proc/loadavg)" >> "$ROOT/progress.log"
    taskset -c "$CPU" "$PERF" stat -x, -e cycles:u,instructions:u -o "$OUT/perf.csv" "$BIN" case-bench \
      validation/benchmark/synthetic/syn-n2-r2.json validation/benchmark/synthetic/syn-n2-r1.json \
      --widths 1,32 --blocks 9 --min-block-ms 20 --reference-dir validation/benchmark/ninja \
      -o "$OUT/cases.json" > "$OUT/cases.log" 2>&1
  done
done
echo "$(date +%T) DONE" >> "$ROOT/progress.log"
