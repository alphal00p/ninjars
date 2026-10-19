#!/usr/bin/env bash
# layout_sweep.sh <cpu> <out-root> <repetitions> <binary-A> <binary-B>
#
# The case-bench part of run_one.sh (27 cases, widths 1,32) on four heap
# layouts: every case file padded with 0, 48, 112 or 192 bytes of trailing
# whitespace (see layout_pad.sh). A and B alternate per layout, the order
# swapped from one layout to the next. Output:
# <out-root>/p<pad>/<rep><0|1>-<A|B>/cases.json, analysed per layout with
# `analyze.py <out-root>/p<pad>` and over layouts with layout_analyze.py.
# Inputs from $INPUTS (default: current directory).
CPU=$1; ROOT=$2; REPS=$3; BINA=$4; BINB=$5
cd "${INPUTS:-$PWD}" || exit 1
for pad in 0 48 112 192; do
  mkdir -p "$ROOT/cases-p$pad"
  for f in validation/benchmark/cases/*.json validation/benchmark/synthetic/*.json; do
    cp "$f" "$ROOT/cases-p$pad/"; printf '%*s' $pad '' >> "$ROOT/cases-p$pad/$(basename "$f")"
  done
done
k=0
for rep in $(seq "$REPS"); do
  for pad in 0 48 112 192; do
    if [ $(( (rep + k) % 2 )) -eq 0 ]; then ORDER="A B"; else ORDER="B A"; fi
    for L in $ORDER; do
      if [ "$L" = A ]; then BIN=$BINA; else BIN=$BINB; fi
      OUT=$ROOT/p$pad/$(printf %02d "$rep")$([ "$L" = A ] && echo 0 || echo 1)-$L; mkdir -p "$OUT"
      echo "$(date +%T) rep $rep pad $pad $L $(cut -d' ' -f1-3 /proc/loadavg)" >> "$ROOT/progress.log"
      taskset -c "$CPU" "$BIN" case-bench "$ROOT/cases-p$pad"/*.json --widths 1,32 --blocks 9 --min-block-ms 20 \
        --reference-dir validation/benchmark/ninja -o "$OUT/cases.json" > "$OUT/cases.log" 2>&1
    done
    k=$((k + 1))
  done
done
echo "$(date +%T) DONE" >> "$ROOT/progress.log"
