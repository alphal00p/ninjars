#!/usr/bin/env bash
# run_one.sh <ninjars-binary> <out-dir> <cpu> [pipeline|cases|both]
#
# One full benchmark pass with the commands of run_final2.sh (design section 8):
# pipeline-bench on the 7 physical graphs (widths 1,8,32,128) and case-bench on
# the 27 benchmark cases (7 physical + 20 synthetic, widths 1,32), all pinned to
# <cpu> with taskset. Inputs are read from $INPUTS (a checkout whose
# examples/graphs/physical and validation/benchmark inputs are used by both
# sides of the A/B; default: the current directory).
BIN=$1; OUT=$2; CPU=$3; WHAT=${4:-both}
INPUTS=${INPUTS:-$PWD}
cd "$INPUTS" || exit 1
mkdir -p "$OUT"
if [ "$WHAT" != cases ]; then
  for spec in hgg:hgg/top-triangle-0 four-photon:four-photon/top-box-0 gg-hh:gg-hh/top-triangle-0 \
              gg-hh:gg-hh/top-box-2 gg-hhh:gg-hhh/top-triangle-0 gg-hhh:gg-hhh/top-box-8 \
              gg-hhh:gg-hhh/top-pentagon-26; do
    proc=${spec%%:*}; graph=${spec#*:}; name=$proc-$(basename "$graph")
    taskset -c "$CPU" "$BIN" pipeline-bench "examples/graphs/physical/$graph-pool.json" \
      --points "validation/benchmark/points/$proc.json" --feynman-rules --widths 1,8,32,128 \
      --blocks 9 --min-block-ms 20 --reference "validation/benchmark/ninja/$name.json" \
      -o "$OUT/pipeline-$name.json" > "$OUT/pipeline-$name.log" 2>&1
    echo "$name exit $?"
  done
fi
if [ "$WHAT" != pipeline ]; then
  taskset -c "$CPU" "$BIN" case-bench validation/benchmark/cases/*.json \
    validation/benchmark/synthetic/*.json --widths 1,32 --blocks 9 --min-block-ms 20 \
    --reference-dir validation/benchmark/ninja -o "$OUT/cases.json" > "$OUT/cases.log" 2>&1
  echo "cases exit $?"
fi
