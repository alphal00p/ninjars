#!/usr/bin/env bash
# layout_pad.sh <cpu> <out.tsv> <repetitions> <case.json> <binary>...
#
# Heap-layout sweep of one case's width-1 replayed reduction: the case file is
# copied with 0..256 bytes of trailing whitespace, which changes only the size
# of the buffer the case-bench worker thread reads it into, and with it the
# addresses (mod 64) of the buffers allocated after it in that thread's malloc
# arena. Each binary runs case-bench (width 1, 9 blocks) on every variant,
# interleaved. Output rows: binary, padding, repetition, ns per point.
# Table: pad_table.py <out.tsv>. Inputs from $INPUTS (default: current dir).
CPU=$1; OUT=$2; REPS=$3; CASE=$4; shift 4
cd "${INPUTS:-$PWD}" || exit 1
TMP=$(mktemp -d)
for pad in 0 16 32 48 64 80 96 112 128 160 192 256; do
  cp "$CASE" "$TMP/case-p$pad.json"; printf '%*s' $pad '' >> "$TMP/case-p$pad.json"
done
: > "$OUT"
for rep in $(seq "$REPS"); do
  for pad in 0 16 32 48 64 80 96 112 128 160 192 256; do
    for bin in "$@"; do
      taskset -c "$CPU" "$bin" case-bench "$TMP/case-p$pad.json" --widths 1 --blocks 9 --min-block-ms 20 \
        -o "$TMP/out.json" > /dev/null 2>&1
      v=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['cases'][0]['per_point_timing'][0]['reduction']['median_ns_per_point'])" "$TMP/out.json")
      echo -e "$(basename "$bin")\t$pad\t$rep\t$v" >> "$OUT"
    done
  done
done
rm -rf "$TMP"
