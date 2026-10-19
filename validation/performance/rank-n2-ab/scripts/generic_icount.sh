#!/usr/bin/env bash
# generic_icount.sh <ninjars-binary> <cpu> <out.tsv>
# User-space instructions per timed point of the generic engine's `ninjars
# bench` (fixed workload): the difference of two runs with --rounds 2 and
# --rounds 12 (setup, warmup and the diagnostic pass cancel), divided by
# 10 rounds x points. Run from a checkout ($INPUTS).
BIN=$1; CPU=$2; OUT=$3
PERF=${PERF:-perf}
INPUTS=${INPUTS:-$PWD}
cd "$INPUTS" || exit 1
: > "$OUT"
count() {
  taskset -c "$CPU" "$PERF" stat -x, -e instructions:u -o "$OUT.tmp" "$BIN" bench "$@" --workload changing -o /dev/null > /dev/null 2>&1
  awk -F, '/instructions/{print $1}' "$OUT.tmp"
}
for spec in "box-f64 examples/graphs/box.json --precision f64 --points 65" \
            "pentagon-rank4-f64 examples/graphs/pentagon-rank4.json --precision f64 --points 65" \
            "heptagon-rank8-f64 examples/graphs/heptagon-rank8.json --precision f64 --points 65" \
            "gg-hh-top-box-2-f64 examples/graphs/physical/gg-hh/top-box-2-pool.json --feynman-rules --precision f64 --points 65" \
            "box-double examples/graphs/box.json --precision double-float --points 17" \
            "box-float128 examples/graphs/box.json --precision float --bits 128 --points 9"; do
  set -- $spec; name=$1; shift
  pts=$(echo "$spec" | sed -E 's/.*--points ([0-9]+).*/\1/')
  for w in 1 32; do
    lo=$(count "$@" --batch-sizes $w --rounds 2)
    hi=$(count "$@" --batch-sizes $w --rounds 12)
    echo -e "$name\t$w\t$(( (hi - lo) / (10 * pts) ))" >> "$OUT"
  done
done
rm -f "$OUT.tmp"
