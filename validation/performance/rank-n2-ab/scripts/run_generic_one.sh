#!/usr/bin/env bash
# run_generic_one.sh <ninjars-binary> <out-dir> <cpu>
#
# One pass of the generic engine's `ninjars bench` (design section 8, check g)
# on four rank <= n+1 fixtures: box (n=4, r=4), pentagon-rank4 (n=5, r=4),
# heptagon-rank8 (n=7, r=8) and the physical gg-hh top-box-2 pool (n=4), in
# binary64, and box and pentagon also in DoubleFloat and Float(128 bits).
# Inputs are read from $INPUTS (default: the current directory).
BIN=$1; OUT=$2; CPU=$3
INPUTS=${INPUTS:-$PWD}
cd "$INPUTS" || exit 1
mkdir -p "$OUT"
run() {
  local name=$1; shift
  taskset -c "$CPU" "$BIN" bench "$@" --workload changing --rounds 5 -o "$OUT/$name.json" > "$OUT/$name.log" 2>&1
  echo "$name exit $?"
}
for f in box pentagon-rank4 heptagon-rank8; do
  run "$f-f64" "examples/graphs/$f.json" --precision f64 --batch-sizes 1,32 --points 65
done
run gg-hh-top-box-2-f64 examples/graphs/physical/gg-hh/top-box-2-pool.json --feynman-rules \
  --precision f64 --batch-sizes 1,32 --points 65
for f in box pentagon-rank4; do
  run "$f-double" "examples/graphs/$f.json" --precision double-float --batch-sizes 1,8 --points 17
  run "$f-float128" "examples/graphs/$f.json" --precision float --bits 128 --batch-sizes 1,8 --points 9
done
