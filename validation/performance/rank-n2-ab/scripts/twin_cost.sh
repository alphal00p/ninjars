#!/usr/bin/env bash
# twin_cost.sh <ninjars-binary> <ab_icount-executable> <out-dir> <cpu> [repetitions]
#
# Rank n+2 cost against the rank n+1 twins (same kinematics), integration
# build only. Run from the integration checkout (or set $INPUTS to it).
# 1. Fast engine, case-bench (widths 1,32; reduction = replayed numerator and
#    masters, total = TensorNumerator/MuTensorNumerator + native masters) on
#    every pair of twin_pairs.py, <repetitions> times (default 3).
# 2. Fast engine, user-space instructions per point (ab_icount.rs) on the same
#    files.
# 3. Generic engine, `ninjars bench` on the n-gon twins of
#    make_twin_fixtures.py (n = 2..7, ranks n+1 and n+2): binary64 for every n,
#    DoubleFloat and Float(128) with fewer points; the fast engine (`--engine
#    fast`, MuTensorNumerator) on the same fixtures.
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=$1; ICOUNT=$2; OUT=$3; CPU=$4; REPS=${5:-3}
INPUTS=${INPUTS:-$PWD}
cd "$INPUTS" || exit 1
mkdir -p "$OUT/generic" "$OUT/fixtures"
python3 "$HERE/twin_pairs.py" . > "$OUT/pairs.tsv"
FILES=$(cut -f3,4 "$OUT/pairs.tsv" | tr '\t' '\n')
for k in $(seq 1 "$REPS"); do
  taskset -c "$CPU" "$BIN" case-bench $FILES --widths 1,32 --blocks 9 --min-block-ms 20 \
    -o "$OUT/cases-$k.json" > "$OUT/cases-$k.log" 2>&1
  echo "case-bench $k exit $?"
done
ICOUNT_OUT=$OUT/icount.tsv ICOUNT_FILES=$(echo $FILES | tr ' ' ':') taskset -c "$CPU" "$ICOUNT" \
  --ignored --exact icount --test-threads=1 > "$OUT/icount.log" 2>&1
echo "icount exit $?"
python3 "$HERE/make_twin_fixtures.py" "$OUT/fixtures" . > /dev/null
for n in 2 3 4 5 6 7; do
  for r in $((n + 1)) $((n + 2)); do
    f=$OUT/fixtures/ngon-n$n-r$r.json; b=ngon-n$n-r$r
    taskset -c "$CPU" "$BIN" bench "$f" --precision f64 --batch-sizes 1,32 --points 65 --rounds 5 \
      --workload changing -o "$OUT/generic/$b-f64.json" > "$OUT/generic/$b-f64.log" 2>&1
    echo "$b f64 exit $?"
    taskset -c "$CPU" "$BIN" bench "$f" --engine fast --batch-sizes 1,32 --points 65 --rounds 5 \
      --workload changing -o "$OUT/generic/$b-fast.json" > "$OUT/generic/$b-fast.log" 2>&1
    echo "$b fast exit $?"
    taskset -c "$CPU" "$BIN" bench "$f" --precision double-float --batch-sizes 1 --points 9 --rounds 3 \
      --workload changing -o "$OUT/generic/$b-double.json" > "$OUT/generic/$b-double.log" 2>&1
    echo "$b double exit $?"
    taskset -c "$CPU" "$BIN" bench "$f" --precision float --bits 128 --batch-sizes 1 --points 3 --rounds 3 \
      --workload changing -o "$OUT/generic/$b-float128.json" > "$OUT/generic/$b-float128.log" 2>&1
    echo "$b float128 exit $?"
  done
done
