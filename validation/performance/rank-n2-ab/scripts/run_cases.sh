#!/usr/bin/env bash
# run_cases.sh <ninjars-binary> <out-dir> <cpu> <case.json> [<case.json> ...]
# case-bench of the given cases only (the rerun of design 8 (c)), with the
# options of run_one.sh. Inputs relative to $INPUTS (default: the current directory).
BIN=$1; OUT=$2; CPU=$3; shift 3
INPUTS=${INPUTS:-$PWD}
cd "$INPUTS" || exit 1
mkdir -p "$OUT"
taskset -c "$CPU" "$BIN" case-bench "$@" --widths 1,32 --blocks 9 --min-block-ms 20 \
  --reference-dir validation/benchmark/ninja -o "$OUT/cases.json" > "$OUT/cases.log" 2>&1
echo "cases exit $?"
