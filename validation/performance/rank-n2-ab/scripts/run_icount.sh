#!/usr/bin/env bash
# run_icount.sh <cpu> <ab_icount-executable> <out.tsv>
# Runs the ignored `icount` test of ab_icount.rs pinned to <cpu>; ICOUNT_DIRS,
# ICOUNT_FILES and ICOUNT_WIDTHS pass through (see ab_icount.rs).
CPU=$1; EXE=$2; OUT=$3
ICOUNT_OUT=$OUT taskset -c "$CPU" "$EXE" --ignored --exact icount --test-threads=1 > "$OUT.log" 2>&1
echo "$(basename "$OUT") exit $?"
