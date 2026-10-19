#!/usr/bin/env bash
# run_rerun.sh <tree-A> <tree-B> <tree-oldA> <tree-oldB> <work-dir> <cpu-x> <cpu-y> <cpu-z> <cpu-w>
#
# The rerun of README.md "Rerun against main d53dd08" as run: the design 8
# guard of A = main d53dd08 against B = main 0d0d7e7, and the layout
# experiments on the previous pair dec3d55 (oldA) / 8346699 (oldB). Trees are
# detached worktrees (`git worktree add --detach <dir> <rev>`). <cpu-x>,
# <cpu-y>, <cpu-z> are idle cores on three L3 complexes (idle_l3.py), <cpu-w>
# a core on a fourth one for the experiments. Needs the dev shell, perf
# ($PERF) and gcc; builds use a separate CARGO_BUILD_BUILD_DIR per tree.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
A=$1; B=$2; OA=$3; OB=$4; W=$5; X=$6; Y=$7; Z=$8; V=$9
mkdir -p "$W/bins"
build() { CARGO_BUILD_BUILD_DIR=$W/build-$2 "$HERE/build.sh" "$1" "$W/target-$2" && cp "$W/target-$2/perf/ninjars" "$W/bins/ninjars-$2"; }
build "$A" a; build "$B" b; build "$OA" olda; build "$OB" oldb
cp "$W/bins/ninjars-a" "$W/bins/ninjars-a2"
for s in a b; do t=$([ $s = a ] && echo "$A" || echo "$B")
  CARGO_BUILD_BUILD_DIR=$W/build-$s "$HERE/build_probes.sh" "$t" "$W/target-$s" > "$W/probes-$s.txt"; done
# 1. the guard (design 8): each side reads its own checkout (DOT model fingerprints differ)
export INPUTS_A=$A INPUTS_B=$B
( "$HERE/ab.sh" "$X" "$W/ab/x-ab" ABBAABBAABBAABBAABBA "$W/bins/ninjars-a" "$W/bins/ninjars-b"
  INPUTS_B=$A "$HERE/ab.sh" "$X" "$W/ab/x-aa" ABBAABBAABBAABBA "$W/bins/ninjars-a" "$W/bins/ninjars-a2" ) &
( INPUTS_B=$A "$HERE/ab.sh" "$Y" "$W/ab/y-aa" ABBAABBAABBAABBA "$W/bins/ninjars-a" "$W/bins/ninjars-a2"
  "$HERE/ab.sh" "$Y" "$W/ab/y-ab" BAABBAABBAABBAABBAAB "$W/bins/ninjars-a" "$W/bins/ninjars-b" ) &
( "$HERE/ab_generic.sh" "$Z" "$W/ab/g-ab" ABBAABBAABBAABBA "$W/bins/ninjars-a" "$W/bins/ninjars-b"
  "$HERE/ab_generic.sh" "$Z" "$W/ab/g-ab2" BAABBAABBAABBAAB "$W/bins/ninjars-a" "$W/bins/ninjars-b" ) &
wait
python3 "$HERE/analyze.py" --label B/A --tsv "$W/fast-ab.tsv" --json "$W/fast-ab.json" "$W/ab/x-ab" "$W/ab/y-ab"
python3 "$HERE/analyze.py" --label A2/A --tsv "$W/fast-aa.tsv" --json "$W/fast-aa.json" "$W/ab/x-aa" "$W/ab/y-aa"
python3 "$HERE/analyze_generic.py" "$W/ab/g-ab"; python3 "$HERE/analyze_generic.py" "$W/ab/g-ab2"
# 2. instructions per point: run_icount.sh three times per side (executables in probes-{a,b}.txt),
#    then icount_compare.py; (cd $A && gate_check.py $W/ab/x-ab/*-A) and (cd $B && gate_check.py $W/ab/x-ab/*-B)
# 3. ratios in (1.03, 1.05]: ab_cases.sh alone, 9 rounds per side (design 8 c)
# 4. layout experiments, syn-n2-r2 width 1 (INPUTS=$OA):
#    ab_screen.sh on (oldA, oldB) and (A, B); x2_symbols.sh + relink_cli.sh --symbol-ordering-file;
#    coldmark.py on a copy of $OB + build + relink_cli.sh -z keep-text-section-prefix;
#    multi_screen.sh "$V" ... oldA oldB variants; multi_analyze.py --ref ninjars-olda
#    layout_pad.sh "$V" out.tsv 2 validation/benchmark/synthetic/syn-n2-r2.json <bins>, with and without
#    LD_PRELOAD=libalign64.so (gcc -O2 -shared -fPIC -o libalign64.so align64.c); pad_table.py
#    layout_sweep.sh "$V" $W/sweep 3 bins/ninjars-a bins/ninjars-b; layout_analyze.py $W/sweep
