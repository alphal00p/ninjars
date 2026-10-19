#!/usr/bin/env bash
# run_all.sh <tree-A> <tree-B> <work-dir> <cpu-x> <cpu-y> <cpu-z>
#
# The full guard as run for README.md (design section 8), on exported trees
# (`git archive <rev> | tar -x -C <tree>`) of the baseline (A) and the
# candidate (B). <cpu-x> and <cpu-y> are idle cores on two different L3
# complexes (scripts/idle_l3.py), <cpu-z> a third core for the
# deterministic and B-only measurements. Needs the project's dev shell.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
A=$1; B=$2; W=$3; X=$4; Y=$5; Z=$6
mkdir -p "$W/bins" "$W/ab" "$W/icount"
for s in A B; do
  t=$([ $s = A ] && echo "$A" || echo "$B"); l=$(echo $s | tr AB ab)
  "$HERE/build.sh" "$t" "$W/target-$l"
  cp "$W/target-$l/perf/ninjars" "$W/bins/ninjars-$l"
  "$HERE/build_probes.sh" "$t" "$W/target-$l" | tee "$W/probes-$l.txt"
done
cp "$W/bins/ninjars-a" "$W/bins/ninjars-a2"   # A/A: the same binary under a second name
export INPUTS=$A                               # identical benchmark inputs for both sides
# 1. fast engine: interleaved A/B and A/A, two cores, swapped halfway
( "$HERE/ab.sh" "$X" "$W/ab/x-ab" ABBAABBAABBAABBAABBA "$W/bins/ninjars-a" "$W/bins/ninjars-b"
  "$HERE/ab.sh" "$X" "$W/ab/x-aa" ABBAABBAABBAABBA "$W/bins/ninjars-a" "$W/bins/ninjars-a2" ) &
( "$HERE/ab.sh" "$Y" "$W/ab/y-aa" ABBAABBAABBAABBA "$W/bins/ninjars-a" "$W/bins/ninjars-a2"
  "$HERE/ab.sh" "$Y" "$W/ab/y-ab" BAABBAABBAABBAABBAAB "$W/bins/ninjars-a" "$W/bins/ninjars-b" ) &
# 2. generic engine (`ninjars bench`), two interleaved replicates
( "$HERE/ab_generic.sh" "$Z" "$W/ab/g-ab" ABBAABBAABBAABBA "$W/bins/ninjars-a" "$W/bins/ninjars-b"
  "$HERE/ab_generic.sh" "$Z" "$W/ab/g-ab2" BAABBAABBAABBAAB "$W/bins/ninjars-a" "$W/bins/ninjars-b" ) &
wait
python3 "$HERE/analyze.py" --label B/A --tsv "$W/fast-ab.tsv" --json "$W/fast-ab.json" "$W/ab/x-ab" "$W/ab/y-ab"
python3 "$HERE/analyze.py" --label A2/A --tsv "$W/fast-aa.tsv" --json "$W/fast-aa.json" "$W/ab/x-aa" "$W/ab/y-aa"
python3 "$HERE/analyze_generic.py" "$W/ab/g-ab"; python3 "$HERE/analyze_generic.py" "$W/ab/g-ab2"
# 3. reruns of ratios > 1.03 alone, 9 rounds per side (design 8 c), e.g.
"$HERE/ab_cases.sh" "$X" "$W/ab/rerun" ABBAABBAABBAABBAAB "$W/bins/ninjars-a" "$W/bins/ninjars-b" \
  validation/benchmark/synthetic/syn-n2-r2.json validation/benchmark/synthetic/syn-n2-r1.json
python3 "$HERE/analyze.py" --label "rerun B/A" "$W/ab/rerun"
# 4. instructions per point (deterministic): the probes' executables are in probes-{a,b}.txt
#    run_icount.sh <cpu> <ab_icount executable> <out.tsv>, three times per side, then
#    icount_compare.py --a a-*.tsv --b b-*.tsv; generic: generic_icount.sh <bin> <cpu> <out.tsv>
# 5. rank n+2 cost against the rank n+1 twins (B only): INPUTS=$B twin_cost.sh ... ; twin_report.py
# 6. absolute gates: (cd $A && gate_check.py $W/ab/x-ab/*-B)
