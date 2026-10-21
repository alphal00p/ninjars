#!/usr/bin/env bash
# Live-total instructions per point of case-bench (new CLI) by difference of --total-calls 0 and N.
# usage: icount_cases.sh CPU OUT.tsv
CPU=$1; OUT=$2
PERF=/nix/store/gyp2si1k1w7jhw8z4xx1bwr2m0pr5445-perf-linux-7.2/bin/perf
W=/shared/external/ninja/references/ninjars_bench_main
T=/shared/external/ninja/references/session-2026-10-20b/bench-main/icount
mkdir -p $T
cd $W
echo -e "case\twidth\tN\tinstr_N0\tinstr_N\tinstr_per_point" > $OUT
for f in validation/benchmark/cases/{four-photon-top-box-0,gg-hhh-top-box-8,gg-hhh-top-pentagon-26,gg-hhh-top-triangle-0,gg-hh-top-box-2,gg-hh-top-triangle-0,hgg-top-triangle-0}.json validation/benchmark/synthetic/syn-*.json; do
  c=$(basename $f .json)
  for w in 1 32; do
    N=$(( w == 1 ? 3000 : 200 ))
    i0=$(taskset -c $CPU $PERF stat -x, -e instructions:u -- target/perf/ninjars case-bench $f --widths $w --total-calls 1 -o $T/x.json 2>&1 >/dev/null | grep instructions | cut -d, -f1)
    i1=$(taskset -c $CPU $PERF stat -x, -e instructions:u -- target/perf/ninjars case-bench $f --widths $w --total-calls $((N+1)) -o $T/x.json 2>&1 >/dev/null | grep instructions | cut -d, -f1)
    pp=$(python3 -c "print(($i1-$i0)/($N*$w))")
    echo -e "$c\t$w\t$N\t$i0\t$i1\t$pp" >> $OUT
  done
done
echo DONE >> $OUT.done
