#!/usr/bin/env bash
# End-to-end arbitrary-precision evaluation of the gg -> HHH top pentagon
# (graph 26) at its three pool points: exact FeynKit preparation, the generic
# NinjaRS engine on Symbolica Float, and oneloopmaster's exact master
# expressions at the working precision. Compares 700- and 1000-bit results,
# DoubleFloat and the binary64 fast engine. summarize.py then writes
# arb_summary.json and docs/benchmark/data/arbitrary_precision.json.
set -euo pipefail
NINJARS=${1:-target/perf/ninjars}
FIXTURE=examples/graphs/physical/gg-hhh/top-pentagon-26-pool.json
OUT=validation/benchmark/arbitrary-precision
time "$NINJARS" reduce "$FIXTURE" --precision float --bits 700 -o "$OUT/p700.json"
time "$NINJARS" reduce "$FIXTURE" --precision float --bits 1000 -o "$OUT/p1000.json"
"$NINJARS" reduce "$FIXTURE" --precision double-float -o "$OUT/dd.json"
"$NINJARS" reduce "$FIXTURE" --engine fast -o "$OUT/fast.json"
