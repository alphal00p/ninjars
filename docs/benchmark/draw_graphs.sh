#!/usr/bin/env bash
# Draw every benchmarked graph with FeynKit + Linnest (GammaLoop's physics
# styles) and compile the drawings to SVG for the report. The Linnest/Kurvst
# Typst packages, the physics drawing templates and GammaLoop's vendored Typst
# packages are staged from the pinned FeynKit checkout into target/linnet-root.
# Usage (from the repository root, after building the CLI, with typst on PATH):
#   docs/benchmark/draw_graphs.sh [path/to/ninjars]
set -euo pipefail
NINJARS=${1:-target/perf/ninjars}
FEYNKIT=${FEYNKIT:-references/gammaloop_ninja_shared}
ROOT=target/linnet-root
OUT=docs/benchmark/linnet
mkdir -p "$ROOT/crates/linnest" "$ROOT/crates/kurvst" "$ROOT/assets/embedded/drawing" "$ROOT/typst-packages"
for tree in crates/linnest/typst crates/kurvst/typst; do
  rsync -a --delete --exclude examples --exclude docs --exclude benches --exclude tests \
    "$FEYNKIT/$tree/" "$ROOT/$tree/"
done
rsync -a --delete "$FEYNKIT/assets/embedded/drawing/templates/" "$ROOT/assets/embedded/drawing/templates/"
rsync -a --delete "$FEYNKIT/vendor/typst-packages/" "$ROOT/typst-packages/"
for g in hgg/top-triangle-0 four-photon/top-box-0 gg-hh/top-triangle-0 gg-hh/top-box-2 \
         gg-hhh/top-triangle-0 gg-hhh/top-box-8 gg-hhh/top-pentagon-26; do
  "$NINJARS" draw --config '(title: none)' examples/graphs/physical/$g.dot \
    -o "$ROOT/graphs/$(dirname $g)" > /dev/null
done
"$NINJARS" draw --model phi3 --config '(title: none, style: (edge-label: none))' \
  examples/graphs/bubble.dot examples/graphs/triangle.dot examples/graphs/box.dot \
  examples/graphs/pentagon-rank4.dot docs/benchmark/graphs/hexagon.dot \
  -o "$ROOT/graphs/synthetic" > /dev/null
for doc in "$ROOT"/graphs/*/*.typ; do
  rel=${doc#"$ROOT/graphs/"}
  mkdir -p "$OUT/$(dirname "$rel")"
  typst compile --root "$ROOT" --package-path "$ROOT/typst-packages" "$doc" "$OUT/${rel%.typ}.svg"
done
