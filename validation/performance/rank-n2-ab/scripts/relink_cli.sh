#!/usr/bin/env bash
# relink_cli.sh <checkout> <target-dir> <out-binary> <rustc args...>
# Rebuilds only the ninjars CLI binary (the perf build of build.sh) with extra
# rustc arguments, typically linker options, for example
#   -C link-arg=-Wl,--symbol-ordering-file=<x2_symbols.sh output> -C link-arg=-Wl,--no-warn-symbol-ordering
#   -C link-arg=-Wl,-z,keep-text-section-prefix
# and copies the binary to <out-binary>. Dependencies are not rebuilt.
CHECKOUT=$1; TARGET=$2; OUT=$3; shift 3
cd "$CHECKOUT" || exit 1
CARGO_TARGET_DIR=$TARGET cargo rustc --locked --profile perf -p ninjars-cli --bin ninjars \
  --features numerator-jit -- "$@" && cp "$TARGET/perf/ninjars" "$OUT"
