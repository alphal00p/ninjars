#!/usr/bin/env bash
# build.sh <checkout> <target-dir>
# perf build of the ninjars CLI with the numerator JIT (the binary of
# run_final2.sh), into its own CARGO_TARGET_DIR. Needs the project's dev
# shell (Symbolica licence in the environment).
CHECKOUT=$1; TARGET=$2
cd "$CHECKOUT" || exit 1
CARGO_TARGET_DIR=$TARGET cargo build --locked --profile perf -p ninjars-cli --features numerator-jit
