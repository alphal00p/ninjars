#!/usr/bin/env bash
# build_probes.sh <checkout> <target-dir>
# Copies the in-process probes (ab_icount.rs, ab_stackprobe.rs,
# ab_bubble_masters.rs) into crates/ninjars-symbolica/tests/ of <checkout>
# (an exported scratch tree, not a working checkout) and builds them with the
# perf profile; prints the test executables. On a tree without the rank n+2
# bubble family (main), ab_bubble_masters.rs is built without BubbleRank4.
HERE=$(cd "$(dirname "$0")" && pwd)
CHECKOUT=$1; TARGET=$2
T=$CHECKOUT/crates/ninjars-symbolica/tests
cp "$HERE/ab_icount.rs" "$HERE/ab_stackprobe.rs" "$T/"
if grep -q BubbleRank4 "$CHECKOUT/crates/ninjars/src/fast/masters.rs"; then
  cp "$HERE/ab_bubble_masters.rs" "$T/"
else
  sed 's/, FastMaster::BubbleRank4\]/]/' "$HERE/ab_bubble_masters.rs" > "$T/ab_bubble_masters.rs"
fi
cd "$CHECKOUT" || exit 1
CARGO_TARGET_DIR=$TARGET cargo test --locked --profile perf -p ninjars-symbolica \
  --test ab_icount --test ab_stackprobe --test ab_bubble_masters --no-run 2>&1 | grep Executable
