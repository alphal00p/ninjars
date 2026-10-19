#!/usr/bin/env bash
# ab.sh <cpu> <out-root> <sequence> <binary-A> <binary-B> [pipeline|cases|both]
#
# Interleaved A/B: runs run_one.sh once per letter of <sequence> (for example
# ABBAABBAABBAABBAABBA), with binary A or B, into <out-root>/<NN>-<letter>.
# An A/A noise run passes two copies of the same binary.
# Inputs: $INPUTS_A for A and $INPUTS_B for B (default $INPUTS, then the
# current directory). Each side needs its own checkout's physical graphs when
# their DOT model fingerprints differ (gammaloop pin bumps).
HERE=$(cd "$(dirname "$0")" && pwd)
CPU=$1; ROOT=$2; SEQ=$3; BINA=$4; BINB=$5; WHAT=${6:-both}
mkdir -p "$ROOT"
for (( k=0; k<${#SEQ}; k++ )); do
  L=${SEQ:$k:1}
  if [ "$L" = A ]; then BIN=$BINA; IN=${INPUTS_A:-${INPUTS:-$PWD}}; else BIN=$BINB; IN=${INPUTS_B:-${INPUTS:-$PWD}}; fi
  OUT=$ROOT/$(printf %02d $((k+1)))-$L
  echo "$(date +%T) run $((k+1)) $L cpu $CPU $(cut -d' ' -f1-3 /proc/loadavg)" >> "$ROOT/progress.log"
  INPUTS=$IN "$HERE/run_one.sh" "$BIN" "$OUT" "$CPU" "$WHAT" >> "$ROOT/progress.log" 2>&1
done
echo "$(date +%T) DONE" >> "$ROOT/progress.log"
