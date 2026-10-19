#!/usr/bin/env bash
# ab_generic.sh <cpu> <out-root> <sequence> <binary-A> <binary-B>: interleaved
# A/B of run_generic_one.sh (see ab.sh).
HERE=$(cd "$(dirname "$0")" && pwd)
CPU=$1; ROOT=$2; SEQ=$3; BINA=$4; BINB=$5
mkdir -p "$ROOT"
for (( k=0; k<${#SEQ}; k++ )); do
  L=${SEQ:$k:1}
  if [ "$L" = A ]; then BIN=$BINA; IN=${INPUTS_A:-${INPUTS:-$PWD}}; else BIN=$BINB; IN=${INPUTS_B:-${INPUTS:-$PWD}}; fi
  OUT=$ROOT/$(printf %02d $((k+1)))-$L
  echo "$(date +%T) run $((k+1)) $L cpu $CPU $(cut -d' ' -f1-3 /proc/loadavg)" >> "$ROOT/progress.log"
  INPUTS=$IN "$HERE/run_generic_one.sh" "$BIN" "$OUT" "$CPU" >> "$ROOT/progress.log" 2>&1
done
echo "$(date +%T) DONE" >> "$ROOT/progress.log"
