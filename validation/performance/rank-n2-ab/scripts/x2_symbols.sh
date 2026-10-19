#!/usr/bin/env bash
# x2_symbols.sh <binary>: mangled names of the rank n+2 text symbols, one per
# line (an LLD --symbol-ordering-file that links them first, out of the
# rank <= n+1 code's neighbourhood; see relink_cli.sh).
paste -d'\t' <(nm --defined-only "$1") <(nm -C --defined-only "$1") \
  | awk -F'\t' '{split($1, a, " "); if (a[2] ~ /^[tTwW]$/) print a[3] "\t" $2}' \
  | grep -E $'\t.*(engine_x2|bases_x2|subtractions_x2|subtractions_series_x2|_x2$|RankN2)' \
  | grep -v 'drop_glue\|RawVec' | cut -f1 | sort -u
