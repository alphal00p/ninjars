#!/usr/bin/env python3
"""pad_table.py OUT.tsv: median (min-max) ns per point per padding and binary (layout_pad.sh)."""
import collections
import statistics as st
import sys

d = collections.defaultdict(list)
for line in open(sys.argv[1]):
    b, p, _, v = line.split()
    d[(b, int(p))].append(float(v))
bins = sorted({k[0] for k in d})
pads = sorted({k[1] for k in d})
print("pad\t" + "\t".join(bins))
for p in pads:
    print(f"{p}\t" + "\t".join(f"{st.median(d[(b, p)]):.0f} ({min(d[(b, p)]):.0f}-{max(d[(b, p)]):.0f})" for b in bins))
print("median\t" + "\t".join(f"{st.median([st.median(d[(b, p)]) for p in pads]):.0f}" for b in bins))
