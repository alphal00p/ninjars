#!/usr/bin/env python3
"""multi_analyze.py ROOT [ROOT ...] --ref NAME: per binary and key (case, width, metric),
the median over rounds of the per-run median ns/point and its ratio to binary NAME;
several ROOTs (cores) are combined by the geometric mean of their ratios."""
import argparse, collections, json, math, pathlib, statistics as st
ap = argparse.ArgumentParser(); ap.add_argument("roots", nargs="+", type=pathlib.Path); ap.add_argument("--ref", required=True)
args = ap.parse_args()
ratios = collections.defaultdict(list); absval = collections.defaultdict(list)
for root in args.roots:
    vals = collections.defaultdict(list)
    for r in sorted(p for p in root.iterdir() if p.is_dir()):
        b = r.name.split("-", 2)[2]; f = r / "cases.json"
        if not f.exists(): continue
        for c in json.loads(f.read_text())["cases"]:
            for t in c["per_point_timing"]:
                for m in ("reduction", "total"):
                    vals[(b, c["case"], t["width"], m)].append(t[m]["median_ns_per_point"])
    for (b, c, w, m), v in vals.items():
        ref = vals.get((args.ref, c, w, m))
        if ref:
            ratios[(b, c, w, m)].append(st.median(v) / st.median(ref)); absval[(b, c, w, m)].append(st.median(v))
bins = sorted({k[0] for k in ratios})
keys = sorted({k[1:] for k in ratios})
print(f"{'case':10s} {'w':>3s} {'metric':9s} " + " ".join(f"{b[:22]:>22s}" for b in bins))
for k in keys:
    row = []
    for b in bins:
        rs = ratios.get((b,) + k, [])
        row.append(f"{math.exp(sum(map(math.log, rs)) / len(rs)):.3f} ({st.median(absval[(b,) + k]):.0f})" if rs else "-")
    print(f"{k[0]:10s} {k[1]:3d} {k[2]:9s} " + " ".join(f"{x:>22s}" for x in row))
