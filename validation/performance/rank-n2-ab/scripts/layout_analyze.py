#!/usr/bin/env python3
"""Ratios of layout_sweep.sh over heap layouts.

usage: layout_analyze.py ROOT [--tsv out.tsv]

Per (case, width, metric) and layout p<pad>: the median over repetitions of
the per-run median ns/point of A and of B. Reported per key:
- the ratio B/A per layout and its geometric mean over layouts;
- the layout spread of each side: max/min over layouts of its medians.
"""
import argparse
import collections
import json
import math
import pathlib
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("root", type=pathlib.Path)
ap.add_argument("--tsv", type=pathlib.Path)
args = ap.parse_args()
vals = collections.defaultdict(lambda: collections.defaultdict(lambda: {"A": [], "B": []}))
pads = sorted(int(p.name[1:]) for p in args.root.iterdir() if p.is_dir() and p.name.startswith("p") and p.name[1:].isdigit())
for pad in pads:
    for r in sorted((args.root / f"p{pad}").iterdir()):
        f = r / "cases.json"
        if not f.exists():
            continue
        for c in json.loads(f.read_text())["cases"]:
            for t in c["per_point_timing"]:
                for m in ("reduction", "total"):
                    if m in t:
                        vals[(c["case"], t["width"], m)][pad][r.name[-1]].append(t[m]["median_ns_per_point"])


def gm(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs))


rows = []
for k in sorted(vals):
    per = vals[k]
    a = {p: st.median(per[p]["A"]) for p in pads if per[p]["A"] and per[p]["B"]}
    b = {p: st.median(per[p]["B"]) for p in pads if per[p]["A"] and per[p]["B"]}
    if not a:
        continue
    ratios = [b[p] / a[p] for p in a]
    rows.append((k, gm(ratios), ratios, max(a.values()) / min(a.values()), max(b.values()) / min(b.values()), a, b))
print(f"{'case':26s} {'w':>3s} {'metric':9s} {'gm B/A':>7s}  per layout B/A ({', '.join(f'p{p}' for p in pads)})"
      f"  layout spread A, B")
for k, g, rs, sa, sb, _, _ in rows:
    print(f"{k[0]:26s} {k[1]:3d} {k[2]:9s} {g:7.4f}  " + " ".join(f"{x:.3f}" for x in rs) + f"   {sa:.3f} {sb:.3f}")
for w in (1, 32):
    for m in ("reduction", "total"):
        sel = [r for r in rows if r[0][1] == w and r[0][2] == m]
        if sel:
            print(f"width {w:2d} {m:9s}: keys {len(sel)}, geomean B/A {gm([r[1] for r in sel]):.4f}, "
                  f"max {max(r[1] for r in sel):.3f}; median layout spread A {st.median([r[3] for r in sel]):.3f} "
                  f"(max {max(r[3] for r in sel):.3f}), B {st.median([r[4] for r in sel]):.3f} (max {max(r[4] for r in sel):.3f})")
if args.tsv:
    with open(args.tsv, "w") as f:
        f.write("case\twidth\tmetric\tgeomean_ratio\t" + "\t".join(f"p{p}:A_ns\tp{p}:B_ns\tp{p}:ratio" for p in pads)
                + "\tspread_A\tspread_B\n")
        for k, g, rs, sa, sb, a, b in rows:
            f.write(f"{k[0]}\t{k[1]}\t{k[2]}\t{g:.4f}\t" + "\t".join(
                f"{a[p]:.1f}\t{b[p]:.1f}\t{b[p] / a[p]:.4f}" for p in pads if p in a) + f"\t{sa:.4f}\t{sb:.4f}\n")
