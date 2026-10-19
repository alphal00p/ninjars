#!/usr/bin/env python3
"""Ratios of ab_screen.sh runs.

usage: screen_analyze.py ROOT [ROOT ...]

Per (case, width, metric) of case-bench (reduction = replayed numerator and
masters, total = live path): the median over runs of the per-run median
ns/point, and the ratio B/A. Also the process's user-space cycles per
instruction (perf stat) per side, and whether the reference checks of all
runs agree. Several ROOTs (replicates on different cores) are combined by the
geometric mean of their ratios.
"""
import collections
import json
import math
import pathlib
import statistics as st
import sys


def load(root):
    vals = collections.defaultdict(lambda: {"A": [], "B": []})
    cpi = {"A": [], "B": []}
    checks = collections.defaultdict(set)
    for r in sorted(p for p in pathlib.Path(root).iterdir() if p.is_dir() and p.name[-2:] in ("-A", "-B")):
        side = r.name[-1]
        f = r / "cases.json"
        if not f.exists():
            continue
        for c in json.loads(f.read_text())["cases"]:
            for t in c["per_point_timing"]:
                for m in ("reduction", "total"):
                    if m in t:
                        vals[(c["case"], t["width"], m)][side].append(t[m]["median_ns_per_point"])
                checks[(c["case"], t["width"])].add(json.dumps(t.get("reference_check"), sort_keys=True))
        p = r / "perf.csv"
        if p.exists():
            d = {}
            for line in p.read_text().splitlines():
                w = line.split(",")
                if len(w) > 3 and w[0][:1].isdigit():
                    d[w[2]] = float(w[0])
            if "cycles:u" in d and "instructions:u" in d:
                cpi[side].append(d["cycles:u"] / d["instructions:u"])
    return vals, cpi, checks


combined = collections.defaultdict(list)
for root in sys.argv[1:]:
    vals, cpi, checks = load(root)
    print(f"== {root}")
    for k in sorted(vals):
        a, b = vals[k]["A"], vals[k]["B"]
        if a and b:
            r = st.median(b) / st.median(a)
            combined[k].append(r)
            print(f"{k[0]:14s} w{k[1]:<3d} {k[2]:9s} A {st.median(a):8.1f} B {st.median(b):8.1f} "
                  f"ratio {r:.4f}  runs {len(a)}+{len(b)}  A {min(a):.0f}..{max(a):.0f}  B {min(b):.0f}..{max(b):.0f}")
    if cpi["A"] and cpi["B"]:
        print(f"process cycles per instruction: A {st.median(cpi['A']):.4f} B {st.median(cpi['B']):.4f} "
              f"ratio {st.median(cpi['B']) / st.median(cpi['A']):.4f}")
    differ = [k for k, v in checks.items() if len(v) != 1]
    print("reference checks identical across runs:", "yes" if not differ else f"NO {differ[:3]}")
if len(sys.argv) > 2:
    print("== geometric mean over replicates")
    for k, rs in sorted(combined.items()):
        g = math.exp(sum(map(math.log, rs)) / len(rs))
        print(f"{k[0]:14s} w{k[1]:<3d} {k[2]:9s} {g:.4f}   " + " ".join(f"{x:.4f}" for x in rs))
