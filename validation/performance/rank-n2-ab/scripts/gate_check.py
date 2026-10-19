#!/usr/bin/env python3
"""Absolute gates (design 8 f): NinjaRS case-bench against original Ninja
(o3-fortranrules, `validation/benchmark/ninja/<case>.json` timings).

usage: gate_check.py RUN_DIR [RUN_DIR ...] [--ref-dir DIR] [--width W]

Per case the median over the given runs' cases.json of reduction, masters
and total ns/point, divided by Ninja's pure_reduction, masters_only and full
medians. Default width 32, default ref dir validation/benchmark/ninja.
"""
import argparse
import json
import pathlib
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+", type=pathlib.Path)
ap.add_argument("--ref-dir", type=pathlib.Path, default=pathlib.Path("validation/benchmark/ninja"))
ap.add_argument("--width", type=int, default=32)
args = ap.parse_args()
acc = {}
for r in args.runs:
    for c in json.loads((r / "cases.json").read_text())["cases"]:
        e = [x for x in c["per_point_timing"] if x["width"] == args.width][0]
        s = e["summary_ns_per_point"]
        acc.setdefault(c["case"], []).append((s["reduction"], s["masters"], s["total"],
                                              (e.get("reference_check") or {}).get("max_scaled_laurent_difference")))
print(f"{'case':26s} {'red x':>6s} {'mast x':>6s} {'tot x':>6s} {'maxdiff':>9s}")
worst = [0, 0, 0]
for case, vals in acc.items():
    rf = args.ref_dir / f"{case}.json"
    if not rf.exists():
        print(f"{case:26s} no reference")
        continue
    t = json.loads(rf.read_text())["timing"]["o3-fortranrules"]["median_ns"]
    red = st.median(v[0] for v in vals) / t["pure_reduction"]
    mast = st.median(v[1] for v in vals) / t["masters_only"]
    tot = st.median(v[2] for v in vals) / t["full"]
    worst = [max(worst[0], red), max(worst[1], mast), max(worst[2], tot)]
    diff = max((v[3] for v in vals if v[3] is not None), default=float("nan"))
    print(f"{case:26s} {red:6.2f} {mast:6.2f} {tot:6.2f} {diff:9.2e}")
print(f"{'max':26s} {worst[0]:6.2f} {worst[1]:6.2f} {worst[2]:6.2f}")
