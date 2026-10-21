#!/usr/bin/env python3
"""MadLoop timing-only re-time (retime_madloop_scratch.py on the exports built in this session):
medians over the accepted rounds against the committed summary.json (CPU 250)."""
import json, pathlib, statistics
S = pathlib.Path("/shared/external/ninja/references/session-2026-10-20b/bench-main")
W = pathlib.Path("/shared/external/ninja/references/ninjars_bench_main/validation/benchmark/madloop")
committed = {e["graph"]: e for e in json.load(open(W / "summary.json"))}
keys = ["numerator_coefficients", "tensor_numerator", "ninja_reduction", "masters", "full"]
res = {}
for gdir in sorted((S / "rounds/mlretime").iterdir()):
    g = gdir.name
    per = {"O3": [], "default": []}
    meta = []
    for rdir in sorted(gdir.glob("round*")):
        acc = json.loads((rdir / "accepted.json").read_text())
        d = json.loads((rdir / f"attempt{acc['attempt']}" / "retime.json").read_text())[g]
        for v in per:
            per[v].append(d[v]["per_point_us"])
        meta.append((acc["cpu"], acc["worst_sibling_busy"], acc["accepted"]))
    res[g] = {"cpus": meta}
    for v in per:
        med = {k: statistics.median(p[k] for p in per[v]) for k in keys}
        old = committed[g]["per_point_us"][v]
        res[g][v] = {"median": med, "committed": {k: old[k] for k in keys}, "ratio": {k: med[k] / old[k] for k in keys},
                     "rounds_full": [p["full"] for p in per[v]]}
        print(g, v, " ".join(f"{k} {med[k]:.2f}/{old[k]:.2f}={med[k]/old[k]:.3f}" for k in keys),
              "| full", [round(p["full"], 2) for p in per[v]])
    print("   ", meta)
(S / "analysis/mlretime.json").write_text(json.dumps(res, indent=1))
