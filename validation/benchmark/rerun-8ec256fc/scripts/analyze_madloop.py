#!/usr/bin/env python3
"""MadLoop re-timing subset: per graph, accepted rounds' O3/default per-point values, medians, ratio to the committed summary."""
import json, pathlib, statistics, importlib.util
S = pathlib.Path("/shared/external/ninja/references/session-2026-10-20b/bench-main")
W = pathlib.Path("/shared/external/ninja/references/ninjars_bench_main/validation/benchmark/madloop")
spec = importlib.util.spec_from_file_location("mlsum", W / "summarize.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
committed = {e["graph"]: e for e in json.load(open(W / "summary.json"))}
keys = ["numerator_coefficients", "tensor_numerator", "ninja_reduction", "masters", "full"]
res = {}
for gdir in sorted((S / "rounds/madloop").iterdir()):
    g = gdir.name
    per = {"O3": [], "default": []}
    meta = []
    for rdir in sorted(gdir.glob("round*")):
        acc = json.loads((rdir / "accepted.json").read_text())
        d = json.loads((rdir / f"attempt{acc['attempt']}" / f"{g}.json").read_text())
        for v in per:
            per[v].append(m.per_point(d["timing"][v]))
        meta.append((acc["cpu"], acc["worst_sibling_busy"], acc["accepted"]))
        checks = d.get("ninja_input_check", {}).get("pass"), d.get("diagram_check", {}).get("pass")
    res[g] = {}
    for v in per:
        med = {k: statistics.median(p[k] for p in per[v]) for k in keys}
        old = committed[g]["per_point_us"][v]
        res[g][v] = {"median": med, "committed": {k: old[k] for k in keys}, "ratio": {k: med[k] / old[k] for k in keys},
                     "rounds_full": [p["full"] for p in per[v]]}
        print(g, v, " ".join(f"{k} {med[k]:.2f}/{old[k]:.2f}={med[k]/old[k]:.3f}" for k in keys),
              "| full per round", [round(p["full"], 2) for p in per[v]])
    print("   cpus/sibling/accepted", meta, "checks (inputs, diagram)", checks)
(S / "analysis/madloop.json").write_text(json.dumps(res, indent=1))
