#!/usr/bin/env python3
"""Ninja + OneLOop replay re-timing (o3-fortranrules, same executables as the committed run): per case and
round the medians of pure_reduction, numerator part, masters_only and full against the committed CPU-250 values."""
import json, pathlib, statistics
S = pathlib.Path("/shared/external/ninja/references/session-2026-10-20b/bench-main")
C = pathlib.Path("/shared/external/ninja/references/ninjars_bench_main/validation/benchmark/ninja")
res = {}
for gdir in sorted((S / "rounds/ninja").iterdir()):
    g = gdir.name
    old = json.loads((C / f"{g}.json").read_text())
    od = old["timing"]["o3-fortranrules"]["derived"]
    rounds = []
    for rdir in sorted(gdir.glob("round*")):
        acc = json.loads((rdir / "accepted.json").read_text())
        d = json.loads((rdir / f"attempt{acc['attempt']}" / "out" / f"{g}.json").read_text())
        t = d["timing"]["o3-fortranrules"]
        assert t["driver_binary_sha256"] == old["timing"]["o3-fortranrules"]["driver_binary_sha256"]
        assert d["all_replays_bitwise_identical"]
        same = all(a[k] == b[k] for a, b in zip(d["lanes"], old["lanes"]) for k in ("eps0", "eps_m1", "eps_m2", "cut_constructible", "rational"))
        rounds.append({"derived": t["derived"], "cpu": acc["cpu"], "sib": acc["worst_sibling_busy"], "same_results": same})
    keys = ["pure_reduction_ns", "numerator_part_ns", "masters_ns", "full_ns"]
    med = {k: statistics.median(r["derived"][k] for r in rounds) for k in keys}
    ratio = {k: med[k] / od[k] for k in keys}
    res[g] = {"median_ns": med, "committed_ns": {k: od[k] for k in keys}, "ratio": ratio,
              "rounds": rounds, "committed_cpu": old["timing_cpu"]}
    print(f"{g:24s} " + " ".join(f"{k[:-3]} {med[k]/1e3:.2f}/{od[k]/1e3:.2f}={ratio[k]:.3f}" for k in keys) +
          f" | cpus {[r['cpu'] for r in rounds]} sib {[r['sib'] for r in rounds]} bitwise-same {[r['same_results'] for r in rounds]}")
(S / "analysis/ninja.json").write_text(json.dumps(res, indent=1))
