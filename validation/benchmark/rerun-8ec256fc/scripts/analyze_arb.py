#!/usr/bin/env python3
"""Arbitrary-precision rounds: wall time (accepted attempts) and instructions:u/cycles:u per run."""
import json, pathlib, statistics
S = pathlib.Path("/shared/external/ninja/references/session-2026-10-20b/bench-main")
acc = {}
for l in open(S / "accepted.jsonl"):
    r = json.loads(l)
    acc[r["label"]] = r
recs = [json.loads(l) for l in open(S / "runs.jsonl")]
out = {}
for bits in (700, 1000):
    for side in ("new", "old"):
        walls, instr = [], []
        for r in (1, 2, 3):
            label = f"arb/p{bits}/round{r}"
            a = acc[label]
            tag = f"{label}/{side}/attempt{a['attempt']}"
            rec = [x for x in recs if x["tag"] == tag][0]
            walls.append((rec["wall_s"], rec["cpu"], rec["busy_sibling"]))
            perf = (S / "rounds/arb" / f"{side}-p{bits}-r{r}.perf").read_text().splitlines()
            vals = {line.split(",")[2]: int(line.split(",")[0]) for line in perf if line and not line.startswith("#") and "," in line}
            instr.append(vals)
        out[f"{side}-{bits}"] = {"walls": walls, "instructions": [v["instructions:u"] for v in instr],
                                  "cycles": [v["cycles:u"] for v in instr]}
        print(f"{side} {bits} bits: wall {[w[0] for w in walls]} median {statistics.median(w[0] for w in walls):.2f} s; "
              f"cpus {[w[1] for w in walls]} sib {[w[2] for w in walls]}; instructions:u {[v['instructions:u'] for v in instr]}; "
              f"cycles:u median {statistics.median(v['cycles:u'] for v in instr)/1e9:.2f}e9")
for bits in (700, 1000):
    n, o = out[f"new-{bits}"], out[f"old-{bits}"]
    print(f"{bits}: new/old wall median {statistics.median(w[0] for w in n['walls'])/statistics.median(w[0] for w in o['walls']):.3f}, "
          f"instructions {statistics.median(n['instructions'])/statistics.median(o['instructions']):.4f}, "
          f"cycles {statistics.median(n['cycles'])/statistics.median(o['cycles']):.3f}")
(S / "analysis").mkdir(exist_ok=True)
(S / "analysis/arb.json").write_text(json.dumps(out, indent=1))
