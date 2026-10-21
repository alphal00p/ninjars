#!/usr/bin/env python3
"""Analyse the accepted rounds of orchestrate.py.

pipeline: per graph, per side (new = aa6a630 CLI, old = d53dd08 CLI), the median over rounds of the
w32/w1 components; new/old ratios; the new round with the median w32 total (the one to commit).
cases: the same for case-bench (per case, w32).
Writes SCRATCH/analysis/<what>.json and prints a table.
"""
import json, pathlib, statistics, sys

SCRATCH = pathlib.Path("/shared/external/ninja/references/session-2026-10-20b/bench-main")
ROUNDS = SCRATCH / "rounds"
OUT = SCRATCH / "analysis"
OUT.mkdir(exist_ok=True)
KEYS = ["numerator", "numerator_preparation", "tensor_numerator", "reduction", "masters", "total"]


def accepted_dir(label_dir):
    acc = json.loads((label_dir / "accepted.json").read_text())
    return label_dir / f"attempt{acc['attempt']}", acc


def width(report, w):
    for e in report["per_point_timing"]:
        if e["width"] == w:
            return e
    raise KeyError(w)


def pipeline():
    res = {}
    for gdir in sorted((ROUNDS / "pipeline").iterdir()):
        g = gdir.name
        rounds = []
        for rdir in sorted(gdir.glob("round*")):
            adir, acc = accepted_dir(rdir)
            entry = {"round": rdir.name, "accepted": acc}
            for side in ("new", "old"):
                p = adir / f"{side}-pipeline-{g}.json"
                d = json.loads(p.read_text())
                entry[side] = {"path": str(p),
                               "w32": {k: width(d, 32)["summary_ns_per_point"][k] / 1e3 for k in KEYS},
                               "w1": {k: width(d, 1)["summary_ns_per_point"][k] / 1e3 for k in KEYS},
                               "generation_s": d["generation_ns"]["total"] / 1e9,
                               "check": width(d, 32).get("reference_check", {}).get("max_scaled_laurent_difference")}
            rounds.append(entry)
        med = {}
        for side in ("new", "old"):
            med[side] = {w: {k: statistics.median(r[side][w][k] for r in rounds) for k in KEYS} for w in ("w32", "w1")}
            med[side]["generation_s"] = statistics.median(r[side]["generation_s"] for r in rounds)
        ratio = {w: {k: med["new"][w][k] / med["old"][w][k] for k in KEYS} for w in ("w32", "w1")}
        # the most representative new round: least squared log deviation from the per-quantity medians
        import math
        def score(r):
            q = [(r["new"]["w32"][k], med["new"]["w32"][k]) for k in ("numerator", "reduction", "masters", "total")]
            q += [(r["new"]["w1"]["total"], med["new"]["w1"]["total"]), (r["new"]["generation_s"], med["new"]["generation_s"])]
            return sum(math.log(a / b) ** 2 for a, b in q)
        chosen = min(rounds, key=score)
        res[g] = {"rounds": rounds, "median": med, "new_over_old": ratio, "chosen_new": chosen["new"]["path"],
                  "checks": {s: [r[s]["check"] for r in rounds] for s in ("new", "old")}}
        m = med
        print(f"{g:24s} w32 new/old: num {ratio['w32']['numerator']:.3f} red {ratio['w32']['reduction']:.3f} "
              f"mas {ratio['w32']['masters']:.3f} tot {ratio['w32']['total']:.3f} | new tot {m['new']['w32']['total']:.2f} "
              f"old tot {m['old']['w32']['total']:.2f} | spread new tot "
              f"{min(r['new']['w32']['total'] for r in rounds):.2f}-{max(r['new']['w32']['total'] for r in rounds):.2f} | "
              f"cpus {[r['accepted']['cpu'] for r in rounds]} sib {[r['accepted']['worst_sibling_busy'] for r in rounds]}")
    (OUT / "pipeline.json").write_text(json.dumps(res, indent=1))


def cases():
    rounds = []
    for rdir in sorted((ROUNDS / "cases").glob("round*")):
        adir, acc = accepted_dir(rdir)
        entry = {"round": rdir.name, "accepted": acc}
        for side in ("new", "old"):
            p = adir / f"{side}-cases.json"
            d = json.loads(p.read_text())
            entry[side] = {"path": str(p), "cases": {}}
            for c in d["cases"]:
                for w in (1, 32):
                    e = [x for x in c["per_point_timing"] if x["width"] == w][0]
                    s = e["summary_ns_per_point"]
                    entry[side]["cases"].setdefault(c["case"], {})[f"w{w}"] = {
                        k: s[k] / 1e3 for k in ("tensor_numerator", "reduction", "masters", "total")}
                    entry[side]["cases"][c["case"]][f"w{w}"]["check"] = (e.get("reference_check") or {}).get("max_scaled_laurent_difference")
        rounds.append(entry)
    names = list(rounds[0]["new"]["cases"])
    res = {"rounds": rounds, "cases": {}}
    phys = [n for n in names if not n.startswith("syn-")]
    # choose the new round whose geometric mean of physical w32 totals is the median
    import math
    def gm(r):
        return math.exp(sum(math.log(r["new"]["cases"][n]["w32"]["total"]) for n in phys) / len(phys))
    order = sorted(rounds, key=gm)
    res["chosen_new"] = order[len(order) // 2]["new"]["path"]
    for n in names:
        med = {side: {k: statistics.median(r[side]["cases"][n]["w32"][k] for r in rounds)
                      for k in ("tensor_numerator", "reduction", "masters", "total")} for side in ("new", "old")}
        ratio = {k: med["new"][k] / med["old"][k] for k in med["new"]}
        res["cases"][n] = {"median_w32": med, "new_over_old": ratio}
        print(f"{n:26s} w32 new/old: tn {ratio['tensor_numerator']:.3f} red {ratio['reduction']:.3f} "
              f"mas {ratio['masters']:.3f} tot {ratio['total']:.3f} | new red {med['new']['reduction']:.2f} mas {med['new']['masters']:.2f} tot {med['new']['total']:.2f}")
    print("chosen", res["chosen_new"], "cpus", [r["accepted"]["cpu"] for r in rounds], "sib", [r["accepted"]["worst_sibling_busy"] for r in rounds])
    (OUT / "cases.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"pipeline": pipeline, "cases": cases}[sys.argv[1]]()
