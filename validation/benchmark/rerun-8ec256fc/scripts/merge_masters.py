#!/usr/bin/env python3
"""Merge the per-stream native-masters rounds (orchestrate.py masters1).

Per call stream: accepted rounds of the 8ec256fc (new) and 3023fa6 (old) benches, run alternately on the
same CPU. The most representative new round (least squared log deviation from the per-family medians) of
each stream goes into the merged results file (same layout as one masters-native-bench run over all streams).
Prints per-stream new/old median ratios per family and the IPC (instructions:u / cycles:u of each process).
usage: merge_masters.py OUT.json
"""
import json, math, pathlib, statistics, sys

SCRATCH = pathlib.Path("/shared/external/ninja/references/session-2026-10-20b/bench-main")
ROUNDS = SCRATCH / "rounds/masters1"
OLDRES = pathlib.Path("/shared/external/ninja/references/ninjars_bench_main/validation/benchmark/masters_native/results/olm-3023fa6-sym-e679e8c1-tensor.json")
FAMS = ("all_in_call_order", "boxes", "triangles", "bubbles", "tadpoles")


def perf(path):
    vals = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith("#") and "," in line:
            f = line.split(",")
            try:
                vals[f[2]] = int(f[0])
            except ValueError:
                pass
    return vals


def main():
    out = pathlib.Path(sys.argv[1])
    order = [c["case"] for c in json.loads(OLDRES.read_text())["cases"]]
    top, cases, analysis = None, [], {}
    for name in order:
        rounds = []
        for rdir in sorted((ROUNDS / name).glob("round*")):
            acc = json.loads((rdir / "accepted.json").read_text())
            adir = rdir / f"attempt{acc['attempt']}"
            rec = {"round": rdir.name, "accepted": acc}
            for side in ("new", "old"):
                d = json.loads((adir / f"{side}.json").read_text())
                (c,) = d["cases"]
                p = perf(adir / f"{side}.perf")
                rec[side] = {"doc": d, "case": c, "t": {k: v["median_ns"] for k, v in c["timing"].items()},
                             "ipc": p.get("instructions:u", 0) / max(1, p.get("cycles:u", 1)),
                             "max_diff": max(c["max_scaled_difference_vs_oneloop"].values())}
            rounds.append(rec)
        fams = [f for f in FAMS if f in rounds[0]["new"]["t"]]
        med = {side: {f: statistics.median(r[side]["t"][f] for r in rounds) for f in fams} for side in ("new", "old")}

        def score(r):
            return sum(math.log(r["new"]["t"][f] / med["new"][f]) ** 2 for f in fams)
        chosen = min(rounds, key=score)
        top = chosen["new"]["doc"]
        cases.append(chosen["new"]["case"])
        ratio = {f: med["new"][f] / med["old"][f] for f in fams}
        ipc = {side: statistics.median(r[side]["ipc"] for r in rounds) for side in ("new", "old")}
        analysis[name] = {"median_ns": med, "new_over_old": ratio, "ipc": ipc, "chosen": chosen["round"],
                          "cpus": [r["accepted"]["cpu"] for r in rounds],
                          "sibling": [r["accepted"]["worst_sibling_busy"] for r in rounds],
                          "max_diff": {s: chosen[s]["max_diff"] for s in ("new", "old")}}
        print(f"{name:32s} " + " ".join(f"{f[:5]} {ratio[f]:.3f}" for f in fams) +
              f" | ipc new {ipc['new']:.3f} old {ipc['old']:.3f} | maxdiff new {chosen['new']['max_diff']:.1e} "
              f"old {chosen['old']['max_diff']:.1e} | cpus {analysis[name]['cpus']} sib {analysis[name]['sibling']}")
    doc = dict(top)
    doc["cases"] = cases
    doc["runs_note"] = ("one masters-native-bench process per call stream, each pinned to one CPU whose SMT sibling was "
                        "at most 5 % busy; per stream the most representative of three rounds run alternately with "
                        "olm-3023fa6-sym-e679e8c1-tensor on the same CPU")
    doc["runs"] = [{"case": n, "round": analysis[n]["chosen"], "cpus": analysis[n]["cpus"]} for n in order]
    out.write_text(json.dumps(doc, indent=2) + "\n")
    (SCRATCH / "analysis/masters1.json").write_text(json.dumps(analysis, indent=1))


if __name__ == "__main__":
    main()
