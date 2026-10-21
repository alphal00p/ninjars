#!/usr/bin/env python3
"""Merge the per-case case-bench rounds (orchestrate.py cases1) into one cases.json.

Per case: the accepted rounds' new (aa6a630) and old (d53dd08) runs; medians, new/old ratios; the most
representative new round (least squared log deviation from the per-quantity medians of w32 tensor
numerator, reduction, masters, total and the w1 total) goes into the merged file, in the suite order.
usage: merge_cases.py OUT.json   (also writes SCRATCH/analysis/cases1.json)
"""
import json, math, pathlib, statistics, sys

SCRATCH = pathlib.Path("/shared/external/ninja/references/session-2026-10-20b/bench-main")
ROUNDS = SCRATCH / "rounds/cases1"
sys.path.insert(0, str(SCRATCH / "scripts"))
from orchestrate import CASES  # noqa: E402

KEYS = ("tensor_numerator", "reduction", "masters", "total")


def entry(path, name):
    d = json.loads(path.read_text())
    (c,) = [x for x in d["cases"] if x["case"] == name]
    vals = {}
    for w in (1, 32):
        e = [x for x in c["per_point_timing"] if x["width"] == w][0]
        vals[f"w{w}"] = {k: e["summary_ns_per_point"][k] / 1e3 for k in KEYS}
        vals[f"w{w}"]["check"] = (e.get("reference_check") or {}).get("max_scaled_laurent_difference")
        vals[f"w{w}"]["failed"] = e.get("failed_lanes")
    return d, c, vals


def main():
    out_path = pathlib.Path(sys.argv[1])
    merged_cases, runs, analysis = [], [], {}
    boundary = None
    for case in CASES:
        name = pathlib.Path(case).stem
        rounds = []
        for rdir in sorted((ROUNDS / name).glob("round*")):
            acc = json.loads((rdir / "accepted.json").read_text())
            adir = rdir / f"attempt{acc['attempt']}"
            rec = {"round": rdir.name, "accepted": acc}
            for side in ("new", "old"):
                d, c, vals = entry(adir / f"{side}-cases.json", name)
                rec[side] = {"vals": vals, "case": c, "doc": d}
            rounds.append(rec)
        med = {side: {w: {k: statistics.median(r[side]["vals"][w][k] for r in rounds) for k in KEYS}
                      for w in ("w1", "w32")} for side in ("new", "old")}

        def score(r):
            q = [(r["new"]["vals"]["w32"][k], med["new"]["w32"][k]) for k in KEYS]
            q.append((r["new"]["vals"]["w1"]["total"], med["new"]["w1"]["total"]))
            return sum(math.log(a / b) ** 2 for a, b in q)
        chosen = min(rounds, key=score)
        c = dict(chosen["new"]["case"])
        c["cpu_affinity"] = chosen["new"]["doc"]["cpu_affinity"]
        merged_cases.append(c)
        boundary = chosen["new"]["doc"]["boundary"]
        runs.append({"case": name, "round": chosen["round"], "cpu_affinity": c["cpu_affinity"],
                     "sibling_busy_max": chosen["accepted"]["worst_sibling_busy"]})
        ratio = {w: {k: med["new"][w][k] / med["old"][w][k] for k in KEYS} for w in ("w1", "w32")}
        analysis[name] = {"median": med, "new_over_old": ratio,
                          "rounds": [{"round": r["round"], "accepted": r["accepted"],
                                      "new": r["new"]["vals"], "old": r["old"]["vals"]} for r in rounds],
                          "chosen": chosen["round"]}
        print(f"{name:26s} w32 new/old: tn {ratio['w32']['tensor_numerator']:.3f} red {ratio['w32']['reduction']:.3f} "
              f"mas {ratio['w32']['masters']:.3f} tot {ratio['w32']['total']:.3f} | new red {med['new']['w32']['reduction']:.2f} "
              f"mas {med['new']['w32']['masters']:.2f} tot {med['new']['w32']['total']:.2f} | cpus "
              f"{[r['accepted']['cpu'] for r in rounds]} sib {[r['accepted']['worst_sibling_busy'] for r in rounds]} "
              f"chosen {chosen['round']}")
    template = ["target/perf/ninjars", "case-bench", "<case>", "--widths", "1,32", "--blocks", "9",
                "--min-block-ms", "20", "--reference-dir", "validation/benchmark/ninja",
                "-o", "validation/benchmark/ninjars/cases.json"]
    doc = {"boundary": boundary, "cases": merged_cases, "command": template,
           "command_note": ("one case-bench process per case (<case> = each file of `cases`, in this order), each pinned "
                            "to one CPU whose SMT sibling was at most 5 % busy during the run (socket 0 of this host "
                            "has SMT); per case, the most representative of three rounds interleaved with the d53dd08 "
                            "CLI; `runs` lists the round and CPU"),
           "case_files": CASES, "runs": runs,
           "cpu_affinity": "per case (see runs)", "operation": "case-bench", "schema_version": 1}
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    (SCRATCH / "analysis/cases1.json").write_text(json.dumps(analysis, indent=1))


if __name__ == "__main__":
    main()
