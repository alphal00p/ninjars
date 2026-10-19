#!/usr/bin/env python3
"""Run `ninjars case-check` on the rank n+2 suite and synthetic cases; keep per-lane verdicts.

Engines and precisions: the fast engine and the generic engine in binary64 (native
masters), and the generic engine in DoubleFloat, Float(128) and Float(256) (exact
masters), with the tolerance policy of docs/rank-n2.md section 8.3. The runs are
untimed and run in parallel. Output: OUT/accuracy.json with, per run and lane, the
verdict, scaled difference, tolerance, tier and status.

usage: run_accuracy.py [--out DIR] [--ninjars PATH] [--runs f64,double,float:128,float:256]
"""
import argparse
import json
import pathlib
import subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
RN2 = REPO / "validation" / "rank-n2"
SYN = REPO / "validation" / "benchmark" / "synthetic-rank-n2"


def run(ninjars, precision, work):
    engines = "fast,generic" if precision == "f64" else "generic"
    out = work / f"case-check-{precision.replace(':', '-')}.json"
    cases = sorted((RN2 / "cases").glob("*.json")) + sorted(SYN.glob("*.json"))
    cmd = [str(ninjars), "case-check", "--engines", engines, "--precision", precision,
           "--reference-dir", str(RN2 / "reference"), "--known-failures", str(RN2 / "known-failures.json"),
           "--twins", str(RN2 / "summary" / "controls.json"), "--twins", str(RN2 / "summary" / "synthetic-twins.json"),
           "--twins", str(RN2 / "summary" / "n7-controls.json"), "-o", str(out), *map(str, cases)]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    report = json.loads(out.read_text())
    return precision, proc.returncode, report


def trim(report):
    runs = {}
    for case in report["cases"]:
        for r in case["runs"]:
            key = f"{r['engine']}/{r['precision']}"
            lanes = runs.setdefault(key, [])
            for lane in r["lanes"]:
                ref = lane.get("reference") or {}
                lanes.append({"case": case["case"], "n": case["n"], "rank": case["rank"],
                              "mu_powers": case["mu_powers"], "id": lane["id"], "verdict": lane["verdict"],
                              "scaled_difference": lane.get("scaled_difference"), "tolerance": lane.get("tolerance"),
                              "status": lane.get("status"), "tier": ref.get("tier"),
                              "uncertainty": ref.get("uncertainty"), "condition": ref.get("condition"),
                              "new_terms": ref.get("new_terms"), "reason": lane.get("reason")})
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "results")
    ap.add_argument("--ninjars", type=pathlib.Path, default=REPO / "target" / "perf" / "ninjars")
    ap.add_argument("--runs", default="f64,double,float:128,float:256")
    args = ap.parse_args()
    work = REPO / "target" / "benchmark" / "collier-1.2.10" / "work"
    work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    precisions = args.runs.split(",")
    result = {"runs": {}, "exit_status": {}, "summary": {}}
    with ThreadPoolExecutor(len(precisions)) as ex:
        for precision, code, report in ex.map(lambda p: run(args.ninjars, p, work), precisions):
            result["exit_status"][precision] = code
            result["runs"].update(trim(report))
            result["summary"].update(report["summary"])
            print(f"case-check {precision}: exit {code}, {json.dumps(report['summary'])}", flush=True)
    rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    result["ninjars_head"] = rev
    (args.out / "accuracy.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {args.out / 'accuracy.json'}")


if __name__ == "__main__":
    main()
