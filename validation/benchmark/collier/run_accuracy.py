#!/usr/bin/env python3
"""Run `ninjars case-check` on the rank n+2 suite and synthetic cases; keep per-lane verdicts.

Engines and precisions: the fast engine and the generic engine in binary64 (native
masters), and the generic engine in DoubleFloat, Float(128) and Float(256) (exact
masters), with the tolerance policy of docs/rank-n2.md section 8.3. The runs are
untimed and run in parallel. Output: OUT/accuracy.json with, per run and lane, the
verdict, scaled difference, tolerance, reference tier and status, and the scaled
differences of the lane's value to COLLIER's (`collier`: `coli` and `dd`, COLLIER 1.2.10
binary64 precise build without the global cache, the lane values of collier.json from
run_bench.py's validation pass; `coli_quad`, COLLIER 1.2.9 COLI quad of the reference,
`collier_coli_quad_decimal`), each the largest |x - x_COLLIER| over the three Laurent
coefficients over the reference's scale, in exact decimal arithmetic.

usage: run_accuracy.py [--out DIR] [--ninjars PATH] [--runs f64,double,float:128,float:256]
                       [--collier results/collier.json]
"""
import argparse
import datetime
import json
import math
import pathlib
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, getcontext

getcontext().prec = 120

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
RN2 = REPO / "validation" / "rank-n2"
SYN = REPO / "validation" / "benchmark" / "synthetic-rank-n2"
KEYS = ("eps0", "eps_m1", "eps_m2")


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


def scaled_pair(a, b, scale):
    """Largest |a_k - b_k| over the three Laurent coefficients over scale; a, b: six
    decimal strings or binary64 numbers (taken exactly) each."""
    diff = 0.0
    for j in range(3):
        dr = Decimal(a[2 * j]) - Decimal(b[2 * j])
        di = Decimal(a[2 * j + 1]) - Decimal(b[2 * j + 1])
        diff = max(diff, math.hypot(float(dr), float(di)))
    return diff / scale


def collier_values(collier, references):
    """{lane id: {"coli"|"dd"|"coli_quad": six values}}: COLLIER 1.2.10 binary64 (precise
    build, uncached) from collier.json and COLLIER 1.2.9 COLI quad from the references."""
    out = {}
    for case, meta in collier["meta"].items():
        for variant, label in (("precise-coli", "coli"), ("precise-dd", "dd")):
            for lane_id, lane in zip(meta["lane_ids"], collier["lanes"].get(case, {}).get(variant) or []):
                out.setdefault(lane_id, {})[label] = [x for k in KEYS for x in lane[k]]
    for path in references.glob("*.json"):
        for lane in json.loads(path.read_text())["lanes"]:
            q = lane.get("collier_coli_quad_decimal")
            if q:
                out.setdefault(lane["id"], {})["coli_quad"] = [x for k in KEYS for x in q[k]]
    return out


def trim(report, collier):
    runs = {}
    for case in report["cases"]:
        for r in case["runs"]:
            key = f"{r['engine']}/{r['precision']}"
            lanes = runs.setdefault(key, [])
            for lane in r["lanes"]:
                ref = lane.get("reference") or {}
                row = {"case": case["case"], "n": case["n"], "id": lane["id"], "verdict": lane["verdict"],
                       "scaled_difference": lane.get("scaled_difference"), "tolerance": lane.get("tolerance"),
                       "status": lane.get("status"), "tier": ref.get("tier")}
                values = lane.get("values")
                if values and lane.get("status") == "Success" and ref.get("scale"):
                    v = [x for k in KEYS for x in values[k]]
                    row["collier"] = {label: scaled_pair(v, c, ref["scale"])
                                      for label, c in collier.get(lane["id"], {}).items()}
                lanes.append(row)
    return runs


def dump(result):
    """One lane per line (the file is committed)."""
    lines = ["{"]
    for key in ("ninjars_head", "date", "oneloopmaster", "symbolica", "exit_status", "summary"):
        lines.append(f" {json.dumps(key)}: {json.dumps(result[key])},")
    lines.append(' "runs": {')
    runs = list(result["runs"].items())
    for i, (key, lanes) in enumerate(runs):
        lines.append(f"  {json.dumps(key)}: [")
        lines += [f"   {json.dumps(lane)}{',' if j + 1 < len(lanes) else ''}" for j, lane in enumerate(lanes)]
        lines.append("  ]" + ("," if i + 1 < len(runs) else ""))
    lines += [" }", "}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "results")
    ap.add_argument("--ninjars", type=pathlib.Path, default=REPO / "target" / "perf" / "ninjars")
    ap.add_argument("--runs", default="f64,double,float:128,float:256")
    ap.add_argument("--collier", type=pathlib.Path, default=HERE / "results" / "collier.json",
                    help="COLLIER's lane values (run_bench.py)")
    args = ap.parse_args()
    collier = collier_values(json.loads(args.collier.read_text()), RN2 / "reference")
    work = REPO / "target" / "benchmark" / "collier-1.2.10" / "work"
    work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    precisions = args.runs.split(",")
    result = {"runs": {}, "exit_status": {}, "summary": {}}
    with ThreadPoolExecutor(len(precisions)) as ex:
        for precision, code, report in ex.map(lambda p: run(args.ninjars, p, work), precisions):
            result["exit_status"][precision] = code
            result["runs"].update(trim(report, collier))
            result["summary"].update(report["summary"])
            print(f"case-check {precision}: exit {code}, {json.dumps(report['summary'])}", flush=True)
    rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    result["ninjars_head"] = rev
    result["date"] = datetime.date.today().isoformat()
    cargo = (REPO / "Cargo.toml").read_text()
    pin = re.search(r'oneloop = \{[^}]*rev = "([0-9a-f]{8})', cargo)
    result["oneloopmaster"] = pin[1] if pin else None
    pin = re.search(r'\nsymbolica = \{ git = [^}]*rev = "([0-9a-f]{8})', cargo)
    result["symbolica"] = pin[1] if pin else None
    (args.out / "accuracy.json").write_text(dump(result))
    print(f"wrote {args.out / 'accuracy.json'}")


if __name__ == "__main__":
    main()
