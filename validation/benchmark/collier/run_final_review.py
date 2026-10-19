#!/usr/bin/env python3
"""Binary64 accuracy of NinjaRS and COLLIER on the final review's fresh rank n+2 cases.

The final review of the rank n+2 work generated 58 fresh cases (n = 1..7; light-like,
equal, mixed, zero and complex masses; mu^2 numerators) with 60-100 digit references
(`final-review-reference-v1`: `lanes[].laurent_decimal`, `scale`), independent of the
suite in validation/rank-n2. This script evaluates every lane with
- NinjaRS: `ninjars case-check --engines fast,generic --precision f64` (no references;
  the lane values of the report), and the generic engine's lane status with its opt-in
  cancellation test (`--stability-test`);
- COLLIER 1.2.10, precise build: COLI, and DD where every piece is within its limits
  (collier_bench's untimed validation pass);
and writes the scaled differences (largest |x - ref| over the Laurent coefficients
over the reference's scale, in exact decimal arithmetic) with COLLIER's flags and error
estimates to OUT/accuracy-final-review.json. Untimed; the COLLIER runs are parallel.

usage: run_final_review.py [--dir /shared/external/ninja/references/final-review]
                           [--out DIR] [--jobs 16]
"""
import argparse
import json
import math
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, getcontext

getcontext().prec = 120
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
BUILD = REPO / "target" / "benchmark" / "collier-1.2.10"
KEYS = ("eps0", "eps_m1", "eps_m2")


def scaled(values, ref):
    """values: [re0, im0, re1, im1, re2, im2] as decimal strings or floats."""
    diff = 0.0
    for j, k in enumerate(KEYS):
        dr = Decimal(str(values[2 * j])) - Decimal(ref["laurent_decimal"][k][0])
        di = Decimal(str(values[2 * j + 1])) - Decimal(ref["laurent_decimal"][k][1])
        diff = max(diff, math.hypot(float(dr), float(di)))
    return diff / ref["scale"] if ref["scale"] else diff


def collier_lanes(path):
    lanes = []
    for line in pathlib.Path(path).read_text().splitlines():
        f = line.split()
        if f and f[0] == "LANE":
            lanes.append({"errflag": int(f[2]), "accflag": int(f[3]), "values": f[4:10],
                          "err_estimate": float(f[10])})
    return lanes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=pathlib.Path, default=pathlib.Path("/shared/external/ninja/references/final-review"))
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "results")
    ap.add_argument("--ninjars", type=pathlib.Path, default=REPO / "target" / "perf" / "ninjars")
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    cases = sorted((args.dir / "cases").glob("*.json"))
    work = BUILD / "work-final-review"
    inputs = work / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(HERE / "prepare.py"), *map(str, cases), "--out-dir", str(inputs),
                    "--jobs", str(args.jobs)], check=True, stdout=subprocess.DEVNULL)

    # NinjaRS: lane values without references
    report = work / "case-check-f64.json"
    subprocess.run([str(args.ninjars), "case-check", "--engines", "fast,generic", "--precision", "f64", "--no-fail",
                    "-o", str(report), *map(str, cases)], check=True, cwd=REPO, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    ninjars = {}
    for case in json.loads(report.read_text())["cases"]:
        for run in case["runs"]:
            for lane in run["lanes"]:
                v = lane.get("values")
                if v and lane.get("status") in ("Success", None):
                    ninjars.setdefault(lane["id"], {})[run["engine"]] = sum(([v[k][0], v[k][1]] for k in KEYS), [])

    # the generic engine's opt-in cancellation test (--stability-test): lane statuses
    report = work / "case-check-f64-stability.json"
    subprocess.run([str(args.ninjars), "case-check", "--engines", "generic", "--precision", "f64",
                    "--stability-test", "--no-fail", "-o", str(report), *map(str, cases)], check=True, cwd=REPO,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    stability = {lane["id"]: lane.get("status") for case in json.loads(report.read_text())["cases"]
                 for run in case["runs"] for lane in run["lanes"]}

    # COLLIER: validation pass only (one block of one repetition of the finite part)
    jobs = []
    for path in cases:
        name = json.loads(path.read_text())["name"]
        meta = json.loads((inputs / f"{name}.meta.json").read_text())
        for mode in (1, 2):
            if mode == 2 and not meta["dd_ok"]:
                continue
            jobs.append((name, mode, [str(BUILD / "bin" / "collier_bench-precise"), str(inputs / f"{name}.in"),
                                      str(work / f"{name}.{mode}.txt"), str(mode), "1", "-1", "0", "0",
                                      "total_finite"]))
    with ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(lambda j: subprocess.run(j[2], check=True, cwd=work, stdout=subprocess.DEVNULL), jobs))

    lanes = []
    for path in cases:
        case = json.loads(path.read_text())
        name = case["name"]
        ref = {l["id"]: l for l in json.loads((args.dir / "refs" / f"{name}.json").read_text())["lanes"]}
        coli = collier_lanes(work / f"{name}.1.txt")
        dd = collier_lanes(work / f"{name}.2.txt") if (work / f"{name}.2.txt").exists() else None
        for i, lane in enumerate(case["lanes"]):
            rl = ref.get(lane["id"])
            if rl is None:
                continue
            row = {"case": name, "id": lane["id"], "n": case["n"], "rank": case["rank"],
                   "mu_powers": case.get("mu_powers", 0), "uncertainty": rl.get("uncertainty"),
                   "condition": rl.get("condition")}
            row["generic_stability_test"] = stability.get(lane["id"])
            for engine in ("fast", "generic"):
                v = ninjars.get(lane["id"], {}).get(engine)
                row[engine] = None if v is None else scaled(v, rl)
            for label, data in (("coli", coli), ("dd", dd)):
                if data:
                    c = data[i]
                    row[label] = scaled(c["values"], rl)
                    row[f"{label}_flags"] = [c["errflag"], c["accflag"], c["err_estimate"] / rl["scale"]]
            lanes.append(row)
    rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    out = {"source": str(args.dir), "ninjars_head": rev, "collier": "1.2.10 precise", "lanes": lanes}
    args.out.mkdir(parents=True, exist_ok=True)
    text = "{\n" + "".join(f" {json.dumps(k)}: {json.dumps(v)},\n" for k, v in out.items() if k != "lanes")
    text += ' "lanes": [\n' + ",\n".join(f"  {json.dumps(l)}" for l in lanes) + "\n ]\n}\n"
    (args.out / "accuracy-final-review.json").write_text(text)
    print(f"wrote {args.out / 'accuracy-final-review.json'}: {len(lanes)} lanes")


if __name__ == "__main__":
    main()
