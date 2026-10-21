#!/usr/bin/env python3
"""Binary64 accuracy of NinjaRS and COLLIER on the final review's fresh rank n+2 cases.

The final review of the rank n+2 work generated 58 fresh cases (n = 1..7; light-like,
equal, mixed, zero and complex masses; mu^2 numerators) with 60-100 digit references
(`final-review-reference-v1`: `lanes[].laurent_decimal`, `scale`), independent of the
suite in validation/rank-n2. This script evaluates every lane with
- NinjaRS: `ninjars case-check --engines fast,generic --precision f64` (no references;
  the lane values of the report), and the generic engine's lane status with its opt-in
  cancellation test (`--stability-test`);
- NinjaRS's generic engine at higher precision (`--precisions`, default DoubleFloat,
  Float(128) and Float(256), exact masters): the lane status (a `Float` master whose
  tracked precision is below max(53, bits/4) fails its lane, 2a7df0d) and the value;
- COLLIER 1.2.10, precise build, without the global cache: COLI, and DD where every
  piece is within its limits (collier_bench's untimed validation pass);
and writes the scaled differences (largest |x - ref| over the Laurent coefficients
over the reference's scale, in exact decimal arithmetic) with COLLIER's flags
(`coli_flags`, `dd_flags`: error flag, accuracy flag, error estimate over the scale and
sum of the contraction terms' magnitudes over the scale), the NinjaRS statuses, and the
scaled differences between NinjaRS's binary64 values and COLLIER's (`fast_vs_coli`,
`generic_vs_dd`, ...) to OUT/accuracy-final-review.json. Untimed; the NinjaRS and
COLLIER runs are parallel.

usage: run_final_review.py [--dir /shared/external/ninja/references/final-review]
                           [--out DIR] [--jobs 16] [--precisions double,float:128,float:256]
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
    return scaled_pair(values, sum(([ref["laurent_decimal"][k][0], ref["laurent_decimal"][k][1]] for k in KEYS), []),
                       ref["scale"])


def scaled_pair(a, b, scale):
    """Largest |a_k - b_k| over the Laurent coefficients over scale (exact decimals)."""
    diff = 0.0
    for j in range(3):
        dr = Decimal(str(a[2 * j])) - Decimal(str(b[2 * j]))
        di = Decimal(str(a[2 * j + 1])) - Decimal(str(b[2 * j + 1]))
        diff = max(diff, math.hypot(float(dr), float(di)))
    return diff / scale if scale else diff


def lane_values(report):
    """{lane id: {engine/precision: (status, [re0, im0, ...] or None)}} of a case-check report."""
    out = {}
    for case in json.loads(pathlib.Path(report).read_text())["cases"]:
        for run in case["runs"]:
            for lane in run["lanes"]:
                v = lane.get("values")
                ok = v and lane.get("status") in ("Success", None)
                out.setdefault(lane["id"], {})[f"{run['engine']}/{run['precision']}"] = (
                    lane.get("status"), sum(([v[k][0], v[k][1]] for k in KEYS), []) if ok else None)
    return out


def collier_lanes(path):
    lanes = []
    for line in pathlib.Path(path).read_text().splitlines():
        f = line.split()
        if f and f[0] == "LANE":
            lanes.append({"errflag": int(f[2]), "accflag": int(f[3]), "values": f[4:10],
                          "err_estimate": float(f[10]), "absum": float(f[11])})
    return lanes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=pathlib.Path, default=pathlib.Path("/shared/external/ninja/references/final-review"))
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "results")
    ap.add_argument("--ninjars", type=pathlib.Path, default=REPO / "target" / "perf" / "ninjars")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--precisions", default="double,float:128,float:256",
                    help="generic-engine precisions beyond binary64 (comma-separated; empty: none)")
    args = ap.parse_args()
    cases = sorted((args.dir / "cases").glob("*.json"))
    work = BUILD / "work-final-review"
    inputs = work / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(HERE / "prepare.py"), *map(str, cases), "--out-dir", str(inputs),
                    "--jobs", str(args.jobs)], check=True, stdout=subprocess.DEVNULL)

    # NinjaRS (no references; untimed, in parallel): binary64 lane values (fast and
    # generic), the generic engine's opt-in cancellation test (--stability-test) and the
    # generic engine at higher precision
    precisions = [p for p in args.precisions.split(",") if p]
    runs = {"f64": ["--engines", "fast,generic", "--precision", "f64"],
            "f64-stability": ["--engines", "generic", "--precision", "f64", "--stability-test"]}
    runs.update({p: ["--engines", "generic", "--precision", p] for p in precisions})

    def ninjars_run(item):
        label, options = item
        report = work / f"case-check-{label.replace(':', '-')}.json"
        subprocess.run([str(args.ninjars), "case-check", *options, "--no-fail", "-o", str(report),
                        *map(str, cases)], check=True, cwd=REPO, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return label, lane_values(report)

    with ThreadPoolExecutor(len(runs)) as ex:
        reports = dict(ex.map(ninjars_run, runs.items()))
    ninjars = {i: {key.split("/")[0]: values for key, (_, values) in d.items()} for i, d in reports["f64"].items()}
    statuses = {i: {key.split("/")[0]: status for key, (status, _) in d.items()} for i, d in reports["f64"].items()}
    stability = {i: d["generic/f64"][0] for i, d in reports["f64-stability"].items()}

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
                row[f"{engine}_status"] = statuses.get(lane["id"], {}).get(engine)
            for p in precisions:
                key = p.replace("float:", "float")  # double, float128, float256 (generic engine)
                status, v = reports[p].get(lane["id"], {}).get(f"generic/{p}", (None, None))
                row[key] = None if v is None else scaled(v, rl)
                row[f"{key}_status"] = status
            for label, data in (("coli", coli), ("dd", dd)):
                if data:
                    c = data[i]
                    row[label] = scaled(c["values"], rl)
                    row[f"{label}_flags"] = [c["errflag"], c["accflag"], c["err_estimate"] / rl["scale"],
                                             c["absum"] / rl["scale"]]
                    for engine in ("fast", "generic"):
                        v = ninjars.get(lane["id"], {}).get(engine)
                        row[f"{engine}_vs_{label}"] = None if v is None else scaled_pair(v, c["values"], rl["scale"])
            lanes.append(row)
    rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    out = {"source": str(args.dir), "ninjars_head": rev, "collier": "1.2.10 precise, no global cache",
           "precisions": ["f64"] + precisions, "lanes": lanes}
    args.out.mkdir(parents=True, exist_ok=True)
    text = "{\n" + "".join(f" {json.dumps(k)}: {json.dumps(v)},\n" for k, v in out.items() if k != "lanes")
    text += ' "lanes": [\n' + ",\n".join(f"  {json.dumps(l)}" for l in lanes) + "\n ]\n}\n"
    (args.out / "accuracy-final-review.json").write_text(text)
    print(f"wrote {args.out / 'accuracy-final-review.json'}: {len(lanes)} lanes")


if __name__ == "__main__":
    main()
