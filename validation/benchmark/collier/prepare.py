#!/usr/bin/env python3
"""Prepare binary64 COLLIER inputs for collier_bench.f90 from bench cases.

Each lane of a "ninjars-bench-case-v1"/"-v2" case is decomposed into COLLIER pieces
exactly as the rank n+2 reference harness does
(validation/rank-n2/harness/collier_reference.py: lane_sink and sink_to_jobs, with the
lane's S-matrix as invariants): the integrand mu^2 is removed by pinching, scaleless
pieces vanish and each piece is shifted to its reference propagator (p_0 = 0). The
preprocessing is exact rational arithmetic; this script then rounds every mass,
momentum, invariant and numerator coefficient of every piece once to binary64. That is
the input a binary64 COLLIER user would hand over; it is written with repr(), which
gfortran reads back to the identical binary64 number.

Output per case: DIR/<case>.in (the driver's input) and DIR/<case>.meta.json (lane ids,
pieces, whether every piece is within the DD branch's limits).

usage: prepare.py CASE.json [...] --out-dir DIR [--jobs N]
"""
import argparse
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "validation" / "rank-n2" / "harness"))
import collier_reference as cr  # noqa: E402


def f64(q):
    return repr(float(q))


def dd_ok(job):
    n = len(job["props"])
    return n <= 6 and job["rank"] <= cr.DD_MAX_RANK.get(n, 99)


def prepare(path, out_dir):
    case = json.loads(pathlib.Path(path).read_text())
    lines, lanes_meta = [], []
    nmax, rmax = 1, 0
    for index, lane in enumerate(case["lanes"]):
        jobs = cr.sink_to_jobs(cr.lane_sink(case, lane), lane, "smatrix")
        lines.append(f"{len(jobs)} {f64(cr.Q(float(lane['mu_squared'])))}")
        pieces = []
        for job in jobs:
            n = len(job["props"])
            coefs = sorted(job["poly"].items())
            lines.append(f"{n} {job['rank']} {len(coefs)}")
            lines += [f"{f64(m[0])} {f64(m[1])}" for m in job["masses2"]]
            lines += [" ".join(f64(x) for x in p) for p in job["momvec"]]
            lines += [f64(x) for x in job["mominv"]]
            lines += [f"{e[0]} {e[1]} {e[2]} {e[3]} {f64(c[0])} {f64(c[1])}" for e, c in coefs]
            nmax, rmax = max(nmax, n), max(rmax, job["rank"])
            pieces.append({"n": n, "rank": job["rank"], "coefficients": len(coefs), "props": job["props"],
                           "dd_ok": dd_ok(job)})
        lanes_meta.append({"id": lane.get("id", f"lane{index}"), "pieces": pieces,
                           "dd_ok": all(p["dd_ok"] for p in pieces)})
    text = f"{len(case['lanes'])} {nmax} {rmax}\n" + "\n".join(lines) + "\n"
    name = case["name"]
    (out_dir / f"{name}.in").write_text(text)
    meta = {"case": name, "case_file": str(path), "n": case["n"], "rank": case["rank"],
            "mu_powers": case.get("mu_powers", 0), "nmax": nmax, "rmax": rmax, "lanes": lanes_meta,
            "dd_ok": all(l["dd_ok"] for l in lanes_meta)}
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return f"{name:36s} n={case['n']} r={case['rank']} pieces={[len(l['pieces']) for l in lanes_meta]} dd={meta['dd_ok']}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="+", type=pathlib.Path)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    work = [(str(p), args.out_dir) for p in args.cases]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            for line in ex.map(prepare, *zip(*work)):
                print(line, flush=True)
    else:
        for w in work:
            print(prepare(*w), flush=True)


if __name__ == "__main__":
    main()
