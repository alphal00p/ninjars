#!/usr/bin/env python3
"""Run e2e_reduction.py on every lane of the given cases in parallel; one result file per
case (format "ninjars-reduction-run-v1": lanes[].laurent_decimal plus the reduction's
diagnostics) in --out-dir.  E2E_MODE (1 COLI, 2 DD) selects COLLIER's branch for the
scalar C0/D0; E2E_SCALARS=DIR takes them from exact values instead (scalar_cases.py).

    run_reduction.py CASE.json... --out-dir DIR [--jobs N] [--timeout S]"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent


def one(task):
    case_path, lane, timeout = task
    with tempfile.TemporaryDirectory(prefix="rn2red-") as tmp:
        out = pathlib.Path(tmp) / "out.json"
        env = dict(os.environ, E2E_TMP=tmp, E2E_OUT=str(out))
        try:
            proc = subprocess.run([sys.executable, "-I", str(HERE / "e2e_reduction.py"), str(case_path), "-", str(lane)],
                                  env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return case_path, lane, {"error": f"timeout after {timeout} s"}
        if proc.returncode != 0 or not out.exists():
            return case_path, lane, {"error": (proc.stderr or proc.stdout)[-2000:]}
        return case_path, lane, json.loads(out.read_text())["lanes"][0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="+", type=pathlib.Path)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=4 * 3600)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cases = {str(p): json.loads(p.read_text()) for p in a.cases}
    tasks = [(p, i, a.timeout) for p, c in cases.items() for i in range(len(c["lanes"]))]
    tasks.sort(key=lambda t: -(cases[t[0]]["n"] * 100 + cases[t[0]]["rank"]))
    res = {}
    with ThreadPoolExecutor(a.jobs) as ex:
        for p, i, lane in ex.map(one, tasks):
            res.setdefault(p, {})[i] = lane
            c = cases[p]
            if len(res[p]) == len(c["lanes"]):
                lanes = []
                for k, src in enumerate(c["lanes"]):
                    l = dict(res[p][k])
                    l["id"] = src.get("id")
                    lanes.append(l)
                doc = {"format": "ninjars-reduction-run-v1", "case": c["name"], "case_file": p, "n": c["n"],
                       "rank": c["rank"], "scalar_masters": ("exact table " + os.environ["E2E_SCALARS"]) if os.environ.get("E2E_SCALARS") else
                       "COLLIER quad " + ("DD" if os.environ.get("E2E_MODE") == "2" else "COLI"),
                       "lanes": lanes}
                (a.out_dir / f"{c['name']}.json").write_text(json.dumps(doc, indent=1) + "\n")
                bad = [k for k, l in enumerate(lanes) if "error" in l]
                print(f"{c['name']:40s} lanes={len(lanes)}" + (f" ERRORS in lanes {bad}" if bad else ""), flush=True)


if __name__ == "__main__":
    main()
