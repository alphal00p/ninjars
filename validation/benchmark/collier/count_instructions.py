#!/usr/bin/env python3
"""Retired user-space instructions per phase-space point, NinjaRS fast engine and COLLIER.

Wall-clock times on the shared host drift with the core clock (1.8 to 3.1 GHz); retired
instructions do not. Each code runs a fixed amount of work twice under
`perf stat -e instructions:u,cycles:u`, and the difference of the two counts divided by
the difference in evaluated points (chosen from a pilot run so that the difference holds
about --target instructions) removes the start-up (input parsing, COLLIER's
Init_cll, Symbolica and oneloopmaster initialisation, recording passes):
- NinjaRS: `ninjars case-bench CASE --widths W --total-calls K` (the live fast-engine
  total, K batches of W lanes), K = K1 and K2;
- COLLIER: `collier_bench IN OUT MODE B -R 0 0 total_finite` (R repetitions per lane and
  block, B blocks, only the finite-part total), B = 1 and 2.
perf comes from `nix shell nixpkgs#perf` when not on PATH.

usage: count_instructions.py [CASE.json ...] [--out DIR] [--cpu CPU]
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
RN2 = REPO / "validation" / "rank-n2"
SYN = REPO / "validation" / "benchmark" / "synthetic-rank-n2"
BUILD = REPO / "target" / "benchmark" / "collier-1.2.10"
DEFAULT = ([SYN / f"syn-n{n}-r{n + 2}.json" for n in range(1, 7)]
           + [RN2 / "cases" / "rn2-int-complex-n7-r9.json"])


def perf(cmd, cpu):
    base = ["perf"] if shutil.which("perf") else ["nix", "shell", "nixpkgs#perf", "-c", "perf"]
    pin = ["taskset", "-c", str(cpu)] if cpu is not None else []
    proc = subprocess.run(base + ["stat", "-x,", "-e", "instructions:u,cycles:u", "--"] + pin + [str(c) for c in cmd],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"failed: {' '.join(map(str, cmd))}\n{proc.stderr[-2000:]}")
    counts = {}
    for line in proc.stderr.splitlines():
        f = line.split(",")
        if len(f) > 2 and f[2].startswith(("instructions", "cycles")):
            counts[f[2].split(":")[0]] = float(f[0])
    return counts


def per_point(a, b, points):
    return {k: (b[k] - a[k]) / points for k in a}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "results")
    ap.add_argument("--cpu", type=int)
    ap.add_argument("--target", type=float, default=2e9,
                    help="instructions in the difference of the two runs (sets the work)")
    ap.add_argument("--ninjars", type=pathlib.Path, default=REPO / "target" / "perf" / "ninjars")
    args = ap.parse_args()
    cases = args.cases or DEFAULT
    inputs = BUILD / "inputs-instructions"  # not run_bench.py's directory, which may be in use
    subprocess.run([sys.executable, str(HERE / "prepare.py"), *map(str, cases), "--out-dir", str(inputs)],
                   check=True, stdout=subprocess.DEVNULL)
    rows = {}
    with tempfile.TemporaryDirectory(prefix="collier-instr-") as tmp:
        for path in cases:
            name = json.loads(path.read_text())["name"]
            lanes = len(json.loads(path.read_text())["lanes"])
            row = {}
            for width in (1, 32):
                def run(k):
                    return perf([args.ninjars, "case-bench", path, "--widths", width, "--total-calls", k], args.cpu)
                a, b = run(1), run(2)  # pilot: one batch more
                k = max(1, round(args.target / max(b["instructions"] - a["instructions"], 1.0)))
                b = run(1 + k)
                row[f"ninjars_fast_w{width}"] = per_point(a, b, k * width) | {"points": k * width}
            meta = json.loads((inputs / f"{name}.meta.json").read_text())
            for variant, mode in (("precise", 1), ("precise", 2), ("stock", 1)):
                if mode == 2 and not meta["dd_ok"]:
                    continue
                exe = BUILD / "bin" / f"collier_bench-{variant}"
                out = pathlib.Path(tmp) / "out.txt"

                def run(blocks, reps):
                    return perf([exe, inputs / f"{name}.in", out, mode, blocks, -reps, 0, 0, "total_finite"], args.cpu)
                a, b = run(1, 1), run(2, 1)  # pilot: one repetition of every lane more
                reps = max(1, round(args.target / max(b["instructions"] - a["instructions"], 1.0)))
                a, b = run(1, reps), run(2, reps)
                row[f"collier_{variant}_{'coli' if mode == 1 else 'dd'}"] = (
                    per_point(a, b, reps * lanes) | {"points": reps * lanes})
            rows[name] = row
            print(name, {k: round(v["instructions"]) for k, v in row.items()}, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "instructions.json").write_text(json.dumps(
        {"method": "perf stat instructions:u,cycles:u; difference of two fixed-work runs per point",
         "cases": rows}, indent=1) + "\n")
    print(f"wrote {args.out / 'instructions.json'}")


if __name__ == "__main__":
    main()
