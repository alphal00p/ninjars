#!/usr/bin/env python3
"""Time ORIGINAL Ninja on bench-case files (scalar, one point at a time).

Builds `ninja_reference.cc` like crosscheck_ninja.py and runs its `--time`
mode on every case: pure reduction (replayed TensorNumerator outputs and
OneLOop masters), reduction + live TensorNumerator, and the complete default
evaluation, cycling over the case's points. Pin the process, e.g.

  taskset -c 200 python3 validation/fast/bench_ninja.py cases/*.json

Output: one JSON line per case and phase (nanoseconds per point).
"""

import argparse
import json
import pathlib
import subprocess

import crosscheck_ninja as cc


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", nargs="+", type=pathlib.Path)
    parser.add_argument("--reference-build", type=pathlib.Path,
                        default=cc.ROOT / "target/reference-uncached-v2")
    parser.add_argument("--executable", type=pathlib.Path,
                        default=cc.ROOT / "target/fast-reference/ninja_reference")
    parser.add_argument("--blocks", type=int, default=7)
    parser.add_argument("--min-block-ms", type=float, default=20.0)
    args = parser.parse_args()
    executable = cc.build(args.reference_build.resolve(), args.executable)
    for path in args.cases:
        case = json.loads(path.read_text())
        text = "".join(cc.request(case, lane, 0) for lane in case["lanes"])
        out = subprocess.run([str(executable), "--time", str(args.blocks), str(args.min_block_ms)],
                             input=text, capture_output=True, text=True, check=True)
        for line in out.stdout.splitlines():
            if line.startswith("{"):
                record = json.loads(line)
                record.update({"case": case["name"], "n": case["n"], "rank": case["rank"],
                               "implementation": "original-ninja"})
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
