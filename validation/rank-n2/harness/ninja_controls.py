#!/usr/bin/env python3
"""Original-Ninja results for rank <= n+1 control cases, trimmed to the Laurent coefficients
like the phase-1 files in validation/rank-n2/controls/ninja/ (format
"ninjars-reference-ninja-bench-v1").

The driver is validation/benchmark/ninja_bench.cc built against original Ninja @05fb0d3 and
OneLOop 3.6 (validation/benchmark/run_ninja_bench.py, ninja-builds.json, variant o3).  Only
the first evaluation of each lane is used: one round, no timing.  v2 cases (integrand mu^2)
are skipped, because Ninja's tensor interface has no mu^2 blocks.

    ninja_controls.py --driver .../ninja-runs/bin/ninja_bench-o3 CASE.json... --out DIR"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import tempfile

KEYS = ("eps0", "eps_m1", "eps_m2", "cut_constructible", "rational")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="+", type=pathlib.Path)
    ap.add_argument("--driver", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--case-prefix", default="controls/cases/")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(a.driver.read_bytes()).hexdigest()
    for path in a.cases:
        case = json.loads(path.read_text())
        if case.get("format") != "ninjars-bench-case-v1" or case["rank"] > case["n"] + 1:
            print(f"{case['name']}: skipped (v2 or rank > n+1)")
            continue
        with tempfile.TemporaryDirectory(prefix="rn2ninja-") as tmp:
            raw = pathlib.Path(tmp) / "out.json"
            subprocess.run([str(a.driver), str(path), str(raw), "--label", "o3", "--rounds", "1",
                            "--block-ms", "1"], check=True, capture_output=True, text=True)
            data = json.loads(raw.read_text())
        doc = {k: data[k] for k in ("format", "case", "build", "oneloop_warnings", "n", "rank", "amplitude",
                                     "all_replays_bitwise_identical")}
        doc["case_file"] = a.case_prefix + path.name
        doc["provenance"] = ("original Ninja @05fb0d3 + OneLOop 3.6, o3 build, validation/benchmark/ninja_bench.cc "
                             f"driver (sha256 {sha[:16]}...), harness/ninja_controls.py, trimmed to the Laurent "
                             "coefficients")
        doc["lanes"] = [{"id": l["id"], "return_status": l["return_status"], **{k: l[k] for k in KEYS}}
                        for l in data["lanes"]]
        (a.out / path.name).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"{case['name']}: statuses {[l['return_status'] for l in data['lanes']]}")


if __name__ == "__main__":
    main()
