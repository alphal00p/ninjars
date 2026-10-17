#!/usr/bin/env python3
"""Collect the review evidence of the fast engine against original Ninja into
validation/fast/evidence/ninja_reference.json.

Inputs are JSON lines written by the examples
  fast_crosscheck (identical OneLOop masters; one line per case) and
  fast_precision  (256-bit generic-engine reference; one line per lane),
grouped by the caller, e.g.

  summarize_review.py --crosscheck physical-and-edge=xcheck_reference.jsonl \
      --crosscheck main-benchmark=xcheck_main.jsonl --precision precision.jsonl \
      --self-noise extended=noise.json

`--self-noise` takes a JSON object {case: [scaled Laurent difference of
original Ninja between the case and its reversed propagator order, per lane]}.
"""

import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def crosscheck(path):
    out = []
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.startswith("{"):
            continue
        d = json.loads(line)
        out.append({
            "case": d["case"], "n": d["n"], "rank": d["rank"], "lanes": len(d["lanes"]),
            "max_scaled_laurent_error": d["max_scaled_laurent"],
            "max_scaled_split_error": d["max_scaled_split"],
            "unmatched_masters": d["unmatched_masters"],
            "statuses_match": d.get("statuses_match"),
            "batches_bitwise": d["batches_bitwise"],
            "lane_errors": {l["id"]: l["laurent"] for l in d["lanes"]},
            "statuses": {l["id"]: [l["status"], l["ninja_status"]] for l in d["lanes"]},
            "native": d.get("native"),
        })
    return out


def native_summary(groups):
    """oneloopmaster native masters (fast_crosscheck --features native-masters)
    against OneLOop on every recorded call: largest relative difference per
    family/output (null: a failed native row), and the complete fast path
    (fast engine + native masters) against Ninja + OneLOop per case."""
    worst, cases = {}, []
    for group in groups.values():
        for c in group:
            native = c.get("native")
            if not native:
                continue
            for family, error in native["native_vs_oneloop"].items():
                error = float("inf") if error is None else error
                worst[family] = max(worst.get(family, 0.0), error)
            bad = {k: v for k, v in native["native_vs_oneloop"].items() if v is None or v > 1e-10}
            if bad or native["native_failed_lanes"]:
                cases.append({"case": c["case"], "masters_above_1e-10": bad,
                              "failed_lanes": native["native_failed_lanes"],
                              "full_path_max_scaled_error": native["native_full_max_scaled"],
                              "worst_calls": {k: native["native_worst_calls"][k] for k in bad}})
    return {"max_relative_difference": {k: (None if v == float("inf") else v) for k, v in worst.items()},
            "any_failed_row": any(v == float("inf") for v in worst.values()),
            "cases_with_defects": cases}


def precision(path):
    out = []
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.startswith("{"):
            continue
        d = json.loads(line)
        out.append({k: d[k] for k in ["case", "lane", "bits", "scale", "fast_laurent_error",
                                      "ninja_laurent_error", "fast_vs_ninja"]})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crosscheck", action="append", default=[])
    parser.add_argument("--precision", action="append", default=[])
    parser.add_argument("--self-noise", action="append", default=[])
    parser.add_argument("--output", type=pathlib.Path, default=HERE / "evidence/ninja_reference.json")
    args = parser.parse_args()
    groups = {}
    for item in args.crosscheck:
        name, path = item.split("=", 1)
        groups[name] = crosscheck(path)
    noise = {}
    for item in args.self_noise:
        name, path = item.split("=", 1)
        noise[name] = json.loads(pathlib.Path(path).read_text())
    lanes = [p for path in args.precision for p in precision(path)]
    ratios = sorted(p["fast_laurent_error"] / max(p["ninja_laurent_error"], 1e-300) for p in lanes)
    summary = {
        "description": (
            "Fast engine vs original Ninja (o3-fortranrules) with identical OneLOop masters, "
            "errors relative to each lane's largest Ninja modulus (no absolute floor); and both "
            "binary64 codes vs an exact (256-bit generic NinjaRS) reduction with the same masters."),
        "crosscheck": {
            name: {
                "cases": len(cases),
                "lanes": sum(c["lanes"] for c in cases),
                "max_scaled_laurent_error": max(c["max_scaled_laurent_error"] for c in cases),
                "all_statuses_match": all(c["statuses_match"] is not False for c in cases),
                "all_batches_bitwise": all(c["batches_bitwise"] for c in cases),
                "unmatched_masters": sum(c["unmatched_masters"] for c in cases),
                "per_case": [{k: v for k, v in c.items() if k != "native"} for c in cases],
            }
            for name, cases in groups.items()
        },
        "native_masters_vs_oneloop": native_summary(groups),
        "ninja_self_noise_reversed_order": noise,
        "precision": {
            "lanes": len(lanes),
            "fast_over_ninja_error_ratio": {
                "min": ratios[0] if ratios else None,
                "median": ratios[len(ratios) // 2] if ratios else None,
                "max": ratios[-1] if ratios else None,
            },
            "per_lane": lanes,
        },
    }
    args.output.write_text(json.dumps(summary, separators=(",", ":")) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
