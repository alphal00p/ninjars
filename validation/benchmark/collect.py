#!/usr/bin/env python3
"""Collect benchmark measurements into docs/benchmark/data/results.json.

Inputs (all relative to the repository root):

- validation/benchmark/ninjars/pipeline-<graph>.json  NinjaRS end-to-end
  (`ninjars pipeline-bench`) for the physical representatives;
- validation/benchmark/ninjars/cases.json             NinjaRS fast engine on the
  shared bench cases (`ninjars case-bench`), physical and synthetic;
- validation/benchmark/madloop/summary.json           MadLoop pipeline (MadLoop
  numerator + original Ninja + OneLOop), per graph;
- validation/benchmark/ninja/<case>.json              original Ninja + OneLOop on
  the identical bench cases (replay driver, build o3-fortranrules).

Per-point times are in microseconds; NinjaRS uses the batched width given by
--width. The MadLoop pipeline uses its faster build for every quantity
(GLOBAL_FLAG=-O3 for evaluation, default flags for compilation time).
"""

import argparse
import json
import pathlib
import platform

ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCH = ROOT / "validation" / "benchmark"

LABELS = {
    "hgg-top-triangle-0": ("gg → H", "top triangle"),
    "four-photon-top-box-0": ("γγ → γγ", "top box"),
    "gg-hh-top-triangle-0": ("gg → HH", "top triangle"),
    "gg-hh-top-box-2": ("gg → HH", "top box"),
    "gg-hhh-top-triangle-0": ("gg → HHH", "top triangle"),
    "gg-hhh-top-box-8": ("gg → HHH", "top box"),
    "gg-hhh-top-pentagon-26": ("gg → HHH", "top pentagon"),
}


def load(path):
    return json.loads(pathlib.Path(path).read_text())


def width_entry(report, width):
    for entry in report["per_point_timing"]:
        if entry["width"] == width:
            return entry
    raise KeyError(f"width {width} missing")


def us(value):
    return None if value is None else value / 1000.0


def ninja_identical(case, build="o3-fortranrules"):
    path = BENCH / "ninja" / f"{case}.json"
    if not path.exists():
        return None
    timing = load(path)["timing"][build]["median_ns"]
    pure = timing["pure_reduction"]
    tensor = timing["reduction_plus_tensor_numerator"] - pure
    return {
        "numerator": us(tensor),
        "reduction": us(pure),
        "masters": us(timing["masters_only"]),
        "total": us(timing["full"]),
    }


def case_difference(cases, name, width):
    """Largest scaled Laurent difference to original Ninja on identical inputs."""
    if name not in cases:
        return None
    check = width_entry(cases[name], width).get("reference_check") or {}
    return check.get("max_scaled_laurent_difference")


def case_summary(cases, name, width):
    """NinjaRS on the shared bench case (identical inputs to original Ninja)."""
    if name not in cases:
        return None
    summary = width_entry(cases[name], width)["summary_ns_per_point"]
    return {
        "numerator": us(summary["tensor_numerator"]),
        "reduction": us(summary["reduction"]),
        "masters": us(summary["masters"]),
        "total": us(summary["total"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--host", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument(
        "--output", default=str(ROOT / "docs" / "benchmark" / "data" / "results.json")
    )
    args = parser.parse_args()

    madloop = {entry["graph"]: entry for entry in load(BENCH / "madloop" / "summary.json")}
    cases = {c["case"]: c for c in load(BENCH / "ninjars" / "cases.json")["cases"]}

    physical = []
    for graph, (process, diagram) in LABELS.items():
        pipeline = load(BENCH / "ninjars" / f"pipeline-{graph}.json")
        batched = width_entry(pipeline, args.width)["summary_ns_per_point"]
        single = width_entry(pipeline, 1)["summary_ns_per_point"]
        check = width_entry(pipeline, args.width)["reference_check"]
        ml = madloop[graph]
        o3 = ml["per_point_us"]["O3"]
        compile_default = ml["compile_seconds"]["default"]
        physical.append(
            {
                "id": graph,
                "process": process,
                "diagram": diagram,
                "label": f"{process}, {diagram}",
                "n": pipeline["propagators"],
                "rank": pipeline["rank"],
                "madloop_rank": ml["madloop_rank"],
                "coefficient_heads": pipeline["coefficient_heads"],
                "generation_s": {
                    "ninjars": pipeline["generation_ns"]["total"] / 1e9,
                    "ninjars_detail": {
                        k: pipeline["generation_ns"][k] / 1e9
                        for k in ("prepare_total", "planning_total", "compilation_total")
                    },
                    "madloop": ml["generation_seconds_median"]
                    + compile_default["source"]
                    + compile_default["pdir"],
                    "madloop_detail": {
                        "output": ml["generation_seconds_median"],
                        "compile_source": compile_default["source"],
                        "compile_process": compile_default["pdir"],
                    },
                },
                "ninjars_us": {
                    "numerator": us(batched["numerator"]),
                    "numerator_preparation": us(batched["numerator_preparation"]),
                    "tensor_numerator": us(batched["tensor_numerator"]),
                    "reduction": us(batched["reduction"]),
                    "masters": us(batched["masters"]),
                    "total": us(batched["total"]),
                },
                "ninjars_width1_us": {
                    "numerator": us(single["numerator"]),
                    "reduction": us(single["reduction"]),
                    "masters": us(single["masters"]),
                    "total": us(single["total"]),
                },
                # MadLoop's NINJA_LOOP interface (tensor copy, momentum
                # inversion, S-matrix, bookkeeping; slightly negative values
                # are timing noise) is counted with the numerator, where
                # NinjaRS counts its own propagator and S-matrix preparation.
                "madloop_us": {
                    "numerator": o3["numerator_coefficients"]
                    + o3["tensor_numerator"]
                    + max(0.0, o3["madloop_glue"]),
                    "numerator_coefficients": o3["numerator_coefficients"],
                    "ninja_interface": max(0.0, o3["madloop_glue"]),
                    "tensor_numerator": o3["tensor_numerator"],
                    "reduction": o3["ninja_reduction"],
                    "masters": o3["masters"],
                    "total": o3["full"],
                },
                "ninja_identical_us": ninja_identical(graph),
                "ninjars_identical_us": case_summary(cases, graph, args.width),
                # Identical inputs (case-bench) for the table; the end-to-end
                # pipeline with NinjaRS's own numerator separately.
                "max_scaled_difference_vs_ninja": case_difference(cases, graph, args.width),
                "max_scaled_difference_end_to_end": check["max_scaled_laurent_difference"]
                if check
                else None,
            }
        )

    synthetic = []
    for name, case in sorted(cases.items()):
        if not name.startswith("syn-"):
            continue
        entry = width_entry(case, args.width)
        summary = entry["summary_ns_per_point"]
        check = entry.get("reference_check") or {}
        synthetic.append(
            {
                "id": name,
                "n": case["n"],
                "rank": case["rank"],
                "ninjars_us": {
                    "numerator": us(summary["tensor_numerator"]),
                    "reduction": us(summary["reduction"]),
                    "masters": us(summary["masters"]),
                    "total": us(summary["total"]),
                },
                "ninja_us": ninja_identical(name),
                "failed_lanes": entry.get("failed_lanes", 0),
                "max_scaled_difference_vs_ninja": check.get("max_scaled_laurent_difference"),
            }
        )

    results = {
        "status": "measured",
        "date": args.date or "",
        "host": args.host or platform.processor() or platform.machine(),
        "batch_width": args.width,
        "cases": physical,
        "synthetic": synthetic,
    }
    pathlib.Path(args.output).write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {args.output}: {len(physical)} physical, {len(synthetic)} synthetic cases")


if __name__ == "__main__":
    main()
