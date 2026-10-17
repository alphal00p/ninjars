#!/usr/bin/env python3
"""Per-point NinjaRS tensor-numerator preparation timing for gg -> HHH graph 26.

Times ``ninjars_graph::PhysicalTensorNumerator`` through ``ninjars
numerator-bench`` on the shared three-point pool: per phase-space point, raw
binary64 momenta, helicities and a model-parameter row -> HELAS/FeynKit
external wavefunctions -> every SymJIT O2 coefficient head (including the
model couplings) -> the dense rank-4 Ninja tensor (70 coefficients). Propagator
data (shifts, squared masses, MadLoop-thresholded S matrix) is timed
separately and together. Setup is excluded and recorded in
generation-ninjars.json. The same process first checks the prepared tensors
and propagator data against validation/benchmark/cases/gg-hhh-top-pentagon-26.json.

Each process is pinned to one idle CPU; inside it every width is warmed and
then measured in --blocks blocks of at least --min-block-ms, reporting the
median block. The whole process is repeated --repeats times and the median of
the per-process medians is reported. Run from the repository root, without
concurrent builds, after ``{build}``.
"""
import argparse
import json
import pathlib
import tempfile

from benchmark_common import BUILD_COMMAND, ROOT, git, host, median, pick_cpu, run_pinned, sha256

__doc__ = __doc__.format(build=BUILD_COMMAND)
KINDS = ["tensor_coefficients", "propagator_data", "tensor_and_propagator_data"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ninjars", type=pathlib.Path, default=ROOT / "target/perf/ninjars")
    parser.add_argument("--cpu", type=int, help="CPU to pin to (default: an idle allowed CPU)")
    parser.add_argument("--widths", default="1,8,32")
    parser.add_argument("--blocks", type=int, default=15)
    parser.add_argument("--min-block-ms", type=float, default=20.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=pathlib.Path,
                        default=ROOT / "validation/benchmark/numerator-prep-ninjars.json")
    args = parser.parse_args()
    if args.blocks < 7 or args.min_block_ms < 20:
        parser.error("use at least 7 blocks of at least 20 ms")
    cpu = args.cpu if args.cpu is not None else pick_cpu()
    fixture = "examples/graphs/physical/gg-hhh/top-pentagon-26-pool.json"
    common = [args.ninjars.resolve(), "numerator-bench", fixture,
              "--points", "validation/benchmark/points/gg-hhh.json", "--feynman-rules",
              "--check-case", "validation/benchmark/cases/gg-hhh-top-pentagon-26.json",
              "--widths", args.widths, "--blocks", str(args.blocks),
              "--min-block-ms", str(args.min_block_ms)]
    record = {"schema_version": 1, "graph": "gg-hhh top-pentagon-26 (n=5, rank 4, ncoeff 70)",
              "implementation": "ninjars_graph::PhysicalTensorNumerator (SymJIT O2, binary64)",
              "boundary": ("per point: binary64 momenta, helicities and model row -> external "
                           "wavefunctions -> model couplings and all coefficient heads -> dense "
                           "Ninja tensor; warm, allocation-free (crates/ninjars-graph/tests/"
                           "physical_numerator.rs); setup excluded"),
              "build": BUILD_COMMAND, "ninjars_sha256": sha256(args.ninjars),
              "git": git(), "host": host(cpu),
              "script": "python3 validation/benchmark/numerator_prep_ninjars.py " +
                        " ".join(f"--{k.replace('_', '-')} {v}" for k, v in
                                 [("widths", args.widths), ("blocks", args.blocks),
                                  ("min_block_ms", args.min_block_ms), ("repeats", args.repeats)]),
              "variants": []}
    with tempfile.TemporaryDirectory() as work:
        for variant, extra in [("symjit-o2", []), ("symjit-o2-factored", ["--factor"])]:
            runs = []
            for repeat in range(args.repeats):
                output = pathlib.Path(work) / f"{variant}-{repeat}.json"
                command = [*common, *extra, "--output", output]
                run_pinned(cpu, command)
                runs.append(json.loads(output.read_text()))
            first = runs[0]
            for run in runs:
                check = run["check_case"]
                assert check["max_tensor_error_relative_to_lane_max"] < 1e-12, check
            widths = []
            for index, timing in enumerate(first["per_point_timing"]):
                entry = {"width": timing["width"]}
                for kind in KINDS:
                    medians = [run["per_point_timing"][index][kind]["median_ns_per_point"]
                               for run in runs]
                    entry[kind] = {
                        "median_ns_per_point": median(medians),
                        "process_medians_ns_per_point": medians,
                        "calls_per_block": [run["per_point_timing"][index][kind]["calls_per_block"]
                                            for run in runs],
                        "block_ns_per_point": [run["per_point_timing"][index][kind]["block_ns_per_point"]
                                               for run in runs],
                    }
                widths.append(entry)
            record["variants"].append({
                "variant": variant,
                "command": ["taskset", "-c", str(cpu), "ninjars", *map(str, command[1:-1]), "<output>"],
                "coefficient_heads": first["coefficient_heads"],
                "evaluator_inputs": first["evaluator_inputs"],
                "nonzero_tensor_slots": first["nonzero_tensor_slots"],
                "runtime_model_parameters": first["runtime_model_parameters"],
                "check_case": first["check_case"],
                "widths": widths,
            })
            print(variant, {w["width"]: round(w["tensor_coefficients"]["median_ns_per_point"], 1)
                            for w in widths}, flush=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
