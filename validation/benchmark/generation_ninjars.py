#!/usr/bin/env python3
"""Generation-time accounting of the NinjaRS tensor-numerator path.

For every benchmark representative graph, ``ninjars numerator-bench
--generation-only --feynman-rules`` builds a ``PhysicalTensorNumerator`` from
the FeynKit DOT topology in a fresh (cold) process pinned to one idle CPU and
reports the wall-clock duration of each setup stage:

  prepare   graph import, model Feynman rules and external-state projector,
            integral family/propagators, routing + tensor contraction, color
            invariants, and expansion into the polynomial in (q0..q3) with
            symbolic external components and model parameters;
  planning  coefficient-head extraction (PolynomialCoefficientPlan), optional
            exact factoring of the heads, and the static tensor scatter;
  compile   Symbolica multi-output evaluator construction and SymJIT O2
            machine-code compilation.

Every graph is run --repeats times (default 3) per variant (plain and with
exact head factoring); per-stage medians are recorded together with the
process wall-clock time (which also includes process start-up, model loading
and the shift/mass evaluators). Generating the DOT graphs themselves (FeynKit
diagram generation, done once ahead of time) is not included. Run from the
repository root, without concurrent builds, after ``{build}``.
"""
import argparse
import json
import pathlib
import tempfile

from benchmark_common import (BUILD_COMMAND, REPRESENTATIVES, ROOT, git, host, median, pick_cpu,
                              run_pinned, sha256)

__doc__ = __doc__.format(build=BUILD_COMMAND)
STAGES = {
    "prepare": ["graph_import", "feynman_rules", "propagators", "contraction", "color_invariants",
                "polynomial"],
    "planning": ["coefficient_planning", "coefficient_factoring", "tensor_plan"],
    "compile": ["evaluator_construction", "symjit_compilation"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ninjars", type=pathlib.Path, default=ROOT / "target/perf/ninjars")
    parser.add_argument("--cpu", type=int, help="CPU to pin to (default: an idle allowed CPU)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=pathlib.Path,
                        default=ROOT / "validation/benchmark/generation-ninjars.json")
    args = parser.parse_args()
    cpu = args.cpu if args.cpu is not None else pick_cpu()
    record = {"schema_version": 1, "units": "seconds (median over cold processes)",
              "build": BUILD_COMMAND, "ninjars_sha256": sha256(args.ninjars), "git": git(),
              "host": host(cpu), "repeats": args.repeats,
              "script": f"python3 validation/benchmark/generation_ninjars.py --repeats {args.repeats}",
              "stages": STAGES, "graphs": []}
    with tempfile.TemporaryDirectory() as work:
        for process, graph in REPRESENTATIVES:
            entry = {"process": process, "graph": graph}
            for variant, extra in [("symjit-o2", []), ("symjit-o2-factored", ["--factor"])]:
                runs, walls = [], []
                for repeat in range(args.repeats):
                    output = pathlib.Path(work) / f"{process}-{graph}-{variant}-{repeat}.json"
                    command = [args.ninjars.resolve(), "numerator-bench",
                               f"examples/graphs/physical/{process}/{graph}-pool.json",
                               "--points", f"validation/benchmark/points/{process}.json",
                               "--feynman-rules", "--generation-only", *extra, "--output", output]
                    walls.append(run_pinned(cpu, command))
                    runs.append(json.loads(output.read_text()))
                stages = {}
                for name in runs[0]["generation_ns"]:
                    stages[name] = median([run["generation_ns"][name] * 1e-9 for run in runs])
                summary = {group: median([sum(run["generation_ns"][s] for s in names) * 1e-9
                                          for run in runs])
                           for group, names in STAGES.items()}
                summary["total"] = median([run["generation_ns"]["total"] * 1e-9 for run in runs])
                summary["process_wall_clock"] = median(walls)
                entry.setdefault("propagators", runs[0]["propagators"])
                entry.setdefault("rank", runs[0]["rank"])
                entry.setdefault("ncoeff", runs[0]["ncoeff"])
                entry[variant] = {
                    "command": ["taskset", "-c", str(cpu), "ninjars", *map(str, command[1:-1]),
                                "<output>"],
                    "coefficient_heads": runs[0]["coefficient_heads"],
                    "summary": summary, "stages": stages,
                    "raw_total": [run["generation_ns"]["total"] * 1e-9 for run in runs],
                    "raw_process_wall_clock": walls,
                }
                print(process, graph, variant, {k: round(v, 3) for k, v in summary.items()},
                      flush=True)
            record["graphs"].append(entry)
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
