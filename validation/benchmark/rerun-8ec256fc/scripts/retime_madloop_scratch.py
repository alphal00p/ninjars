#!/usr/bin/env python3
"""Re-run only the per-point MadLoop timings on the existing exports (reproducibility).

  python3 validation/benchmark/madloop/retime_madloop.py [--graphs ...] [--cpu 29] [--out FILE]

For every graph and both exports of repetition 1 (target/benchmark/madloop/runs/
<graph>/rep1-{default,O3}, built by run_madloop.py), repeats exactly the
single-helicity timing sequence of run_madloop.py (HelicityFilterLevel=0, one
untimed warm-up run, then twice: bench, bench_skip, bench_instr; 9 blocks of
>= 25 ms each) with the same executables, computes the split with
summarize.per_point and prints it next to the committed summary.json values.
Raw results go to --out (default target/benchmark/madloop/retime.json). The
MadLoopParams.dat helicity-filter level is restored to the value found.
Nothing under validation/ is modified.
"""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path("/shared/external/ninja/references/ninjars_bench_main/validation/benchmark/madloop")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_madloop_scratch as rm  # noqa: E402
import summarize  # noqa: E402



def read_filter_level(export):
    sys.path.insert(0, str(rm.MG))
    from madgraph.various.banner import MadLoopParam
    return int(MadLoopParam(str(export / "Cards/MadLoopParams.dat"))["HelicityFilterLevel"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", nargs="+", default=summarize.ORDER)
    ap.add_argument("--cpu", type=int, default=29)
    ap.add_argument("--out", type=pathlib.Path, default=rm.TARGET / "retime.json")
    args = ap.parse_args()
    committed = {r["graph"]: r for r in json.loads((HERE / "summary.json").read_text())}
    out = {}
    keys = ["full", "numerator_coefficients", "tensor_numerator", "masters", "ninja_reduction", "madloop_glue"]
    print("graph variant: " + " ".join(f"{k} (new/committed)" for k in keys))
    for name in args.graphs:
        spec = rm.SPECS[name]
        pool = json.loads((rm.POOLS / spec["pool"]).read_text())
        mur = json.loads((HERE / f"{name}.json").read_text())["mu_R"]
        out[name] = {}
        for variant in rm.VARIANTS:
            export = rm.TARGET / "runs" / name / f"rep1-{variant}" / "export"
            p = rm.pdir(export)
            level = read_filter_level(export)
            hel, _ = rm.hel_index(p, pool["helicities"])
            rm.set_helicity_filter(export, 0)
            try:
                rm.time_exe(p, "bench", args.cpu, hel, pool["points"], mur, nblocks=1, block_ms=1.0, warm=3)
                t = {"runs": [], "helicity_summed": {"median_s": float("nan")}}
                for _ in range(2):
                    t["runs"].append({
                        "full": rm.time_exe(p, "bench", args.cpu, hel, pool["points"], mur),
                        "skip_loop_eval": rm.time_exe(p, "bench_skip", args.cpu, hel, pool["points"], mur),
                        "instrumented": rm.time_exe(p, "bench_instr", args.cpu, hel, pool["points"], mur)})
            finally:
                rm.set_helicity_filter(export, level)
            new = summarize.per_point(t)
            old = committed[name]["per_point_us"][variant]
            out[name][variant] = {"per_point_us": new, "raw": t}
            print(f"{name} {variant}: " + " ".join(f"{new[k]:.2f}/{old[k]:.2f}" for k in keys), flush=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
