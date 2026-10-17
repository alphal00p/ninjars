#!/usr/bin/env python3
"""Tabulate the MadLoop reference results (validation/benchmark/madloop/<graph>.json).

Writes validation/benchmark/madloop/SUMMARY.md and summary.json. Per-point times
are microseconds per phase-space point for ONE helicity configuration
(ML5_0_SLOOPMATRIXHEL), median of all timed blocks of both timing runs.
Split (instrumented Ninja, time-stamp counters, same executable otherwise):
  numerator_coefficients = bench_skip (loop_matrix.f with SKIPLOOPEVAL=.TRUE.:
                           external wavefunctions + loop-coefficient construction)
  ninja_total            = inside ninja_tensor_evaluate (incl. TensorNumerator/Amplitude setup)
  tensor_numerator       = TensorNumerator callbacks inside Ninja
  masters                = AvHOneLoop/OneLOop master-integral calls
  ninja_reduction        = ninja_total - tensor_numerator - masters
  madloop_glue           = full(instr) - numerator_coefficients - ninja_total
                           (MadLoop's NINJA_LOOP interface, loop_matrix bookkeeping)
"""
import json
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parent
NINJA = HERE.parent / "ninja"
ORDER = ["hgg-top-triangle-0", "four-photon-top-box-0", "gg-hh-top-triangle-0", "gg-hh-top-box-2",
         "gg-hhh-top-triangle-0", "gg-hhh-top-box-8", "gg-hhh-top-pentagon-26"]


def med(xs):
    return statistics.median(xs) if xs else None


def per_point(t):
    blocks = lambda key: [b for r in t["runs"] for b in r[key]["blocks_s"]]
    out = {"full": med(blocks("full")) * 1e6, "numerator_coefficients": med(blocks("skip_loop_eval")) * 1e6,
           "full_instrumented": med(blocks("instrumented")) * 1e6,
           "helicity_summed": t["helicity_summed"]["median_s"] * 1e6}
    tot = {"evaluate": 0.0, "numerator": 0.0, "masters": 0.0, "calls_eval": 0.0, "calls_num": 0.0, "calls_mast": 0.0}
    evals = 0.0
    for r in t["runs"]:
        x = r["instrumented"]
        c, hz, ne = x["counters"], x["tsc_hz"], x["evaluations"]
        tot["evaluate"] += c[0] / hz * 1e6
        tot["numerator"] += c[1] / hz * 1e6
        tot["masters"] += c[2] / hz * 1e6
        tot["calls_eval"] += c[4]
        tot["calls_num"] += c[5]
        tot["calls_mast"] += c[6]
        evals += ne
    out["ninja_total"] = tot["evaluate"] / evals
    out["tensor_numerator"] = tot["numerator"] / evals
    out["masters"] = tot["masters"] / evals
    out["ninja_reduction"] = out["ninja_total"] - out["tensor_numerator"] - out["masters"]
    out["madloop_glue"] = out["full_instrumented"] - out["numerator_coefficients"] - out["ninja_total"]
    out["ninja_calls_per_point"] = tot["calls_eval"] / evals
    out["tensor_numerator_calls_per_point"] = tot["calls_num"] / evals
    out["master_calls_per_point"] = tot["calls_mast"] / evals
    out["n_rank"] = t["runs"][0]["instrumented"]["n_rank"]
    return out


def main():
    rows = []
    for name in ORDER:
        path = HERE / f"{name}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        gen = [rep[v]["generation_seconds"] for rep in d["reps"] for v in rep]
        gen_wall = [rep[v]["generation_wall_seconds"] for rep in d["reps"] for v in rep]
        comp = {v: {"source": med([rep[v]["source_make_seconds"] for rep in d["reps"]]),
                    "pdir": med([rep[v]["pdir_make_seconds"] for rep in d["reps"]]),
                    "source_all": [rep[v]["source_make_seconds"] for rep in d["reps"]],
                    "pdir_all": [rep[v]["pdir_make_seconds"] for rep in d["reps"]]} for v in d["reps"][0]}
        timing = {v: per_point(t) for v, t in d["timing"].items()}
        replay = None
        rp = NINJA / f"madloop-{name}.json"
        if rp.exists():
            r = json.loads(rp.read_text())
            replay = {v: {**{k: x / 1000 for k, x in r["timing"][v]["derived"].items() if k.endswith("_ns")},
                          "entry_ns": r["timing"][v]["median_ns"].get("full_madloop_entry", 0.0) / 1000}
                      for v in r["timing"]}
        # Same diagram and helicity: original Ninja's finite part of MadLoop's
        # exact inputs over that of the NinjaRS case, per pool point, must be one
        # point-independent constant (MadLoop's colour/normalisation factor).
        amplitude_ratio = None
        rs = NINJA / f"{name}.json"
        if rp.exists() and rs.exists():
            a, b = json.loads(rp.read_text())["lanes"], json.loads(rs.read_text())["lanes"]
            ratios = [complex(*x["eps0"]) / complex(*y["eps0"]) for x, y in zip(a, b)]
            mags = [abs(z) for z in ratios]
            amplitude_ratio = {"abs_eps0_madloop_over_ninjars": mags,
                               "max_relative_spread": (max(mags) - min(mags)) / max(mags),
                               "pass": len(a) == len(b) and (max(mags) - min(mags)) / max(mags) < 1e-9}
        rows.append({"graph": name, "filter_id": d["spec"]["filter_id"],
                     "helicity": d["helicities"], "hel_index": d["timing"]["default"]["hel_index"],
                     "n_helicity_configs": d["timing"]["default"]["n_helicity_configs"],
                     "madloop_rank": d["ninja_input_check"]["madloop_rank"],
                     "ninjars_rank": d["ninja_input_check"]["ninjars_rank"],
                     "n": d["ninja_input_check"]["n"][0],
                     "diagram_check": d["diagram_check"]["pass"], "ninja_input_check": d["ninja_input_check"]["pass"],
                     "amplitude_ratio_check": amplitude_ratio,
                     "generation_seconds_median": med(gen), "generation_seconds_all": gen,
                     "generation_wall_seconds_median": med(gen_wall),
                     "compile_seconds": comp, "per_point_us": timing, "replay_of_madloop_inputs_us": replay})
    (HERE / "summary.json").write_text(json.dumps(rows, indent=1) + "\n")
    f = lambda x: f"{x:.2f}" if x is not None else "-"
    L = ["# MadLoop reference (MadLoop numerator + original Ninja + OneLOop)", "",
         "Generated by `validation/benchmark/madloop/summarize.py` from the per-graph JSON files",
         "(`run_madloop.py`). See README.md for the commands and the definitions.", "",
         "## Generation and compilation (seconds, median of 3 cold repetitions; generation: 6 samples)", "",
         "| graph | id | gen | gen wall | Source (default) | P dir (default) | Source (-O3) | P dir (-O3) |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        c = r["compile_seconds"]
        L.append(f"| {r['graph']} | {r['filter_id']} | {f(r['generation_seconds_median'])} | {f(r['generation_wall_seconds_median'])} | "
                 f"{f(c['default']['source'])} | {f(c['default']['pdir'])} | {f(c['O3']['source'])} | {f(c['O3']['pdir'])} |")
    for v, label in (("default", "MadLoop default flags (gfortran -O0, polynomial.f -O3)"), ("O3", "GLOBAL_FLAG=-O3")):
        L += ["", f"## Per phase-space point, one helicity, {label} (microseconds)", "",
              "| graph | n | rank ML/RS | full | coef. constr. | Ninja total | TensorNum | masters | Ninja reduction | glue | hel. summed | Ninja calls/pt |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in rows:
            t = r["per_point_us"][v]
            L.append(f"| {r['graph']} | {r['n']} | {r['madloop_rank']}/{r['ninjars_rank']} | {f(t['full'])} | {f(t['numerator_coefficients'])} | "
                     f"{f(t['ninja_total'])} | {f(t['tensor_numerator'])} | {f(t['masters'])} | {f(t['ninja_reduction'])} | "
                     f"{f(t['madloop_glue'])} | {f(t['helicity_summed'])} | {t['ninja_calls_per_point']:.2f} |")
    L += ["", "## MadLoop pipeline in the report's columns", "",
          "Per point, one helicity, microseconds: numerator = coef. constr. + TensorNum (MadLoop's",
          "loop-coefficient construction plus Ninja's evaluations of that tensor at its sampling points);",
          "reduction = Ninja reduction + glue; masters = OneLOop; total = full (uninstrumented).",
          "Generation = gen (in-process) + Source make + P dir make, seconds.", "",
          "| graph | flags | generation+compile | numerator | reduction | masters | total |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        for v in ("default", "O3"):
            t = r["per_point_us"][v]
            c = r["compile_seconds"][v]
            gen_total = r["generation_seconds_median"] + c["source"] + c["pdir"]
            L.append(f"| {r['graph']} | {v} | {f(gen_total)} | {f(t['numerator_coefficients'] + t['tensor_numerator'])} | "
                     f"{f(t['ninja_reduction'] + t['madloop_glue'])} | {f(t['masters'])} | {f(t['full'])} |")
    L += ["", "## Cross-check: Part-2 replay driver on MadLoop's exact Ninja inputs (microseconds)", "",
          "| graph | build | full | entry | pure reduction | numerator part | masters |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        if not r["replay_of_madloop_inputs_us"]:
            continue
        for v, x in r["replay_of_madloop_inputs_us"].items():
            L.append(f"| {r['graph']} | {v} | {f(x['full_ns'])} | {f(x['entry_ns'])} | {f(x['pure_reduction_ns'])} | {f(x['numerator_part_ns'])} | {f(x['masters_ns'])} |")
    L += ["", "Checks:", "",
          "* diagram (vs the raw MadLoop/FeynKit matching evidence): "
          + ", ".join(f"{r['graph']}:{'ok' if r['diagram_check'] else 'FAIL'}" for r in rows),
          "* MadLoop's Ninja inputs vs NinjaRS case (S-matrix and mass multisets per point): "
          + ", ".join(f"{r['graph']}:{'ok' if r['ninja_input_check'] else 'FAIL'}" for r in rows),
          "* |eps0(MadLoop inputs) / eps0(NinjaRS case)| from original Ninja is one constant over the pool",
          "  points (same diagram and helicity; MadLoop's colour/normalisation factor): "
          + ", ".join(f"{r['graph']}:" + (f"{r['amplitude_ratio_check']['abs_eps0_madloop_over_ninjars'][0]:.10g}"
                                          f" (spread {r['amplitude_ratio_check']['max_relative_spread']:.0e})"
                                          if r["amplitude_ratio_check"] and r["amplitude_ratio_check"]["pass"] else "FAIL")
                      for r in rows), ""]
    (HERE / "SUMMARY.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
