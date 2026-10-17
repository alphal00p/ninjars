#!/usr/bin/env python3
"""Full-process MadLoop timing: all top-quark loop diagrams of g g > h h h.

Context number for the single-graph comparison (run_madloop.py): the same
MadLoop/Ninja/OneLOop setup, but the loop filter keeps every top-loop diagram
('n in [3,4,5] and all(abs(p)==6 for p in loop_pdgs)', 50 diagrams: 8
triangles, 18 boxes, 24 pentagons), which MadLoop groups by denominator
structure. One generation per flag variant; timings of the helicity-summed
ML5_0_SLOOPMATRIX (default helicity filter) and of the ++ helicity alone, over
the gg-hhh pool. Writes validation/benchmark/madloop/full-gg-hhh-top-loops.json.

  python3 validation/benchmark/madloop/run_madloop_full.py [--cpu 29]
"""
import argparse
import importlib.util
import json
import pathlib
import shutil

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run_madloop", HERE / "run_madloop.py")
rm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu", type=int, default=29)
    args = parser.parse_args()
    rm.ARGS = args
    pool_path = rm.POOLS / "gg-hhh.json"
    pool = json.loads(pool_path.read_text())
    mur = float(pool["scale_squared"]) ** 0.5
    base = rm.TARGET / "runs" / "full-gg-hhh-top-loops"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    log = base / "log.txt"
    result = {"process": rm.GG_HHH, "loop_filter": "n in [3,4,5] and all(abs(p)==6 for p in loop_pdgs)",
              "pool": str(pool_path), "pool_sha256": rm.sha256(pool_path), "mu_R": mur, "variants": {}}
    for variant, flag in rm.VARIANTS.items():
        dest = base / variant
        cmd = [rm.sys.executable, HERE / "mg5_generate.py", "--process", rm.GG_HHH, "--loop-filter",
               result["loop_filter"], "--model", "loop_sm", "--ninja-lib", rm.NINJA_LIB, "--cuttools-dir",
               rm.CUTTOOLS, "--output", dest / "export", "--record", dest / "generation.json", "--allow-many"]
        wall, _ = rm.run(["taskset", "-c", str(args.cpu), *cmd], log=log)
        gen = json.loads((dest / "generation.json").read_text())
        export = dest / "export"
        p = rm.pdir(export)
        rm.configure(export, pool)
        source_s = gen["timings_seconds"]["source_make_inside_output"]
        if flag:
            for f in list((export / "Source/DHELAS").glob("*.o")) + list((export / "Source/MODEL").glob("*.o")):
                f.unlink()
            for lib in ("libdhelas.a", "libmodel.a"):
                (export / "lib" / lib).unlink()
            source_s, _ = rm.run(["taskset", "-c", str(args.cpu), "make", f"SHELL={rm.BASH}", f"GLOBAL_FLAG={flag}"],
                                 cwd=export / "Source", log=log)
        pdir_s, _ = rm.run(["taskset", "-c", str(args.cpu), "make", f"SHELL={rm.BASH}", f"GLOBAL_FLAG={flag}", "check"],
                           cwd=p, log=log)
        bins = rm.build_bench(p, flag, log)
        hel, nhel = rm.hel_index(p, pool["helicities"])
        t = {"generation_seconds": gen["timings_seconds"]["generation_total_in_process"],
             "generation_wall_seconds": wall, "source_make_seconds": source_s, "pdir_make_seconds": pdir_s,
             "n_loop_diagrams": gen["n_loop_diagrams_after_filter"], "n_loop_groups": gen["n_loop_groups"],
             "executables_sha256": bins, "hel_index": hel, "n_helicity_configs": nhel}
        # helicity sum with MadLoop's default filter (level 2): the full set is gauge invariant
        rm.set_helicity_filter(export, 2)
        rm.time_exe(p, "bench", args.cpu, 0, pool["points"], mur, nblocks=1, block_ms=1.0, warm=6)
        t["helicity_summed"] = rm.time_exe(p, "bench", args.cpu, 0, pool["points"], mur)
        t["helicity_summed_instrumented"] = rm.time_exe(p, "bench_instr", args.cpu, 0, pool["points"], mur)
        t["helicity_summed_skip_loop_eval"] = rm.time_exe(p, "bench_skip", args.cpu, 0, pool["points"], mur)
        rm.set_helicity_filter(export, 0)
        rm.time_exe(p, "bench", args.cpu, hel, pool["points"], mur, nblocks=1, block_ms=1.0, warm=3)
        t["one_helicity"] = rm.time_exe(p, "bench", args.cpu, hel, pool["points"], mur)
        t["one_helicity_instrumented"] = rm.time_exe(p, "bench_instr", args.cpu, hel, pool["points"], mur)
        t["one_helicity_skip_loop_eval"] = rm.time_exe(p, "bench_skip", args.cpu, hel, pool["points"], mur)
        result["variants"][variant] = t
        print(variant, {k: (v["median_s"] * 1e6 if isinstance(v, dict) and "median_s" in v else v)
                        for k, v in t.items() if k != "executables_sha256"}, flush=True)
    result["sources_sha256"] = {f: rm.sha256(HERE / f) for f in ("run_madloop_full.py", "run_madloop.py", "mg5_generate.py",
                                                                  "bench_madloop.f", "bench_stubs.c")}
    (HERE / "full-gg-hhh-top-loops.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
