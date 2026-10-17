#!/usr/bin/env python3
"""MadLoop reference pipeline: generation, compilation and per-point timings.

For every graph of the shared phase-space pools (NUMERATOR stream,
validation/benchmark/points/*.json of the main checkout, merged in e6d7fa0 with
the same sha256 as the references/claude_numerator files first used; they
record the MadLoop loop-filter id of each NinjaRS graph):

1. generation (mg5_generate.py, stock MG5_aMC 3.7.0, loop_optimized_output
   True, single filtered loop diagram, Ninja from build 'o3-fortranrules'),
   repeated cold (fresh Python process, fresh output directory) for two exports
   per repetition: 'default' (MadLoop's standalone flags: no -O for anything
   but polynomial.f, i.e. gfortran -O0, polynomial.f -O3) and 'O3'
   (GLOBAL_FLAG=-O3 for Source and the P* directory);
2. compilation: `make` in Source (inside `output standalone` for 'default';
   rebuilt from scratch with GLOBAL_FLAG=-O3 for 'O3') and `make` in the P*
   directory, serial, timed separately;
3. MadLoopParams: MLReductionLib=6, CTModeRun=CTModeInit=1, ImprovePSPoint=-1,
   NRotations_DP=NRotations_QP=0, UseQPIntegrandForNinja=UseQPIntegrandForCutTools=
   .false.; param_card: zero top and Higgs widths (other values = pool model);
4. timing (first repetition, both exports, pinned): bench_madloop.f calling
   ML5_0_SLOOPMATRIXHEL for the pool helicity (index looked up in
   MadLoop5_resources/ML5_0_HelConfigs.dat) over the 3-point pool, mu_R from the
   pool; executables: 'bench' (plain Ninja), 'bench_skip' (loop_matrix.f with
   SKIPLOOPEVAL=.TRUE.: loop-coefficient construction only), 'bench_instr'
   (instrumented Ninja: TensorNumerator / OneLOop / remainder split) and the
   helicity-summed ML5_0_SLOOPMATRIX ('bench', HEL=0);
5. the exact Ninja inputs (NINJA_BENCH_DUMP of the instrumented build) are
   written as ninjars-bench-case-v1 files validation/benchmark/madloop/cases/
   madloop-<graph>.json for the Part-2 replay cross-check (run_ninja_bench.py);
6. checks: the selected loop diagram equals the one of the existing raw
   MadLoop/FeynKit matching evidence (same filter id, vertices, loop/attached
   PDGs), and per point MadLoop's Ninja propagator invariants (S-matrix
   multiset) and masses equal those of the NinjaRS bench case of the graph.

  python3 validation/benchmark/madloop/run_madloop.py [--graphs ...] [--reps 3] [--cpu 29]
"""
import argparse
import datetime
import gzip
import hashlib
import json
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TARGET = ROOT / "target/benchmark/madloop"
MG = ROOT / "references/MG5_aMC_v3_7_0"
# NUMERATOR stream inputs, committed on main (e6d7fa0); the result files recorded
# them from references/claude_numerator with the same sha256.
POOLS = pathlib.Path("/shared/external/ninja/validation/benchmark/points")
NINJARS_CASES = pathlib.Path("/shared/external/ninja/validation/benchmark/cases")
BUILDS = json.loads((HERE.parent / "ninja-builds.json").read_text())
NINJA_LIB = pathlib.Path(BUILDS["variants"]["o3-fortranrules"]["madgraph_ninja_dir"])
NINJA_INSTR_LIB = pathlib.Path(BUILDS["variants"]["o3-fortranrules-instr"]["madgraph_ninja_dir"])
CUTTOOLS = TARGET / "cuttools"

GG_HHH = "g g > h h h QCD=2 QED=3 [virt=QCD]"
GG_HH = "g g > h h QCD=2 QED=2 [virt=QCD]"
SPECS = {
    "hgg-top-triangle-0": dict(process="h > g g QED=1 [virt=QCD]", model="loop_sm", pool="hgg.json",
                               base_filter="n==3 and all(abs(p)==6 for p in loop_pdgs)", filter_id=3,
                               evidence=("graphs.json", "/shared/external/ninja/target/reference/madloop-hgg-validated/graphs.json")),
    "four-photon-top-box-0": dict(process="a a > a a [virt=QED]", model="loop_sm_qed_validation", pool="four-photon.json",
                                  base_filter="n==4 and all(abs(p)==6 for p in loop_pdgs)", filter_id=182,
                                  evidence=("evidence", "validation/evidence/four-photon-madloop.json")),
    "gg-hh-top-triangle-0": dict(process=GG_HH, model="loop_sm", pool="gg-hh.json",
                                 base_filter="n in [3,4] and all(abs(p)==6 for p in loop_pdgs)", filter_id=11,
                                 evidence=("evidence", "validation/evidence/gg-hh-madloop.json")),
    "gg-hh-top-box-2": dict(process=GG_HH, model="loop_sm", pool="gg-hh.json",
                            base_filter="n in [3,4] and all(abs(p)==6 for p in loop_pdgs)", filter_id=10,
                            evidence=("evidence", "validation/evidence/gg-hh-madloop.json")),
    "gg-hhh-top-triangle-0": dict(process=GG_HHH, model="loop_sm", pool="gg-hhh.json",
                                  base_filter="n in [3,4,5] and all(abs(p)==6 for p in loop_pdgs)", filter_id=66,
                                  evidence=("evidence", "validation/evidence/gg-hhh-madloop.json.gz")),
    "gg-hhh-top-box-8": dict(process=GG_HHH, model="loop_sm", pool="gg-hhh.json",
                             base_filter="n in [3,4,5] and all(abs(p)==6 for p in loop_pdgs)", filter_id=58,
                             evidence=("evidence", "validation/evidence/gg-hhh-madloop.json.gz")),
    "gg-hhh-top-pentagon-26": dict(process=GG_HHH, model="loop_sm", pool="gg-hhh.json",
                                   base_filter="n in [3,4,5] and all(abs(p)==6 for p in loop_pdgs)", filter_id=55,
                                   evidence=("evidence", "validation/evidence/gg-hhh-madloop.json.gz")),
}
ML_PARAMS = {"MLReductionLib": "6", "CTModeRun": 1, "CTModeInit": 1, "ImprovePSPoint": -1,
             "NRotations_DP": 0, "NRotations_QP": 0, "UseQPIntegrandForNinja": False,
             "UseQPIntegrandForCutTools": False}
VARIANTS = {"default": "", "O3": "-O3"}
BASH = shutil.which("bash")


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def env():
    e = dict(os.environ)
    e["NIX_HARDENING_ENABLE"] = ""   # exact MadLoop flags (nix would add -O2 for fortify)
    return e


def run(cmd, cwd=None, log=None, env_extra=None, check=True):
    e = env()
    if env_extra:
        e.update(env_extra)
    t0 = time.perf_counter()
    if log:
        with open(log, "a") as handle:
            handle.write("+ " + " ".join(map(str, cmd)) + "\n")
            handle.flush()
            proc = subprocess.run(list(map(str, cmd)), cwd=cwd, env=e, stdout=handle, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.run(list(map(str, cmd)), cwd=cwd, env=e, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n"
                           f"{'' if log else proc.stdout[-3000:] + proc.stderr[-3000:]}")
    return dt, proc


def qed_model():
    """Copy of loop_sm with QED perturbative expansion enabled (as prepare_four_photons.py)."""
    model = TARGET / "models/loop_sm_qed_validation"
    if model.exists():
        return model
    model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(MG / "models/loop_sm", model, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pkl"))
    orders = model / "coupling_orders.py"
    text = orders.read_text()
    assert text.count("perturbative_expansion =0") == 1
    orders.write_text(text.replace("perturbative_expansion =0", "perturbative_expansion =1"))
    return model


def configure(export, pool):
    sys.path.insert(0, str(MG))
    from madgraph.various.banner import MadLoopParam
    from models.check_param_card import ParamCard
    path = export / "Cards/MadLoopParams.dat"
    ml = MadLoopParam(str(path))
    for k, v in ML_PARAMS.items():
        ml[k] = v
    ml.write(str(path))
    card_path = export / "Cards/param_card.dat"
    card = ParamCard(str(card_path))
    widths = {}
    for pdg in (6, 25):
        card["decay"].get((pdg,)).value = 0.0
        widths[pdg] = card["decay"].get((pdg,)).value
    card.write(str(card_path))
    return {"MadLoopParams": {k: str(v) for k, v in ML_PARAMS.items()},
            "widths": widths, "param_card_sha256": sha256(card_path), "MadLoopParams_sha256": sha256(path)}


def set_helicity_filter(export, level):
    sys.path.insert(0, str(MG))
    from madgraph.various.banner import MadLoopParam
    path = export / "Cards/MadLoopParams.dat"
    ml = MadLoopParam(str(path))
    ml["HelicityFilterLevel"] = level
    ml["DoubleCheckHelicityFilter"] = level > 0
    ml.write(str(path))
    for f in (pdir(export) / "MadLoop5_resources").glob("*Filter*"):
        f.unlink()


def pdir(export):
    dirs = sorted((export / "SubProcesses").glob("P*_*"))
    assert len(dirs) == 1, dirs
    return dirs[0]


BENCH_MK = r"""
include makefile
BENCH_NINJA_PLAIN := {plain}
BENCH_NINJA_INSTR := {instr}
bench_madloop.o: {bench_f}
	$(FC) $(FFLAGS) -I. -c $< -o $@ $(LOOP_INCLUDE)
bench_stubs.o: {stubs_c}
	gcc -O2 -c $< -o $@
loop_matrix_skip.f: loop_matrix.f
	sed 's/PARAMETER (SKIPLOOPEVAL=.FALSE.)/PARAMETER (SKIPLOOPEVAL=.TRUE.)/' $< > $@
	grep -q 'PARAMETER (SKIPLOOPEVAL=.TRUE.)' $@
loop_matrix_skip.o: loop_matrix_skip.f polynomial.o
	$(FC) $(FFLAGS) -c $< -o $@ $(LOOP_INCLUDE)
bench: bench_madloop.o bench_stubs.o $(PROCESS) $(LIBS)
	$(FC) $(FFLAGS) -o $@ bench_madloop.o bench_stubs.o $(PROCESS) $(LINKLIBS)
bench_skip: bench_madloop.o bench_stubs.o loop_matrix_skip.o $(PROCESS) $(LIBS)
	$(FC) $(FFLAGS) -o $@ bench_madloop.o bench_stubs.o $(subst loop_matrix.o,loop_matrix_skip.o,$(PROCESS)) $(LINKLIBS)
bench_instr: bench_madloop.o bench_stubs.o $(PROCESS) $(LIBS)
	$(FC) $(FFLAGS) -o $@ bench_madloop.o bench_stubs.o $(PROCESS) $(subst $(BENCH_NINJA_PLAIN),$(BENCH_NINJA_INSTR),$(LINKLIBS))
"""


def build_bench(p, flag, log):
    (p / "bench.mk").write_text(BENCH_MK.format(
        plain=str(NINJA_LIB), instr=str(NINJA_INSTR_LIB),
        bench_f=str(HERE / "bench_madloop.f"), stubs_c=str(HERE / "bench_stubs.c")))
    run(["make", "-f", "bench.mk", f"SHELL={BASH}", f"GLOBAL_FLAG={flag}", "bench", "bench_skip", "bench_instr"],
        cwd=p, log=log)
    # the instrumented executable must contain the strong accessors
    nm = subprocess.run(["nm", str(p / "bench_instr")], capture_output=True, text=True).stdout
    assert re.search(r" T ninja_bench_get_\b", nm), "instrumented accessors missing"
    nm = subprocess.run(["nm", str(p / "bench")], capture_output=True, text=True).stdout
    assert not re.search(r" T ninja_bench_get_\b", nm), "plain bench must not contain instrumentation"
    return {k: sha256(p / k) for k in ("bench", "bench_skip", "bench_instr")}


def hel_index(p, helicities):
    rows = [list(map(int, l.split())) for l in (p / "MadLoop5_resources").glob("*HelConfigs.dat").__next__().read_text().splitlines() if l.strip()]
    idx = [i + 1 for i, r in enumerate(rows) if r == helicities]
    assert len(idx) == 1, (rows, helicities)
    return idx[0], len(rows)


def write_input(p, hel, points, mur, nblocks, block_ms, warm):
    lines = [f"{hel} {len(points)} {nblocks} {block_ms} {warm} {mur!r}"]
    for pt in points:
        for mom in pt["momenta"]:
            lines.append(" ".join(repr(float(x)) for x in mom))
    (p / "bench_input.txt").write_text("\n".join(lines) + "\n")


def parse_bench(text):
    out = {"results": [], "blocks_s": [], "n_rank": []}
    for line in text.splitlines():
        w = line.split()
        if not w:
            continue
        if w[0] == "RESULT":
            out["results"].append([float(x) for x in w[2:]])
        elif w[0] == "BLOCK":
            out["blocks_s"].append(float(w[2]))
        elif w[0] == "CALIBRATION":
            out["sweeps_per_block"] = int(w[1])
        elif w[0] == "EVALUATIONS":
            out["evaluations"] = float(w[1])
        elif w[0] == "TSC_HZ":
            out["tsc_hz"] = float(w[1])
        elif w[0] == "NINJA_COUNTERS":
            out["counters"] = [float(x) for x in w[1:]]
        elif w[0] == "NINJA_N_RANK":
            out["n_rank"].append({"n": int(w[1]), "rank": int(w[2]), "calls": float(w[3])})
        elif w[0] == "CHECKSUM":
            out["checksum"] = float(w[1])
    out["median_s"] = statistics.median(out["blocks_s"]) if out["blocks_s"] else None
    return out


def time_exe(p, exe, cpu, hel, points, mur, nblocks=9, block_ms=25.0, warm=3, env_extra=None):
    write_input(p, hel, points, mur, nblocks, block_ms, warm)
    _, proc = run(["taskset", "-c", str(cpu), f"./{exe}"], cwd=p, env_extra=env_extra)
    data = parse_bench(proc.stdout)
    data["madloop_warnings"] = sum(1 for l in proc.stdout.splitlines() if l.strip().startswith(("##W", "##E")))
    data["load_before"] = pathlib.Path("/proc/loadavg").read_text().split()[:3]
    return data


def dump_case(p, name, cpu, hel, pool, mur):
    dump = p / "ninja_dump.jsonl"
    if dump.exists():
        dump.unlink()
    time_exe(p, "bench_instr", cpu, hel, pool["points"], mur, nblocks=1, block_ms=0.001, warm=3,
             env_extra={"NINJA_BENCH_DUMP": str(dump)})
    calls = [json.loads(l) for l in dump.read_text().splitlines()]
    npts = len(pool["points"])
    # MadLoop's first evaluations run its helicity-filter initialization (several
    # helicities, i.e. several Ninja calls, per point). The steady state used by the
    # timings is one call per evaluation, cycling over the points in order: take the
    # last sweep and check that the two sweeps before it are identical.
    first = calls[-npts:]
    for i, c in enumerate(calls[-3 * npts:-npts]):
        assert c == first[i % npts], "steady-state Ninja calls are not one-per-evaluation in point order"
    lanes = [{"id": f"{name}-madloop-{pt['id']}", "shifts": c["shifts"], "masses_squared": c["masses_squared"],
              "mu_squared": c["mu_squared"], "s_matrix": c["s_matrix"], "tensor": c["tensor"]}
             for pt, c in zip(pool["points"], first)]
    n = {c["n"] for c in first}
    rank = {c["rank"] for c in first}
    assert len(n) == 1 and len(rank) == 1
    case = {"format": "ninjars-bench-case-v1", "name": f"madloop-{name}", "n": n.pop(), "rank": rank.pop(),
            "description": "exact inputs MadLoop (loop_optimized_output, MLReductionLib=6) passes to "
                           "ninja_tensor_evaluate for the pool points, recorded with NINJA_BENCH_DUMP",
            "lanes": lanes}
    path = HERE / "cases" / f"madloop-{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(case) + "\n")
    return path, case


def invariants_check(case, ninjars_case_path):
    """Compare routing-invariant propagator data per lane with the NinjaRS case."""
    ref = json.loads(pathlib.Path(ninjars_case_path).read_text())
    out = []
    for a, b in zip(case["lanes"], ref["lanes"]):
        def inv(lane):
            n = len(lane["shifts"])
            if lane.get("s_matrix"):
                s = [lane["s_matrix"][i][j] for i in range(n) for j in range(i + 1, n)]
            else:
                def d(x, y):
                    v = [x[k] - y[k] for k in range(4)]
                    return v[0] ** 2 - v[1] ** 2 - v[2] ** 2 - v[3] ** 2
                s = [d(lane["shifts"][i], lane["shifts"][j]) for i in range(n) for j in range(i + 1, n)]
            return sorted(s), sorted(tuple(m) for m in lane["masses_squared"])
        sa, ma = inv(a)
        sb, mb = inv(b)
        scale = max(abs(x) for x in sa + sb) or 1.0
        ds = max(abs(x - y) for x, y in zip(sa, sb)) / scale if len(sa) == len(sb) else float("inf")
        dm = max(abs(x[0] - y[0]) + abs(x[1] - y[1]) for x, y in zip(ma, mb)) if len(ma) == len(mb) else float("inf")
        out.append({"lane": a["id"], "ninjars_lane": b["id"], "max_scaled_invariant_difference": ds,
                    "max_mass_squared_difference": dm})
    return {"ninjars_case": str(ninjars_case_path), "ninjars_case_sha256": sha256(ninjars_case_path),
            "ninjars_rank": ref["rank"], "madloop_rank": case["rank"], "n": [case["n"], ref["n"]],
            "lanes": out, "pass": all(x["max_scaled_invariant_difference"] < 1e-12 and x["max_mass_squared_difference"] == 0
                                      for x in out) and case["n"] == ref["n"]}


def diagram_check(selected, spec):
    kind, path = spec["evidence"]
    p = pathlib.Path(path) if path.startswith("/") else ROOT / path
    raw = p.read_bytes()
    if p.suffix == ".gz":
        raw = gzip.decompress(raw)
    data = json.loads(raw)
    if kind == "graphs.json":
        graphs = [g for g in data["graphs"] if g["filter_id"] == spec["filter_id"]]
        ref = graphs[0] if graphs else None
    else:
        graphs = [g for g in data["graphs"] if g["graph_filter_id"] == spec["filter_id"]]
        ref = graphs[0]["graph_metadata"] if graphs else None
    if ref is None:
        return {"evidence": str(p), "pass": False, "reason": "filter id not found in evidence"}

    def key(g):
        return (g["filter_id"], list(g["loop_pdgs"]), list(g["attached_pdgs"]),
                [(v["id"], [(l["id"], l["number"], l["state"], l["loop_line"]) for l in v["legs"]]) for v in g["vertices"]])
    same = key(selected) == key(ref)
    return {"evidence": str(p), "evidence_sha256": sha256(p), "pass": same,
            "compared": "filter id, loop PDGs, attached PDGs, vertex ids and (leg id, number, state, loop_line)"}


def generate(name, spec, dest, record, log):
    model = str(qed_model()) if spec["model"] == "loop_sm_qed_validation" else spec["model"]
    cmd = [sys.executable, HERE / "mg5_generate.py", "--process", spec["process"],
           "--loop-filter", f"{spec['base_filter']} and id=={spec['filter_id']}", "--model", model,
           "--ninja-lib", NINJA_LIB, "--cuttools-dir", CUTTOOLS, "--output", dest, "--record", record]
    wall, _ = run(["taskset", "-c", str(ARGS.cpu), *cmd], log=log)
    data = json.loads(pathlib.Path(record).read_text())
    data["wall_seconds_including_python_startup"] = wall
    pathlib.Path(record).write_text(json.dumps(data, indent=2) + "\n")
    return data


def process_graph(name, args):
    spec = SPECS[name]
    pool_path = POOLS / spec["pool"]
    pool = json.loads(pool_path.read_text())
    mur = float(pool["scale_squared"]) ** 0.5
    base = TARGET / "runs" / name
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    log = base / "log.txt"
    result = {"graph": name, "spec": {k: v for k, v in spec.items() if k != "evidence"},
              "pool": str(pool_path), "pool_sha256": sha256(pool_path),
              "helicities": pool["helicities"], "mu_R": mur, "reps": []}
    for rep in range(1, args.reps + 1):
        entry = {}
        for variant, flag in VARIANTS.items():
            dest = base / f"rep{rep}-{variant}"
            gen = generate(name, spec, dest / "export", dest / "generation.json", log)
            export = dest / "export"
            p = pdir(export)
            cfg = configure(export, pool)
            source_s = gen["timings_seconds"]["source_make_inside_output"]
            if flag:
                # rebuild Source from scratch with the optimization flag
                for f in list((export / "Source/DHELAS").glob("*.o")) + list((export / "Source/MODEL").glob("*.o")):
                    f.unlink()
                for lib in ("libdhelas.a", "libmodel.a"):
                    (export / "lib" / lib).unlink()
                source_s, _ = run(["taskset", "-c", str(args.cpu), "make", f"SHELL={BASH}", f"GLOBAL_FLAG={flag}"],
                                  cwd=export / "Source", log=log)
            # `make check`: MadLoop's Fortran standalone executable (plain `make` would also try
            # to build the f2py python modules). SHELL: the makefile hard-codes /bin/bash,
            # which does not exist on this NixOS host.
            pdir_s, _ = run(["taskset", "-c", str(args.cpu), "make", f"SHELL={BASH}", f"GLOBAL_FLAG={flag}", "check"],
                            cwd=p, log=log)
            entry[variant] = {"generation_seconds": gen["timings_seconds"]["generation_total_in_process"],
                              "generation_wall_seconds": gen["wall_seconds_including_python_startup"],
                              "generation_breakdown": gen["timings_seconds"],
                              "source_make_seconds": source_s, "pdir_make_seconds": pdir_s,
                              "global_flag": flag, "export": str(export), "config": cfg,
                              "selected_loop_diagram": gen["selected_loop_diagrams"][0],
                              "n_loop_groups": gen["n_loop_groups"], "multipliers": gen["multipliers"]}
            print(f"{name} rep{rep} {variant}: gen {entry[variant]['generation_seconds']:.2f}s "
                  f"Source {source_s:.2f}s P {pdir_s:.2f}s", flush=True)
        result["reps"].append(entry)
    result["diagram_check"] = diagram_check(result["reps"][0]["default"]["selected_loop_diagram"], spec)
    # timings on the first repetition
    result["timing"] = {}
    for variant, flag in VARIANTS.items():
        export = pathlib.Path(result["reps"][0][variant]["export"])
        p = pdir(export)
        bins = build_bench(p, flag, log)
        hel, nhel = hel_index(p, pool["helicities"])
        t = {"executables_sha256": bins, "hel_index": hel, "n_helicity_configs": nhel, "runs": []}
        # Single-helicity timings run with MadLoop's helicity filter off
        # (HelicityFilterLevel=0): SLOOPMATRIXHEL then evaluates only the requested
        # helicity, without the filter's initialization/double-check phase, which for
        # an isolated (not gauge-invariant) diagram can fail and re-initialize during
        # the timed blocks (W15/E01). One untimed run first (warm-up of files/caches).
        set_helicity_filter(export, 0)
        time_exe(p, "bench", args.cpu, hel, pool["points"], mur, nblocks=1, block_ms=1.0, warm=3)
        for _ in range(2):
            t["runs"].append({
                "full": time_exe(p, "bench", args.cpu, hel, pool["points"], mur),
                "skip_loop_eval": time_exe(p, "bench_skip", args.cpu, hel, pool["points"], mur),
                "instrumented": time_exe(p, "bench_instr", args.cpu, hel, pool["points"], mur)})
        if variant == "default":
            case_path, case = dump_case(p, name, args.cpu, hel, pool, mur)
            t["ninja_case"] = str(case_path)
            result["ninja_input_check"] = invariants_check(case, NINJARS_CASES / f"{name}.json")
        # Helicity-summed ML5_0_SLOOPMATRIX with MadLoop's default helicity filter
        # (level 2, double check); if its initialization fails for this isolated
        # diagram, fall back to no filter and record it.
        set_helicity_filter(export, 2)
        try:
            time_exe(p, "bench", args.cpu, 0, pool["points"], mur, nblocks=1, block_ms=1.0, warm=6)
            t["helicity_summed"] = time_exe(p, "bench", args.cpu, 0, pool["points"], mur)
            t["helicity_filter_level_summed"] = 2
            ok = t["helicity_summed"]["madloop_warnings"] == 0
        except RuntimeError as error:
            if "Could not initialize the filters" not in str(error):
                raise
            ok = False
        if not ok:
            t["helicity_summed_filter_failure"] = t.pop("helicity_summed", "initialization failed")
            set_helicity_filter(export, 0)
            t["helicity_summed"] = time_exe(p, "bench", args.cpu, 0, pool["points"], mur)
            t["helicity_filter_level_summed"] = 0
        result["timing"][variant] = t
    result["date_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    result["sources_sha256"] = {f: sha256(HERE / f) for f in ("run_madloop.py", "mg5_generate.py", "bench_madloop.f", "bench_stubs.c")}
    (HERE / f"{name}.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def main():
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graphs", nargs="+", default=list(SPECS), choices=list(SPECS))
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--cpu", type=int, default=29)
    ARGS = parser.parse_args()
    for name in ARGS.graphs:
        process_graph(name, ARGS)


if __name__ == "__main__":
    main()
