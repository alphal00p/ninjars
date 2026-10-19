#!/usr/bin/env python3
"""Measure COLLIER 1.2.10 and NinjaRS on the rank n+2 bench cases, interleaved on one CPU.

Every timed process is single-threaded and pinned with `taskset -c CPU`. A round runs, in
this order: NinjaRS fast engine (`ninjars case-bench --widths 1,32`) on every case, then
COLLIER (precise and stock builds, COLI and, where every piece is within its limits, DD) on
every case, so the two codes are interleaved per round. `--rounds` repeats this; collect.py
takes per quantity the best (smallest) of the rounds' block medians. Precise COLI with the
global cache is run in the first round for validation only, untimed: the cache returns wrong
7-point rank-9 results (collier_cache_repro.f90, README.md). The NinjaRS generic
engine (`case-bench --engines generic`, binary64, native masters) is timed once, after the
rounds, on GENERIC_CASES (the slow engine; one round). Before every process the CPU's busy
time is sampled from /proc/stat; the time other processes spent on it during the run
(busy minus the child's own CPU time) is recorded as `foreign_s`.

Outputs (raw, consolidated, committed): OUT/ninjars-fast.json, OUT/ninjars-generic.json,
OUT/collier.json, OUT/environment.json. COLLIER inputs are prepared into BUILD/inputs
(prepare.py; deterministic, not committed).

`--lanes-only` re-runs only COLLIER's untimed validation pass of every variant (unpinned, in
parallel) and replaces the lane values of OUT/collier.json, keeping its timing rounds (and
dropping the timings of variants that are no longer timed); environment.json records it under
`collier_lanes`. Use it after a change of the driver's validation pass.

usage: run_bench.py [--cpu CPU] [--rounds 2] [--blocks 9] [--collier-blocks 5] [--min-block-ms 20]
                    [--cases CASE.json ...] [--skip-generic] [--out DIR] [--lanes-only [--jobs 8]]
"""
import argparse
import concurrent.futures
import datetime
import json
import os
import pathlib
import platform
import re
import resource
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
RN2 = REPO / "validation" / "rank-n2"
SYN = REPO / "validation" / "benchmark" / "synthetic-rank-n2"
BUILD = pathlib.Path(os.environ.get("COLLIER_BENCH_BUILD", REPO / "target" / "benchmark" / "collier-1.2.10"))
NINJARS = REPO / "target" / "perf" / "ninjars"

# rank n+1 twins of rn2-int-complex (the cost table of docs/rank-n2.md section 7.2)
TWINS = [RN2 / "controls" / "cases" / f"rn2-int-complex-n{n}-r{n + 1}.json" for n in range(1, 8)]
# the generic engine (slow) is timed on the synthetic cases and the rn2-int-complex family
GENERIC_CASES = ([SYN / f"syn-n{n}-r{n + 2}.json" for n in range(1, 7)]
                 + [RN2 / "cases" / f"rn2-int-complex-n{n}-r{n + 2}.json" for n in range(1, 8)]
                 + [RN2 / "cases" / f"rn2-int-complex-n{n}-r{n + 2}-mu2.json" for n in range(1, 8)])


# COLLIER variants: (build, mode 1 = COLI / 2 = DD, global cache, timed). The global cache
# is validated but not timed: it returns wrong 7-point rank-9 tensor integrals (README.md).
VARIANTS = [("precise", 1, 0, True), ("precise", 2, 0, True), ("stock", 1, 0, True), ("stock", 2, 0, True),
            ("precise", 1, 1, False)]


def variant_key(build, mode, cache):
    return f"{build}-{'coli' if mode == 1 else 'dd'}{'-cache' if cache else ''}"


def collier_cmd(binary, inp, res, mode, cache, timed, blocks, min_block_ms):
    """collier_bench IN OUT MODE BLOCKS MIN_BLOCK_MS CACHE [REQACC ONLY]; ONLY=none skips timing."""
    cmd = [binary, inp, res, mode, blocks, min_block_ms, cache]
    return cmd if timed else cmd + [-1, "none"]


def default_cases():
    return sorted(SYN.glob("syn-*.json")) + sorted((RN2 / "cases").glob("*.json"))


def cpu_busy(cpu):
    """(busy, total) jiffies of one CPU."""
    for line in open("/proc/stat"):
        f = line.split()
        if f[0] == f"cpu{cpu}":
            v = list(map(int, f[1:]))
            return sum(v) - v[3] - v[4], sum(v)
    raise SystemExit(f"cpu{cpu} not in /proc/stat")


def pick_cpu(seconds=5.0):
    def snap():
        out = {}
        for line in open("/proc/stat"):
            f = line.split()
            if f[0].startswith("cpu") and f[0][3:].isdigit():
                v = list(map(int, f[1:]))
                out[int(f[0][3:])] = (sum(v) - v[3] - v[4], sum(v))
        return out
    a = snap()
    time.sleep(seconds)
    b = snap()
    busy = {c: (b[c][0] - a[c][0]) / max(1, b[c][1] - a[c][1]) for c in a}
    return min(busy, key=lambda c: (busy[c], c)), busy


def run_pinned(cpu, cmd, log):
    """Run cmd pinned to cpu; returns (wall_s, child_cpu_s, foreign_s)."""
    hz = os.sysconf("SC_CLK_TCK")
    b0, _ = cpu_busy(cpu)
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.time()
    with open(log, "a") as fh:
        fh.write("$ " + " ".join(map(str, cmd)) + "\n")
        fh.flush()
        proc = subprocess.run(["taskset", "-c", str(cpu)] + [str(c) for c in cmd], stdout=fh, stderr=fh)
    wall = time.time() - t0
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    b1, _ = cpu_busy(cpu)
    child = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    busy = (b1 - b0) / hz
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}; see {log}")
    return wall, child, max(0.0, busy - child)


def trim_case_bench(report, key):
    """Keep the per-width summaries, statuses and block statistics (no raw block lists)."""
    out = {}
    for case in report["cases"]:
        widths = []
        for w in case.get(key, []):
            entry = {k: w[k] for k in ("width", "failed_lanes", "failure_kinds", "replay_bitwise",
                                       "summary_ns_per_point", "master_rows_per_batch") if k in w}
            if "reference_check" in w:
                entry["reference_check"] = w["reference_check"]
            for q in ("reduction", "reduction_and_tensor_numerator", "reduction_and_numerator", "masters", "total"):
                if q in w:
                    entry[q] = {k: w[q][k] for k in ("median_ns_per_point", "min_ns_per_point", "max_ns_per_point",
                                                     "blocks", "calls_per_block")}
            widths.append(entry)
        out[case["case"]] = {"file": case["file"], "n": case["n"], "rank": case["rank"],
                             "mu_powers": case["mu_powers"], "lanes": case["lanes"], "widths": widths}
    return out


def dump(doc, depth=3):
    """Compact JSON for the committed results: containers are expanded one entry per line
    down to `depth` levels (one line per case), everything deeper is written compactly."""
    def node(value, indent):
        pad = " " * indent
        nested = isinstance(value, (dict, list)) and any(
            isinstance(v, (dict, list)) for v in (value.values() if isinstance(value, dict) else value))
        if indent < depth and nested and value:
            if isinstance(value, dict):
                items = [f"{pad} {json.dumps(k)}: {node(v, indent + 1)}" for k, v in value.items()]
                return "{\n" + ",\n".join(items) + "\n" + pad + "}"
            items = [f"{pad} {node(v, indent + 1)}" for v in value]
            return "[\n" + ",\n".join(items) + "\n" + pad + "]"
        return json.dumps(value, separators=(",", ":"))
    return node(doc, 0) + "\n"


def parse_collier(path):
    lanes, times = [], {}
    for line in pathlib.Path(path).read_text().splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == "LANE":
            v = [float(x) for x in f[4:]]
            lanes.append({"lane": int(f[1]), "errflag": int(f[2]), "accflag": int(f[3]),
                          "eps0": v[0:2], "eps_m1": v[2:4], "eps_m2": v[4:6], "err_estimate": v[6], "absum": v[7]})
        elif f[0] == "TIME":
            times[f[1]] = {"median_ns_per_point": float(f[2]), "min_ns_per_point": float(f[3]),
                           "max_ns_per_point": float(f[4]), "blocks": int(f[5]), "reps": int(f[6])}
    return lanes, times


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu", type=int)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--blocks", type=int, default=9)
    ap.add_argument("--min-block-ms", type=float, default=20.0)
    ap.add_argument("--collier-blocks", type=int, default=5)
    ap.add_argument("--generic-blocks", type=int, default=5)
    ap.add_argument("--cases", nargs="*", type=pathlib.Path)
    ap.add_argument("--skip-generic", action="store_true")
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "results")
    ap.add_argument("--ninjars", type=pathlib.Path, default=NINJARS)
    ap.add_argument("--lanes-only", action="store_true",
                    help="re-run only COLLIER's validation pass and replace the lanes of OUT/collier.json")
    ap.add_argument("--jobs", type=int, default=8, help="parallel validation runs with --lanes-only")
    args = ap.parse_args()
    cases = [p.resolve() for p in (args.cases or default_cases())]
    twins = [p for p in TWINS if p.exists()]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    work = BUILD / "work"
    work.mkdir(parents=True, exist_ok=True)
    log = work / "run_bench.log"
    if not args.lanes_only:
        log.write_text("")
    if args.cpu is None and not args.lanes_only:
        args.cpu, busy = pick_cpu()
        print(f"picked cpu {args.cpu} (busy {busy[args.cpu]:.3f} over 5 s)", flush=True)
    rel = lambda p: str(pathlib.Path(p).resolve().relative_to(REPO))  # noqa: E731

    # COLLIER inputs (exact preprocessing, rounded once to binary64; untimed)
    inputs = BUILD / "inputs"
    subprocess.run([sys.executable, str(HERE / "prepare.py"), *map(str, cases + twins), "--out-dir", str(inputs),
                    "--jobs", "8"], check=True, stdout=subprocess.DEVNULL)
    meta = {json.loads(p.read_text())["case"]: json.loads(p.read_text()) for p in inputs.glob("*.meta.json")}
    names = [json.loads(p.read_text())["name"] for p in cases + twins]

    bins = {v: BUILD / "bin" / f"collier_bench-{v}" for v in ("precise", "stock")}
    collier_meta = {name: {k: meta[name][k] for k in ("case_file", "n", "rank", "mu_powers", "nmax", "rmax",
                                                        "dd_ok")}
                    | {"pieces": [[{k: p[k] for k in ("n", "rank", "coefficients")} for p in lane["pieces"]]
                                  for lane in meta[name]["lanes"]],
                       "lane_ids": [lane["id"] for lane in meta[name]["lanes"]]}
                    for name in names}
    if args.lanes_only:
        lanes_only(args, names, meta, bins, inputs, work, collier_meta)
        return
    fast = {"rounds": []}
    collier = {"version": "1.2.10", "rounds": [], "lanes": {}}
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True).stdout.strip()  # the measured tree (a dirty tree is recorded as such)
    if subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", "crates"]).returncode:
        head += "+dirty"
    env = {"cpu": args.cpu, "date": datetime.date.today().isoformat(), "host": platform.node(),
           "processor": open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": \t"),
           "blocks": args.blocks, "collier_blocks": args.collier_blocks, "generic_blocks": args.generic_blocks,
           "min_block_ms": args.min_block_ms, "rounds": args.rounds, "runs": []}
    for rnd in range(args.rounds):
        print(f"round {rnd + 1}/{args.rounds}", flush=True)
        # NinjaRS fast engine: suite and synthetic cases, then the twins (their own references)
        entry = {}
        for label, files, refs in (("cases", cases, RN2 / "reference"), ("twins", twins, RN2 / "controls" / "reference")):
            raw = work / f"ninjars-fast-{label}.json"
            cmd = [args.ninjars, "case-bench", *files, "--widths", "1,32", "--blocks", args.blocks,
                   "--min-block-ms", args.min_block_ms, "--reference-dir", refs, "-o", raw]
            wall, child, foreign = run_pinned(args.cpu, cmd, log)
            env["runs"].append({"round": rnd, "what": f"ninjars-fast-{label}", "wall_s": wall, "cpu_s": child,
                                "foreign_s": foreign})
            print(f"  ninjars fast {label}: {wall:.0f} s (foreign {foreign:.1f} s)", flush=True)
            entry.update(trim_case_bench(json.loads(raw.read_text()), "per_point_timing"))
        fast["rounds"].append(entry)
        # COLLIER
        entry = {}
        for build, mode, cache, timed in VARIANTS:
            if not timed and rnd > 0:
                continue
            key = variant_key(build, mode, cache)
            t_wall, t_foreign = 0.0, 0.0
            for name in names:
                if mode == 2 and not meta[name]["dd_ok"]:
                    continue
                res = work / f"{name}.{key}.txt"
                cmd = collier_cmd(bins[build], inputs / f"{name}.in", res, mode, cache, timed, args.collier_blocks,
                                  args.min_block_ms)
                wall, child, foreign = run_pinned(args.cpu, cmd, log)
                t_wall += wall
                t_foreign += foreign
                lanes, times = parse_collier(res)
                if timed:
                    entry.setdefault(name, {})[key] = times
                if rnd == 0:
                    collier["lanes"].setdefault(name, {})[key] = lanes
            env["runs"].append({"round": rnd, "what": f"collier-{key}", "wall_s": t_wall, "foreign_s": t_foreign})
            print(f"  collier {key}: {t_wall:.0f} s (foreign {t_foreign:.1f} s)", flush=True)
        collier["rounds"].append(entry)
    collier["meta"] = collier_meta
    (out / "ninjars-fast.json").write_text(dump(fast))
    (out / "collier.json").write_text(dump(collier))
    if not args.skip_generic:
        files = [p for p in GENERIC_CASES if p.exists()]
        raw = work / "ninjars-generic.json"
        cmd = [args.ninjars, "case-bench", *files, "--engines", "generic", "--widths", "1,32",
               "--blocks", args.generic_blocks, "--min-block-ms", args.min_block_ms, "-o", raw]
        wall, child, foreign = run_pinned(args.cpu, cmd, log)
        env["runs"].append({"what": "ninjars-generic", "wall_s": wall, "cpu_s": child, "foreign_s": foreign})
        print(f"ninjars generic: {wall:.0f} s (foreign {foreign:.1f} s)", flush=True)
        (out / "ninjars-generic.json").write_text(
            dump({"rounds": [trim_case_bench(json.loads(raw.read_text()), "generic_per_point_timing")]}))
    env["ninjars_head"] = head
    pin = re.search(r'oneloop = \{[^}]*rev = "([0-9a-f]{7})', (REPO / "Cargo.toml").read_text())
    env["oneloopmaster"] = pin[1] if pin else None
    env["cases"] = [rel(p) for p in cases]
    env["twins"] = [rel(p) for p in twins]
    env["generic_cases"] = [] if args.skip_generic else [rel(p) for p in GENERIC_CASES if p.exists()]
    (out / "environment.json").write_text(dump(env))
    print(f"wrote {out}", flush=True)


def lanes_only(args, names, meta, bins, inputs, work, collier_meta):
    """Re-run COLLIER's validation pass (untimed, unpinned) and replace the lanes of collier.json."""
    path = args.out / "collier.json"
    collier = json.loads(path.read_text())
    jobs = []
    for build, mode, cache, _ in VARIANTS:
        key = variant_key(build, mode, cache)
        for name in names:
            if mode == 2 and not meta[name]["dd_ok"]:
                continue
            res = work / f"{name}.{key}.lanes.txt"
            jobs.append((name, key, res, collier_cmd(bins[build], inputs / f"{name}.in", res, mode, cache, False, 1,
                                                     args.min_block_ms)))

    def run(job):
        name, key, res, cmd = job
        proc = subprocess.run([str(c) for c in cmd], cwd=work, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n{proc.stderr}")
        return name, key, parse_collier(res)[0]

    lanes = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for name, key, rows in pool.map(run, jobs):
            lanes.setdefault(name, {})[key] = rows
    collier["lanes"] = {name: lanes[name] for name in names if name in lanes}
    timed = {variant_key(b, m, c) for b, m, c, t in VARIANTS if t}
    collier["rounds"] = [{name: {k: v for k, v in entry.items() if k in timed} for name, entry in rnd.items()}
                         for rnd in collier["rounds"]]
    collier["meta"] = collier_meta
    path.write_text(dump(collier))
    env_path = args.out / "environment.json"
    env = json.loads(env_path.read_text())
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    # the files that determine the lane values (driver, inputs, build)
    if subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--",
                       *(f"validation/benchmark/collier/{f}" for f in ("collier_bench.f90", "prepare.py", "build.sh")),
                       "validation/rank-n2/harness/collier_reference.py"]).returncode:
        head += "+dirty"
    env["collier_lanes"] = {"date": datetime.date.today().isoformat(), "head": head,
                            "note": "lane values of collier.json from run_bench.py --lanes-only "
                                    "(validation pass only); timing rounds from the run recorded above"}
    env_path.write_text(dump(env))
    print(f"wrote {path}: lanes of {len(collier['lanes'])} cases, {len(jobs)} runs", flush=True)


if __name__ == "__main__":
    main()
