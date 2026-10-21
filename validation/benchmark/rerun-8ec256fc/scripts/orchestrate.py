#!/usr/bin/env python3
"""Guarded, interleaved timing runs for the bench-main refresh.

Every timed process runs pinned to one CPU of 0-127. Socket 0 has SMT: CPU c shares its core with
c+256, which other jobs use. A run is accepted only if the SMT sibling was busy for at most THR of
the run's wall time (measured from /proc/stat); otherwise the round is repeated (at most TRIES times,
then the attempt with the least sibling load is kept and flagged). Before each attempt the CPU is
chosen among CANDIDATES as the first whose core (CPU and sibling) was idle over a 2 s window.

usage: orchestrate.py pipeline|cases|ninja|masters [--rounds 3] [--thr 0.10]
Records: SCRATCH/runs.jsonl (one line per process), outputs under SCRATCH/rounds/.
"""
import argparse, json, os, shutil, subprocess, sys, time, pathlib

SCRATCH = pathlib.Path("/shared/external/ninja/references/session-2026-10-20b/bench-main")
NEW = pathlib.Path("/shared/external/ninja/references/ninjars_bench_main")
OLD = SCRATCH / "old-d53dd08"
ROUNDS = SCRATCH / "rounds"
LOG = SCRATCH / "runs.jsonl"
CANDIDATES = [25, 9, 93, 70, 53, 4, 45, 11, 84, 67, 17, 37, 43]
GRAPHS = [("four-photon", "top-box-0", "four-photon"), ("gg-hhh", "top-box-8", "gg-hhh"),
          ("gg-hhh", "top-pentagon-26", "gg-hhh"), ("gg-hhh", "top-triangle-0", "gg-hhh"),
          ("gg-hh", "top-box-2", "gg-hh"), ("gg-hh", "top-triangle-0", "gg-hh"),
          ("hgg", "top-triangle-0", "hgg")]
CASES = (["validation/benchmark/cases/%s.json" % g for g in
          ["four-photon-top-box-0", "gg-hhh-top-box-8", "gg-hhh-top-pentagon-26", "gg-hhh-top-triangle-0",
           "gg-hh-top-box-2", "gg-hh-top-triangle-0", "hgg-top-triangle-0"]] +
         ["validation/benchmark/synthetic/%s.json" % s for s in
          ["syn-gghhh-ggHHH-n5-r4", "syn-gghhh-ggHHH-n5-r5", "syn-gghhh-gHgHH-n5-r4", "syn-gghhh-gHgHH-n5-r5",
           "syn-gghhh-nosmat-n5-r4", "syn-n2-r1", "syn-n2-r2", "syn-n2-r3", "syn-n3-r2", "syn-n3-r3", "syn-n3-r4",
           "syn-n4-r3", "syn-n4-r4", "syn-n4-r5", "syn-n5-r4", "syn-n5-r5", "syn-n5-r6", "syn-n6-r5", "syn-n6-r6",
           "syn-n6-r7"]])


def snap(cpus=None):
    d = {}
    for line in open('/proc/stat'):
        if line.startswith('cpu') and line[3].isdigit():
            w = line.split()
            c = int(w[0][3:])
            if cpus is None or c in cpus:
                v = list(map(int, w[1:]))
                d[c] = (v[3] + v[4], sum(v))
    return d


def busy(a, b, c):
    return 1 - (b[c][0] - a[c][0]) / max(1, b[c][1] - a[c][1])


def sibling(c):
    return [int(x) for x in open(f'/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list').read().strip().split(',') if int(x) != c]


def choose_cpu(max_wait=180):
    t0 = time.time()
    best = None
    while time.time() - t0 < max_wait:
        cpus = set(CANDIDATES) | {s for c in CANDIDATES for s in sibling(c)}
        a = snap(cpus); time.sleep(2); b = snap(cpus)
        loads = []
        for c in CANDIDATES:
            m = max([busy(a, b, c)] + [busy(a, b, s) for s in sibling(c)])
            loads.append((m, c))
            if m < 0.03:
                return c, m
        best = min(loads)
    return best[1], best[0]


def run(cmd, cwd, cpu, tag, pin=True):
    """Run cmd (pinned to cpu unless pin is False: the command pins itself); return the record."""
    sib = sibling(cpu)
    cpus = [cpu] + sib
    load0 = open('/proc/loadavg').read().split()[:3]
    a = snap(set(cpus)); t0 = time.time()
    full = (['taskset', '-c', str(cpu)] if pin else []) + cmd
    proc = subprocess.Popen(full, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    delay = None
    out_chunks = []
    import threading
    def reader():
        for chunk in iter(lambda: proc.stdout.read(65536), b''):
            out_chunks.append(chunk)
    th = threading.Thread(target=reader); th.start()
    while proc.poll() is None:
        try:
            # run-queue wait of the pinned process (or of the driver it starts)
            pids = [proc.pid] + [int(p) for p in open(f'/proc/{proc.pid}/task/{proc.pid}/children').read().split()]
            tot = 0
            for p in pids:
                for t in os.listdir(f'/proc/{p}/task'):
                    tot += int(open(f'/proc/{p}/task/{t}/schedstat').read().split()[1])
            delay = max(delay or 0, tot)
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    th.join()
    t1 = time.time(); b = snap(set(cpus))
    rec = {"tag": tag, "cmd": cmd, "cwd": str(cwd), "cpu": cpu, "sibling": sib,
           "load_before": load0, "load_after": open('/proc/loadavg').read().split()[:3],
           "busy_cpu": round(busy(a, b, cpu), 4), "busy_sibling": round(max(busy(a, b, s) for s in sib), 4) if sib else 0.0,
           "runqueue_wait_s": None if delay is None else round(delay / 1e9, 4),
           "wall_s": round(t1 - t0, 3), "exit": proc.returncode,
           "start_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t0))}
    with open(LOG, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps({k: rec[k] for k in ("tag", "cpu", "busy_sibling", "runqueue_wait_s", "wall_s", "exit")}), flush=True)
    return rec, b''.join(out_chunks)


def guarded_round(steps, thr, tries, label):
    """steps: list of (tag, cmd_fn(cpu), cwd, pin, collect_fn(attempt_dir) ). All steps of a round share the CPU."""
    attempts = []
    for attempt in range(1, tries + 1):
        cpu, pre = choose_cpu()
        recs = []
        adir = ROUNDS / label / f"attempt{attempt}"
        adir.mkdir(parents=True, exist_ok=True)
        for tag, cmd_fn, cwd, pin, collect in steps:
            rec, out = run(cmd_fn(cpu), cwd, cpu, f"{label}/{tag}/attempt{attempt}", pin)
            (adir / f"{tag}.stdout").write_bytes(out)
            if rec["exit"] != 0:
                print(out.decode(errors='replace')[-3000:], file=sys.stderr)
                raise SystemExit(f"{label}/{tag} failed")
            collect(adir, tag)
            recs.append(rec)
        worst = max(r["busy_sibling"] for r in recs)
        attempts.append((worst, attempt, recs))
        (adir / "records.json").write_text(json.dumps(recs, indent=1))
        if worst <= thr:
            break
    worst, attempt, recs = min(attempts)
    accepted = {"label": label, "attempt": attempt, "worst_sibling_busy": worst, "accepted": worst <= thr,
                "cpu": recs[0]["cpu"], "attempts": len(attempts)}
    (ROUNDS / label / "accepted.json").write_text(json.dumps(accepted, indent=1))
    with open(SCRATCH / "accepted.jsonl", 'a') as f:
        f.write(json.dumps(accepted) + '\n')
    return accepted


def mover(src_root, names):
    def collect(adir, tag):
        for n in names:
            src = src_root / n
            shutil.move(str(src), str(adir / f"{tag}-{pathlib.Path(n).name}"))
    return collect


def pipeline(args):
    for proc_, diag, pts in GRAPHS:
        g = f"{proc_}-{diag}"
        if args.only and g not in args.only:
            continue
        def cmd(cpu, root_cli):
            return [root_cli, "pipeline-bench", f"examples/graphs/physical/{proc_}/{diag}-pool.json",
                    "--points", f"validation/benchmark/points/{pts}.json", "--feynman-rules",
                    "--widths", "1,8,32,128", "--blocks", "9", "--min-block-ms", "20",
                    "--reference", f"validation/benchmark/ninja/{g}.json",
                    "-o", f"validation/benchmark/ninjars/pipeline-{g}.json"]
        files = [f"validation/benchmark/ninjars/pipeline-{g}.json"]
        for r in range(1, args.rounds + 1):
            new = ("new", lambda cpu: cmd(cpu, "target/perf/ninjars"), NEW, True, mover(NEW, files))
            old = ("old", lambda cpu: cmd(cpu, "target/perf/ninjars"), OLD, True, mover(OLD, files))
            steps = [new, old] if r % 2 == 1 else [old, new]
            guarded_round(steps, args.thr, args.tries, f"pipeline/{g}/round{r}")


def cases(args):
    files = ["validation/benchmark/ninjars/cases.json"]
    def cmd(cpu):
        return ["target/perf/ninjars", "case-bench", *CASES, "--widths", "1,32", "--blocks", "9",
                "--min-block-ms", "20", "--reference-dir", "validation/benchmark/ninja",
                "-o", "validation/benchmark/ninjars/cases.json"]
    for r in range(1, args.rounds + 1):
        new = ("new", cmd, NEW, True, mover(NEW, files))
        old = ("old", cmd, OLD, True, mover(OLD, files))
        steps = [new, old] if r % 2 == 1 else [old, new]
        guarded_round(steps, args.thr, args.tries, f"cases/round{r}")


def cases1(args):
    """case-bench one case per process (short runs, so the sibling guard is effective)."""
    files = ["validation/benchmark/ninjars/cases.json"]
    for case in CASES:
        name = pathlib.Path(case).stem
        if args.only and name not in args.only:
            continue
        def cmd(cpu, case=case):
            return ["target/perf/ninjars", "case-bench", case, "--widths", "1,32", "--blocks", "9",
                    "--min-block-ms", "20", "--reference-dir", "validation/benchmark/ninja",
                    "-o", "validation/benchmark/ninjars/cases.json"]
        for r in range(1, args.rounds + 1):
            new = ("new", cmd, NEW, True, mover(NEW, files))
            old = ("old", cmd, OLD, True, mover(OLD, files))
            steps = [new, old] if r % 2 == 1 else [old, new]
            guarded_round(steps, args.thr, args.tries, f"cases1/{name}/round{r}")


def ninja(args):
    harness = SCRATCH / "harness/run_ninja_bench_scratch.py"
    for g in ["four-photon-top-box-0", "gg-hhh-top-box-8", "gg-hhh-top-pentagon-26", "gg-hhh-top-triangle-0",
              "gg-hh-top-box-2", "gg-hh-top-triangle-0", "hgg-top-triangle-0"] + (args.extra or []):
        if args.only and g not in args.only:
            continue
        sub = "synthetic" if g.startswith("syn-") else "cases"
        for r in range(1, args.rounds + 1):
            label = f"ninja/{g}/round{r}"
            if (ROUNDS / label / "accepted.json").exists():
                continue
            def cmd(cpu, g=g, sub=sub, label=label):
                return ["env", f"REF_OUT={ROUNDS / label / 'out'}", sys.executable, str(harness),
                        str(NEW / f"validation/benchmark/{sub}/{g}.json"), "--variants", "o3-fortranrules",
                        "--cpu", str(cpu), "--rounds", "9", "--block-ms", "25"]
            def collect(adir, tag, label=label):
                out = ROUNDS / label / "out"
                if out.exists():
                    shutil.move(str(out), str(adir / "out"))
            guarded_round([("ref", cmd, NEW, False, collect)], args.thr, args.tries, label)


def masters(args):
    labels = {"new": "olm-8ec256fc-sym-9cdc2949-tensor", "old": "olm-3023fa6-sym-e679e8c1-tensor"}
    calls = sorted(str(p.relative_to(NEW)) for p in (NEW / "validation/benchmark/ninja").glob("*.masters.json"))
    for r in range(1, args.rounds + 1):
        steps = []
        for side in (["new", "old"] if r % 2 == 1 else ["old", "new"]):
            lab = labels[side]
            exe = str(NEW / f"target/benchmark/masters-native/{lab}/target/release/masters-native-bench")
            def cmd(cpu, exe=exe, lab=lab, side=side, r=r):
                out = ROUNDS / "masters" / f"round{r}" / f"{side}-{lab}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                stat = ROUNDS / "masters" / f"round{r}" / f"{side}-{lab}.perf"
                return ["/nix/store/gyp2si1k1w7jhw8z4xx1bwr2m0pr5445-perf-linux-7.2/bin/perf", "stat", "-x,", "-o", str(stat),
                        "-e", "instructions:u,cycles:u", "--", exe, str(out), *calls, "--label", lab]
            def collect(adir, tag, lab=lab, side=side, r=r):
                for ext in (".json", ".perf"):
                    src = ROUNDS / "masters" / f"round{r}" / f"{side}-{lab}{ext}"
                    shutil.move(str(src), str(adir / src.name))
            steps.append((side, cmd, NEW, True, collect))
        guarded_round(steps, args.thr, args.tries, f"masters/round{r}")


def masters1(args):
    """masters-native-bench one call stream per process (short runs for the sibling guard)."""
    labels = {"new": "olm-8ec256fc-sym-9cdc2949-tensor", "old": "olm-3023fa6-sym-e679e8c1-tensor"}
    calls = sorted(str(p.relative_to(NEW)) for p in (NEW / "validation/benchmark/ninja").glob("*.masters.json"))
    for call in calls:
        name = pathlib.Path(call).name[:-len(".masters.json")]
        if args.only and name not in args.only:
            continue
        for r in range(1, args.rounds + 1):
            label = f"masters1/{name}/round{r}"
            if (ROUNDS / label / "accepted.json").exists():
                continue
            steps = []
            for side in (["new", "old"] if r % 2 == 1 else ["old", "new"]):
                lab = labels[side]
                exe = str(NEW / f"target/benchmark/masters-native/{lab}/target/release/masters-native-bench")
                out = ROUNDS / label / f"{side}.json"
                stat = ROUNDS / label / f"{side}.perf"
                def cmd(cpu, exe=exe, out=out, stat=stat, lab=lab, call=call):
                    out.parent.mkdir(parents=True, exist_ok=True)
                    return [PERF, "stat", "-x,", "-o", str(stat), "-e", "instructions:u,cycles:u", "--",
                            exe, str(out), call, "--label", lab]
                def collect(adir, tag, out=out, stat=stat):
                    for f in (out, stat):
                        shutil.move(str(f), str(adir / f.name))
                steps.append((side, cmd, NEW, True, collect))
            guarded_round(steps, args.thr, args.tries, label)


PERF = "/nix/store/gyp2si1k1w7jhw8z4xx1bwr2m0pr5445-perf-linux-7.2/bin/perf"
ARB_CANDIDATES = [36, 33, 39, 60, 57, 78, 75]


def arb(args):
    global CANDIDATES
    CANDIDATES = ARB_CANDIDATES
    fixture = "examples/graphs/physical/gg-hhh/top-pentagon-26-pool.json"
    if not args.skip_data:
        # the data run: run.sh with the new CLI, writes validation/benchmark/arbitrary-precision/*.json
        def cmd(cpu):
            return ["bash", "-c", "time bash validation/benchmark/arbitrary-precision/run.sh target/perf/ninjars"]
        guarded_round([("runsh", cmd, NEW, True, lambda adir, tag: None)], args.thr, args.tries, "arb/data")
    for r in range(1, args.rounds + 1):
        for bits in (700, 1000):
            steps = []
            for side in (["new", "old"] if r % 2 == 1 else ["old", "new"]):
                root = NEW if side == "new" else OLD
                out = ROUNDS / "arb" / f"{side}-p{bits}-r{r}.json"
                stat = ROUNDS / "arb" / f"{side}-p{bits}-r{r}.perf"
                (ROUNDS / "arb").mkdir(parents=True, exist_ok=True)
                def cmd(cpu, out=out, stat=stat, bits=bits):
                    return [PERF, "stat", "-x,", "-o", str(stat), "-e", "instructions:u,cycles:u", "--",
                            "target/perf/ninjars", "reduce", fixture, "--precision", "float", "--bits", str(bits),
                            "-o", str(out)]
                steps.append((side, cmd, root, True, lambda adir, tag: None))
            guarded_round(steps, args.thr, args.tries, f"arb/p{bits}/round{r}")


ML_CANDIDATES = [90, 92, 18, 19, 26, 28]


def madloop(args):
    global CANDIDATES
    CANDIDATES = ML_CANDIDATES
    harness = SCRATCH / "harness/run_madloop_scratch.py"
    outdir = SCRATCH / "refretime/madloop"
    for g in args.only or ["gg-hhh-top-pentagon-26", "gg-hh-top-box-2", "hgg-top-triangle-0"]:
        for r in range(1, args.rounds + 1):
            label = f"madloop/{g}/round{r}"
            def cmd(cpu, g=g):
                return [sys.executable, str(harness), "--graphs", g, "--reps", "1", "--cpu", str(cpu)]
            def collect(adir, tag, g=g):
                for src in (outdir / f"{g}.json", outdir / "cases" / f"madloop-{g}.json"):
                    if src.exists():
                        shutil.move(str(src), str(adir / src.name))
            guarded_round([("ml", cmd, NEW, False, collect)], args.thr, args.tries, label)


def mlretime(args):
    """Re-time only the per-point MadLoop sequence on the exports built by the madloop mode."""
    global CANDIDATES
    CANDIDATES = ML_CANDIDATES
    harness = SCRATCH / "harness/retime_madloop_scratch.py"
    for g in args.only or ["gg-hhh-top-pentagon-26", "gg-hh-top-box-2", "hgg-top-triangle-0"]:
        for r in range(1, args.rounds + 1):
            label = f"mlretime/{g}/round{r}"
            out = ROUNDS / label / "retime.json"
            def cmd(cpu, g=g, out=out):
                out.parent.mkdir(parents=True, exist_ok=True)
                return [sys.executable, str(harness), "--graphs", g, "--cpu", str(cpu), "--out", str(out)]
            def collect(adir, tag, out=out):
                shutil.move(str(out), str(adir / out.name))
            guarded_round([("ml", cmd, NEW, False, collect)], args.thr, args.tries, label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["pipeline", "cases", "cases1", "ninja", "masters", "masters1", "arb", "madloop", "mlretime"])
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--cands", type=int, nargs="*")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--thr", type=float, default=0.10)
    ap.add_argument("--tries", type=int, default=5)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--extra", nargs="*")
    args = ap.parse_args()
    global CANDIDATES, ML_CANDIDATES
    if args.cands:
        CANDIDATES = args.cands
        ML_CANDIDATES = args.cands
    {"pipeline": pipeline, "cases": cases, "cases1": cases1, "ninja": ninja, "masters": masters, "masters1": masters1, "arb": arb, "madloop": madloop, "mlretime": mlretime}[args.what](args)
    print("DONE", args.what, flush=True)


if __name__ == "__main__":
    main()
