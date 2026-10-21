#!/usr/bin/env python3
"""Run the original-Ninja reference benchmark (ninja_bench.cc) on bench cases.

For every case file (ninjars-bench-case-v1, .json or .json.gz) and every
requested Ninja build variant (see build_ninja.py / ninja-builds.json), run
the pinned driver and merge the results into validation/benchmark/ninja/<case>.json
(+ <case>.masters.json with the exact master-integral calls), then rewrite
validation/benchmark/ninja/SUMMARY.md.

  python3 validation/benchmark/run_ninja_bench.py [CASE ...] [--discover]
      [--variants o3 o3-fortranrules o3-ol34] [--cpu 200] [--rounds 9] [--block-ms 25]
      [--oneloop-warnings]

OneLOop warnings are off by default in the driver (see ninja_bench.cc): one
stdout line per D0 call with a vanishing diagonal invariant would otherwise
be timed as master-integral work. --oneloop-warnings restores them.

--discover adds validation/benchmark/synthetic/*.json and the ENGINE/NUMERATOR
case directories (validation/fast/cases, validation/benchmark/cases) of this
worktree, of the main checkout and of references/claude_{engine,numerator}.
"""
import argparse
import datetime
import gzip
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path("/shared/external/ninja/references/ninjars_bench_main/validation/benchmark")
ROOT = HERE.parents[1]
SCRATCH = pathlib.Path("/shared/external/ninja/references/session-2026-10-20b/bench-main")
TARGET = SCRATCH / "refbuild"
OUT = pathlib.Path(os.environ.get("REF_OUT", str(SCRATCH / "refretime/ninja")))
DEFAULT_VARIANTS = ["o3", "o3-fortranrules"]


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def env():
    e = dict(os.environ)
    e["NIX_HARDENING_ENABLE"] = ""
    return e


def builds():
    return json.loads((TARGET / "ninja-builds.json").read_text())


def driver(variant, record):
    spec = record["variants"][variant]
    prefix = pathlib.Path(spec["prefix"])
    binary = TARGET / "ninja-runs/bin" / f"ninja_bench-{variant}"
    stamp = binary.with_suffix(".json")
    gfortran = subprocess.check_output(["gfortran", "-print-file-name=libgfortran.so"], text=True).strip()
    command = ["g++", *spec["cxxflags"].split(), "-std=c++11", f"-I{prefix / 'include'}",
               str(HERE / "ninja_bench.cc"), str(prefix / "lib/libninja.a"), spec["oneloop_library"],
               gfortran, f"-Wl,-rpath,{pathlib.Path(gfortran).parent}", "-lquadmath", "-o", str(binary)]
    want = {"driver_sha256": sha256(HERE / "ninja_bench.cc"), "libninja_sha256": spec["library_sha256"]["libninja.a"],
            "libavh_olo_sha256": spec["library_sha256"]["libavh_olo.a"], "command": command}
    if binary.exists() and stamp.exists():
        have = json.loads(stamp.read_text())
        if all(have.get(k) == v for k, v in want.items()) and have.get("binary_sha256") == sha256(binary):
            return binary, have
    binary.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env())
    want["binary_sha256"] = sha256(binary)
    want["compiler"] = subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]
    stamp.write_text(json.dumps(want, indent=2) + "\n")
    return binary, want


def discover():
    found = sorted((HERE / "synthetic").glob("*.json"))
    roots = [ROOT, pathlib.Path("/shared/external/ninja"),
             pathlib.Path("/shared/external/ninja/references/claude_engine"),
             pathlib.Path("/shared/external/ninja/references/claude_numerator")]
    seen = {p.name for p in found}
    for root in roots:
        for sub in ["validation/fast/cases", "validation/benchmark/cases"]:
            d = root / sub
            if d.is_dir():
                for p in sorted(list(d.glob("*.json")) + list(d.glob("*.json.gz"))):
                    if p.name not in seen:
                        seen.add(p.name)
                        found.append(p)
    return found


def materialize(path, tmpdir):
    """The driver reads plain JSON; decompress .gz inputs to a temporary file."""
    path = pathlib.Path(path)
    if path.suffix == ".gz":
        target = pathlib.Path(tmpdir) / path.name[:-3]
        target.write_bytes(gzip.decompress(path.read_bytes()))
        return target
    return path


def case_name(path):
    raw = pathlib.Path(path).read_bytes()
    if str(path).endswith(".gz"):
        raw = gzip.decompress(raw)
    return json.loads(raw)["name"]


def rel(a, b, scale):
    """|a-b| relative to the lane's scale (largest of |eps0|, |eps_m1|, |eps_m2|, |cc|, |rational|)."""
    return abs(complex(*a) - complex(*b)) / max(scale, 1e-300)


def lane_scale(lane):
    return max(abs(complex(*lane[k])) for k in ["eps0", "eps_m1", "eps_m2", "cut_constructible", "rational"])


def run_case(path, variants, record, args):
    name = case_name(path)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        plain = materialize(path, tmp)
        for variant in variants:
            binary, stamp = driver(variant, record)
            raw = TARGET / "ninja-runs" / variant / f"{name}.json"
            raw.parent.mkdir(parents=True, exist_ok=True)
            command = ["taskset", "-c", str(args.cpu), str(binary), str(plain), str(raw),
                       "--label", variant, "--rounds", str(args.rounds), "--block-ms", str(args.block_ms)]
            if args.oneloop_warnings:
                command.append("--oneloop-warnings")
            load = pathlib.Path("/proc/loadavg").read_text().split()[:3]
            print("+", " ".join(command), flush=True)
            proc = subprocess.run(command, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"{name}/{variant}: driver failed\n{proc.stderr}")
            data = json.loads(raw.read_text())
            data["host_loadavg_before"] = load
            data["driver_binary"] = stamp
            results[variant] = data
    first = results[variants[0]]
    lanes = []
    for i, lane in enumerate(first["lanes"]):
        entry = {k: lane[k] for k in ["id", "return_status", "eps0", "eps_m1", "eps_m2",
                                      "cut_constructible", "rational", "checks",
                                      "numerator_calls", "master_calls"]}
        entry["max_scaled_difference_between_builds"] = max(
            rel(lane[k], results[v]["lanes"][i][k], lane_scale(lane)) for v in variants[1:]
            for k in ["eps0", "eps_m1", "eps_m2", "cut_constructible", "rational"]) if len(variants) > 1 else 0.0
        entry["checks_all_builds"] = all(all(results[v]["lanes"][i]["checks"].values()) for v in variants)
        lanes.append(entry)
    masters = {"format": "ninjars-reference-master-calls-v1", "case": name,
               "note": "Exact IntegralLibrary calls made by original Ninja (in call order) and the OneLOop "
                       f"results, from build {variants[0]}; s = Ninja's S-matrix arguments, m2 complex masses.",
               "lanes": [{"id": lane["id"], "calls": lane["master_call_list"]} for lane in first["lanes"]]}
    timing = {}
    for v in variants:
        t = results[v]["timing"]
        timing[v] = {"median_ns": {m: d["median_ns"] for m, d in t["modes"].items()},
                     "blocks_ns": {m: d["blocks_ns"] for m, d in t["modes"].items()},
                     "iterations_per_block": {m: d["iterations_per_block"] for m, d in t["modes"].items()},
                     "derived": t["derived"], "protocol": t["protocol"],
                     "oneloop_warnings": results[v].get("oneloop_warnings", True),
                     "host_loadavg_before": results[v]["host_loadavg_before"],
                     "libninja_sha256": record["variants"][v]["library_sha256"]["libninja.a"],
                     "libavh_olo_sha256": record["variants"][v]["library_sha256"]["libavh_olo.a"],
                     "driver_binary_sha256": results[v]["driver_binary"]["binary_sha256"]}
    report = {
        "format": "ninjars-reference-ninja-results-v1",
        "case": name, "case_file": str(path), "case_sha256": sha256(path),
        "n": first["n"], "rank": first["rank"], "lanes_count": first["lanes_count"],
        "amplitude": first["amplitude"],
        "all_replays_bitwise_identical": all(results[v]["all_replays_bitwise_identical"] for v in variants),
        "lanes": lanes,
        "timing_unit": "ns per phase-space point (one lane), lanes cycled; median of interleaved blocks",
        "timing_cpu": args.cpu,
        "timing_modes": {m: d["description"] for m, d in first["timing"]["modes"].items()},
        "timing": timing,
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "driver_sha256": sha256(HERE / "ninja_bench.cc"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(report, indent=1) + "\n")
    (OUT / f"{name}.masters.json").write_text(json.dumps(masters) + "\n")
    return report


def fmt_us(ns):
    return f"{ns / 1000:.2f}"


def summary():
    rows = []
    for path in sorted(OUT.glob("*.json")):
        if path.name.endswith(".masters.json"):
            continue
        rows.append(json.loads(path.read_text()))
    lines = ["# Original Ninja + OneLOop reference timings", "",
             "Generated by `validation/benchmark/run_ninja_bench.py` (driver `ninja_bench.cc`).",
             "Times are microseconds per phase-space point (median of interleaved >=25 ms blocks, lanes cycled,",
             "pinned to one CPU of the shared host). `full` = TensorNumerator + OneLOop (no cache);",
             "`pure` = replayed numerator and masters; `num` = (TensorNumerator + replayed masters) - pure;",
             "`num*` = recorded TensorNumerator calls alone; `masters` = recorded OneLOop calls alone;",
             "`sum/full` = (pure + num + masters)/full; `entry` = MadLoop's C entry point ninja_tensor_evaluate_[smat_]cm_",
             "(new TensorNumerator + Amplitude per call); D0/C0/B/A0 = recorded box/triangle/bubble(rank-2 B11)/tadpole",
             "calls alone. Builds: see `../ninja-builds.json`. OneLOop warnings off (see ninja_bench.cc) unless a",
             "build's `oneloop_warnings` is true in the case JSON.", ""]
    variants = []
    for r in rows:
        for v in r["timing"]:
            if v not in variants:
                variants.append(v)
    for v in variants:
        lines += [f"## Build `{v}`", "",
                  "| case | n | rank | full | entry | pure | num | num* | masters | D0 | C0 | B | A0 | sum/full | num. calls | master calls |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in rows:
            if v not in r["timing"]:
                continue
            d = r["timing"][v]["derived"]
            nc = r["lanes"][0]["numerator_calls"]["total"]
            mc = r["lanes"][0]["master_calls"]["total"]
            med = r["timing"][v]["median_ns"]
            grp = lambda name: fmt_us(med[name]) if name in med else "-"
            lines.append(f"| {r['case']} | {r['n']} | {r['rank']} | {fmt_us(d['full_ns'])} | {grp('full_madloop_entry')} | "
                         f"{fmt_us(d['pure_reduction_ns'])} | "
                         f"{fmt_us(d['numerator_part_ns'])} | {fmt_us(d['numerator_only_ns'])} | {fmt_us(d['masters_ns'])} | "
                         f"{grp('masters_boxes')} | {grp('masters_triangles')} | {grp('masters_bubbles')} | {grp('masters_tadpoles')} | "
                         f"{d['sum_over_full']:.3f} | {nc} | {mc} |")
        lines.append("")
    lines += ["Correctness: every lane's replays are bitwise identical to the direct evaluation "
              "(`all_replays_bitwise_identical` in each JSON); per-lane Laurent coefficients, cut-constructible "
              "and rational parts are in `<case>.json`, the exact master calls in `<case>.masters.json`.", ""]
    (OUT / "SUMMARY.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", nargs="*", type=pathlib.Path)
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--cpu", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--block-ms", type=float, default=25.0)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--oneloop-warnings", action="store_true",
                        help="keep OneLOop's warnings on stdout (timed); default: off")
    args = parser.parse_args()
    if not args.summary_only:
        record = builds()
        cases = list(args.cases) + (discover() if args.discover else [])
        if not cases:
            parser.error("no cases")
        for path in cases:
            report = run_case(path.resolve(), args.variants, record, args)
            d = report["timing"][args.variants[0]]["derived"]
            print(f"{report['case']}: full {d['full_ns']/1e3:.2f} us, pure {d['pure_reduction_ns']/1e3:.2f}, "
                  f"num {d['numerator_part_ns']/1e3:.2f}, masters {d['masters_ns']/1e3:.2f}, "
                  f"sum/full {d['sum_over_full']:.3f}", flush=True)
    summary()


if __name__ == "__main__":
    main()
