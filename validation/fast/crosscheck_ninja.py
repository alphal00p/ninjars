#!/usr/bin/env python3
"""Cross-check the fast engine against ORIGINAL Ninja on bench-case files.

Builds `ninja_reference.cc` against the uncached reference Ninja + OneLOop
libraries (validation/reference/build.py --benchmark layout). For every point
of each "ninjars-bench-case-v1" file:

1. the Rust fast engine (`fast_bench --dump`: TensorNumerator + oneloopmaster
   native masters) reports its Laurent coefficients, cut-constructible and
   rational parts, and the master rows it used;
2. original Ninja (ninja::TensorNumerator) reduces the same point with exactly
   those master values (`masters_mode 1`): this isolates the reductions;
3. original Ninja with its own OneLOop masters (`masters_mode 2`) reports its
   results and every OneLOop master call, which are compared with the native
   rows of the same arguments.

The scaled error of a point is max_k |rust_k - ninja_k| / max(1, |ninja|),
with |ninja| the largest modulus among the point's Laurent coefficients; the
master comparison uses max |native - oneloop| / max(1e-300, |oneloop|) per
family. Only the reduction comparison (2) gates the exit status: a point
passes if its scaled error is below the tolerance or below ten times original
Ninja's own numerical noise, estimated as the scaled difference between
original Ninja on the point and on the same point with its propagators listed
in reverse order (an identical integral).

Usage:
  crosscheck_ninja.py --fast-bench PATH [--reference-build DIR]
                      [--tolerance 1e-9] [--json OUT] cases/*.json
"""

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent


def build(reference, output):
    source = reference / "ninja-source/src"
    ninja_build = reference / "ninja-build"
    libraries = [ninja_build / "src/.libs/libninja.a", reference / "oneloop/libavh_olo.a"]
    for path in libraries:
        if not path.exists():
            sys.exit(f"missing reference library {path}; run validation/reference/build.py")
    fortran = pathlib.Path(subprocess.check_output(
        ["gfortran", "-print-file-name=libgfortran.so"], text=True).strip())
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["g++", "-O3", "-DNDEBUG", "-std=c++11", f"-I{source}", f"-I{ninja_build / 'src'}",
               f"-I{ninja_build}", str(HERE / "ninja_reference.cc"), *map(str, libraries),
               str(fortran), f"-Wl,-rpath,{fortran.parent}", "-lquadmath", "-o", str(output)]
    subprocess.run(command, check=True)
    return output


def request(case, lane, mode, table=None, repetitions=0):
    n, rank = case["n"], case["rank"]
    matrix = lane.get("s_matrix")
    lines = [f"{n} {rank} {repetitions} {lane['mu_squared']!r} {1 if matrix else 0} {mode}"]
    for v, m in zip(lane["shifts"], lane["masses_squared"]):
        lines.append(" ".join(repr(float(x)) for x in [*v, *m]))
    if matrix:
        lines.append(" ".join(repr(float(x)) for row in matrix for x in row))
    lines.extend(f"{c[0]!r} {c[1]!r}" for c in lane["tensor"])
    if mode == 1:
        families = ["D0", "C0", "B", "A0"]
        lines.append(" ".join(str(len(table[f])) for f in families))
        for family in families:
            for row in table[family]:
                lines.append(" ".join(f"{z[0]!r} {z[1]!r}" for z in row))
    return "\n".join(lines) + "\n"


def run_ninja(executable, text):
    out = subprocess.run([str(executable)], input=text, capture_output=True, text=True, check=True)
    results, masters, current = [], [], []
    for line in out.stdout.splitlines():
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        if "master" in record:
            current.append(record)
        else:
            results.append(record)
            masters.append(current)
            current = []
    return results, masters


def run_rust(fast_bench, path):
    out = subprocess.run([str(fast_bench), "--dump", str(path)], capture_output=True, text=True,
                         check=True)
    return [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]


def cx(z):
    return complex(z[0], z[1])


def reversed_case(case):
    out = dict(case)
    out["lanes"] = []
    for lane in case["lanes"]:
        lane = dict(lane)
        lane["shifts"] = lane["shifts"][::-1]
        lane["masses_squared"] = lane["masses_squared"][::-1]
        if lane.get("s_matrix"):
            lane["s_matrix"] = [row[::-1] for row in lane["s_matrix"][::-1]]
        out["lanes"].append(lane)
    return out


def scaled_differences(ninja, other):
    out = []
    for a, b in zip(ninja, other):
        if None in a["epsilon"] or None in b["epsilon"]:
            out.append(float("inf"))
            continue
        scale = max(1.0, max(abs(cx(z)) for z in a["epsilon"]))
        out.append(max(abs(cx(x) - cx(y)) for x, y in zip(a["epsilon"], b["epsilon"])) / scale)
    return out


def compare_results(case, ninja, rust, noise=None, tolerance=0.0):
    worst = {"laurent": 0.0, "cut_constructible": 0.0, "rational": 0.0}
    failures = []
    within_noise = 0
    for index, (lane, a, b) in enumerate(zip(case["lanes"], ninja, rust)):
        if a["status"] != 0 or b["status"] != "Success" or None in a["epsilon"] \
                or a.get("master_misses", 0):
            failures.append([lane["id"], a["status"], b["status"], a.get("master_misses", 0)])
            continue
        scale = max(1.0, max(abs(cx(z)) for z in a["epsilon"]))
        d = lambda x, y: abs(cx(x) - cx(y)) / scale
        error = max(d(x, y) for x, y in zip(a["epsilon"], b["laurent"]))
        worst["laurent"] = max(worst["laurent"], error)
        if noise is not None and error > tolerance:
            if error <= 10 * noise[index]:
                within_noise += 1
            else:
                failures.append([lane["id"], "error %.3e exceeds 10x noise %.3e" % (error, noise[index])])
        worst["cut_constructible"] = max(worst["cut_constructible"],
                                         d(a["cut_constructible"], b["cut_constructible"]))
        worst["rational"] = max(worst["rational"], d(a["rational"], b["rational"]))
    return worst, failures, within_noise


def compare_masters(rust, ninja_masters):
    """Largest relative deviation of native masters from OneLOop, per family,
    matching rows by their arguments."""
    worst = {}
    for point, calls in zip(rust, ninja_masters):
        for call in calls:
            family = call["master"]
            arity = len(call["args"])
            rows = point["masters"][family]
            key = [cx(z) for z in call["args"]]
            best = None
            for row in rows:
                if all(abs(cx(z) - k) <= 1e-12 * max(1.0, abs(k)) for z, k in zip(row[:arity], key)):
                    best = row
                    break
            if best is None:
                worst.setdefault(family, {"max_relative": 0.0, "unmatched": 0})["unmatched"] += 1
                continue
            values = [cx(z) for z in call["values"]]
            native = [cx(z) for z in best[arity:arity + len(values)]]
            for triple in range(len(values) // 3):
                reference = values[3 * triple:3 * triple + 3]
                scale = max(abs(v) for v in reference)
                if scale == 0:
                    continue
                error = max(abs(a - b) for a, b in zip(native[3 * triple:], reference)) / scale
                entry = worst.setdefault(family, {"max_relative": 0.0, "unmatched": 0})
                entry["max_relative"] = max(entry["max_relative"], error)
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", nargs="+", type=pathlib.Path)
    parser.add_argument("--fast-bench", type=pathlib.Path, required=True)
    parser.add_argument("--reference-build", type=pathlib.Path,
                        default=ROOT / "target/reference-uncached-v2")
    parser.add_argument("--executable", type=pathlib.Path,
                        default=ROOT / "target/fast-reference/ninja_reference")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args()
    executable = build(args.reference_build.resolve(), args.executable)
    report = []
    ok = True
    for path in args.cases:
        case = json.loads(path.read_text())
        rust = run_rust(args.fast_bench, path)
        table_text = "".join(request(case, lane, 1, point["masters"])
                             for lane, point in zip(case["lanes"], rust))
        same_masters, _ = run_ninja(executable, table_text)
        oneloop, calls = run_ninja(executable, "".join(request(case, lane, 2) for lane in case["lanes"]))
        flipped, _ = run_ninja(executable, "".join(request(reversed_case(case), lane, 0)
                                                    for lane in reversed_case(case)["lanes"]))
        noise = scaled_differences(oneloop, flipped)
        reduction, failures, within_noise = compare_results(case, same_masters, rust, noise,
                                                            args.tolerance)
        end_to_end, _, _ = compare_results(case, oneloop, rust)
        passed = not failures
        ok &= passed
        report.append({"case": case["name"], "points": len(case["lanes"]),
                       "same_masters_max_scaled_error": reduction,
                       "points_above_tolerance_within_ninja_noise": within_noise,
                       "ninja_noise_max_scaled": max(noise),
                       "oneloop_masters_max_scaled_error": end_to_end,
                       "native_vs_oneloop_masters": compare_masters(rust, calls),
                       "failures": failures, "passed": passed})
        print(json.dumps(report[-1]), flush=True)
    if args.json:
        args.json.write_text(json.dumps({"tolerance": args.tolerance, "cases": report}, indent=1) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
