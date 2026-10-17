#!/usr/bin/env python3
"""Validate the shared phase-space pools and exported tensor bench cases.

1. Pools (validation/benchmark/points/*.json): binary64 momentum conservation,
   mass shells, positive energies and the physical production threshold.
2. Bench cases (validation/benchmark/cases/*.json, "ninjars-bench-case-v1"):
   shapes, lane ids, symmetric zero-diagonal S matrices consistent with the
   shifts, and squared masses.
3. Original C++ Ninja with ``ninja::TensorNumerator`` on every exported lane
   (validation/reference/tensor_runner.cc, linked against unchanged Ninja and
   OneLOop libraries) against ``ninjars reduce`` of the corresponding pool
   fixture. Both integrate the same FeynKit loop momentum, so no routing or
   orientation conversion enters; Ninja runs once with the exported S matrix
   and once deriving it from the shifts. The pass criterion uses the
   double-float reduction of the exact fixture inputs; the binary64 reduction
   is compared against the same Ninja values and recorded as a diagnostic.

Run from the repository root inside the development shell, after building the
CLI and the reference libraries (validation/reference/build.py):

    cargo build --profile perf -p ninjars-cli --features numerator-jit
    python3 validation/benchmark/validate_cases.py --ninjars target/perf/ninjars \
        --reference-build /path/to/target/reference-uncached-v2
"""
import argparse
import fractions
import hashlib
import json
import math
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCH = ROOT / "validation/benchmark"
CRITERION = 1e-9
PRECISIONS = ["double-float", "f64"]
REPRESENTATIVES = [
    ("hgg", "top-triangle-0"),
    ("four-photon", "top-box-0"),
    ("gg-hh", "top-triangle-0"),
    ("gg-hh", "top-box-2"),
    ("gg-hhh", "top-triangle-0"),
    ("gg-hhh", "top-box-8"),
    ("gg-hhh", "top-pentagon-26"),
]


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def minkowski(p):
    return p[0] * p[0] - p[1] * p[1] - p[2] * p[2] - p[3] * p[3]


def ncoeff(rank):
    return (rank + 1) * (rank + 2) * (rank + 3) * (rank + 4) // 24


def check_pool(process):
    pool = json.loads((BENCH / "points" / f"{process}.json").read_text())
    assert pool["format"] == "ninjars-phase-space-pool-v1"
    legs = pool["legs"]
    masses = [float(fractions.Fraction(leg["mass"])) for leg in legs]
    worst = {"conservation": 0.0, "mass_shell": 0.0}
    points = []
    for point in pool["points"]:
        momenta = point["momenta"]
        assert len(momenta) == len(legs) and len(point["helicities"]) == len(legs)
        scale = max(p[0] for p in momenta)
        assert all(p[0] > 0 for p in momenta), point["id"]
        total = [sum((1 if leg["state"] == "incoming" else -1) * p[k]
                     for p, leg in zip(momenta, legs)) for k in range(4)]
        conservation = max(abs(x) for x in total) / scale
        shell = max(abs(minkowski(p) - m * m) / (p[0] * p[0]) for p, m in zip(momenta, masses))
        incoming = [sum(p[k] for p, leg in zip(momenta, legs) if leg["state"] == "incoming")
                    for k in range(4)]
        sqrt_s = math.sqrt(minkowski(incoming))
        threshold = sum(m for m, leg in zip(masses, legs) if leg["state"] == "outgoing")
        assert sqrt_s > threshold, point["id"]
        assert point["scale_squared"] == float(fractions.Fraction(pool["scale_squared"]))
        worst["conservation"] = max(worst["conservation"], conservation)
        worst["mass_shell"] = max(worst["mass_shell"], shell)
        points.append({"id": point["id"], "sqrt_s": sqrt_s, "production_threshold": threshold,
                       "conservation_relative_to_max_energy": conservation,
                       "mass_shell_relative_to_energy_squared": shell})
    # Rounding a momentum component once costs at most half an ulp (2^-53).
    assert worst["conservation"] < 1e-15 and worst["mass_shell"] < 1e-15, worst
    return pool, {"process": process, "points": points, "worst": worst}


def check_case(case, pool, graph):
    assert case["format"] == "ninjars-bench-case-v1"
    n, rank = case["n"], case["rank"]
    assert [lane["id"] for lane in case["lanes"]] == [p["id"] for p in pool["points"]]
    worst = 0.0
    for lane, point in zip(case["lanes"], pool["points"]):
        assert len(lane["shifts"]) == n and all(len(v) == 4 for v in lane["shifts"])
        assert len(lane["masses_squared"]) == n
        assert len(lane["tensor"]) == ncoeff(rank)
        assert lane["mu_squared"] == point["scale_squared"]
        matrix = lane["s_matrix"]
        assert matrix is not None and len(matrix) == n
        scale = max(abs(v[0]) for v in lane["shifts"]) ** 2
        for i in range(n):
            assert matrix[i][i] == 0
            for j in range(n):
                assert matrix[i][j] == matrix[j][i]
                d = [a - b for a, b in zip(lane["shifts"][i], lane["shifts"][j])]
                worst = max(worst, abs(matrix[i][j] - minkowski(d)) / scale)
    assert worst < 1e-12, (graph, worst)
    return {"s_matrix_vs_rounded_shifts_relative_to_energy_squared": worst}


def build_runner(reference, output):
    fortran = pathlib.Path(subprocess.check_output(
        ["gfortran", "-print-file-name=libgfortran.so"], text=True).strip())
    libraries = [reference / "ninja-build/src/.libs/libninja.a", reference / "oneloop/libavh_olo.a"]
    command = ["g++", "-O3", "-DNDEBUG", "-std=c++11",
               "-I" + str(reference / "ninja-source/src"), "-I" + str(reference / "ninja-build/src"),
               "-I" + str(reference / "ninja-build"), str(ROOT / "validation/reference/tensor_runner.cc"),
               *map(str, libraries), str(fortran), "-Wl,-rpath," + str(fortran.parent),
               "-lquadmath", "-o", str(output)]
    subprocess.run(command, check=True)
    revisions = json.loads((reference / "revisions.json").read_text())
    return {"command": [*command[:-1], "<tmp>/ninja-tensor-runner"], "libraries_sha256": {p.name: sha256(p) for p in libraries},
            "ninja_revision": revisions.get("ninja"), "configuration": revisions.get("configuration"),
            "runner_source_sha256": sha256(ROOT / "validation/reference/tensor_runner.cc")}


def request(case, lane):
    rows = [f"{case['n']} {case['rank']} 1 {lane['mu_squared']!r}"]
    for v, m in zip(lane["shifts"], lane["masses_squared"]):
        rows.append(" ".join(repr(float(x)) for x in [*v, *m]))
    rows.extend(f"{re!r} {im!r}" for re, im in lane["tensor"])
    return "\n".join(rows) + "\n"


def ninja(runner, case, lane, directory):
    arguments = [str(runner)]
    if lane is not None and lane.get("s_matrix") is not None:
        path = directory / "s_matrix.txt"
        path.write_text(f"{case['n']}\n" + "\n".join(" ".join(repr(float(x)) for x in row)
                                                   for row in lane["s_matrix"]) + "\n")
        arguments += ["--s-matrix", str(path)]
    result = subprocess.run(arguments, input=request(case, lane), text=True,
                            capture_output=True, check=True)
    # OneLOop prints its banner on standard output before the first result.
    row = json.loads([line for line in result.stdout.splitlines() if line.startswith("{")][-1])
    assert row["status"] == 0, row
    return [complex(*value) for value in row["epsilon"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ninjars", type=pathlib.Path, default=ROOT / "target/perf/ninjars")
    parser.add_argument("--reference-build", type=pathlib.Path,
                        default=ROOT / "target/reference-uncached-v2")
    parser.add_argument("--output", type=pathlib.Path, default=BENCH / "case-validation.json")
    args = parser.parse_args()
    reference = args.reference_build.resolve()
    report = {"schema_version": 1, "criterion":
              f"each complex Laurent coefficient abs(a-b)/max(1,abs(b)) <= {CRITERION}",
              "pools": [], "cases": []}
    with tempfile.TemporaryDirectory() as work:
        work = pathlib.Path(work)
        runner = work / "ninja-tensor-runner"
        report["reference"] = build_runner(reference, runner)
        pools = {}
        for process in dict.fromkeys(p for p, _ in REPRESENTATIVES):
            pools[process], summary = check_pool(process)
            report["pools"].append(summary)
        passed = {precision: True for precision in PRECISIONS}
        for process, graph in REPRESENTATIVES:
            name = f"{process}-{graph}"
            path = BENCH / "cases" / f"{name}.json"
            case = json.loads(path.read_text())
            structure = check_case(case, pools[process], name)
            fixture = ROOT / "examples/graphs/physical" / process / f"{graph}-pool.json"
            reduced, commands = {}, {}
            for precision in PRECISIONS:
                output = work / f"{name}-{precision}.json"
                command = [str(args.ninjars), "reduce", str(fixture.relative_to(ROOT)),
                           "--precision", precision, "--output", str(output)]
                subprocess.run(command, cwd=ROOT, check=True)
                commands[precision] = ["ninjars", *command[1:-1], "<tmp>"]
                reduced[precision] = json.loads(output.read_text())["results"]
            lanes, worst = [], {precision: [0.0, 0.0] for precision in PRECISIONS}
            for index, lane in enumerate(case["lanes"]):
                row = {"id": lane["id"]}
                ninja_values = {}
                for label in ("exported_s_matrix", "s_matrix_from_shifts"):
                    probe = lane if label == "exported_s_matrix" else {**lane, "s_matrix": None}
                    ninja_values[label] = ninja(runner, case, probe, work)
                    row[f"ninja_{label}"] = [[v.real, v.imag] for v in ninja_values[label]]
                for precision in PRECISIONS:
                    result = reduced[precision][index]
                    assert result["status"]["kind"] == "success", (name, precision, result["status"])
                    rust = [complex(float(re), float(im)) for re, im in result["values"][:3]]
                    largest = max(abs(v) for v in rust)
                    entry = {"values": [[v.real, v.imag] for v in rust]}
                    for label, values in ninja_values.items():
                        scaled = [abs(a - b) / max(1.0, abs(b)) for a, b in zip(rust, values)]
                        relative = [abs(a - b) / largest for a, b in zip(rust, values)]
                        entry[f"scaled_errors_vs_ninja_{label}"] = scaled
                        entry[f"relative_errors_vs_ninja_{label}"] = relative
                        worst[precision][0] = max(worst[precision][0], *scaled)
                        worst[precision][1] = max(worst[precision][1], *relative)
                    row[f"ninjars_reduce_{precision}"] = entry
                lanes.append(row)
            summary = {}
            for precision in PRECISIONS:
                passed[precision] &= worst[precision][0] <= CRITERION
                summary[precision] = {"max_scaled_error": worst[precision][0],
                                      "max_error_relative_to_largest_laurent_coefficient": worst[precision][1],
                                      "pass": worst[precision][0] <= CRITERION}
            report["cases"].append({
                "case": str(path.relative_to(ROOT)), "case_sha256": sha256(path),
                "fixture": str(fixture.relative_to(ROOT)), "reduce_commands": commands,
                "structure": structure, "summary": summary, "lanes": lanes})
            print(name, json.dumps(summary), flush=True)
        # The pass criterion uses the double-float reduction; binary64 NinjaRS
        # reductions are recorded against the same Ninja values as diagnostics.
        report["pass"] = passed["double-float"]
        report["pass_by_reduce_precision"] = passed
        report["laurent_order"] = ["finite", "single_pole", "double_pole"]
        report["ninjars_build"] = {"executable": str(args.ninjars), "sha256": sha256(args.ninjars)}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["pass"]:
        raise SystemExit("bench cases disagree with original Ninja")


if __name__ == "__main__":
    main()
