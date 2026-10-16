#!/usr/bin/env python3
"""Check Symbolica-generated callbacks through original Ninja and OneLOop."""
import argparse
import json
import math
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILD = ROOT / "target/reference"
TRIANGLE = [(0, 0, 0, 0), (2, 1, 0, 0), (3, 0, 1, 0)]
BOX = [*TRIANGLE, (4, 0, 0, 1)]


def request(shifts, rank, coefficients, repetitions=3):
    lines = [f"{len(shifts)} {rank} {repetitions} 1"]
    lines.extend(" ".join(map(str, (*shift, 1, 0))) for shift in shifts)
    lines.extend(f"{value.real} {value.imag}" for value in map(complex, coefficients))
    return "\n".join(lines) + "\n"


def evaluate(executable, requests):
    result = subprocess.run([executable], input="".join(requests),
                            text=True, capture_output=True, check=True)
    values = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(values) == len(requests), result.stdout
    for value in values:
        assert value["status"] == 0, value
        assert all(pair is not None for pair in value["epsilon"]), value
        assert value["rational"] is not None, value
    return values


def close(actual, expected):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        if isinstance(left, list):
            close(left, right)
        else:
            assert math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-11), (actual, expected)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--generator", type=pathlib.Path, required=True,
                        help="compiled ninjars-symbolica generate_reference example executable")
    args = parser.parse_args()
    generator = args.generator.resolve()
    environment = {**os.environ, "SYMBOLICA_HIDE_BANNER": "1"}
    evidence = {"checks": {}, "timings_are_acceptance_evidence": False}

    def build(name, propagators, expression, parameters=()):
        header = BUILD / f"{name}.hh"
        executable = BUILD / f"ninja-{name}"
        subprocess.run([generator, str(propagators), expression, header, *parameters],
                       cwd=ROOT, env=environment, check=True)
        subprocess.run([sys.executable, ROOT / "validation/reference/build.py",
                        "--runner-only", "--numerator-header", header, "--output", executable],
                       cwd=ROOT, check=True)
        return executable

    triangle_mu2 = build("triangle-mu2", 3, "mu2")
    value, = evaluate(triangle_mu2, [request(TRIANGLE, 2, [])])
    close(value["epsilon"], [[0.5, 0], [0, 0], [0, 0]])
    close(value["rational"], [0.5, 0])
    evidence["checks"]["triangle_mu2"] = value

    box_mu4 = build("box-mu4", 4, "mu2^2")
    value, = evaluate(box_mu4, [request(BOX, 4, [])])
    close(value["epsilon"], [[-1 / 6, 0], [0, 0], [0, 0]])
    close(value["rational"], [-1 / 6, 0])
    evidence["checks"]["box_mu4"] = value

    # Coefficients are literal component monomials in documented Ninja order.
    polynomial = "q0^2+2*q0*q1+3*q1^2+5*q2+7*q3+11"
    tensor = [11, 0, 0, 5, 7, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
    generated = build("triangle-tensor-parity", 3, polynomial)
    direct, = evaluate(generated, [request(TRIANGLE, 2, [])])
    control, = evaluate(BUILD / "ninja-reference", [request(TRIANGLE, 2, tensor)])
    for key in ["epsilon", "rational", "cut_constructible"]:
        close(direct[key], control[key])
    evidence["checks"]["mu_free_tensor_parity"] = {"generated": direct, "tensor": control}

    # Change a complex parameter between requests, checking output order and
    # preservation of the runtime parameter rather than a compiled constant.
    generated = build("triangle-parameter", 3, "a*q0^2+mu2", ["a"])
    parameters = [2 + 3j, -4 + 1j, 0j]
    direct = evaluate(generated, [request(TRIANGLE, 2, [a]) for a in parameters])
    controls = evaluate(BUILD / "ninja-reference",
                        [request(TRIANGLE, 2, [0] * 5 + [a] + [0] * 9) for a in parameters])
    for value, control in zip(direct, controls):
        expected = [pair[:] for pair in control["epsilon"]]
        expected[0][0] += 0.5
        close(value["epsilon"], expected)
        close(value["rational"], [control["rational"][0] + 0.5, control["rational"][1]])
    evidence["checks"]["complex_runtime_parameters"] = direct

    # R=n+1 exercises the higher-rank callback layouts. The independent check
    # here is linearity against a separate generated mu4 callback and a tensor
    # q0^2 numerator, not an independent closed-form higher-rank master value.
    mu4 = build("triangle-mu4", 3, "mu2^2")
    combined = build("triangle-q0sq-mu4", 3, "q0^2+mu2^2")
    high, = evaluate(combined, [request(TRIANGLE, 4, [])])
    pure, = evaluate(mu4, [request(TRIANGLE, 4, [])])
    tensor_value, = evaluate(BUILD / "ninja-reference",
                            [request(TRIANGLE, 2, [0] * 5 + [1] + [0] * 9)])
    for key in ["epsilon", "rational", "cut_constructible"]:
        if key == "epsilon":
            expected = [[a + b for a, b in zip(x, y)]
                        for x, y in zip(pure[key], tensor_value[key])]
        else:
            expected = [a + b for a, b in zip(pure[key], tensor_value[key])]
        close(high[key], expected)
    evidence["checks"]["higher_rank_linearity"] = {"combined": high, "mu4": pure,
                                                     "tensor_q0sq": tensor_value}
    output = BUILD / "generated-callback-validation.json"
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Passed {len(evidence['checks'])} integrated callback checks; evidence: {output}")


if __name__ == "__main__":
    main()
