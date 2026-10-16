#!/usr/bin/env python3
"""Verify DOT preparation exports through original Ninja and its tensor interface."""
import argparse
import json
import pathlib
import subprocess
import sys

from test_generated_runner import BUILD, ROOT, close, evaluate, request


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--cli", type=pathlib.Path, required=True,
                        help="ninjars built with --features reference-codegen")
    args = parser.parse_args()
    evidence = {"checks": [], "comparison": "original generated callbacks versus original tensor interface plus analytic mu controls",
                "timings_are_acceptance_evidence": False}
    cases = [
        ("bubble", "examples/graphs/bubble.json", 2,
         [1, 1, 0, 0, 0, 0, 0, 2] + [0] * 7, [7 / 6, 5 / 3, 133 / 96]),
        ("triangle", "examples/graphs/triangle-reference.json", 3,
         [1, 0, 0, 0, 0, 0, 1] + [0] * 17 + [1] + [0] * 10, [2 / 3]),
    ]
    for name, fixture, rank, tensor, rational_controls in cases:
        directory = BUILD / f"prepared-{name}"
        preparation = subprocess.run([args.cli.resolve(), "prepare", ROOT / fixture,
                                      "--reference-dir", directory],
                                     cwd=ROOT, text=True, capture_output=True, check=True)
        prepared = json.loads(preparation.stdout)
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["reference_precision"] == "binary64"
        assert manifest["overall_factor_included_in_callback"] is True
        assert len(manifest["samples"]) == len(rational_controls)
        for sample, mu_rational in zip(manifest["samples"], rational_controls):
            lane = sample["lane"]
            assert sample["rank"] == rank
            executable = directory / f"reference-{lane}"
            subprocess.run([sys.executable, ROOT / "validation/reference/build.py",
                            "--runner-only", "--numerator-header", directory / sample["header"],
                            "--output", executable], check=True, cwd=ROOT)
            value, = evaluate(executable, [(directory / sample["input"]).read_text()])
            # Fixtures have unit squared masses, unit graph factors and scale.
            # The mu-free polynomial is independently supplied to Ninja's
            # TensorNumerator in its documented component-coefficient order.
            control, = evaluate(BUILD / "ninja-reference",
                                [request(sample["shifts_binary64"], rank, tensor)])
            expected = [pair[:] for pair in control["epsilon"]]
            expected[0][0] += mu_rational
            close(value["epsilon"], expected)
            close(value["rational"], [control["rational"][0] + mu_rational,
                                       control["rational"][1]])
            evidence["checks"].append({
                "fixture": fixture, "lane": lane,
                "prepared_numerator": prepared["samples"][lane]["numerator"],
                "reference": value, "tensor": control, "analytic_mu_rational": mu_rational,
                "artifact": json.loads(executable.with_suffix(".json").read_text()),
            })
    output = BUILD / "prepared-graph-validation.json"
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Passed {len(evidence['checks'])} prepared graph checks; evidence: {output}")


if __name__ == "__main__":
    main()
