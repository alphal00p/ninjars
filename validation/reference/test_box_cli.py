#!/usr/bin/env python3
"""Compare exact DOT preparation, original Ninja, and both Rust box modes."""
import argparse
import json
import pathlib
import subprocess
import sys

from test_generated_runner import BUILD, ROOT, close, evaluate, request


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--cli", type=pathlib.Path, default=ROOT / "target/debug/ninjars"
    )
    cli = parser.parse_args().cli.resolve()
    fixture = "examples/graphs/box.json"
    directory = BUILD / "prepared-box"
    prepared = json.loads(
        subprocess.check_output(
            [cli, "prepare", ROOT / fixture, "--reference-dir", directory],
            cwd=ROOT,
            text=True,
        )
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    assert len(manifest["samples"]) == 3
    assert manifest["overall_factor_included_in_callback"] is True
    references = []
    # Original TensorNumerator rank ordering for 1 + q0 + 2*q1*q2.
    tensor = [1, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0]
    for sample in manifest["samples"]:
        lane = sample["lane"]
        assert sample["rank"] == 4 and sample["propagator_count"] == 4
        assert sample["parameter_count"] == 0
        assert sample["mass_squared_binary64"] == [[1, 0]] * 4
        assert sample["full_numerator"] == prepared["samples"][lane]["numerator"]
        reference_input = (directory / sample["input"]).read_text()
        assert float(reference_input.splitlines()[0].split()[3]) == 1
        executable = directory / f"reference-{lane}"
        subprocess.run(
            [
                sys.executable,
                ROOT / "validation/reference/build.py",
                "--runner-only",
                "--numerator-header",
                directory / sample["header"],
                "--output",
                executable,
            ],
            check=True,
            cwd=ROOT,
        )
        # A fresh process avoids original Ninja's incomplete D0 cache key.
        (value,) = evaluate(executable, [reference_input])
        (control,) = evaluate(
            BUILD / "ninja-reference",
            [request(sample["shifts_binary64"], 2, tensor)],
        )
        expected = [v[:] for v in control["epsilon"]]
        expected[0][0] -= 1 / 6
        close(value["epsilon"], expected)
        references.append(
            {
                "lane": lane,
                "numerator": prepared["samples"][lane]["numerator"],
                "full_numerator": sample["full_numerator"],
                "shifts_binary64": sample["shifts_binary64"],
                "mass_squared_binary64": sample["mass_squared_binary64"],
                "reference": value,
                "tensor": control,
                "analytic_mu4_rational": -1 / 6,
                "artifact": json.loads(executable.with_suffix(".json").read_text()),
            }
        )
    runs = []
    for precision in ["f64", "double-float", "float"]:
        for mode in ["expansion", "sampling"]:
            for batch in [1, 2]:
                command = [
                    cli,
                    "reduce",
                    ROOT / fixture,
                    "--precision",
                    precision,
                    "--batch-size",
                    str(batch),
                    "--mu-extraction",
                    mode,
                    "--full-box-coefficients",
                    "--mu-sampling-scale",
                    "2",
                ]
                if precision == "float":
                    command += ["--bits", "256"]
                result = json.loads(
                    subprocess.check_output(command, cwd=ROOT, text=True)
                )
                assert result["success"] and result["rank"] == 4
                assert len(result["results"]) == len(references)
                for row, reference in zip(result["results"], references):
                    assert row["status"]["kind"] == "success"
                    assert row["lane"] == reference["lane"]
                    values = [[float(x) for x in pair] for pair in row["values"]]
                    close(values[:3], reference["reference"]["epsilon"])
                    close(values[0], [x + y for x, y in zip(values[3], values[4])])
                runs.append(
                    {
                        "precision": result["precision"],
                        "batch_size": batch,
                        "box_reduction": result["box_reduction"],
                        "results": result["results"],
                    }
                )
    evidence = {
        "fixture": fixture,
        "graph_source": prepared["graph_source"],
        "input_fixture": prepared["input_fixture"],
        "propagator_edges": prepared["propagator_edges"],
        "reference_cache_policy": (
            "Fresh original Ninja process per phase-space sample; "
            "no cross-point cache reuse"
        ),
        "epsilon_order": ["finite", "single_pole", "double_pole"],
        "reference_checks": references,
        "rust_runs": runs,
        "comparison_tolerance": (
            "scaled relative 1e-9; rational split may depend on residue basis"
        ),
        "timings_are_acceptance_evidence": False,
    }
    path = ROOT / "validation/evidence/box-cli.json"
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        f"Validated three prepared boxes against original Ninja and {len(runs)} "
        f"Rust precision/mode/batch configurations: {path}"
    )


if __name__ == "__main__":
    main()
