#!/usr/bin/env python3
"""Record the controls milestone from a complete --nocapture workspace test log."""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("test_log", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "validation/evidence/controls-stage.json")
    args = parser.parse_args()
    log = args.test_log.read_text()
    results = [tuple(map(int, match)) for match in re.findall(
        r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored", log)]
    if "Doc-tests ninjars_symbolica" not in log or "test result: FAILED" in log or not results:
        raise RuntimeError("a complete successful workspace test log is required")
    if any(failed or ignored for _, failed, ignored in results):
        raise RuntimeError("all reference tests must be enabled and pass")
    comparisons = []
    for line in log.splitlines():
        if "NINJARS_COMPARISON " in line:
            record = json.loads(line.split("NINJARS_COMPARISON ", 1)[1])
            record["comparison"].pop("artifact_directory", None)
            comparisons.append(record)
    if len(comparisons) != 4 or not all(
        check["passed"] for item in comparisons for check in item["comparison"]["checks"]
    ):
        raise RuntimeError("four passing comparison configurations are required")
    scale = re.search(r"automatic scale error ([^;]+); fixed scale three error (\S+)", log)
    if not scale:
        raise RuntimeError("sampling-scale diagnostic is missing")
    dependencies = {}
    manifest = json.loads((ROOT / "dependencies/manifest.json").read_text())
    for dependency in manifest["dependencies"]:
        dependencies[dependency["name"]] = subprocess.check_output(
            ["git", "-C", str(ROOT / dependency["path"]), "rev-parse", "HEAD"], text=True).strip()
    report = {
        "dependencies": dependencies,
        "workspace_tests": {"passed": sum(row[0] for row in results), "failed": 0, "ignored": 0},
        "command": "RUST_MIN_STACK=134217728 cargo test --workspace --all-features --locked -- --include-ignored --test-threads=1 --nocapture",
        "comparisons": comparisons,
        "sampling_scale_regression": {"automatic_scale": 247, "automatic_max_scaled_error": float(scale[1]),
                                      "fixed_scale": 3, "fixed_scaled_error": float(scale[2])},
        "timings_are_acceptance_evidence": False,
        "physical_integrated_acceptance": False,
        "full_goal_complete": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Recorded {report['workspace_tests']['passed']} tests and four comparison configurations: {args.output}")


if __name__ == "__main__":
    main()
