#!/usr/bin/env python3
"""Collect the generic-engine runs of scalar_cases.py sub-cases into one table file
(validation/rank-n2/scalars/<set>.json), readable by e2e_reduction.py through
E2E_SCALARS=<directory of tables>.

    export_scalars.py RUN_DIR OUT.json"""
import json
import pathlib
import sys


def main():
    run, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    lanes = []
    for f in sorted(run.glob("*.json")):
        for l in json.loads(f.read_text())["lanes"]:
            lanes.append({"id": l["id"], "laurent_decimal": l["laurent_decimal"]})
    lanes.sort(key=lambda l: l["id"])
    doc = {"format": "ninjars-generic-float-run-v1", "case": out.stem + "-scalars",
           "description": "exact scalar C0/D0 (NinjaRS generic engine, Float 256, SymbolicaMasters) of the "
                          "scalar_cases.py sub-cases; lane id <lane id>#<propagator subset>",
           "lanes": lanes}
    out.parent.mkdir(parents=True, exist_ok=True)
    body = ",\n".join(json.dumps(l, separators=(",", ":")) for l in lanes)
    head = json.dumps({k: v for k, v in doc.items() if k != "lanes"})[:-1]
    out.write_text(head + ', "lanes": [\n' + body + "\n]}\n")
    json.loads(out.read_text())
    print(out, len(lanes))


if __name__ == "__main__":
    main()
