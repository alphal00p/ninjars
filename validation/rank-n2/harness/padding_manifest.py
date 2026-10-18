#!/usr/bin/env python3
"""V5 (rank padding) manifest: every rank <= n+1 bench case of the repository that has an
original-Ninja result, with the paths of both (relative to the repository root).  The
test declares each case as rank n+2 (zero top coefficients) and compares the x2 path with
the Ninja result (fast engine, 1e-11) and with the generic rank <= n+1 result (Float).

    padding_manifest.py --out validation/rank-n2/identity/padding-manifest.json"""
import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCES = [("validation/fast/ninja_reference/cases", "validation/fast/ninja_reference/ninja"),
           ("validation/benchmark/cases", "validation/benchmark/ninja"),
           ("validation/benchmark/synthetic", "validation/benchmark/ninja"),
           ("validation/rank-n2/controls/cases", "validation/rank-n2/controls/ninja")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    rows = []
    for cdir, ndir in SOURCES:
        for c in sorted((REPO / cdir).glob("*.json")):
            n = REPO / ndir / c.name
            if not n.exists():
                continue
            case = json.loads(c.read_text())
            if case["rank"] > case["n"] + 1 or case.get("format") != "ninjars-bench-case-v1":
                continue
            rows.append({"case": f"{cdir}/{c.name}", "ninja": f"{ndir}/{c.name}", "n": case["n"],
                         "rank": case["rank"], "declared_rank": case["n"] + 2, "lanes": len(case["lanes"])})
    doc = {"format": "ninjars-rank-n2-padding-v1",
           "description": "rank <= n+1 cases with original-Ninja results, to be run declared as rank n+2 (V5)",
           "cases": rows}
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{len(rows)} cases, {sum(r['lanes'] for r in rows)} lanes")


if __name__ == "__main__":
    main()
