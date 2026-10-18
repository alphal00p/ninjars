#!/usr/bin/env python3
"""Rank-0 sub-cases for the scalar C0/D0 masters of the integrand reduction.

For every lane of the given cases and every subset S of 3 or 4 propagators, one lane of
a rank-0 bench case ("ninjars-bench-case-v1", numerator 1) with the shifts, masses,
mu_R^2 and s_matrix block of S, exactly as review_reduction/e2e_reduction.py's
collier_scalar builds it.  Lane id "<lane id>#<i>,<j>,<k>[,<l>]" (S sorted).  Output:
DIR/scalars-n{3,4}-NNN.json (--chunk lanes each).  Their integrals are the scalar
C0/D0; evaluated with oneloopmaster's exact masters through the NinjaRS generic engine
at 256 bits (see README, "Scalar masters") they make the reduction independent of COLLIER.

    scalar_cases.py CASE.json... --out DIR [--invariants shifts]"""
import argparse
import itertools
import json
import pathlib


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="+", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--invariants", choices=["smatrix", "shifts"], default="smatrix",
                    help="shifts: s_matrix null, the invariants come from the exact shifts (input-ambiguity runs)")
    a = ap.parse_args()
    groups = {3: [], 4: []}
    seen = set()
    for p in a.cases:
        case = json.loads(p.read_text())
        n = case["n"]
        for lane in case["lanes"]:
            for k in (3, 4):
                if n < k:
                    continue
                for S in itertools.combinations(range(n), k):
                    lid = f"{lane['id']}#{','.join(map(str, S))}"
                    if lid in seen:
                        continue
                    seen.add(lid)
                    smat = lane.get("s_matrix")
                    groups[k].append({
                        "id": lid,
                        "shifts": [lane["shifts"][i] for i in S],
                        "masses_squared": [lane["masses_squared"][i] for i in S],
                        "mu_squared": lane["mu_squared"],
                        "s_matrix": [[smat[i][j] for j in S] for i in S]
                        if smat is not None and a.invariants == "smatrix" else None,
                        "tensor": [[1.0, 0.0]],
                    })
    a.out.mkdir(parents=True, exist_ok=True)
    for k, lanes in groups.items():
        for part in range(0, len(lanes), a.chunk):
            name = f"scalars-n{k}-{part // a.chunk:03d}"
            doc = {"format": "ninjars-bench-case-v1", "name": name, "n": k, "rank": 0,
                   "description": "scalar masters of the integrand reduction (scalar_cases.py)",
                   "lanes": lanes[part:part + a.chunk]}
            (a.out / f"{name}.json").write_text(json.dumps(doc) + "\n")
        print(f"n={k}: {len(lanes)} lanes")


if __name__ == "__main__":
    main()
