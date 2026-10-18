#!/usr/bin/env python3
"""Exact 1/eps coefficient of IR-finite lanes with rank R = 2N - 4 (log UV divergence).

Only the degree-R block of T_0 is UV divergent (mu^2 blocks give finite rational terms), and
    T^{mu1..mu2k} UV pole = {g..g} / (2^(N-2) (N-1)!)
contracted with Ninja's monomials (prod_mu (a_mu - 1)!! (-1)^((a1+a2+a3)/2)).  Compares with
eps_m1 of the reference files (laurent_decimal), exact rational arithmetic.

    uv_check.py CASE_DIR REF_DIR [--skip-family irml,irsoft]"""
import argparse
import json
import pathlib
import sys
from decimal import Decimal, getcontext
from math import factorial

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collier_reference as cr  # noqa: E402
from mp_lowpoint import pairing_factor  # noqa: E402

getcontext().prec = 60


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", type=pathlib.Path)
    ap.add_argument("refs", type=pathlib.Path)
    ap.add_argument("--skip-family", default="irml,irsoft")
    a = ap.parse_args()
    skip = set(a.skip_family.split(","))
    worst, count = 0.0, 0
    for cf in sorted(a.cases.glob("*.json")):
        case = json.loads(cf.read_text())
        N, R = case["n"], case["rank"]
        if R != 2 * N - 4 or N < 3 or case.get("family") in skip or not (a.refs / cf.name).exists():
            continue
        if any(f"-{f}-" in case["name"] for f in skip):
            continue
        refs = {l["id"]: l for l in json.loads((a.refs / cf.name).read_text())["lanes"]}
        for lane in case["lanes"]:
            T0 = dict(cr.numerator_blocks(case, lane))[0]
            pole = cr.ZERO
            for m, c in T0.items():
                if sum(m) == R:
                    P = pairing_factor(m)
                    if P:
                        pole = cr.cadd(pole, cr.cscale(c, cr.Q(P, 2 ** (N - 2) * factorial(N - 1))))
            ref = refs.get(lane["id"])
            if not ref or "laurent_decimal" not in ref:
                continue
            got = [Decimal(x) for x in ref["laurent_decimal"]["eps_m1"]]
            ex = [cr.qdec(pole[0]), cr.qdec(pole[1])]
            sc = Decimal(ref.get("scale") or 1)
            d = float(((got[0] - ex[0]) ** 2 + (got[1] - ex[1]) ** 2).sqrt() / sc)
            worst = max(worst, d)
            count += 1
            print(f"{lane['id']:40s} source {ref.get('source', '?'):16s} |pole - analytic| / scale {d:.1e}")
    print(f"{count} lanes, worst {worst:.2e}")


if __name__ == "__main__":
    main()
