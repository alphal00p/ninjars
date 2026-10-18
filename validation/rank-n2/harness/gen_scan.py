#!/usr/bin/env python3
"""Conditioning-scan cases (DESIGN.md V9, report only) with exact dyadic inputs.

  rn2-scan-bubp2-{real,complex}-n2-r4[-mu2]   bubble, p = (5a, 0, 0, 3a) with p^2 = 16 a^2 =
        m0 2^-k, k = 0, 4, 8, ..., 24, plus p = 0 (lanes in that order); m0 = 173^2
  rn2-scan-gram-n3-r5                          triangle, p1 = (10,0,0,6), p2 = (5,0,d,3):
        Gram det = -64 d^2, d = 2^-k, k = 0, 2, ..., 12
  rn2-scan-gram-n4-r6                          box, p3 = (p1 + p2)/2 + (0,0,0,0) + d e_y-ish:
        the third momentum approaches the plane of p1, p2 (normalised Gram ~ d^2)
Complex masses throughout except the real bubble; mu_R^2 = 64 (bubbles 10^4).

    gen_scan.py --out validation/rank-n2/scan"""
import argparse
import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collier_reference as cr  # noqa: E402
from gen_identity import rand_poly, tensor_v2, LAYOUT_V2  # noqa: E402
from gen_suite import exact_s_matrix  # noqa: E402
from conditioning import gram_min  # noqa: E402


def smat(shifts):
    n = len(shifts)
    out = []
    for i in range(n):
        row = []
        for j in range(n):
            d = [cr.Q(a) - cr.Q(b) for a, b in zip(shifts[i], shifts[j])]
            v = cr.mdot(d, d)
            assert cr.Q(float(v)) == v
            row.append(float(v))
        out.append(row)
    return out


def doc(name, n, rank, lanes, desc, K=0, extra=None):
    d = {"format": "ninjars-bench-case-v1", "name": name, "n": n, "rank": rank}
    if K:
        d.update({"format": "ninjars-bench-case-v2", "mu_powers": K, "r2": "synthetic", "tensor_layout": LAYOUT_V2})
    d.update({"description": desc, "generator": "validation/rank-n2/harness/gen_scan.py", "exact_binary64": True})
    if extra:
        d.update(extra)
    d["lanes"] = lanes
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    cdir = a.out / "cases"
    cdir.mkdir(parents=True, exist_ok=True)
    docs = []
    m0 = 173.0 ** 2
    for masses, m2 in (("real", [[m0, 0.0], [m0, 0.0]]), ("complex", [[m0, -259.5], [6400.0, -168.0]])):
        for mu in (False, True):
            rng = random.Random(f"scan-bub-{masses}-{mu}")
            lanes, ks = [], []
            for k in list(range(0, 25, 4)) + [None]:
                if k is None:
                    p = [0.0, 0.0, 0.0, 0.0]
                else:
                    aa = 173.0 / 4 * 2.0 ** (-k / 2)
                    p = [5 * aa, 0.0, 0.0, 3 * aa]
                shifts = [[0.0] * 4, p]
                blocks = [(0, rand_poly(rng, 4, wbits=7))] + ([(1, rand_poly(rng, 2, wbits=7)), (2, rand_poly(rng, 0))] if mu else [])
                t, K = tensor_v2(blocks, 4) if mu else (cr.poly_to_tensor(blocks[0][1], 4), 0)
                lanes.append({"id": f"rn2-scan-bubp2-{masses}-n2-r4{'-mu2' if mu else ''}-k{'inf' if k is None else k}",
                              "shifts": shifts, "masses_squared": m2, "mu_squared": 1.0e4, "s_matrix": smat(shifts),
                              "tensor": t})
                ks.append(k)
            name = f"rn2-scan-bubp2-{masses}-n2-r4{'-mu2' if mu else ''}"
            docs.append(doc(name, 2, 4, lanes, "V9: bubble p^2 = m0 2^-k ladder (lane order k = 0, 4, ..., 24, p = 0)",
                            2 if mu else 0, {"scan": {"parameter": "p^2/m0", "k": ks}}))
    m3 = [[25.0, -1.25], [49.0, -0.5], [36.0, -1.5], [16.0, -0.75]]
    for n, rank in ((3, 5), (4, 6)):
        rng = random.Random(f"scan-gram-n{n}")
        lanes, gs = [], []
        for k in range(0, 13, 2):
            d = 2.0 ** (-k)
            p1 = [10.0, 0.0, 0.0, 6.0]
            p2 = [5.0, 0.0, d, 3.0] if n == 3 else [6.0, 2.0, 0.0, 1.0]
            legs = [p1, p2]
            if n == 4:
                legs.append([8.0, 1.0, d, 3.5])  # (p1 + p2)/2 + d e_y
            shifts = [[0.0] * 4]
            for p in legs:
                shifts.append([x + y for x, y in zip(shifts[-1], p)])
            g = gram_min(shifts)[0]
            t = cr.poly_to_tensor(rand_poly(rng, rank), rank)
            lanes.append({"id": f"rn2-scan-gram-n{n}-r{rank}-k{k}", "shifts": shifts, "masses_squared": m3[:n],
                          "mu_squared": 64.0, "s_matrix": smat(shifts), "tensor": t})
            gs.append(g)
        docs.append(doc(f"rn2-scan-gram-n{n}-r{rank}", n, rank, lanes,
                        "V9: Gram-determinant ladder, d = 2^-k (lane order k = 0, 2, ..., 12)", 0,
                        {"scan": {"parameter": "min normalised Gram determinant", "values": gs}}))
    for dd in docs:
        (cdir / f"{dd['name']}.json").write_text(json.dumps(dd, indent=1) + "\n")
    print(f"{len(docs)} scan cases")


if __name__ == "__main__":
    main()
