#!/usr/bin/env python3
"""Integrand mu^2 handling of the harness against exact closed forms.

Dimension shift gives, for k propagators and b >= 1 (normalisation of collier_reference.py),
    int (mu^2)^b / (D_0 .. D_(k-1)) = (-1)^(b+1) (b-1)! / (b+2-k)!  int_simplex Delta^(b+2-k)
for b + 2 - k >= 0 and 0 (O(eps)) otherwise, with Delta = sum_i x_i m_i^2 - sum_(i<j) x_i x_j
(V_i - V_j)^2 on the simplex sum x_i = 1 (Dirichlet: int prod x_i^(a_i) = prod a_i! /
(k - 1 + sum a_i)!), no poles.  Special cases are Ninja's coded rational terms (m^4/2,
(m0+m1)/2 - p^2/6, 1/2, -1/6) and the new rank n+2 ones (box mu^6: sum m/12 - sum s/60;
bubble mu^4; triangle mu^4).  Plus int mu^2 q^mu / (D0 D1 D2) = -(V1 + V2 - 2 V0)^mu/6 - V0^mu/2.
Every (k, b) with k = 1..6, b = 1..4 is evaluated as a v2 case (T_b = 1) with the quad
COLLIER build (exact inputs) and, for k <= 2, the mp backend; the closed form is exact
rational arithmetic.  Prints the worst scaled deviation.

    test_mu2.py [--build quad|mp]"""
import argparse
import itertools
import sys
from decimal import Decimal, getcontext
from math import factorial

import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collier_reference as cr  # noqa: E402

getcontext().prec = 60
Q = cr.Q
V = [[0, 0, 0, 0], [13, 2, -4, 7], [31, -6, 1, 15], [42, 3, 9, 11], [52, -1, 6, 30], [61, 5, -2, 24]]
M2 = [[299.0, -2.5], [156.25, 0.0], [83.0, -2.25], [0.0, 0.0], [64.5, -1.75], [100.0, -0.5]]
W = (7, -3, 5, 2)


def dirichlet(a):
    k = len(a)
    num = 1
    for x in a:
        num *= factorial(x)
    return Q(num, factorial(k - 1 + sum(a)))


def simplex_delta_power(k, p, m2, V):
    """int_simplex Delta^p (exact complex rational)."""
    terms = {}
    lin = {}
    for i in range(k):
        e = [0] * k
        e[i] = 1
        lin[tuple(e)] = cr.cq(m2[i][0], m2[i][1])
    for i, j in itertools.combinations(range(k), 2):
        d = [Q(x) - Q(y) for x, y in zip(V[i], V[j])]
        e = [0] * k
        e[i] += 1
        e[j] += 1
        lin[tuple(e)] = cr.cadd(lin.get(tuple(e), cr.ZERO), cr.cq(-cr.mdot(d, d)))
    poly = {tuple([0] * k): cr.cq(1)}
    for _ in range(p):
        new = {}
        for ea, ca in poly.items():
            for eb, cb in lin.items():
                e = tuple(x + y for x, y in zip(ea, eb))
                new[e] = cr.cadd(new.get(e, cr.ZERO), cr.cmul(ca, cb))
        poly = new
    tot = cr.ZERO
    for e, c in poly.items():
        tot = cr.cadd(tot, cr.cscale(c, dirichlet(e)))
    return tot


def closed(k, b, shift0):
    p = b + 2 - k
    if p < 0:
        return cr.ZERO
    Vs = [[x + s for x, s in zip(v, shift0)] for v in V[:k]]
    return cr.cscale(simplex_delta_power(k, p, M2, Vs), Q((-1) ** (b + 1) * factorial(b - 1), factorial(p)))


def case_for(k, b, shift0, mono=(0, 0, 0, 0)):
    """v2 case: N = mu^(2b) q^mono, propagators 0..k-1 shifted by shift0."""
    deg = sum(mono)
    rank = 2 * b + deg
    blocks = []
    for kk in range(b + 1):
        r = rank - 2 * kk
        t = [[0.0, 0.0] for _ in range(cr.ncoeff(r))]
        if kk == b:
            t[cr.monomials(r).index(tuple(mono))] = [1.0, 0.0]
        blocks += t
    lane = {"id": f"k{k}-b{b}-{mono}-{shift0}", "shifts": [[float(x + s) for x, s in zip(v, shift0)] for v in V[:k]],
            "masses_squared": M2[:k], "mu_squared": 1.0e4, "s_matrix": None, "tensor": blocks}
    return {"format": "ninjars-bench-case-v2", "name": lane["id"], "n": k, "rank": rank, "mu_powers": b,
            "lanes": [lane]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default="quad")
    a = ap.parse_args()
    tests = []
    for k in range(1, 7):
        for b in range(1, 5):
            if a.build == "mp" and k > 2:
                continue
            for s0 in ((0, 0, 0, 0), W):
                tests.append((case_for(k, b, s0), closed(k, b, s0)))
    for mu in range(4):
        e = [0, 0, 0, 0]
        e[mu] = 1
        for s0 in ((0, 0, 0, 0), W):
            if a.build == "mp":
                continue
            expect = Q(-(V[1][mu] + V[2][mu]), 6) - Q(s0[mu], 2)
            tests.append((case_for(3, 1, s0, tuple(e)), cr.cq(expect)))
    worst = 0.0
    for case, expect in tests:
        r = cr.evaluate_case(case, build=a.build)[0]
        got = [(Decimal(r["laurent_decimal"][k][0]), Decimal(r["laurent_decimal"][k][1])) for k in cr.KEYS]
        ex = (cr.qdec(expect[0]), cr.qdec(expect[1]))
        sc = max(abs(ex[0]) + abs(ex[1]), Decimal(1))
        d = max(abs(got[0][0] - ex[0]) + abs(got[0][1] - ex[1]),
                abs(got[1][0]) + abs(got[1][1]), abs(got[2][0]) + abs(got[2][1])) / sc
        worst = max(worst, float(d))
        print(f"{case['name']:44s} expected {float(ex[0]):+.6e}{float(ex[1]):+.6e}i  deviation {float(d):.1e}")
    print(f"{len(tests)} integrals, worst scaled deviation {worst:.2e} ({a.build})")


if __name__ == "__main__":
    main()
