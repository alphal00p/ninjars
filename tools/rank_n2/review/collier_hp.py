#!/usr/bin/env python3
"""Re-run the reference harness (COLLIER quad, unchanged driver) but with the polynomial
pre-processing (mu^2 removal, loop-momentum shifts) and the contraction done in 60-digit
mpmath instead of binary64, to measure the harness' own preprocessing error.
usage: collier_hp.py CASE.json OUT.json"""
import importlib.util
import json
import os
import pathlib
import sys
from decimal import Decimal
from math import comb

import mpmath as mp

mp.mp.dps = 60
HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cref', os.environ.get('NINJARS_COLLIER_HARNESS', HERE.parents[2] / 'validation' / 'rank-n2' / 'harness' / 'collier_reference.py'))
cref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cref)


def tensor_to_poly(tensor):
    mons = cref.monomials(cref.rank_of_tensor(len(tensor)))
    poly = {}
    for m, c in zip(mons, tensor):
        z = mp.mpc(mp.mpf(c[0]), mp.mpf(c[1]))
        if z != 0:
            poly[m] = poly.get(m, 0) + z
    return poly


def a_poly(v, m2):
    v = [mp.mpf(x) for x in v]
    p = {(2, 0, 0, 0): mp.mpf(1), (0, 2, 0, 0): mp.mpf(-1), (0, 0, 2, 0): mp.mpf(-1), (0, 0, 0, 2): mp.mpf(-1)}
    met = (1, -1, -1, -1)
    for mu in range(4):
        if v[mu] != 0:
            e = [0, 0, 0, 0]
            e[mu] = 1
            p[tuple(e)] = p.get(tuple(e), 0) + 2 * met[mu] * v[mu]
    p[(0, 0, 0, 0)] = p.get((0, 0, 0, 0), 0) + (v[0] ** 2 - v[1] ** 2 - v[2] ** 2 - v[3] ** 2) \
        - mp.mpc(mp.mpf(m2.real), mp.mpf(m2.imag))
    return p


def pshift(poly, v):
    if all(x == 0 for x in v):
        return dict(poly)
    v = [mp.mpf(x) for x in v]
    out = {}
    for m, c in poly.items():
        terms = [((0, 0, 0, 0), c)]
        for mu in range(4):
            e = m[mu]
            if e == 0:
                continue
            new = []
            for k in range(e + 1):
                f = comb(e, k) * (-v[mu]) ** (e - k)
                if f == 0:
                    continue
                for mm, cc in terms:
                    t = list(mm)
                    t[mu] += k
                    new.append((tuple(t), cc * f))
            terms = new
        for mm, cc in terms:
            out[mm] = out.get(mm, 0) + cc
    return out


def contract(job, table):
    acc = [[Decimal(0), Decimal(0)] for _ in range(4)]
    for m, c in job["poly"].items():
        c = mp.mpc(c)
        cr, ci = Decimal(mp.nstr(c.real, 60)), Decimal(mp.nstr(c.imag, 60))
        t = table["T"][m]
        for k in range(4):
            tr, ti = t[k]
            acc[k][0] += cr * tr - ci * ti
            acc[k][1] += cr * ti + ci * tr
    return [complex(float(a[0]), float(a[1])) for a in acc] + [0.0, [0.0, 0.0, 0.0]], acc


cref.tensor_to_poly = tensor_to_poly
cref.a_poly = a_poly
cref.pshift = pshift
cref.contract = contract
cref.getcontext().prec = 80

case = json.loads(pathlib.Path(sys.argv[1]).read_text())
# exact invariants are irrelevant for 'int' lanes (s_matrix exact); keep the harness default
lanes = cref.evaluate_case(case, mode=1, build='quad')
pathlib.Path(sys.argv[2]).write_text(json.dumps({'case': case['name'], 'lanes': lanes}, indent=1))
for l in lanes:
    print(l['id'], l['laurent_decimal']['eps0'], l['laurent_decimal']['eps_m1'])
