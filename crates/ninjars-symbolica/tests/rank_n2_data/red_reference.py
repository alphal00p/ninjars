#!/usr/bin/env python3
"""Independent 60-digit reference for the rank n+2 smoke cases.

The review's reduction (`tools/rank_n2/review/red.py`, shares no code with
NinjaRS) assembled with the theory's integrated values as in
`tools/rank_n2/review/e2e.py`: A0 in closed form and B0..B1111 from Feynman
parameters (mpmath quadrature), but scalar C0/D0 read from MASTERS.json
(oneloopmaster's exact expressions at 256 bits, written by the ignored test
`rank_n2_dump_scalar_masters` in `tests/rank_n2_integration.rs`) instead of
COLLIER. COLLIER's quad-precision tensor integrals (the `hpref` references)
are only accurate to about 1e-17..1e-14 relative on some lanes (its A0 and
tensor coefficients; see `tools/rank_n2/README.md`), which hides the
reduction's own accuracy; with exact masters this reference tests the
reduction alone.

usage: red_reference.py CASE.json MASTERS.json OUT.json
(run with tools/rank_n2/review/py.sh, with tools/rank_n2/review on PYTHONPATH)
"""
import itertools
import json
import pathlib
import sys

import mpmath as mp

import e2e
from red import FieldMp, set_field, Reduction, dot, vsub


def main():
    case = json.loads(pathlib.Path(sys.argv[1]).read_text())
    masters = json.loads(pathlib.Path(sys.argv[2]).read_text())[case['name']]
    n = case['n']
    F = FieldMp(seed=11, dps=60)
    set_field(F)
    lanes = []
    for li, lane in enumerate(case['lanes']):
        V = [tuple(F.num(float(x)) for x in v) for v in lane['shifts']]
        m = [F.num(complex(c[0], c[1])) for c in lane['masses_squared']]
        mu2R = mp.mpf(lane['mu_squared'])
        N = e2e.build_numerator(lane, F)
        R = Reduction(n, V, m, N)
        summary = R.run(check_fits=True, log=print)
        worst = max((max(w, d) for v in summary.values() for _, w, d in v), default=0)
        scalar = lambda S: [mp.mpc(mp.mpf(z[0]), mp.mpf(z[1]))
                            for z in masters[li][','.join(map(str, S))]]
        tot = [mp.mpc(0), mp.mpc(0), mp.mpc(0)]
        sij = lambda i, j: dot(vsub(V[i], V[j]), vsub(V[i], V[j]))
        for S, res in R.res.items():
            k = len(S)
            c = res.c
            if k == 4:
                e2e.add(tot, c[(0, (0,))], scalar(S))
                tot[0] += -c.get((2, (0,)), 0) / 6
                rat6 = sum(m[i] for i in S) / 12 - sum(sij(i, j) for i, j in itertools.combinations(S, 2)) / 60
                tot[0] += c.get((3, (0,)), 0) * rat6
            elif k == 3:
                e2e.add(tot, c[(0, (0, 0))], scalar(S))
                tot[0] += c.get((1, (0, 0)), 0) / 2
                tot[0] += c.get((2, (0, 0)), 0) * (sum(sij(i, j) for i, j in itertools.combinations(S, 2)) / 24
                                                    - sum(m[i] for i in S) / 6)
            elif k == 2:
                i1, i2 = S
                s = sij(i2, i1)
                m1, m2 = m[i1], m[i2]
                ke2 = dot(vsub(V[i2], V[i1]), R.e2ref)
                for a in range(R.R(2) + 1):
                    ca = c.get((0, (a, 0, 0)), 0)
                    if ca == 0:
                        continue
                    e2e.add(tot, ca * ke2 ** a, e2e.Bn(a, s, m1, m2, mu2R))
                tot[0] += c.get((1, (0, 0, 0)), 0) * (3 * (m1 + m2) - s) / 6
                tot[0] += c.get((1, (1, 0, 0)), 0) * ke2 * (s - 2 * m1 - 4 * m2) / 12
                tot[0] += c.get((1, (2, 0, 0)), 0) * ke2 ** 2 * (5 * m1 + 15 * m2 - 3 * s) / 60
                tot[0] -= c.get((2, (0, 0, 0)), 0) * (s * s - 5 * s * (m1 + m2) + 10 * (m1 * m1 + m1 * m2 + m2 * m2)) / 60
            elif k == 1:
                i = S[0]
                rc = R.tadpole_recipe(S, R.tadbasis_common)
                mm = m[i]
                e2e.add(tot, rc['c0'] + rc['c15'] * rc['mp34'] * mm / 4, e2e.A0(mm, mu2R))
                tot[0] += mm * mm / 2 * (rc['c14'] + rc['c15'] * rc['mp34'] / 4)
        print(f"{case['name']} lane {li}: recipe-vs-fit worst {worst:.1e}; eps0 = {mp.nstr(tot[0], 30)}", flush=True)
        lanes.append({
            'id': lane['id'],
            'recipe_vs_fit': worst,
            'laurent_decimal': {key: [mp.nstr(z.real, 50), mp.nstr(z.imag, 50)]
                                for key, z in zip(('eps0', 'eps_m1', 'eps_m2'), tot)},
        })
    out = {
        'format': 'ninjars-rank-n2-red-reference-v1',
        'case': case['name'],
        'method': 'tools/rank_n2/review/red.py reduction (60 digits, residues checked against an '
                  'independent fit), A0 closed form, B0..B1111 Feynman-parameter quadrature, exact '
                  'oneloopmaster C0/D0 (256 bits)',
        'lanes': lanes,
    }
    pathlib.Path(sys.argv[3]).write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
