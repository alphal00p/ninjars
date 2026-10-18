#!/usr/bin/env python3
"""Referee check (3)+(4): high-precision (mpmath) reduction of a rank n+2 reference-suite
lane with the theory recipe, assembled with the theory's integrated values; bubble/tadpole
masters computed here from Feynman parameters (mpmath), scalar C0/D0 from COLLIER (quad)
through the reference harness (rank-0 numerator).  Compared with the COLLIER rank-(n+2)
tensor reference of the suite.

usage: e2e.py CASE.json REF.json [lanes]"""
import itertools
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import mpmath as mp
import red
from red import FieldMp, set_field, Numerator, Reduction, dot, vsub

HERE = pathlib.Path(__file__).resolve().parent
# The COLLIER reference harness (phase-1 `reference/harness/collier_reference.py`).
HARNESS = pathlib.Path(os.environ.get('NINJARS_COLLIER_HARNESS', HERE.parents[2] / 'validation' / 'rank-n2' / 'harness' / 'collier_reference.py'))


def monomials(rank):
    allm = [(0, 0, 0, 0)]
    lower = [(0, 0, 0, 0)]
    for r in range(1, rank + 1):
        r1, r2, r3 = r, r * (r + 1) // 2, r * (r + 1) * (r + 2) // 6

        def bump(m, k):
            m = list(m)
            m[k] += 1
            return tuple(m)
        block = [bump(lower[0], 0)]
        block += [bump(m, 1) for m in lower[:r1]]
        block += [bump(m, 2) for m in lower[:r2]]
        block += [bump(m, 3) for m in lower[:r3]]
        allm += block
        lower = block
    return allm


def rank_of(length):
    r = 0
    while (r + 1) * (r + 2) * (r + 3) * (r + 4) // 24 < length:
        r += 1
    return r


def build_numerator(lane, F):
    terms = []
    T = lane['tensor']
    r0 = rank_of(len(T))
    for e, c in zip(monomials(r0), T):
        terms.append((e, 0, F.num(complex(c[0], c[1]))))
    rank = r0
    for ex in lane.get('mu2_tensors', []) or []:
        k = int(ex['power'])
        rk = rank_of(len(ex['tensor']))
        for e, c in zip(monomials(rk), ex['tensor']):
            terms.append((e, k, F.num(complex(c[0], c[1]))))
        rank = max(rank, rk + 2 * k)
    return Numerator(terms, rank)


# ----------------------------------------------------------------- masters (eps^0, eps^-1, eps^-2)

def A0(m, mu2):
    if m == 0:
        return [mp.mpc(0), mp.mpc(0), mp.mpc(0)]
    return [m * (1 - mp.log(m / mu2)), m, mp.mpc(0)]


def Bn(a, s, m1, m2, mu2):
    """B_{1^a} (OneLOop convention): (-1)^a int_0^1 x^a [1/eps - log(Delta/mu2)]."""
    eps_i0 = mp.mpc(0, mp.mpf(10) ** -45) * (abs(s) + abs(m1) + abs(m2))
    D = lambda x: -x * (1 - x) * s + (1 - x) * m1 + x * m2 - eps_i0
    # breakpoints at the zeros of Re(Delta) inside (0,1)
    A, B, C = mp.re(s), mp.re(m2 - m1 - s), mp.re(m1)
    pts = [mp.mpf(0)]
    if A != 0:
        disc = B * B - 4 * A * C
        if disc >= 0:
            for rt in sorted([(-B - mp.sqrt(disc)) / (2 * A), (-B + mp.sqrt(disc)) / (2 * A)]):
                if 0 < rt < 1:
                    pts.append(rt)
    pts.append(mp.mpf(1))
    val = mp.quad(lambda x: x ** a * mp.log(D(x) / mu2), pts)
    sg = (-1) ** a
    return [-sg * val, mp.mpc(sg) / (a + 1), mp.mpc(0)]


def collier_scalar(lane, props, workdir):
    """Scalar n-point integral of the propagator subset via COLLIER quad (rank-0 numerator)."""
    sub = {
        'format': 'ninjars-bench-case-v1', 'name': 'scalar-' + '-'.join(map(str, props)), 'n': len(props),
        'rank': 0,
        'lanes': [{
            'id': 'x',
            'shifts': [lane['shifts'][i] for i in props],
            'masses_squared': [lane['masses_squared'][i] for i in props],
            'mu_squared': lane['mu_squared'],
            's_matrix': [[lane['s_matrix'][i][j] for j in props] for i in props],
            'tensor': [[1.0, 0.0]],
        }]}
    p = pathlib.Path(workdir) / (sub['name'] + '.json')
    p.write_text(json.dumps(sub))
    outd = pathlib.Path(workdir) / 'out'
    subprocess.run([sys.executable, '-I', str(HARNESS), str(p), '--build', 'quad', '--mode', os.environ.get('E2E_MODE', '1'), '--out-dir', str(outd)],
                   check=True, capture_output=True, text=True)
    res = json.loads((outd / (sub['name'] + '.collier.json')).read_text())['lanes'][0]
    dec = res['laurent_decimal']
    return [mp.mpc(mp.mpf(dec[k][0]), mp.mpf(dec[k][1])) for k in ('eps0', 'eps_m1', 'eps_m2')]


def add(acc, c, v):
    for i in range(3):
        acc[i] += c * v[i]


def main():
    case = json.loads(pathlib.Path(sys.argv[1]).read_text())
    ref = json.loads(pathlib.Path(sys.argv[2]).read_text())
    lanes = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else range(len(case['lanes']))
    n = case['n']
    F = FieldMp(seed=11, dps=60)
    set_field(F)
    workdir = tempfile.mkdtemp(prefix='e2e-', dir=str(HERE))
    for li in lanes:
        lane = case['lanes'][li]
        V = [tuple(F.num(float(x)) for x in v) for v in lane['shifts']]
        m = [F.num(complex(c[0], c[1])) for c in lane['masses_squared']]
        mu2R = mp.mpf(lane['mu_squared'])
        N = build_numerator(lane, F)
        R = Reduction(n, V, m, N)
        summ = R.run(check_fits=True, log=print)
        worst = max((max(w, d) for v in summ.values() for _, w, d in v), default=0)
        tot = [mp.mpc(0)] * 3
        tot = [mp.mpc(0), mp.mpc(0), mp.mpc(0)]
        new = [mp.mpc(0), mp.mpc(0), mp.mpc(0)]  # contributions of the new rank-(n+2) terms only
        wcoll = mp.mpf(0)  # sum |c0 * (C0 or D0 from COLLIER)| (error budget of the external masters)
        absall = mp.mpf(0)
        part = {'box_mu6': mp.mpc(0), 'B1111': mp.mpc(0), 'bub_mu2qe2^2': mp.mpc(0), 'bub_mu4': mp.mpc(0)}
        sij = lambda i, j: dot(vsub(V[i], V[j]), vsub(V[i], V[j]))
        for S, res in R.res.items():
            k = len(S)
            c = res.c
            if k == 4:
                D0 = collier_scalar(lane, S, workdir)
                add(tot, c[(0, (0,))], D0)
                wcoll += abs(c[(0, (0,))] * D0[0])
                tot[0] += -c.get((2, (0,)), 0) / 6
                rat6 = sum(m[i] for i in S) / 12 - sum(sij(i, j) for i, j in itertools.combinations(S, 2)) / 60
                tot[0] += c.get((3, (0,)), 0) * rat6
                new[0] += c.get((3, (0,)), 0) * rat6
                part['box_mu6'] += c.get((3, (0,)), 0) * rat6
            elif k == 3:
                C0 = collier_scalar(lane, S, workdir)
                add(tot, c[(0, (0, 0))], C0)
                wcoll += abs(c[(0, (0, 0))] * C0[0])
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
                    b = Bn(a, s, m1, m2, mu2R)
                    add(tot, ca * ke2 ** a, b)
                    if a == 4:
                        add(new, ca * ke2 ** a, b)
                        part['B1111'] += ca * ke2 ** a * b[0]
                tot[0] += c.get((1, (0, 0, 0)), 0) * (3 * (m1 + m2) - s) / 6
                tot[0] += c.get((1, (1, 0, 0)), 0) * ke2 * (s - 2 * m1 - 4 * m2) / 12
                r21 = c.get((1, (2, 0, 0)), 0) * ke2 ** 2 * (5 * m1 + 15 * m2 - 3 * s) / 60
                r22 = -c.get((2, (0, 0, 0)), 0) * (s * s - 5 * s * (m1 + m2) + 10 * (m1 * m1 + m1 * m2 + m2 * m2)) / 60
                tot[0] += r21 + r22
                new[0] += r21 + r22
                part['bub_mu2qe2^2'] += r21
                part['bub_mu4'] += r22
            elif k == 1:
                i = S[0]
                rc = R.tadpole_recipe(S, R.tadbasis_common)
                mm = m[i]
                coefA0 = rc['c0'] + rc['c15'] * rc['mp34'] * mm / 4
                add(tot, coefA0, A0(mm, mu2R))
                tot[0] += mm * mm / 2 * (rc['c14'] + rc['c15'] * rc['mp34'] / 4)
        rl = ref['lanes'][li]
        refv = [mp.mpc(mp.mpf(rl['laurent_decimal'][kk][0]), mp.mpf(rl['laurent_decimal'][kk][1]))
                for kk in ('eps0', 'eps_m1', 'eps_m2')]
        scale = max(abs(x) for x in refv)
        diff = [abs(a - b) / scale for a, b in zip(tot, refv)]
        print(f"{case['name']} lane {li}: tier {rl.get('tier')} unc {rl.get('uncertainty')} | recipe-vs-fit worst {worst:.1e}")
        print(f"   ours  eps0 = {mp.nstr(tot[0], 22)}  eps-1 = {mp.nstr(tot[1], 18)}  eps-2 = {mp.nstr(tot[2], 6)}")
        print(f"   ref   eps0 = {mp.nstr(refv[0], 22)}  eps-1 = {mp.nstr(refv[1], 18)}  eps-2 = {mp.nstr(refv[2], 6)}")
        print(f"   rel diff (to lane scale) eps0 {mp.nstr(diff[0], 3)}  eps-1 {mp.nstr(diff[1], 3)}  eps-2 {mp.nstr(diff[2], 3)};"
              f"  |new rank-n+2 terms|/scale = {mp.nstr(abs(new[0]) / scale, 3)}", flush=True)
        print(f'   sum|c0*M| over COLLIER scalar masters / scale = {mp.nstr(wcoll/scale, 3)}')
        print('   new-term parts / scale: ' + ', '.join(f'{k} {mp.nstr(abs(v)/scale, 3)}' for k, v in part.items()))


if __name__ == "__main__":
    main()
