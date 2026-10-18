#!/usr/bin/env python3
"""60-digit integrand reduction of one bench case (review e2e driver, patched).

Provenance: SP/review/e2e.py of the rank n+2 referee (scratchpad
rank-n2/review/, sha256 of the original below in README).  The reduction itself is
red.py, copied verbatim (sha256 a03275f3659320db...): a from-scratch Ninja-style
reduction for any rank in mpmath (60 digits), with residues checked against an
independent fit on every cut ("recipe_vs_fit").  Integrated with the theory's values;
masters: A0 closed form, B_{1^a} from Feynman parameters (mpmath, Bn below), scalar
C0/D0 from collier_reference.py (COLLIER quad; E2E_MODE=1 COLI, 2 DD) with exact inputs,
or, with $E2E_SCALARS set, from a table of exact values (scalar_cases.py sub-cases).

Patches against the review's e2e.py (most are marked "patch" in the code):
  - HARNESS points at this harness (exact driver inputs; the original passed repr(float));
  - v2 bench cases are accepted;
  - per-lane values go to $E2E_OUT as 40-digit decimals; the reference argument may be '-';
  - temporary files go to $E2E_TMP (else the system temp dir) and are removed;
  - $E2E_DPS sets the working precision (default 60);
  - scaleless bubbles return 0 (the original returned NaN);
  - B_n uses the exact -i0 prescription instead of a 1e-45 imaginary shift;
  - lanes whose value vanishes exactly (massless tadpole) no longer divide by zero;
  - $E2E_SCALARS takes C0/D0 from a table instead of COLLIER.

usage: e2e_reduction.py CASE.json REF.json|- [lanes]"""
import itertools
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import mpmath as mp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import red  # noqa: E402
from red import FieldMp, set_field, Numerator, Reduction, dot, vsub

HERE = pathlib.Path(__file__).resolve().parent
HARNESS = HERE.parent / 'collier_reference.py'


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


def build_numerator(lane, F, case=None):
    terms = []
    if case is not None and case.get('format') == 'ninjars-bench-case-v2':
        K, R0 = case['mu_powers'], case['rank']
        T, off, ext = lane['tensor'], 0, []
        nc = lambda r: (r + 1) * (r + 2) * (r + 3) * (r + 4) // 24
        blocks = []
        for k in range(K + 1):
            blocks.append(T[off:off + nc(R0 - 2 * k)]); off += nc(R0 - 2 * k)
        lane = dict(lane, tensor=blocks[0], mu2_tensors=[{'power': k, 'tensor': blocks[k]} for k in range(1, K + 1)])
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
    if s == 0 and m1 == 0 and m2 == 0:  # scaleless (patch: the review's version returned NaN)
        return [mp.mpc(0), mp.mpc(0), mp.mpc(0)]
    # patch: exact -i0 prescription; the review shifted Delta by -1e-45 i |scale|, which costs
    # ~1e-21 at on-shell thresholds (Delta = x^2 M^2, irsoft lanes)
    D = lambda x: -x * (1 - x) * s + (1 - x) * m1 + x * m2

    def logm(z):
        if mp.im(z) == 0 and mp.re(z) < 0:
            return mp.log(-mp.re(z)) - mp.pi * 1j
        return mp.log(z)
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
    val = mp.quad(lambda x: x ** a * logm(D(x) / mu2), pts)
    sg = (-1) ** a
    return [-sg * val, mp.mpc(sg) / (a + 1), mp.mpc(0)]


_SCALARS = None
WORKDIR = None


def table_scalar(lane, props):
    """Scalar C0/D0 from $E2E_SCALARS (directory of generic-engine runs of scalar_cases.py
    sub-cases, lane ids "<lane id>#<sorted props>"); patch, see the module docstring."""
    global _SCALARS
    if _SCALARS is None:
        _SCALARS = {}
        for f in sorted(pathlib.Path(os.environ['E2E_SCALARS']).glob('*.json')):
            for l in json.loads(f.read_text())['lanes']:
                if 'laurent_decimal' in l:
                    _SCALARS[l['id']] = [mp.mpc(mp.mpf(l['laurent_decimal'][k][0]), mp.mpf(l['laurent_decimal'][k][1]))
                                         for k in ('eps0', 'eps_m1', 'eps_m2')]
    key = f"{lane['id']}#{','.join(map(str, sorted(props)))}"
    if key not in _SCALARS:
        raise KeyError(f'scalar master {key} missing in $E2E_SCALARS')
    return _SCALARS[key]


def collier_scalar(lane, props, workdir):
    if os.environ.get('E2E_SCALARS'):
        return table_scalar(lane, props)
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
    res = json.loads((outd / (sub['name'] + '.json')).read_text())['lanes'][0]
    dec = res['laurent_decimal']
    return [mp.mpc(mp.mpf(dec[k][0]), mp.mpf(dec[k][1])) for k in ('eps0', 'eps_m1', 'eps_m2')]


def add(acc, c, v):
    for i in range(3):
        acc[i] += c * v[i]


def main():
    case = json.loads(pathlib.Path(sys.argv[1]).read_text())
    ref = json.loads(pathlib.Path(sys.argv[2]).read_text()) if sys.argv[2] != '-' else None
    lanes = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else range(len(case['lanes']))
    n = case['n']
    F = FieldMp(seed=11, dps=int(os.environ.get('E2E_DPS', '60')))  # patch: E2E_DPS
    set_field(F)
    global WORKDIR
    workdir = WORKDIR = tempfile.mkdtemp(prefix='e2e-', dir=os.environ.get('E2E_TMP') or None)
    outlanes = []
    for li in lanes:
        lane = case['lanes'][li]
        V = [tuple(F.num(float(x)) for x in v) for v in lane['shifts']]
        m = [F.num(complex(c[0], c[1])) for c in lane['masses_squared']]
        mu2R = mp.mpf(lane['mu_squared'])
        N = build_numerator(lane, F, case)
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
        lane_scale = max(abs(x) for x in tot) or mp.mpf(1)  # patch: exactly vanishing lanes
        outlanes.append({'id': lane.get('id'), 'laurent_decimal': {kk: [mp.nstr(mp.re(tot[i]), 40), mp.nstr(mp.im(tot[i]), 40)] for i, kk in enumerate(('eps0', 'eps_m1', 'eps_m2'))},
                         'recipe_vs_fit': float(worst), 'collier_scalar_weight': float(wcoll / lane_scale),
                         'new_terms_over_scale': float(abs(new[0]) / lane_scale),
                         'parts_over_scale': {k: float(abs(v) / lane_scale) for k, v in part.items()}})
        if os.environ.get('E2E_OUT'):
            pathlib.Path(os.environ['E2E_OUT']).write_text(json.dumps({'case': case['name'], 'lanes': outlanes}, indent=1))
        if ref is None:
            continue
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
    try:
        main()
    finally:
        import shutil
        if WORKDIR:
            shutil.rmtree(WORKDIR, ignore_errors=True)
