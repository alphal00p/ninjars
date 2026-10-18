#!/usr/bin/env python3
"""Independent check of theory/subtractions_rank_n2_cse.txt (the code meant to be ported):
evaluate the CSE expressions exactly in F_p[i] on kinematics built by red.py and compare
with the definition (Laurent series of new parent monomials / uncut denominators)."""
import re
import sys
import itertools
from fractions import Fraction
import red
from red import (FieldFp, set_field, Numerator, Reduction, Residue, dot, vadd, vsub, vsc, ldiv)

CSE = sys.argv[1]


def parse(path):
    secs = {}
    cur = None
    for line in open(path):
        line = line.strip()
        m = re.match(r'// -+ (\w+):', line)
        if m:
            cur = m.group(1)
            secs[cur] = []
            continue
        if not line or line.startswith('//') or cur is None:
            continue
        line = line.rstrip(';')
        line = re.sub(r'(?<![\w.])(\d+)\.0\b', r'Fraction(\1)', line)
        line = re.sub(r'\b(?<!\.)(\d+)\b(?!\))', lambda mm: mm.group(0), line)
        secs[cur].append(line)
    return secs


def run_section(lines, env):
    env = dict(env)
    env['Fraction'] = Fraction
    out = {}
    env['out'] = out

    class O(dict):
        def __setitem__(self, k, v):
            dict.__setitem__(self, tuple(k), v)
    env['out'] = O()
    for ln in lines:
        lhs, rhs = ln.split('=', 1)
        lhs = lhs.strip()
        val = eval(rhs, {'pow': pow, 'Fraction': Fraction}, env)
        if lhs.startswith('out['):
            key = tuple(int(x) for x in re.findall(r'-?\d+', lhs))
            env['out'][key] = val
        else:
            env[lhs] = val
    return dict(env['out'])


def series_poly_check(outs, series_fn, nsamp=6, lvl_vars=('x', 'M')):
    """Compare sum_slot out[b,a,j] x^a M^j with series_fn(x, M)[b] at random x, M."""
    F = red.F
    worst = 0
    nlev = {}
    for _ in range(nsamp):
        x, M = F.rand(), F.rand()
        s = series_fn(x, M)
        levels = set(b for b, _, _ in outs) | set(p for p in s if p >= 0)
        for b in levels:
            pred = F.zero
            for (bb, a, j), v in outs.items():
                if bb == b:
                    pred = pred + v * x ** a * M ** j
            got = s.get(b, F.zero)
            if not F.iszero(pred - got):
                worst += 1
            nlev[b] = nlev.get(b, 0) + 1
    return worst, sorted(nlev)


def main():
    secs = parse(CSE)
    f = FieldFp(77)
    set_field(f)
    n, r = 3, 5
    V = [tuple(f.rand() for _ in range(4)) for _ in range(n)]
    m = [f.rand() for _ in range(n)]
    N = Numerator.random(r)
    R = Reduction(n, V, m, N)
    # ---- triangle (0,1,2) -> bubble (0,1), family A and B ; extra line 2
    T = (0, 1, 2)
    _, _, _, a_t, et3, et4 = R.triangle_geometry(T)
    tri = Residue(T, 3, 5, V[0], [et3, et4])
    newtri = {'c15': (0, (5, 0)), 'c16': (0, (0, 5)), 'c17': (1, (3, 0)), 'c18': (1, (0, 3)),
              'c19': (2, (1, 0)), 'c20': (2, (0, 1))}
    vals = {k: f.rand() for k in newtri}
    tri.c = {newtri[k]: v for k, v in vals.items()}
    results = []
    for B, l in (((0, 1), 2), ((0, 2), 1), ((1, 2), 0)):
        V0, m0, uncut, e1, e2, e3, e4, a0, a1, mp12 = R.bubble_geometry(B)
        mp34 = dot(e3, e4)
        for fam, (g3, g4) in (('A', (e3, e4)), ('B', (e4, e3))):
            eb = [None, a0, a1, g3, vsc(1 / (2 * mp34), g4)]
            kk, fl = R.lin_den(B, l)
            env = dict(vals)
            P0 = m0 - dot(a0, a0)
            P1 = -2 * dot(a0, a1)
            P2 = -dot(a1, a1)
            env.update(P0=P0, P1=P1, P2=P2, f=fl)
            for i in (1, 2, 3, 4):
                env[f'K{i}'] = dot(eb[i], kk)
                env[f"U{i}_3"] = dot(eb[i], et3) + (1 if i == 2 else 0)
                env[f'U{i}_4'] = dot(eb[i], et4)
            outs = run_section(secs['tri_to_bub'], env)

            def sfn(x, M, B=B, fam=fam, a0=a0, a1=a1, g3=g3, g4=g4, mp34=mp34, m0=m0, kk=kk, fl=fl, V0=V0):
                P0v = vadd(a0, vsc(x, a1))
                beta = m0 - dot(P0v, P0v)
                P2v = vsc((beta + M) / (2 * mp34), g4)
                A = vsub(P0v, V0)
                g = tri.series(A, g3, P2v, M, 1)
                return ldiv(g, 2 * dot(g3, kk), 2 * dot(P0v, kk) + fl, 2 * dot(P2v, kk), 0)
            bad, lev = series_poly_check(outs, sfn)
            results.append(('tri_to_bub', B, fam, len(outs), bad))
    # ---- triangle -> tadpole, bubble -> tadpole on tadpole families
    e1t, e2t, e3t, e4t = R.tadbasis_common
    bub = Residue((0, 1), 2, 4, V[0], None)
    for C in ((0,), (1,), (2,)):
        V0, m0, _, _, _ = R.geom(C)
        others = [i for i in T if i != C[0]]
        for fam, (u1, u2) in (('A', (e1t, e2t)), ('B', (e3t, e4t))):
            mp = dot(u1, u2)
            e4eff = vsc(1 / (2 * mp), u2)
            env = dict(vals)
            env['P'] = m0
            for idx, l in enumerate(others):
                kk, fl = R.lin_den(C, l)
                env[f'A{idx}'] = dot(u1, kk)
                env[f'B{idx}'] = dot(e4eff, kk)
                env[f'f{idx}'] = fl
            env['U3_3'], env['U4_3'] = dot(u1, et3), dot(e4eff, et3)
            env['U3_4'], env['U4_4'] = dot(u1, et4), dot(e4eff, et4)
            outs = run_section(secs['tri_to_tad'], env)

            def sfn(x, M, C=C, u1=u1, e4eff=e4eff, m0=m0, others=others, V0=V0):
                zero = tuple(red.F.zero for _ in range(4))
                P2v = vsc(m0 + M, e4eff)
                A = vsub(zero, V0)
                g = tri.series(A, u1, P2v, M, 2)
                mm = 2
                for l in others:
                    kk, fl = R.lin_den(C, l)
                    g = ldiv(g, 2 * dot(u1, kk), fl, 2 * dot(P2v, kk), mm - 1)
                    mm -= 1
                return g
            outs3 = {(b, 0, j): v for (b, a, j), v in outs.items()}
            bad, lev = series_poly_check(outs3, sfn)
            results.append(('tri_to_tad', C, fam, len(outs), bad))
        # bubble -> tadpole for bubbles containing C
        for Bb in ((0, 1), (0, 2), (1, 2)):
            if C[0] not in Bb:
                continue
            V0b, m0b, _, e1b, e2b, e3b, e4b, a0b, a1b, mp12b = R.bubble_geometry(Bb)
            bres = Residue(Bb, 2, 4, V0b, [e2b, e3b, e4b])
            newbub = {'c20': (0, (4, 0, 0)), 'c21': (1, (2, 0, 0)), 'c22': (2, (0, 0, 0)),
                      'c23': (0, (0, 4, 0)), 'c24': (0, (0, 0, 4)), 'c25': (0, (1, 3, 0)),
                      'c26': (0, (1, 0, 3)), 'c27': (0, (2, 2, 0)), 'c28': (0, (2, 0, 2)),
                      'c29': (0, (3, 1, 0)), 'c30': (0, (3, 0, 1)), 'c31': (1, (0, 2, 0)),
                      'c32': (1, (0, 0, 2)), 'c33': (1, (1, 1, 0)), 'c34': (1, (1, 0, 1))}
            bvals = {k: f.rand() for k in newbub}
            bres.c = {newbub[k]: v for k, v in bvals.items()}
            l = [i for i in Bb if i != C[0]][0]
            for fam, (u1, u2) in (('A', (e1t, e2t)), ('B', (e3t, e4t))):
                mp = dot(u1, u2)
                e4eff = vsc(1 / (2 * mp), u2)
                kk, fl = R.lin_den(C, l)
                env = dict(bvals)
                env.update(P=m0, A=dot(u1, kk), B=dot(e4eff, kk), f=fl)
                for i, ebv in ((2, e2b), (3, e3b), (4, e4b)):
                    env[f'S{i}'] = dot(u1, ebv)
                    env[f'T{i}'] = dot(e4eff, ebv)
                kbol = vsub(V0b, V0)
                env["W"] = dot(e2b, kbol) + 1
                outs = run_section(secs['bub_to_tad'], env)

                def sfn(x, M, u1=u1, e4eff=e4eff, m0=m0, V0=V0, kk=kk, fl=fl, bres=bres):
                    zero = tuple(red.F.zero for _ in range(4))
                    P2v = vsc(m0 + M, e4eff)
                    A = vsub(zero, V0)
                    g = bres.series(A, u1, P2v, M, 1)
                    return ldiv(g, 2 * dot(u1, kk), fl, 2 * dot(P2v, kk), 0)
                outs3 = {(b, 0, j): v for (b, a, j), v in outs.items()}
                bad, lev = series_poly_check(outs3, sfn)
                results.append(('bub_to_tad', (Bb, C), fam, len(outs), bad))
    nbad = 0
    for rr in results:
        print(rr)
        nbad += rr[-1]
    print("CSE vs definition: total mismatching level-evaluations =", nbad)


main()
