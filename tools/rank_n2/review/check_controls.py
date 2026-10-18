#!/usr/bin/env python3
"""Negative controls (the checks of check_decomp.py must fail when the recipe is broken),
the independent-tadpole-basis variant, and the box mu-expansion convention test."""
import red
from red import FieldFp, set_field, Numerator, Reduction, zero_point_degree, dot


def setup(n, r, seed, with_mu=True):
    f = FieldFp(seed)
    set_field(f)
    V = [tuple(f.rand() for _ in range(4)) for _ in range(n)]
    m = [f.rand() for _ in range(n)]
    N = Numerator.random(r, with_mu=with_mu)
    return V, m, N


def summarize(tag, summ):
    bad = {k: sum(1 for _, w, d in v if w > 0 or d > 0) for k, v in summ.items()}
    print(f"  {tag}: cuts failing per level {bad}")


def control(name, n, r, patch):
    V, m, N = setup(n, r, 4242)
    R = Reduction(n, V, m, N)
    patch(R)
    try:
        summ = R.run(log=lambda s: None)
        summarize(name, summ)
    except AssertionError as e:
        print(f"  {name}: recipe fit raised: {e}")


print("== negative controls (each must report failures) ==")
# 1. bubbles without triangle subtraction
orig = Reduction.series_remainder


def no_tri(self, S, P0, P1, P2, mu2, lo, subtract_levels=(3, 2)):
    if len(S) == 2:
        subtract_levels = ()
    return orig(self, S, P0, P1, P2, mu2, lo, subtract_levels)


control("bubble w/o triangle subtraction (n=3,r=5)", 3, 5, lambda R: setattr(R, 'series_remainder', no_tri.__get__(R)))


def no_bub(self, S, P0, P1, P2, mu2, lo, subtract_levels=(3, 2)):
    if len(S) == 1:
        subtract_levels = (3,)
    return orig(self, S, P0, P1, P2, mu2, lo, subtract_levels)


control("tadpole w/o bubble subtraction (n=3,r=5)", 3, 5, lambda R: setattr(R, 'series_remainder', no_bub.__get__(R)))

# 2. rank n+1 residue spaces at rank n+2 (R_k lowered by one)
for n in (1, 2, 3, 4):
    V, m, N = setup(n, n + 2, 99)
    R = Reduction(n, V, m, N)
    R.R = lambda k, R=R: R.r - R.n + k - 1
    try:
        summ = R.run(log=lambda s: None)
        summarize(f"rank n+1 bases at r=n+2, n={n}", summ)
    except AssertionError as e:
        print(f"  rank n+1 bases at r=n+2, n={n}: recipe fit raised: {str(e)[:90]}")

# 3. triangle c0/c7/c14 from family A only (no A/B average) with boxes present
orig_tri = Reduction.triangle


def tri_A_only(self, S):
    res = orig_tri(self, S)
    d = self.info['tri_t0_AminusB'][S]
    for j, v in d.items():
        res.c[(j, (0, 0))] = res.c[(j, (0, 0))] + v / 2
    return res


control("triangle t^0 from A only (n=5,r=7)", 5, 7, lambda R: setattr(R, 'triangle', tri_A_only.__get__(R)))

print("== A-B difference at triangle t^0 (box contamination), n=5 r=7: nonzero per mu^2j? ==")
V, m, N = setup(5, 7, 4243)
R = Reduction(5, V, m, N)
R.run(check_fits=False, log=print)
for S, d in list(R.info['tri_t0_AminusB'].items())[:3]:
    print("  ", S, {j: (not red.F.iszero(v)) for j, v in d.items()})

print("== independent tadpole bases: degree of Delta_0 ==")
for n in range(1, 6):
    for dr in (0, 1, 2):
        V, m, N = setup(n, n + dr, 3000 + n * 10 + dr)
        for tb in ('common', 'independent'):
            R = Reduction(n, V, m, N, tadpole_basis=tb)
            summ = R.run(log=print)
            deg = zero_point_degree(R.zero_point(npts=30))
            ok = all(w == 0 and d == 0 for v in summ.values() for _, w, d in v)
            print(f"  n={n} r=n{dr:+d} {tb:11s}: residues ok={ok}  Delta_0 "
                  f"{'degree<=%d' % deg[0] if deg else '??'}{' (identically 0)' if deg and deg[1] else ''}")

print("== box mu-expansion: exact shift (beta) vs Ninja convention mu2 = tau^2 v^2 ==")
for n in (4, 5, 6):
    for with_mu in (True, False):
        V, m, N = setup(n, n + 2, 5000 + n)
        if not with_mu:
            N = Numerator.random(n + 2, with_mu=False)
        res = {}
        for beta in (True, False):
            R = Reduction(n, V, m, N, beta_box=beta)
            summ = R.run(log=lambda s: None)
            res[beta] = (R, summ)
        bad = {beta: sum(1 for _, w, d in res[beta][1].get(4, []) if d > 0) for beta in res}
        # compare c4, c5, c6 between conventions
        Rb, Rn = res[True][0], res[False][0]
        diff = {}
        for S in Rb.res:
            if len(S) == 4:
                for key, name in (((2, (0,)), 'c4'), ((2, (1,)), 'c5'), ((3, (0,)), 'c6')):
                    diff.setdefault(name, 0)
                    diff[name] += 0 if red.F.iszero(Rb.res[S].c[key] - Rn.res[S].c[key]) else 1
        print(f"  n={n} r={n+2} mu2-numerator={with_mu}: boxes failing fit with beta={bad[True]}, "
              f"Ninja-convention={bad[False]}; #boxes where conventions differ {diff}")
