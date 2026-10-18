"""Cross-check of the exported closed-form subtraction formulas (all monomials,
old and new) against an independent implementation: the truncated-Laurent-series
routines of laurent_lib.py (the ones used by the full reduction prototype).
Random complex inputs, 40-digit arithmetic."""
import json
import random

import mpmath as mp
import sympy as sp

from laurent_lib import Param, divide, smul, padd, mpdot
import subtraction_formulas as SF

mp.mp.dps = 40


def rc(rng):
    return mp.mpc(rng.uniform(-1, 1), rng.uniform(-1, 1))


def rv(rng):
    return [rc(rng) for _ in range(4)]


def monomial_series(forms, j, depth_tmin):
    """forms: list of z-series components [c_t, c_0(dict), c_m1(dict)] -> Laurent dict."""
    a = len(forms)
    depth = a - depth_tmin
    ser = [{(0, 0): mp.mpf(1)}] + [dict() for _ in range(depth)]
    for lf in forms:
        ser = smul(ser, lf, depth)
    out = {}
    for s, cd in enumerate(ser):
        for (ix, imu), v in cd.items():
            out[(a - s, ix, imu + j)] = out.get((a - s, ix, imu + j), 0) + v
    return out


def lf(ct, c0, cm1):
    return [{(0, 0): ct}, c0, cm1]


def evalF(F, subs):
    return {k: mp.mpc(complex(sp.N(e.subs(subs), 30))) for k, e in F.items()}


def compare(a, b):
    keys = set(a) | set(b)
    err = max((abs(a.get(k, 0) - b.get(k, 0)) for k in keys), default=0)
    sc = max((abs(v) for v in list(a.values()) + list(b.values())), default=1)
    return err / sc


def main(ntests=3, seed=11):
    rng = random.Random(seed)
    worst = 0
    for _ in range(ntests):
        # ---- tri -> bub
        u, eb1, eb2, eb3, eb4, k = rv(rng), rv(rng), rv(rng), rv(rng), rv(rng), rv(rng)
        P = [rc(rng) for _ in range(3)]
        fv = rc(rng)
        subs = {SF.U1: complex(mpdot(eb1, u)), SF.U2: complex(mpdot(eb2, u)), SF.U3: complex(mpdot(eb3, u)),
                SF.U4: complex(mpdot(eb4, u)), SF.K1: complex(mpdot(eb1, k)), SF.K2: complex(mpdot(eb2, k)),
                SF.K3: complex(mpdot(eb3, k)), SF.K4: complex(mpdot(eb4, k)), SF.f: complex(fv),
                SF.P0: complex(P[0]), SF.P1: complex(P[1]), SF.P2: complex(P[2])}
        Xd = lambda s: {(0, 0): P[0] * s, (1, 0): P[1] * s, (2, 0): P[2] * s, (0, 1): s}
        Lu = lf(mpdot(eb3, u), {(0, 0): mpdot(eb1, u), (1, 0): mpdot(eb2, u)}, Xd(mpdot(eb4, u)))
        den = (2 * mpdot(eb3, k), {(0, 0): 2 * mpdot(eb1, k) + fv, (1, 0): 2 * mpdot(eb2, k)}, Xd(2 * mpdot(eb4, k)))
        for idx in range(1, 21):
            jj, a3, a4 = SF.TRI[idx]
            a = a3 + a4
            ser = monomial_series([Lu] * a, jj, -1)
            q = {kk: v for kk, v in divide(ser, den, 0).items() if kk[0] >= 0}
            F, _ = SF.tri_formula(idx, "bub")
            e = compare(q, evalF(F, subs))
            worst = max(worst, e)
            print(f"tri->bub c{idx:<2d}: closed form vs series rel.err {mp.nstr(e, 3)}")
        # ---- tri -> tad
        u, e3, e4, k0, k1 = rv(rng), rv(rng), rv(rng), rv(rng), rv(rng)
        Pp, fa, fb = rc(rng), rc(rng), rc(rng)
        subs = {SF.U3: complex(mpdot(e3, u)), SF.U4: complex(mpdot(e4, u)), SF.A0: complex(mpdot(e3, k0)),
                SF.B0: complex(mpdot(e4, k0)), SF.A1: complex(mpdot(e3, k1)), SF.B1: complex(mpdot(e4, k1)),
                SF.f0: complex(fa), SF.f1: complex(fb), SF.P: complex(Pp)}
        Md = lambda s: {(0, 0): Pp * s, (0, 1): s}
        Lu = lf(mpdot(e3, u), {}, Md(mpdot(e4, u)))
        d0 = (2 * mpdot(e3, k0), {(0, 0): fa}, Md(2 * mpdot(e4, k0)))
        d1 = (2 * mpdot(e3, k1), {(0, 0): fb}, Md(2 * mpdot(e4, k1)))
        for idx in range(1, 21):
            jj, a3, a4 = SF.TRI[idx]
            a = a3 + a4
            ser = monomial_series([Lu] * a, jj, -2)
            q = divide(divide(ser, d0, -1), d1, 0)
            q = {kk: v for kk, v in q.items() if kk[0] >= 0}
            F, _ = SF.tri_formula(idx, "tad")
            e = compare(q, evalF(F, subs))
            worst = max(worst, e)
            print(f"tri->tad c{idx:<2d}: closed form vs series rel.err {mp.nstr(e, 3)}")
        # ---- bub -> tad
        eb2, eb3, eb4, e3, e4, k = rv(rng), rv(rng), rv(rng), rv(rng), rv(rng), rv(rng)
        Wv, Pp, fv = rc(rng), rc(rng), rc(rng)
        subs = {SF.S2: complex(mpdot(e3, eb2)), SF.S3: complex(mpdot(e3, eb3)), SF.S4: complex(mpdot(e3, eb4)),
                SF.T2: complex(mpdot(e4, eb2)), SF.T3: complex(mpdot(e4, eb3)), SF.T4: complex(mpdot(e4, eb4)),
                SF.W: complex(Wv), SF.A: complex(mpdot(e3, k)), SF.B: complex(mpdot(e4, k)),
                SF.f: complex(fv), SF.P: complex(Pp)}
        Md = lambda s: {(0, 0): Pp * s, (0, 1): s}
        L2 = lf(mpdot(e3, eb2), {(0, 0): Wv}, Md(mpdot(e4, eb2)))
        L3 = lf(mpdot(e3, eb3), {}, Md(mpdot(e4, eb3)))
        L4 = lf(mpdot(e3, eb4), {}, Md(mpdot(e4, eb4)))
        den = (2 * mpdot(e3, k), {(0, 0): fv}, Md(2 * mpdot(e4, k)))
        for idx in range(1, 35):
            jj, p2, p3, p4 = SF.BUB[idx]
            ser = monomial_series([L2] * p2 + [L3] * p3 + [L4] * p4, jj, -1)
            q = {kk: v for kk, v in divide(ser, den, 0).items() if kk[0] >= 0}
            e = compare(q, evalF(SF.bub_formula(idx), subs))
            worst = max(worst, e)
            print(f"bub->tad c{idx:<2d}: closed form vs series rel.err {mp.nstr(e, 3)}")
    print(f"WORST closed-form vs series relative error: {mp.nstr(worst, 3)} "
          f"{'PASS' if worst < 1e-12 else 'FAIL'}")


if __name__ == "__main__":
    main()
