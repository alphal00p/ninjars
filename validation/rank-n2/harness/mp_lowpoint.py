#!/usr/bin/env python3
"""Independent high-precision tadpole and bubble tensor integrals (no COLLIER code).

Backend "mp" of collier_reference.py: evaluates one job (props order, exact polynomial
N(l) in the loop momentum l of the job's reference propagator, exact masses, momentum p,
invariant s = p^2 as passed to COLLIER) in the normalisation of collier_reference.py.

Symmetric integration of the 4-dimensional components of the D-dimensional l,
    l^{mu1} .. l^{mu2j} -> (lbar^2)^j / (D (D+2) .. (D+2j-2)) * sum_pairings g..g,
gives with D = 4 - 2 eps, to O(eps^0) and after dividing by r_Gamma:
  tadpole (D_0 = lbar^2 - m0):
      sum_j  P(m) m0^(j+1) / (2^j (j+1)!) [1/eps + H_(j+1) - log(m0/mu_R^2)]
  bubble  (Feynman x: (l + x p)^2 - Delta(x), Delta = x m1 + (1-x) m0 - x (1-x) s):
      int_0^1 dx sum_j C_j(x) Delta(x)^j / (2^j j!) [1/eps + H_j - log(Delta(x)/mu_R^2)]
with C_j(x) the coefficients of the even degree-2j monomials of N(k - x p), P(m) the
pairing factor prod_mu (a_mu - 1)!! (-1)^((a1+a2+a3)/2), H_j harmonic numbers and the
-i0 prescription Delta -> Delta - i0.  All polynomial algebra is exact (rationals); the
only transcendental pieces are log(m0) and L_b = int_0^1 x^b log(Delta(x)/mu_R^2) dx,
computed with mpmath's tanh-sinh quadrature (breakpoints at the real parts of the roots
of Delta) at RN2_MP_DPS (default 50) and RN2_MP_DPS + 20 digits; the difference is the
error estimate.  Scaleless integrals (m0 = 0 tadpole; m0 = m1 = s = 0 bubble) vanish."""
import math
import os
from decimal import Decimal
from math import comb, factorial

import mpmath as mp

from collier_reference import Q, cq, cadd, cmul, cscale, is_zero, ZERO

DPS = int(os.environ.get("RN2_MP_DPS", "50"))


def qmp(q):
    return mp.mpf(int(q.numerator)) / int(q.denominator)


def cmpc(c):
    return mp.mpc(qmp(c[0]), qmp(c[1]))


def dfact(n):
    return 1 if n <= 0 else n * dfact(n - 2)


def pairing_factor(m):
    if any(e % 2 for e in m):
        return 0
    return (-1) ** ((m[1] + m[2] + m[3]) // 2) * dfact(m[0] - 1) * dfact(m[1] - 1) * dfact(m[2] - 1) * dfact(m[3] - 1)


def harmonic(j):
    return sum((Q(1, k) for k in range(1, j + 1)), Q(0))


def log_minus_i0(z):
    """log(z - i0)."""
    if mp.im(z) == 0 and mp.re(z) < 0:
        return mp.log(-mp.re(z)) - mp.pi * 1j
    return mp.log(z)


def to_dec(z):
    return [Decimal(mp.nstr(mp.re(z), DPS + 15, strip_zeros=False)), Decimal(mp.nstr(mp.im(z), DPS + 15, strip_zeros=False))]


def xpoly_mul(a, b):
    out = [ZERO] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if is_zero(x):
            continue
        for j, y in enumerate(b):
            out[i + j] = cadd(out[i + j], cmul(x, y))
    return out


def xpoly_add(acc, a, f):
    if len(acc) < len(a):
        acc.extend([ZERO] * (len(a) - len(acc)))
    for i, x in enumerate(a):
        acc[i] = cadd(acc[i], cmul(x, f))


def tadpole(job):
    m0 = job["masses2"][0]
    zero = [mp.mpc(0)] * 3
    if is_zero(m0):
        return zero, 0.0, 0.0
    with mp.workdps(DPS + 20):
        A = {}
        for m, c in job["poly"].items():
            P = pairing_factor(m)
            if P == 0:
                continue
            j = sum(m) // 2
            A[j] = cadd(A.get(j, ZERO), cscale(c, Q(P, 2 ** j * factorial(j + 1))))
        pole, fin_h = ZERO, ZERO
        for j, a in A.items():
            mj = cq(1)
            for _ in range(j + 1):
                mj = cmul(mj, m0)
            t = cmul(a, mj)
            pole = cadd(pole, t)
            fin_h = cadd(fin_h, cscale(t, harmonic(j + 1)))
        L = log_minus_i0(cmpc(m0) / qmp(job["mu2"]))
        fin = cmpc(fin_h) - cmpc(pole) * L
        absum = float(abs(cmpc(fin_h)) + abs(cmpc(pole) * L))
        return [fin, cmpc(pole), mp.mpc(0)], absum, 0.0


def bubble(job):
    m0, m1 = job["masses2"]
    s = job["mominv"][0]
    p = job["momvec"][0]
    zero = [mp.mpc(0)] * 3
    if is_zero(m0) and is_zero(m1) and s == 0:
        return zero, 0.0, 0.0
    delta = [m0, cadd(cadd(m1, (-m0[0], -m0[1])), cq(-s)), cq(s)]
    # N(k - x p) = sum over k-monomials of x-polynomials
    kpoly = {}
    for m, c in job["poly"].items():
        terms = [((0, 0, 0, 0), [c])]
        for mu in range(4):
            a = m[mu]
            if a == 0:
                continue
            new = []
            for t in range(a + 1):
                # C(a, t) k^t (-x p_mu)^(a - t)
                f = comb(a, t) * (-p[mu]) ** (a - t)
                if f == 0:
                    continue
                for km, xp in terms:
                    kk = list(km)
                    kk[mu] += t
                    new.append((tuple(kk), [ZERO] * (a - t) + [cscale(x, f) for x in xp]))
            terms = new
        for km, xp in terms:
            acc = kpoly.setdefault(km, [])
            xpoly_add(acc, xp, cq(1))
    A = {}  # j -> x-polynomial of C_j(x) Delta^j / (2^j j!) (pairing factors included)
    dpow = {0: [cq(1)]}
    for km, xp in kpoly.items():
        P = pairing_factor(km)
        if P == 0:
            continue
        j = sum(km) // 2
        if j not in dpow:
            d = dpow[max(dpow)]
            for _ in range(j - max(dpow)):
                d = xpoly_mul(d, delta)
            dpow[j] = d
        poly = xpoly_mul(xp, dpow[j])
        xpoly_add(A.setdefault(j, []), poly, cq(Q(P, 2 ** j * factorial(j))))
    bmax = max((len(a) for a in A.values()), default=0)
    B = [ZERO] * bmax
    pole, fin_h = ZERO, ZERO
    for j, a in A.items():
        for b, x in enumerate(a):
            B[b] = cadd(B[b], x)
            t = cscale(x, Q(1, b + 1))
            pole = cadd(pole, t)
            fin_h = cadd(fin_h, cscale(t, harmonic(j)))
    Lb = [None, None]
    for k, dps in enumerate((DPS, DPS + 20)):
        with mp.workdps(dps):
            d = [cmpc(x) for x in delta]
            mu2 = qmp(job["mu2"])
            pts = [mp.mpf(0)]
            if d[2] != 0:
                disc = mp.sqrt(d[1] ** 2 - 4 * d[2] * d[0])
                roots = [(-d[1] - disc) / (2 * d[2]), (-d[1] + disc) / (2 * d[2])]
            elif d[1] != 0:
                roots = [-d[0] / d[1]]
            else:
                roots = []
            for r in sorted(mp.re(r) for r in roots):
                if 0 < r < 1 and all(abs(r - q) > mp.mpf(10) ** (-dps // 2) for q in pts):
                    pts.append(r)
            pts.append(mp.mpf(1))
            vals = []
            for b in range(bmax):
                if is_zero(B[b]):
                    vals.append(mp.mpc(0))
                    continue
                f = (lambda x, b=b: x ** b * log_minus_i0((d[0] + x * (d[1] + x * d[2])) / mu2))
                vals.append(mp.quad(f, pts))
            Lb[k] = vals
    with mp.workdps(DPS + 20):
        fin = cmpc(fin_h)
        err = mp.mpf(0)
        absum = abs(cmpc(fin_h))
        for b in range(bmax):
            cb = cmpc(B[b])
            fin -= cb * Lb[1][b]
            err += abs(cb) * abs(Lb[1][b] - Lb[0][b])
            absum += abs(cb * Lb[1][b])
        return [fin, cmpc(pole), mp.mpc(0)], float(absum), float(err)


def job_value(job):
    N = len(job["props"])
    if N == 1:
        vals, absum, err = tadpole(job)
    elif N == 2:
        vals, absum, err = bubble(job)
    else:
        raise ValueError("mp backend: N <= 2 only")
    with mp.workdps(DPS + 20):
        lau = [to_dec(v) for v in vals] + [[Decimal(0), Decimal(0)]]
        ab = [absum, float(abs(vals[1])), 0.0]
    return {"laurent": lau, "absum": ab, "err": err, "errflag": 0, "accflag": 0}
