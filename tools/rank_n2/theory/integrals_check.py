"""Part (d): integrated values of the non-spurious residue terms for rank n+2.

A. Rational terms, derived symbolically by three independent methods:
   (F) Feynman parameters + dimension shift  I^d[mu^(2a) f] = (-eps)_a I^(d+2a)[f],
   (U) UV pole of the shifted integral from the large-q expansion with D0-dim
       angular averaging (no Feynman parameters),
   (P) projection of Passarino-Veltman tensor coefficients (bubbles/tadpoles) on
       the (-2eps)-dimensional metric, using the pole parts of the PV recursions
       implemented in oneloopmaster (two_point.rs: b00, b001, b0000, b0011).
B. End-to-end numerical check for n = 1..4, rank n+2 (and n+1 control),
   Euclidean kinematics: the integral of N/prod(D) computed DIRECTLY with
   Feynman parameters (finite part and 1/eps pole), versus the sum over cuts of
   (Laurent-extracted Ninja coefficients) x (masters / rational terms) using the
   formulas of REPORT.md section 5.  Normalisation: d^dq/(i pi^(d/2)) e^(eps gamma)
   mu^(2 eps), mu = 1 (equivalent to OneLOop's r_Gamma for single poles).
"""
import itertools
import math
import random
import sys

import mpmath as mp
import numpy as np
import sympy as sp
from sympy import QQ

# ============================================================== A. rational terms
eps = sp.symbols("epsilon")
m = sp.symbols("m1:5", positive=True)          # squared masses
s = {}
for i in range(4):
    for j in range(i + 1, 4):
        s[(i, j)] = sp.Symbol(f"s{i+1}{j+1}")    # s_ij = (V_i - V_j)^2


def simplex_moment(n, powers):
    """int_simplex prod x_i^a_i  (delta(1-sum x)) = prod a_i! / (sum a_i + n - 1)!"""
    num = 1
    for a in powers:
        num *= sp.factorial(a)
    return sp.Rational(num, sp.factorial(sum(powers) + n - 1))


def feynman_rational(n, alpha, extra=None):
    """Rational term  I_n[mu^(2 alpha) * extra] via Feynman parameters, where extra is
    None (scalar) or ('pe', a): (qbar.e)^a with e null, only for n=2 (bubble)."""
    # I^(d+2a) = (-1)^n Gamma(n) int dF Gamma(n - D/2)/Gamma(n) Delta^(D/2-n) (scalar)
    # (-eps)(1-eps)...(alpha-1-eps) * (-1)^n Gamma(n-2-alpha+eps) int dF Delta^(2+alpha-n-eps)
    pref = 1
    for i in range(alpha):
        pref *= (i - eps)
    # Delta for n points with x_n = 1 - sum
    xs = sp.symbols(f"x1:{n+1}")
    Delta = sum(xs[i] * m[i] for i in range(n)) - sum(xs[i] * xs[j] * s[(i, j)]
                                                       for i in range(n) for j in range(i + 1, n))
    w = 2 + alpha - n                                 # power of Delta at eps = 0
    # Gamma(n - D/2) = Gamma(n-2-alpha+eps); its residue at eps=0 multiplies int Delta^w
    g = sp.gamma(n - 2 - alpha + eps)
    val = sp.limit(pref * g * (-1) ** n, eps, 0)
    if extra is None:
        integrand = sp.expand(Delta ** w)
    else:
        assert n == 2
        a = extra[1]
        # qbar = l - x2 p  ->  (qbar.e)^a -> (-x2)^a (p.e)^a (e null kills l.e pairs)
        integrand = sp.expand(Delta ** w * (-xs[1]) ** a)
    poly = sp.Poly(integrand, *xs)
    tot = 0
    for mon, c in poly.terms():
        tot += c * simplex_moment(n, mon)
    return sp.factor(sp.expand(val * tot))


def uv_rational(n, alpha, extra_pe=0):
    """Same rational term from the UV pole of I^(4+2 alpha) via large-q expansion.
    Shifts: V_1 = 0, k_i = V_i - V_1 symbolic through dot products; e null (bubble)."""
    D0 = 4 + 2 * alpha
    # dot-product symbols among k_2..k_n and e
    names = [f"k{i}" for i in range(2, n + 1)] + (["e"] if extra_pe else [])
    dots = {}
    for a_ in names:
        for b_ in names:
            key = tuple(sorted((a_, b_)))
            if key not in dots:
                dots[key] = sp.Symbol("d_" + "_".join(key))
    if extra_pe:
        dots[("e", "e")] = 0
    lam, Q2 = sp.symbols("lam Q2")
    qv = {nm: sp.Symbol("q_" + nm) for nm in names}
    # integrand: (q.e)^a / prod_i ((q+k_i)^2 - m_i) with k_1 = 0
    expr = qv["e"] ** extra_pe if extra_pe else sp.Integer(1)
    deg_f = extra_pe
    order = deg_f - 2 * n + D0                     # needed sub-leading order in 1/lam
    for i in range(n):
        if i == 0:
            a_i = -m[0]
            lin = 0
        else:
            nm = f"k{i+1}"
            lin = 2 * qv[nm]
            a_i = dots[(nm, nm)] - m[i]
        # 1/(Q2 + lin/lam + a_i/lam^2) = 1/Q2 sum_j (-(lin/lam + a_i/lam^2)/Q2)^j
        ser = sum((-(lin / lam + a_i / lam ** 2) / Q2) ** j for j in range(order + 1))
        expr = sp.expand(expr * ser / Q2)
    expr = sp.expand(expr)
    part = sp.expand(expr.coeff(lam, -order))
    # angular average of monomials in q_v in D0 dims
    res = 0
    for term in sp.Add.make_args(part):
        c, factors = term.as_coeff_mul()
        powq2 = 0
        vec = []
        coeff = c
        for f_ in factors:
            b_, e_ = f_.as_base_exp()
            if b_ == Q2:
                powq2 += e_
            elif b_ in qv.values():
                nm = [k_ for k_, v_ in qv.items() if v_ == b_][0]
                vec += [nm] * int(e_)
            else:
                coeff *= f_
        if len(vec) % 2:
            continue
        sct = len(vec) // 2
        denom = 1
        for i in range(sct):
            denom *= (D0 + 2 * i)
        coeff = coeff * matchings(vec, dots) / denom
        res += coeff
    c_uv = sp.expand(res)
    # pole of int d^Dq/(i pi^{D/2}) c (q^2)^(-D0/2) = (-1)^(D0/2) c / (Gamma(D0/2) eps)
    # rational = -(alpha-1)! * (-1)^(D0/2) c / Gamma(D0/2)
    rat = -sp.factorial(alpha - 1) * (-1) ** (D0 // 2) * c_uv / sp.factorial(D0 // 2 - 1)
    # re-express dot products of k_i via s_ij: k_i^2 = s_{1i}, k_i.k_j = (s_1i + s_1j - s_ij)/2
    sub = {}
    for i in range(2, n + 1):
        sub[dots[(f"k{i}", f"k{i}")]] = s[(0, i - 1)]
        for j in range(i + 1, n + 1):
            sub[dots[tuple(sorted((f"k{i}", f"k{j}")))]] = (s[(0, i - 1)] + s[(0, j - 1)] - s[(i - 1, j - 1)]) / 2
    if extra_pe:
        # (k2.e) is the bubble's p.e : keep it as symbol pe
        sub[dots[("e", "k2")]] = sp.Symbol("pe")
    return sp.factor(sp.expand(rat.subs(sub)))


def matchings(vec, dots):
    if not vec:
        return 1
    first, rest = vec[0], vec[1:]
    tot = 0
    for i in range(len(rest)):
        key = tuple(sorted((first, rest[i])))
        d = dots[key]
        if d == 0:
            continue
        tot += d * matchings(rest[:i] + rest[i + 1:], dots)
    return tot


def pv_bubble_rational():
    """Pole parts of oneloopmaster's PV recursions (two_point.rs) and their
    projections on the (-2eps)-dim metric (Mastrolia-Mirabella-Peraro App. B)."""
    m0, m1, p = sp.symbols("m1 m2 s12")
    A0p = lambda mm: mm
    A00p = lambda mm: mm ** 2 / 4
    B0p, B1p, B11p, B111p = 1, sp.Rational(-1, 2), sp.Rational(1, 3), sp.Rational(-1, 4)
    kin = m1 - m0 - p
    B00p = (A0p(m1) + 2 * m0 * B0p - kin * B1p) / 6
    B001p = (2 * m0 * B1p - A0p(m1) - kin * B11p) / 8
    B0000p = (2 * m0 * B00p + A00p(m1) - kin * B001p) / 10
    B0011p = (2 * m0 * B11p + A0p(m1) - kin * B111p) / 10
    return {"I2[mu2]": sp.factor(2 * B00p),
            "I2[mu2 (q.e)]/(p.e)": sp.factor(2 * B001p),
            "I2[mu2 (q.e)^2]/(p.e)^2": sp.factor(2 * B0011p),
            "I2[mu4]": sp.factor(-4 * B0000p)}


def part_A():
    print("== A. rational terms (F: Feynman, U: UV expansion, P: PV-pole projection)")
    pe = sp.Symbol("pe")
    m1_, m2_ = m[0], m[1]
    results = {}
    cases = [("I1[mu2]", 1, 1, 0), ("I2[mu2]", 2, 1, 0), ("I2[mu2 (q.e)]", 2, 1, 1),
             ("I2[mu2 (q.e)^2]", 2, 1, 2), ("I2[mu4]", 2, 2, 0), ("I3[mu2]", 3, 1, 0),
             ("I3[mu4]", 3, 2, 0), ("I4[mu4]", 4, 2, 0), ("I4[mu6]", 4, 3, 0)]
    pv = pv_bubble_rational()
    ok = True
    for label, n, alpha, a in cases:
        F = feynman_rational(n, alpha, ("pe", a) if a else None)
        if a:
            F = sp.factor(F * pe ** a)
        U = uv_rational(n, alpha, a)
        same = sp.simplify(F - U) == 0
        line = f"   {label:18s} = {sp.sstr(F):60s}  [F == U: {same}]"
        ok &= same
        if n == 2:
            key = {0: "I2[mu2]", 1: "I2[mu2 (q.e)]/(p.e)", 2: "I2[mu2 (q.e)^2]/(p.e)^2"}[a] if alpha == 1 else "I2[mu4]"
            Pv = pv[key] * (pe ** a)
            Pv = Pv.subs({sp.Symbol("m1"): m1_, sp.Symbol("m2"): m2_, sp.Symbol("s12"): s[(0, 1)]})
            samep = sp.simplify(F - Pv) == 0
            ok &= samep
            line += f" [F == P: {samep}]"
        print(line)
        results[label] = F
    print(f"   all rational terms agree between methods: {ok}")
    return results, ok


# ============================================================== B. end-to-end numerics

def gl_simplex(n, npts):
    """Gauss-Legendre rule on the (n-1)-simplex {x_i>=0, sum x_i = 1}; returns X (N x n), W."""
    if n == 1:
        return np.ones((1, 1)), np.ones(1)
    u, wu = np.polynomial.legendre.leggauss(npts)
    u = (u + 1) / 2
    wu = wu / 2
    grids = np.meshgrid(*([u] * (n - 1)), indexing="ij")
    wgr = np.meshgrid(*([wu] * (n - 1)), indexing="ij")
    U = [g.ravel() for g in grids]
    Wt = np.prod([g.ravel() for g in wgr], axis=0)
    X = np.zeros((U[0].size, n))
    rem = np.ones(U[0].size)
    jac = np.ones(U[0].size)
    for i in range(n - 1):
        X[:, i] = rem * U[i]
        jac *= (1 - U[i]) ** (n - 2 - i)
        rem = rem * (1 - U[i])
    X[:, n - 1] = rem
    return X, Wt * jac


def mp_simplex(n, degree):
    """Gauss-Legendre product rule on the simplex in mpmath precision: list of (x tuple, w)."""
    from mpmath.calculus.quadrature import GaussLegendre
    if n == 1:
        return [((mp.mpf(1),), mp.mpf(1))]
    nodes = GaussLegendre(mp.mp).calc_nodes(degree, mp.mp.prec)
    u1 = [((x + 1) / 2, w / 2) for x, w in nodes]
    out = []
    for combo in itertools.product(u1, repeat=n - 1):
        rem = mp.mpf(1)
        w = mp.mpf(1)
        xs = []
        for i, (u, wu) in enumerate(combo):
            xs.append(rem * u)
            w *= wu * (1 - u) ** (n - 2 - i)
            rem = rem * (1 - u)
        xs.append(rem)
        out.append((tuple(xs), w))
    return out


_RULES = {}


def rule(n, degree):
    if (n, degree) not in _RULES:
        _RULES[(n, degree)] = mp_simplex(n, degree)
    return _RULES[(n, degree)]


def eps_coeffs(n, k, alpha):
    """E(eps) = (-1)^(n+k) (k+1)! Gamma(alpha-eps)/Gamma(-eps) Gamma(n-2-k-alpha+eps)/Gamma(n) e^(gamma eps)
       = cm1/eps + c0 + O(eps).  Exact rationals."""
    e = sp.symbols("e")
    expr = ((-1) ** (n + k) * sp.factorial(k + 1) * sp.gamma(alpha - e) / sp.gamma(-e)
            * sp.gamma(n - 2 - k - alpha + e) / sp.gamma(n) * sp.exp(sp.EulerGamma * e))
    ser = sp.series(expr, e, 0, 1).removeO()
    cm1 = sp.nsimplify(sp.simplify(ser.coeff(e, -1)))
    c0 = sp.nsimplify(sp.simplify(ser.coeff(e, 0)))
    return cm1, c0


_EPS_CACHE = {}


def direct_integral(V, m2, N, npts):
    """int d^dq/(i pi^{d/2}) e^{eps gamma} N(q,mu2)/prod D  ->  (finite, pole)."""
    n = len(V)
    xs = sp.symbols(f"x1:{n+1}")
    l = sp.symbols("l0:4")
    P = [sum(xs[i] * V[i][c] for i in range(n)) for c in range(4)]
    # Delta = sum x m + P^2 - sum x V^2
    mdot = lambda a, b: a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    Delta = sum(xs[i] * m2[i] for i in range(n)) + mdot(P, P) - sum(xs[i] * mdot(V[i], V[i]) for i in range(n))
    Delta = sp.expand(Delta)
    # N(q, mu2) with q = l - P
    from common import QR
    q = [l[c] - P[c] for c in range(4)]
    mu2 = sp.Symbol("mu2")
    expr = 0
    for e, c in N.items():
        expr += sp.Rational(int(c.numerator), int(c.denominator)) * q[0] ** e[0] * q[1] ** e[1] * q[2] ** e[2] * q[3] ** e[3] * mu2 ** e[4]
    poly = sp.Poly(sp.expand(expr), *l, mu2)
    # group by (alpha, degree in l)
    groups = {}
    for mon, c in poly.terms():
        a = mon[4]
        d = sum(mon[:4])
        if d % 2:
            continue
        groups.setdefault((a, d // 2), []).append((mon[:4], c))
    pts = rule(n, npts)
    Dfun = sp.lambdify(xs, Delta, "mpmath")
    Dl = [Dfun(*x) for x, w in pts]
    if any(d <= 0 for d in Dl):
        raise ValueError("Delta not positive on simplex: choose Euclidean kinematics")
    logD = [mp.log(d) for d in Dl]
    fin, pole = mp.mpf(0), mp.mpf(0)
    for (alpha, k), terms in groups.items():
        f = sum(c * l[0] ** mon[0] * l[1] ** mon[1] * l[2] ** mon[2] * l[3] ** mon[3] for mon, c in terms)
        for _ in range(k):
            f = sp.diff(f, l[0], 2) - sp.diff(f, l[1], 2) - sp.diff(f, l[2], 2) - sp.diff(f, l[3], 2)
        den = 2 ** k * math.factorial(k)
        for i in range(k):
            den *= (4 + 2 * i)
        Cx = sp.expand(f / den)
        if Cx == 0:
            continue
        key = (n, k, alpha)
        if key not in _EPS_CACHE:
            _EPS_CACHE[key] = eps_coeffs(n, k, alpha)
        cm1, c0 = _EPS_CACHE[key]
        cm1, c0 = mp.mpf(sp.Rational(cm1).p) / sp.Rational(cm1).q, mp.mpf(sp.Rational(c0).p) / sp.Rational(c0).q
        w_ = 2 + k + alpha - n
        Cf = sp.lambdify(xs, Cx, "mpmath")
        fac = math.factorial(n - 1)
        for (x, w), d, ld in zip(pts, Dl, logD):
            v = w * Cf(*x) * d ** w_
            fin += fac * v * (c0 - cm1 * ld)
            pole += fac * v * cm1
    return fin, pole


def masters(n_, V, m2, S, npts):
    """Scalar/tensor masters in the same normalisation (finite, pole), mpmath precision."""
    mdot = lambda a, b: a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    Vs = [[tompf(c) for c in V[i]] for i in S]
    ms = [tompf(m2[i]) for i in S]
    k = len(S)
    sij = {(a, b): mdot([Vs[a][c] - Vs[b][c] for c in range(4)], [Vs[a][c] - Vs[b][c] for c in range(4)])
           for a in range(k) for b in range(k)}
    if k == 1:
        mm = ms[0]
        return {"A0": (mm * (1 - mp.log(mm)), mm)}
    pts = rule(k, npts)
    Dl = [sum(x[a] * ms[a] for a in range(k)) - sum(x[a] * x[b] * sij[(a, b)] for a in range(k)
                                                   for b in range(a + 1, k)) for x, w in pts]
    if k == 2:
        out = {}
        for a in range(5):
            out[f"B{a}"] = ((-1) ** a * sum(w * x[1] ** a * (-mp.log(d)) for (x, w), d in zip(pts, Dl)),
                            mp.mpf((-1) ** a) / (a + 1))
        return out
    if k == 3:
        return {"C0": (-sum(w / d for (x, w), d in zip(pts, Dl)), 0)}
    if k == 4:
        return {"D0": (sum(w / d ** 2 for (x, w), d in zip(pts, Dl)), 0)}


def tompf(x):
    return mp.mpf(int(x.numerator)) / int(x.denominator)


def part_B(nlist, seed, npts):
    from laurent_prototype import run
    from common import Kinematics, random_numerator
    print("== B. end-to-end: direct Feynman-parameter integral vs sum of Laurent-extracted "
          "coefficients x masters/rational terms")
    rng = random.Random(seed)
    worst = 0
    for n in nlist:
        for rank in (n + 1, n + 2):
            # Euclidean kinematics: purely spatial shifts, positive masses -> Delta > 0
            V = [(QQ(0),) + tuple(QQ(rng.randint(-3, 3), rng.randint(1, 3)) for _ in range(3)) for _ in range(n)]
            m2 = [QQ(rng.randint(5, 14), rng.randint(1, 2)) for _ in range(n)]
            kin = Kinematics(V, m2)
            while True:
                e1 = Kinematics.random(1, rng).refs["tadpole"]
                break
            kin.refs["tadpole"] = e1
            N = random_numerator(rng, rank)
            worst_proto, ext, res, kin, N = run(n, rank, seed + n, verbose=False, kin=kin, N=N)
            fin_d, pole_d = direct_integral(V, m2, N, npts)
            fin_r, pole_r = mp.mpf(0), mp.mpf(0)
            mdot = lambda a, b: a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
            for S, data in ext.items():
                c = dict(data[0])
                k_ = len(S)
                if k_ == 4:
                    M = masters(n, V, m2, S, npts)
                    ms = [tompf(m2[i]) for i in S]
                    Vs = [[tompf(x) for x in V[i]] for i in S]
                    ssum = sum(mdot([Vs[a][cc] - Vs[b][cc] for cc in range(4)], [Vs[a][cc] - Vs[b][cc] for cc in range(4)])
                               for a in range(4) for b in range(a + 1, 4))
                    fin_r += c[0] * M["D0"][0] - c.get(4, 0) / 6 + c.get(6, 0) * (sum(ms) / 12 - ssum / 60)
                elif k_ == 3:
                    M = masters(n, V, m2, S, npts)
                    ms = [tompf(m2[i]) for i in S]
                    Vs = [[tompf(x) for x in V[i]] for i in S]
                    ssum = sum(mdot([Vs[a][cc] - Vs[b][cc] for cc in range(4)], [Vs[a][cc] - Vs[b][cc] for cc in range(4)])
                               for a in range(3) for b in range(a + 1, 3))
                    fin_r += c[0] * M["C0"][0] + c.get(7, 0) / 2 + c.get(14, 0) * (ssum / 24 - sum(ms) / 6)
                elif k_ == 2:
                    i1, i2 = S
                    M = masters(n, V, m2, S, npts)
                    e2 = data[2]["e2"]
                    pvec = [tompf(V[i2][cc] - V[i1][cc]) for cc in range(4)]
                    ke2 = mdot(pvec, e2)
                    s12 = mdot(pvec, pvec)
                    m1_, m2_ = tompf(m2[i1]), tompf(m2[i2])
                    tens = {0: 0, 1: 1, 2: 2, 13: 3, 20: 4}
                    for idx, a in tens.items():
                        if idx in c:
                            fin_r += c[idx] * ke2 ** a * M[f"B{a}"][0]
                            pole_r += c[idx] * ke2 ** a * M[f"B{a}"][1]
                    fin_r += c.get(9, 0) * (3 * (m1_ + m2_) - s12) / 6
                    fin_r += c.get(10, 0) * ke2 * (s12 - 2 * m1_ - 4 * m2_) / 12
                    fin_r += c.get(21, 0) * ke2 ** 2 * (5 * m1_ + 15 * m2_ - 3 * s12) / 60
                    fin_r += -c.get(22, 0) * (s12 ** 2 - 5 * s12 * (m1_ + m2_) + 10 * (m1_ ** 2 + m1_ * m2_ + m2_ ** 2)) / 60
                elif k_ == 1:
                    (i1,) = S
                    mm = tompf(m2[i1])
                    e34 = mdot(data[2]["e3"], data[2]["e4"])
                    A0f, A0p = mm * (1 - mp.log(mm)), mm
                    cc0 = c[0] + c.get(15, 0) * e34 * mm / 4
                    fin_r += cc0 * A0f + mm ** 2 / 2 * (c.get(14, 0) + c.get(15, 0) * e34 / 4)
                    pole_r += cc0 * A0p
            ef = abs(fin_r - fin_d) / max(1, abs(fin_d))
            mag = max(abs(v) for d_ in ext.values() for v in d_[0].values())
            print(f"   [prototype self-check worst rel.err {mp.nstr(worst_proto, 3)}; max |coefficient| {mp.nstr(mag, 3)}]")
            ep = abs(pole_r - pole_d) / max(1, abs(pole_d))
            worst = max(worst, ef, ep)
            print(f"   n={n} r={rank}: finite: direct {mp.nstr(fin_d, 15)}  reduction {mp.nstr(fin_r, 15)}"
                  f"  rel.diff {mp.nstr(ef, 2)};  1/eps: direct {mp.nstr(pole_d, 15)}  reduction "
                  f"{mp.nstr(pole_r, 15)}  rel.diff {mp.nstr(ep, 2)}")
    print(f"   worst relative difference {mp.nstr(worst, 3)}  {'PASS' if worst < mp.mpf(10)**-15 else 'FAIL'}")
    return worst


if __name__ == "__main__":
    res, okA = part_A()
    nlist = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 2, 3, 4]
    npts = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    part_B(nlist, 4242, npts)
