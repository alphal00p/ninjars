"""Part (c): numerical prototype of Ninja's Laurent-expansion reduction extended to
rank r = n+2, validated against the EXACT residues of decomposition_check.top_down.

For every cut the prototype follows the recipe of REPORT.md section 4:
  boxes     : c0,c1 from the two 4-dim solutions; c6,c5,c4 from the 3-term
              large-t expansion ON the d-dim cut  qbar=q0+t v, mu2=t^2 v^2+mu0^2;
              c2,c3 (tests only) from finite-mu2 sampling minus pentagons;
  triangles : two T3 expansions (t along e3 / e4), depth R=r-n+3, division by
              the uncut denominators, t^0 coefficients averaged (box terms cancel);
  bubbles   : T2 expansion depth r-n+2 and swapped depth r-n+1, division,
              coefficient-level subtraction of the parent triangles;
  tadpoles  : T3-type expansions along e1 and e3, depth r-n+1, division,
              subtraction of parent bubbles and triangles -> c0, c14, c15.
Higher-point residues used in the subtractions are the ones EXTRACTED by the
prototype itself (not the exact ones).  Every corrected Laurent coefficient
(t >= 0) is compared with the Laurent expansion of the exact residue on the same
parametrisation, and every Ninja-basis coefficient with the exact one.
"""
import itertools
import random
import sys

import mpmath as mp

from common import Kinematics, random_numerator, dot as qdot
from decomposition_check import top_down
from laurent_lib import (Param, expand_polynomial, divide, den_on_param, poly_part, levels,
                         mpdot, vadd, vsub, vsc, tompf, tovec, null_pair_in_plane, smul, padd,
                         pmul, pscale, unlevel)

mp.mp.dps = 40

# ------------------------------------------------------------------ Ninja monomial tables
# (index, mu2 power j, {basis-vector name: power})
TRIANGLE = [(0, 0, {}), (1, 0, {"e3": 1}), (2, 0, {"e3": 2}), (3, 0, {"e3": 3}), (4, 0, {"e4": 1}),
            (5, 0, {"e4": 2}), (6, 0, {"e4": 3}), (7, 1, {}), (8, 1, {"e3": 1}), (9, 1, {"e4": 1}),
            (10, 1, {"e3": 2}), (11, 1, {"e4": 2}), (12, 0, {"e3": 4}), (13, 0, {"e4": 4}), (14, 2, {}),
            # rank n+2
            (15, 0, {"e3": 5}), (16, 0, {"e4": 5}), (17, 1, {"e3": 3}), (18, 1, {"e4": 3}),
            (19, 2, {"e3": 1}), (20, 2, {"e4": 1})]
BUBBLE = [(0, 0, {}), (1, 0, {"e2": 1}), (2, 0, {"e2": 2}), (3, 0, {"e3": 1}), (4, 0, {"e3": 2}),
          (5, 0, {"e4": 1}), (6, 0, {"e4": 2}), (7, 0, {"e2": 1, "e3": 1}), (8, 0, {"e2": 1, "e4": 1}),
          (9, 1, {}), (10, 1, {"e2": 1}), (11, 1, {"e3": 1}), (12, 1, {"e4": 1}), (13, 0, {"e2": 3}),
          (14, 0, {"e3": 3}), (15, 0, {"e4": 3}), (16, 0, {"e2": 2, "e3": 1}), (17, 0, {"e2": 2, "e4": 1}),
          (18, 0, {"e2": 1, "e3": 2}), (19, 0, {"e2": 1, "e4": 2}),
          # rank n+2
          (20, 0, {"e2": 4}), (21, 1, {"e2": 2}), (22, 2, {}), (23, 0, {"e3": 4}), (24, 0, {"e4": 4}),
          (25, 0, {"e2": 1, "e3": 3}), (26, 0, {"e2": 1, "e4": 3}), (27, 0, {"e2": 2, "e3": 2}),
          (28, 0, {"e2": 2, "e4": 2}), (29, 0, {"e2": 3, "e3": 1}), (30, 0, {"e2": 3, "e4": 1}),
          (31, 1, {"e3": 2}), (32, 1, {"e4": 2}), (33, 1, {"e2": 1, "e3": 1}), (34, 1, {"e2": 1, "e4": 1})]
TADPOLE = [(0, 0, {}), (1, 0, {"e1": 1}), (2, 0, {"e2": 1}), (3, 0, {"e3": 1}), (4, 0, {"e4": 1}),
           (5, 0, {"e1": 2}), (6, 0, {"e2": 2}), (7, 0, {"e3": 2}), (8, 0, {"e4": 2}),
           (10, 0, {"e1": 1, "e3": 1}), (11, 0, {"e1": 1, "e4": 1}), (12, 0, {"e2": 1, "e3": 1}),
           (13, 0, {"e2": 1, "e4": 1}), (14, 1, {}), (15, 0, {"e3": 1, "e4": 1}),
           # rank n+2
           (16, 0, {"e1": 3}), (17, 0, {"e2": 3}), (18, 0, {"e3": 3}), (19, 0, {"e4": 3}),
           (20, 0, {"e1": 2, "e3": 1}), (21, 0, {"e1": 2, "e4": 1}), (22, 0, {"e2": 2, "e3": 1}),
           (23, 0, {"e2": 2, "e4": 1}), (24, 0, {"e1": 1, "e3": 2}), (25, 0, {"e1": 1, "e4": 2}),
           (26, 0, {"e2": 1, "e3": 2}), (27, 0, {"e2": 1, "e4": 2}), (28, 0, {"e3": 2, "e4": 1}),
           (29, 0, {"e3": 1, "e4": 2}), (30, 0, {"e1": 1, "e3": 1, "e4": 1}),
           (31, 0, {"e2": 1, "e3": 1, "e4": 1}), (32, 1, {"e1": 1}), (33, 1, {"e2": 1}),
           (34, 1, {"e3": 1}), (35, 1, {"e4": 1})]
BOX = [(0, 0, {}), (1, 0, {"v": 1}), (2, 1, {}), (3, 1, {"v": 1}), (4, 2, {}), (5, 2, {"v": 1}),
       (6, 3, {})]


def table_for(k, R):
    tab = {1: TADPOLE, 2: BUBBLE, 3: TRIANGLE, 4: BOX}[k]
    return [m for m in tab if sum(m[2].values()) + 2 * m[1] <= R]


# ------------------------------------------------------------------ helpers

def qq_monos(poly):
    return [(e, tompf(c)) for e, c in poly.items()]


def eval_qq(poly, q, mu2):
    s = 0
    for e, c in poly.items():
        s += tompf(c) * q[0] ** e[0] * q[1] ** e[1] * q[2] ** e[2] * q[3] ** e[3] * mu2 ** e[4]
    return s


def eval_monomial(m, vecs, V0, q, mu2):
    qb = vadd(q, V0)
    val = mu2 ** m[1]
    for name, p in m[2].items():
        val *= mpdot(qb, vecs[name]) ** p
    return val


def fit_ninja(poly, table, vecs, V0, rng):
    """Coefficients of an exact residue representative in Ninja's monomial basis."""
    npts = len(table)
    pts = []
    for _ in range(npts + 4):
        q = [mp.mpc(rng.uniform(-2, 2), rng.uniform(-2, 2)) for _ in range(4)]
        mu2 = mp.mpc(rng.uniform(-2, 2), rng.uniform(-2, 2))
        pts.append((q, mu2))
    A = mp.matrix(npts, npts)
    b = mp.matrix(npts, 1)
    for i, (q, mu2) in enumerate(pts[:npts]):
        for j, m in enumerate(table):
            A[i, j] = eval_monomial(m, vecs, V0, q, mu2)
        b[i] = eval_qq(poly, q, mu2)
    x = mp.lu_solve(A, b)
    # verify on extra points (representative must lie in the span)
    err = 0
    for q, mu2 in pts[npts:]:
        val = sum(x[j] * eval_monomial(m, vecs, V0, q, mu2) for j, m in enumerate(table))
        ref = eval_qq(poly, q, mu2)
        err = max(err, abs(val - ref) / (1 + abs(ref)))
    return {m[0]: x[j] for j, m in enumerate(table)}, err


def residue_on_param(param, coeffs, table, vecs, V0, tmin):
    """Laurent expansion (t >= tmin) of a Ninja-basis residue on a parametrisation."""
    depth_needed = {}
    out = {}
    for m in table:
        c = coeffs.get(m[0], 0)
        if c == 0:
            continue
        d = sum(m[2].values())
        depth = d - tmin
        if depth < 0:
            continue
        ser = [{(0, 0): c}] + [dict() for _ in range(depth)]
        for name, p in m[2].items():
            lf = param.linear_form(vecs[name], const=mpdot(V0, vecs[name]))
            for _ in range(p):
                ser = smul(ser, lf, depth)
        for s, cd in enumerate(ser):
            for (ix, imu), v in cd.items():
                key = (d - s, ix, imu + m[1])
                out[key] = out.get(key, 0) + v
    return out


def maxrel(a, b, keys=None):
    keys = keys if keys is not None else set(a) | set(b)
    err, scale = 0, 0
    for k in keys:
        err = max(err, abs(a.get(k, 0) - b.get(k, 0)))
        scale = max(scale, abs(b.get(k, 0)), abs(a.get(k, 0)))
    return err / (scale if scale else 1)


# ------------------------------------------------------------------ main reduction

def run(n, rank, seed, verbose=True, kin=None, N=None):
    rng = random.Random(seed)
    if kin is None:
        kin = Kinematics.random(n, rng, massless_fraction=0.0)
    if N is None:
        N = random_numerator(rng, rank)
    res, msg = top_down(kin, N, rank, random.Random(seed + 1))
    assert res is not None, msg
    info = res["info"]
    V = [tovec(v) for v in kin.V]
    m2 = [tompf(x) for x in kin.m2]
    Nm = qq_monos(N)
    report = []
    worst = 0

    def note(label, err, tol=mp.mpf(10) ** (-20)):
        nonlocal worst
        worst = max(worst, err)
        report.append((label, err))
        if verbose:
            flag = "ok" if err < tol else "MISMATCH"
            print(f"   {label:62s} rel.err {mp.nstr(err, 3):>10s}  {flag}")

    frng = random.Random(seed + 99)
    extracted = {}   # S -> (coeff dict, table, vecs, V0)

    # ---------------- pentagons (only needed for the finite-mu2 box test)
    pent = {}
    for S in itertools.combinations(range(n), 5):
        cut = info[S]["cut"]
        qb = tovec(cut.qpart)
        mu2 = mpdot(qb, qb) - m2[S[0]]
        q = vsub(qb, V[S[0]])
        val = eval_qq(N, q, mu2)
        for j in range(n):
            if j not in S:
                val /= mpdot(vadd(q, V[j]), vadd(q, V[j])) - m2[j] - mu2
        pent[S] = val / mu2
        exact = tompf(info[S]["coeffs"][0])
        note(f"pentagon {S}: c0 (mu2 coefficient)", abs(pent[S] - exact) / abs(exact))

    def pent_sub(Sbox, q, mu2):
        s = 0
        for S5, c in pent.items():
            if set(Sbox) < set(S5):
                t = c * mu2
                for j in range(n):
                    if j not in S5:
                        t *= mpdot(vadd(q, V[j]), vadd(q, V[j])) - m2[j] - mu2
                s += t
        return s

    # ---------------- boxes
    for S in itertools.combinations(range(n), 4):
        cut = info[S]["cut"]
        V0, m0 = V[S[0]], m2[S[0]]
        v = tovec(cut.vperp)
        v2 = mpdot(v, v)
        qp = tovec(cut.qpart)
        q0 = vsub(qp, vsc(mpdot(qp, v) / v2, v))           # point of the 4-d cut line with q0.v = 0
        mu0 = mpdot(q0, q0) - m0                            # mu2 = t^2 v^2 + mu0  on the d-dim cut
        unc = [j for j in range(n) if j not in S]
        # 3-term mu-expansion on the exact d-dimensional cut
        R = rank - n + 4
        # N(q0 - V0 + t v, t^2 v^2 + mu0) as polynomial in t (coefficient list)

        def npoly_t(offset):
            coeffs = [mp.mpc(0)] * (rank + 1)
            qt = [[q0[i] - V0[i], v[i]] for i in range(4)]          # q_i = a + b t
            mt = [offset, 0, v2]                                     # mu2 = offset + v^2 t^2
            for e, c in Nm:
                poly = [c]
                for i in range(4):
                    for _ in range(e[i]):
                        poly = pmul_t(poly, qt[i])
                for _ in range(e[4]):
                    poly = pmul_t(poly, mt)
                for p, x in enumerate(poly):
                    coeffs[p] += x
            return coeffs

        top_on = npoly_t(mu0)[::-1]       # descending powers t^rank, t^rank-1, ...
        top_off = npoly_t(0)[::-1]        # Ninja's rank<=n+1 convention mu2 = t^2 v^2
        num = list(top_on[:3])
        for j in unc:
            kk = vsub(V[j], V0)
            f = mpdot(kk, kk) + m0 - m2[j]
            d0, d1 = 2 * mpdot(v, kk), 2 * mpdot(q0, kk) + f
            num[0] /= d0
            num[1] = (num[1] - d1 * num[0]) / d0
            num[2] = (num[2] - d1 * num[1]) / d0
        # top three t-powers t^R, t^(R-1), t^(R-2) of Delta_4 on the d-dim cut determine
        # c_R, c_(R-1), c_(R-2) (index = leading t power); c_R also feeds t^(R-2) via mu0.
        def lead(e):        # leading coefficient of monomial c_e on the parametrisation
            return v2 ** (e // 2) if e % 2 == 0 else v2 ** ((e + 1) // 2)

        def sub2(e):        # its t^(e-2) coefficient
            j = e // 2
            return j * mu0 * (v2 ** (j - 1) if e % 2 == 0 else v2 ** j)
        # pentagons (c mu2 / D_l ~ t) contaminate t^1 and below: use only t-powers >= 2
        c = {}
        if R >= 2:
            c[R] = num[0] / lead(R)
        if R - 1 >= 2:
            c[R - 1] = num[1] / lead(R - 1)
        if R - 2 >= 2:
            c[R - 2] = (num[2] - c[R] * sub2(R)) / lead(R - 2)
        # 4-d solutions (mu2 = 0): qbar = q0 +- x0 v  ->  c0, c1
        x0 = mp.sqrt(-mu0 / v2)
        vals = []
        for sgn in (1, -1):
            q = vsub(vadd(q0, vsc(sgn * x0, v)), V0)
            val = eval_qq(N, q, 0)
            for j in unc:
                val /= mpdot(vadd(q, V[j]), vadd(q, V[j])) - m2[j]
            vals.append(val)
        c0_, c1_ = (vals[0] + vals[1]) / 2, (vals[0] - vals[1]) / (2 * x0 * v2)
        if 0 not in c and R >= 0:
            c[0] = c0_
        if 1 not in c and R >= 1:
            c[1] = c1_
        # remaining (c2, c3 for R=6; c2 for R=5): finite mu2 sampling minus pentagons
        missing = [e for e in range(R + 1) if e not in c]
        if missing:
            s = mp.mpf(3) / 7
            xs = mp.sqrt((s - mu0) / v2)
            vals = []
            for sgn in (1, -1):
                q = vsub(vadd(q0, vsc(sgn * xs, v)), V0)
                val = eval_qq(N, q, s) - pent_sub(S, q, s)
                for j in unc:
                    val /= mpdot(vadd(q, V[j]), vadd(q, V[j])) - m2[j] - s
                vals.append(val)
            Ps, Qs = (vals[0] + vals[1]) / 2, (vals[0] - vals[1]) / (2 * xs * v2)
            for e in missing:
                if e % 2 == 0:
                    rest = Ps - sum(c[k_] * s ** (k_ // 2) for k_ in c if k_ % 2 == 0)
                    c[e] = rest / s ** (e // 2)
                else:
                    rest = Qs - sum(c[k_] * s ** (k_ // 2) for k_ in c if k_ % 2 == 1)
                    c[e] = rest / s ** (e // 2)
        tab = table_for(4, R)
        c = {k_: c[k_] for k_ in [m[0] for m in tab]}
        exact, ferr = fit_ninja(res["residues"][S], tab, {"v": v}, V0, frng)
        note(f"box {S}: exact representative in Ninja basis (fit residual)", ferr)
        note(f"box {S}: all {len(tab)} coefficients (c6,c5,c4 via on-cut mu-exp.)", maxrel(c, exact))
        if rank == n + 2 and N.degree(4) > 0:
            # illustrate: Ninja's rank<=n+1 off-cut convention spoils the 3rd term
            num_off = list(top_off[:3])
            for j in unc:
                kk = vsub(V[j], V0)
                f = mpdot(kk, kk) + m0 - m2[j]
                d0, d1 = 2 * mpdot(v, kk), 2 * mpdot(q0, kk) + f
                num_off[0] /= d0
                num_off[1] = (num_off[1] - d1 * num_off[0]) / d0
                num_off[2] = (num_off[2] - d1 * num_off[1]) / d0
            c4_off = num_off[2] / v2 ** 2 - 3 * mu0 * num_off[0] / v2 ** 3
            if verbose:
                print(f"   box {S}: c4 with OFF-cut mu-expansion (mu2=t^2 v^2) rel.dev "
                      f"{mp.nstr(abs(c4_off - exact[4]) / abs(exact[4]), 3)} (expected O(1))")
        extracted[S] = (c, tab, {"v": v}, V0)

    # ---------------- triangles
    for S in itertools.combinations(range(n), 3):
        cut = info[S]["cut"]
        V0, m0 = V[S[0]], m2[S[0]]
        n1, n2 = tovec(cut.ws[0]), tovec(cut.ws[1])
        qp = tovec(cut.qpart)
        # a = qbar offset in span(p1,p2):  a = qp + al n1 + be n2 with a.n1 = a.n2 = 0
        G = mp.matrix([[mpdot(n1, n1), mpdot(n1, n2)], [mpdot(n2, n1), mpdot(n2, n2)]])
        sol = mp.lu_solve(G, mp.matrix([-mpdot(qp, n1), -mpdot(qp, n2)]))
        a = vadd(qp, vadd(vsc(sol[0], n1), vsc(sol[1], n2)))
        e3, e4 = null_pair_in_plane(n1, n2)
        mp34 = mpdot(e3, e4)
        beta = m0 - mpdot(a, a)
        R = rank - n + 3
        unc = [j for j in range(n) if j not in S]
        Q = {}
        for name, (E, Gv) in {"t||e3": (e3, vsc(1 / (2 * mp34), e4)),
                               "t||e4": (e4, vsc(1 / (2 * mp34), e3))}.items():
            par = Param(E, vsub(a, V0), [0] * 4, Gv, (beta, 0, 0))
            lau = expand_polynomial(Nm, par, len(unc))
            for j in unc:
                lau = divide(lau, den_on_param(par, V[j], m2[j], V0, m0), len(unc) - unc.index(j) - 1)
            lau = poly_part(lau)
            ex = poly_part(expand_polynomial(qq_monos(res["residues"][S]), par, 0))
            hi = [k_ for k_ in set(lau) | set(ex) if k_[0] >= 1]
            note(f"triangle {S} [{name}]: Laurent coeffs t>=1 vs exact residue", maxrel(lau, ex, hi))
            Q[name] = (lau, ex)
        # t^0 averages
        t0keys = [(0, 0, j) for j in range(R // 2 + 1)]
        avg = {k_: (Q["t||e3"][0].get(k_, 0) + Q["t||e4"][0].get(k_, 0)) / 2 for k_ in t0keys}
        avg_ex = {k_: (Q["t||e3"][1].get(k_, 0) + Q["t||e4"][1].get(k_, 0)) / 2 for k_ in t0keys}
        note(f"triangle {S}: averaged t^0 (c0,c7,c14) vs exact", maxrel(avg, avg_ex))
        if n >= 4:
            single = maxrel({k_: Q["t||e3"][0].get(k_, 0) for k_ in t0keys},
                            {k_: Q["t||e3"][1].get(k_, 0) for k_ in t0keys})
            if verbose:
                print(f"   triangle {S}: single-orientation t^0 deviation (box terms) {mp.nstr(single, 3)}")
        # Ninja coefficients
        L1, L2 = Q["t||e3"][0], Q["t||e4"][0]
        c = {}
        for m in table_for(3, R):
            idx, j, pw = m
            if not pw:
                c[idx] = avg[(0, 0, j)]
            elif "e4" in pw:      # t||e3 -> qbar.e4 = t mp34
                b = pw["e4"]; c[idx] = L1.get((b, 0, j), 0) / mp34 ** b
            else:
                b = pw["e3"]; c[idx] = L2.get((b, 0, j), 0) / mp34 ** b
        tab = table_for(3, R)
        vecs = {"e3": e3, "e4": e4}
        exact, ferr = fit_ninja(res["residues"][S], tab, vecs, V0, frng)
        note(f"triangle {S}: exact representative in Ninja basis (fit residual)", ferr)
        note(f"triangle {S}: all {len(tab)} Ninja coefficients", maxrel(c, exact))
        extracted[S] = (c, tab, vecs, V0)

    # ---------------- bubbles
    for S in itertools.combinations(range(n), 2):
        cut = info[S]["cut"]
        i1, i2 = S
        V0, m0, m1 = V[i1], m2[i1], m2[i2]
        k = vsub(V[i1], V[i2])                              # Ninja: p[1] = V[cut1]-V[cut2]
        r = tovec(cut.ref)
        e2 = r
        e1 = vsub(k, vsc(mpdot(k, k) / (2 * mpdot(k, r)), r))
        mp12 = mpdot(e1, e2)
        from common import orth_complement
        T = orth_complement([cut.p[0], cut.ref])
        e3, e4 = null_pair_in_plane(tovec(T[0]), tovec(T[1]))
        e4 = vsc(-mp12 / mpdot(e3, e4), e4)                 # Ninja normalisation e3.e4 = -e1.e2
        mp34 = mpdot(e3, e4)
        kq = mpdot(k, k)
        e12, e1k, e2k = 2 * mp12, 2 * mpdot(e1, k), 2 * mpdot(e2, k)
        b0 = (-m1 + m0 + kq) / e2k
        b1 = -e1k / e2k
        a0 = vsc(b0, e2)
        a1 = vadd(e1, vsc(b1, e2))
        prm = (m0, -b0 * e12, -b1 * e12)
        R = rank - n + 2
        unc = [j for j in range(n) if j not in S]
        Q = {}
        for name, (E, Gv, tmin_final) in {"t||e3": (e3, vsc(1 / (2 * mp34), e4), 0),
                                           "t||e4": (e4, vsc(1 / (2 * mp34), e3), 1)}.items():
            par = Param(E, vsub(a0, V0), a1, Gv, prm)
            lau = expand_polynomial(Nm, par, len(unc) + tmin_final)
            for idx_j, j in enumerate(unc):
                lau = divide(lau, den_on_param(par, V[j], m2[j], V0, m0),
                             len(unc) + tmin_final - idx_j - 1)
            lau = {k_: v_ for k_, v_ in lau.items() if k_[0] >= tmin_final}
            # coefficient-level subtraction of parent triangles
            for S3 in itertools.combinations(range(n), 3):
                if set(S) < set(S3):
                    (ell,) = set(S3) - set(S)
                    cT, tabT, vecT, V0T = extracted[S3]
                    sub = residue_on_param(par, cT, tabT, vecT, V0T, tmin_final - 1)
                    sub = divide(sub, den_on_param(par, V[ell], m2[ell], V0, m0), tmin_final)
                    for k_, v_ in sub.items():
                        if k_[0] >= tmin_final:
                            lau[k_] = lau.get(k_, 0) - v_
            ex = expand_polynomial(qq_monos(res["residues"][S]), par, tmin_final)
            note(f"bubble {S} [{name}]: corrected Laurent coeffs t>={tmin_final} vs exact",
                 maxrel(lau, ex))
            Q[name] = lau
        L1, L2 = Q["t||e3"], Q["t||e4"]
        c = {}
        tab = table_for(2, R)
        for m in tab:
            idx, j, pw = m
            a_ = pw.get("e2", 0)
            if "e4" in pw:
                b = pw["e4"]; c[idx] = L1.get((b, a_, j), 0) / (mp12 ** a_ * mp34 ** b)
            elif "e3" in pw:
                b = pw["e3"]; c[idx] = L2.get((b, a_, j), 0) / (mp12 ** a_ * mp34 ** b)
            else:
                c[idx] = L1.get((0, a_, j), 0) / mp12 ** a_
        vecs = {"e2": e2, "e3": e3, "e4": e4}
        exact, ferr = fit_ninja(res["residues"][S], tab, vecs, V0, frng)
        note(f"bubble {S}: exact representative in Ninja basis (fit residual)", ferr)
        note(f"bubble {S}: all {len(tab)} Ninja coefficients", maxrel(c, exact))
        extracted[S] = (c, tab, vecs, V0)
        extracted[S] = extracted[S] + ({"k": k, "mp12": mp12},)

    # ---------------- tadpoles
    for S in itertools.combinations(range(n), 1):
        cut = info[S]["cut"]
        (i1,) = S
        V0, m0 = V[i1], m2[i1]
        e1, e2 = tovec(cut.e1), tovec(cut.e2)
        mp12 = mpdot(e1, e2)
        from common import orth_complement
        T = orth_complement([cut.e1, cut.e2])
        e3, e4 = null_pair_in_plane(tovec(T[0]), tovec(T[1]))
        e4 = vsc(-mp12 / mpdot(e3, e4), e4)
        mp34 = mpdot(e3, e4)
        R = rank - n + 1
        unc = [j for j in range(n) if j not in S]
        Q = {}
        for name, (E, Gv) in {"t||e1": (e1, vsc(1 / (2 * mp12), e2)),
                               "t||e2": (e2, vsc(1 / (2 * mp12), e1)),
                               "t||e3": (e3, vsc(1 / (2 * mp34), e4)),
                               "t||e4": (e4, vsc(1 / (2 * mp34), e3))}.items():
            par = Param(E, vsc(-1, V0), [0] * 4, Gv, (m0, 0, 0))
            lau = expand_polynomial(Nm, par, len(unc))
            for idx_j, j in enumerate(unc):
                lau = divide(lau, den_on_param(par, V[j], m2[j], V0, m0), len(unc) - idx_j - 1)
            lau = poly_part(lau)
            for S2 in itertools.combinations(range(n), 2):
                if set(S) < set(S2):
                    (ell,) = set(S2) - set(S)
                    cB, tabB, vecB, V0B, _ = extracted[S2]
                    sub = residue_on_param(par, cB, tabB, vecB, V0B, -1)
                    sub = divide(sub, den_on_param(par, V[ell], m2[ell], V0, m0), 0)
                    for k_, v_ in poly_part(sub).items():
                        lau[k_] = lau.get(k_, 0) - v_
            for S3 in itertools.combinations(range(n), 3):
                if set(S) < set(S3):
                    l1, l2 = sorted(set(S3) - set(S))
                    cT, tabT, vecT, V0T = extracted[S3]
                    sub = residue_on_param(par, cT, tabT, vecT, V0T, -2)
                    sub = divide(sub, den_on_param(par, V[l1], m2[l1], V0, m0), -1)
                    sub = divide(sub, den_on_param(par, V[l2], m2[l2], V0, m0), 0)
                    for k_, v_ in poly_part(sub).items():
                        lau[k_] = lau.get(k_, 0) - v_
            ex = poly_part(expand_polynomial(qq_monos(res["residues"][S]), par, 0))
            note(f"tadpole {S} [{name}]: corrected Laurent coeffs t>=0 vs exact", maxrel(lau, ex))
            Q[name] = lau
        c = {0: Q["t||e1"].get((0, 0, 0), 0), 14: Q["t||e1"].get((0, 0, 1), 0)}
        c[15] = 2 * (Q["t||e3"].get((0, 0, 1), 0) - c[14]) / mp34
        present = [m[0] for m in table_for(1, R)]
        c = {k_: v_ for k_, v_ in c.items() if k_ in present}
        # pure powers from the four directions (spurious, tests only)
        for idx, j, pw in table_for(1, R):
            if len(pw) == 1:
                (nm, b), = pw.items()
                grow = {"e2": ("t||e1", mp12), "e1": ("t||e2", mp12),
                        "e4": ("t||e3", mp34), "e3": ("t||e4", mp34)}[nm]
                if nm in ("e3", "e4") and b + 2 <= R:
                    continue   # t^b mixes with qe3*qe4^(b+1) (or qe4*qe3^(b+1)) on this family
                c[idx] = Q[grow[0]].get((b, 0, j), 0) / grow[1] ** b
        tab = table_for(1, R)
        vecs = {"e1": e1, "e2": e2, "e3": e3, "e4": e4}
        exact, ferr = fit_ninja(res["residues"][S], tab, vecs, V0, frng)
        note(f"tadpole {S}: exact representative in Ninja basis (fit residual)", ferr)
        ns = [k_ for k_ in (0, 14, 15) if k_ in c]
        note(f"tadpole {S}: non-spurious c0,c14,c15", maxrel({k_: c[k_] for k_ in ns},
                                                         {k_: exact[k_] for k_ in ns}))
        others = [k_ for k_ in c if k_ not in (0, 14, 15)]
        note(f"tadpole {S}: {len(others)} pure-power spurious coefficients",
             maxrel({k_: c[k_] for k_ in others}, {k_: exact[k_] for k_ in others}))
        extracted[S] = (c, tab, vecs, V0)
    return worst, extracted, res, kin, N


def pmul_t(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    nlist = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 3, 4, 5]
    ranks = sys.argv[3] if len(sys.argv) > 3 else "n+2"
    allworst = 0
    for n in nlist:
        rk = {"n+2": [n + 2], "all": sorted({max(0, n - 1), n, n + 1, n + 2})}[ranks]
        for rank in rk:
            print(f"== n={n} rank={rank} seed={seed}")
            w, *_ = run(n, rank, seed + 13 * n + rank)
            print(f"== n={n} rank={rank}: worst relative error {mp.nstr(w, 3)}")
            allworst = max(allworst, w)
    print(f"OVERALL worst relative error {mp.nstr(allworst, 3)}  "
          f"{'PASS' if allworst < mp.mpf(10)**-20 else 'FAIL'}")
