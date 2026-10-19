"""Shared exact (rational) utilities for the rank n+2 integrand-decomposition checks.

Conventions (identical to Ninja / NinjaRS):
  * metric (+,-,-,-); q is the 4-dimensional loop momentum, mu2 = mu^2 the
    (-2 eps)-dimensional part, D_i = (q + V_i)^2 - m_i^2 - mu2.
  * weighted degree: each q component counts 1, mu2 counts 2.  A numerator of
    rank r is a polynomial of weighted degree <= r.
  * a cut S = (i1 < ... < ik) uses qbar = q + V_{i1}; residues are polynomials
    in qbar and mu2.

Residue representative spaces (Ninja's monomial bases, written basis-free so
that they have rational generators):
  k=5  pentagon : { mu2 }
  k=4  box      : { mu2^j , mu2^j (qbar.v) }        v spans {p1,p2,p3}^perp
  k=3  triangle : { mu2^j H_a(qbar_T) }             H_a harmonic (traceless) of
                                                     degree a in T={p1,p2}^perp
                                                     == span{(qbar.e3)^a,(qbar.e4)^a}
  k=2  bubble   : { mu2^j (qbar.r)^a H_b(qbar_T) }  r = Ninja's null reference
                                                     (= e2), T = {p,r}^perp
                                                     == monomials qe2^a qe3^b, qe2^a qe4^b
  k=1  tadpole  : { mu2^j (qbar.e1)^a M(qbar_T), mu2^j (qbar.e2)^a M(qbar_T) }
                  e1,e2 null, T={e1,e2}^perp, M any monomial in two transverse
                  coordinates  == monomials not divisible by qe1*qe2
  k=0  (no cut) : c(q, mu2) prod_all D, integrating to zero, with c of weighted
                  degree <= max(r-2n, r-n-2) for one common tadpole basis
                  (Ninja's default) and <= max(r-2n, r-n-1) for independent
                  tadpole bases per cut (decomposition_check.zero_point_bound;
                  tools/rank_n2/README.md, correction 2)
with weighted degree <= R_k = r - n + k for the cut residues (k >= 1).
"""
from fractions import Fraction
import itertools
import random

import sympy as sp
from sympy import QQ
from sympy.polys.rings import ring

# ---------------------------------------------------------------- vectors

METRIC = (1, -1, -1, -1)


def dot(a, b):
    return sum(METRIC[i] * a[i] * b[i] for i in range(4))


def vadd(a, b):
    return tuple(a[i] + b[i] for i in range(4))


def vsub(a, b):
    return tuple(a[i] - b[i] for i in range(4))


def vscale(s, a):
    return tuple(s * a[i] for i in range(4))


def rand_rat(rng, num=9, den=5):
    while True:
        d = rng.randint(1, den)
        n = rng.randint(-num, num)
        if n != 0:
            return QQ(n, d)


def rand_vec(rng, num=9, den=4):
    return tuple(rand_rat(rng, num, den) for _ in range(4))


def rand_null(rng):
    """Random rational null vector (stereographic parametrisation of the light cone)."""
    while True:
        s, u, sc = rand_rat(rng, 7, 5), rand_rat(rng, 7, 5), rand_rat(rng, 5, 3)
        w = s * s + u * u
        v = (sc * (1 + w), sc * 2 * s, sc * 2 * u, sc * (w - 1))
        # random spatial rotation by a rational orthogonal-ish shuffle: permute axes
        perm = rng.sample([1, 2, 3], 3)
        v = (v[0], v[perm[0]], v[perm[1]], v[perm[2]])
        assert dot(v, v) == 0
        return v


def orth_complement(vectors):
    """Rational basis of {w : w.p = 0 for all p in vectors} (Minkowski dot)."""
    if not vectors:
        return [tuple(QQ(int(i == j)) for j in range(4)) for i in range(4)]
    M = sp.Matrix([[METRIC[i] * sp.Rational(int(p[i].numerator), int(p[i].denominator))
                    for i in range(4)] for p in vectors])
    ns = M.nullspace()
    out = []
    for v in ns:
        v = v * sp.ilcm(*[sp.fraction(x)[1] for x in v])
        out.append(tuple(QQ(int(sp.fraction(x)[0]), int(sp.fraction(x)[1])) for x in v))
    return out


def solve_linear(vectors, rhs):
    """A particular rational solution x of  p_j . x = rhs_j."""
    if not vectors:
        return (QQ(0),) * 4
    M = sp.Matrix([[METRIC[i] * sp.Rational(int(p[i].numerator), int(p[i].denominator))
                    for i in range(4)] for p in vectors])
    b = sp.Matrix([sp.Rational(int(c.numerator), int(c.denominator)) for c in rhs])
    sol, params = M.gauss_jordan_solve(b)
    sol = sol.subs({p: 0 for p in params})
    return tuple(QQ(int(sp.fraction(x)[0]), int(sp.fraction(x)[1])) for x in sol)


# ---------------------------------------------------------------- polynomial rings

QR, Q0, Q1, Q2, Q3, MU2 = ring("q0,q1,q2,q3,mu2", QQ)
QVARS = (Q0, Q1, Q2, Q3)


def wdeg(poly):
    """Weighted degree (q:1, mu2:2) of a 5-variable polynomial."""
    if not poly:
        return -1
    return max(e[0] + e[1] + e[2] + e[3] + 2 * e[4] for e in poly.keys())


def lin(vec, shift=None):
    """The polynomial (q + shift).vec in the q-ring."""
    shift = shift or (QQ(0),) * 4
    return sum((METRIC[i] * vec[i] * (QVARS[i] + shift[i]) for i in range(4)), QR(0))


def den_poly(V, m2):
    qb = [QVARS[i] + V[i] for i in range(4)]
    return qb[0] ** 2 - qb[1] ** 2 - qb[2] ** 2 - qb[3] ** 2 - m2 - MU2


def weighted_monomials(maxdeg):
    """Exponent tuples (a0,a1,a2,a3,j) with a0+..+a3+2j <= maxdeg."""
    out = []
    for j in range(maxdeg // 2 + 1):
        d = maxdeg - 2 * j
        for a in itertools.product(range(d + 1), repeat=4):
            if sum(a) <= d:
                out.append(a + (j,))
    return out


def random_numerator(rng, rank, coeff=5):
    p = QR(0)
    for e in weighted_monomials(rank):
        c = rng.randint(-coeff, coeff)
        if c:
            p += QR({e: QQ(c)})
    return p


# ---------------------------------------------------------------- harmonic polynomials

def harmonic_basis(G, a):
    """Basis (list of dicts {(i,j): coeff} meaning l1^i l2^j) of homogeneous degree-a
    polynomials in two linear forms l1,l2 with sum_ij G_ij d_i d_j P = 0 (traceless
    w.r.t. the Gram matrix G of the transverse vectors)."""
    if a == 0:
        return [{(0, 0): QQ(1)}]
    if a == 1:
        return [{(1, 0): QQ(1)}, {(0, 1): QQ(1)}]
    # unknowns c_m for l1^m l2^(a-m), m = 0..a ; Laplacian maps to degree a-2
    rows = []
    for mm in range(a - 1):  # target monomial l1^mm l2^(a-2-mm)
        row = [QQ(0)] * (a + 1)
        # d1^2 term: from l1^(mm+2) l2^(a-2-mm): coeff (mm+2)(mm+1) G11
        row[mm + 2] += G[0][0] * (mm + 2) * (mm + 1)
        # d2^2 term: from l1^mm l2^(a-mm): coeff (a-mm)(a-mm-1) G22
        row[mm] += G[1][1] * (a - mm) * (a - mm - 1)
        # 2 G12 d1 d2: from l1^(mm+1) l2^(a-1-mm): coeff 2 G12 (mm+1)(a-1-mm)
        row[mm + 1] += 2 * G[0][1] * (mm + 1) * (a - 1 - mm)
        rows.append(row)
    M = sp.Matrix([[sp.Rational(int(x.numerator), int(x.denominator)) for x in row] for row in rows])
    ns = M.nullspace()
    assert len(ns) == 2, (a, len(ns))
    out = []
    for v in ns:
        out.append({(m, a - m): QQ(int(sp.fraction(v[m])[0]), int(sp.fraction(v[m])[1]))
                    for m in range(a + 1) if v[m] != 0})
    return out


def poly_from_forms(coeffs, l1, l2):
    return sum((c * l1 ** i * l2 ** j for (i, j), c in coeffs.items()), QR(0))


# ---------------------------------------------------------------- cuts

class Kinematics:
    def __init__(self, V, m2, refs=None):
        self.V = V
        self.m2 = m2
        self.n = len(V)
        self.D = [den_poly(V[i], m2[i]) for i in range(self.n)]
        self.refs = refs or {}

    @staticmethod
    def random(n, rng, massless_fraction=0.0, common_tadpole_basis=True):
        V = [rand_vec(rng) for _ in range(n)]
        m2 = []
        for _ in range(n):
            if rng.random() < massless_fraction:
                m2.append(QQ(0))
            else:
                m2.append(abs(rand_rat(rng, 9, 4)))
        refs = {}
        if common_tadpole_basis:
            # Ninja's tadpole_basis() normally returns the same {e1,e2,e3,e4} for
            # every single cut of the integral.
            while True:
                e1, e2 = rand_null(rng), rand_null(rng)
                if dot(e1, e2) != 0:
                    break
            refs["tadpole"] = (e1, e2)
        return Kinematics(V, m2, refs)


class Cut:
    """Exact parametrisation of the cut S and Ninja-type residue basis."""

    def __init__(self, kin, S, rank, rng):
        self.kin, self.S, self.k = kin, tuple(S), len(S)
        n = kin.n
        self.R = rank - n + self.k            # maximal weighted degree of the residue
        i1 = S[0]
        self.V0 = kin.V[i1]
        self.m0 = kin.m2[i1]
        self.p = [vsub(kin.V[j], self.V0) for j in S[1:]]
        # linear cut conditions: 2 qbar.p_j + p_j^2 - m_j^2 + m_i1^2 = 0
        rhs = [-(dot(pj, pj) - kin.m2[j] + self.m0) / 2 for pj, j in zip(self.p, S[1:])]
        self.qpart = solve_linear(self.p, rhs)       # a particular qbar
        self.ws = orth_complement(self.p)            # free directions
        self.nfree = len(self.ws)
        assert self.nfree == 5 - self.k
        self.uncut = [j for j in range(n) if j not in S]
        self._build_basis(rng)

    # qbar.vec as polynomial in q
    def qbar_dot(self, vec):
        return lin(vec, self.V0)

    def _build_basis(self, rng):
        k, R = self.k, self.R
        basis, labels = [], []
        if R < 0:
            self.basis, self.labels = [], []
            return
        if k == 5:
            basis, labels = [MU2], ["mu2"]
        elif k == 4:
            v = self.ws[0]
            self.vperp = v
            x = self.qbar_dot(v)
            for j in range(R // 2 + 1):
                basis.append(MU2 ** j); labels.append(f"mu2^{j}")
            for j in range((R - 1) // 2 + 1):
                basis.append(MU2 ** j * x); labels.append(f"mu2^{j}*x")
        elif k == 3:
            n1, n2 = self.ws
            G = [[dot(n1, n1), dot(n1, n2)], [dot(n2, n1), dot(n2, n2)]]
            l1, l2 = self.qbar_dot(n1), self.qbar_dot(n2)
            for j in range(R // 2 + 1):
                for a in range(R - 2 * j + 1):
                    for h_i, h in enumerate(harmonic_basis(G, a)):
                        basis.append(MU2 ** j * poly_from_forms(h, l1, l2))
                        labels.append(f"mu2^{j}*H{a}.{h_i}")
        elif k == 2:
            p = self.p[0]
            while True:
                r = rand_null(rng)
                if dot(r, p) != 0:
                    break
            self.ref = r
            T = orth_complement([p, r])
            n1, n2 = T
            G = [[dot(n1, n1), dot(n1, n2)], [dot(n2, n1), dot(n2, n2)]]
            l1, l2, lr = self.qbar_dot(n1), self.qbar_dot(n2), self.qbar_dot(r)
            for j in range(R // 2 + 1):
                for a in range(R - 2 * j + 1):
                    for b in range(R - 2 * j - a + 1):
                        for h_i, h in enumerate(harmonic_basis(G, b)):
                            basis.append(MU2 ** j * lr ** a * poly_from_forms(h, l1, l2))
                            labels.append(f"mu2^{j}*qr^{a}*H{b}.{h_i}")
        elif k == 1:
            shared = self.kin.refs.get("tadpole")
            if shared is not None:
                e1, e2 = shared
            else:
                while True:
                    e1, e2 = rand_null(rng), rand_null(rng)
                    if dot(e1, e2) != 0:
                        break
            self.e1, self.e2 = e1, e2
            n1, n2 = orth_complement([e1, e2])
            l1, l2 = self.qbar_dot(n1), self.qbar_dot(n2)
            x1, x2 = self.qbar_dot(e1), self.qbar_dot(e2)
            for j in range(R // 2 + 1):
                d = R - 2 * j
                for b in range(d + 1):
                    for c in range(d - b + 1):
                        T = l1 ** b * l2 ** c
                        basis.append(MU2 ** j * T); labels.append(f"mu2^{j}*l1^{b}l2^{c}")
                        for a in range(1, d - b - c + 1):
                            basis.append(MU2 ** j * x1 ** a * T); labels.append(f"mu2^{j}*qe1^{a}*l1^{b}l2^{c}")
                            basis.append(MU2 ** j * x2 ** a * T); labels.append(f"mu2^{j}*qe2^{a}*l1^{b}l2^{c}")
        self.basis, self.labels = basis, labels

    # ---- restriction to the cut: polynomial ring in the free variables y
    def make_ring(self):
        if self.nfree == 0:
            self.Y = None
            return
        names = ",".join(f"y{i}" for i in range(self.nfree))
        res = ring(names, QQ)
        self.YR, self.ys = res[0], res[1:]
        # qbar(y) and q(y)
        qbar = [self.YR(self.qpart[i]) + sum((self.ws[s][i] * self.ys[s] for s in range(self.nfree)), self.YR(0))
                for i in range(4)]
        self.q_of_y = [qbar[i] - self.V0[i] for i in range(4)]
        self.mu2_of_y = qbar[0] ** 2 - qbar[1] ** 2 - qbar[2] ** 2 - qbar[3] ** 2 - self.m0

    def restrict(self, poly):
        """Restrict a 5-variable polynomial to the cut (exact)."""
        if self.nfree == 0:
            qv = [self.qpart[i] - self.V0[i] for i in range(4)]
            mu2 = dot(self.qpart, self.qpart) - self.m0
            return poly(*qv, mu2)
        return poly.compose(list(zip(QR.gens, list(self.q_of_y) + [self.mu2_of_y]))) if False else \
            _substitute(poly, self.YR, self.q_of_y + [self.mu2_of_y])


def _substitute(poly, YR, images):
    """Substitute the 5 ring variables by polynomials of ring YR (cached powers)."""
    cache = [dict() for _ in range(5)]

    def pw(i, e):
        c = cache[i]
        if e not in c:
            if e == 0:
                c[e] = YR(1)
            elif e == 1:
                c[e] = images[i]
            else:
                h = e // 2
                c[e] = pw(i, h) * pw(i, e - h)
        return c[e]

    # group by (e0,e1,e2,e3) prefix: Horner in mu2 not needed (sizes are small)
    out = YR(0)
    for e, coeff in poly.items():
        term = YR(coeff)
        for i in range(5):
            if e[i]:
                term = term * pw(i, e[i])
        out += term
    return out


def ydeg(p):
    if not hasattr(p, "keys"):
        return 0
    if not p:
        return -1
    return max(sum(e) for e in p.keys())
