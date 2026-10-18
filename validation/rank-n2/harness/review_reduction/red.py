#!/usr/bin/env python3
"""Independent referee implementation of a Ninja-style integrand reduction at any
rank (in particular r = n+2), following the recipe derived in rank-n2/theory.

Arithmetic is generic: either exact in F_p[i] (p = 2^61-1 = 3 mod 4, so i = sqrt(-1)
is outside F_p and every element of F_p has a square root in F_p[i]) or mpmath mpc.

Conventions: metric (+,-,-,-); D_i = (q+V_i)^2 - m_i - mu2 (m_i = squared masses);
cut S=(i1<..<ik), qbar = q + V_{i1}; a residue is a polynomial in mu2 and qbar.u for
Ninja-style null basis vectors u (all built here independently of the theory scripts).
"""
import itertools
import random
import sys
from fractions import Fraction

P = (1 << 61) - 1
HALF = (P + 1) // 4


class G:
    """Element a + b i of F_p[i]."""
    __slots__ = ('a', 'b')

    def __init__(self, a, b=0):
        self.a = a % P
        self.b = b % P

    @staticmethod
    def c(o):
        if isinstance(o, G):
            return o
        if isinstance(o, Fraction):
            return G(o.numerator * pow(o.denominator % P, P - 2, P))
        return G(o)

    def __add__(s, o):
        o = G.c(o)
        return G(s.a + o.a, s.b + o.b)
    __radd__ = __add__

    def __sub__(s, o):
        o = G.c(o)
        return G(s.a - o.a, s.b - o.b)

    def __rsub__(s, o):
        return G.c(o) - s

    def __neg__(s):
        return G(-s.a, -s.b)

    def __mul__(s, o):
        if isinstance(o, G):
            return G(s.a * o.a - s.b * o.b, s.a * o.b + s.b * o.a)
        o = G.c(o)
        return G(s.a * o.a - s.b * o.b, s.a * o.b + s.b * o.a)
    __rmul__ = __mul__

    def inv(s):
        nn = (s.a * s.a + s.b * s.b) % P
        if nn == 0:
            raise ZeroDivisionError("G(0)")
        ni = pow(nn, P - 2, P)
        return G(s.a * ni, -s.b * ni)

    def __truediv__(s, o):
        return s * G.c(o).inv()

    def __rtruediv__(s, o):
        return G.c(o) * s.inv()

    def __pow__(s, k):
        if k < 0:
            return s.inv() ** (-k)
        r = G(1)
        b = s
        while k:
            if k & 1:
                r = r * b
            b = b * b
            k >>= 1
        return r

    def __eq__(s, o):
        o = G.c(o)
        return s.a == o.a and s.b == o.b

    def __hash__(s):
        return hash((s.a, s.b))

    def __repr__(s):
        return f"G({s.a},{s.b})"


class FieldFp:
    name = 'Fp[i]'

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.zero = G(0)
        self.one = G(1)

    def num(self, x):
        return G.c(x)

    def rand(self):
        return G(self.rng.randrange(1, P))

    def rand_small(self):
        # small rational, mimicking 'physical' rational kinematics
        return G.c(Fraction(self.rng.randint(-30, 30), self.rng.randint(1, 7)))

    def sqrt(self, x):
        x = G.c(x)
        assert x.b == 0, "sqrt only for F_p elements"
        r = pow(x.a, HALF, P)
        if r * r % P == x.a:
            return G(r)
        r = pow((-x.a) % P, HALF, P)
        assert r * r % P == (-x.a) % P
        return G(0, r)

    def iszero(self, x, scale=None):
        x = G.c(x)
        return x.a == 0 and x.b == 0

    def piv(self, x):
        return 0 if self.iszero(x) else 1

    def err(self, x, ref):
        return 0.0 if self.iszero(x) else 1.0


class FieldMp:
    name = 'mpmath'

    def __init__(self, seed, dps=50):
        import mpmath
        self.mp = mpmath
        mpmath.mp.dps = dps
        self.rng = random.Random(seed)
        self.zero = mpmath.mpc(0)
        self.one = mpmath.mpc(1)
        self.dps = dps

    def num(self, x):
        if isinstance(x, Fraction):
            return self.mp.mpc(self.mp.mpf(x.numerator) / x.denominator)
        if isinstance(x, complex):
            return self.mp.mpc(x.real, x.imag)
        return self.mp.mpc(x)

    def rand(self):
        return self.mp.mpc(self.rng.uniform(-1, 1), self.rng.uniform(-1, 1))

    def rand_small(self):
        return self.mp.mpc(self.rng.uniform(-3, 3))

    def sqrt(self, x):
        return self.mp.sqrt(x)

    def iszero(self, x, scale=1):
        return abs(x) <= self.mp.mpf(10) ** (-(self.dps // 2)) * max(1, abs(scale))

    def piv(self, x):
        return abs(x)

    def err(self, x, ref):
        return float(abs(x) / max(abs(ref), self.mp.mpf(10) ** -300))


F = None  # active field


def set_field(f):
    global F
    F = f


# ------------------------------------------------------------------ vectors / linear algebra

def dot(a, b):
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def vadd(*vs):
    return tuple(sum((v[i] for v in vs[1:]), vs[0][i]) for i in range(4))


def vsub(a, b):
    return tuple(a[i] - b[i] for i in range(4))


def vsc(c, a):
    return tuple(c * a[i] for i in range(4))


def lower(a):
    return (a[0], -a[1], -a[2], -a[3])


def solve(A, b):
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: F.piv(M[r][col]))
        if F.iszero(M[piv][col]):
            raise ZeroDivisionError("singular system")
        M[col], M[piv] = M[piv], M[col]
        inv = F.one / M[col][col]
        M[col] = [x * inv for x in M[col]]
        for r in range(n):
            if r != col and not F.iszero(M[r][col]):
                f = M[r][col]
                M[r] = [x - f * y for x, y in zip(M[r], M[col])]
    return [M[i][n] for i in range(n)]


def nullspace(rows, dim=4):
    """Basis of {u : rows . u = 0} (plain contraction)."""
    M = [list(r) for r in rows]
    pivcols = []
    r = 0
    for col in range(dim):
        piv = None
        for i in range(r, len(M)):
            if not F.iszero(M[i][col]):
                piv = i
                break
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        inv = F.one / M[r][col]
        M[r] = [x * inv for x in M[r]]
        for i in range(len(M)):
            if i != r and not F.iszero(M[i][col]):
                f = M[i][col]
                M[i] = [x - f * y for x, y in zip(M[i], M[r])]
        pivcols.append(col)
        r += 1
    free = [c for c in range(dim) if c not in pivcols]
    basis = []
    for fc in free:
        u = [F.zero] * dim
        u[fc] = F.one
        for i, pc in enumerate(pivcols):
            u[pc] = -M[i][fc]
        basis.append(tuple(u))
    return basis


def transverse(ks):
    return nullspace([lower(k) for k in ks])


def null_pair(u, w, target=None):
    """Two null vectors spanning span(u,w); e4 rescaled so that e3.e4 = target."""
    uu, uw, ww = dot(u, u), dot(u, w), dot(w, w)
    if F.iszero(ww):
        u, w = w, u
        uu, ww = ww, uu
    disc = uw * uw - uu * ww
    sq = F.sqrt(disc)
    e3 = vadd(u, vsc((-uw + sq) / ww, w))
    e4 = vadd(u, vsc((-uw - sq) / ww, w))
    if target is not None:
        e4 = vsc(target / dot(e3, e4), e4)
    return e3, e4


# ------------------------------------------------------------------ Laurent series in t (dicts)

def lmul(a, b, lo):
    out = {}
    for pa, ca in a.items():
        for pb, cb in b.items():
            p = pa + pb
            if p < lo:
                continue
            v = ca * cb
            if p in out:
                out[p] = out[p] + v
            else:
                out[p] = v
    return out


def ladd_into(acc, a, c=None):
    for p, v in a.items():
        if c is not None:
            v = c * v
        acc[p] = acc[p] + v if p in acc else v


def ldiv(f, d0, d1, d2, lo):
    """g = f/(d0 t + d1 + d2/t) for powers >= lo (needs f exact for powers >= lo+1)."""
    if not f:
        return {}
    top = max(f) - 1
    g = {}
    inv0 = F.one / d0
    for p in range(top, lo - 1, -1):
        v = f.get(p + 1, F.zero)
        if p + 1 in g:
            v = v - d1 * g[p + 1]
        if p + 2 in g:
            v = v - d2 * g[p + 2]
        g[p] = v * inv0
    return g


# ------------------------------------------------------------------ numerator

class Numerator:
    """N(q, mu2) = sum c * q0^e0 q1^e1 q2^e2 q3^e3 * mu2^k (contravariant components)."""

    def __init__(self, terms, rank):
        self.terms = terms  # list of ((e0,e1,e2,e3), k, c)
        self.rank = rank
        # group by q exponent
        g = {}
        for e, k, c in terms:
            g.setdefault(e, []).append((k, c))
        self.group = sorted(g.items(), key=lambda it: it[0])
        self.maxk = max((k for _, k, _ in terms), default=0)

    @staticmethod
    def random(rank, with_mu=True, field=None):
        terms = []
        for k in range(0, rank // 2 + 1 if with_mu else 1):
            d = rank - 2 * k
            for e in itertools.product(range(d + 1), repeat=4):
                if sum(e) <= d:
                    terms.append((e, k, F.rand()))
        return Numerator(terms, rank)

    def eval(self, q, mu2):
        mp = [F.one]
        for _ in range(self.maxk):
            mp.append(mp[-1] * mu2)
        pw = [[F.one] for _ in range(4)]
        for c in range(4):
            for _ in range(self.rank):
                pw[c].append(pw[c][-1] * q[c])
        tot = F.zero
        for e, lst in self.group:
            cf = F.zero
            for k, c in lst:
                cf = cf + c * mp[k]
            tot = tot + cf * pw[0][e[0]] * pw[1][e[1]] * pw[2][e[2]] * pw[3][e[3]]
        return tot

    def series(self, A, B, C, mu2, lo):
        """Laurent series of N(A + B t + C/t, mu2) for powers >= lo (mu2 numeric)."""
        r = self.rank
        mp = [F.one]
        for _ in range(self.maxk):
            mp.append(mp[-1] * mu2)
        comps = [{0: A[c], 1: B[c], -1: C[c]} for c in range(4)]
        pw = [[{0: F.one}] for _ in range(4)]
        for c in range(4):
            for e in range(1, r + 1):
                pw[c].append(lmul(pw[c][-1], comps[c], lo - (r - e)))
        cache01 = {}
        cache012 = {}
        tot = {}
        for e, lst in self.group:
            cf = F.zero
            for k, c in lst:
                cf = cf + c * mp[k]
            d01 = e[0] + e[1]
            key01 = (e[0], e[1])
            if key01 not in cache01:
                cache01[key01] = lmul(pw[0][e[0]], pw[1][e[1]], lo - (r - d01))
            key012 = (e[0], e[1], e[2])
            if key012 not in cache012:
                cache012[key012] = lmul(cache01[key01], pw[2][e[2]], lo - (r - d01 - e[2]))
            f = lmul(cache012[key012], pw[3][e[3]], lo)
            ladd_into(tot, f, cf)
        return tot


# ------------------------------------------------------------------ residues

def mono_set(kind, R):
    """Ninja-style monomial set: list of (j, powers)."""
    out = []
    if R < 0:
        return out
    if kind == 5:
        return [(1, ())]
    if kind == 4:
        for j in range(R // 2 + 1):
            out.append((j, (0,)))
            if 1 + 2 * j <= R:
                out.append((j, (1,)))
        return out
    if kind == 3:
        for j in range(R // 2 + 1):
            out.append((j, (0, 0)))
            for b in range(1, R - 2 * j + 1):
                out.append((j, (b, 0)))
                out.append((j, (0, b)))
        return out
    if kind == 2:  # powers of (e2, e3, e4)
        for j in range(R // 2 + 1):
            for a in range(R - 2 * j + 1):
                out.append((j, (a, 0, 0)))
                for b in range(1, R - 2 * j - a + 1):
                    out.append((j, (a, b, 0)))
                    out.append((j, (a, 0, b)))
        return out
    if kind == 1:  # powers of (e1, e2, e3, e4), no e1*e2 products
        for j in range(R // 2 + 1):
            for p in itertools.product(range(R - 2 * j + 1), repeat=4):
                if sum(p) + 2 * j <= R and not (p[0] > 0 and p[1] > 0):
                    out.append((j, p))
        return out
    raise ValueError(kind)


class Residue:
    def __init__(self, S, kind, R, V0, vecs):
        self.S = S
        self.kind = kind
        self.R = R
        self.V0 = V0
        self.vecs = vecs  # basis vectors u_i (qbar.u_i)
        self.monos = mono_set(kind, R)
        self.c = {}

    def eval(self, q, mu2):
        qb = vadd(q, self.V0)
        s = [dot(qb, u) for u in self.vecs]
        tot = F.zero
        for (j, p), c in self.c.items():
            v = c * mu2 ** j if j else c
            for si, pi in zip(s, p):
                if pi:
                    v = v * si ** pi
            tot = tot + v
        return tot

    def series(self, A, B, C, mu2, lo):
        """Delta(q(t), mu2) with q(t) = A + B t + C/t, powers >= lo."""
        Ab = vadd(A, self.V0)
        sers = [{0: dot(Ab, u), 1: dot(B, u), -1: dot(C, u)} for u in self.vecs]
        R = self.R
        pw = []
        for s in sers:
            lst = [{0: F.one}]
            for e in range(1, R + 1):
                lst.append(lmul(lst[-1], s, lo - (R - e)))
            pw.append(lst)
        tot = {}
        for (j, p), c in self.c.items():
            f = {0: c * mu2 ** j if j else c}
            for i, pi in enumerate(p):
                if pi:
                    f = lmul(f, pw[i][pi], lo - R)
            ladd_into(tot, f)
        return {k: v for k, v in tot.items() if k >= lo}


# ------------------------------------------------------------------ the reduction

class Reduction:
    def __init__(self, n, V, m, N, tadpole_basis='common', beta_box=True, verbose=False):
        self.n, self.V, self.m, self.N = n, V, m, N
        self.r = N.rank
        self.res = {}
        self.verbose = verbose
        self.beta_box = beta_box
        self.tadpole_basis = tadpole_basis
        self.info = {}
        # Ninja-like fixed references (rational null vectors)
        self.e2ref = tuple(F.num(x) for x in (7, 2, 3, 6))      # 49-4-9-36 = 0
        e1t = tuple(F.num(x) for x in (3, 1, 2, 2))             # 9-1-4-4 = 0
        e2t = tuple(F.num(x) for x in (3, -2, 2, -1))           # 9-4-4-1 = 0
        self.tadbasis_common = self.make_tadbasis(e1t, e2t)

    def make_tadbasis(self, e1, e2):
        mp12 = dot(e1, e2)
        u, w = transverse([e1, e2])
        e3, e4 = null_pair(u, w, -mp12)
        return (e1, e2, e3, e4)

    # --- helpers
    def R(self, k):
        return self.r - self.n + k

    def geom(self, S):
        i1 = S[0]
        V0, m0 = self.V[i1], self.m[i1]
        ks = [vsub(self.V[j], V0) for j in S[1:]]
        fs = [dot(k, k) + m0 - self.m[j] for k, j in zip(ks, S[1:])]
        uncut = [l for l in range(self.n) if l not in S]
        return V0, m0, ks, fs, uncut

    def lin_den(self, S, l):
        """(kk, f): on the cut of S, D_l = 2 qbar.kk + f."""
        V0, m0 = self.V[S[0]], self.m[S[0]]
        kk = vsub(self.V[l], V0)
        return kk, dot(kk, kk) + m0 - self.m[l]

    def D(self, l, q, mu2):
        p = vadd(q, self.V[l])
        return dot(p, p) - self.m[l] - mu2

    def parents(self, S):
        return [H for H in self.res if len(H) > len(S) and set(S) <= set(H)]

    def remainder_point(self, S, q, mu2, upto=None):
        """[N - sum_{H > S} Delta_H prod_{l not in H} D_l] / prod_{l not in S} D_l at a point."""
        val = self.N.eval(q, mu2)
        den = F.one
        for l in range(self.n):
            if l not in S:
                den = den * self.D(l, q, mu2)
        val = val / den
        for H in self.parents(S):
            d = F.one
            for l in H:
                if l not in S:
                    d = d * self.D(l, q, mu2)
            val = val - self.res[H].eval(q, mu2) / d
        return val

    def series_remainder(self, S, P0, P1, P2, mu2, lo, subtract_levels=(3, 2)):
        """t-series of N/prod D_uncut - sum over parent triangles/bubbles (as in the recipe)."""
        V0, m0, ks, fs, uncut = self.geom(S)
        m = len(uncut)
        A = vsub(P0, V0)
        f = self.N.series(A, P1, P2, mu2, lo + m)
        for l in uncut:
            kk, fl = self.lin_den(S, l)
            f = ldiv(f, 2 * dot(P1, kk), 2 * dot(P0, kk) + fl, 2 * dot(P2, kk), lo + m - 1)
            m -= 1
        for H in self.parents(S):
            if len(H) not in subtract_levels:
                continue
            extra = [l for l in H if l not in S]
            g = self.res[H].series(A, P1, P2, mu2, lo + len(extra))
            mm = len(extra)
            for l in extra:
                kk, fl = self.lin_den(S, l)
                g = ldiv(g, 2 * dot(P1, kk), 2 * dot(P0, kk) + fl, 2 * dot(P2, kk), lo + mm - 1)
                mm -= 1
            for p, v in g.items():
                if p >= lo:
                    f[p] = f.get(p, F.zero) - v
        return {p: v for p, v in f.items() if p >= lo}

    # --- pentagon
    def pentagon(self, S):
        V0, m0, ks, fs, uncut = self.geom(S)
        qb = solve([[2 * x for x in lower(k)] for k in ks], [-f for f in fs])
        qb = tuple(qb)
        mu2 = dot(qb, qb) - m0
        res = Residue(S, 5, 7, V0, [])
        q = vsub(qb, V0)
        val = self.N.eval(q, mu2)
        for l in uncut:
            val = val / self.D(l, q, mu2)
        res.c[(1, ())] = val / mu2
        self.res[S] = res
        return res

    # --- box
    def box_geometry(self, S):
        V0, m0, ks, fs, uncut = self.geom(S)
        G3 = [[2 * dot(ki, kj) for ki in ks] for kj in ks]
        al = solve(G3, [-f for f in fs])
        lc = vadd(vsc(al[0], ks[0]), vsc(al[1], ks[1]), vsc(al[2], ks[2]))
        v = transverse(ks)[0]
        return V0, m0, uncut, lc, v

    def box_line_poly(self, S, lc, v, mu_shift):
        """Coefficients (descending, top 3) of N(lc - V0 + tau v, mu2 = tau^2 v^2 + mu_shift)/prod D_uncut."""
        V0, m0, ks, fs, uncut = self.geom(S)
        r = self.r
        v2 = dot(v, v)
        # N along the line is a polynomial of degree r in tau: interpolate exactly
        taus = [F.num(i + 2) for i in range(r + 1)]
        vals = [self.N.eval(vsub(vadd(lc, vsc(t, v)), V0), t * t * v2 + mu_shift) for t in taus]
        Vm = [[t ** e for e in range(r + 1)] for t in taus]
        coef = solve(Vm, vals)  # ascending
        desc = list(reversed(coef))  # tau^r, tau^{r-1}, ...
        top = desc[:3]
        while len(top) < 3:
            top.append(F.zero)
        n0, n1, n2 = top
        for l in uncut:
            kk, fl = self.lin_den(S, l)
            d0 = 2 * dot(v, kk)
            d1 = 2 * dot(lc, kk) + fl
            n0 = n0 / d0
            n1 = (n1 - d1 * n0) / d0
            n2 = (n2 - d1 * n1) / d0
        return n0, n1, n2

    def box(self, S):
        V0, m0, uncut, lc, v = self.box_geometry(S)
        R = self.R(4)
        res = Residue(S, 4, R, V0, [v])
        v2 = dot(v, v)
        mu0 = dot(lc, lc) - m0
        c = {}
        # c0, c1 from the two mu2 = 0 solutions
        tau0 = F.sqrt(-mu0 / v2)
        vals = []
        for sg in (1, -1):
            qb = vadd(lc, vsc(sg * tau0, v))
            q = vsub(qb, V0)
            val = self.N.eval(q, F.zero)
            for l in uncut:
                val = val / self.D(l, q, F.zero)
            vals.append(val)
        c[(0, (0,))] = (vals[0] + vals[1]) / 2
        if R >= 1:
            c[(0, (1,))] = (vals[0] - vals[1]) / (2 * tau0 * v2)
        # mu-expansion (top three terms)
        if R >= 4:
            shift = mu0 if self.beta_box else F.zero
            n0, n1, n2 = self.box_line_poly(S, lc, v, shift)
            if R == 6:
                c[(3, (0,))] = n0 / v2 ** 3
                c[(2, (1,))] = n1 / v2 ** 3
                c[(2, (0,))] = n2 / v2 ** 2 - 3 * mu0 * c[(3, (0,))]
            elif R == 5:
                c[(2, (1,))] = n0 / v2 ** 3
                c[(2, (0,))] = n1 / v2 ** 2
            elif R == 4:
                c[(2, (0,))] = n0 / v2 ** 2
            else:
                raise NotImplementedError(R)
            if not self.beta_box:
                # Ninja's convention mu2 = tau^2 v^2 (no shift) for comparison
                pass
        # remaining (spurious) coefficients from finite-mu2 samples, pentagons subtracted
        unknown_even = [j for j in range(R // 2 + 1) if (j, (0,)) not in c]
        unknown_odd = [j for j in range((R - 1) // 2 + 1) if R >= 1 and (j, (1,)) not in c]
        nunk = max(len(unknown_even), len(unknown_odd))
        rowsE, rhsE, rowsO, rhsO = [], [], [], []
        for _ in range(nunk):
            ts = F.rand()
            s = ts * ts * v2 + mu0
            vv = []
            for sg in (1, -1):
                qb = vadd(lc, vsc(sg * ts, v))
                q = vsub(qb, V0)
                vv.append(self.remainder_point(S, q, s))
            Pe = (vv[0] + vv[1]) / 2
            Qo = (vv[0] - vv[1]) / (2 * ts * v2)
            for j in range(R // 2 + 1):
                if (j, (0,)) in c:
                    Pe = Pe - c[(j, (0,))] * s ** j
            for j in range((R - 1) // 2 + 1):
                if R >= 1 and (j, (1,)) in c:
                    Qo = Qo - c[(j, (1,))] * s ** j
            rowsE.append([s ** j for j in unknown_even])
            rhsE.append(Pe)
            rowsO.append([s ** j for j in unknown_odd])
            rhsO.append(Qo)
        if unknown_even:
            sol = solve(rowsE[:len(unknown_even)], rhsE[:len(unknown_even)])
            for j, x in zip(unknown_even, sol):
                c[(j, (0,))] = x
        if unknown_odd:
            sol = solve(rowsO[:len(unknown_odd)], rhsO[:len(unknown_odd)])
            for j, x in zip(unknown_odd, sol):
                c[(j, (1,))] = x
        res.c = c
        self.res[S] = res
        self.info.setdefault('box', {})[S] = dict(mu0=mu0, v2=v2)
        return res

    # --- triangle
    def triangle_geometry(self, S):
        V0, m0, ks, fs, uncut = self.geom(S)
        G2 = [[2 * dot(ki, kj) for ki in ks] for kj in ks]
        al = solve(G2, [-f for f in fs])
        a = vadd(vsc(al[0], ks[0]), vsc(al[1], ks[1]))
        u, w = transverse(ks)
        e3, e4 = null_pair(u, w)
        return V0, m0, uncut, a, e3, e4

    def fit_levels(self, samples, levels, monos_of_level, nverify=3):
        """samples: list of (point-dict of variables, series dict). Fit polynomials per level.
        monos_of_level(b) -> list of exponent tuples; variables given as tuple values."""
        out = {}
        for b in levels:
            monos = monos_of_level(b)
            if not monos:
                out[b] = {}
                continue
            k = len(monos)
            assert len(samples) >= k + nverify
            rows = [[self._mono_val(x, mo) for mo in monos] for x, _ in samples[:k]]
            rhs = [s.get(b, F.zero) for _, s in samples[:k]]
            sol = solve(rows, rhs)
            for x, s in samples[k:k + nverify]:
                pred = sum((cc * self._mono_val(x, mo) for cc, mo in zip(sol, monos)), F.zero)
                got = s.get(b, F.zero)
                if not F.iszero(pred - got, got):
                    raise AssertionError(f"level t^{b}: remainder not of the claimed form "
                                         f"(err {F.err(pred - got, got)})")
            out[b] = dict(zip(monos, sol))
        return out

    @staticmethod
    def _mono_val(x, mo):
        v = None
        for xi, ei in zip(x, mo):
            t = xi ** ei if ei else None
            if t is not None:
                v = t if v is None else v * t
        return v if v is not None else F.one

    def triangle(self, S):
        V0, m0, uncut, a, e3, e4 = self.triangle_geometry(S)
        R = self.R(3)
        res = Residue(S, 3, R, V0, [e3, e4])
        mp34 = dot(e3, e4)
        beta = m0 - dot(a, a)
        lv = {}
        for fam, (u1, u2) in (('A', (e3, e4)), ('B', (e4, e3))):
            samples = []
            for _ in range(R // 2 + 1 + 3):
                M = F.rand()
                P2 = vsc((beta + M) / (2 * mp34), u2)
                s = self.series_remainder(S, a, u1, P2, M, 0, subtract_levels=())
                samples.append(((M,), s))
            lv[fam] = self.fit_levels(samples, range(R + 1), lambda b: [(j,) for j in range((R - b) // 2 + 1)])
        c = {}
        for b in range(R + 1):
            for (j,), val in lv['A'][b].items():
                if b == 0:
                    c[(j, (0, 0))] = (val + lv['B'][0][(j,)]) / 2
                else:
                    c[(j, (0, b))] = val / mp34 ** b
            if b >= 1:
                for (j,), val in lv['B'][b].items():
                    c[(j, (b, 0))] = val / mp34 ** b
        res.c = c
        self.res[S] = res
        self.info.setdefault('tri_t0_AminusB', {})[S] = {j: lv['A'][0][(j,)] - lv['B'][0][(j,)] for (j,) in lv['A'][0]}
        return res

    # --- bubble
    def bubble_geometry(self, S):
        V0, m0, ks, fs, uncut = self.geom(S)
        k, f = ks[0], fs[0]
        e2 = self.e2ref
        r1 = dot(k, k) / (2 * dot(k, e2))
        e1 = vsub(k, vsc(r1, e2))
        mp12 = dot(e1, e2)
        u, w = transverse([k, e2])
        e3, e4 = null_pair(u, w, -mp12)
        b0 = -f / (2 * mp12)
        a0 = vsc(b0, e2)
        a1 = vsub(e1, vsc(r1, e2))
        return V0, m0, uncut, e1, e2, e3, e4, a0, a1, mp12

    def bubble(self, S):
        V0, m0, uncut, e1, e2, e3, e4, a0, a1, mp12 = self.bubble_geometry(S)
        R = self.R(2)
        res = Residue(S, 2, R, V0, [e2, e3, e4])
        mp34 = dot(e3, e4)
        lv = {}

        def monos(b):
            return [(a, j) for j in range((R - b) // 2 + 1) for a in range(R - b - 2 * j + 1)]
        nmax = len(monos(0))
        for fam, (u1, u2) in (('A', (e3, e4)), ('B', (e4, e3))):
            samples = []
            for _ in range(nmax + 3):
                x, M = F.rand(), F.rand()
                P0 = vadd(a0, vsc(x, a1))
                beta = m0 - dot(P0, P0)
                P2 = vsc((beta + M) / (2 * mp34), u2)
                s = self.series_remainder(S, P0, u1, P2, M, 0 if fam == 'A' else 1, subtract_levels=(3,))
                samples.append(((x, M), s))
            levels = range(R + 1) if fam == 'A' else range(1, R + 1)
            lv[fam] = self.fit_levels(samples, levels, monos)
        c = {}
        for b, d in lv['A'].items():
            for (a, j), val in d.items():
                c[(j, (a, 0, b))] = val / (mp12 ** a * mp34 ** b)
        for b, d in lv['B'].items():
            for (a, j), val in d.items():
                c[(j, (a, b, 0))] = val / (mp12 ** a * mp34 ** b)
        res.c = c
        self.res[S] = res
        self.info.setdefault('bub', {})[S] = dict(mp12=mp12, e2=e2, k=vsub(self.V[S[1]], V0))
        return res

    # --- tadpole (recipe gives c0, c14, c15 only)
    def tadpole_recipe(self, S, basis):
        V0, m0, ks, fs, uncut = self.geom(S)
        e1, e2, e3, e4 = basis
        mp12, mp34 = dot(e1, e2), dot(e3, e4)
        R = self.R(1)
        out = {}
        zero = tuple(F.zero for _ in range(4))
        for fam, (u1, u2, mp) in (('A', (e1, e2, mp12)), ('B', (e3, e4, mp34))):
            samples = []
            for _ in range(R // 2 + 1 + 3):
                M = F.rand()
                P2 = vsc((m0 + M) / (2 * mp), u2)
                s = self.series_remainder(S, zero, u1, P2, M, 0, subtract_levels=(3, 2))
                samples.append(((M,), s))
            out[fam] = self.fit_levels(samples, [0], lambda b: [(j,) for j in range(R // 2 + 1)])[0]
        c0 = out['A'][(0,)]
        c14 = out['A'].get((1,), F.zero)
        c15 = 2 * (out['B'].get((1,), F.zero) - c14) / mp34
        consistency = out['B'][(0,)] - (c0 + c15 * m0 * mp34 / 2)
        return dict(c0=c0, c14=c14, c15=c15, consistency=consistency, mp34=mp34)

    # --- generic fit of any residue from on-cut points (independent of the Laurent recipe)
    def cut_point(self, S, geomcache):
        V0, m0, ks, fs, uncut = self.geom(S)
        k = len(S)
        if k == 5:
            qb = tuple(solve([[2 * x for x in lower(kk)] for kk in ks], [-f for f in fs]))
        elif k == 4:
            _, _, _, lc, v = geomcache
            qb = vadd(lc, vsc(F.rand(), v))
        elif k == 3:
            _, _, _, a, e3, e4 = geomcache
            qb = vadd(a, vsc(F.rand(), e3), vsc(F.rand(), e4))
        elif k == 2:
            _, _, _, e1, e2, e3, e4, a0, a1, _ = geomcache
            qb = vadd(a0, vsc(F.rand(), a1), vsc(F.rand(), e3), vsc(F.rand(), e4))
        else:
            qb = tuple(F.rand() for _ in range(4))
        mu2 = dot(qb, qb) - m0
        return vsub(qb, V0), mu2

    def fit_residue(self, S, vecs, kind, extra=4):
        V0 = self.V[S[0]]
        R = self.R(kind)
        res = Residue(S, kind, R, V0, vecs)
        monos = res.monos
        if kind == 4:
            gc = self.box_geometry(S)
        elif kind == 3:
            gc = self.triangle_geometry(S)
        elif kind == 2:
            gc = self.bubble_geometry(S)
        else:
            gc = None
        rows, rhs = [], []
        for _ in range(len(monos) + extra):
            q, mu2 = self.cut_point(S, gc)
            qb = vadd(q, V0)
            s = [dot(qb, u) for u in vecs]
            row = []
            for j, p in monos:
                v = mu2 ** j if j else F.one
                for si, pi in zip(s, p):
                    if pi:
                        v = v * si ** pi
                row.append(v)
            rows.append(row)
            rhs.append(self.remainder_point(S, q, mu2))
        k = len(monos)
        sol = solve(rows[:k], rhs[:k])
        worst = 0.0
        for row, y in zip(rows[k:], rhs[k:]):
            pred = sum((a * b for a, b in zip(row, sol)), F.zero)
            worst = max(worst, F.err(pred - y, y))
        res.c = dict(zip(monos, sol))
        return res, worst

    # --- driver
    def run(self, check_fits=True, log=print):
        n = self.n
        summary = {}
        for k in (5, 4, 3, 2):
            if k > n or self.R(k) < 0:
                continue
            for S in itertools.combinations(range(n), k):
                if k == 5:
                    self.pentagon(S)
                    continue
                if k == 4:
                    rec = self.box(S)
                    gc = self.box_geometry(S)
                    vecs = [gc[4]]
                elif k == 3:
                    rec = self.triangle(S)
                    gc = self.triangle_geometry(S)
                    vecs = [gc[4], gc[5]]
                else:
                    rec = self.bubble(S)
                    gc = self.bubble_geometry(S)
                    vecs = [gc[4], gc[5], gc[6]]
                if check_fits:
                    fit, worst = self.fit_residue(S, vecs, k)
                    dmax = 0.0
                    for key in fit.monos:
                        a, b = rec.c.get(key, F.zero), fit.c[key]
                        dmax = max(dmax, F.err(a - b, b))
                    summary.setdefault(k, []).append((S, worst, dmax))
                    if worst > 1e-25 or dmax > 1e-20:
                        log(f"  MISMATCH k={k} S={S}: fit-consistency {worst:.2e}, recipe-vs-fit {dmax:.2e}")
        # tadpoles
        if self.R(1) >= 0:
            rng = random.Random(1234)
            for S in itertools.combinations(range(n), 1):
                if self.tadpole_basis == 'common':
                    basis = self.tadbasis_common
                else:
                    def rnull():
                        while True:
                            s_, u_ = F.rand_small(), F.rand_small()
                            w_ = s_ * s_ + u_ * u_
                            return (1 + w_, 2 * s_, 2 * u_, w_ - 1)
                    basis = self.make_tadbasis(rnull(), rnull())
                fit, worst = self.fit_residue(S, list(basis), 1)
                self.res[S] = fit
                rc = self.tadpole_recipe(S, basis)
                d = [F.err(rc['c0'] - fit.c[(0, (0, 0, 0, 0))], fit.c[(0, (0, 0, 0, 0))])]
                if self.R(1) >= 2:
                    d.append(F.err(rc['c14'] - fit.c[(1, (0, 0, 0, 0))], fit.c[(1, (0, 0, 0, 0))]))
                    d.append(F.err(rc['c15'] - fit.c[(0, (0, 0, 1, 1))], fit.c[(0, (0, 0, 1, 1))]))
                d.append(F.err(rc['consistency'], rc['c0']))
                summary.setdefault(1, []).append((S, worst, max(d)))
                if worst > 1e-25 or max(d) > 1e-20:
                    log(f"  MISMATCH tadpole S={S}: fit-consistency {worst:.2e}, recipe-vs-fit {max(d):.2e}")
        return summary

    def zero_point(self, npts=8):
        """Return samples of (q, mu2, F/prod D) with F = N - sum Delta_S prod D."""
        out = []
        for _ in range(npts):
            q = tuple(F.rand() for _ in range(4))
            mu2 = F.rand()
            val = self.N.eval(q, mu2)
            for S, res in self.res.items():
                d = F.one
                for l in range(self.n):
                    if l not in S:
                        d = d * self.D(l, q, mu2)
                val = val - res.eval(q, mu2) * d
            dall = F.one
            for l in range(self.n):
                dall = dall * self.D(l, q, mu2)
            out.append((q, mu2, val / dall))
        return out


def zero_point_degree(samples):
    """Smallest weighted degree d (q weight 1, mu2 weight 2) such that F/prod D fits a
    polynomial of weighted degree <= d (checked on extra points); returns (d, coeffs)."""
    for d in range(0, 4):
        monos = []
        for k in range(d // 2 + 1):
            for e in itertools.product(range(d + 1), repeat=4):
                if sum(e) + 2 * k <= d:
                    monos.append((e, k))
        if len(monos) + 2 > len(samples):
            return None
        rows = []
        for q, mu2, _ in samples:
            row = []
            for e, k in monos:
                v = mu2 ** k if k else F.one
                for c in range(4):
                    if e[c]:
                        v = v * q[c] ** e[c]
                row.append(v)
            rows.append(row)
        kk = len(monos)
        try:
            sol = solve(rows[:kk], [s[2] for s in samples[:kk]])
        except ZeroDivisionError:
            continue
        ok = True
        for row, s in zip(rows[kk:], samples[kk:]):
            pred = sum((a * b for a, b in zip(row, sol)), F.zero)
            if not F.iszero(pred - s[2], s[2]):
                ok = False
        if ok:
            allzero = all(F.iszero(x) for x in sol)
            return d, allzero
    return None
