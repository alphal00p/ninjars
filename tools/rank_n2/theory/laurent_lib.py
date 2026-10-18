"""Numerical (mpmath) building blocks of Ninja's Laurent-expansion reduction,
generalised to arbitrary rank (used here for r <= n+2).

A "Laurent coefficient" is a dict {(it, ix, imu): value} meaning t^it x^ix mu2^imu.
"""
import itertools
import mpmath as mp

METRIC = (1, -1, -1, -1)


def mpdot(a, b):
    return sum(METRIC[i] * a[i] * b[i] for i in range(4))


def vadd(a, b):
    return [a[i] + b[i] for i in range(4)]


def vsub(a, b):
    return [a[i] - b[i] for i in range(4)]


def vsc(s, a):
    return [s * a[i] for i in range(4)]


def tompf(x):
    return mp.mpf(int(x.numerator)) / int(x.denominator)


def tovec(v):
    return [tompf(c) for c in v]


# ---------------------------------------------------------------- dict polynomials in (x, mu2)

def padd(a, b, s=1):
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + s * v
    return out


def pmul(a, b):
    out = {}
    for ka, va in a.items():
        for kb, vb in b.items():
            k = (ka[0] + kb[0], ka[1] + kb[1])
            out[k] = out.get(k, 0) + va * vb
    return out


def pscale(a, s):
    return {k: s * v for k, v in a.items()}


# z-series: list of poly-dicts, index = power of z (z = 1/t)

def smul(a, b, depth):
    out = [dict() for _ in range(depth + 1)]
    for i, ai in enumerate(a):
        if i > depth or not ai:
            continue
        for j, bj in enumerate(b):
            if i + j > depth:
                break
            if bj:
                out[i + j] = padd(out[i + j], pmul(ai, bj))
    return out


class Param:
    """q(t,x,mu2) = t*E + C0 + x*C1 + (1/t) * F(x,mu2),  F = (f0 + f1 x + f2 x^2 + mu2) * G.

    Laurent components per Minkowski component i:
       z^0 : E_i ;  z^1 : C0_i + x C1_i ;  z^2 : F_i(x,mu2)
    (q = (1/z) [E + z C + z^2 F]).  For T3-type (triangle/tadpole) C1 = 0, f1=f2=0.
    """

    def __init__(self, E, C0, C1, G, f):
        self.E, self.C0, self.C1, self.G, self.f = E, C0, C1, G, f
        self.comp = []
        for i in range(4):
            s0 = {(0, 0): E[i]}
            s1 = {(0, 0): C0[i], (1, 0): C1[i]}
            Fx = {(0, 0): f[0] * G[i], (1, 0): f[1] * G[i], (2, 0): f[2] * G[i], (0, 1): G[i]}
            self.comp.append([s0, s1, Fx])

    def linear_form(self, u, const=0):
        """(q.u + const) as z-series components [z^0, z^1, z^2] (times 1/z)."""
        return [{(0, 0): mpdot(self.E, u)},
                {(0, 0): mpdot(self.C0, u) + const, (1, 0): mpdot(self.C1, u)},
                {(0, 0): self.f[0] * mpdot(self.G, u), (1, 0): self.f[1] * mpdot(self.G, u),
                 (2, 0): self.f[2] * mpdot(self.G, u), (0, 1): mpdot(self.G, u)}]


def expand_polynomial(monos, param, tmin, shift=None):
    """Laurent expansion in t of P(q, mu2) = sum c * q0^a0..q3^a3 mu2^j on the
    parametrisation q = param(t,x,mu2) - shift ... here q is the loop momentum
    (not qbar): the param vectors already include -V0.  Returns {(it,ix,imu): val}
    for it >= tmin."""
    maxd = max((sum(e[:4]) for e, c in monos), default=0)
    depth_max = maxd - tmin
    if depth_max < 0:
        return {}
    cache = {}

    def power(i, e, depth):
        key = (i, e)
        if key not in cache:
            if e == 0:
                cache[key] = [{(0, 0): mp.mpf(1)}] + [dict() for _ in range(depth_max)]
            else:
                cache[key] = smul(power(i, e - 1, depth_max), param.comp[i], depth_max)
        return [d for d in cache[key][:depth + 1]]

    out = {}
    for e, c in monos:
        d = sum(e[:4])
        depth = d - tmin
        if depth < 0:
            continue
        ser = [{(0, 0): c}] + [dict() for _ in range(depth)]
        for i in range(4):
            if e[i]:
                ser = smul(ser, power(i, e[i], depth), depth)
        j = e[4]
        for s, coeff in enumerate(ser):
            it = d - s
            for (ix, imu), v in coeff.items():
                k = (it, ix, imu + j)
                out[k] = out.get(k, 0) + v
    return out


def levels(lau):
    """Group a Laurent dict by t power: {it: {(ix,imu): v}}."""
    out = {}
    for (it, ix, imu), v in lau.items():
        out.setdefault(it, {})[(ix, imu)] = v
    return out


def unlevel(lv):
    out = {}
    for it, d in lv.items():
        for (ix, imu), v in d.items():
            out[(it, ix, imu)] = v
    return out


def divide(lau, den, tmin):
    """Divide a Laurent series (known from its top power down to tmin) by a
    denominator D = t*d1 + d0(x) + (1/t) dm1(x,mu2) (exact on the cut).
    den = (d1 scalar, d0 poly-dict, dm1 poly-dict). Returns quotient levels from
    top-1 down to tmin-1 ... we return only powers >= tmin - 1 that are fully
    determined, i.e. quotient powers p with numerator powers p+1 >= tmin."""
    lv = levels(lau)
    if not lv:
        return {}
    top = max(lv)
    d1, d0, dm1 = den
    q = {}
    for p in range(top - 1, tmin - 2, -1):
        acc = dict(lv.get(p + 1, {}))
        if p + 1 in q:
            acc = padd(acc, pmul(d0, q[p + 1]), -1)
        if p + 2 in q:
            acc = padd(acc, pmul(dm1, q[p + 2]), -1)
        q[p] = pscale(acc, 1 / d1)
    return unlevel({p: v for p, v in q.items() if p >= tmin - 1})


def den_on_param(param, Vl, ml2, V0, m02):
    """Uncut denominator D_l = 2 qbar.kk + f on the cut, kk = V_l - V0,
    f = kk^2 + m0^2 - ml^2, qbar = q + V0."""
    kk = vsub(Vl, V0)
    f = mpdot(kk, kk) + m02 - ml2
    lf = param.linear_form(kk, const=mpdot(V0, kk))  # (q+V0).kk
    d1 = 2 * lf[0][(0, 0)]
    d0 = {k: 2 * v for k, v in lf[1].items()}
    d0[(0, 0)] = d0.get((0, 0), 0) + f
    dm1 = {k: 2 * v for k, v in lf[2].items()}
    return d1, d0, dm1


def poly_part(lau):
    return {k: v for k, v in lau.items() if k[0] >= 0}


def null_pair_in_plane(n1, n2):
    """Two null vectors e3,e4 spanning span(n1,n2) (complex in general)."""
    G11, G12, G22 = mpdot(n1, n1), mpdot(n1, n2), mpdot(n2, n2)
    if abs(G11) > abs(G22):
        # alpha^2 G11 + 2 alpha G12 + G22 = 0 for e = alpha n1 + n2
        disc = mp.sqrt(G12 * G12 - G11 * G22)
        a1, a2 = (-G12 + disc) / G11, (-G12 - disc) / G11
        e3 = vadd(vsc(a1, n1), n2)
        e4 = vadd(vsc(a2, n1), n2)
    else:
        disc = mp.sqrt(G12 * G12 - G11 * G22)
        b1, b2 = (-G12 + disc) / G22, (-G12 - disc) / G22
        e3 = vadd(n1, vsc(b1, n2))
        e4 = vadd(n1, vsc(b2, n2))
    return e3, e4
