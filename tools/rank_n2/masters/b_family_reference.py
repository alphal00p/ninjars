"""Feynman-parameter references of the finite parts of B0, B1, B11, B111, B1111
at 200 digits, for the arbitrary-precision tests of SymbolicaMasters.

Writes crates/ninjars-symbolica/tests/rank_n2_data/b_family_feynman.txt.

B_n(p2; m0, m1) finite = (-1)^n * ( - int_0^1 x^n log((D(x) - i0)/mu2) dx ),
D(x) = x m1 + (1-x) m0 - x(1-x) p2, Im m <= 0 (OneLOop's normalisation; the
poles are (-1)^n/(n+1) and 0, all zero for the scaleless bubble).

The quadrature is that of the S1 verification of the native rank-four bubble
(scratchpad verify-s1/py/reference.py): mpmath tanh-sinh, Re D in factored form
p2 (x - r1)(x - r2) (no cancellation near its zeros), Im D = x Im m1 +
(1-x) Im m0 (both terms <= 0, so the -i0 side is never lost to rounding),
[0, 1] split at the zeros of Re D and at Re(root) +- k |Im root| of the complex
roots of D, every row rescaled exactly to O(1) by a power of two. Every row is
evaluated twice at DIGITS and at CHECK_DIGITS; the largest difference relative
to the row's largest coefficient is printed (6.5e-157 for the committed file).

Rows (all inputs binary64 numbers, taken exactly): p2 = 0 (unequal, equal,
nearly equal, one massless, complex masses), |p2|/m2 = 1e-1 .. 1e-14 of both
signs for unequal, equal, complex, one-massless and heavy (m2 = 29929, 30276)
mass pairs, the dyadic ladder p2 = 2^-16 .. 2^-46, thresholds and
pseudo-thresholds exactly and at 2^-20 and 2^-40 relative distance, on-shell,
generic, large-p2 and complex-mass rows, the scaleless bubble, and the rows of
b1111_rows() in tests/masters.rs. Unequal-mass rows are followed by their mass
swap (label suffix ~sw), which takes the other branch of OneLOop's mass order.

File format, one row per line: label, then the binary64 bits (16 hex digits) of
p2, Re m0, Im m0, Re m1, Im m1, mu2, then Re and Im of the five finite parts
B0, B1, B11, B111, B1111 rounded to 160 significant digits.

usage: tools/rank_n2/review/py.sh b_family_reference.py OUT.txt [nproc]
"""
import struct
import sys
from multiprocessing import Pool

import mpmath as mp

DIGITS = 200
CHECK_DIGITS = 170
OUTPUT_DIGITS = 160
RANKS = 5


def bits(x):
    return format(struct.unpack('<Q', struct.pack('<d', float(x)))[0], '016x')


def rows():
    out = []

    def add(region, p2, m0, m1, mu2=1.0, swap=True):
        m0 = complex(m0)
        m1 = complex(m1)
        i = len(out)
        out.append((f'{region}:{i}', p2, m0, m1, mu2))
        if swap and m0 != m1:
            out.append((f'{region}:{i}~sw', p2, m1, m0, mu2))

    add('p2zero', 0.0, 2.0, 5.0)
    add('p2zero', 0.0, 3.0, 3.0, 7.0)
    add('p2zero', 0.0, 1.0, 1.0 + 2.0**-30)
    add('p2zero', 0.0, 0.0, 2.0)
    add('p2zero', 0.0, 0.7 - 0.03j, 1.4 - 0.08j)
    for r in (1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 4e-13, 1e-14):
        for sign in (1.0, -1.0):
            add('smallp', sign * r, 1.0, 1.5)
            add('smallp', sign * r, 1.0, 1.0, swap=False)
            add('smallp', sign * r * 1.4, 0.7 - 0.03j, 1.4 - 0.08j)
            add('smallp', sign * r * 2.0, 0.0, 2.0)
            add('smallp', sign * r * 29929.0, 29929.0, 30276.0, 10000.0)
    for k in (16, 24, 32, 40, 46):
        add('ladder', 2.0**-k, 1.0, 1.5)
        add('ladder', 2.0**-k, 1.0, 1.0, swap=False)
    for t in (9.0, 1.0):
        add('thr', t, 1.0, 4.0)
        for e in (2.0**-20, 2.0**-40):
            add('thr', t * (1 + e), 1.0, 4.0)
            add('thr', t * (1 - e), 1.0, 4.0)
    add('thr', 4.0, 1.0, 1.0, swap=False)
    add('thr', 4.0 * (1 + 2.0**-30), 1.0, 1.0, swap=False)
    add('onshell', 2.0, 0.0, 2.0)
    add('generic', 3.0, 1.0, 2.0)
    add('generic', -2.5, 1.0, 1.5)
    add('generic', 10.0, 1.0, 2.0, 4.0)
    add('generic', 1e6, 1.0, 2.0)
    add('complex', 3.0, 0.7 - 0.03j, 1.4 - 0.08j)
    add('complex', -1.0, 1.0 - 0.25j, 0.5)
    add('complex', 8315.18, 8315.18 - 91.0j, 100.0, 10000.0)
    add('scaleless', 0.0, 0.0, 0.0, swap=False)
    for p2, m0, m1, mu2 in [
            (0., 2., 5., 1.), (0., 5., 2., 1.), (0., 3., 3., 7.), (-3., 0., 2., 1.),
            (5., 0., 2., 1.), (1., 2., 0., 1.), (2., 0., 2., 1.), (3., 1., 2., 1.),
            (10., 1., 2., 4.), (-2.5, 1., 1.5, 1.), (1e6, 1., 2., 1.),
            (3., 0.7 - 0.03j, 1.4 - 0.08j, 1.), (-1., 1. - 0.25j, 0.5, 2.)]:
        add('masters', p2, m0, m1, mu2, swap=False)
    for k in (4, 8, 12, 16, 20, 24):
        p = 0.5**k
        add('masters', p, 1., 1.5, swap=False)
        add('masters', -p, 1.5, 1., swap=False)
        add('masters', p, 1., 1., swap=False)
    return out


def quadratic_roots(a, b, c):
    """Roots of a x^2 + b x + c (a != 0), stable form, as mpc."""
    disc = mp.sqrt(mp.mpc(b) * b - 4 * a * c)
    q1 = -b - disc
    q2 = -b + disc
    q = q1 if abs(q1) >= abs(q2) else q2
    if q == 0:
        return [mp.mpc(0), mp.mpc(0)]
    return [q / (2 * a), 2 * c / q]


class Row:
    def __init__(self, p2, m0, m1, mu2):
        self.p2 = p2
        a0, a1 = m0.real, m1.real
        self.im0, self.im1 = m0.imag, m1.imag
        # Re D = p2 x^2 + (a1 - a0 - p2) x + a0
        if p2 == 0:
            self.slope = a1 - a0
            self.re_roots = [a0 / (a0 - a1)] if a1 != a0 else []
            self.const = a0
            self.roots = [m0 / (m0 - m1)] if m1 != m0 else []
        else:
            self.re_roots = quadratic_roots(p2, a1 - a0 - p2, a0)
            self.roots = quadratic_roots(p2, m1 - m0 - p2, m0)
        self.logmu = mp.log(mu2)

    def re_d(self, x):
        if self.p2 == 0:
            if not self.re_roots:
                return self.const
            return self.slope * (x - self.re_roots[0])
        r1, r2 = self.re_roots
        return mp.re(self.p2 * (x - r1) * (x - r2))

    def logd(self, x):
        re = self.re_d(x)
        im = x * self.im1 + (1 - x) * self.im0
        if im == 0:
            if re == 0:
                return None
            if re < 0:
                return mp.mpc(mp.log(-re), -mp.pi)
            return mp.mpc(mp.log(re), 0)
        return mp.log(mp.mpc(re, im))

    def breakpoints(self):
        pts = set()
        for r in self.re_roots:
            r = mp.mpc(r)
            if abs(r.imag) == 0 and 0 < r.real < 1:
                pts.add(+r.real)
        for r in self.roots:
            r = mp.mpc(r)
            re, im = r.real, abs(r.imag)
            if 0 < re < 1:
                pts.add(+re)
            if im > 0:
                for k in (1, 10, 100, 1000):
                    for s in (-1, 1):
                        q = re + s * k * im
                        if 0 < q < 1:
                            pts.add(+q)
        full = [mp.mpf(0)] + sorted(pts) + [mp.mpf(1)]
        extra = []
        for a, b in zip(full, full[1:]):
            if b - a > 0.25:
                n = int((b - a) / 0.25) + 1
                extra += [a + (b - a) * j / n for j in range(1, n)]
        return [mp.mpf(0)] + sorted(set(pts) | set(extra)) + [mp.mpf(1)]

    def integrals(self):
        pts = self.breakpoints()
        out = []
        for n in range(RANKS):
            def f(x, n=n):
                log = self.logd(x)
                if log is None:
                    return mp.mpc(0)
                return x ** n * (log - self.logmu)
            out.append((-1) ** n * (-mp.quad(f, pts, maxdegree=10)))
        return out


def finite_parts(row, digits):
    _, p2, m0, m1, mu2 = row
    with mp.workdps(digits):
        if p2 == 0 and m0 == 0 and m1 == 0:
            return [mp.mpc(0)] * RANKS
        # Homogeneity: rescale the row exactly to O(1) (a power of two).
        scale = max(abs(p2), abs(m0), abs(m1))
        e = mp.mpf(2) ** (-int(mp.floor(mp.log(scale, 2))))
        return Row(mp.mpf(p2) * e, mp.mpc(m0) * e, mp.mpc(m1) * e,
                   mp.mpf(mu2) * e).integrals()


def work(row):
    values = finite_parts(row, DIGITS)
    check = finite_parts(row, CHECK_DIGITS)
    with mp.workdps(DIGITS):
        top = max([abs(v) for v in values] + [mp.mpf(10) ** -DIGITS])
        difference = max(abs(a - b) for a, b in zip(values, check)) / top

        def fmt(x):
            return '0' if x == 0 else mp.nstr(x, OUTPUT_DIGITS, min_fixed=1, max_fixed=0)

        label, p2, m0, m1, mu2 = row
        fields = [label] + [bits(x) for x in (p2, m0.real, m0.imag, m1.real, m1.imag, mu2)]
        fields += [fmt(part) for v in values for part in (mp.re(v), mp.im(v))]
        return ' '.join(fields), difference


def main():
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    worst = mp.mpf(0)
    with Pool(nproc) as pool, open(sys.argv[1], 'w') as out:
        for line, difference in pool.imap(work, rows()):
            out.write(line + '\n')
            worst = max(worst, difference)
    print(f'{len(rows())} rows; largest |{DIGITS} - {CHECK_DIGITS} digits| / row scale:',
          mp.nstr(worst, 3))


if __name__ == '__main__':
    main()
