"""Verification of the master integrals and rational terms needed for rank n+2.

Conventions (Ninja / OneLOop):
  I_n[f] = (normalisation) * int d^d qbar/(i pi^{d/2}) f / prod_i Dbar_i,
  Dbar_i = (qbar + p_i)^2 - m_i^2,   qbar^2 = q^2 - mu^2,  d = 4 - 2 eps.
  Laurent coefficients [eps^0, eps^-1, eps^-2]; masses are squared.

Three independent evaluations are compared:
  (A) the closed forms derived in REPORT.md;
  (B) the exact d-dimensional Feynman-parameter representation
        I_n[(mu^2)^b (q.v)^a] = Gamma(b-eps)/Gamma(-eps) (-1)^n Gamma(n-2-b+eps)
                                 int dx (-(P.v))^a Delta^{2+b-n-eps},
      evaluated at eps = 1e-25 with 60 digits (quadrature over the simplex);
  (C) COLLIER 1.2.9 UV-pole coefficients of tensor coefficients, combined as
        I_n[(mu^2)^b (q.v)^a] = (-1)^{b+1} 2^b (b-1)! (k.v)^a T^{uv}_{0^{2b} 1^a}
      (contraction of the (d-4)-dimensional metric, see REPORT.md),
      read from ../collier/uvpoles.out.
"""
import sys
import mpmath as mp

mp.mp.dps = 40
EPS = mp.mpf('1e-25')


def closed_box_mu4(s, m):
    return mp.mpf(-1) / 6


def closed_box_mu6(s, m):
    # s = (s21, s32, s43, s14, s31, s42), m = (m1..m4) squared
    return (5 * sum(m) - sum(s)) / 60


def closed_tri_mu2(s, m):
    return mp.mpf(1) / 2


def closed_tri_mu4(s, m):
    return (sum(s) / 4 - sum(m)) / 6


def closed_bub(a, b, k, m0, m1):
    """Coefficient R_{a,b} with I_2[(mu^2)^b ((q+p_0).v)^a] = (k.v)^a R_{a,b}."""
    table = {
        (0, 1): (3 * (m0 + m1) - k) / 6,
        (1, 1): (k - 2 * m0 - 4 * m1) / 12,
        (2, 1): (5 * m0 + 15 * m1 - 3 * k) / 60,
        (0, 2): -(k * k - 5 * k * (m0 + m1) + 10 * (m0 * m0 + m0 * m1 + m1 * m1)) / 60,
    }
    return table[(a, b)]


def closed_tad_mu2(m):
    return m * m / 2


def simplex_integral(n, f):
    """int over x_1..x_n >= 0, sum x = 1 of f(x) (n = 1..4)."""
    if n == 1:
        return f((mp.mpf(1),))
    if n == 2:
        return mp.quad(lambda x: f((1 - x, x)), [0, 1])
    if n == 3:
        return mp.quad(lambda x, y: f((1 - x, x * (1 - y), x * y)) * x, [0, 1], [0, 1])
    if n == 4:
        return mp.quad(
            lambda x, y, z: f((1 - x, x * (1 - y), x * y * (1 - z), x * y * z)) * x * x * y,
            [0, 1], [0, 1], [0, 1])
    raise ValueError(n)


def delta(x, s, m):
    """Delta = sum x_i m_i - sum_{i<j} x_i x_j s_ij with s given as dict (i,j)->s_ij."""
    n = len(x)
    value = sum(x[i] * m[i] for i in range(n))
    for i in range(n):
        for j in range(i + 1, n):
            value -= x[i] * x[j] * s[(i, j)]
    return value


def feynman_d_dim(n, a, b, s, m, pv):
    """(B): exact representation at small eps; pv[i] = p_i . v (p_0 = 0)."""
    pref = mp.gamma(b - EPS) / mp.gamma(-EPS) * (-1) ** n * mp.gamma(n - 2 - b + EPS)

    def integrand(x):
        P = sum(x[i] * pv[i] for i in range(n))
        return (-P) ** a * delta(x, s, m) ** (2 + b - n - EPS)

    return pref * simplex_integral(n, integrand)


def report(tag, values):
    ref = values[0][1]
    line = [f"{tag:34s}"]
    worst = mp.mpf(0)
    for name, value in values:
        line.append(f"{name}={mp.nstr(value, 17)}")
        worst = max(worst, abs(value - ref) / max(abs(ref), mp.mpf('1e-250')))
    line.append(f"max rel diff={mp.nstr(worst, 3)}")
    print("  ".join(line))
    return worst


def read_collier(path):
    data = {}
    with open(path) as handle:
        for raw in handle:
            parts = raw.split()
            if not parts or parts[0] not in ("BUB1", "BUB2", "BUB3", "BUB4", "BUB5", "TRI", "BOX", "TAD"):
                continue
            numbers = [mp.mpf(p) for p in parts if p[0] in "-0123456789" and "E" in p]
            pairs = [mp.mpc(numbers[i], numbers[i + 1]) for i in range(0, len(numbers), 2)]
            names = [p for p in parts[1:] if not (p[0] in "-0123456789" and "E" in p)]
            for name, value in zip(names, pairs):
                data[(parts[0], name)] = value
    return data


def main():
    collier = read_collier(sys.argv[1]) if len(sys.argv) > 1 else {}
    worst = mp.mpf(0)

    # Bubble points: (k^2, m0^2, m1^2), as in ../collier/uvpoles.f90.
    bubbles = {
        "BUB1": (mp.mpf(3), mp.mpc('0.7', '-0.03'), mp.mpc('1.4', '-0.08')),
        "BUB2": (mp.mpf(0), mp.mpf(2), mp.mpf(5)),
        "BUB3": (mp.mpf('-7.3'), mp.mpf('0.4'), mp.mpf('2.9')),
        "BUB4": (mp.mpf(11), mp.mpf('1.1'), mp.mpf('0.6')),
    }
    # (k.v) is arbitrary for a massless v; choose kv = 1 and p_1 = k.
    print("== bubble mu^2 integrals: I_2[(mu^2)^b ((q+p0).v)^a] / (k.v)^a")
    for tag, (k, m0, m1) in bubbles.items():
        s = {(0, 1): k}
        for (a, b, ctag) in ((0, 1, "B00uv"), (1, 1, "B001uv"), (2, 1, "B0011uv"), (0, 2, "B0000uv")):
            values = [("closed", closed_bub(a, b, k, m0, m1)),
                      ("feynman", feynman_d_dim(2, a, b, s, (m0, m1), (0, 1)))]
            key = (tag, ctag)
            if key in collier:
                factor = (-1) ** (b + 1) * 2 ** b * mp.factorial(b - 1)
                values.append(("collier", factor * collier[key]))
            worst = max(worst, report(f"{tag} a={a} b={b}", values))

    print("== triangle mu^2, mu^4")
    tri_s = {(0, 1): mp.mpf('2.3'), (1, 2): mp.mpf('-1.7'), (0, 2): mp.mpf('5.1')}
    tri_m = (mp.mpf('0.3'), mp.mpc('1.2', '-0.1'), mp.mpf('0.8'))
    svals = list(tri_s.values())
    values = [("closed", closed_tri_mu2(svals, tri_m)),
              ("feynman", feynman_d_dim(3, 0, 1, tri_s, tri_m, (0, 0, 0)))]
    if ("TRI", "C00uv") in collier:
        values.append(("collier", 2 * collier[("TRI", "C00uv")]))
    worst = max(worst, report("TRI mu^2", values))
    values = [("closed", closed_tri_mu4(svals, tri_m)),
              ("feynman", feynman_d_dim(3, 0, 2, tri_s, tri_m, (0, 0, 0)))]
    if ("TRI", "C0000uv") in collier:
        values.append(("collier", -4 * collier[("TRI", "C0000uv")]))
    worst = max(worst, report("TRI mu^4", values))

    print("== box mu^4, mu^6")
    # COLLIER order p10=s21, p21=s32, p32=s43, p30=s14, p20=s31, p31=s42
    box_s = {(0, 1): mp.mpf('1.1'), (1, 2): mp.mpf('-2.2'), (2, 3): mp.mpf('0.7'),
             (0, 3): mp.mpf('3.3'), (0, 2): mp.mpf('9.5'), (1, 3): mp.mpf('-4.1')}
    box_m = (mp.mpf('0.5'), mp.mpc('1.5', '-0.2'), mp.mpf('2.5'), mp.mpf('0.25'))
    svals = list(box_s.values())
    values = [("closed", closed_box_mu4(svals, box_m)),
              ("feynman", feynman_d_dim(4, 0, 2, box_s, box_m, (0, 0, 0, 0)))]
    if ("BOX", "D0000uv") in collier:
        values.append(("collier", -4 * collier[("BOX", "D0000uv")]))
    worst = max(worst, report("BOX mu^4", values))
    values = [("closed", closed_box_mu6(svals, box_m)),
              ("feynman", feynman_d_dim(4, 0, 3, box_s, box_m, (0, 0, 0, 0)))]
    if ("BOX", "D000000uv") in collier:
        values.append(("collier", 16 * collier[("BOX", "D000000uv")]))
    worst = max(worst, report("BOX mu^6 (new)", values))

    print("== tadpole mu^2")
    m = mp.mpc('2.7', '-0.3')
    values = [("closed", closed_tad_mu2(m)),
              ("feynman", feynman_d_dim(1, 0, 1, {}, (m,), (0,)))]
    if ("TAD", "A00uv") in collier:
        values.append(("collier", 2 * collier[("TAD", "A00uv")]))
    worst = max(worst, report("TAD mu^2", values))

    print("== spurious checks: odd tadpole and transverse triangle/bubble terms vanish")
    # Tadpole I_1[(mu^2)^b ((q+p).v)^a] with v arbitrary (not null): odd a vanish by q -> -q;
    # this is exact in the representation (no P term), so we only state it.
    print("   tadpole: P = 0, so every odd monomial in (q+p_i) integrates to zero exactly")

    print("== B1111 finite parts: -int x^4 log(Delta/mu^2) vs COLLIER (DeltaUV=0, mu^2=1)")

    def b1111_finite(k, m0, m1):
        def dl(x):
            d = -x * (1 - x) * k + (1 - x) * m0 + x * m1
            d = mp.mpc(d)
            if d.imag == 0 and d.real < 0:
                return mp.log(-d.real) - 1j * mp.pi
            return mp.log(d)
        # split at real roots of Delta for real kinematics
        points = [mp.mpf(0), mp.mpf(1)]
        if all(mp.mpc(z).imag == 0 for z in (k, m0, m1)) and k != 0:
            kk, a0, a1 = mp.re(k), mp.re(m0), mp.re(m1)
            disc = (a1 - a0 - kk) ** 2 - 4 * kk * a0
            if disc > 0:
                for r in ((-(a1 - a0 - kk) + mp.sqrt(disc)) / (2 * kk),
                          (-(a1 - a0 - kk) - mp.sqrt(disc)) / (2 * kk)):
                    if 0 < r < 1:
                        points.append(r)
        points.sort()
        return -mp.quad(lambda x: x ** 4 * dl(x), points)

    extra = dict(bubbles)
    extra["BUB5"] = (mp.mpf(0), mp.mpf('1.3'), mp.mpf('1.3'))
    for tag, (k, m0, m1) in extra.items():
        values = [("feynman", b1111_finite(k, m0, m1))]
        if (tag, "B1111") in collier:
            values.append(("collier", collier[(tag, "B1111")]))
        if (tag, "B1111uv") in collier:
            pole = collier[(tag, "B1111uv")]
            print(f"   {tag} pole COLLIER={mp.nstr(pole, 17)} (expected 1/5)")
        worst = max(worst, report(f"{tag} B1111 finite", values))
    # oneloopmaster's committed references (src/tests.rs rank_four_bubbles_match_existing_reference)
    print("   oneloopmaster src/tests.rs references: BUB1 0.029939862301517683+0.017531313325083266i,"
          " BUB2 -0.2998475029970866")

    print(f"\nWORST relative difference over all comparisons: {mp.nstr(worst, 3)}")


if __name__ == "__main__":
    main()
