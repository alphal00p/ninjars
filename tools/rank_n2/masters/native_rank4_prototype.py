"""Prototype of the proposed native rank-4 bubble (OneLOop root representation, bub1111
analogue with the corrected mass-swap sign), checked against Feynman-parameter integrals.
Masses carry a tiny negative imaginary part so that principal logs implement the -i0."""
import mpmath as mp
mp.mp.dps = 40
TINY = mp.mpf('1e-30')

def bnlog(n, x):
    # (n+1) int_0^1 t^n log(1 - t/x) dt = (1 - x^(n+1)) log(1-1/x) - sum_k x^k/(n+1-k)
    if x == 1:
        return -sum(mp.mpf(1) / k for k in range(1, n + 2))
    return (1 - x ** (n + 1)) * mp.log(1 - 1 / x) - sum(x ** k / (n + 1 - k) for k in range(n + 1))

def native(p, m0, m1, rank=4, mu2=1):
    switch = abs(m1) < abs(m0)
    a0, a1 = (m1, m0) if switch else (m0, m1)
    fin = [None] * (rank + 1)
    if p == 0:
        lna = -mp.log(a1 / mu2)
        if a0 == a1:
            fin = [lna / (n + 1) for n in range(rank + 1)]
        else:
            x1 = a1 / (a1 - a0)
            fin = [(lna - bnlog(n, x1)) / (n + 1) for n in range(rank + 1)]
        reconstruct = not switch
    elif a0 == 0:
        lna = -mp.log((a1 - p) / mu2)
        x1 = (p - a1) / p
        fin = [((n + 1) * (lna - bnlog(n, x1)) + 1) / (n + 1) ** 2 for n in range(rank + 1)]
        reconstruct = switch
    else:
        lna = -mp.log(a0 / mu2)
        b = a1 - a0 - p
        d = mp.sqrt(b * b - 4 * p * a0)
        x1, x2 = (-b + d) / (2 * p), (-b - d) / (2 * p)
        fin = [(lna - bnlog(n, x1) - bnlog(n, x2)) / (n + 1) for n in range(rank + 1)]
        reconstruct = switch
    fin = [(-1) ** n * f for n, f in enumerate(fin)]
    if reconstruct:
        fin = [(-1) ** n * sum(mp.binomial(n, k) * fin[k] for k in range(n + 1)) for n in range(rank + 1)]
    return fin

def feynman(n, p, m0, m1):
    f = lambda x: x ** n * mp.log(-x * (1 - x) * p + (1 - x) * m0 + x * m1)
    pts = [mp.mpf(0), mp.mpf(1)]
    if p != 0:
        b = m1 - m0 - p
        d = mp.sqrt(b * b - 4 * p * m0)
        for r in ((-b + d) / (2 * p), (-b - d) / (2 * p)):
            if 0 < mp.re(r) < 1 and abs(mp.im(r)) < 1e-10:
                pts.append(mp.re(r))
    pts.sort()
    return (-1) ** (n + 1) * mp.quad(f, pts)

pts = [(3, 0.75, 1.5), (11, 1.125, 0.625), (-7.25, 0.375, 2.875), (0, 2, 5), (0, 5, 2),
       (2.5, 0, 1.5), (2.5, 1.5, 0), (3, 1.5, 0.75), (0.0625, 1.5, 1.0), (10000, 2, 1)]
worst = 0
for p, m0, m1 in pts:
    m0c = mp.mpc(m0, -TINY) if m0 else mp.mpc(0)
    m1c = mp.mpc(m1, -TINY) if m1 else mp.mpc(0)
    nat = native(mp.mpf(p), m0c, m1c)
    errs = []
    for n in range(5):
        ref = feynman(n, mp.mpf(p), m0c, m1c)
        e = abs(nat[n] - ref) / abs(ref)
        errs.append(e)
        worst = max(worst, e)
    print(f"p2={p:8} m0={m0:6} m1={m1:6}  rel.err B0..B1111: " + " ".join(mp.nstr(e, 2) for e in errs)
          + f"   B1111={mp.nstr(nat[4], 15)}")
print("worst", mp.nstr(worst, 3))
# the OneLOop 3.6 sign (negated rank-4 swap sum) for comparison at a swapped zero-momentum point
nat = native(mp.mpf(0), mp.mpc(2, -TINY), mp.mpc(5, -TINY))
print("B1111(0;2,5) corrected", mp.nstr(nat[4], 16), " OneLOop-3.6-like (negated)", mp.nstr(-nat[4], 16))
