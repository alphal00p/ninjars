"""Replicates oneloopmaster src/native/tensor.rs BnlogSeries::new to size the rank-4 series."""
from fractions import Fraction
from math import comb, factorial, log, exp
SERIES_RADIUS = 0.5
def series(rank, bits):
    target = -(bits + 4.0) * log(2)
    exact, logs, radii = [], [], []
    m = rank + 2
    while True:
        s = sum((-1) ** j * comb(rank, j) * (j + 1) ** (m - 2) for j in range(rank + 1))
        c = Fraction((m - 1) * s, factorial(m))
        logs.append(log(abs(float(c))) if c != 0 else float('-inf'))
        exact.append(c)
        count = len(exact) - 1
        if count > 0:
            radius = exp((logs[0] - logs[count] + target) / count)
            radii.append(radius)
            if radius >= SERIES_RADIUS * 1.0625:
                break
        m += 1
    exact.pop()
    return exact, radii
def needed(radii, n, r):
    for i, b in enumerate(radii):
        if r <= b:
            return i + 1
    return n
for bits in (53, 106):
    for rank in range(6):
        ex, radii = series(rank, bits)
        print(f"bits={bits} rank={rank}: coefficients={len(ex)} needed(0.5)={needed(radii,len(ex),0.5)} "
              f"needed(0.2)={needed(radii,len(ex),0.2)} needed(0.01)={needed(radii,len(ex),0.01)}")
for rank in (3, 4):
    ex, radii = series(rank, 53)
    print(f"rank {rank} @53 bits: radius covered by 8 coeffs={radii[7]:.4g}, by 16={radii[15]:.4g}, by 24={radii[min(23,len(radii)-1)]:.4g}")
