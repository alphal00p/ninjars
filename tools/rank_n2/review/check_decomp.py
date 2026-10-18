#!/usr/bin/env python3
"""Referee checks (1) and (2): exact (F_p[i]) Ninja-style reduction at ranks n-1..n+2
with mu2-dependent numerators; per-cut fit consistency (completeness of the residue
basis), Laurent-recipe vs fit agreement, and the global identity N = sum Delta prod D
+ Delta_0 prod D (degree of Delta_0)."""
import sys
import time
import red
from red import FieldFp, set_field, Numerator, Reduction, zero_point_degree


def run_case(n, r, seed, tadbasis='common', with_mu=True, check_fits=True):
    f = FieldFp(seed)
    set_field(f)
    V = [tuple(f.rand() for _ in range(4)) for _ in range(n)]
    m = [f.rand() for _ in range(n)]
    N = Numerator.random(r, with_mu=with_mu)
    t0 = time.time()
    red_ = Reduction(n, V, m, N, tadpole_basis=tadbasis)
    summ = red_.run(check_fits=check_fits, log=print)
    zp = red_.zero_point(npts=24)
    deg = zero_point_degree(zp)
    counts = {k: len(red.mono_set(k, red_.R(k))) for k in (5, 4, 3, 2, 1) if k <= n and red_.R(k) >= 0}
    worst_fit = max((w for v in summ.values() for _, w, _ in v), default=0)
    worst_rec = max((d for v in summ.values() for _, _, d in v), default=0)
    ncuts = {k: len(v) for k, v in summ.items()}
    print(f"n={n} r={r} (r-n={r-n:+d}) seed={seed} tadbasis={tadbasis} mu2-numerator={with_mu}: "
          f"coeffs/cut {counts} cuts {ncuts} fit-inconsistent={worst_fit>0} recipe!=fit={worst_rec>0} "
          f"Delta_0: {'degree<=%d' % deg[0] if deg else 'NOT POLY<=3'}{' (identically 0)' if deg and deg[1] else ''} "
          f"[{time.time()-t0:.1f}s]", flush=True)
    return red_, summ, deg


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'main'
    if which == 'main':
        for n in range(1, 7):
            for dr in (-1, 0, 1, 2):
                r = n + dr
                if r < 0:
                    continue
                if n == 6 and dr < 2:
                    continue
                run_case(n, r, 1000 + 10 * n + dr)
    elif which == 'tad':
        for n in range(1, 5):
            for dr in (1, 2):
                run_case(n, n + dr, 2000 + 10 * n + dr, tadbasis='independent')
    elif which == 'one':
        n, r = int(sys.argv[2]), int(sys.argv[3])
        run_case(n, r, int(sys.argv[4]) if len(sys.argv) > 4 else 7)
