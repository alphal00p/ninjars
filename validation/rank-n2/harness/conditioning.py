#!/usr/bin/env python3
"""Kinematic conditioning diagnostics of a lane: for every subset of k+1 propagators
(k = 2..min(4, n-1)) the normalised Gram determinant
    g = |det(G)| / prod_i |G_ii|-ish  with G_ij = 2 (V_i - V_0).(V_j - V_0)
normalised by the product of Euclidean |v_i|^2 (scale-free), and the normalised modified
Cayley determinant det(Y)/prod(|Y_ii| + max|Y_ij|), Y_ij = m_i^2 + m_j^2 - (V_i - V_j)^2."""
import itertools, json, sys
from fractions import Fraction
def mdot(a, b): return a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3]
def det(M):
    M = [row[:] for row in M]; n = len(M); d = 1
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if M[p][c] == 0: return 0
        if p != c: M[c], M[p] = M[p], M[c]; d = -d
        d *= M[c][c]
        for r in range(c+1, n):
            f = M[r][c] / M[c][c]
            for k in range(c, n): M[r][k] -= f * M[c][k]
    return d
def gram_min(shifts, kmax=4):
    n = len(shifts); worst = (1.0, None)
    for k in range(2, min(kmax, n - 1) + 1):
        for sub in itertools.combinations(range(n), k + 1):
            v = [[Fraction(x) - Fraction(y) for x, y in zip(shifts[i], shifts[sub[0]])] for i in sub[1:]]
            G = [[mdot(a, b) for b in v] for a in v]
            norm = 1
            for a in v: norm *= sum(x * x for x in a)
            g = abs(float(det(G) / norm)) if norm else 0.0
            if g < worst[0]: worst = (g, sub)
    return worst
def cayley_min(shifts, m2):
    n = len(shifts); worst = (1.0, None)
    for k in range(3, n + 1):
        for sub in itertools.combinations(range(n), k):
            Y = [[complex(*m2[i]) + complex(*m2[j]) - mdot([a-b for a, b in zip(shifts[i], shifts[j])], [a-b for a, b in zip(shifts[i], shifts[j])]) for j in sub] for i in sub]
            sc = max(abs(x) for r in Y for x in r) ** k
            y = abs(det(Y)) / sc if sc else 0.0
            if y < worst[0]: worst = (y, sub)
    return worst
if __name__ == "__main__":
    for f in sys.argv[1:]:
        c = json.load(open(f))
        for l in c["lanes"]:
            g = gram_min(l["shifts"]); y = cayley_min(l["shifts"], l["masses_squared"])
            print(f"{l['id']:36s} min normalised Gram {g[0]:.2e} {g[1]}  min normalised Cayley {y[0]:.2e} {y[1]}")
