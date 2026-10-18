"""Part (b): exact, constructive check of the d-dimensional integrand decomposition
for numerator rank r = n+2 (and controls r <= n+1) with Ninja's residue bases.

For n = 1..6 propagators with random rational kinematics/masses and a random
rational numerator N(q, mu2) of weighted degree r (all monomials present):

 1. top-down exact reduction over Q:  for k = 5..1 and every k-cut S,
       Delta_S|cut = (N - sum_{S' > S} Delta_S' prod_{j not in S'} D_j)|cut / prod_{j not in S} D_j|cut
    must be an EXACT polynomial division (zero remainder) on the rationally
    parametrised cut, of degree <= R_k = r-n+k; its coefficients in the Ninja-type
    basis are obtained by solving a square, non-singular rational linear system.
 2. the remainder N - sum_S Delta_S prod D must vanish identically (n >= 3), or be
    exactly divisible by prod_all D with quotient of weighted degree <= r-2n
    (the scaleless "0-point" term, needed only when r >= 2n, i.e. n <= 2 here).
 3. global linear system (mod a large prime p, exact modular arithmetic):
    the matrix of all basis polynomials times their uncut denominators has full
    column rank (=> unique decomposition) and its span contains every polynomial
    of weighted degree <= r (=> solvable for every numerator).
 4. control: with the rank n+1 residue bases a rank n+2 numerator is NOT
    decomposable (the quotient on the cut exceeds the old degree bound).
"""
import itertools
import random
import sys
import time

import numpy as np
import sympy as sp
from sympy import QQ

from common import (QR, MU2, Kinematics, Cut, wdeg, random_numerator, weighted_monomials,
                    ydeg)


def zero_point_bound(kin, rank):
    """Maximal weighted degree of the scaleless 0-point term c(q,mu2)*prod_all D.

    Common tadpole basis (Ninja's default): max(r-2n, r-n-2); independent random
    tadpole bases per single cut: max(r-2n, r-n-1).  (Empirical rule established
    by this script; for r=n+2 and a common basis it is a single constant for every
    n >= 2.)"""
    n = kin.n
    if "tadpole" in kin.refs:
        return max(rank - 2 * n, rank - n - 2)
    return max(rank - 2 * n, rank - n - 1)


def solve_basis(cut, delta_y):
    """Express the restricted residue in the restricted basis (square system)."""
    if cut.nfree == 0:
        val = cut.restrict(cut.basis[0])
        return [delta_y / val]
    cols = [cut.restrict(b) for b in cut.basis]
    monos = sorted({e for c in cols for e in c.keys()} | set(delta_y.keys()))
    # square-ness check
    nb = len(cols)
    M = sp.zeros(len(monos), nb)
    rhs = sp.zeros(len(monos), 1)
    idx = {m: i for i, m in enumerate(monos)}
    for j, c in enumerate(cols):
        for e, v in c.items():
            M[idx[e], j] = sp.Rational(int(v.numerator), int(v.denominator))
    for e, v in delta_y.items():
        rhs[idx[e], 0] = sp.Rational(int(v.numerator), int(v.denominator))
    assert len(monos) == nb, f"basis not square: {len(monos)} monomials vs {nb} elements"
    det = M.det(method="bareiss")
    assert det != 0, "singular residue basis"
    sol = M.LUsolve(rhs)
    assert M * sol == rhs
    return [QQ(int(sp.fraction(x)[0]), int(sp.fraction(x)[1])) for x in sol]


def top_down(kin, N, rank, rng, rank_for_basis=None, verbose=True):
    n = kin.n
    rb = rank if rank_for_basis is None else rank_for_basis
    residues = {}
    info = {}
    for k in range(min(5, n), 0, -1):
        for S in itertools.combinations(range(n), k):
            cut = Cut(kin, S, rb, rng)
            cut.make_ring()
            Nres = cut.restrict(N)
            for S2, d2 in residues.items():
                if set(S) < set(S2):
                    term = cut.restrict(d2)
                    for j in range(n):
                        if j not in S2:
                            term = term * cut.restrict(kin.D[j])
                    Nres = Nres - term
            U = None
            for j in cut.uncut:
                f = cut.restrict(kin.D[j])
                U = f if U is None else U * f
            if cut.nfree == 0:
                delta_y = Nres if U is None else Nres / U
                deg = 0
            else:
                if U is None:
                    delta_y = Nres
                else:
                    qq, rr = Nres.div([U])
                    if rr != 0:
                        return None, f"cut {S}: non-zero remainder in division by uncut denominators"
                    delta_y = qq[0]
                deg = ydeg(delta_y)
            if deg > cut.R:
                return None, f"cut {S}: residue degree {deg} exceeds bound R_{k}={cut.R}"
            if cut.R < 0:
                if delta_y != 0:
                    return None, f"cut {S}: non-zero residue but none allowed"
                continue
            coeffs = solve_basis(cut, delta_y)
            rep = sum((c * b for c, b in zip(coeffs, cut.basis)), QR(0))
            residues[S] = rep
            info[S] = dict(k=k, ncoef=len(cut.basis), nonzero=sum(1 for c in coeffs if c != 0),
                           deg=deg, R=cut.R, cut=cut, coeffs=coeffs)
    # global reconstruction
    total = QR(0)
    for S, rep in residues.items():
        t = rep
        for j in range(n):
            if j not in S:
                t = t * kin.D[j]
        total += t
    rem = N - total
    allD = QR(1)
    for j in range(n):
        allD = allD * kin.D[j]
    if rem == 0:
        zero_point = QR(0)
    else:
        qq, rr = rem.div([allD])
        if rr != 0:
            return None, "global remainder not divisible by prod_all D"
        zero_point = qq[0]
        # common tadpole basis (Ninja's default): 0-point term only if r >= 2n, of
        # weighted degree <= r-2n.  Independent per-tadpole bases: the tadpole
        # representatives differ by multiples h_i D_i, wdeg(h_i) <= r-n-1, which
        # pile up into sum_i h_i prod_all D.
        bound = zero_point_bound(kin, rank)
        if wdeg(zero_point) > bound:
            return None, f"0-point term of weighted degree {wdeg(zero_point)} > bound {bound}"
    # final exact identity
    total2 = total + zero_point * allD
    assert total2 == N
    return dict(residues=residues, info=info, zero_point=zero_point), "ok"


# ------------------------------------------------------------------ global rank mod p
P = 2147483629  # prime < 2^31


def qq_mod(x):
    return (int(x.numerator) % P) * pow(int(x.denominator) % P, P - 2, P) % P


def rank_mod_p(A):
    """Rank of an int64 matrix modulo P (Gaussian elimination, vectorised)."""
    A = A.copy() % P
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        if r == rows:
            break
        piv = np.nonzero(A[r:, c])[0]
        if piv.size == 0:
            continue
        p = r + piv[0]
        if p != r:
            A[[r, p]] = A[[p, r]]
        inv = pow(int(A[r, c]), P - 2, P)
        A[r] = (A[r] * inv) % P
        nz = np.nonzero(A[:, c])[0]
        nz = nz[nz != r]
        if nz.size:
            f = A[nz, c][:, None]
            A[nz] = (A[nz] - (f * A[r][None, :]) % P) % P
        r += 1
    return r


def global_rank_check(kin, rank, rng):
    n = kin.n
    cols_polys = []
    ncols_per_level = {}
    for k in range(min(5, n), 0, -1):
        for S in itertools.combinations(range(n), k):
            cut = Cut(kin, S, rank, rng)
            U = QR(1)
            for j in cut.uncut:
                U = U * kin.D[j]
            for b in cut.basis:
                cols_polys.append(b * U)
            ncols_per_level[k] = ncols_per_level.get(k, 0) + len(cut.basis)
    allD = QR(1)
    for j in range(n):
        allD = allD * kin.D[j]
    nzero = 0
    zb = zero_point_bound(kin, rank)
    if zb >= 0:
        for e in weighted_monomials(zb):
            cols_polys.append(QR({e: QQ(1)}) * allD)
            nzero += 1
    target = [QR({e: QQ(1)}) for e in weighted_monomials(rank)]
    monos = sorted({e for p in cols_polys + target for e in p.keys()})
    idx = {m: i for i, m in enumerate(monos)}

    def mat(polys):
        A = np.zeros((len(monos), len(polys)), dtype=np.int64)
        for j, p in enumerate(polys):
            for e, v in p.items():
                A[idx[e], j] = qq_mod(v)
        return A

    A = mat(cols_polys)
    B = mat(target)
    rA = rank_mod_p(A)
    rAB = rank_mod_p(np.concatenate([A, B], axis=1))
    return dict(ncols=A.shape[1], rows=A.shape[0], rankM=rA, rank_with_Pr=rAB,
                dimPr=len(target), per_level=ncols_per_level, zero_point_cols=nzero)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20261018
    nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    do_global = (sys.argv[3] != "noglobal") if len(sys.argv) > 3 else True
    rng = random.Random(seed)
    print(f"# exact decomposition check, seed={seed}")
    all_ok = True
    for n in range(1, nmax + 1):
        kin = Kinematics.random(n, rng, massless_fraction=0.2)
        if n <= 4:
            # variant with an independent random tadpole basis per single cut
            kin_ind = Kinematics(kin.V, kin.m2, refs={})
            for rank in (n + 1, n + 2):
                N = random_numerator(rng, rank)
                res, msg = top_down(kin_ind, N, rank, random.Random(seed + 11 * n + rank))
                zp = res["zero_point"] if res else None
                print(f"n={n} r={rank} [independent tadpole bases]: "
                      + (f"OK; 0-point term: {'none' if zp == 0 else f'wdeg {wdeg(zp)} ({len(zp)} terms)'}"
                         if res else f"FAILED: {msg}"))
                all_ok &= res is not None
        for rank in sorted({max(0, n - 1), n, n + 1, n + 2}):
            N = random_numerator(rng, rank)
            t0 = time.time()
            res, msg = top_down(kin, N, rank, random.Random(seed + 7 * n + rank))
            dt = time.time() - t0
            if res is None:
                print(f"n={n} r={rank}: FAILED: {msg}")
                all_ok = False
                continue
            counts = {}
            for S, inf in res["info"].items():
                counts.setdefault(inf["k"], set()).add(inf["ncoef"])
            zp = res["zero_point"]
            print(f"n={n} r={rank} (r-n={rank-n:+d}): exact top-down reduction OK "
                  f"[{dt:.1f}s]; coefficients per cut k: "
                  + ", ".join(f"k={k}:{sorted(v)}" for k, v in sorted(counts.items(), reverse=True))
                  + f"; 0-point term: {'none' if zp == 0 else f'wdeg {wdeg(zp)} ({len(zp)} terms)'};"
                  f" N == sum Delta*prod(D) exactly")
            if rank == n + 2:
                # control: rank n+1 bases are insufficient
                res2, msg2 = top_down(kin, N, rank, random.Random(seed + 7 * n + rank),
                                      rank_for_basis=n + 1)
                print(f"      control with rank-(n+1) residue bases: "
                      f"{'unexpectedly OK' if res2 else 'fails as expected: ' + msg2}")
                if res2 is not None:
                    all_ok = False
        if do_global:
            rank = n + 2
            t0 = time.time()
            g = global_rank_check(kin, rank, random.Random(seed + 7 * n + rank))
            ok = g["rankM"] == g["ncols"] and g["rank_with_Pr"] == g["rankM"]
            all_ok &= ok
            print(f"      global system mod p (r=n+2): {g['rows']} rows x {g['ncols']} residue columns "
                  f"(per k: {dict(sorted(g['per_level'].items(), reverse=True))}, 0-point: {g['zero_point_cols']}); "
                  f"rank={g['rankM']} (full column rank: {g['rankM']==g['ncols']}); "
                  f"dim P_r={g['dimPr']}, rank[M|P_r]={g['rank_with_Pr']} (P_r in span: {g['rank_with_Pr']==g['rankM']}) "
                  f"[{time.time()-t0:.1f}s]")
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
