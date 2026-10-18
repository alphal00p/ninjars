"""Part (c): symbolic derivation of Ninja's coefficient-level subtraction terms
(the analogue of the FORM output in coefficient_level_subs.cc) for rank r = n+2.

A higher-point residue monomial M = mu2^j * prod_i L_i, written on the lower cut's
Laurent parametrisation (L_i = alpha_i t + beta_i + gamma_i / t), divided by the
uncut denominators D_l = d0 t + d1 + d2/t, has a large-t expansion whose
polynomial part P[M / prod D] is what must be subtracted from the lower cut's
Laurent coefficients.  With z = 1/t:
     M / prod_{l=1..m} D_l = mu2^j z^(m-a) * prod_i(alpha_i + beta_i z + gamma_i z^2)
                                           / prod_l(d0_l + d1_l z + d2_l z^2),
so the coefficient of t^b (b >= 0) is mu2^j [z^(a-m-b)] of the Taylor series.

Three kinds (as in Ninja):
  tri->bub : triangle monomial on the bubble parametrisation
             qbar = eb1 + x eb2 + t eb3 + X/t eb4,  X = P0 + P1 x + P2 x^2 + mu2,
             L_u = t U3 + (U1 + x U2) + X U4/t   with Ui = eb_i.u, u in {et3, et4},
             D   = 2 q.k + f  = t 2K3 + (2K1 + 2 x K2 + f) + X 2K4 / t,  Ki = eb_i.k
  tri->tad : triangle monomial on the tadpole parametrisation qbar = t e3 + (P+mu2)/t e4,
             L_u = t U3 + (P+mu2) U4/t,  U3 = e3.u, U4 = e4.u,
             D_i = t 2A_i + f_i + (P+mu2) 2B_i/t,  A_i = e3.k_i, B_i = e4.k_i  (i = 0,1)
  bub->tad : bubble monomial qb2^a qb3^b qb4^c mu2^j on the tadpole parametrisation,
             qb2 = t S2 + W + (P+mu2) T2/t,  qb3 = t S3 + (P+mu2) T3/t,  qb4 = t S4 + (P+mu2) T4/t
             (S_i = e3.eb_i, T_i = e4.eb_i, W = eb2.kbol),  D = t 2A + f + (P+mu2) 2B/t.

Outputs: formulas.json section "subtractions" (sympy-printable strings) for every
NEW rank n+2 monomial and every Laurent slot it feeds, plus a numerical
validation of the same engine against original Ninja's FORM-generated x1cuts
routines (old monomials, rank n+1 layouts) through ninja_harness/harness.
"""
import json
import os
import random
import subprocess
import sys

import sympy as sp

t, x, mu2, z = sp.symbols("t x mu2 z")
HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ engine

def series_inv(d0, d1, d2, order):
    c = [1 / d0]
    for s in range(1, order + 1):
        v = -d1 * c[s - 1]
        if s >= 2:
            v -= d2 * c[s - 2]
        c.append(sp.expand(v / d0))
    return c


def smul(a, b, order):
    out = [0] * (order + 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            if i + j > order:
                break
            out[i + j] += ai * bj
    return [sp.expand(o) for o in out]


def polypart(forms, mu2pow, dens):
    """forms: list of (alpha, beta, gamma); dens: list of (d0, d1, d2).
    Returns {(b, ax, amu): expr} for the t^b x^ax mu2^amu coefficients, b >= 0."""
    a, m = len(forms), len(dens)
    order = a - m
    if order < 0:
        return {}
    G = [sp.Integer(1)] + [0] * order
    for al, be, ga in forms:
        G = smul(G, [al, be, ga], order)
    for d0, d1, d2 in dens:
        G = smul(G, series_inv(d0, d1, d2, order), order)
    out = {}
    for b in range(0, order + 1):
        s = order - b
        expr = sp.expand(G[s] * mu2 ** mu2pow)
        if expr == 0:
            continue
        poly = sp.Poly(expr, x, mu2)
        for (ax, amu), coeff in poly.terms():
            out[(b, ax, amu)] = sp.factor(coeff)
    return out


# ------------------------------------------------------------------ symbols
U1, U2, U3, U4, K1, K2, K3, K4, f, P0, P1, P2 = sp.symbols("U1 U2 U3 U4 K1 K2 K3 K4 f P0 P1 P2")
A0, B0, A1, B1, f0, f1, P = sp.symbols("A0 B0 A1 B1 f0 f1 P")
S2, S3, S4, T2, T3, T4, W, A, B = sp.symbols("S2 S3 S4 T2 T3 T4 W A B")

X = P0 + P1 * x + P2 * x ** 2 + mu2


def tribub(a, j):
    L = (U3, U1 + x * U2, X * U4)
    D = (2 * K3, 2 * K1 + 2 * x * K2 + f, 2 * X * K4)
    return polypart([L] * a, j, [D])


def tritad(a, j):
    L = (U3, 0, (P + mu2) * U4)
    D0_ = (2 * A0, f0, 2 * (P + mu2) * B0)
    D1_ = (2 * A1, f1, 2 * (P + mu2) * B1)
    return polypart([L] * a, j, [D0_, D1_])


def bubtad(e2, e3, e4, j):
    L2 = (S2, W, (P + mu2) * T2)
    L3 = (S3, 0, (P + mu2) * T3)
    L4 = (S4, 0, (P + mu2) * T4)
    D = (2 * A, f, 2 * (P + mu2) * B)
    return polypart([L2] * e2 + [L3] * e3 + [L4] * e4, j, [D])


# Ninja residue monomial tables (index, mu2 power, powers) -- see REPORT.md
TRI = {0: (0, 0, 0), 1: (0, 1, 0), 2: (0, 2, 0), 3: (0, 3, 0), 4: (0, 0, 1), 5: (0, 0, 2), 6: (0, 0, 3),
       7: (1, 0, 0), 8: (1, 1, 0), 9: (1, 0, 1), 10: (1, 2, 0), 11: (1, 0, 2), 12: (0, 4, 0), 13: (0, 0, 4),
       14: (2, 0, 0), 15: (0, 5, 0), 16: (0, 0, 5), 17: (1, 3, 0), 18: (1, 0, 3), 19: (2, 1, 0), 20: (2, 0, 1)}
# (j, e3 power, e4 power)
BUB = {0: (0, 0, 0, 0), 1: (0, 1, 0, 0), 2: (0, 2, 0, 0), 3: (0, 0, 1, 0), 4: (0, 0, 2, 0), 5: (0, 0, 0, 1),
       6: (0, 0, 0, 2), 7: (0, 1, 1, 0), 8: (0, 1, 0, 1), 9: (1, 0, 0, 0), 10: (1, 1, 0, 0), 11: (1, 0, 1, 0),
       12: (1, 0, 0, 1), 13: (0, 3, 0, 0), 14: (0, 0, 3, 0), 15: (0, 0, 0, 3), 16: (0, 2, 1, 0),
       17: (0, 2, 0, 1), 18: (0, 1, 2, 0), 19: (0, 1, 0, 2),
       20: (0, 4, 0, 0), 21: (1, 2, 0, 0), 22: (2, 0, 0, 0), 23: (0, 0, 4, 0), 24: (0, 0, 0, 4),
       25: (0, 1, 3, 0), 26: (0, 1, 0, 3), 27: (0, 2, 2, 0), 28: (0, 2, 0, 2), 29: (0, 3, 1, 0),
       30: (0, 3, 0, 1), 31: (1, 0, 2, 0), 32: (1, 0, 0, 2), 33: (1, 1, 1, 0), 34: (1, 1, 0, 1)}
# (j, e2, e3, e4)
NEW_TRI = range(15, 21)
NEW_BUB = range(20, 35)

LAYOUT = {
    "bubble_rank_n+1": [(3, 0, 0), (2, 0, 0), (2, 1, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 2, 0),
                        (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 3, 0)],
    "bubble_rank_n+2": [(4, 0, 0), (3, 0, 0), (3, 1, 0), (2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 2, 0),
                        (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 3, 0),
                        (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1),
                        (0, 3, 0), (0, 4, 0)],
    "triangle_rank_n+2": [(5, 0, 0), (4, 0, 0), (3, 0, 0), (3, 0, 1), (2, 0, 0), (2, 0, 1), (1, 0, 0),
                          (1, 0, 1), (1, 0, 2), (0, 0, 0), (0, 0, 1), (0, 0, 2)],
    "tadpole_rank_n+1": [(2, 0, 0), (1, 0, 0), (0, 0, 0), (0, 0, 1)],
    "tadpole_rank_n+2": [(3, 0, 0), (2, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 0), (0, 0, 1)],
}


def tri_formula(idx, kind):
    j, a3, a4 = TRI[idx]
    a = a3 + a4
    if a == 0:
        return {}, None
    which = "et3" if a3 else "et4"
    return (tribub(a, j) if kind == "bub" else tritad(a, j)), which


def bub_formula(idx):
    j, e2, e3, e4 = BUB[idx]
    return bubtad(e2, e3, e4, j)


# ------------------------------------------------------------------ numerics

def cdot(a, b):
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def rnd(rng):
    return complex(rng.uniform(-1, 1), rng.uniform(-1, 1))


def rvec(rng, real=False):
    return [rng.uniform(-1, 1) if real else rnd(rng) for _ in range(4)]


def numeric_tribub(rng, coeff_idx, all_slots):
    et3, et4 = rvec(rng), rvec(rng)
    c = [rnd(rng) for _ in range(21)]
    eb = [rvec(rng) for _ in range(4)]
    prm = [rnd(rng) for _ in range(3)]
    k = rvec(rng, real=True)
    fv = rnd(rng)
    total = {}
    for idx in coeff_idx:
        F, which = tri_formula(idx, "bub")
        u = et3 if which == "et3" else et4
        subs = {U1: cdot(eb[0], u), U2: cdot(eb[1], u), U3: cdot(eb[2], u), U4: cdot(eb[3], u),
                K1: cdot(eb[0], k), K2: cdot(eb[1], k), K3: cdot(eb[2], k), K4: cdot(eb[3], k),
                f: fv, P0: prm[0], P1: prm[1], P2: prm[2]}
        for slot, expr in F.items():
            total[slot] = total.get(slot, 0) + c[idx] * complex(expr.evalf(subs=subs))
    line = (f"tribub {1 if all_slots else 0} " + " ".join(f"{v.real} {v.imag}" for v in et3 + et4) + " "
            + " ".join(f"{v.real} {v.imag}" for v in c[:15]) + " "
            + " ".join(f"{v.real} {v.imag}" for vec in eb for v in vec) + " "
            + " ".join(f"{v.real} {v.imag}" for v in prm) + " " + " ".join(str(v) for v in k)
            + f" {fv.real} {fv.imag}")
    return total, line


def numeric_tritad(rng, coeff_idx, full):
    et3, et4 = rvec(rng), rvec(rng)
    c = [rnd(rng) for _ in range(21)]
    e3, e4 = rvec(rng), rvec(rng)
    prm = rnd(rng)
    k0, k1 = rvec(rng, True), rvec(rng, True)
    fa, fb = rnd(rng), rnd(rng)
    total = {}
    for idx in coeff_idx:
        F, which = tri_formula(idx, "tad")
        u = et3 if which == "et3" else et4
        subs = {U3: cdot(e3, u), U4: cdot(e4, u), A0: cdot(e3, k0), B0: cdot(e4, k0),
                A1: cdot(e3, k1), B1: cdot(e4, k1), f0: fa, f1: fb, P: prm}
        for slot, expr in F.items():
            total[slot] = total.get(slot, 0) + c[idx] * complex(expr.evalf(subs=subs))
    line = (("tritadfull " if full else "tritad ") + " ".join(f"{v.real} {v.imag}" for v in et3 + et4) + " "
            + " ".join(f"{v.real} {v.imag}" for v in c[:15]) + " "
            + " ".join(f"{v.real} {v.imag}" for v in e3 + e4) + f" {prm.real} {prm.imag} "
            + " ".join(str(v) for v in k0) + f" {fa.real} {fa.imag} " + " ".join(str(v) for v in k1)
            + f" {fb.real} {fb.imag}")
    return total, line


def numeric_bubtad(rng, coeff_idx, full):
    eb2, eb3, eb4 = rvec(rng), rvec(rng), rvec(rng)
    kbol = rvec(rng, True)
    c = [rnd(rng) for _ in range(35)]
    e3, e4 = rvec(rng), rvec(rng)
    prm = rnd(rng)
    k = rvec(rng, True)
    fv = rnd(rng)
    subs = {S2: cdot(e3, eb2), S3: cdot(e3, eb3), S4: cdot(e3, eb4), T2: cdot(e4, eb2), T3: cdot(e4, eb3),
            T4: cdot(e4, eb4), W: cdot(eb2, kbol), A: cdot(e3, k), B: cdot(e4, k), f: fv, P: prm}
    total = {}
    for idx in coeff_idx:
        for slot, expr in bub_formula(idx).items():
            total[slot] = total.get(slot, 0) + c[idx] * complex(expr.evalf(subs=subs))
    line = (("bubtadfull " if full else "bubtad ") + " ".join(f"{v.real} {v.imag}" for v in eb2 + eb3 + eb4) + " "
            + " ".join(str(v) for v in kbol) + " " + " ".join(f"{v.real} {v.imag}" for v in c[:20]) + " "
            + " ".join(f"{v.real} {v.imag}" for v in e3 + e4) + f" {prm.real} {prm.imag} "
            + " ".join(str(v) for v in k) + f" {fv.real} {fv.imag}")
    return total, line


def run_harness(lines):
    exe = os.path.join(HERE, "ninja_harness", "harness")
    out = subprocess.run([exe], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
    rows = []
    for l in out.stdout.strip().split("\n"):
        v = [float(s) for s in l.split()]
        rows.append([complex(v[2 * i], v[2 * i + 1]) for i in range(len(v) // 2)])
    return rows


def validate_against_ninja(ntests=6, seed=3):
    rng = random.Random(seed)
    worst = 0.0
    report = []
    cases = []
    for _ in range(ntests):
        for all_slots in (True, False):
            tot, line = numeric_tribub(rng, range(15), all_slots)
            cases.append(("tri->bub" + ("" if all_slots else " (swapped, t>=1)"), tot, line,
                          LAYOUT["bubble_rank_n+1"], (lambda s, a=all_slots: a or s[0] >= 1)))
        for full in (False, True):
            tot, line = numeric_tritad(rng, range(15), full)
            cases.append(("tri->tad" + (" full" if full else ""), tot, line, LAYOUT["tadpole_rank_n+1"],
                          (lambda s, fu=full: fu or s[0] == 0)))
            tot, line = numeric_bubtad(rng, range(20), full)
            cases.append(("bub->tad" + (" full" if full else ""), tot, line, LAYOUT["tadpole_rank_n+1"],
                          (lambda s, fu=full: fu or s[0] == 0)))
    rows = run_harness([c[2] for c in cases])
    for (label, tot, line, layout, active), row in zip(cases, rows):
        err, scale = 0.0, 0.0
        for i, slot in enumerate(layout):
            mine = tot.get(slot, 0) if active(slot) else 0
            err = max(err, abs(mine - row[i]))
            scale = max(scale, abs(row[i]), abs(mine))
        rel = err / scale
        worst = max(worst, rel)
        report.append((label, rel))
    return worst, report


# ------------------------------------------------------------------ export

def to_str(e):
    return sp.sstr(e)


def build_json():
    out = {"conventions": __doc__,
           "layouts": {k: [list(s) for s in v] for k, v in LAYOUT.items()},
           "triangle_monomials": {str(k): {"mu2": v[0], "qe3": v[1], "qe4": v[2]} for k, v in TRI.items()},
           "bubble_monomials": {str(k): {"mu2": v[0], "qe2": v[1], "qe3": v[2], "qe4": v[3]} for k, v in BUB.items()},
           "tri_to_bub": {}, "tri_to_tad": {}, "bub_to_tad": {}}
    for idx in NEW_TRI:
        F, which = tri_formula(idx, "bub")
        out["tri_to_bub"][f"c{idx}"] = {"u": which, "slots": {str(list(s)): to_str(e) for s, e in sorted(F.items(), reverse=True)}}
        F, which = tri_formula(idx, "tad")
        out["tri_to_tad"][f"c{idx}"] = {"u": which, "slots": {str(list(s)): to_str(e) for s, e in sorted(F.items(), reverse=True)}}
    for idx in NEW_BUB:
        F = bub_formula(idx)
        out["bub_to_tad"][f"c{idx}"] = {"slots": {str(list(s)): to_str(e) for s, e in sorted(F.items(), reverse=True)}}
    return out


if __name__ == "__main__":
    worst, rep = validate_against_ninja()
    for label, rel in rep:
        print(f"engine vs original Ninja x1cuts ({label:28s}): rel.err {rel:.2e}")
    print(f"WORST engine-vs-Ninja relative error (old monomials, rank n+1 layouts): {worst:.2e}")
    data = build_json()
    # summary of the new terms
    for kind in ("tri_to_bub", "tri_to_tad", "bub_to_tad"):
        for name, d in data[kind].items():
            print(f"{kind:10s} {name:4s} -> slots {sorted(d['slots'].keys())}")
    with open(os.path.join(HERE, "subtractions_rank_n2.json"), "w") as fh:
        json.dump(data, fh, indent=1)
    print("wrote subtractions_rank_n2.json")
