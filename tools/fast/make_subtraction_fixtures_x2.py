#!/usr/bin/env python3
"""Golden fixtures for the rank n+2 fast subtractions (validation V3).

Random binary64 inputs in the argument layout of the generated routines
(crates/ninjars/src/fast/subtractions_x2.rs) mapped to the subtraction terms
of EVERY parent coefficient (old and new), computed from the series
definition of `tools/rank_n2/theory/subtraction_formulas.py` (the polynomial
part of the parent monomial divided by the uncut denominators on the child's
Laurent parametrisation), independently of both Ninja's FORM output and the
CSE code the routines are built from. Values are evaluated with mpmath at 40
digits and rounded to binary64.

Usage (sympy and mpmath are needed, e.g. through tools/rank_n2/theory/py.sh):
    make_subtraction_fixtures_x2.py <output .json>
"""

import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rank_n2" / "theory"))
import mpmath  # noqa: E402
import sympy as sp  # noqa: E402
import subtraction_formulas as sf  # noqa: E402

mpmath.mp.dps = 40
SEED = 20261018
SAMPLES = 6

BUBBLE_X2 = sf.LAYOUT["bubble_rank_n+2"]
TADPOLE_X2 = sf.LAYOUT["tadpole_rank_n+2"]


def cdot(a, b):
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def rnd(rng):
    return complex(rng.uniform(-1, 1), rng.uniform(-1, 1))


def away_from_zero(rng):
    """A complex number with modulus in [0.5, 1.5] (uncut products K3, A)."""
    r = rng.uniform(0.5, 1.5)
    phase = rng.uniform(-3.14159, 3.14159)
    return complex(r * mpmath.cos(phase), r * mpmath.sin(phase))


def cvec(rng):
    return [rnd(rng) for _ in range(4)]


def rvec(rng):
    return [rng.uniform(-1, 1) for _ in range(4)]


def mp(z):
    return mpmath.mpc(z.real, z.imag) if isinstance(z, complex) else mpmath.mpf(z)


def mpdot(a, b):
    a, b = [mp(x) for x in a], [mp(x) for x in b]
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def pair(z):
    z = complex(z)
    return [z.real, z.imag]


class Formulas:
    """Lambdified per-(coefficient, slot) series formulas."""

    def __init__(self, symbols, table):
        self.symbols = symbols
        self.table = {}
        for idx, formulas in table.items():
            self.table[idx] = {slot: sp.lambdify(symbols, expr, modules="mpmath")
                               for slot, expr in formulas.items()}

    def total(self, coefficients, values_for, slots):
        out = {slot: mpmath.mpc(0) for slot in slots}
        for idx, formulas in self.table.items():
            args = values_for(idx)
            for slot, fn in formulas.items():
                if slot in out:
                    out[slot] += mp(coefficients[idx]) * fn(*args)
        return [out[slot] for slot in slots]


def tri_to_bub_formulas():
    symbols = [sf.U1, sf.U2, sf.U3, sf.U4, sf.K1, sf.K2, sf.K3, sf.K4, sf.f, sf.P0, sf.P1, sf.P2]
    table, which = {}, {}
    for idx in range(21):
        formulas, u = sf.tri_formula(idx, "bub")
        if formulas:
            table[idx], which[idx] = formulas, u
    return Formulas(symbols, table), which


def tri_to_tad_formulas():
    symbols = [sf.U3, sf.U4, sf.A0, sf.B0, sf.A1, sf.B1, sf.f0, sf.f1, sf.P]
    table, which = {}, {}
    for idx in range(21):
        formulas, u = sf.tri_formula(idx, "tad")
        if formulas:
            table[idx], which[idx] = formulas, u
    return Formulas(symbols, table), which


def bub_to_tad_formulas():
    symbols = [sf.S2, sf.S3, sf.S4, sf.T2, sf.T3, sf.T4, sf.W, sf.A, sf.B, sf.f, sf.P]
    table = {}
    for idx in range(35):
        formulas = sf.bub_formula(idx)
        if formulas:
            table[idx] = formulas
    return Formulas(symbols, table)


def tri_to_bub_case(rng, formulas, which, kind):
    et3, et4, eb3, eb4 = cvec(rng), cvec(rng), cvec(rng), cvec(rng)
    c = [rnd(rng) for _ in range(21)]
    param = [rnd(rng) for _ in range(3)]
    eb1k, eb2k, eb3k, eb4k = rnd(rng), rng.uniform(-1, 1), away_from_zero(rng), rnd(rng)
    eb1t = [rnd(rng) for _ in range(4)]
    f = rnd(rng)

    def values(idx):
        u3 = which[idx] == "et3"
        u = et3 if u3 else et4
        return [mp(eb1t[0 if u3 else 1]), mp(eb1t[2 if u3 else 3]), mpdot(eb3, u), mpdot(eb4, u),
                mp(eb1k), mp(eb2k), mp(eb3k), mp(eb4k), mp(f), mp(param[0]), mp(param[1]),
                mp(param[2])]
    slots = BUBBLE_X2 if kind == "all" else [s for s in BUBBLE_X2 if s[0] >= 1]
    expected = formulas.total(c, values, slots)
    return {
        "kind": f"tri_to_bub_{kind}",
        "et3": [pair(z) for z in et3], "et4": [pair(z) for z in et4],
        "c": [pair(z) for z in c],
        "eb3": [pair(z) for z in eb3], "eb4": [pair(z) for z in eb4],
        "param": [pair(z) for z in param],
        "eb1k": pair(eb1k), "eb2k": eb2k, "eb3k": pair(eb3k), "eb4k": pair(eb4k),
        "eb1t": [pair(z) for z in eb1t], "f": pair(f),
        "slots": [list(s) for s in slots],
        "indices": [BUBBLE_X2.index(s) for s in slots],
        "expected": [pair(z) for z in expected],
    }


def tri_to_tad_case(rng, formulas, which):
    et3, et4, e3, e4 = cvec(rng), cvec(rng), cvec(rng), cvec(rng)
    c = [rnd(rng) for _ in range(21)]
    param = rnd(rng)
    e3k0, e4k0, f0 = away_from_zero(rng), rnd(rng), rnd(rng)
    e3k1, e4k1, f1 = away_from_zero(rng), rnd(rng), rnd(rng)

    def values(idx):
        u = et3 if which[idx] == "et3" else et4
        return [mpdot(e3, u), mpdot(e4, u), mp(e3k0), mp(e4k0), mp(e3k1), mp(e4k1), mp(f0),
                mp(f1), mp(param)]
    slots = [(0, 0, 0), (0, 0, 1)]
    expected = formulas.total(c, values, slots)
    return {
        "kind": "tri_to_tad",
        "et3": [pair(z) for z in et3], "et4": [pair(z) for z in et4],
        "c": [pair(z) for z in c],
        "e3": [pair(z) for z in e3], "e4": [pair(z) for z in e4],
        "param": pair(param),
        "e3k0": pair(e3k0), "e4k0": pair(e4k0), "f0": pair(f0),
        "e3k1": pair(e3k1), "e4k1": pair(e4k1), "f1": pair(f1),
        "slots": [list(s) for s in slots],
        "indices": [TADPOLE_X2.index(s) for s in slots],
        "expected": [pair(z) for z in expected],
    }


def bub_to_tad_case(rng, formulas):
    eb2, eb3, eb4, kbol = rvec(rng), cvec(rng), cvec(rng), rvec(rng)
    c = [rnd(rng) for _ in range(35)]
    e3, e4 = cvec(rng), cvec(rng)
    param = rnd(rng)
    e3k, e4k, f = away_from_zero(rng), rnd(rng), rnd(rng)
    args = [mpdot(e3, eb2), mpdot(e3, eb3), mpdot(e3, eb4), mpdot(e4, eb2), mpdot(e4, eb3),
            mpdot(e4, eb4), mpdot(eb2, kbol), mp(e3k), mp(e4k), mp(f), mp(param)]
    slots = [(0, 0, 0), (0, 0, 1)]
    expected = formulas.total(c, lambda idx: args, slots)
    return {
        "kind": "bub_to_tad",
        "eb2": eb2, "eb3": [pair(z) for z in eb3], "eb4": [pair(z) for z in eb4], "kbol": kbol,
        "c": [pair(z) for z in c],
        "e3": [pair(z) for z in e3], "e4": [pair(z) for z in e4],
        "param": pair(param), "e3k": pair(e3k), "e4k": pair(e4k), "f": pair(f),
        "slots": [list(s) for s in slots],
        "indices": [TADPOLE_X2.index(s) for s in slots],
        "expected": [pair(z) for z in expected],
    }


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    rng = random.Random(SEED)
    tb, tb_which = tri_to_bub_formulas()
    tt, tt_which = tri_to_tad_formulas()
    bt = bub_to_tad_formulas()
    cases = []
    for _ in range(SAMPLES):
        cases.append(tri_to_bub_case(rng, tb, tb_which, "all"))
        cases.append(tri_to_bub_case(rng, tb, tb_which, "spurious"))
        cases.append(tri_to_tad_case(rng, tt, tt_which))
        cases.append(bub_to_tad_case(rng, bt))
    data = {
        "format": "ninjars-rank-n2-subtraction-fixtures-v1",
        "description": (
            "Rank n+2 coefficient-level subtractions (all parent coefficients, old and new) "
            "from the series definition of tools/rank_n2/theory/subtraction_formulas.py at 40 "
            "digits, for random inputs in the argument layout of "
            "crates/ninjars/src/fast/subtractions_x2.rs. `expected[k]` is the amount "
            "subtracted from divnum[indices[k]] (slot label slots[k] = [t, x, mu2])."),
        "generator": "tools/fast/make_subtraction_fixtures_x2.py",
        "seed": SEED,
        "cases": cases,
    }
    pathlib.Path(sys.argv[1]).write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    main()
