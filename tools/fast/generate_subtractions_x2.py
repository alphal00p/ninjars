#!/usr/bin/env python3
"""Generate the closed-form rank n+2 coefficient-level subtractions
(crates/ninjars/src/fast/subtractions_x2.rs), the test reference of the series
routines the fast engine uses (subtractions_series_x2.rs).

At rank r = n+2 every subtraction splits exactly into (design section 2.6,
review section 3):

* the rank n+1 routines of original Ninja (`x1cuts::correctbubcoeffs` and
  `x1cuts::correcttadcoeffs`, FORM-generated in `coefficient_level_subs.cc`),
  unchanged statement by statement and translated exactly as
  `translate_subtractions.py` does for `subtractions.rs`, with their output
  slots re-indexed from the rank n+1 layouts to the rank n+2 ones by their
  (t, x, mu^2) label; and
* the terms of the new parent coefficients (triangle c15..c20, bubble
  c20..c34), ported mechanically from the CSE code of the theory derivation
  (`tools/rank_n2/theory/subtractions_rank_n2_cse.txt`, checked against the
  series definition by `tools/rank_n2/review/check_cse.py`).

Old parent monomials never reach the new slots (give t, x and X/t weight one:
a degree-d monomial over m uncut denominators reaches weight <= d - m), so the
sum is exact.

Symbols of the CSE code map onto the arguments of the rank n+1 routines:
tri->bub U1/U2 = `eb1t`, U3/U4 = eb3/eb4 . et3/et4, K1..K4 = `eb1k`, `eb2k`,
`eb3k`, `eb4k`, P0..P2 = `param`; tri->tad U3/U4 = e3/e4 . et3/et4,
A_i = `e3k_i`, B_i = `e4k_i`, P = `param`; bub->tad S_i/T_i = e3/e4 . eb_i,
W = eb2 . kbol, A = `e3k`, B = `e4k`. Every division by K3, A_i or A becomes a
product of the precomputed reciprocals (`ieb3k`, `ie3k0`, `ie3k1`, `ie3k`).
Temporaries feeding only dropped slots (orientation B keeps t >= 1, tadpoles
keep t^0 and t^0 mu^2) are removed.

Usage: generate_subtractions_x2.py <ninja checkout> <CSE file> <output .rs>
(the output is formatted with rustfmt afterwards).
"""

import hashlib
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import translate_subtractions as ts  # noqa: E402

# The reviewed CSE file (tools/rank_n2/theory/subtractions_rank_n2_cse.txt).
CSE_SHA256 = "d5d7fc60ccf2155493085d2dd36e0c8eef5f8c5960aee1e0936057fae49bae7b"

# Rank n+2 slot layouts, (t, x, mu^2) labels in index order (design 1.3).
BUBBLE_X2 = [(4, 0, 0), (3, 0, 0), (3, 1, 0), (2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 2, 0),
             (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 3, 0),
             (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1),
             (0, 3, 0), (0, 4, 0)]
TADPOLE_X2 = [(3, 0, 0), (2, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 0), (0, 0, 1)]

# Original Ninja's rank n+1 slot names (translate_subtractions.X1_*_JEXT) by label.
X1_BUBBLE_LABELS = {
    "jext3": (3, 0, 0), "jext2": (2, 0, 0), "jext2x1": (2, 1, 0), "jext1": (1, 0, 0),
    "jext1mu2": (1, 0, 1), "jext1x1": (1, 1, 0), "jext1x2": (1, 2, 0), "jext0": (0, 0, 0),
    "jext0mu2": (0, 0, 1), "jext0x1": (0, 1, 0), "jext0x1mu2": (0, 1, 1), "jext0x2": (0, 2, 0),
    "jext0x3": (0, 3, 0),
}
X1_TADPOLE_LABELS = {"jext0": (0, 0, 0), "jext0mu2": (0, 0, 1)}
X2_BUBBLE_JEXT = {name: BUBBLE_X2.index(label) for name, label in X1_BUBBLE_LABELS.items()}
X2_TADPOLE_JEXT = {name: TADPOLE_X2.index(label) for name, label in X1_TADPOLE_LABELS.items()}

BUBBLE_SIGNATURE = ts.BUBBLE_SIGNATURE.replace("c: &[C64]", "c: &[C64; 21]")
TAD_TRI_SIGNATURE = ts.TAD_TRI_SIGNATURE.replace("c: &[C64]", "c: &[C64; 21]")
TAD_BUB_SIGNATURE = ts.TAD_BUB_SIGNATURE.replace("c: &[C64]", "c: &[C64; 35]")


# ---------------------------------------------------------------- CSE input

def cse_sections(text):
    """{kind: [(lhs, rhs AST)]} with lhs a temporary name or a slot label."""
    sections = {}
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        header = re.match(r"// -+ (\w+):", line)
        if header:
            current = header.group(1)
            # The file repeats a section header once; restart it.
            sections[current] = []
            continue
        if not line or line.startswith("//") or current is None:
            continue
        lhs, rhs = line.rstrip(";").split("=", 1)
        lhs = lhs.strip()
        slot = re.match(r"out\[\[(\d+), (\d+), (\d+)\]\]$", lhs)
        target = tuple(int(g) for g in slot.groups()) if slot else lhs
        sections[current].append((target, ts.parse(rhs)))
    return sections


# ---------------------------------------------------------------- lowering

def integer(node):
    """Value of an integer literal (possibly negated), else None."""
    if node[0] == "neg":
        inner = integer(node[1])
        return None if inner is None else -inner
    if node[0] == "num" and node[1] == int(node[1]):
        return int(node[1])
    return None


class Lowering:
    """Lower CSE statements to Rust `let` bindings on C64 values."""

    # Sums with at least this many terms are accumulated term by term in a
    # hoisted `let mut` (same left-to-right order, so the same rounding) to
    # keep the generated lines short.
    HOIST = 4

    def __init__(self, symbols, inverses, coefficient_names, prefix):
        self.symbols = symbols          # CSE symbol -> (rust name, defining expression or None)
        self.inverses = inverses        # invertible symbol -> rust name of its reciprocal
        self.coefficients = coefficient_names
        self.prefix = prefix
        self.temps = {}                 # CSE temporary -> rust name
        self.monomials = {}             # CSE temporary -> {invertible symbol: power}
        self.used_symbols = set()
        self.pre = []                   # hoisted accumulator lines of the current statement
        self.sums = 0

    def take_pre(self):
        lines, self.pre = self.pre, []
        return lines

    def sum_chain(self, node):
        """Flatten a left-associated +/- chain into [(op, term)]."""
        terms = []
        while node[0] == "bin" and node[1] in "+-":
            terms.append((node[1], node[3]))
            node = node[2]
        terms.append(("+", node))
        terms.reverse()
        return terms

    def monomial(self, node):
        kind = node[0]
        if kind == "var":
            name = node[1]
            if name in self.inverses:
                return {name: 1}
            return self.monomials.get(name)
        if kind == "call" and node[1] == "pow":
            base, exponent = node[2]
            inner, n = self.monomial(base), integer(exponent)
            if inner is None or n is None:
                return None
            return {k: v * n for k, v in inner.items()}
        if kind == "bin" and node[1] == "*":
            left, right = self.monomial(node[2]), self.monomial(node[3])
            if left is None or right is None:
                return None
            out = dict(left)
            for k, v in right.items():
                out[k] = out.get(k, 0) + v
            return {k: v for k, v in out.items() if v}
        if kind == "bin" and node[1] == "/" and node[2] == ("num", 1.0):
            inner = self.monomial(node[3])
            return None if inner is None else {k: -v for k, v in inner.items()}
        return None

    def power(self, text, n):
        return text if n == 1 else f"p{n}({text})"

    def monomial_text(self, mono):
        """Rust product of a monomial in the invertible symbols."""
        factors = []
        for name in sorted(mono):
            n = mono[name]
            if n > 0:
                factors.append(self.power(self.symbol(name), n))
            elif n < 0:
                factors.append(self.power(self.inverses[name], -n))
        if not factors:
            raise ValueError("empty monomial")
        return " * ".join(factors) if len(factors) == 1 else "(" + " * ".join(factors) + ")"

    def symbol(self, name):
        if name in self.coefficients:
            return self.coefficients[name]
        if name not in self.symbols:
            raise ValueError(f"unknown CSE symbol {name}")
        self.used_symbols.add(name)
        return self.symbols[name][0]

    def expr(self, node):
        """(text, is_number, value)"""
        kind = node[0]
        if kind == "num":
            return ts.fmt_num(node[1]), True, node[1]
        if kind == "var":
            name = node[1]
            if name in self.temps:
                return self.temps[name], False, None
            return self.symbol(name), False, None
        if kind == "neg":
            text, number, value = self.expr(node[1])
            if number:
                return ts.fmt_num(-value), True, -value
            return f"(-{text})", False, None
        if kind == "call":
            if node[1] != "pow":
                raise ValueError(f"unsupported call {node[1]}")
            base, exponent = node[2]
            n = integer(exponent)
            if n is None or n == 0:
                raise ValueError("non-integer power")
            if n < 0:
                mono = self.monomial(base)
                if mono is None:
                    raise ValueError("negative power of a non-monomial")
                return self.monomial_text({k: v * n for k, v in mono.items()}), False, None
            text, number, value = self.expr(base)
            if number:
                return ts.fmt_num(value ** n), True, value ** n
            return self.power(text, n), False, None
        if kind == "bin" and node[1] in "+-":
            terms = self.sum_chain(node)
            if len(terms) >= self.HOIST:
                name = f"{self.prefix}s{self.sums}"
                self.sums += 1
                for k, (op, term) in enumerate(terms):
                    text, number, value = self.expr(term)
                    if k == 0:
                        if number:
                            text = f"C64::real({text})"
                        elif text.startswith("(") and text.endswith(")") and balanced(text[1:-1]):
                            text = text[1:-1]
                        self.pre.append(f"    let mut {name} = {text};")
                    else:
                        if text.startswith("(") and text.endswith(")") and balanced(text[1:-1]):
                            text = text[1:-1]
                        self.pre.append(f"    {name} {op}= {text};")
                return name, False, None
        if kind == "bin":
            op, left, right = node[1], node[2], node[3]
            if op == "/":
                lt, ln, lv = self.expr(left)
                rt, rn, rv = self.expr(right)
                if ln and rn:
                    return ts.fmt_num(lv / rv), True, lv / rv
                if rn:
                    return self.combine(lt, ln, lv, "*", ts.fmt_num(1.0 / rv), True, 1.0 / rv)
                mono = self.monomial(right)
                if mono is None:
                    raise ValueError("division by a non-monomial")
                inverse = self.monomial_text({k: -v for k, v in mono.items()})
                if ln and lv == 1.0:
                    return inverse, False, None
                return self.combine(lt, ln, lv, "*", inverse, False, None)
            lt, ln, lv = self.expr(left)
            rt, rn, rv = self.expr(right)
            return self.combine(lt, ln, lv, op, rt, rn, rv)
        raise ValueError(f"unsupported node {node}")

    def combine(self, lt, ln, lv, op, rt, rn, rv):
        if ln and rn:
            value = lv + rv if op == "+" else lv - rv if op == "-" else lv * rv
            return ts.fmt_num(value), True, value
        if op == "*" and ln and lv == 1.0:
            return rt, False, None
        if op == "*" and rn and rv == 1.0:
            return lt, False, None
        return f"({lt} {op} {rt})", False, None


def lower_section(statements, symbols, inverses, coefficients, slot_index, prefix):
    """Rust lines for one CSE section: `let` bindings of the live temporaries
    and `divnum[i] -= ...` for the kept slots (`slot_index` maps labels)."""
    # Liveness: keep only temporaries reaching a kept slot.
    live = set()
    kept = []
    for target, node in reversed(statements):
        if isinstance(target, tuple):
            if target not in slot_index:
                continue
        elif target not in live:
            continue
        names = set()
        ts.uses(node, names)
        live |= names
        kept.append((target, node))
    kept.reverse()
    low = Lowering(symbols, inverses, coefficients, prefix)
    body = []
    for target, node in kept:
        if isinstance(target, tuple):
            text, number, _ = low.expr(node)
            body += low.take_pre()
            if number:
                raise ValueError("constant subtraction")
            if text.startswith("(") and text.endswith(")") and balanced(text[1:-1]):
                text = text[1:-1]
            body.append(f"    divnum[{slot_index[target]}] -= {text};")
        else:
            mono = low.monomial(node)
            if mono is not None:
                low.monomials[target] = mono
            text, number, _ = low.expr(node)
            body += low.take_pre()
            name = f"{prefix}{target[1:]}"
            low.temps[target] = name
            if number:
                text = f"C64::real({text})"
            elif text.startswith("(") and text.endswith(")") and balanced(text[1:-1]):
                text = text[1:-1]
            body.append(f"    let {name} = {text};")
    lets = []
    for name in sorted(low.used_symbols, key=lambda s: symbols[s][0]):
        rust, definition = symbols[name]
        if definition is not None:
            lets.append(f"    let {rust} = {definition};")
    return lets + body


def balanced(text):
    depth = 0
    for ch in text:
        depth += ch == "("
        depth -= ch == ")"
        if depth < 0:
            return False
    return depth == 0


# ---------------------------------------------------------------- symbols

def tri_bub_symbols():
    symbols = {
        "U1_3": ("eb1t[0]", None), "U1_4": ("eb1t[1]", None),
        "U2_3": ("eb1t[2]", None), "U2_4": ("eb1t[3]", None),
        "U3_3": ("u3_3", "mp_cc(eb3, et3)"), "U3_4": ("u3_4", "mp_cc(eb3, et4)"),
        "U4_3": ("u4_3", "mp_cc(eb4, et3)"), "U4_4": ("u4_4", "mp_cc(eb4, et4)"),
        "K1": ("eb1k", None), "K2": ("k2", "C64::real(eb2k)"), "K3": ("eb3k", None),
        "K4": ("eb4k", None), "f": ("f", None),
        "P0": ("param[0]", None), "P1": ("param[1]", None), "P2": ("param[2]", None),
    }
    coefficients = {f"c{i}": f"c[{i}]" for i in range(15, 21)}
    return symbols, {"K3": "ieb3k"}, coefficients


def tri_tad_symbols():
    symbols = {
        "U3_3": ("u3_3", "mp_cc(e3, et3)"), "U3_4": ("u3_4", "mp_cc(e3, et4)"),
        "U4_3": ("u4_3", "mp_cc(e4, et3)"), "U4_4": ("u4_4", "mp_cc(e4, et4)"),
        "A0": ("e3k0", None), "A1": ("e3k1", None), "B0": ("e4k0", None), "B1": ("e4k1", None),
        "f0": ("f0", None), "f1": ("f1", None), "P": ("param", None),
    }
    coefficients = {f"c{i}": f"c[{i}]" for i in range(15, 21)}
    return symbols, {"A0": "ie3k0", "A1": "ie3k1"}, coefficients


def bub_tad_symbols():
    symbols = {
        "S2": ("s2", "mp_cr(e3, eb2)"), "S3": ("s3", "mp_cc(e3, eb3)"),
        "S4": ("s4", "mp_cc(e3, eb4)"), "T2": ("t2", "mp_cr(e4, eb2)"),
        "T3": ("t3", "mp_cc(e4, eb3)"), "T4": ("t4", "mp_cc(e4, eb4)"),
        "W": ("w", "C64::real(mp_rr(eb2, kbol))"), "A": ("e3k", None), "B": ("e4k", None),
        "f": ("f", None), "P": ("param", None),
    }
    coefficients = {f"c{i}": f"c[{i}]" for i in range(20, 35)}
    return symbols, {"A": "ie3k"}, coefficients


# ---------------------------------------------------------------- emission

def x1_lines(stmts, real_momenta):
    lines = []
    for stmt in stmts:
        if stmt[0] == "set":
            text = ts.emit(stmt[2], real_momenta)
            if stmt[2][0] == "bin":
                text = text[1:-1]
            lines.append(f"    let {stmt[1]} = {text};")
        else:
            lines.append(f"    divnum[{stmt[1]}] -= {stmt[2]};")
    return lines


def function(name, doc, signature, lines):
    return "\n".join([*doc, "#[inline(never)]",
                      "#[allow(clippy::too_many_arguments, unused_variables)]",
                      f"pub(super) fn {name}({signature}) {{", *lines, "}"])


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    source = pathlib.Path(sys.argv[1]) / "src/coefficient_level_subs.cc"
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != ts.EXPECTED_SHA256:
        sys.exit(f"unexpected {source} (sha256 {digest})")
    funcs = ts.functions(ts.strip_comments(data.decode()))
    names = [name for name, _ in funcs]
    expected = ["correctbubcoeffs", "correcttadcoeffs", "correcttadcoeffs", "correcttadcoeffsfull",
                "correcttadcoeffsfull"] * 2
    if names != expected:
        sys.exit(f"unexpected function layout {names}")
    x1_bubble, x1_tad_tri, x1_tad_bub = funcs[5][1], funcs[6][1], funcs[7][1]
    cse_path = pathlib.Path(sys.argv[2])
    cse_text = cse_path.read_text()
    cse_digest = hashlib.sha256(cse_text.encode()).hexdigest()
    if cse_digest != CSE_SHA256:
        sys.exit(f"unexpected {cse_path} (sha256 {cse_digest})")
    sections = cse_sections(cse_text)

    out = [
        "//! Rank n+2 coefficient-level subtractions in closed form. Generated by",
        "//! `tools/fast/generate_subtractions_x2.py`; do not edit by hand.",
        "//!",
        "//! The fast engine evaluates the same quantities by series division",
        "//! (`subtractions_series_x2.rs`); these routines are its test reference.",
        "//!",
        "//! Each routine is original Ninja's rank n+1 routine (`coefficient_level_subs.cc`,",
        "//! translated as in `subtractions.rs`) with its output slots re-indexed to the",
        "//! rank n+2 layouts, followed by the terms of the new parent coefficients",
        "//! (`tools/rank_n2/theory/subtractions_rank_n2_cse.txt`). Divisions by the",
        "//! uncut products become products of the precomputed reciprocals.",
        f"//! Ninja source SHA-256: {ts.EXPECTED_SHA256}.",
        f"//! CSE source SHA-256: {CSE_SHA256}.",
        "",
        "#![allow(clippy::all)]",
        "",
        "use super::complex::{C64, CMom, RMom, mp_cc, mp_cr, mp_rr};",
        "",
        "#[inline(always)]",
        "fn p2(x: C64) -> C64 {",
        "    x * x",
        "}",
        "",
        "#[inline(always)]",
        "fn p3(x: C64) -> C64 {",
        "    x * x * x",
        "}",
        "",
        "#[inline(always)]",
        "fn p4(x: C64) -> C64 {",
        "    let y = x * x;",
        "    y * y",
        "}",
        "",
        "#[inline(always)]",
        "fn p5(x: C64) -> C64 {",
        "    let y = x * x;",
        "    y * y * x",
        "}",
        "",
    ]
    bubble_real = {"eb2"}
    all_block, else_block = ts.split_allcoeffs(x1_bubble)
    all_slots = {label: i for i, label in enumerate(BUBBLE_X2)}
    spurious_slots = {label: i for i, label in enumerate(BUBBLE_X2) if label[0] >= 1}
    symbols, inverses, coefficients = tri_bub_symbols()
    for suffix, block, slots, doc in (
            ("all", all_block, all_slots,
             ["/// Rank n+2 triangle -> bubble subtraction, orientation A: all 22 slots",
              "/// (`x1cuts::correctbubcoeffs` re-indexed, plus triangle c15..c20)."]),
            ("spurious", else_block, spurious_slots,
             ["/// Rank n+2 triangle -> bubble subtraction, orientation B (e3 <-> e4):",
              "/// the 13 slots with t >= 1 (`x1cuts` spurious branch re-indexed, plus",
              "/// triangle c15..c18)."])):
        stmts = ts.specialize(ts.statements(block, X2_BUBBLE_JEXT, 1), ts.TRIANGLE_ZEROS[1],
                              ts.BUBBLE_PRODUCTS, {"eb3k"})
        lines = x1_lines(stmts, bubble_real)
        lines += lower_section(sections["tri_to_bub"], symbols, inverses, coefficients, slots, "w")
        out.append(function(f"bubble_from_triangle_x2_{suffix}", doc, BUBBLE_SIGNATURE, lines))
        out.append("")
    tad_slots = {(0, 0, 0): TADPOLE_X2.index((0, 0, 0)), (0, 0, 1): TADPOLE_X2.index((0, 0, 1))}
    stmts = ts.specialize(ts.statements(x1_tad_tri, X2_TADPOLE_JEXT, 1), set(),
                          ts.TAD_TRI_PRODUCTS, {"e3k0", "e3k1"})
    symbols, inverses, coefficients = tri_tad_symbols()
    lines = x1_lines(stmts, set())
    lines += lower_section(sections["tri_to_tad"], symbols, inverses, coefficients, tad_slots, "w")
    out.append(function("tadpole_from_triangle_x2",
                        ["/// Rank n+2 triangle -> tadpole subtraction on slots 4 (t^0) and 5",
                         "/// (t^0 mu²): `x1cuts::correcttadcoeffs` re-indexed, plus c15..c18."],
                        TAD_TRI_SIGNATURE, lines))
    out.append("")
    stmts = ts.specialize(ts.statements(x1_tad_bub, X2_TADPOLE_JEXT, 1), set(),
                          ts.TAD_BUB_PRODUCTS, {"e3k"})
    symbols, inverses, coefficients = bub_tad_symbols()
    lines = x1_lines(stmts, {"eb2", "kbol"})
    lines += lower_section(sections["bub_to_tad"], symbols, inverses, coefficients, tad_slots, "w")
    out.append(function("tadpole_from_bubble_x2",
                        ["/// Rank n+2 bubble -> tadpole subtraction on slots 4 (t^0) and 5",
                         "/// (t^0 mu²): `x1cuts::correcttadcoeffs` re-indexed, plus",
                         "/// c20, c21 and c23..c34."],
                        TAD_BUB_SIGNATURE, lines))
    out.append("")
    pathlib.Path(sys.argv[3]).write_text("\n".join(out))


if __name__ == "__main__":
    main()
