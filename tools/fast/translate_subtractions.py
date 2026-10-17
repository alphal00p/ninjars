#!/usr/bin/env python3
"""Mechanically translate original Ninja's FORM-generated coefficient-level
subtractions (src/coefficient_level_subs.cc) into Rust for the fast engine.

The generated statements are kept in their original order and grouping. Two
safe transformations are applied for each specialization:

* rank-dependent structural zeros: parent coefficients which original Ninja
  never sets for the given rank (they stay zero-initialized there) are replaced
  by an exact zero and identically-zero products/sums are folded away;
* dead-code elimination of temporaries that only fed removed terms.

Both only drop exact `0 * x` / `x + 0` contributions, so finite results agree
with original Ninja up to rounding. Every temporary is renamed into static
single assignment form (`zN_v`).

Usage: translate_subtractions.py <ninja checkout> <output .rs>
"""

import hashlib
import pathlib
import re
import sys

EXPECTED_SHA256 = "cf408bc630a23ffa2a95a1f2479c3f6059d1f862fa6851897200deb788becc42"

CONSTANTS = {
    "HALF": 0.5, "ONE": 1.0, "TWO": 2.0, "THREE": 3.0, "FOUR": 4.0,
    "SIX": 6.0, "EIGHT": 8.0, "INV8": 0.125, "INV4": 0.25,
}


# ---------------------------------------------------------------- parsing

def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def matching_brace(text, open_index):
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced braces")


def functions(text):
    """Bodies of every subtraction function in source order."""
    out = []
    for match in re.finditer(r"void\s+(correctbubcoeffs|correcttadcoeffs|correcttadcoeffsfull)\s*\(", text):
        start = text.index("{", match.end())
        end = matching_brace(text, start)
        out.append((match.group(1), text[start + 1:end]))
    return out


def split_allcoeffs(body):
    head = body.index("if (allcoeffs)")
    first_open = body.index("{", head)
    first_close = matching_brace(body, first_open)
    rest = body[first_close + 1:]
    else_open = rest.index("{", rest.index("else"))
    else_close = matching_brace(rest, else_open)
    return body[first_open + 1:first_close], rest[else_open + 1:else_close]


TOKEN = re.compile(r"\s*(?:(\d+\.\d*|\d*\.\d+|\d+)|([A-Za-z_][A-Za-z_0-9]*)|(.))")


def tokenize(expr):
    tokens = []
    pos = 0
    expr = expr.strip()
    while pos < len(expr):
        m = TOKEN.match(expr, pos)
        if not m:
            raise ValueError(expr[pos:])
        pos = m.end()
        if m.group(1):
            tokens.append(("num", float(m.group(1))))
        elif m.group(2):
            tokens.append(("id", m.group(2)))
        elif m.group(3) and not m.group(3).isspace():
            tokens.append(("op", m.group(3)))
    return tokens


class Parser:
    def __init__(self, tokens):
        self.t = tokens
        self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self, kind=None, value=None):
        tok = self.peek()
        if kind and tok[0] != kind or value and tok[1] != value:
            raise ValueError(f"expected {kind} {value}, got {tok}")
        self.i += 1
        return tok

    def expr(self):
        node = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            node = ("bin", op, node, self.term())
        return node

    def term(self):
        node = self.unary()
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.take()[1]
            node = ("bin", op, node, self.unary())
        return node

    def unary(self):
        if self.peek() == ("op", "-"):
            self.take()
            return ("neg", self.unary())
        if self.peek() == ("op", "+"):
            self.take()
            return self.unary()
        return self.primary()

    def primary(self):
        kind, value = self.peek()
        if kind == "num":
            self.take()
            return ("num", value)
        if kind == "op" and value == "(":
            self.take()
            node = self.expr()
            self.take("op", ")")
            return node
        if kind == "id":
            self.take()
            if value in CONSTANTS:
                return ("num", CONSTANTS[value])
            if value == "Real":
                self.take("op", "(")
                node = self.expr()
                self.take("op", ")")
                return node
            if self.peek() == ("op", "("):
                self.take()
                args = [self.expr()]
                while self.peek() == ("op", ","):
                    self.take()
                    args.append(self.expr())
                self.take("op", ")")
                return ("call", value, args)
            if self.peek() == ("op", "["):
                self.take()
                index = self.take("num")[1]
                self.take("op", "]")
                return ("index", value, int(index))
            return ("var", value)
        raise ValueError(f"unexpected token {kind} {value}")


def parse(expr):
    p = Parser(tokenize(expr))
    node = p.expr()
    if p.i != len(p.t):
        raise ValueError(f"trailing tokens in {expr}")
    return node


# ---------------------------------------------------------------- statements

def statements(block, jext, rmn):
    """(target, expression) assignments and (output index, Z name) subtractions."""
    stmts = []
    for raw in block.split(";"):
        s = " ".join(raw.split())
        if not s or s.startswith("Complex ") or s.startswith("const int"):
            continue
        guard = re.match(r"if \(rmn >= (-?\d+)\) (.*)", s)
        if guard:
            if rmn < int(guard.group(1)):
                continue
            s = guard.group(2)
        sub = re.match(r"divnum\[(\w+)\] -= (\w+)$", s)
        if sub:
            stmts.append(("sub", jext[sub.group(1)], sub.group(2)))
            continue
        assign = re.match(r"(Z\d+_)\s*=\s*(.*)$", s)
        if not assign:
            raise ValueError(f"unrecognized statement: {s}")
        stmts.append(("set", assign.group(1), parse(assign.group(2))))
    return stmts


# ---------------------------------------------------------------- simplification

def is_num(node, value=None):
    return node[0] == "num" and (value is None or node[1] == value)


def simplify(node, env, zeros):
    """Rename temporaries to SSA names, substitute structural zeros, fold."""
    kind = node[0]
    if kind == "num":
        return node
    if kind == "var":
        name = node[1]
        if name.startswith("Z"):
            ssa = env[name]
            return ("num", 0.0) if ssa is None else ("var", ssa)
        return node
    if kind == "index":
        if node[1] == "c" and node[2] in zeros:
            return ("num", 0.0)
        return node
    if kind == "call":
        return ("call", node[1], [simplify(a, env, zeros) for a in node[2]])
    if kind == "neg":
        inner = simplify(node[1], env, zeros)
        if is_num(inner):
            return ("num", -inner[1])
        return ("neg", inner)
    op, left, right = node[1], simplify(node[2], env, zeros), simplify(node[3], env, zeros)
    if is_num(left) and is_num(right):
        a, b = left[1], right[1]
        if op == "/" and b == 0.0:
            raise ValueError("constant division by zero")
        return ("num", a + b if op == "+" else a - b if op == "-" else a * b if op == "*" else a / b)
    if op == "*":
        if is_num(left, 0.0) or is_num(right, 0.0):
            return ("num", 0.0)
    if op == "/":
        if is_num(right, 0.0):
            raise ValueError("division by a structural zero")
        if is_num(left, 0.0):
            return ("num", 0.0)
    if op == "+":
        if is_num(left, 0.0):
            return right
        if is_num(right, 0.0):
            return left
    if op == "-":
        if is_num(right, 0.0):
            return left
        if is_num(left, 0.0):
            return ("neg", right)
    return ("bin", op, left, right)


def uses(node, out):
    if node[0] == "var":
        out.add(node[1])
    elif node[0] == "call":
        for a in node[2]:
            uses(a, out)
    elif node[0] == "neg":
        uses(node[1], out)
    elif node[0] == "bin":
        uses(node[2], out)
        uses(node[3], out)


def specialize(stmts, zeros):
    env = {}
    version = {}
    lowered = []
    for stmt in stmts:
        if stmt[0] == "set":
            _, target, expr = stmt
            expr = simplify(expr, env, zeros)
            if is_num(expr, 0.0):
                env[target] = None
                continue
            version[target] = version.get(target, 0) + 1
            ssa = f"{target[:-1].lower()}_{version[target]}"
            env[target] = ssa
            lowered.append(("set", ssa, expr))
        else:
            _, index, name = stmt
            if env.get(name) is not None:
                lowered.append(("sub", index, env[name]))
    live = set()
    kept = []
    for stmt in reversed(lowered):
        if stmt[0] == "sub":
            live.add(stmt[2])
            kept.append(stmt)
        elif stmt[1] in live:
            uses(stmt[2], live)
            kept.append(stmt)
    return list(reversed(kept))


# ---------------------------------------------------------------- emission

def fmt_num(value):
    text = repr(float(value))
    return text if ("." in text or "e" in text or "inf" in text) else text + ".0"


def emit(node, real_momenta):
    kind = node[0]
    if kind == "num":
        return fmt_num(node[1])
    if kind == "var":
        return node[1]
    if kind == "index":
        return f"{node[1]}[{node[2]}]"
    if kind == "neg":
        return f"(-{emit(node[1], real_momenta)})"
    if kind == "call":
        if node[1] != "mp" or len(node[2]) != 2:
            raise ValueError(f"unsupported call {node[1]}")
        a, b = node[2]
        if a[0] != "var" or b[0] != "var":
            raise ValueError("mp arguments must be momenta")
        ra, rb = a[1] in real_momenta, b[1] in real_momenta
        if ra and rb:
            return f"mp_rr({a[1]}, {b[1]})"
        if ra:
            return f"mp_cr({b[1]}, {a[1]})"
        if rb:
            return f"mp_cr({a[1]}, {b[1]})"
        return f"mp_cc({a[1]}, {b[1]})"
    return f"({emit(node[2], real_momenta)} {node[1]} {emit(node[3], real_momenta)})"


def rust_function(name, doc, signature, stmts, real_momenta):
    lines = [f"/// {doc}", "#[inline(never)]", "#[allow(clippy::too_many_arguments, unused_variables)]",
             f"pub(super) fn {name}({signature}) {{"]
    for stmt in stmts:
        if stmt[0] == "set":
            text = emit(stmt[2], real_momenta)
            if stmt[2][0] == "bin":
                text = text[1:-1]
            lines.append(f"    let {stmt[1]} = {text};")
        else:
            lines.append(f"    divnum[{stmt[1]}] -= {stmt[2]};")
    lines.append("}")
    return "\n".join(lines)


def normal_bubble_jext(rmn):
    return {
        "jext2": rmn, "jext1": rmn + 1, "jext1x1": 2,
        "jext0": rmn + 2 + (1 if rmn >= 0 else 0), "jext0mu2": 4,
        "jext0x1": 2 + (3 if rmn >= 0 else 0), "jext0x2": 6,
    }


X1_BUBBLE_JEXT = {
    "jext3": 0, "jext2": 1, "jext2x1": 2, "jext1": 3, "jext1mu2": 4, "jext1x1": 5,
    "jext1x2": 6, "jext0": 7, "jext0mu2": 8, "jext0x1": 9, "jext0x1mu2": 10,
    "jext0x2": 11, "jext0x3": 12,
}
X1_TADPOLE_JEXT = {"jext0": 2, "jext0mu2": 3}

# Triangle coefficients never set by original Ninja for rank - n = rmn.
TRIANGLE_ZEROS = {-2: {2, 3, 5, 6, 7, 8, 9}, -1: {3, 6, 8, 9}, 0: set(), 1: set()}

BUBBLE_SIGNATURE = ("divnum: &mut [C64], et3: &CMom, et4: &CMom, c: &[C64], eb1: &CMom, "
                    "eb2: &RMom, eb3: &CMom, eb4: &CMom, param: &[C64; 3], k: &RMom, f: C64")
TAD_TRI_SIGNATURE = ("divnum: &mut [C64], et3: &CMom, et4: &CMom, c: &[C64], e3: &CMom, e4: &CMom, "
                     "param: C64, k0: &RMom, f0: C64, k1: &RMom, f1: C64")
TAD_BUB_SIGNATURE = ("divnum: &mut [C64], eb2: &RMom, eb3: &CMom, eb4: &CMom, kbol: &RMom, c: &[C64], "
                     "e3: &CMom, e4: &CMom, param: C64, k: &RMom, f: C64")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    source = pathlib.Path(sys.argv[1]) / "src/coefficient_level_subs.cc"
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256:
        sys.exit(f"unexpected {source} (sha256 {digest})")
    funcs = functions(strip_comments(data.decode()))
    names = [name for name, _ in funcs]
    expected = ["correctbubcoeffs", "correcttadcoeffs", "correcttadcoeffs", "correcttadcoeffsfull",
                "correcttadcoeffsfull"] * 2
    if names != expected:
        sys.exit(f"unexpected function layout {names}")
    normal_bubble, x1_bubble = funcs[0][1], funcs[5][1]
    x1_tad_tri, x1_tad_bub = funcs[6][1], funcs[7][1]

    out = [
        "//! Coefficient-level subtractions of original Ninja (`coefficient_level_subs.cc`,",
        "//! FORM-generated), translated mechanically by `tools/fast/translate_subtractions.py`.",
        "//! Do not edit by hand. Parent coefficients that original Ninja never sets",
        "//! for a rank are folded as exact zeros; statement order is preserved.",
        f"//! Source SHA-256: {EXPECTED_SHA256}.",
        "",
        "#![allow(clippy::all)]",
        "",
        "use super::complex::{C64, CMom, RMom, mp_cc, mp_cr, mp_rr};",
        "",
    ]
    bubble_real = {"eb2", "k"}
    for rmn in (-2, -1, 0):
        all_block, else_block = split_allcoeffs(normal_bubble)
        jext = normal_bubble_jext(rmn)
        variants = [("all", all_block, "all"), ("spurious", else_block, "second-orientation")]
        # Without linear bubble terms (rank = n-2) original Ninja has no second orientation.
        for suffix, block, which in variants[:1 if rmn < -1 else 2]:
            stmts = specialize(statements(block, jext, rmn), TRIANGLE_ZEROS[rmn])
            name = f"bubble_from_triangle_r{'m' + str(-rmn) if rmn < 0 else rmn}_{suffix}"
            doc = (f"`cuts::correctbubcoeffs` for rank - n = {rmn}, {which} coefficients.")
            out.append(rust_function(name, doc, BUBBLE_SIGNATURE, stmts, bubble_real))
            out.append("")
    all_block, else_block = split_allcoeffs(x1_bubble)
    for suffix, block, which in (("all", all_block, "all"), ("spurious", else_block, "second-orientation")):
        stmts = specialize(statements(block, X1_BUBBLE_JEXT, 1), TRIANGLE_ZEROS[1])
        name = f"bubble_from_triangle_x1_{suffix}"
        doc = f"`x1cuts::correctbubcoeffs` (rank n+1), {which} coefficients."
        out.append(rust_function(name, doc, BUBBLE_SIGNATURE, stmts, bubble_real))
        out.append("")
    stmts = specialize(statements(x1_tad_tri, X1_TADPOLE_JEXT, 1), set())
    out.append(rust_function("tadpole_from_triangle_x1",
                             "`x1cuts::correcttadcoeffs` (triangle parent, rank n+1).",
                             TAD_TRI_SIGNATURE, stmts, {"k0", "k1"}))
    out.append("")
    stmts = specialize(statements(x1_tad_bub, X1_TADPOLE_JEXT, 1), set())
    out.append(rust_function("tadpole_from_bubble_x1",
                             "`x1cuts::correcttadcoeffs` (bubble parent, rank n+1).",
                             TAD_BUB_SIGNATURE, stmts, {"eb2", "kbol", "k"}))
    out.append("")
    pathlib.Path(sys.argv[2]).write_text("\n".join(out))


if __name__ == "__main__":
    main()
