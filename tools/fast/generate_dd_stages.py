#!/usr/bin/env python3
"""Generate `crates/ninjars/src/fast/dd_stages.rs`: the fast engine's triangle,
bubble and tadpole stages, cut geometry, coefficient-level subtractions and
tensor-numerator jets in double-double (`fast/dd.rs`), for the re-reduction of
lanes whose binary64 reduction cancels (`docs/triangle-gram-conditioning.md`).

The items are copied from the binary64 sources (`geometry.rs`,
`subtractions.rs`, `tensor_jets.rs`, `tensor.rs`, `engine.rs`) with the
binary64 scalar type replaced: inside the generated module `C64`, `RMom` and
`CMom` are aliases of the double-double types and `f64` is replaced by `Dd`
except in constants and literals; a few anchored edits (EDITS) adapt what the
type change cannot (zero initialisers, the stability checks, which the
binary64 pass already ran). The binary64 sources are not changed.

    python3 tools/fast/generate_dd_stages.py            # write the file
    python3 tools/fast/generate_dd_stages.py --check    # exit 1 if stale
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAST = ROOT / "crates/ninjars/src/fast"
OUT = FAST / "dd_stages.rs"


def read(name):
    return (FAST / name).read_text()


def item(src, first_line):
    """The item whose definition starts with the line `first_line` (exact
    match after stripping), with its doc comments and attributes, through the
    brace that closes it."""
    lines = src.split("\n")
    hits = [i for i, l in enumerate(lines) if l.strip() == first_line.strip() or l.startswith(first_line)]
    if len(hits) != 1:
        raise SystemExit(f"anchor {first_line!r}: {len(hits)} matches")
    start = hits[0]
    while start > 0 and (lines[start - 1].strip().startswith(("///", "#[", "//!")) ):
        start -= 1
    depth, seen, end = 0, False, None
    for i in range(hits[0], len(lines)):
        code = re.sub(r"//.*", "", lines[i])
        for ch in code:
            if ch == "{":
                depth += 1
                seen = True
            elif ch == "}":
                depth -= 1
        if (seen and depth == 0) or (not seen and code.rstrip().endswith(";")):
            end = i
            break
    if end is None:
        raise SystemExit(f"unterminated item {first_line!r}")
    return "\n".join(lines[start : end + 1]) + "\n"


def body(src):
    """A whole source file without its `use` lines and module docs."""
    out = []
    for l in src.split("\n"):
        if l.startswith("use ") or l.startswith("//!") or l.startswith("#![allow"):
            continue
        out.append(l)
    return "\n".join(out)


KEEP_F64 = [
    r"f64::EPSILON",
    r"f64::NAN",
    r"f64::MIN_POSITIVE",
    r"f64::MANTISSA_DIGITS",
    r"as f64",
]


def retype(text):
    """`f64` -> `Dd` outside constants and the kept expressions."""
    keep = {}
    for k, pat in enumerate(KEEP_F64):
        tag = f"@@KEEP{k}@@"
        keep[tag] = pat.replace("\\", "")
        text = re.sub(pat, tag, text)
    lines = []
    for l in text.split("\n"):
        if re.match(r"\s*(pub(\(\w+\))? )?const [A-Z0-9_]+: f64", l):
            lines.append(l)
            continue
        l = re.sub(r"\b(\d+)_f64\b", r"Dd::from(\1.)", l)
        l = re.sub(r"\bf64\b", "Dd", l)
        lines.append(l)
    text = "\n".join(lines)
    for tag, pat in keep.items():
        text = text.replace(tag, pat)
    return text


def edit(text, old, new, count=1):
    n = text.count(old)
    if n != count:
        raise SystemExit(f"edit anchor {old[:60]!r}: {n} matches, expected {count}")
    return text.replace(old, new)


HEADER = '''//! The fast engine's triangle, bubble and tadpole stages in double-double
//! (`fast/dd.rs`) for the re-reduction of lanes whose binary64 reduction
//! cancels (`docs/triangle-gram-conditioning.md`).
//!
//! Generated from `geometry.rs`, `subtractions.rs`, `tensor_jets.rs`,
//! `tensor.rs` and `engine.rs` by `tools/fast/generate_dd_stages.py`; do not
//! edit by hand (`python3 tools/fast/generate_dd_stages.py`). Inside this
//! module `C64`, `RMom` and `CMom` are the double-double types.

#![allow(clippy::all, dead_code, unused_variables, unused_mut, unused_assignments)]

use super::dd::{Cd as C64, CMomD as CMom, Dd, RMomD as RMom, mp_cc, mp_cr, mp_rr};
use super::dd_driver::{
    DdStages, FastInput, Stability, den_scale, difference_scale, dot_scale, euclidean_or, fail,
    gram_scale, taxicab_sum,
};
use super::numerator::{t2_len_le3, t3_len};
use super::tensor_layout::{block_offset, ncoeff_homogeneous};
use super::InstabilityTest;
use crate::Result;

'''


def geometry():
    g = body(read("geometry.rs"))
    g = retype(g)
    g = edit(g, "fn sign(x: Dd, s: Dd) -> Dd {\n    if s < 0. { -x } else { x }", "fn sign(x: impl Into<Dd>, s: Dd) -> Dd {\n    let x = x.into();\n    if s < 0. { -x } else { x }")
    # Zero initialisers that the type change cannot infer.
    g = edit(g, "let (mut r1, mut r2) = (0., 0.);", "let (mut r1, mut r2) = (Dd::ZERO, Dd::ZERO);")
    g = edit(g, "let mut r1 = 0.;", "let mut r1 = Dd::ZERO;")
    g = edit(g, "Self::complete(e1, k2, r1, 0., k1k2)", "Self::complete(e1, k2, r1, Dd::ZERO, k1k2)")
    # Ninja's fixed reference of `Basis(k)` and the tadpole candidates: exact
    # binary64 vectors, promoted.
    g = edit(g, """        let k2 = [
            sign(1., k[0]),
            -sign(INVSQRT3, k[1]),
            -sign(INVSQRT3, k[2]),
            -sign(INVSQRT3, k[3]),
        ];""", """        // Ninja's reference, light-like in double-double: its energy is
        // the double-double norm of the binary64 spatial part (binary64's
        // `1 - 3 (1/sqrt 3)²` is a rounding residue that the reduction of a
        // degenerate lane amplifies like any other).
        let third = Dd::from(INVSQRT3);
        let k2 = [
            sign((third * third * 3.).sqrt(), k[0]),
            -sign(third, k[1]),
            -sign(third, k[2]),
            -sign(third, k[3]),
        ];""")
    g = edit(g, "    refs.map(|(r1, r2)| Basis::from_pair(&r1, &r2, eps))",
             "    refs.map(|(r1, r2): ([f64; 4], [f64; 4])| Basis::from_pair(&r1.map(Dd::from), &r2.map(Dd::from), eps))")
    g = edit(g, "pub(crate) fn tadpole_candidates(eps: Dd) -> [Basis; 3] {", "pub(crate) fn tadpole_candidates(eps: f64) -> [Basis; 3] {")
    g = edit(g, """    let sqrt2 = Dd::from(2.).sqrt();
    let sqrt3 = Dd::from(3.).sqrt();
    let invsqrt2 = 0.5 * sqrt2;
    let invsqrt3 = 1. / Dd::from(3.).sqrt();""", """    let sqrt2 = 2_f64.sqrt();
    let sqrt3 = 3_f64.sqrt();
    let invsqrt2 = 0.5 * sqrt2;
    let invsqrt3 = 1. / 3_f64.sqrt();""")
    # Basis thresholds stay binary64 numbers.
    g = edit(g, "pub(crate) fn from_pair(k1: &RMom, k2: &RMom, eps: Dd) -> Self {", "pub(crate) fn from_pair(k1: &RMom, k2: &RMom, eps: f64) -> Self {")
    g = edit(g, "pub(crate) fn from_single(k: &RMom, eps: Dd) -> Self {", "pub(crate) fn from_single(k: &RMom, eps: f64) -> Self {")
    g = edit(g, "    eps: Dd,\n) -> bool {", "    eps: f64,\n) -> bool {")
    g = edit(g, "            mp12 = if one_plus.abs() < ORTHOGONAL {", "            mp12 = if one_plus.abs() < ORTHOGONAL {")
    # The spinor's axis tests compare exact zeros.
    g = edit(g, "        let zero = C64::ZERO;", "        let zero = C64::ZERO;\n        let (plus, minus) = (p[0] + p[3], p[0] - p[3]);")
    g = edit(g, "        let plus = p[0] + p[3];\n        let minus = p[0] - p[3];\n", "")
    g = edit(g, "            let v = signed_root(2. * p[0]);", "            let v = signed_root(p[0] * 2.);")
    return g


def subtractions():
    s = body(read("subtractions.rs"))
    return retype(s)


def jets():
    j = body(read("tensor_jets.rs"))
    return retype(j)


def tensor():
    t = read("tensor.rs")
    parts = [
        item(t, "fn splits(d: usize) -> (usize, usize, usize) {"),
        item(t, "pub(super) fn horner_scalar("),
        item(t, "pub(super) const fn t3_len_const(i: usize) -> usize {"),
        item(t, "pub(super) const fn t2_offset(i: usize) -> usize {"),
        item(t, "pub(super) fn horner_jet<P: Jet<L>, const L: usize>("),
        item(t, "pub(super) fn t3_output<const L: usize>("),
        item(t, "pub(super) fn t2_output<const L: usize>("),
    ]
    return retype("\n".join(parts))


ENGINE_ITEMS = [
    "struct TriangleData {",
    "struct BubbleData {",
    "pub(super) struct TadpoleData {",
    "pub(super) struct Uncut {",
    "struct BoxRow {",
    "pub(super) fn den(q: &CMom, v: &RMom, m2: C64) -> C64 {",
    "struct TriangleRow {",
    "struct BubbleRow {",
    "struct TadpoleRow {",
    "struct Calls {",
    "impl Calls {",
    "pub(super) fn sub_cr(a: &CMom, b: &RMom) -> CMom {",
    "pub(super) fn scale_c(a: &CMom, s: f64) -> CMom {",
    "pub(super) fn is_zero(z: C64) -> bool {",
    "pub(super) fn divide_t3(num: &mut [C64], nterms: usize, d0inv: C64, d1: C64, d2: [C64; 2]) {",
    "fn divide_t3_n<const NT: usize>(num: &mut [C64], d0inv: C64, d1: C64, d2: [C64; 2]) {",
    "pub(super) fn divide_t2(num: &mut [C64], nterms: usize, d0inv: C64, d1: [C64; 2], d2: [C64; 4]) {",
    "fn divide_t2_n<const NT: usize>(num: &mut [C64], d0inv: C64, d1: [C64; 2], d2: [C64; 4]) {",
    "type BubbleCorrection = fn(",
    "fn tadpole_from_triangle(",
    "fn tadpole_from_bubble(",
]

STAGES = [
    "    fn boxes<N: FastNumerator + ?Sized>(&mut self, input: &FastInput, num: &mut N) -> Result<()> {",
    "    fn triangles<N: FastNumerator + ?Sized>(",
    "    fn bubbles<N: FastNumerator + ?Sized>(&mut self, input: &FastInput, num: &mut N) -> Result<()> {",
    "    fn tadpoles<N: FastNumerator + ?Sized>(",
    "    fn rank3_bubble(&self, b: usize, lane: usize, v: &[RMom]) -> bool {",
]


def engine():
    e = read("engine.rs")
    items = [item(e, a) for a in ENGINE_ITEMS]
    stages = [item(e, a) for a in STAGES]
    text = "\n".join(items)
    text += "\nimpl DdStages {\n" + "\n".join(stages) + "}\n"
    text = retype(text)
    text = text.replace("<N: FastNumerator + ?Sized>", "<N: DdNumerator + ?Sized>")
    # Without the diagnosis hooks of the binary64 stages, if present.
    text = re.sub(r"\n\s*if super::diag::enabled\(\)\n\s*&& let Some\(inj\) = super::diag::injected\(&p\)\n\s*\{\n\s*c = inj;\n\s*\}", "", text)
    text = edit(text, "#[derive(Clone, Copy, Debug, Default)]\nstruct TriangleData {", "#[derive(Clone, Copy, Debug, Default)]\npub(super) struct TriangleData {")
    text = edit(text, "#[derive(Clone, Copy, Debug, Default)]\nstruct BubbleData {", "#[derive(Clone, Copy, Debug, Default)]\npub(super) struct BubbleData {")
    for f in ["    e3: CMom,\n    e4: CMom,\n    c: [C64; 15],", ]:
        text = edit(text, f, "    pub(super) e3: CMom,\n    pub(super) e4: CMom,\n    pub(super) c: [C64; 15],")
    text = edit(text, "    e2: RMom,\n    e3: CMom,\n    e4: CMom,\n    c: [C64; 20],", "    pub(super) e2: RMom,\n    pub(super) e3: CMom,\n    pub(super) e4: CMom,\n    pub(super) c: [C64; 20],")
    for s in ["struct TriangleRow {", "struct BubbleRow {", "struct TadpoleRow {", "struct Calls {"]:
        text = edit(text, "\n" + s, "\npub(super) " + s)
    text = edit(text, "\nimpl Calls {\n    fn clear(&mut self) {", "\nimpl Calls {\n    pub(super) fn clear(&mut self) {")
    text = edit(text, "    fn prepare_out(&mut self, width: usize) {", "    pub(super) fn prepare_out(&mut self, width: usize) {")
    # Zero initialisers.
    text = edit(text, "                            e2k: 0.,", "                            e2k: Dd::ZERO,")
    text = edit(text, "pub(super) fn is_zero(z: C64) -> bool {\n    z.re == 0. && z.im == 0.\n}", "pub(super) fn is_zero(z: C64) -> bool {\n    z.is_zero()\n}")
    # The box stage without the binary64 mu-expansion conditioning screen
    # (`crate::conditioning`): the re-reduction's own estimate judges the
    # double-double result.
    for line in [
        "            let mut suspects: Vec<(usize, [Dd; 2], Option<BoxMuConditioning>)> = Vec::new();\n",
        "                let mut conditioning = BoxMuConditioning::ONE;\n",
        "                        conditioning.line(BOX_MU_PRETEST * bound, factor.taxicab());\n",
        "                    let vnorm = euclidean_norm(&row.vort);\n",
        "                    let t = euclidean_norm(&sub_cc(&row.q[0], &row.mid)) / vnorm;\n",
        "                        let length = kk.iter().map(|x| x * x).sum::<Dd>().sqrt();\n",
        "                        conditioning.line(2. * vnorm * length, d0.abs());\n",
        "                        conditioning.expansion(d1.abs(), d0.abs(), t);\n",
    ]:
        text = edit(text, line, "")
    text = edit(text, """                    if conditioning.exceeds(0, 1., f64::MANTISSA_DIGITS) {
                        suspects.push((r, [self.calls.out[r].taxicab(), 0.], None));
                    }
""", "")
    text = edit(text, """                    if conditioning.exceeds(1, 1., f64::MANTISSA_DIGITS) {
                        let numerator = [
                            self.calls.out[2 * r].taxicab(),
                            self.calls.out[2 * r + 1].taxicab(),
                        ];
                        suspects.push((r, numerator, Some(conditioning)));
                    }
""", "")
    text = edit(text, """            if !suspects.is_empty() {
                self.box_mu_conditioning(input, num, &p, b, &suspects, u32::from(x1))?;
            }
""", "")
    # Record each box's conditioning `log2 λ²` (`crate::conditioning`, the
    # box-Gram bound): λ² = (|l+|² + |l-|²) / (2 S²) for the two solutions
    # (the parallelogram identity of |a|² + |t v|²), S² the largest squared
    # shift or squared mass of the box's lines.
    text = edit(text, """                // The box's conditioning, for the cancellation rescue's trigger
                // (no work when the rescue is off).
                if self.options.cancellation_rescue {
                    self.box_lambda[b * lanes + lane] = box_lambda_bits(&l, v, m, &p);
                }
                let q = [sub_cr(&l[0], &v[c1]), sub_cr(&l[1], &v[c1])];""", """                let q = [sub_cr(&l[0], &v[c1]), sub_cr(&l[1], &v[c1])];
                let norm = |x: &CMom| x.iter().map(|z| z.norm_sqr()).sum::<Dd>();
                let scale = p
                    .iter()
                    .map(|&i| v[i].iter().map(|x| *x * *x).sum::<Dd>().max(m[i].abs()))
                    .fold(Dd::ZERO, Dd::max);
                self.box_lambda[b * lanes + lane] =
                    ((norm(&l[0]) + norm(&l[1])) / (scale * 2.)).to_f64().log2();""")
    text = text.replace("    fn boxes<", "    pub(super) fn boxes<")
    text = edit(text, "\nstruct BoxRow {", "\npub(super) struct BoxRow {")
    # The stages are methods of the double-double workspace.
    text = text.replace("    fn triangles<", "    pub(super) fn triangles<")
    text = text.replace("    fn bubbles<", "    pub(super) fn bubbles<")
    text = text.replace("    fn tadpoles<", "    pub(super) fn tadpoles<")
    text = text.replace("    fn rank3_bubble(", "    pub(super) fn rank3_bubble(")
    return text


def numerator_trait():
    n = read("numerator.rs")
    t = item(n, "    fn t3_expansion(")
    b = item(n, "    fn t2_expansion(")
    ev = item(n, "    fn evaluate(")
    mu = item(n, "    fn mu_expansion(")
    text = "/// The expansions the double-double stages request (`FastNumerator`'s).\npub(super) trait DdNumerator {\n" + ev + "\n" + mu + "\n" + t + "\n" + b + "}\n"
    return retype(text)


def generate():
    out = HEADER
    out += "// ---- geometry.rs\n" + geometry()
    out += "\n// ---- subtractions.rs\nmod subs {\n#![allow(clippy::all, dead_code, unused_variables)]\nuse super::super::dd::{Cd as C64, CMomD as CMom, Dd, RMomD as RMom, mp_cc, mp_cr, mp_rr};\n" + subtractions() + "\n}\n"
    out += "\n// ---- tensor_jets.rs\npub(crate) mod jets {\n#![allow(clippy::all, dead_code)]\nuse super::super::dd::Cd as C64;\n" + jets().replace("pub(super) ", "pub(crate) ") + "\n}\npub(crate) use jets::Jet;\n"
    out += "\n// ---- tensor.rs\n" + tensor()
    out += "\n// ---- numerator.rs\n" + numerator_trait()
    out += "\n// ---- engine.rs\n" + engine()
    return out


def main():
    text = generate()
    if "--check" in sys.argv:
        if OUT.read_text() != text:
            print(f"{OUT} is stale", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
