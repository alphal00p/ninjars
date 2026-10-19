//! Generator of `engine/conditioned.rs`, the rank <= n+1 stages with the
//! conditioned cut geometry ([`super::FastOptions::conditioned_cuts`]).
//!
//! The conditioned stages are `evaluate_up_to_n1`, `boxes`, `bubbles` and
//! `tadpoles` of `engine.rs`, copied verbatim except for the edits in
//! [`FUNCTIONS`]: the box normal, center and solutions of `box_x2.rs`, the
//! bubble reference of `bases_x2.rs` and its tadpole bases. A copy, rather
//! than one body for both variants, leaves the default stages' source, and
//! with it their machine code, exactly as before the option existed (one
//! body selected by a const generic, inlined into the two entries, changed
//! the default instances' instruction scheduling and register allocation).
//!
//! `conditioned_stages_are_generated_from_the_default_ones` fails while the
//! file differs from what the current `engine.rs` gives; `NINJARS_BLESS=1
//! cargo test -p ninjars --lib conditioned_stages` rewrites it. An edit
//! whose anchor no longer occurs exactly once fails as well, so changes to
//! the default stages cannot silently miss the conditioned ones.

const ENGINE: &str = include_str!("engine.rs");
const GENERATED: &str = "src/fast/engine/conditioned.rs";

const HEADER: &str = r#"//! The rank <= n+1 stages with [`super::FastOptions::conditioned_cuts`]:
//! `evaluate_up_to_n1`, `boxes`, `bubbles` and `tadpoles` of `engine.rs` with
//! the rank n+2 engine's cut geometry (the box normal, center and solutions
//! of `box_x2.rs`; per lane the bubble reference and the tadpole basis of
//! `bases_x2.rs` that best condition the uncut lines). Reduced by
//! [`super::ConditionedEngine`].
//!
//! Generated from `engine.rs` by `fast/tests_conditioned.rs`; do not edit by
//! hand (`NINJARS_BLESS=1 cargo test -p ninjars --lib conditioned_stages`).

use super::*;
use super::super::bases_x2::bubble_basis_x2;
use super::super::box_x2::box_geometry_x2;

impl FastAmplitude {
"#;

/// Per function: the first line of its definition in `engine.rs`, its new
/// first lines, and the edits of its body (each anchor must occur exactly
/// once in the function).
type Function = (
    &'static str,
    &'static str,
    &'static [(&'static str, &'static str)],
);

const FUNCTIONS: &[Function] = &[
    (
        "    fn evaluate_up_to_n1<N: FastNumerator + ?Sized, M: FastMasters + ?Sized>(
        &mut self,
        input: &FastInput,
        numerator: &mut N,
        masters: &mut M,
        results: &mut [FastResult],
        statuses: &mut [FastStatus],
    ) -> Result<()> {
",
        "    /// [`FastAmplitude::evaluate_up_to_n1`] with the conditioned stages.
    pub(super) fn evaluate_up_to_n1_conditioned<N: FastNumerator + ?Sized, M: FastMasters + ?Sized>(
        &mut self,
        input: &FastInput,
        numerator: &mut N,
        masters: &mut M,
        results: &mut [FastResult],
        statuses: &mut [FastStatus],
        tadpole_bases: &TadpoleBasesX2,
    ) -> Result<()> {
",
        &[
            (
                "self.boxes(input, numerator)?;",
                "self.boxes_conditioned(input, numerator)?;",
            ),
            (
                "self.bubbles(input, numerator)?;",
                "self.bubbles_conditioned(input, numerator)?;",
            ),
            (
                "self.tadpoles(input, numerator)?;",
                "self.tadpoles_conditioned(input, numerator, tadpole_bases)?;",
            ),
        ],
    ),
    (
        "    fn boxes<N: FastNumerator + ?Sized>(&mut self, input: &FastInput, num: &mut N) -> Result<()> {
",
        "    /// [`FastAmplitude::boxes`] with the rank n+2 engine's box geometry.
    fn boxes_conditioned<N: FastNumerator + ?Sized>(&mut self, input: &FastInput, num: &mut N) -> Result<()> {
",
        &[(
            "                let q = [sub_cr(&l[0], &v[c1]), sub_cr(&l[1], &v[c1])];
                let mut mid = [C64::ZERO; 4];
                if x1 {
                    for mu in 0..4 {
                        mid[mu] = (l[0][mu] + l[1][mu]) * 0.5 - v[c1][mu];
                    }
                }
                self.box_rows.push(BoxRow { lane, vort, q, mid });
                for qq in q {
",
            "                // The rank n+2 engine's geometry (box_x2.rs).
                let Some(g) = box_geometry_x2(
                    &[v[c1], v[c2], v[c3], v[c4]],
                    &[m[c1], m[c2], m[c3], m[c4]],
                    &l,
                ) else {
                    fail(&mut self.alive, &mut self.lane_status, lane);
                    continue;
                };
                self.box_rows.push(BoxRow {
                    lane,
                    vort: g.normal,
                    q: g.q,
                    mid: g.mid,
                });
                for qq in g.q {
",
        )],
    ),
    (
        "    fn bubbles<N: FastNumerator + ?Sized>(&mut self, input: &FastInput, num: &mut N) -> Result<()> {
",
        "    /// [`FastAmplitude::bubbles`] with the rank n+2 engine's bubble references.
    fn bubbles_conditioned<N: FastNumerator + ?Sized>(&mut self, input: &FastInput, num: &mut N) -> Result<()> {
",
        &[(
            "                let e = Basis::from_single(&k, eps);
",
            "                let e = {
                    // The reference that best conditions the uncut lines.
                    let cp = &self.topo.bubble_complements[b * (n - 2)..(b + 1) * (n - 2)];
                    bubble_basis_x2(&k, v, c1, cp, eps)
                };
",
        )],
    ),
    (
        "    fn tadpoles<N: FastNumerator + ?Sized>(
        &mut self,
        input: &FastInput,
        num: &mut N,
    ) -> Result<()> {
",
        "    /// [`FastAmplitude::tadpoles`] with the rank n+2 engine's tadpole bases.
    fn tadpoles_conditioned<N: FastNumerator + ?Sized>(
        &mut self,
        input: &FastInput,
        num: &mut N,
        bases: &TadpoleBasesX2,
    ) -> Result<()> {
",
        &[
            (
                "                    .unwrap_or(2);
                self.tad_rows.push(TadpoleRow { lane, basis });
",
                "                    .unwrap_or(2);
                // The basis that best conditions the uncut lines.
                let basis = {
                    let uncut = &self.topo.tadpole_complements[a * (n - 1)..(a + 1) * (n - 1)];
                    bases.choose(basis, v, a, uncut)
                };
                self.tad_rows.push(TadpoleRow { lane, basis });
",
            ),
            (
                "vectors(&self.tadpole_bases[row.basis], orientation)",
                "vectors(&bases.bases[row.basis], orientation)",
            ),
            (
                "let e = self.tadpole_bases[row.basis];",
                "let e = bases.bases[row.basis];",
            ),
        ],
    ),
];

/// The text of the function of `engine` whose definition starts with
/// `start`, up to its closing brace (rustfmt's layout: the only line `    }`
/// inside an `impl` after the start).
fn function<'a>(engine: &'a str, start: &str) -> &'a str {
    assert_eq!(
        engine.matches(start).count(),
        1,
        "engine.rs: the definition {start:?} does not occur exactly once"
    );
    let begin = engine.find(start).unwrap();
    let end = begin + engine[begin..].find("\n    }\n").unwrap() + "\n    }\n".len();
    &engine[begin..end]
}

/// `engine/conditioned.rs` for this `engine.rs`.
fn generate(engine: &str) -> String {
    let mut out = HEADER.to_string();
    for (k, (start, signature, edits)) in FUNCTIONS.iter().enumerate() {
        let mut text = function(engine, start).replacen(start, signature, 1);
        for (anchor, replacement) in edits.iter() {
            assert_eq!(
                text.matches(anchor).count(),
                1,
                "engine.rs: the anchor {anchor:?} does not occur exactly once in {start:?}"
            );
            text = text.replacen(anchor, replacement, 1);
        }
        if k > 0 {
            out.push('\n');
        }
        out.push_str(&text);
    }
    out.push_str("}\n");
    out
}

#[test]
fn conditioned_stages_are_generated_from_the_default_ones() {
    let expected = generate(ENGINE);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(GENERATED);
    if std::env::var_os("NINJARS_BLESS").is_some() {
        std::fs::write(&path, &expected).unwrap();
    }
    let actual = std::fs::read_to_string(&path).unwrap();
    assert!(
        actual == expected,
        "{GENERATED} is stale; regenerate it with NINJARS_BLESS=1 cargo test -p ninjars --lib conditioned_stages"
    );
}
