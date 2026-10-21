//! Conditioned bubble references and tadpole bases of the rank n+2 engine
//! (and of the rank <= n+1 engine with
//! [`super::FastOptions::conditioned_cuts`]).
//!
//! Original Ninja fixes the bubble reference vector (`Basis(k)`: the
//! sign-adapted `r = (±1, ∓1/√3, ∓1/√3, ∓1/√3)`, which keeps `k.r` large) and
//! takes the first of its three tadpole bases whose reference vectors are not
//! exactly orthogonal to an uncut line (`check_ref_vecs`). Neither choice
//! looks at how close to orthogonal the uncut lines are. Every uncut line
//! `v = V_j - V_cut` of a cut enters the Laurent division, and the parent
//! subtraction terms, through `1/(e.v)` with `e` the growing expansion vector
//! (bubble `e3`, and `e4` in the swapped orientation; tadpole `e1` and `e3`):
//! when `v` is nearly orthogonal to `e`, the residue coefficients grow like
//! powers of `|e||v|/|e.v|` and cancel against the other cuts and the
//! integrated terms. At rank n+1 these powers reach 4; at rank n+2 the
//! expansions are one order deeper and the subtraction terms divide by up to
//! `(e.v)^5`, so the same geometry costs several more digits (up to 1e-4
//! relative, and O(1) on rare points, where the generic engine in binary64
//! gives 1e-14).
//!
//! The rank n+2 engine therefore scores candidates per cut and lane against
//! the uncut lines and takes the best one; Ninja's choice wins ties and is
//! kept without uncut lines. Selection is per lane, so batches stay bitwise
//! identical to single lanes. The tadpole basis only changes the rounding of
//! the result; the bubble reference also moves terms between the
//! cut-constructible and the rational part (only their sum is reference
//! independent, given true master integrals), as the generic engine's own
//! reference choice does. A rank <= n+1 numerator run through the rank n+2
//! engine (`AtDeclaredRank`) therefore gives the rank <= n+1 engine's Laurent
//! coefficients, but not its split.
//!
//! - **Bubbles** (`k = V_first - V_second`, real null reference `r`):
//!   `score = |k.r|/(|k||r⁰|) · min_v |e3.v|/(|e3||v|)` (Euclidean norms of
//!   the components), evaluated without spinors from the Gram determinant
//!   `G(k, r, v) = 2 (k.r)(e3.v)(e4.v)` and `|e3|² = 2 |e1⁰ r⁰|`
//!   (`e1 = k - r1 r`); for real kinematics `|e4.v| = |e3.v|`, so both
//!   orientations are covered. Candidates: Ninja's reference and the 24
//!   signed permutations of `(3/4, 1/2, 1/2, 1/4)` (exactly null in binary64).
//!   Since 5ed3b22 the generic engine scores the same 24 (scaled by 4/3; the
//!   score is scale invariant) with this score at every rank class
//!   (`ninjars-symbolica` `cuts.rs::bubble_reference`).
//! - **Tadpoles**: `score = min_v min(|e1.v|/|e1|, |e3.v|/|e3|)/|v|` over
//!   original Ninja's three bases and six exactly null pairs built like the
//!   generic engine's rational Weyl spinors (`tadpole_plans.rs`), whose six
//!   the generic engine scores the same way (`tadpole_line_scores`).
//!
//! The scores were chosen on forced experiments (every candidate for one cut
//! at a time, against Float(256) results) over the rank n+2 suite and random
//! n = 3..7 kinematics: the largest error increase over the best candidate
//! of a cut fell from 4e7 (Ninja's choices) to 2.4e2 (bubbles) and 40
//! (tadpoles). On the rank n+2 suite and 1260 random lanes (n = 1..7, against
//! Float(256)), keeping Ninja's choice unless a candidate scores 1.3 or 2
//! times better left 25% and 120% more lanes outside 10 times the generic
//! binary64 error than taking the best candidate, and adding the eight cube
//! diagonals `(1, ±1/√3, ±1/√3, ±1/√3)` as candidates gained nothing.
//!
//! The spinor code is a copy of `geometry.rs` (`Spinor`, `Basis::complete`),
//! kept separate so that new call sites cannot change the inlining, and
//! hence the machine code, of the rank <= n+1 path. The scoring keeps its
//! per-candidate minima on the stack: no vector routine is instantiated here
//! (see `engine_x2.rs` `Push1`).

// Index loops mirror original Ninja's component loops.
#![allow(clippy::needless_range_loop)]

use super::complex::{C64, CMom, RMom, mp_rr};
use super::geometry::{Basis, mp2, scale, sub, tadpole_candidates};

/// `x` with the sign of `s` (Ninja's `sign`).
#[inline(always)]
fn sign(x: f64, s: f64) -> f64 {
    if s < 0. { -x } else { x }
}

#[inline(always)]
fn signed_root(x: f64) -> C64 {
    if x < 0. {
        C64::new(0., (-x).sqrt())
    } else {
        C64::real(x.sqrt())
    }
}

/// Massless spinor of a real momentum (`geometry.rs` `Spinor`).
#[derive(Clone, Copy, Debug)]
struct SpinorX2 {
    ap: C64,
    ame: C64,
    bp: C64,
    bme: C64,
}

impl SpinorX2 {
    fn new(p: &RMom, eps: f64) -> Self {
        let zero = C64::ZERO;
        if (p[0] - p[3]).abs() < eps {
            let v = signed_root(2. * p[0]);
            Self {
                ap: v,
                ame: zero,
                bp: v,
                bme: zero,
            }
        } else if (p[0] + p[3]).abs() < eps {
            let v = signed_root(2. * p[0]);
            Self {
                ap: zero,
                ame: v,
                bp: zero,
                bme: v,
            }
        } else {
            let plus = p[0] + p[3];
            if plus < 0. {
                let s = (-plus).sqrt();
                let inv = 1. / s;
                Self {
                    ap: C64::new(0., s),
                    ame: C64::new(p[2] * inv, -p[1] * inv),
                    bp: C64::new(0., s),
                    bme: C64::new(-p[2] * inv, -p[1] * inv),
                }
            } else {
                let s = plus.sqrt();
                let inv = 1. / s;
                Self {
                    ap: C64::real(s),
                    ame: C64::new(p[1] * inv, p[2] * inv),
                    bp: C64::real(s),
                    bme: C64::new(p[1] * inv, -p[2] * inv),
                }
            }
        }
    }
}

/// `0.5 <i|gamma^mu|j]`.
#[inline(always)]
fn momentum_from_spinors(i: &SpinorX2, j: &SpinorX2) -> CMom {
    let x = i.ap * j.bme - i.ame * j.bp;
    [
        (i.ame * j.bme + i.ap * j.bp) * 0.5,
        (i.ap * j.bme + i.ame * j.bp) * 0.5,
        C64::new(-x.im, x.re) * 0.5,
        (i.ap * j.bp - i.ame * j.bme) * 0.5,
    ]
}

/// `Basis::complete`: `e3 = <1|γ|2]/2` and `e4 = <2|γ|1]/2` of the real
/// massless `e1`, `e2`.
fn complete(e1: RMom, e2: RMom, r1: f64, r2: f64, mp12: f64, eps: f64) -> Basis {
    let sp1 = SpinorX2::new(&e1, eps);
    let sp2 = SpinorX2::new(&e2, eps);
    Basis {
        e1,
        e2,
        e3: momentum_from_spinors(&sp1, &sp2),
        e4: momentum_from_spinors(&sp2, &sp1),
        r1,
        r2,
        mp12,
    }
}

/// Ninja's `Basis(k)` with the massless reference `e2 = r` (`k.r != 0`):
/// `k = e1 + r1 e2`. With Ninja's reference it is `Basis::from_single`
/// operation for operation.
fn bubble_basis(k: &RMom, r: &RMom, eps: f64) -> Basis {
    let kq = mp2(k);
    let k1k2 = mp_rr(k, r);
    let mut e1 = *k;
    let mut r1 = 0.;
    if kq.abs() >= eps {
        r1 = 0.5 * kq / k1k2;
        e1 = sub(&e1, &scale(r, r1));
    }
    complete(e1, *r, r1, 0., k1k2, eps)
}

/// Original Ninja's bubble reference for `k` (`Basis(k)`).
#[inline(always)]
fn ninja_reference(k: &RMom) -> RMom {
    const INVSQRT3: f64 = 0.577_350_269_189_625_8; // 1./sqrt(3.)
    [
        sign(1., k[0]),
        -sign(INVSQRT3, k[1]),
        -sign(INVSQRT3, k[2]),
        -sign(INVSQRT3, k[3]),
    ]
}

/// Number of candidate bubble references besides Ninja's.
const BUBBLE_CANDIDATES: usize = 24;

/// The 24 signed permutations of `(3/4, 1/2, 1/2, 1/4)`: exactly null, with
/// `r⁰ > 0`.
const fn bubble_candidates() -> [RMom; BUBBLE_CANDIDATES] {
    let mut out = [[0.; 4]; BUBBLE_CANDIDATES];
    let mut c = 0;
    let mut small = 1;
    while small < 4 {
        let mut s = 0;
        while s < 8 {
            let mut r = [0.75, 0., 0., 0.];
            let mut i = 1;
            while i < 4 {
                let magnitude = if i == small { 0.25 } else { 0.5 };
                r[i] = if (s >> (i - 1)) & 1 == 1 {
                    -magnitude
                } else {
                    magnitude
                };
                i += 1;
            }
            out[c] = r;
            c += 1;
            s += 1;
        }
        small += 1;
    }
    out
}

const BUBBLE_REFERENCES: [RMom; BUBBLE_CANDIDATES] = bubble_candidates();

/// [`BUBBLE_REFERENCES`] by component (structure of arrays: the scoring
/// loops over the candidates vectorize).
const BUBBLE_COMPONENTS: [[f64; BUBBLE_CANDIDATES]; 4] = {
    let mut out = [[0.; BUBBLE_CANDIDATES]; 4];
    let mut i = 0;
    while i < BUBBLE_CANDIDATES {
        let mut mu = 0;
        while mu < 4 {
            out[mu][i] = BUBBLE_REFERENCES[i][mu];
            mu += 1;
        }
        i += 1;
    }
    out
};

/// Euclidean norm squared of the components.
#[inline(always)]
fn norm2(v: &RMom) -> f64 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
}

/// The square of the bubble score of a reference with time component `r0`
/// and `kr = k.r` (module docs), given `plane = min_v |G(k, r, v)|/|v|²`:
/// `(k.r)²/(|k|² r⁰²) · min_v |e3.v|²/(|e3|² |v|²)` with
/// `|e3.v|² = |G(k, r, v)|/(2 |k.r|)` and `|e3|² = 2 |e1⁰ r⁰|`. 0 when it
/// is not finite (`k.r = 0`).
#[inline(always)]
fn bubble_score2(k: &RMom, kk: f64, knorm2: f64, r0: f64, kr: f64, plane: f64, eps: f64) -> f64 {
    let r1 = if kk.abs() >= eps { 0.5 * kk / kr } else { 0. };
    let e1r = ((k[0] - r1 * r0) * r0).abs();
    // (k.r)²/(|k|² r⁰²) / (2 |k.r| 2 |e1⁰ r⁰|) multiplies the plane term.
    let score = kr.abs() / (4. * (knorm2 * r0 * r0) * e1r) * plane;
    if score.is_finite() { score } else { 0. }
}

/// The bubble basis of the cut momentum `k = V_first - V_second` given the
/// uncut lines `V_j - V_first` (`j` in `uncut`): `Basis(k)` with the best
/// scoring reference, original Ninja's on ties.
pub(super) fn bubble_basis_x2(
    k: &RMom,
    shifts: &[RMom],
    first: usize,
    uncut: &[usize],
    eps: f64,
) -> Basis {
    let ninja = ninja_reference(k);
    #[cfg(test)]
    if NINJA_CHOICES.get() {
        return bubble_basis(k, &ninja, eps);
    }
    // Without uncut lines the reference enters only through k.r, and Ninja's
    // sign-adapted reference keeps k.r >= |k⁰| + |k|/√3 (a candidate gains at
    // most a factor 2/√3): nothing to condition.
    if uncut.is_empty() {
        return bubble_basis(k, &ninja, eps);
    }
    const C: usize = BUBBLE_CANDIDATES;
    let [r0, r1, r2, r3] = &BUBBLE_COMPONENTS;
    let kk = mp2(k);
    let knorm2 = norm2(k);
    let kr_ninja = mp_rr(k, &ninja);
    let kr: [f64; C] =
        std::array::from_fn(|i| k[0] * r0[i] - k[1] * r1[i] - k[2] * r2[i] - k[3] * r3[i]);
    let (mut plane_ninja, mut plane) = (f64::INFINITY, [f64::INFINITY; C]);
    for &j in uncut {
        let v = sub(&shifts[j], &shifts[first]);
        let (kv, vv, vnorm2) = (mp_rr(k, &v), mp2(&v), norm2(&v));
        // det [[k.k, k.r, k.v], [k.r, 0, r.v], [k.v, r.v, v.v]] / |v|²
        let ratio =
            |kr: f64, rv: f64| (-kk * rv * rv - kr * kr * vv + 2. * kr * rv * kv).abs() / vnorm2;
        let x = ratio(kr_ninja, mp_rr(&ninja, &v));
        plane_ninja = if x < plane_ninja { x } else { plane_ninja };
        for i in 0..C {
            let rv = r0[i] * v[0] - r1[i] * v[1] - r2[i] * v[2] - r3[i] * v[3];
            let x = ratio(kr[i], rv);
            plane[i] = if x < plane[i] { x } else { plane[i] };
        }
    }
    let mut best = bubble_score2(k, kk, knorm2, ninja[0], kr_ninja, plane_ninja, eps);
    let mut choice = None;
    for i in 0..C {
        let s = bubble_score2(k, kk, knorm2, r0[i], kr[i], plane[i], eps);
        if s > best {
            best = s;
            choice = Some(i);
        }
    }
    let Some(i) = choice else {
        return bubble_basis(k, &ninja, eps);
    };
    // Ninja's convention k.r > 0.
    let mut r = BUBBLE_REFERENCES[i];
    if kr[i] < 0. {
        r = r.map(|x| -x);
    }
    bubble_basis(k, &r, eps)
}

#[cfg(test)]
thread_local! {
    /// Unit tests: force original Ninja's bubble references and tadpole
    /// bases, i.e. the rank <= n+1 engine's conventions (and its
    /// cut-constructible/rational split).
    static NINJA_CHOICES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Unit tests: run `f` with original Ninja's bubble references and tadpole
/// bases on this thread.
#[cfg(test)]
pub(super) fn with_ninja_choices<R>(f: impl FnOnce() -> R) -> R {
    NINJA_CHOICES.set(true);
    let out = f();
    NINJA_CHOICES.set(false);
    out
}

/// Number of tadpole bases: original Ninja's three, then six more.
pub(super) const TADPOLE_BASES: usize = 9;

/// Candidate tadpole bases, and their `e1` and `e3` divided by their
/// Euclidean norms by component (structure of arrays, so the scoring loops
/// vectorize).
#[derive(Clone, Copy, Debug)]
pub(super) struct TadpoleBasesX2 {
    pub(super) bases: [Basis; TADPOLE_BASES],
    e1: [[f64; TADPOLE_BASES]; 4],
    e3_re: [[f64; TADPOLE_BASES]; 4],
    e3_im: [[f64; TADPOLE_BASES]; 4],
}

impl TadpoleBasesX2 {
    pub(super) fn new(eps: f64) -> Self {
        // Null pairs from the rational Weyl spinors |i> = (1, a_i + i b_i) of
        // the generic engine: e = (1 + |z|², 2a, 2b, 1 - |z|²)/2, z = a + i b
        // (exact in binary64).
        const PAIRS: [(f64, f64, f64, f64); 6] = [
            (1., 2., -2., 1.),
            (2., -1., 1., 3.),
            (-1., 3., 2., 1.),
            (3., 1., -1., -2.),
            (1., -3., 3., -2.),
            (2., 3., -3., 1.),
        ];
        let null = |a: f64, b: f64| -> RMom {
            let z2 = a * a + b * b;
            [0.5 * (1. + z2), a, b, 0.5 * (1. - z2)]
        };
        let ninja = tadpole_candidates(eps);
        let bases: [Basis; TADPOLE_BASES] = std::array::from_fn(|i| {
            if i < 3 {
                ninja[i]
            } else {
                let (a, b, c, d) = PAIRS[i - 3];
                let (e1, e2) = (null(a, b), null(c, d));
                complete(e1, e2, 0., 0., mp_rr(&e1, &e2), eps)
            }
        });
        let inv1 = bases.map(|e| 1. / norm2(&e.e1).sqrt());
        let inv3 = bases.map(|e| {
            let n: f64 = e.e3.iter().map(|z| z.norm_sqr()).sum();
            1. / n.sqrt()
        });
        Self {
            bases,
            e1: std::array::from_fn(|mu| std::array::from_fn(|i| bases[i].e1[mu] * inv1[i])),
            e3_re: std::array::from_fn(|mu| std::array::from_fn(|i| bases[i].e3[mu].re * inv3[i])),
            e3_im: std::array::from_fn(|mu| std::array::from_fn(|i| bases[i].e3[mu].im * inv3[i])),
        }
    }

    /// The squared score of every candidate,
    /// `min_v min(|e1.v|²/|e1|², |e3.v|²/|e3|²)/|v|²`.
    #[inline(always)]
    fn scores2(&self, shifts: &[RMom], cut: usize, uncut: &[usize]) -> [f64; TADPOLE_BASES] {
        const T: usize = TADPOLE_BASES;
        let (e1, re, im) = (&self.e1, &self.e3_re, &self.e3_im);
        let mut score = [f64::INFINITY; T];
        for &j in uncut {
            let v = sub(&shifts[j], &shifts[cut]);
            let inv = 1. / norm2(&v);
            for i in 0..T {
                let a = e1[0][i] * v[0] - e1[1][i] * v[1] - e1[2][i] * v[2] - e1[3][i] * v[3];
                let b = re[0][i] * v[0] - re[1][i] * v[1] - re[2][i] * v[2] - re[3][i] * v[3];
                let c = im[0][i] * v[0] - im[1][i] * v[1] - im[2][i] * v[2] - im[3][i] * v[3];
                let (a, b) = (a * a, b * b + c * c);
                let s = if a < b { a } else { b } * inv;
                score[i] = if s < score[i] { s } else { score[i] };
            }
        }
        score
    }

    /// The basis of the tadpole `cut` with the uncut lines `V_j - V_cut` (`j`
    /// in `uncut`), given original Ninja's choice `ninja`: the candidate with
    /// the best score `min_v min(|e1.v|/|e1|, |e3.v|/|e3|)/|v|`, Ninja's on
    /// ties.
    pub(super) fn choose(
        &self,
        ninja: usize,
        shifts: &[RMom],
        cut: usize,
        uncut: &[usize],
    ) -> usize {
        #[cfg(test)]
        if NINJA_CHOICES.get() {
            return ninja;
        }
        if uncut.is_empty() {
            return ninja;
        }
        let score = self.scores2(shifts, cut, uncut);
        let finite = |s: f64| if s.is_finite() { s } else { 0. };
        let mut best = finite(score[ninja]);
        let mut choice = ninja;
        for i in 0..TADPOLE_BASES {
            let s = finite(score[i]);
            if s > best {
                best = s;
                choice = i;
            }
        }
        choice
    }
}

#[cfg(test)]
mod tests {
    use super::super::complex::{mp_cc, mp_cr};
    use super::*;

    /// The spinor-free bubble score agrees with the spinor basis
    /// (`|e3.v|` and `|e3|` from `e3` itself), for every candidate.
    #[test]
    fn bubble_score_matches_the_spinor_basis() {
        let eps = 1e-9;
        for (k, vs) in [
            (
                [0.3, -1.7, 0.4, 2.1],
                vec![[1.1, 0.2, -0.7, 0.5], [-0.4, 2.0, 1.3, -0.9]],
            ),
            (
                [-2.5, 0.3, 0.1, -0.2],
                vec![
                    [0.2, 1.5, -0.3, 0.8],
                    [3.0, 0.1, 0.2, 0.3],
                    [0.5, -0.5, 0.5, -0.5],
                ],
            ),
        ] {
            let kk = mp2(&k);
            let knorm2 = norm2(&k);
            for r in BUBBLE_REFERENCES.iter().chain([ninja_reference(&k)].iter()) {
                let r = if mp_rr(&k, r) < 0. { r.map(|x| -x) } else { *r };
                let e = bubble_basis(&k, &r, eps);
                let e3n2: f64 = e.e3.iter().map(|z| z.norm_sqr()).sum();
                let (mut want, mut plane) = (f64::INFINITY, f64::INFINITY);
                for v in &vs {
                    let a = mp_cr(&e.e3, v).abs();
                    want = want.min(a * a / (e3n2 * norm2(v)));
                    // G(k, r, v) from the Gram matrix of (k, r, v).
                    let (kr, kv, rv, vv) = (mp_rr(&k, &r), mp_rr(&k, v), mp_rr(&r, v), mp2(v));
                    let gram = -kk * rv * rv - kr * kr * vv + 2. * kr * rv * kv;
                    plane = plane.min(gram.abs() / norm2(v));
                }
                want *= e.mp12 * e.mp12 / (knorm2 * r[0] * r[0]);
                let got = bubble_score2(&k, kk, knorm2, r[0], mp_rr(&k, &r), plane, eps);
                assert!(
                    (got - want).abs() <= 1e-12 * want,
                    "k={k:?} r={r:?}: score² {got:e} against the spinor basis {want:e}"
                );
            }
        }
    }

    /// The scores do not depend on the normalization of the reference.
    #[test]
    fn bubble_score_is_scale_invariant() {
        let eps = 1e-9;
        let k = [0.3, -1.7, 0.4, 2.1];
        let v = [1.1, 0.2, -0.7, 0.5];
        let (kk, knorm2) = (mp2(&k), norm2(&k));
        for r in BUBBLE_REFERENCES {
            let mut scores = [0.; 2];
            for (s, factor) in scores.iter_mut().zip([1., 4. / 3.]) {
                let r = r.map(|x| x * factor);
                let (kr, rv) = (mp_rr(&k, &r), mp_rr(&r, &v));
                let gram = -kk * rv * rv - kr * kr * mp2(&v) + 2. * kr * rv * mp_rr(&k, &v);
                *s = bubble_score2(&k, kk, knorm2, r[0], kr, gram.abs() / norm2(&v), eps);
            }
            assert!(
                (scores[0] - scores[1]).abs() <= 1e-13 * scores[0],
                "{r:?}: {scores:?}"
            );
        }
    }

    /// Every candidate basis is a null tetrad with `e3.e4 = -e1.e2`, and the
    /// new candidates are exactly null.
    #[test]
    fn candidate_bases_are_null_tetrads() {
        let eps = 1e-9;
        let k = [0.3, -1.7, 0.4, 2.1];
        let tadpoles = TadpoleBasesX2::new(eps);
        let bubbles = BUBBLE_REFERENCES.map(|r| {
            let r = if mp_rr(&k, &r) < 0. { r.map(|x| -x) } else { r };
            bubble_basis(&k, &r, eps)
        });
        for r in &BUBBLE_REFERENCES {
            assert_eq!(mp2(r), 0., "{r:?}");
        }
        for e in &tadpoles.bases[3..] {
            assert_eq!((mp2(&e.e1), mp2(&e.e2)), (0., 0.), "{e:?}");
        }
        for e in tadpoles.bases.iter().chain(bubbles.iter()) {
            let scale = 1. + e.mp12.abs();
            assert!(mp2(&e.e1).abs() < 1e-12 * scale, "{e:?}");
            assert!(mp2(&e.e2).abs() < 1e-12 * scale, "{e:?}");
            assert!(mp_cc(&e.e3, &e.e3).abs() < 1e-12 * scale, "{e:?}");
            assert!(mp_cc(&e.e4, &e.e4).abs() < 1e-12 * scale, "{e:?}");
            assert!((mp_cc(&e.e3, &e.e4) + C64::real(e.mp12)).abs() < 1e-12 * scale);
            assert!((mp_rr(&e.e1, &e.e2) - e.mp12).abs() < 1e-12 * scale);
        }
    }

    /// With Ninja's reference the basis is `Basis::from_single` bitwise.
    #[test]
    fn ninja_reference_reproduces_basis_from_single() {
        let eps = 1e-9;
        for k in [
            [0.3, -1.7, 0.4, 2.1],
            [-2.5, 0.3, 0.1, -0.2],
            [1., 0., 0., 1.],
        ] {
            let a = bubble_basis(&k, &ninja_reference(&k), eps);
            let b = Basis::from_single(&k, eps);
            assert_eq!(format!("{a:?}"), format!("{b:?}"));
            // Without uncut lines Ninja's sign-adapted reference is kept.
            let kept = bubble_basis_x2(&k, &[k], 0, &[], eps);
            assert_eq!(format!("{kept:?}"), format!("{b:?}"));
        }
    }

    /// The best scoring bubble reference and tadpole basis are chosen,
    /// Ninja's on ties; an uncut line in the plane of k and Ninja's bubble
    /// reference, or orthogonal to Ninja's tadpole `e1`, moves away from it.
    #[test]
    fn best_scoring_choices_win() {
        let eps = 1e-9;
        let k = [0.3, -1.7, 0.4, 2.1];
        let (kk, knorm2) = (mp2(&k), norm2(&k));
        let ninja = ninja_reference(&k);
        let ninja_basis = bubble_basis(&k, &ninja, eps);
        // v = k + 0.5 r lies in the plane of k and Ninja's reference.
        let v = [0, 1, 2, 3].map(|i| k[i] + 0.5 * ninja[i]);
        let replaced = bubble_basis_x2(&k, &[[0.; 4], v], 0, &[1], eps);
        assert_ne!(replaced.e2, ninja_basis.e2);
        assert!(mp_cr(&replaced.e3, &v).abs() > 1e-3 * norm2(&v).sqrt());
        let lines = [
            [1.1, 0.2, -0.7, 0.5],
            [-0.4, 2.0, 1.3, -0.9],
            [0.2, 1.5, -0.3, 0.8],
            [3.0, 0.1, 0.2, 0.3],
            ninja_basis.e3.map(|z| z.re),
        ];
        let mut kept = 0;
        for uncut in [&[1][..], &[2], &[3, 4], &[1, 2, 3], &[5]] {
            let shifts: Vec<RMom> = [[0.; 4]].into_iter().chain(lines).collect();
            let score = |r: &RMom| {
                let kr = mp_rr(&k, r);
                let plane = uncut
                    .iter()
                    .map(|&j| {
                        let v = &shifts[j];
                        let rv = mp_rr(r, v);
                        let gram = -kk * rv * rv - kr * kr * mp2(v) + 2. * kr * rv * mp_rr(&k, v);
                        gram.abs() / norm2(v)
                    })
                    .fold(f64::INFINITY, f64::min);
                bubble_score2(&k, kk, knorm2, r[0], kr, plane, eps)
            };
            let (mut best, mut want) = (score(&ninja), ninja);
            for r in &BUBBLE_REFERENCES {
                if score(r) > best {
                    best = score(r);
                    want = if mp_rr(&k, r) < 0. { r.map(|x| -x) } else { *r };
                }
            }
            kept += usize::from(want == ninja);
            let got = bubble_basis_x2(&k, &shifts, 0, uncut, eps);
            assert_eq!(got.e2, want, "uncut {uncut:?}");
        }
        assert!(kept < 5);
        // Tadpoles, with the score from the bases themselves.
        let tadpoles = TadpoleBasesX2::new(eps);
        let score = |i: usize, v: &RMom| {
            let e = &tadpoles.bases[i];
            let e3n2: f64 = e.e3.iter().map(|z| z.norm_sqr()).sum();
            let a = mp_rr(&e.e1, v).powi(2) / norm2(&e.e1);
            let b = mp_cr(&e.e3, v).norm_sqr() / e3n2;
            a.min(b) / norm2(v)
        };
        let e1 = tadpoles.bases[0].e1;
        let w = [e1[1], e1[0], 0., 0.]; // e1.w = e1⁰ e1¹ - e1¹ e1⁰ = 0
        assert_eq!(score(0, &w), 0.);
        assert_ne!(tadpoles.choose(0, &[[0.; 4], w], 0, &[1]), 0);
        for ninja in 0..3 {
            for w in lines.iter().chain([[0.5, -0.5, 0.5, -0.5], w].iter()) {
                let shifts = [[0.; 4], *w];
                let scores2 = tadpoles.scores2(&shifts, 0, &[1]);
                for i in 0..TADPOLE_BASES {
                    let want = score(i, w);
                    assert!(
                        (scores2[i] - want).abs() <= 1e-12 * want + 1e-15,
                        "{w:?} {i}"
                    );
                }
                let mut want = ninja;
                for i in 0..TADPOLE_BASES {
                    if scores2[i] > scores2[want] {
                        want = i;
                    }
                }
                assert_eq!(tadpoles.choose(ninja, &shifts, 0, &[1]), want, "{w:?}");
            }
        }
    }
}
