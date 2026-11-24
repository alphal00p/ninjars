//! The fast engine's triangle, bubble and tadpole stages in double-double
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

// ---- geometry.rs

// Index loops mirror original Ninja's component and lane loops.


/// `x` with the sign of `s` (Ninja's `sign`).
#[inline(always)]
fn sign(x: impl Into<Dd>, s: Dd) -> Dd {
    let x = x.into();
    if s < 0. { -x } else { x }
}

#[inline(always)]
pub(crate) fn sub(a: &RMom, b: &RMom) -> RMom {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

#[inline(always)]
pub(crate) fn add(a: &RMom, b: &RMom) -> RMom {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

#[inline(always)]
pub(crate) fn scale(a: &RMom, s: Dd) -> RMom {
    [a[0] * s, a[1] * s, a[2] * s, a[3] * s]
}

#[inline(always)]
pub(crate) fn mp2(a: &RMom) -> Dd {
    mp_rr(a, a)
}

/// Gram determinant `k1² k2² - (k1.k2)²`.
#[inline(always)]
pub(crate) fn gram_det(k1: &RMom, k2: &RMom) -> Dd {
    let k12 = mp_rr(k1, k2);
    mp2(k1) * mp2(k2) - k12 * k12
}

/// Massless spinor of a real momentum (`Spinor(const RealMomentum&)`).
#[derive(Clone, Copy, Debug)]
struct Spinor {
    ap: C64,
    ame: C64,
    bp: C64,
    bme: C64,
}

#[inline(always)]
fn signed_root(x: Dd) -> C64 {
    // x < 0 ? I*sqrt(-x) : sqrt(x)
    if x < 0. {
        C64::new(0., (-x).sqrt())
    } else {
        C64::real(x.sqrt())
    }
}

impl Spinor {
    /// The spinor of a light-like `p`, formed from its larger light-cone
    /// component, with no threshold (`p± = p⁰ ± p³`, `p⊥ = p¹ + i p²`,
    /// `p⊥ p⊥* = p⁺ p⁻`; signed roots, `√x = i √-x` for `x < 0`):
    /// - on the `z` axis (`p⊥ = 0`), Ninja's axis spinors: `ap = bp = √(2p⁰)`
    ///   if `|p⁺| ≥ |p⁻|`, otherwise `ame = bme = √(2p⁰)`, the others zero;
    /// - for `|p⁺| ≥ |p⁻|`, Ninja's form: `ap = bp = √p⁺`, `ame = p⊥/√p⁺`,
    ///   `bme = p⊥*/√p⁺`;
    /// - otherwise the same spinor up to the phase `√p⁺ √p⁻ / p⊥`:
    ///   `ame = bme = √p⁻`, `ap = p⊥*/√p⁻`, `bp = p⊥/√p⁻`.
    ///
    /// The component taken is never formed by cancellation. Ninja took the
    /// axis spinors whenever `|p⁰ ∓ p³| < INFRARED_EPS` (absolute, in the
    /// units of the inputs), which drops a transverse momentum of up to
    /// `√(2 |p⁰| INFRARED_EPS)`, and `√p⁺` everywhere else, which near the
    /// `-z` axis divides by a cancelling `p⁺`: for a beam along `-z` reached
    /// as a difference of loop shifts, a rounding residue that is above
    /// `INFRARED_EPS` once the energies are large in those units. Ninja's
    /// bits are kept for `|p⁺| ≥ |p⁻|` outside those windows and on the `z`
    /// axis inside them (except for `|p⁰| < INFRARED_EPS / 2` along `-z`,
    /// where Ninja took the `+z` spinor).
    fn new(p: &RMom) -> Self {
        let zero = C64::ZERO;
        let (plus, minus) = (p[0] + p[3], p[0] - p[3]);
        // On the z axis, and for p⁰ = p³ = 0 (the zero spinor), Ninja's
        // axis spinors.
        if (p[1] == 0. && p[2] == 0.) || (plus == 0. && minus == 0.) {
            let v = signed_root(p[0] * 2.);
            if plus.abs() >= minus.abs() {
                Self {
                    ap: v,
                    ame: zero,
                    bp: v,
                    bme: zero,
                }
            } else {
                Self {
                    ap: zero,
                    ame: v,
                    bp: zero,
                    bme: v,
                }
            }
        } else if plus.abs() >= minus.abs() {
            // ap = sqrt(p+) is real or purely imaginary: divide by it with a
            // single real reciprocal.
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
        } else if minus < 0. {
            // The same with ame = sqrt(p-).
            let s = (-minus).sqrt();
            let inv = 1. / s;
            Self {
                ap: C64::new(-p[2] * inv, -p[1] * inv),
                ame: C64::new(0., s),
                bp: C64::new(p[2] * inv, -p[1] * inv),
                bme: C64::new(0., s),
            }
        } else {
            let s = minus.sqrt();
            let inv = 1. / s;
            Self {
                ap: C64::new(p[1] * inv, -p[2] * inv),
                ame: C64::real(s),
                bp: C64::new(p[1] * inv, p[2] * inv),
                bme: C64::real(s),
            }
        }
    }
}

/// `0.5 <i|gamma^mu|j]`.
#[inline(always)]
fn momentum_from_spinors(i: &Spinor, j: &Spinor) -> CMom {
    let x = i.ap * j.bme - i.ame * j.bp;
    [
        (i.ame * j.bme + i.ap * j.bp) * 0.5,
        (i.ap * j.bme + i.ame * j.bp) * 0.5,
        // HALF * I * x
        C64::new(-x.im, x.re) * 0.5,
        (i.ap * j.bp - i.ame * j.bme) * 0.5,
    ]
}

/// Massless basis `e1, e2` (real) and `e3, e4` (complex) with
/// `e3.e4 = -e1.e2` (Ninja's `Basis`).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Basis {
    pub e1: RMom,
    pub e2: RMom,
    pub e3: CMom,
    pub e4: CMom,
    pub r1: Dd,
    pub r2: Dd,
    pub mp12: Dd,
}

/// `|1 + r1 r2| = 2 |k1.k2 / gamma|` below which [`Basis::from_pair`] takes
/// `e1.e2 = gamma / 2` instead of Ninja's cancelling `k1.k2 / (1 + r1 r2)`
/// (more than ten bits lost).
const ORTHOGONAL: f64 = 1. / 1024.;

impl Basis {
    /// `Basis(k1, k2)`: `k1 = e1 + r1 e2`, `k2 = e2 + r2 e1`.
    pub(crate) fn from_pair(k1: &RMom, k2: &RMom, eps: f64) -> Self {
        let k1q = mp2(k1);
        let k2q = mp2(k2);
        let k1k2 = mp_rr(k1, k2);
        let k1n = k1q.abs() < eps;
        let k2n = k2q.abs() < eps;
        let mut e1 = *k1;
        let mut e2 = *k2;
        let (mut r1, mut r2) = (Dd::ZERO, Dd::ZERO);
        let mp12;
        if k1n && k2n {
            mp12 = k1k2;
        } else if k1n {
            r2 = 0.5 * k2q / k1k2;
            e2 = sub(&e2, &scale(k1, r2));
            mp12 = k1k2;
        } else if k2n {
            r1 = 0.5 * k1q / k1k2;
            e1 = sub(&e1, &scale(k2, r1));
            mp12 = k1k2;
        } else {
            // Reciprocals are hoisted: one division per denominator.
            let gamma = k1k2 + sign(1., k1k2) * (k1k2 * k1k2 - k1q * k2q).sqrt();
            let inv_gamma = 1. / gamma;
            r1 = k1q * inv_gamma;
            r2 = k2q * inv_gamma;
            let inv_den = 1. / (1. - r1 * r2);
            e1 = scale(&sub(&e1, &scale(k2, r1)), inv_den);
            e2 = scale(&sub(&e2, &scale(k1, r2)), inv_den);
            // 1 + r1 r2 = 2 k1.k2 / gamma, so Ninja's k1.k2 / (1 + r1 r2) is
            // gamma / 2. For nearly orthogonal momenta Ninja's form divides
            // rounding noise by rounding noise (a wrong basis normalization
            // and silently wrong cut solutions); use gamma / 2 there and keep
            // Ninja's arithmetic elsewhere.
            let one_plus = 1. + r1 * r2;
            mp12 = if one_plus.abs() < ORTHOGONAL {
                0.5 * gamma
            } else {
                k1k2 / one_plus
            };
        }
        Self::complete(e1, e2, r1, r2, mp12)
    }

    /// `Basis(k)`: `k = e1 + r1 e2` with a fixed massless reference `e2`.
    pub(crate) fn from_single(k: &RMom, eps: f64) -> Self {
        const INVSQRT3: f64 = 0.577_350_269_189_625_8; // 1./sqrt(3.)
        // Ninja's reference, light-like in double-double: its energy is
        // the double-double norm of the binary64 spatial part (binary64's
        // `1 - 3 (1/sqrt 3)²` is a rounding residue that the reduction of a
        // degenerate lane amplifies like any other).
        let third = Dd::from(INVSQRT3);
        let k2 = [
            sign((third * third * 3.).sqrt(), k[0]),
            -sign(third, k[1]),
            -sign(third, k[2]),
            -sign(third, k[3]),
        ];
        let kq = mp2(k);
        let k1k2 = mp_rr(k, &k2);
        let mut e1 = *k;
        let mut r1 = Dd::ZERO;
        if kq.abs() >= eps {
            r1 = 0.5 * kq / k1k2;
            e1 = sub(&e1, &scale(&k2, r1));
        }
        Self::complete(e1, k2, r1, Dd::ZERO, k1k2)
    }

    fn complete(e1: RMom, e2: RMom, r1: Dd, r2: Dd, mp12: Dd) -> Self {
        let sp1 = Spinor::new(&e1);
        let sp2 = Spinor::new(&e2);
        Self {
            e1,
            e2,
            e3: momentum_from_spinors(&sp1, &sp2),
            e4: momentum_from_spinors(&sp2, &sp1),
            r1,
            r2,
            mp12,
        }
    }

    #[inline(always)]
    pub(crate) fn mp34(&self) -> Dd {
        -self.mp12
    }
}

/// The three reference bases tried by Ninja's `tadpole_basis`, in order.
pub(crate) fn tadpole_candidates(eps: f64) -> [Basis; 3] {
    let sqrt2 = 2_f64.sqrt();
    let sqrt3 = 3_f64.sqrt();
    let invsqrt2 = 0.5 * sqrt2;
    let invsqrt3 = 1. / 3_f64.sqrt();
    let refs = [
        (
            [2., invsqrt3, -invsqrt3, invsqrt3],
            [sqrt3, invsqrt3, -invsqrt2, invsqrt2],
        ),
        (
            [2., 0.5 * (1. + invsqrt3), 0.5 * (1. - invsqrt3), invsqrt3],
            [
                sqrt3,
                0.5 * (sqrt3 * invsqrt2 + invsqrt3),
                0.5 * (1. - invsqrt2),
                invsqrt2,
            ],
        ),
        (
            [2., 0.5 * (1. - invsqrt3), -0.5 * (1. + invsqrt3), invsqrt3],
            [
                sqrt3,
                0.5 * (1. - invsqrt2),
                -0.5 * (sqrt3 * invsqrt2 + invsqrt3),
                invsqrt2,
            ],
        ),
    ];
    refs.map(|(r1, r2): ([f64; 4], [f64; 4])| Basis::from_pair(&r1.map(Dd::from), &r2.map(Dd::from), eps))
}

/// Ninja's `check_ref_vecs`: `false` if some `V_i - V_cut` is (nearly)
/// orthogonal to one of the real reference vectors.
pub(crate) fn reference_vectors_ok(
    shifts: &[RMom],
    cut: usize,
    e1: &RMom,
    e2: &RMom,
    eps: f64,
) -> bool {
    let v0 = shifts[cut];
    for (i, v) in shifts.iter().enumerate() {
        if i == cut {
            continue;
        }
        let d = sub(v, &v0);
        let norm = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3];
        let a = mp_rr(&d, e1);
        let b = mp_rr(&d, e2);
        if a * a / norm < eps || b * b / norm < eps {
            return false;
        }
    }
    true
}

#[inline(always)]
pub(crate) fn cmom_from_real(a: &RMom) -> CMom {
    [
        C64::real(a[0]),
        C64::real(a[1]),
        C64::real(a[2]),
        C64::real(a[3]),
    ]
}

/// `x1 e1 + x2 e2` for complex coefficients and real vectors.
#[inline(always)]
pub(crate) fn combine_real2(x1: C64, e1: &RMom, x2: C64, e2: &RMom) -> CMom {
    [
        x1 * e1[0] + x2 * e2[0],
        x1 * e1[1] + x2 * e2[1],
        x1 * e1[2] + x2 * e2[2],
        x1 * e1[3] + x2 * e2[3],
    ]
}

/// The two quadruple-cut solutions (`CutBox::getLoopMomentum`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn box_solutions(
    k1: &RMom,
    k2: &RMom,
    k3: &RMom,
    e: &Basis,
    m1q: C64,
    m2q: C64,
    m3q: C64,
    m4q: C64,
) -> [CMom; 2] {
    let mk11 = mp2(k1);
    let mk22 = mp2(k2);
    let beta = 1. / (1. - e.r1 * e.r2);
    let half12 = 0.5 / e.mp12;
    let a1 = (m1q - m4q + mk11) * half12;
    let a2 = (m2q - m1q - mk22) * half12;
    let x1 = (a2 - a1 * e.r2) * beta;
    let x2 = (a1 - a2 * e.r1) * beta;
    let k23 = add(k2, k3);
    let mk33 = mp2(&k23);
    let me31 = mp_rr(&k23, &e.e1);
    let me32 = mp_rr(&k23, &e.e2);
    let me33 = mp_cr(&e.e3, &k23);
    let me34 = mp_cr(&e.e4, &k23);
    let inv33 = me33.inv();
    let a3 = (m3q - m1q - mk33 - x1 * (2. * me31) - x2 * (2. * me32)) * inv33 * 0.5;
    let a4 = -(me34 * inv33);
    let b1 = a3 * (-2. * e.mp12);
    let b2 = a4 * (-2. * e.mp12);
    let b0 = x1 * x2 * (2. * e.mp12) - m1q;
    let rtdel = (b1 * b1 - b0 * b2 * 4.).sqrt();
    let half_b2 = b2.inv() * 0.5;
    let x41 = (-b1 + rtdel) * half_b2;
    let x42 = (-b1 - rtdel) * half_b2;
    let x31 = x41 * a4 + a3;
    let x32 = x42 * a4 + a3;
    let base = combine_real2(x1, &e.e1, x2, &e.e2);
    let mut out = [base, base];
    for mu in 0..4 {
        out[0][mu] = out[0][mu] + x31 * e.e3[mu] + x41 * e.e4[mu];
        out[1][mu] = out[1][mu] + x32 * e.e3[mu] + x42 * e.e4[mu];
    }
    out
}

/// Triple-cut parametrization `q = a + t e3 + (param + mu²)/t e4/(2 e3.e4)`
/// (`CutTriangle::getLoopMomentum`). Returns `(x1, x2, param)` with
/// `a = x1 e1 + x2 e2`.
pub(crate) fn triangle_solution(
    k1: &RMom,
    k2: &RMom,
    e: &Basis,
    m1q: C64,
    m2q: C64,
    m3q: C64,
) -> (C64, C64, C64) {
    let mk11 = mp2(k1);
    let mk22 = mp2(k2);
    let beta = 1. / (1. - e.r1 * e.r2);
    let half12 = 0.5 / e.mp12;
    let a1 = (m1q - m3q + mk11) * half12;
    let a2 = (m2q - m1q - mk22) * half12;
    let x1 = (a2 - a1 * e.r2) * beta;
    let x2 = (a1 - a2 * e.r1) * beta;
    let param = m1q - x1 * x2 * (2. * e.mp12);
    (x1, x2, param)
}

/// Double-cut parametrization (`CutBubble::getLoopMomentum`): returns
/// `(b0, b1, param)` with `a0 = b0 e2`, `a1 = e1 + b1 e2` (real).
pub(crate) fn bubble_solution(k: &RMom, e: &Basis, m1q: C64, m2q: C64) -> (C64, Dd, [C64; 3]) {
    let kq = mp2(k);
    let e12 = 2. * e.mp12;
    let e1k = 2. * mp_rr(&e.e1, k);
    let e2k = 2. * mp_rr(&e.e2, k);
    let inv_e2k = 1. / e2k;
    let b0 = (-m2q + m1q + kq) * inv_e2k;
    let b1 = -e1k * inv_e2k;
    (b0, b1, [m1q, -(b0 * e12), C64::real(-b1 * e12)])
}

// ---- subtractions.rs
mod subs {
#![allow(clippy::all, dead_code, unused_variables)]
use super::super::dd::{Cd as C64, CMomD as CMom, Dd, RMomD as RMom, mp_cc, mp_cr, mp_rr};



/// `cuts::correctbubcoeffs` for rank - n = -2, all coefficients.
#[inline(never)]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(super) fn bubble_from_triangle_rm2_all(
    divnum: &mut [C64],
    et3: &CMom,
    et4: &CMom,
    c: &[C64],
    eb3: &CMom,
    eb4: &CMom,
    param: &[C64; 3],
    eb1k: C64,
    eb2k: Dd,
    eb3k: C64,
    ieb3k: C64,
    eb4k: C64,
    eb1t: &[C64; 4],
    f: C64,
) {
    let z1_1 = (ieb3k * ieb3k) * ieb3k;
    let z2_1 = eb3k;
    let z3_1 = mp_cc(eb3, et3);
    let z5_1 = mp_cc(eb3, et4);
    let z19_1 = 0.5 * c[1];
    let z19_2 = z3_1 * z19_1;
    let z20_2 = 0.5 * c[4];
    let z20_3 = z5_1 * z20_2;
    let z15_2 = z20_3 + z19_2;
    let z15_3 = z2_1 * z15_2;
    let z15_4 = z15_3;
    let z15_5 = z2_1 * z15_4;
    let z15_6 = z15_5;
    let z15_7 = z1_1 * z15_6;
    divnum[0] -= z15_7;
}

/// `cuts::correctbubcoeffs` for rank - n = -1, all coefficients.
#[inline(never)]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(super) fn bubble_from_triangle_rm1_all(
    divnum: &mut [C64],
    et3: &CMom,
    et4: &CMom,
    c: &[C64],
    eb3: &CMom,
    eb4: &CMom,
    param: &[C64; 3],
    eb1k: C64,
    eb2k: Dd,
    eb3k: C64,
    ieb3k: C64,
    eb4k: C64,
    eb1t: &[C64; 4],
    f: C64,
) {
    let z1_1 = (ieb3k * ieb3k) * ieb3k;
    let z2_1 = eb3k;
    let z3_1 = mp_cc(eb3, et3);
    let z5_1 = mp_cc(eb3, et4);
    let z8_1 = eb1k;
    let z9_1 = eb1t[0];
    let z10_1 = eb1t[1];
    let z11_1 = eb2k;
    let z12_1 = eb1t[2];
    let z13_1 = eb1t[3];
    let z14_1 = z3_1 * z3_1;
    let z18_1 = z5_1 * z5_1;
    let z20_1 = 0.5 * z1_1;
    let z21_1 = z20_1 * z2_1;
    let z23_1 = 0.5 * f;
    let z26_1 = c[5];
    let z27_1 = z26_1 * z18_1;
    let z29_1 = c[2];
    let z30_1 = z14_1 * z29_1;
    let z27_2 = z27_1 + z30_1;
    let z30_2 = z2_1 * z27_2;
    let z30_3 = z30_2;
    let z30_4 = z30_3 * z21_1;
    let z32_1 = (-z11_1) * z27_2;
    let z18_2 = z23_1 + z8_1;
    let z23_2 = z32_1;
    let z29_2 = (z12_1 * z3_1) * z29_1;
    let z26_2 = (z5_1 * z13_1) * z26_1;
    let z26_3 = z26_2 + z29_2;
    let z26_4 = z2_1 * z26_3;
    let z23_3 = z26_4 + (0.5 * z23_2);
    let z23_4 = z2_1 * z23_3;
    let z23_5 = z23_4;
    let z23_6 = z1_1 * z23_5;
    let z18_3 = (-z27_2) * z18_2;
    let z18_4 = z18_3;
    let z19_1 = c[2];
    let z19_2 = z9_1 * z19_1;
    let z19_3 = z19_2 + (0.5 * c[1]);
    let z19_4 = z3_1 * z19_3;
    let z20_2 = c[5];
    let z20_3 = z10_1 * z20_2;
    let z20_4 = z20_3 + (0.5 * c[4]);
    let z20_5 = z5_1 * z20_4;
    let z15_2 = z20_5 + z19_4;
    let z15_3 = z2_1 * z15_2;
    let z15_4 = (0.5 * z18_4) + z15_3;
    let z15_5 = z2_1 * z15_4;
    let z15_6 = z15_5;
    let z15_7 = z1_1 * z15_6;
    divnum[0] -= z30_4;
    divnum[2] -= z23_6;
    divnum[1] -= z15_7;
}

/// `cuts::correctbubcoeffs` for rank - n = -1, second-orientation coefficients.
#[inline(never)]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(super) fn bubble_from_triangle_rm1_spurious(
    divnum: &mut [C64],
    et3: &CMom,
    et4: &CMom,
    c: &[C64],
    eb3: &CMom,
    eb4: &CMom,
    param: &[C64; 3],
    eb1k: C64,
    eb2k: Dd,
    eb3k: C64,
    ieb3k: C64,
    eb4k: C64,
    eb1t: &[C64; 4],
    f: C64,
) {
    let z1_1 = mp_cc(eb3, et3);
    let z2_1 = ieb3k;
    let z3_1 = mp_cc(eb3, et4);
    let z11_1 = 0.5 * z2_1;
    let z13_1 = z3_1 * z3_1;
    let z16_1 = z1_1 * z1_1;
    let z16_2 = c[2] * z16_1;
    let z13_2 = c[5] * z13_1;
    let z10_1 = z13_2 + z16_2;
    let z10_2 = z10_1 * z11_1;
    divnum[0] -= z10_2;
}

/// `cuts::correctbubcoeffs` for rank - n = 0, all coefficients.
#[inline(never)]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(super) fn bubble_from_triangle_r0_all(
    divnum: &mut [C64],
    et3: &CMom,
    et4: &CMom,
    c: &[C64],
    eb3: &CMom,
    eb4: &CMom,
    param: &[C64; 3],
    eb1k: C64,
    eb2k: Dd,
    eb3k: C64,
    ieb3k: C64,
    eb4k: C64,
    eb1t: &[C64; 4],
    f: C64,
) {
    let z1_1 = (ieb3k * ieb3k) * ieb3k;
    let z2_1 = eb3k;
    let z3_1 = mp_cc(eb3, et3);
    let z4_1 = mp_cc(eb4, et3);
    let z5_1 = mp_cc(eb3, et4);
    let z6_1 = mp_cc(eb4, et4);
    let z7_1 = eb4k;
    let z8_1 = eb1k;
    let z9_1 = eb1t[0];
    let z10_1 = eb1t[1];
    let z11_1 = eb2k;
    let z12_1 = eb1t[2];
    let z13_1 = eb1t[3];
    let z14_1 = z3_1 * z3_1;
    let z15_1 = z14_1 * z4_1;
    let z16_1 = 3.0 * c[3];
    let z16_2 = z16_1 * z15_1;
    let z17_1 = 3.0 * c[6];
    let z18_1 = z5_1 * z5_1;
    let z17_2 = (z17_1 * z18_1) * z6_1;
    let z19_1 = z3_1 * c[8];
    let z20_1 = c[9] * z5_1;
    let z16_3 = ((z20_1 + z17_2) + z19_1) + z16_2;
    let z16_4 = z2_1 * z16_3;
    let z17_3 = c[3] * ((z3_1 * z3_1) * z3_1);
    let z19_2 = c[6] * ((z5_1 * z5_1) * z5_1);
    let z17_4 = z17_3 + z19_2;
    let z19_3 = z7_1 * z17_4;
    let z16_5 = z16_4 - z19_3;
    let z20_2 = 0.5 * z1_1;
    let z21_1 = z20_2 * z2_1;
    let z16_6 = z16_5 * z21_1;
    let z22_1 = z8_1 * z17_4;
    let z23_1 = 0.5 * f;
    let z24_1 = z23_1 * z17_4;
    let z24_2 = z22_1 + z24_1;
    let z25_1 = z10_1 * c[6];
    let z26_1 = c[5] + (3.0 * z25_1);
    let z27_1 = z26_1 * z18_1;
    let z28_1 = z9_1 * c[3];
    let z29_1 = c[2] + (3.0 * z28_1);
    let z30_1 = z14_1 * z29_1;
    let z27_2 = z27_1 + z30_1;
    let z30_2 = z2_1 * z27_2;
    let z30_3 = z30_2 - z24_2;
    let z30_4 = z30_3 * z21_1;
    let z31_1 = (z17_4 * (z2_1 * z2_1)) * z20_2;
    let z32_1 = (-z11_1) * z27_2;
    let z15_2 = z15_1 * c[3];
    let z18_2 = z18_1 * c[6];
    let z33_1 = z18_2 * z6_1;
    let z15_3 = z15_2 + z33_1;
    let z33_2 = 3.0 * z2_1;
    let z34_1 = z15_3 * z33_2;
    let z34_2 = z34_1 - z19_3;
    let z34_3 = param[1] * z34_2;
    let z18_3 = z18_2 * z13_1;
    let z14_2 = (z14_1 * z12_1) * c[3];
    let z14_3 = z18_3 + z14_2;
    let z18_4 = z23_1 + z8_1;
    let z23_2 = (-z14_3) * z18_4;
    let z23_3 = (z34_3 + (3.0 * z23_2)) + z32_1;
    let z29_2 = (z12_1 * z3_1) * z29_1;
    let z26_2 = (z5_1 * z13_1) * z26_1;
    let z26_3 = z26_2 + z29_2;
    let z26_4 = z2_1 * z26_3;
    let z23_4 = z26_4 + (0.5 * z23_3);
    let z23_5 = z2_1 * z23_4;
    let z24_3 = z11_1 * z24_2;
    let z23_6 = z24_3 + z23_5;
    let z23_7 = z1_1 * z23_6;
    let z24_4 = (-z11_1) * z17_4;
    let z26_5 = z14_3 * z33_2;
    let z24_5 = z24_4 + z26_5;
    let z21_2 = z24_5 * z21_1;
    let z24_6 = param[2] * z15_3;
    let z26_6 = (z5_1 * c[6]) * (z13_1 * z13_1);
    let z29_3 = (c[3] * z3_1) * (z12_1 * z12_1);
    let z24_7 = (z29_3 + z26_6) + z24_6;
    let z24_8 = z24_7 * z33_2;
    let z26_7 = (-param[2]) * z19_3;
    let z14_4 = z11_1 * z14_3;
    let z14_5 = (z24_8 - (3.0 * z14_4)) + z26_7;
    let z14_6 = z2_1 * z14_5;
    let z24_9 = z17_4 * (z11_1 * z11_1);
    let z14_7 = z24_9 + z14_6;
    let z14_8 = z14_7 * z20_2;
    let z18_5 = (-z27_2) * z18_4;
    let z19_4 = (-param[0]) * z19_3;
    let z18_6 = z19_4 + z18_5;
    let z15_4 = param[0] * z15_3;
    let z19_5 = c[2] + (1.5 * z28_1);
    let z19_6 = z9_1 * z19_5;
    let z19_7 = z19_6 + (0.5 * c[1]);
    let z19_8 = z3_1 * z19_7;
    let z20_3 = c[5] + (1.5 * z25_1);
    let z20_4 = z10_1 * z20_3;
    let z20_5 = z20_4 + (0.5 * c[4]);
    let z20_6 = z5_1 * z20_5;
    let z15_5 = ((1.5 * z15_4) + z20_6) + z19_8;
    let z15_6 = z2_1 * z15_5;
    let z15_7 = (0.5 * z18_6) + z15_6;
    let z15_8 = z2_1 * z15_7;
    let z18_7 = f * z17_4;
    let z18_8 = (0.25 * z18_7) + z22_1;
    let z18_9 = f * z18_8;
    let z17_5 = z17_4 * (z8_1 * z8_1);
    let z17_6 = z18_9 + z17_5;
    let z15_9 = (0.5 * z17_6) + z15_8;
    let z15_10 = z1_1 * z15_9;
    divnum[0] -= z31_1;
    divnum[2] -= z21_2;
    divnum[4] -= z16_6;
    divnum[6] -= z14_8;
    divnum[1] -= z30_4;
    divnum[5] -= z23_7;
    divnum[3] -= z15_10;
}

/// `cuts::correctbubcoeffs` for rank - n = 0, second-orientation coefficients.
#[inline(never)]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(super) fn bubble_from_triangle_r0_spurious(
    divnum: &mut [C64],
    et3: &CMom,
    et4: &CMom,
    c: &[C64],
    eb3: &CMom,
    eb4: &CMom,
    param: &[C64; 3],
    eb1k: C64,
    eb2k: Dd,
    eb3k: C64,
    ieb3k: C64,
    eb4k: C64,
    eb1t: &[C64; 4],
    f: C64,
) {
    let z1_1 = mp_cc(eb3, et3);
    let z2_1 = ieb3k;
    let z3_1 = mp_cc(eb3, et4);
    let z4_1 = eb2k;
    let z5_1 = eb1t[2];
    let z6_1 = eb1t[3];
    let z7_1 = eb1k;
    let z8_1 = eb1t[0];
    let z9_1 = eb1t[1];
    let z10_1 = c[3] * ((z1_1 * z1_1) * z1_1);
    let z11_1 = c[6] * ((z3_1 * z3_1) * z3_1);
    let z10_2 = z10_1 + z11_1;
    let z11_2 = 0.5 * z2_1;
    let z12_1 = z10_2 * z11_2;
    let z13_1 = z3_1 * z3_1;
    let z14_1 = z13_1 * c[6];
    let z15_1 = z6_1 * z14_1;
    let z16_1 = z1_1 * z1_1;
    let z17_1 = z16_1 * c[3];
    let z18_1 = z5_1 * z17_1;
    let z15_2 = z15_1 + z18_1;
    let z10_3 = z10_2 * z2_1;
    let z18_2 = (-z4_1) * z10_3;
    let z15_3 = (3.0 * z15_2) + z18_2;
    let z15_4 = z15_3 * z11_2;
    let z17_2 = z8_1 * z17_1;
    let z14_2 = z9_1 * z14_1;
    let z14_3 = z17_2 + z14_2;
    let z17_3 = (-z7_1) - (0.5 * f);
    let z10_4 = z17_3 * z10_3;
    let z16_2 = c[2] * z16_1;
    let z13_2 = c[5] * z13_1;
    let z10_5 = ((z10_4 + z13_2) + z16_2) + (3.0 * z14_3);
    let z10_6 = z10_5 * z11_2;
    divnum[0] -= z12_1;
    divnum[2] -= z15_4;
    divnum[1] -= z10_6;
}

/// `x1cuts::correctbubcoeffs` (rank n+1), all coefficients.
#[inline(never)]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(super) fn bubble_from_triangle_x1_all(
    divnum: &mut [C64],
    et3: &CMom,
    et4: &CMom,
    c: &[C64],
    eb3: &CMom,
    eb4: &CMom,
    param: &[C64; 3],
    eb1k: C64,
    eb2k: Dd,
    eb3k: C64,
    ieb3k: C64,
    eb4k: C64,
    eb1t: &[C64; 4],
    f: C64,
) {
    let z1_1 = ((ieb3k * ieb3k) * ieb3k) * ieb3k;
    let z2_1 = eb1k;
    let z3_1 = eb3k;
    let z4_1 = mp_cc(eb3, et3);
    let z5_1 = mp_cc(eb4, et3);
    let z6_1 = mp_cc(eb3, et4);
    let z7_1 = mp_cc(eb4, et4);
    let z8_1 = eb4k;
    let z9_1 = eb1t[0];
    let z10_1 = eb1t[1];
    let z11_1 = eb2k;
    let z12_1 = eb1t[2];
    let z13_1 = eb1t[3];
    let z14_1 = z9_1 * c[12];
    let z15_1 = (2.0 * z14_1) + (0.5 * c[3]);
    let z16_1 = z4_1 * z4_1;
    let z17_1 = z16_1 * z5_1;
    let z18_1 = z17_1 * z15_1;
    let z19_1 = 0.5 * c[6];
    let z20_1 = z6_1 * z6_1;
    let z21_1 = z19_1 * z20_1;
    let z22_1 = z21_1 * z7_1;
    let z18_2 = z18_1 + z22_1;
    let z22_2 = 3.0 * z3_1;
    let z18_3 = z18_2 * z22_2;
    let z23_1 = (z4_1 * z4_1) * z4_1;
    let z24_1 = z15_1 * z23_1;
    let z25_1 = (z6_1 * z6_1) * z6_1;
    let z26_1 = z19_1 * z25_1;
    let z24_2 = z24_1 + z26_1;
    let z26_2 = z24_2 * z8_1;
    let z18_4 = z18_3 - z26_2;
    let z26_3 = z3_1 * z3_1;
    let z18_5 = z18_4 * z26_3;
    let z27_1 = z7_1 * c[13];
    let z28_1 = z27_1 * z25_1;
    let z29_1 = z23_1 * c[12];
    let z30_1 = z29_1 * z5_1;
    let z28_2 = z28_1 + z30_1;
    let z30_2 = 2.0 * z3_1;
    let z31_1 = z28_2 * z30_2;
    let z32_1 = c[12] * (z16_1 * z16_1);
    let z33_1 = c[13] * (((z6_1 * z6_1) * z6_1) * z6_1);
    let z32_2 = z32_1 + z33_1;
    let z33_2 = z32_2 * z8_1;
    let z33_3 = z31_1 - z33_2;
    let z34_1 = z2_1 * z3_1;
    let z35_1 = z33_3 * z34_1;
    let z28_3 = z28_2 * z3_1;
    let z36_1 = 0.5 * z32_2;
    let z37_1 = z36_1 * z8_1;
    let z28_4 = z28_3 - z37_1;
    let z38_1 = f * z3_1;
    let z39_1 = z28_4 * z38_1;
    let z27_2 = z27_1 * z20_1;
    let z40_1 = z27_2 * z22_2;
    let z41_1 = z25_1 * c[13];
    let z42_1 = z41_1 * z8_1;
    let z40_2 = z42_1 - z40_1;
    let z42_2 = 2.0 * z26_3;
    let z40_3 = z40_2 * z42_2;
    let z43_1 = z40_3 * z10_1;
    let z35_2 = ((z39_1 - z18_5) + z35_1) + z43_1;
    let z39_2 = 0.5 * z16_1;
    let z43_2 = z26_3 * z2_1;
    let z44_1 = (-z43_2) * z39_2;
    let z45_1 = z26_3 * f;
    let z46_1 = z16_1 * z45_1;
    let z47_1 = (z3_1 * z3_1) * z3_1;
    let z48_1 = z47_1 * z4_1;
    let z49_1 = z9_1 * z48_1;
    let z44_2 = ((-0.25 * z46_1) + z49_1) + z44_1;
    let z44_3 = c[10] * z44_2;
    let z46_2 = z20_1 * z26_3;
    let z49_2 = 0.5 * f;
    let z50_1 = z49_2 + z2_1;
    let z50_2 = z46_2 * z50_1;
    let z51_1 = z47_1 * z6_1;
    let z52_1 = z10_1 * z51_1;
    let z52_2 = (-0.5 * z50_2) + z52_1;
    let z52_3 = c[11] * z52_2;
    let z53_1 = 0.5 * z48_1;
    let z54_1 = c[8] * z53_1;
    let z55_1 = 0.5 * z51_1;
    let z56_1 = c[9] * z55_1;
    let z44_4 = (((z56_1 + z44_3) + z52_3) + z54_1) - z35_2;
    let z44_5 = z1_1 * z44_4;
    let z52_4 = z36_1 * z34_1;
    let z54_2 = z14_1 + (0.25 * c[3]);
    let z56_2 = z54_2 * z23_1;
    let z25_2 = z25_1 * c[6];
    let z56_3 = z56_2 + (0.25 * z25_2);
    let z57_1 = (-z56_3) * z26_3;
    let z58_1 = z32_2 * z38_1;
    let z57_2 = ((0.125 * z58_1) + z57_1) + z52_4;
    let z57_3 = f * z57_2;
    let z58_2 = z42_2 * z2_1;
    let z45_2 = z58_2 + z45_1;
    let z58_3 = (-z41_1) * z45_2;
    let z59_1 = z20_1 * c[6];
    let z60_1 = z47_1 * z59_1;
    let z61_1 = 3.0 * z47_1;
    let z62_1 = z61_1 * z20_1;
    let z63_1 = (z10_1 * c[13]) * z62_1;
    let z58_4 = (z63_1 + (1.5 * z60_1)) + z58_3;
    let z58_5 = z10_1 * z58_4;
    let z31_2 = z37_1 - z31_1;
    let z31_3 = z31_2 * z26_3;
    let z37_2 = (-param[0]) * z31_3;
    let z60_2 = z24_2 * z26_3;
    let z52_5 = (-z60_2) + z52_4;
    let z52_6 = z2_1 * z52_5;
    let z63_2 = z9_1 * z9_1;
    let z64_1 = z63_2 * c[12];
    let z65_1 = c[3] * z9_1;
    let z64_2 = ((3.0 * z64_1) + (1.5 * z65_1)) + (0.5 * c[2]);
    let z64_3 = z64_2 * z16_1;
    let z66_1 = 0.5 * z20_1;
    let z67_1 = z66_1 * c[5];
    let z64_4 = z64_3 + z67_1;
    let z67_2 = z64_4 * z47_1;
    let z37_3 = (((z58_5 + z57_3) + z52_6) + z67_2) + z37_2;
    let z37_4 = z1_1 * z37_3;
    let z52_7 = c[11] * z66_1;
    let z57_4 = c[10] * z39_2;
    let z52_8 = z57_4 + z52_7;
    let z52_9 = z47_1 * z52_8;
    let z52_10 = (-z31_3) + z52_9;
    let z52_11 = z1_1 * z52_10;
    let z57_5 = 2.0 * z10_1;
    let z58_6 = z41_1 * z57_5;
    let z58_7 = z58_6 + z24_2;
    let z58_8 = z47_1 * z58_7;
    let z66_2 = (-z36_1) * z43_2;
    let z67_3 = z32_2 * f;
    let z68_1 = z26_3 * z67_3;
    let z58_9 = ((-0.25 * z68_1) + z66_2) + z58_8;
    let z58_10 = z1_1 * z58_9;
    let z66_3 = (z1_1 * z47_1) * z36_1;
    let z68_2 = param[0] * z3_1;
    let z69_1 = z68_2 * z33_3;
    let z64_5 = z64_4 * z26_3;
    let z69_2 = z64_5 + z69_1;
    let z70_1 = z32_2 * z2_1;
    let z71_1 = 1.5 * z70_1;
    let z14_2 = (4.0 * z14_1) + c[3];
    let z14_3 = z14_2 * z23_1;
    let z14_4 = z25_2 + z14_3;
    let z14_5 = z3_1 * z14_4;
    let z14_6 = z14_5 - z71_1;
    let z14_7 = z2_1 * z14_6;
    let z23_2 = z24_2 * z3_1;
    let z24_3 = z71_1 - z23_2;
    let z25_3 = (-0.375 * z67_3) - z24_3;
    let z25_4 = f * z25_3;
    let z14_8 = (z25_4 + z14_7) - z69_2;
    let z14_9 = z11_1 * z14_8;
    let z25_5 = (-param[1]) * z35_2;
    let z29_2 = z29_1 * z12_1;
    let z71_2 = z41_1 * z13_1;
    let z29_3 = z29_2 + z71_2;
    let z71_3 = z29_3 * z30_2;
    let z72_1 = z71_3 * z2_1;
    let z73_1 = z16_1 * z12_1;
    let z54_3 = (-z54_2) * z73_1;
    let z59_2 = z13_1 * z59_1;
    let z54_4 = (-0.25 * z59_2) + z54_3;
    let z59_3 = 3.0 * z26_3;
    let z54_5 = z54_4 * z59_3;
    let z74_1 = 0.5 * z38_1;
    let z75_1 = z29_3 * z74_1;
    let z54_6 = (z75_1 + z54_5) + z72_1;
    let z54_7 = f * z54_6;
    let z75_2 = z13_1 * c[13];
    let z76_1 = z75_2 * z51_1;
    let z77_1 = z46_2 * z11_1;
    let z78_1 = (-c[13]) * z77_1;
    let z76_2 = (2.0 * z76_1) + z78_1;
    let z76_3 = z10_1 * z76_2;
    let z78_2 = z46_2 * z75_2;
    let z79_1 = (-f) - (2.0 * z2_1);
    let z79_2 = z78_2 * z79_1;
    let z80_1 = z51_1 * z13_1;
    let z81_1 = c[6] * z80_1;
    let z76_4 = (z76_3 + z81_1) + z79_2;
    let z46_3 = z46_2 * c[6];
    let z79_3 = 1.5 * z46_3;
    let z81_2 = 4.0 * z34_1;
    let z82_1 = f * z30_2;
    let z82_2 = z82_1 + z81_2;
    let z82_3 = z41_1 * z82_2;
    let z82_4 = (-z79_3) + z82_3;
    let z82_5 = z11_1 * z82_4;
    let z76_5 = z82_5 + (3.0 * z76_4);
    let z76_6 = z10_1 * z76_5;
    let z27_3 = z27_2 * z13_1;
    let z82_6 = c[12] * z12_1;
    let z17_2 = z17_1 * z82_6;
    let z17_3 = z27_3 + z17_2;
    let z17_4 = z17_3 * z22_2;
    let z22_3 = z29_3 * z8_1;
    let z17_5 = z17_4 - z22_3;
    let z17_6 = z17_5 * z42_2;
    let z22_4 = param[0] * z17_6;
    let z27_4 = z15_1 * z73_1;
    let z21_2 = z21_1 * z13_1;
    let z21_3 = z27_4 + z21_2;
    let z27_5 = z21_3 * z59_3;
    let z72_2 = (-z27_5) + z72_1;
    let z72_3 = z2_1 * z72_2;
    let z73_2 = z63_2 * z82_6;
    let z65_2 = (3.0 * z65_1) + c[2];
    let z65_3 = z12_1 * z65_2;
    let z65_4 = (6.0 * z73_2) + z65_3;
    let z65_5 = z4_1 * z65_4;
    let z73_3 = (c[5] * z6_1) * z13_1;
    let z65_6 = z73_3 + z65_5;
    let z65_7 = z65_6 * z47_1;
    let z14_10 = (((((z25_5 + z76_6) + z14_9) + z54_7) + z72_3) + z65_7) + z22_4;
    let z14_11 = z1_1 * z14_10;
    let z22_5 = z11_1 * z3_1;
    let z25_6 = z22_5 * z33_3;
    let z17_7 = z17_6 - z25_6;
    let z25_7 = z80_1 - (0.5 * z77_1);
    let z25_8 = c[11] * z25_7;
    let z33_4 = z26_3 * z11_1;
    let z39_3 = (-z33_4) * z39_2;
    let z54_8 = z12_1 * z48_1;
    let z39_4 = z54_8 + z39_3;
    let z39_5 = c[10] * z39_4;
    let z25_9 = (z39_5 + z25_8) + z17_7;
    let z25_10 = z1_1 * z25_9;
    let z32_3 = z32_2 * z34_1;
    let z34_2 = z36_1 * z38_1;
    let z32_4 = (z34_2 - z60_2) + z32_3;
    let z32_5 = z11_1 * z32_4;
    let z34_3 = (-z29_3) * z45_2;
    let z21_4 = z21_3 * z61_1;
    let z38_2 = z75_2 * z62_1;
    let z39_6 = (-z41_1) * z33_4;
    let z38_3 = z38_2 + z39_6;
    let z38_4 = z38_3 * z57_5;
    let z39_7 = (-param[1]) * z31_3;
    let z21_5 = (((z39_7 + z38_4) + z32_5) + z21_4) + z34_3;
    let z21_6 = z1_1 * z21_5;
    let z32_6 = 2.0 * z47_1;
    let z34_4 = z29_3 * z32_6;
    let z33_5 = (-z36_1) * z33_4;
    let z33_6 = z34_4 + z33_5;
    let z33_7 = z1_1 * z33_6;
    let z24_4 = (-0.75 * z67_3) - z24_3;
    let z24_5 = z11_1 * z24_4;
    let z34_5 = z29_3 * z81_2;
    let z38_5 = f * z71_3;
    let z24_6 = ((z24_5 + z38_5) - z27_5) + z34_5;
    let z24_7 = z11_1 * z24_6;
    let z27_6 = (-param[2]) * z35_2;
    let z34_6 = z12_1 * z12_1;
    let z16_2 = (c[12] * z34_6) * z16_1;
    let z35_3 = z13_1 * z13_1;
    let z38_6 = z35_3 * c[13];
    let z20_2 = z38_6 * z20_1;
    let z16_3 = z16_2 + z20_2;
    let z20_3 = (-z26_3) * z49_2;
    let z20_4 = z20_3 - z43_2;
    let z20_5 = z16_3 * z20_4;
    let z26_4 = (z6_1 * z35_3) * z19_1;
    let z15_2 = (z4_1 * z34_6) * z15_1;
    let z15_3 = z26_4 + z15_2;
    let z15_4 = z15_3 * z47_1;
    let z15_5 = z15_4 + z20_5;
    let z20_6 = param[1] * z17_7;
    let z26_5 = z41_1 * z22_5;
    let z26_6 = (-3.0 * z78_2) + z26_5;
    let z26_7 = z11_1 * z26_6;
    let z34_7 = z51_1 * z38_6;
    let z26_8 = (3.0 * z34_7) + z26_7;
    let z26_9 = z26_8 * z57_5;
    let z15_6 = (((z27_6 + z20_6) + z26_9) + (3.0 * z15_5)) + z24_7;
    let z15_7 = z1_1 * z15_6;
    let z20_7 = z16_3 * z61_1;
    let z24_8 = (-z29_3) * z42_2;
    let z22_6 = z36_1 * z22_5;
    let z22_7 = z24_8 + z22_6;
    let z22_8 = z11_1 * z22_7;
    let z24_9 = (-param[2]) * z31_3;
    let z20_8 = (z24_9 + z20_7) + z22_8;
    let z20_9 = z1_1 * z20_8;
    let z16_4 = (-z16_3) * z59_3;
    let z22_9 = (-z11_1) * z36_1;
    let z22_10 = z71_3 + z22_9;
    let z22_11 = z11_1 * z22_10;
    let z16_5 = z16_4 + z22_11;
    let z16_6 = z11_1 * z16_5;
    let z17_8 = param[2] * z17_7;
    let z22_12 = (z6_1 * c[13]) * ((z13_1 * z13_1) * z13_1);
    let z24_10 = (z4_1 * c[12]) * ((z12_1 * z12_1) * z12_1);
    let z22_13 = z22_12 + z24_10;
    let z22_14 = z22_13 * z32_6;
    let z16_7 = (z17_8 + z22_14) + z16_6;
    let z16_8 = z1_1 * z16_7;
    let z17_9 = 0.75 * z70_1;
    let z22_15 = z3_1 * z56_3;
    let z22_16 = ((-0.125 * z67_3) + z22_15) - z17_9;
    let z22_17 = z22_16 * z49_2;
    let z24_11 = (-z28_4) * z68_2;
    let z17_10 = z23_2 - z17_9;
    let z17_11 = z2_1 * z17_10;
    let z17_12 = ((z22_17 + z17_11) - (0.5 * z64_5)) + z24_11;
    let z17_13 = f * z17_12;
    let z22_18 = z30_2 * z2_1;
    let z24_12 = z41_1 * z22_18;
    let z24_13 = (-z79_3) + z24_12;
    let z24_14 = z2_1 * z24_13;
    let z22_19 = z74_1 + z22_18;
    let z22_20 = z41_1 * z22_19;
    let z22_21 = (-0.75 * z46_3) + z22_20;
    let z22_22 = f * z22_21;
    let z26_10 = (-c[13]) * z50_2;
    let z19_2 = z19_1 * z51_1;
    let z19_3 = z19_2 + z26_10;
    let z26_11 = (c[13] * z51_1) * z57_5;
    let z19_4 = (3.0 * z19_3) + z26_11;
    let z19_5 = z10_1 * z19_4;
    let z26_12 = (-param[0]) * z40_3;
    let z27_7 = c[5] * z51_1;
    let z19_6 = (((z19_5 + z22_22) + z24_14) + z27_7) + z26_12;
    let z19_7 = z10_1 * z19_6;
    let z18_6 = param[0] * z18_5;
    let z22_23 = (-z2_1) * z36_1;
    let z22_24 = z23_2 + z22_23;
    let z22_25 = z2_1 * z22_24;
    let z22_26 = z22_25 - z69_2;
    let z22_27 = z2_1 * z22_26;
    let z23_3 = 2.0 * c[12];
    let z23_4 = z63_2 * z23_3;
    let z23_5 = c[2] + z23_4;
    let z23_6 = z9_1 * z23_5;
    let z24_15 = 1.5 * c[3];
    let z24_16 = z63_2 * z24_15;
    let z23_7 = z23_6 + z24_16;
    let z23_8 = z23_7 * z48_1;
    let z24_17 = c[4] * z55_1;
    let z26_13 = c[1] * z53_1;
    let z17_14 = (((((z26_13 + z19_7) + z24_17) + z17_13) + z22_27) + z23_8) + z18_6;
    let z17_15 = z1_1 * z17_14;
    divnum[7] -= z17_15;
    divnum[8] -= z44_5;
    divnum[3] -= z37_4;
    divnum[4] -= z52_11;
    divnum[1] -= z58_10;
    divnum[0] -= z66_3;
    divnum[9] -= z14_11;
    divnum[10] -= z25_10;
    divnum[5] -= z21_6;
    divnum[2] -= z33_7;
    divnum[11] -= z15_7;
    divnum[6] -= z20_9;
    divnum[12] -= z16_8;
}

/// `x1cuts::correctbubcoeffs` (rank n+1), second-orientation coefficients.
#[inline(never)]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(super) fn bubble_from_triangle_x1_spurious(
    divnum: &mut [C64],
    et3: &CMom,
    et4: &CMom,
    c: &[C64],
    eb3: &CMom,
    eb4: &CMom,
    param: &[C64; 3],
    eb1k: C64,
    eb2k: Dd,
    eb3k: C64,
    ieb3k: C64,
    eb4k: C64,
    eb1t: &[C64; 4],
    f: C64,
) {
    let z1_1 = ((ieb3k * ieb3k) * ieb3k) * ieb3k;
    let z2_1 = eb1k;
    let z3_1 = eb3k;
    let z4_1 = mp_cc(eb3, et3);
    let z5_1 = mp_cc(eb3, et4);
    let z6_1 = eb1t[0];
    let z7_1 = eb1t[1];
    let z8_1 = mp_cc(eb4, et3);
    let z9_1 = mp_cc(eb4, et4);
    let z10_1 = eb4k;
    let z11_1 = eb2k;
    let z12_1 = eb1t[2];
    let z13_1 = eb1t[3];
    let z14_1 = z5_1 * z5_1;
    let z15_1 = c[13] * z1_1;
    let z16_1 = z14_1 * z15_1;
    let z17_1 = z7_1 * z16_1;
    let z18_1 = 0.5 * z1_1;
    let z19_1 = z14_1 * z18_1;
    let z20_1 = c[6] * z19_1;
    let z20_2 = z20_1 + z17_1;
    let z20_3 = z7_1 * z20_2;
    let z21_1 = z4_1 * z4_1;
    let z22_1 = z21_1 * z18_1;
    let z23_1 = (z6_1 * c[3]) * z22_1;
    let z24_1 = c[12] * z1_1;
    let z25_1 = z21_1 * z24_1;
    let z26_1 = (z6_1 * z6_1) * z25_1;
    let z20_4 = (z20_3 + z23_1) + z26_1;
    let z23_2 = (z4_1 * z4_1) * z4_1;
    let z26_2 = z23_2 * z24_1;
    let z27_1 = z26_2 * z8_1;
    let z28_1 = (z5_1 * z5_1) * z5_1;
    let z29_1 = z28_1 * z15_1;
    let z30_1 = z29_1 * z9_1;
    let z27_2 = z27_1 + z30_1;
    let z30_2 = 2.0 * z27_2;
    let z31_1 = param[0] * z30_2;
    let z32_1 = c[2] * z22_1;
    let z20_5 = (z32_1 + (3.0 * z20_4)) + z31_1;
    let z20_6 = z3_1 * z20_5;
    let z31_2 = z26_2 * z6_1;
    let z32_2 = z29_1 * z7_1;
    let z31_3 = z31_2 + z32_2;
    let z28_2 = z28_1 * c[6];
    let z23_3 = z23_2 * c[3];
    let z23_4 = z28_2 + z23_3;
    let z28_3 = z1_1 * z23_4;
    let z28_4 = (-0.25 * z28_3) - z31_3;
    let z28_5 = f * z28_4;
    let z15_2 = z15_1 * (z14_1 * z14_1);
    let z32_3 = z24_1 * (((z4_1 * z4_1) * z4_1) * z4_1);
    let z15_3 = z15_2 + z32_3;
    let z32_4 = z10_1 * z15_3;
    let z32_5 = 0.5 * z32_4;
    let z33_1 = (-param[0]) * z32_5;
    let z20_7 = (z20_6 + z33_1) + z28_5;
    let z20_8 = z3_1 * z20_7;
    let z23_5 = z23_4 * z18_1;
    let z23_6 = z23_5 + (2.0 * z31_3);
    let z28_6 = z23_6 * z3_1;
    let z31_4 = 0.5 * z15_3;
    let z33_2 = z31_4 * z2_1;
    let z28_7 = z28_6 - z33_2;
    let z33_3 = z31_4 * f;
    let z34_1 = z33_3 - z28_7;
    let z34_2 = z2_1 * z34_1;
    let z35_1 = z15_3 * (f * f);
    let z20_9 = (z34_2 + (0.125 * z35_1)) + z20_8;
    let z20_10 = z3_1 * z20_9;
    let z34_3 = (z3_1 * z3_1) * z3_1;
    let z35_2 = (c[5] * z34_3) * z19_1;
    let z20_11 = z35_2 + z20_10;
    let z19_2 = c[11] * z19_1;
    let z22_2 = c[10] * z22_1;
    let z19_3 = (z22_2 + z19_2) + z30_2;
    let z19_4 = z3_1 * z19_3;
    let z19_5 = (-z32_5) + z19_4;
    let z22_3 = z3_1 * z3_1;
    let z19_6 = z19_5 * z22_3;
    let z35_3 = z15_3 * f;
    let z28_8 = (-0.25 * z35_3) + z28_7;
    let z28_9 = z22_3 * z28_8;
    let z34_4 = z34_3 * z31_4;
    let z21_2 = z21_1 * z12_1;
    let z24_2 = (z21_2 * z24_1) * z6_1;
    let z17_2 = z13_1 * z17_1;
    let z17_3 = z24_2 + z17_2;
    let z14_2 = (c[6] * z14_1) * z13_1;
    let z21_3 = c[3] * z21_2;
    let z14_3 = z14_2 + z21_3;
    let z14_4 = z14_3 * z18_1;
    let z14_5 = z14_4 + (2.0 * z17_3);
    let z17_4 = param[1] * z30_2;
    let z14_6 = (3.0 * z14_5) + z17_4;
    let z14_7 = z3_1 * z14_6;
    let z17_5 = (-z11_1) * z23_6;
    let z18_2 = z26_2 * z12_1;
    let z21_4 = z29_1 * z13_1;
    let z18_3 = z18_2 + z21_4;
    let z21_5 = (-f) * z18_3;
    let z23_7 = (-param[1]) * z32_5;
    let z14_8 = ((z14_7 + z21_5) + z17_5) + z23_7;
    let z14_9 = z3_1 * z14_8;
    let z17_6 = 2.0 * z18_3;
    let z18_4 = z17_6 * z3_1;
    let z15_4 = z15_3 * z11_1;
    let z15_5 = z15_4 - z18_4;
    let z15_6 = z2_1 * z15_5;
    let z21_6 = z11_1 * z33_3;
    let z14_10 = (z15_6 + z21_6) + z14_9;
    let z14_11 = z3_1 * z14_10;
    let z15_7 = (-z11_1) * z31_4;
    let z15_8 = z15_7 + z18_4;
    let z15_9 = z15_8 * z22_3;
    let z17_7 = (-z11_1) * z17_6;
    let z18_5 = (z12_1 * z12_1) * z25_1;
    let z16_2 = (z13_1 * z13_1) * z16_1;
    let z16_3 = z18_5 + z16_2;
    let z16_4 = z3_1 * z16_3;
    let z16_5 = z17_7 + (3.0 * z16_4);
    let z16_6 = z3_1 * z16_5;
    let z17_8 = (z11_1 * z11_1) * z31_4;
    let z16_7 = z17_8 + z16_6;
    let z16_8 = z3_1 * z16_7;
    let z17_9 = 2.0 * z3_1;
    let z17_10 = z27_2 * z17_9;
    let z17_11 = (-z32_5) + z17_10;
    let z17_12 = (param[2] * z17_11) * z22_3;
    let z16_9 = z16_8 + z17_12;
    divnum[3] -= z20_11;
    divnum[4] -= z19_6;
    divnum[1] -= z28_9;
    divnum[0] -= z34_4;
    divnum[5] -= z14_11;
    divnum[2] -= z15_9;
    divnum[6] -= z16_9;
}

/// `x1cuts::correcttadcoeffs` (triangle parent, rank n+1).
#[inline(never)]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(super) fn tadpole_from_triangle_x1(
    divnum: &mut [C64],
    et3: &CMom,
    et4: &CMom,
    c: &[C64],
    e3: &CMom,
    e4: &CMom,
    param: C64,
    e3k0: C64,
    ie3k0: C64,
    e4k0: C64,
    f0: C64,
    e3k1: C64,
    ie3k1: C64,
    e4k1: C64,
    f1: C64,
) {
    let z1_1 = ((((ie3k0 * ie3k0) * ie3k0) * ie3k1) * ie3k1) * ie3k1;
    let z2_1 = mp_cc(e3, et3);
    let z3_1 = e3k0;
    let z4_1 = e3k1;
    let z5_1 = e4k1;
    let z6_1 = e4k0;
    let z7_1 = mp_cc(e4, et3);
    let z8_1 = mp_cc(e3, et4);
    let z9_1 = mp_cc(e4, et4);
    let z10_1 = z3_1 * z3_1;
    let z11_1 = z10_1 * z4_1;
    let z12_1 = z5_1 * z11_1;
    let z13_1 = z4_1 * z4_1;
    let z14_1 = (z13_1 * z6_1) * z3_1;
    let z12_2 = z12_1 + z14_1;
    let z14_2 = 0.25 * z8_1;
    let z14_3 = z14_2 * z12_2;
    let z15_1 = z4_1 * z3_1;
    let z16_1 = z15_1 * z15_1;
    let z17_1 = z16_1 * z9_1;
    let z14_4 = z14_3 - z17_1;
    let z14_5 = c[13] * z14_4;
    let z17_2 = (-z8_1) * z14_5;
    let z18_1 = 0.25 * z16_1;
    let z19_1 = c[11] * z18_1;
    let z17_3 = z19_1 + z17_2;
    let z19_2 = z8_1 * z8_1;
    let z17_4 = z17_3 * z19_2;
    let z20_1 = z2_1 * c[12];
    let z12_3 = z12_2 * z20_1;
    let z21_1 = (z7_1 * z16_1) * c[12];
    let z12_4 = (-z21_1) + (0.25 * z12_3);
    let z21_2 = (-z2_1) * z12_4;
    let z18_2 = c[10] * z18_1;
    let z18_3 = z18_2 + z21_2;
    let z21_3 = z2_1 * z2_1;
    let z18_4 = z18_3 * z21_3;
    let z17_5 = z17_4 + z18_4;
    let z17_6 = z1_1 * z17_5;
    let z11_2 = f1 * z11_1;
    let z13_2 = z13_1 * f0;
    let z18_5 = z13_2 * z3_1;
    let z11_3 = z11_2 + z18_5;
    let z18_6 = (-c[3]) * z11_3;
    let z15_2 = z15_1 * f1;
    let z13_3 = z15_2 + z13_2;
    let z13_4 = z13_3 * f0;
    let z10_2 = z10_1 * (f1 * f1);
    let z10_3 = z10_2 + z13_4;
    let z13_5 = z10_3 * z20_1;
    let z13_6 = (0.5 * z13_5) + z18_6;
    let z13_7 = z2_1 * z13_6;
    let z15_3 = c[2] * z16_1;
    let z13_8 = z15_3 + (0.5 * z13_7);
    let z13_9 = z13_8 * z21_3;
    let z10_4 = (z19_2 * c[13]) * z10_3;
    let z15_4 = c[5] * z16_1;
    let z10_5 = z15_4 + (0.25 * z10_4);
    let z10_6 = z10_5 * z19_2;
    let z15_5 = (z8_1 * z8_1) * z8_1;
    let z11_4 = (c[6] * z11_3) * z15_5;
    let z10_7 = (z13_9 + z10_6) - (0.5 * z11_4);
    let z11_5 = (-z14_5) * z15_5;
    let z12_5 = (-z12_4) * ((z2_1 * z2_1) * z2_1);
    let z11_6 = z11_5 + z12_5;
    let z11_7 = param * z11_6;
    let z10_8 = (0.25 * z10_7) + z11_7;
    let z10_9 = z1_1 * z10_8;
    divnum[3] -= z17_6;
    divnum[2] -= z10_9;
}

/// `x1cuts::correcttadcoeffs` (bubble parent, rank n+1).
#[inline(never)]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(super) fn tadpole_from_bubble_x1(
    divnum: &mut [C64],
    eb2: &RMom,
    eb3: &CMom,
    eb4: &CMom,
    kbol: &RMom,
    c: &[C64],
    e3: &CMom,
    e4: &CMom,
    param: C64,
    e3k: C64,
    ie3k: C64,
    e4k: C64,
    f: C64,
) {
    let z1_1 = (ie3k * ie3k) * ie3k;
    let z2_1 = e3k;
    let z3_1 = mp_cr(e3, eb2);
    let z4_1 = mp_cr(e4, eb2);
    let z5_1 = mp_cc(e4, eb3);
    let z6_1 = mp_cc(e4, eb4);
    let z7_1 = mp_cc(e3, eb3);
    let z8_1 = mp_cc(e3, eb4);
    let z9_1 = e4k;
    let z10_1 = mp_rr(eb2, kbol);
    let z11_1 = z4_1 * z8_1;
    let z12_1 = 0.5 * z3_1;
    let z13_1 = z12_1 * z6_1;
    let z11_2 = z11_1 + z13_1;
    let z11_3 = z11_2 * c[17];
    let z13_2 = z4_1 * z7_1;
    let z12_2 = z12_1 * z5_1;
    let z12_3 = z13_2 + z12_2;
    let z12_4 = z12_3 * c[16];
    let z13_3 = (c[19] * z6_1) * z8_1;
    let z14_1 = (z5_1 * c[18]) * z7_1;
    let z15_1 = (c[13] * z3_1) * z4_1;
    let z11_4 = (((z14_1 + z13_3) + (1.5 * z15_1)) + z11_3) + z12_4;
    let z12_5 = (0.5 * c[10]) + z11_4;
    let z12_6 = z3_1 * z12_5;
    let z13_4 = z7_1 * z7_1;
    let z14_2 = (z5_1 * z13_4) * c[14];
    let z15_2 = z8_1 * z8_1;
    let z16_1 = (c[15] * z15_2) * z6_1;
    let z14_3 = z14_2 + z16_1;
    let z16_2 = z13_4 * c[18];
    let z17_1 = z15_2 * c[19];
    let z16_3 = z16_2 + z17_1;
    let z17_2 = z16_3 * z4_1;
    let z14_4 = z17_2 + (3.0 * z14_3);
    let z17_3 = z7_1 * c[11];
    let z18_1 = c[12] * z8_1;
    let z17_4 = (z18_1 + z17_3) + z14_4;
    let z12_7 = z12_6 + (0.5 * z17_4);
    let z12_8 = z2_1 * z12_7;
    let z17_5 = z3_1 * z3_1;
    let z18_2 = z17_5 * c[13];
    let z18_3 = z16_3 + z18_2;
    let z18_4 = z18_3 * z3_1;
    let z19_1 = c[17] * z8_1;
    let z20_1 = c[16] * z7_1;
    let z19_2 = z19_1 + z20_1;
    let z17_6 = z17_5 * z19_2;
    let z20_2 = c[14] * ((z7_1 * z7_1) * z7_1);
    let z21_1 = c[15] * ((z8_1 * z8_1) * z8_1);
    let z17_7 = ((z18_4 + z21_1) + z17_6) + z20_2;
    let z18_5 = z17_7 * z9_1;
    let z12_9 = (-0.5 * z18_5) + z12_8;
    let z20_3 = z2_1 * z1_1;
    let z12_10 = z12_9 * z20_3;
    let z16_4 = (-z16_3) * z10_1;
    let z13_5 = (-c[4]) * z13_4;
    let z15_3 = (-c[6]) * z15_2;
    let z13_6 = (z16_4 + z15_3) + z13_5;
    let z13_7 = f * z13_6;
    let z15_4 = (-c[8]) * z8_1;
    let z16_5 = (-c[7]) * z7_1;
    let z15_5 = z15_4 + z16_5;
    let z15_6 = f * z15_5;
    let z16_6 = z3_1 * f;
    let z21_2 = 3.0 * c[13];
    let z22_1 = (-z10_1) * z21_2;
    let z22_2 = z22_1 - c[2];
    let z22_3 = z22_2 * z16_6;
    let z15_7 = z22_3 + z15_6;
    let z15_8 = z3_1 * z15_7;
    let z13_8 = z15_8 + z13_7;
    let z15_9 = (-param) * z18_5;
    let z16_7 = ((-z19_2) * z16_6) * z10_1;
    let z13_9 = (z15_9 + (0.5 * z13_8)) + z16_7;
    let z11_5 = z3_1 * z11_4;
    let z11_6 = (0.5 * z14_4) + z11_5;
    let z11_7 = param * z11_6;
    let z14_5 = 0.5 * z19_2;
    let z15_10 = z10_1 * z10_1;
    let z14_6 = z15_10 * z14_5;
    let z16_8 = c[8] * z10_1;
    let z16_9 = z16_8 + c[5];
    let z16_10 = z8_1 * z16_9;
    let z18_6 = c[7] * z10_1;
    let z18_7 = z18_6 + c[3];
    let z18_8 = z7_1 * z18_7;
    let z16_11 = z18_8 + z16_10;
    let z15_11 = z15_10 * z21_2;
    let z15_12 = c[1] + z15_11;
    let z18_9 = c[2] * z10_1;
    let z15_13 = (0.5 * z15_12) + z18_9;
    let z15_14 = z3_1 * z15_13;
    let z11_8 = ((z11_7 + (0.5 * z16_11)) + z15_14) + z14_6;
    let z11_9 = z2_1 * z11_8;
    let z11_10 = (0.5 * z13_9) + z11_9;
    let z11_11 = z11_10 * z20_3;
    let z13_10 = (z1_1 * z17_7) * (f * f);
    let z11_12 = (0.125 * z13_10) + z11_11;
    divnum[3] -= z12_10;
    divnum[2] -= z11_12;
}

}

// ---- tensor_jets.rs
pub(crate) mod jets {
#![allow(clippy::all, dead_code)]
use super::super::dd::Cd as C64;


/// Truncated jet algebra of one expansion kind (see `tensor.rs`).
pub(crate) trait Jet<const L: usize> {
    /// Component of the pure power `u^k`, if kept.
    const PURE: [Option<usize>; 5];
    /// `out = c x` for a scalar (`u^0`) input.
    fn first_set(out: &mut [C64; L], c: C64, x: &[C64; 4]);
    /// `out += c x` for a scalar (`u^0`) input.
    fn first_acc(out: &mut [C64; L], c: C64, x: &[C64; 4]);
    /// `out = g x` (truncated).
    fn mul_set(out: &mut [C64; L], g: &[C64; L], x: &[C64; 4]);
    /// `out += g x` (truncated).
    fn mul_acc(out: &mut [C64; L], g: &[C64; L], x: &[C64; 4]);
}

pub(crate) struct T3K0;

impl Jet<1> for T3K0 {
    const PURE: [Option<usize>; 5] = [Some(0), None, None, None, None];

    #[inline(always)]
    fn first_set(out: &mut [C64; 1], c: C64, x: &[C64; 4]) {
        out[0] = c * x[0];
    }

    #[inline(always)]
    fn first_acc(out: &mut [C64; 1], c: C64, x: &[C64; 4]) {
        out[0] += c * x[0];
    }

    #[inline(always)]
    fn mul_set(out: &mut [C64; 1], g: &[C64; 1], x: &[C64; 4]) {
        out[0] = g[0] * x[0];
    }

    #[inline(always)]
    fn mul_acc(out: &mut [C64; 1], g: &[C64; 1], x: &[C64; 4]) {
        out[0] += g[0] * x[0];
    }
}

pub(crate) struct T3K1;

impl Jet<2> for T3K1 {
    const PURE: [Option<usize>; 5] = [Some(0), Some(1), None, None, None];

    #[inline(always)]
    fn first_set(out: &mut [C64; 2], c: C64, x: &[C64; 4]) {
        out[0] = c * x[0];
        out[1] = c * x[1];
    }

    #[inline(always)]
    fn first_acc(out: &mut [C64; 2], c: C64, x: &[C64; 4]) {
        out[0] += c * x[0];
        out[1] += c * x[1];
    }

    #[inline(always)]
    fn mul_set(out: &mut [C64; 2], g: &[C64; 2], x: &[C64; 4]) {
        out[0] = g[0] * x[0];
        out[1] = g[1] * x[0];
        out[1] += g[0] * x[1];
    }

    #[inline(always)]
    fn mul_acc(out: &mut [C64; 2], g: &[C64; 2], x: &[C64; 4]) {
        out[0] += g[0] * x[0];
        out[1] += g[1] * x[0];
        out[1] += g[0] * x[1];
    }
}

pub(crate) struct T3K2;

impl Jet<4> for T3K2 {
    const PURE: [Option<usize>; 5] = [Some(0), Some(1), Some(2), None, None];

    #[inline(always)]
    fn first_set(out: &mut [C64; 4], c: C64, x: &[C64; 4]) {
        out[0] = c * x[0];
        out[1] = c * x[1];
        out[3] = c * x[2];
        out[2] = C64::ZERO;
    }

    #[inline(always)]
    fn first_acc(out: &mut [C64; 4], c: C64, x: &[C64; 4]) {
        out[0] += c * x[0];
        out[1] += c * x[1];
        out[3] += c * x[2];
    }

    #[inline(always)]
    fn mul_set(out: &mut [C64; 4], g: &[C64; 4], x: &[C64; 4]) {
        out[0] = g[0] * x[0];
        out[1] = g[1] * x[0];
        out[1] += g[0] * x[1];
        out[2] = g[2] * x[0];
        out[2] += g[1] * x[1];
        out[3] = g[3] * x[0];
        out[3] += g[0] * x[2];
    }

    #[inline(always)]
    fn mul_acc(out: &mut [C64; 4], g: &[C64; 4], x: &[C64; 4]) {
        out[0] += g[0] * x[0];
        out[1] += g[1] * x[0];
        out[1] += g[0] * x[1];
        out[2] += g[2] * x[0];
        out[2] += g[1] * x[1];
        out[3] += g[3] * x[0];
        out[3] += g[0] * x[2];
    }
}

pub(crate) struct T3K3;

impl Jet<6> for T3K3 {
    const PURE: [Option<usize>; 5] = [Some(0), Some(1), Some(2), Some(4), None];

    #[inline(always)]
    fn first_set(out: &mut [C64; 6], c: C64, x: &[C64; 4]) {
        out[0] = c * x[0];
        out[1] = c * x[1];
        out[3] = c * x[2];
        out[2] = C64::ZERO;
        out[4] = C64::ZERO;
        out[5] = C64::ZERO;
    }

    #[inline(always)]
    fn first_acc(out: &mut [C64; 6], c: C64, x: &[C64; 4]) {
        out[0] += c * x[0];
        out[1] += c * x[1];
        out[3] += c * x[2];
    }

    #[inline(always)]
    fn mul_set(out: &mut [C64; 6], g: &[C64; 6], x: &[C64; 4]) {
        out[0] = g[0] * x[0];
        out[1] = g[1] * x[0];
        out[1] += g[0] * x[1];
        out[2] = g[2] * x[0];
        out[2] += g[1] * x[1];
        out[3] = g[3] * x[0];
        out[3] += g[0] * x[2];
        out[4] = g[4] * x[0];
        out[4] += g[2] * x[1];
        out[5] = g[5] * x[0];
        out[5] += g[3] * x[1];
        out[5] += g[1] * x[2];
    }

    #[inline(always)]
    fn mul_acc(out: &mut [C64; 6], g: &[C64; 6], x: &[C64; 4]) {
        out[0] += g[0] * x[0];
        out[1] += g[1] * x[0];
        out[1] += g[0] * x[1];
        out[2] += g[2] * x[0];
        out[2] += g[1] * x[1];
        out[3] += g[3] * x[0];
        out[3] += g[0] * x[2];
        out[4] += g[4] * x[0];
        out[4] += g[2] * x[1];
        out[5] += g[5] * x[0];
        out[5] += g[3] * x[1];
        out[5] += g[1] * x[2];
    }
}

pub(crate) struct T3K4;

impl Jet<9> for T3K4 {
    const PURE: [Option<usize>; 5] = [Some(0), Some(1), Some(2), Some(4), Some(6)];

    #[inline(always)]
    fn first_set(out: &mut [C64; 9], c: C64, x: &[C64; 4]) {
        out[0] = c * x[0];
        out[1] = c * x[1];
        out[3] = c * x[2];
        out[2] = C64::ZERO;
        out[4] = C64::ZERO;
        out[5] = C64::ZERO;
        out[6] = C64::ZERO;
        out[7] = C64::ZERO;
        out[8] = C64::ZERO;
    }

    #[inline(always)]
    fn first_acc(out: &mut [C64; 9], c: C64, x: &[C64; 4]) {
        out[0] += c * x[0];
        out[1] += c * x[1];
        out[3] += c * x[2];
    }

    #[inline(always)]
    fn mul_set(out: &mut [C64; 9], g: &[C64; 9], x: &[C64; 4]) {
        out[0] = g[0] * x[0];
        out[1] = g[1] * x[0];
        out[1] += g[0] * x[1];
        out[2] = g[2] * x[0];
        out[2] += g[1] * x[1];
        out[3] = g[3] * x[0];
        out[3] += g[0] * x[2];
        out[4] = g[4] * x[0];
        out[4] += g[2] * x[1];
        out[5] = g[5] * x[0];
        out[5] += g[3] * x[1];
        out[5] += g[1] * x[2];
        out[6] = g[6] * x[0];
        out[6] += g[4] * x[1];
        out[7] = g[7] * x[0];
        out[7] += g[5] * x[1];
        out[7] += g[2] * x[2];
        out[8] = g[8] * x[0];
        out[8] += g[3] * x[2];
    }

    #[inline(always)]
    fn mul_acc(out: &mut [C64; 9], g: &[C64; 9], x: &[C64; 4]) {
        out[0] += g[0] * x[0];
        out[1] += g[1] * x[0];
        out[1] += g[0] * x[1];
        out[2] += g[2] * x[0];
        out[2] += g[1] * x[1];
        out[3] += g[3] * x[0];
        out[3] += g[0] * x[2];
        out[4] += g[4] * x[0];
        out[4] += g[2] * x[1];
        out[5] += g[5] * x[0];
        out[5] += g[3] * x[1];
        out[5] += g[1] * x[2];
        out[6] += g[6] * x[0];
        out[6] += g[4] * x[1];
        out[7] += g[7] * x[0];
        out[7] += g[5] * x[1];
        out[7] += g[2] * x[2];
        out[8] += g[8] * x[0];
        out[8] += g[3] * x[2];
    }
}

pub(crate) struct T2K0;

impl Jet<1> for T2K0 {
    const PURE: [Option<usize>; 5] = [Some(0), None, None, None, None];

    #[inline(always)]
    fn first_set(out: &mut [C64; 1], c: C64, x: &[C64; 4]) {
        out[0] = c * x[0];
    }

    #[inline(always)]
    fn first_acc(out: &mut [C64; 1], c: C64, x: &[C64; 4]) {
        out[0] += c * x[0];
    }

    #[inline(always)]
    fn mul_set(out: &mut [C64; 1], g: &[C64; 1], x: &[C64; 4]) {
        out[0] = g[0] * x[0];
    }

    #[inline(always)]
    fn mul_acc(out: &mut [C64; 1], g: &[C64; 1], x: &[C64; 4]) {
        out[0] += g[0] * x[0];
    }
}

pub(crate) struct T2K1;

impl Jet<3> for T2K1 {
    const PURE: [Option<usize>; 5] = [Some(0), Some(1), None, None, None];

    #[inline(always)]
    fn first_set(out: &mut [C64; 3], c: C64, x: &[C64; 4]) {
        out[0] = c * x[0];
        out[1] = c * x[1];
        out[2] = c * x[2];
    }

    #[inline(always)]
    fn first_acc(out: &mut [C64; 3], c: C64, x: &[C64; 4]) {
        out[0] += c * x[0];
        out[1] += c * x[1];
        out[2] += c * x[2];
    }

    #[inline(always)]
    fn mul_set(out: &mut [C64; 3], g: &[C64; 3], x: &[C64; 4]) {
        out[0] = g[0] * x[0];
        out[1] = g[1] * x[0];
        out[1] += g[0] * x[1];
        out[2] = g[2] * x[0];
        out[2] += g[0] * x[2];
    }

    #[inline(always)]
    fn mul_acc(out: &mut [C64; 3], g: &[C64; 3], x: &[C64; 4]) {
        out[0] += g[0] * x[0];
        out[1] += g[1] * x[0];
        out[1] += g[0] * x[1];
        out[2] += g[2] * x[0];
        out[2] += g[0] * x[2];
    }
}

pub(crate) struct T2K2;

impl Jet<7> for T2K2 {
    const PURE: [Option<usize>; 5] = [Some(0), Some(1), Some(3), None, None];

    #[inline(always)]
    fn first_set(out: &mut [C64; 7], c: C64, x: &[C64; 4]) {
        out[0] = c * x[0];
        out[1] = c * x[1];
        out[2] = c * x[2];
        out[6] = c * x[3];
        out[3] = C64::ZERO;
        out[4] = C64::ZERO;
        out[5] = C64::ZERO;
    }

    #[inline(always)]
    fn first_acc(out: &mut [C64; 7], c: C64, x: &[C64; 4]) {
        out[0] += c * x[0];
        out[1] += c * x[1];
        out[2] += c * x[2];
        out[6] += c * x[3];
    }

    #[inline(always)]
    fn mul_set(out: &mut [C64; 7], g: &[C64; 7], x: &[C64; 4]) {
        out[0] = g[0] * x[0];
        out[1] = g[1] * x[0];
        out[1] += g[0] * x[1];
        out[2] = g[2] * x[0];
        out[2] += g[0] * x[2];
        out[3] = g[3] * x[0];
        out[3] += g[1] * x[1];
        out[4] = g[4] * x[0];
        out[4] += g[2] * x[1];
        out[4] += g[1] * x[2];
        out[5] = g[5] * x[0];
        out[5] += g[2] * x[2];
        out[6] = g[6] * x[0];
        out[6] += g[0] * x[3];
    }

    #[inline(always)]
    fn mul_acc(out: &mut [C64; 7], g: &[C64; 7], x: &[C64; 4]) {
        out[0] += g[0] * x[0];
        out[1] += g[1] * x[0];
        out[1] += g[0] * x[1];
        out[2] += g[2] * x[0];
        out[2] += g[0] * x[2];
        out[3] += g[3] * x[0];
        out[3] += g[1] * x[1];
        out[4] += g[4] * x[0];
        out[4] += g[2] * x[1];
        out[4] += g[1] * x[2];
        out[5] += g[5] * x[0];
        out[5] += g[2] * x[2];
        out[6] += g[6] * x[0];
        out[6] += g[0] * x[3];
    }
}

pub(crate) struct T2K3;

impl Jet<13> for T2K3 {
    const PURE: [Option<usize>; 5] = [Some(0), Some(1), Some(3), Some(7), None];

    #[inline(always)]
    fn first_set(out: &mut [C64; 13], c: C64, x: &[C64; 4]) {
        out[0] = c * x[0];
        out[1] = c * x[1];
        out[2] = c * x[2];
        out[6] = c * x[3];
        out[3] = C64::ZERO;
        out[4] = C64::ZERO;
        out[5] = C64::ZERO;
        out[7] = C64::ZERO;
        out[8] = C64::ZERO;
        out[9] = C64::ZERO;
        out[10] = C64::ZERO;
        out[11] = C64::ZERO;
        out[12] = C64::ZERO;
    }

    #[inline(always)]
    fn first_acc(out: &mut [C64; 13], c: C64, x: &[C64; 4]) {
        out[0] += c * x[0];
        out[1] += c * x[1];
        out[2] += c * x[2];
        out[6] += c * x[3];
    }

    #[inline(always)]
    fn mul_set(out: &mut [C64; 13], g: &[C64; 13], x: &[C64; 4]) {
        out[0] = g[0] * x[0];
        out[1] = g[1] * x[0];
        out[1] += g[0] * x[1];
        out[2] = g[2] * x[0];
        out[2] += g[0] * x[2];
        out[3] = g[3] * x[0];
        out[3] += g[1] * x[1];
        out[4] = g[4] * x[0];
        out[4] += g[2] * x[1];
        out[4] += g[1] * x[2];
        out[5] = g[5] * x[0];
        out[5] += g[2] * x[2];
        out[6] = g[6] * x[0];
        out[6] += g[0] * x[3];
        out[7] = g[7] * x[0];
        out[7] += g[3] * x[1];
        out[8] = g[8] * x[0];
        out[8] += g[4] * x[1];
        out[8] += g[3] * x[2];
        out[9] = g[9] * x[0];
        out[9] += g[5] * x[1];
        out[9] += g[4] * x[2];
        out[10] = g[10] * x[0];
        out[10] += g[5] * x[2];
        out[11] = g[11] * x[0];
        out[11] += g[6] * x[1];
        out[11] += g[1] * x[3];
        out[12] = g[12] * x[0];
        out[12] += g[6] * x[2];
        out[12] += g[2] * x[3];
    }

    #[inline(always)]
    fn mul_acc(out: &mut [C64; 13], g: &[C64; 13], x: &[C64; 4]) {
        out[0] += g[0] * x[0];
        out[1] += g[1] * x[0];
        out[1] += g[0] * x[1];
        out[2] += g[2] * x[0];
        out[2] += g[0] * x[2];
        out[3] += g[3] * x[0];
        out[3] += g[1] * x[1];
        out[4] += g[4] * x[0];
        out[4] += g[2] * x[1];
        out[4] += g[1] * x[2];
        out[5] += g[5] * x[0];
        out[5] += g[2] * x[2];
        out[6] += g[6] * x[0];
        out[6] += g[0] * x[3];
        out[7] += g[7] * x[0];
        out[7] += g[3] * x[1];
        out[8] += g[8] * x[0];
        out[8] += g[4] * x[1];
        out[8] += g[3] * x[2];
        out[9] += g[9] * x[0];
        out[9] += g[5] * x[1];
        out[9] += g[4] * x[2];
        out[10] += g[10] * x[0];
        out[10] += g[5] * x[2];
        out[11] += g[11] * x[0];
        out[11] += g[6] * x[1];
        out[11] += g[1] * x[3];
        out[12] += g[12] * x[0];
        out[12] += g[6] * x[2];
        out[12] += g[2] * x[3];
    }
}

pub(crate) struct Mu;

impl Jet<2> for Mu {
    const PURE: [Option<usize>; 5] = [Some(0), Some(1), None, None, None];

    #[inline(always)]
    fn first_set(out: &mut [C64; 2], c: C64, x: &[C64; 4]) {
        out[0] = c * x[0];
        out[1] = c * x[1];
    }

    #[inline(always)]
    fn first_acc(out: &mut [C64; 2], c: C64, x: &[C64; 4]) {
        out[0] += c * x[0];
        out[1] += c * x[1];
    }

    #[inline(always)]
    fn mul_set(out: &mut [C64; 2], g: &[C64; 2], x: &[C64; 4]) {
        out[0] = g[0] * x[0];
        out[1] = g[1] * x[0];
        out[1] += g[0] * x[1];
    }

    #[inline(always)]
    fn mul_acc(out: &mut [C64; 2], g: &[C64; 2], x: &[C64; 4]) {
        out[0] += g[0] * x[0];
        out[1] += g[1] * x[0];
        out[1] += g[0] * x[1];
    }
}

/// Jets of the rank n+2 expansions (`tensor.rs` `horner_jet_x2`): the
/// pure-power table also keeps `u^5`.
pub(crate) trait JetX2<const L: usize>: Jet<L> {
    /// Component of the pure power `u^k`, `k <= 5`, if kept.
    const PURE_X2: [Option<usize>; 6];
}

pub(crate) struct T3K5;

impl Jet<12> for T3K5 {
    const PURE: [Option<usize>; 5] = [Some(0), Some(1), Some(2), Some(4), Some(6)];

    #[inline(always)]
    fn first_set(out: &mut [C64; 12], c: C64, x: &[C64; 4]) {
        out[0] = c * x[0];
        out[1] = c * x[1];
        out[3] = c * x[2];
        out[2] = C64::ZERO;
        out[4] = C64::ZERO;
        out[5] = C64::ZERO;
        out[6] = C64::ZERO;
        out[7] = C64::ZERO;
        out[8] = C64::ZERO;
        out[9] = C64::ZERO;
        out[10] = C64::ZERO;
        out[11] = C64::ZERO;
    }

    #[inline(always)]
    fn first_acc(out: &mut [C64; 12], c: C64, x: &[C64; 4]) {
        out[0] += c * x[0];
        out[1] += c * x[1];
        out[3] += c * x[2];
    }

    #[inline(always)]
    fn mul_set(out: &mut [C64; 12], g: &[C64; 12], x: &[C64; 4]) {
        out[0] = g[0] * x[0];
        out[1] = g[1] * x[0];
        out[1] += g[0] * x[1];
        out[2] = g[2] * x[0];
        out[2] += g[1] * x[1];
        out[3] = g[3] * x[0];
        out[3] += g[0] * x[2];
        out[4] = g[4] * x[0];
        out[4] += g[2] * x[1];
        out[5] = g[5] * x[0];
        out[5] += g[3] * x[1];
        out[5] += g[1] * x[2];
        out[6] = g[6] * x[0];
        out[6] += g[4] * x[1];
        out[7] = g[7] * x[0];
        out[7] += g[5] * x[1];
        out[7] += g[2] * x[2];
        out[8] = g[8] * x[0];
        out[8] += g[3] * x[2];
        out[9] = g[9] * x[0];
        out[9] += g[6] * x[1];
        out[10] = g[10] * x[0];
        out[10] += g[7] * x[1];
        out[10] += g[4] * x[2];
        out[11] = g[11] * x[0];
        out[11] += g[8] * x[1];
        out[11] += g[5] * x[2];
    }

    #[inline(always)]
    fn mul_acc(out: &mut [C64; 12], g: &[C64; 12], x: &[C64; 4]) {
        out[0] += g[0] * x[0];
        out[1] += g[1] * x[0];
        out[1] += g[0] * x[1];
        out[2] += g[2] * x[0];
        out[2] += g[1] * x[1];
        out[3] += g[3] * x[0];
        out[3] += g[0] * x[2];
        out[4] += g[4] * x[0];
        out[4] += g[2] * x[1];
        out[5] += g[5] * x[0];
        out[5] += g[3] * x[1];
        out[5] += g[1] * x[2];
        out[6] += g[6] * x[0];
        out[6] += g[4] * x[1];
        out[7] += g[7] * x[0];
        out[7] += g[5] * x[1];
        out[7] += g[2] * x[2];
        out[8] += g[8] * x[0];
        out[8] += g[3] * x[2];
        out[9] += g[9] * x[0];
        out[9] += g[6] * x[1];
        out[10] += g[10] * x[0];
        out[10] += g[7] * x[1];
        out[10] += g[4] * x[2];
        out[11] += g[11] * x[0];
        out[11] += g[8] * x[1];
        out[11] += g[5] * x[2];
    }
}

impl JetX2<12> for T3K5 {
    const PURE_X2: [Option<usize>; 6] = [Some(0), Some(1), Some(2), Some(4), Some(6), Some(9)];
}

pub(crate) struct T2K4;

impl Jet<22> for T2K4 {
    const PURE: [Option<usize>; 5] = [Some(0), Some(1), Some(3), Some(7), Some(13)];

    #[inline(always)]
    fn first_set(out: &mut [C64; 22], c: C64, x: &[C64; 4]) {
        out[0] = c * x[0];
        out[1] = c * x[1];
        out[2] = c * x[2];
        out[6] = c * x[3];
        out[3] = C64::ZERO;
        out[4] = C64::ZERO;
        out[5] = C64::ZERO;
        out[7] = C64::ZERO;
        out[8] = C64::ZERO;
        out[9] = C64::ZERO;
        out[10] = C64::ZERO;
        out[11] = C64::ZERO;
        out[12] = C64::ZERO;
        out[13] = C64::ZERO;
        out[14] = C64::ZERO;
        out[15] = C64::ZERO;
        out[16] = C64::ZERO;
        out[17] = C64::ZERO;
        out[18] = C64::ZERO;
        out[19] = C64::ZERO;
        out[20] = C64::ZERO;
        out[21] = C64::ZERO;
    }

    #[inline(always)]
    fn first_acc(out: &mut [C64; 22], c: C64, x: &[C64; 4]) {
        out[0] += c * x[0];
        out[1] += c * x[1];
        out[2] += c * x[2];
        out[6] += c * x[3];
    }

    #[inline(always)]
    fn mul_set(out: &mut [C64; 22], g: &[C64; 22], x: &[C64; 4]) {
        out[0] = g[0] * x[0];
        out[1] = g[1] * x[0];
        out[1] += g[0] * x[1];
        out[2] = g[2] * x[0];
        out[2] += g[0] * x[2];
        out[3] = g[3] * x[0];
        out[3] += g[1] * x[1];
        out[4] = g[4] * x[0];
        out[4] += g[2] * x[1];
        out[4] += g[1] * x[2];
        out[5] = g[5] * x[0];
        out[5] += g[2] * x[2];
        out[6] = g[6] * x[0];
        out[6] += g[0] * x[3];
        out[7] = g[7] * x[0];
        out[7] += g[3] * x[1];
        out[8] = g[8] * x[0];
        out[8] += g[4] * x[1];
        out[8] += g[3] * x[2];
        out[9] = g[9] * x[0];
        out[9] += g[5] * x[1];
        out[9] += g[4] * x[2];
        out[10] = g[10] * x[0];
        out[10] += g[5] * x[2];
        out[11] = g[11] * x[0];
        out[11] += g[6] * x[1];
        out[11] += g[1] * x[3];
        out[12] = g[12] * x[0];
        out[12] += g[6] * x[2];
        out[12] += g[2] * x[3];
        out[13] = g[13] * x[0];
        out[13] += g[7] * x[1];
        out[14] = g[14] * x[0];
        out[14] += g[8] * x[1];
        out[14] += g[7] * x[2];
        out[15] = g[15] * x[0];
        out[15] += g[9] * x[1];
        out[15] += g[8] * x[2];
        out[16] = g[16] * x[0];
        out[16] += g[10] * x[1];
        out[16] += g[9] * x[2];
        out[17] = g[17] * x[0];
        out[17] += g[10] * x[2];
        out[18] = g[18] * x[0];
        out[18] += g[11] * x[1];
        out[18] += g[3] * x[3];
        out[19] = g[19] * x[0];
        out[19] += g[12] * x[1];
        out[19] += g[11] * x[2];
        out[19] += g[4] * x[3];
        out[20] = g[20] * x[0];
        out[20] += g[12] * x[2];
        out[20] += g[5] * x[3];
        out[21] = g[21] * x[0];
        out[21] += g[6] * x[3];
    }

    #[inline(always)]
    fn mul_acc(out: &mut [C64; 22], g: &[C64; 22], x: &[C64; 4]) {
        out[0] += g[0] * x[0];
        out[1] += g[1] * x[0];
        out[1] += g[0] * x[1];
        out[2] += g[2] * x[0];
        out[2] += g[0] * x[2];
        out[3] += g[3] * x[0];
        out[3] += g[1] * x[1];
        out[4] += g[4] * x[0];
        out[4] += g[2] * x[1];
        out[4] += g[1] * x[2];
        out[5] += g[5] * x[0];
        out[5] += g[2] * x[2];
        out[6] += g[6] * x[0];
        out[6] += g[0] * x[3];
        out[7] += g[7] * x[0];
        out[7] += g[3] * x[1];
        out[8] += g[8] * x[0];
        out[8] += g[4] * x[1];
        out[8] += g[3] * x[2];
        out[9] += g[9] * x[0];
        out[9] += g[5] * x[1];
        out[9] += g[4] * x[2];
        out[10] += g[10] * x[0];
        out[10] += g[5] * x[2];
        out[11] += g[11] * x[0];
        out[11] += g[6] * x[1];
        out[11] += g[1] * x[3];
        out[12] += g[12] * x[0];
        out[12] += g[6] * x[2];
        out[12] += g[2] * x[3];
        out[13] += g[13] * x[0];
        out[13] += g[7] * x[1];
        out[14] += g[14] * x[0];
        out[14] += g[8] * x[1];
        out[14] += g[7] * x[2];
        out[15] += g[15] * x[0];
        out[15] += g[9] * x[1];
        out[15] += g[8] * x[2];
        out[16] += g[16] * x[0];
        out[16] += g[10] * x[1];
        out[16] += g[9] * x[2];
        out[17] += g[17] * x[0];
        out[17] += g[10] * x[2];
        out[18] += g[18] * x[0];
        out[18] += g[11] * x[1];
        out[18] += g[3] * x[3];
        out[19] += g[19] * x[0];
        out[19] += g[12] * x[1];
        out[19] += g[11] * x[2];
        out[19] += g[4] * x[3];
        out[20] += g[20] * x[0];
        out[20] += g[12] * x[2];
        out[20] += g[5] * x[3];
        out[21] += g[21] * x[0];
        out[21] += g[6] * x[3];
    }
}

impl JetX2<22> for T2K4 {
    const PURE_X2: [Option<usize>; 6] = [Some(0), Some(1), Some(3), Some(7), Some(13), None];
}

pub(crate) struct Mu3;

impl Jet<3> for Mu3 {
    const PURE: [Option<usize>; 5] = [Some(0), Some(1), Some(2), None, None];

    #[inline(always)]
    fn first_set(out: &mut [C64; 3], c: C64, x: &[C64; 4]) {
        out[0] = c * x[0];
        out[1] = c * x[1];
        out[2] = C64::ZERO;
    }

    #[inline(always)]
    fn first_acc(out: &mut [C64; 3], c: C64, x: &[C64; 4]) {
        out[0] += c * x[0];
        out[1] += c * x[1];
    }

    #[inline(always)]
    fn mul_set(out: &mut [C64; 3], g: &[C64; 3], x: &[C64; 4]) {
        out[0] = g[0] * x[0];
        out[1] = g[1] * x[0];
        out[1] += g[0] * x[1];
        out[2] = g[2] * x[0];
        out[2] += g[1] * x[1];
    }

    #[inline(always)]
    fn mul_acc(out: &mut [C64; 3], g: &[C64; 3], x: &[C64; 4]) {
        out[0] += g[0] * x[0];
        out[1] += g[1] * x[0];
        out[1] += g[0] * x[1];
        out[2] += g[2] * x[0];
        out[2] += g[1] * x[1];
    }
}

impl JetX2<3> for Mu3 {
    const PURE_X2: [Option<usize>; 6] = [Some(0), Some(1), Some(2), None, None, None];
}

}
pub(crate) use jets::Jet;

// ---- tensor.rs
/// Ten-build split sizes of the homogeneous rank-`d` block.
#[inline(always)]
fn splits(d: usize) -> (usize, usize, usize) {
    (d, d * (d + 1) / 2, ncoeff_homogeneous(d - 1))
}

/// Scalar Horner evaluation of `N(q)` (all blocks) or of `N_R(q)` (top block).
#[inline]
pub(super) fn horner_scalar(
    c: &[C64],
    rank: usize,
    q: &CMom,
    all_blocks: bool,
    work: &mut [C64],
) -> C64 {
    if rank == 0 {
        return c[0];
    }
    let level = ncoeff_homogeneous(rank - 1);
    let (buf_a, buf_b) = work.split_at_mut(level);
    let (mut cur, mut nxt) = (buf_a, &mut buf_b[..level]);
    let mut d = rank;
    loop {
        let (r1, r2, r3) = splits(d);
        let next = &mut nxt[..r3];
        let g: &[C64] = if d == rank {
            let off = block_offset(d);
            &c[off..off + ncoeff_homogeneous(d)]
        } else {
            &cur[..ncoeff_homogeneous(d)]
        };
        for i in 0..r3 {
            next[i] = g[1 + r1 + r2 + i] * q[3];
        }
        for i in 0..r2 {
            next[i] += g[1 + r1 + i] * q[2];
        }
        for i in 0..r1 {
            next[i] += g[1 + i] * q[1];
        }
        next[0] += g[0] * q[0];
        if all_blocks {
            let low = &c[block_offset(d - 1)..block_offset(d - 1) + r3];
            for i in 0..r3 {
                next[i] += low[i];
            }
        }
        std::mem::swap(&mut cur, &mut nxt);
        if d == 1 {
            return cur[0];
        }
        d -= 1;
    }
}

/// Component index of `u^i s^j` in a T3 jet (Ninja's t3 layout).
pub(super) const fn t3_len_const(i: usize) -> usize {
    // t3_len(i - 1) with t3_len(-1) = 0
    let mut total = 0;
    let mut k = 0;
    while k < i {
        total += 1 + k / 2;
        k += 1;
    }
    total
}

/// Offset of order `u^i` in a T2 jet.
pub(super) const fn t2_offset(i: usize) -> usize {
    let mut total = 0;
    let mut k = 0;
    while k < i {
        total += 1 + k + k / 2 + k / 3;
        k += 1;
    }
    total
}

/// Truncated-jet transposed Horner sweep; returns the jet of
/// `sum_d u^(R-d) N_d(x(u))` with `x_mu(u) = sum_s slot_s(u) xv[s][mu]`.
///
/// Level `d` maps the homogeneous block of degree `d` (ten_build order:
/// `[x0 M0] ++ [x1 M(<d)] ++ [x2 M(<d(d+1)/2)] ++ [x3 M(all)]`) onto the
/// block of degree `d-1`: output entry `i` collects at most four products,
/// so every output jet is written once per level. The lower block `d-1`
/// enters at the pure power `u^(R-d+1)` (dropped beyond truncation).
#[inline(always)]
pub(super) fn horner_jet<P: Jet<L>, const L: usize>(
    c: &[C64],
    rank: usize,
    xv: &[CMom; 4],
    work: &mut [C64],
) -> [C64; L] {
    let mut result = [C64::ZERO; L];
    if rank == 0 {
        result[0] = c[0];
        return result;
    }
    let slot = |mu: usize| -> [C64; 4] { [xv[0][mu], xv[1][mu], xv[2][mu], xv[3][mu]] };
    let xs = [slot(0), slot(1), slot(2), slot(3)];
    let level = ncoeff_homogeneous(rank - 1);
    let (buf_a, buf_b) = work.split_at_mut(level * L);
    let (mut cur, _) = buf_a.as_chunks_mut::<L>();
    let (mut nxt, _) = buf_b[..level * L].as_chunks_mut::<L>();
    // Level R: scalar coefficients times the cut-vector slots.
    {
        let d = rank;
        let off = block_offset(d);
        let (r1, r2, r3) = splits(d);
        let g = &c[off..off + ncoeff_homogeneous(d)];
        let (g1, g2, g3) = (&g[1..1 + r1], &g[1 + r1..1 + r1 + r2], &g[1 + r1 + r2..]);
        let out = &mut cur[..r3];
        for i in 0..r3 {
            P::first_set(&mut out[i], g3[i], &xs[3]);
            if i < r2 {
                P::first_acc(&mut out[i], g2[i], &xs[2]);
                if i < r1 {
                    P::first_acc(&mut out[i], g1[i], &xs[1]);
                }
            }
        }
        P::first_acc(&mut out[0], g[0], &xs[0]);
        if let Some(pure) = P::PURE[1] {
            let low = &c[block_offset(d - 1)..block_offset(d - 1) + r3];
            for i in 0..r3 {
                out[i][pure] += low[i];
            }
        }
    }
    let mut cur_len = level;
    let mut d = rank - 1;
    while d >= 1 {
        let (r1, r2, r3) = splits(d);
        let g = &cur[..cur_len];
        let (g1, g2, g3) = (&g[1..1 + r1], &g[1 + r1..1 + r1 + r2], &g[1 + r1 + r2..]);
        let out = &mut nxt[..r3];
        for i in 0..r3 {
            P::mul_set(&mut out[i], &g3[i], &xs[3]);
            if i < r2 {
                P::mul_acc(&mut out[i], &g2[i], &xs[2]);
                if i < r1 {
                    P::mul_acc(&mut out[i], &g1[i], &xs[1]);
                }
            }
        }
        P::mul_acc(&mut out[0], &g[0], &xs[0]);
        let power = rank - d + 1;
        if let Some(pure) = P::PURE.get(power).copied().flatten() {
            let low = &c[block_offset(d - 1)..block_offset(d - 1) + r3];
            for i in 0..r3 {
                out[i][pure] += low[i];
            }
        }
        cur_len = r3;
        std::mem::swap(&mut cur, &mut nxt);
        d -= 1;
    }
    result.copy_from_slice(&cur[0]);
    result
}

/// Convert the internal T3 jet (powers of `s = param + mu²`) to Ninja's
/// layout (powers of `mu²`).
#[inline(always)]
pub(super) fn t3_output<const L: usize>(
    jet: &[C64; L],
    mindeg: usize,
    param: C64,
    out: &mut [C64],
) {
    for i in 0..=mindeg {
        let o = t3_len_const(i);
        match i / 2 {
            0 => out[o] = jet[o],
            1 => {
                out[o] = jet[o] + param * jet[o + 1];
                out[o + 1] = jet[o + 1];
            }
            _ => {
                out[o] = jet[o] + param * jet[o + 1] + param * param * jet[o + 2];
                out[o + 1] = jet[o + 1] + param * jet[o + 2] * 2.;
                out[o + 2] = jet[o + 2];
            }
        }
    }
}

/// Convert the internal T2 jet (`x^k S^j`, `S = p0 + p1 x + p2 x² + mu²`) to
/// Ninja's layout.
#[inline(always)]
pub(super) fn t2_output<const L: usize>(
    jet: &[C64; L],
    mindeg: usize,
    p: &[C64; 3],
    out: &mut [C64],
) {
    for i in 0..=mindeg {
        let o = t2_offset(i);
        match i {
            0 => out[o] = jet[o],
            1 => {
                out[o] = jet[o];
                out[o + 1] = jet[o + 1];
            }
            2 => {
                // internal [1, x, x², S] -> [1, mu², x, x²]
                let (c1, cx, cx2, cs) = (jet[o], jet[o + 1], jet[o + 2], jet[o + 3]);
                out[o] = c1 + p[0] * cs;
                out[o + 1] = cs;
                out[o + 2] = cx + p[1] * cs;
                out[o + 3] = cx2 + p[2] * cs;
            }
            _ => {
                // internal [1, x, x², x³, S, xS] -> [1, mu², x, x mu², x², x³]
                let (c1, cx, cx2, cx3) = (jet[o], jet[o + 1], jet[o + 2], jet[o + 3]);
                let (cs, cxs) = (jet[o + 4], jet[o + 5]);
                out[o] = c1 + p[0] * cs;
                out[o + 1] = cs;
                out[o + 2] = cx + p[1] * cs + p[0] * cxs;
                out[o + 3] = cxs;
                out[o + 4] = cx2 + p[2] * cs + p[1] * cxs;
                out[o + 5] = cx3 + p[2] * cxs;
            }
        }
    }
}

// ---- numerator.rs
/// The expansions the double-double stages request (`FastNumerator`'s).
pub(super) trait DdNumerator {
    /// Direct evaluation `N(q, mu²)`; `out[i]` for row `i`.
    fn evaluate(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        q: &[CMom],
        mu_squared: &[C64],
        out: &mut [C64],
    ) -> Result<()>;

    /// Ninja's `muExpansion`: `q = t v0 + v1`, `mu² = t² v0²`, returning the
    /// leading coefficients in `t`. Without `v1` (rank ≤ n) one coefficient,
    /// `t^r`, is returned per row; with `v1` (rank n+1) two, `t^r, t^(r-1)`.
    fn mu_expansion(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        v0: &[CMom],
        v1: Option<&[CMom]>,
        out: &mut [C64],
    ) -> Result<()>;

    /// Ninja's `t3Expansion`; `t3_len(mindeg)` outputs per row.
    #[allow(clippy::too_many_arguments)]
    fn t3_expansion(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        a: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[C64],
        mindeg: usize,
        out: &mut [C64],
    ) -> Result<()>;

    /// Ninja's `t2Expansion`; `t2_len(mindeg)` outputs per row.
    #[allow(clippy::too_many_arguments)]
    fn t2_expansion(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        a0: &[CMom],
        a1: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[[C64; 3]],
        mindeg: usize,
        out: &mut [C64],
    ) -> Result<()>;
}

// ---- engine.rs
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct TriangleData {
    pub(super) e3: CMom,
    pub(super) e4: CMom,
    pub(super) c: [C64; 15],
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct BubbleData {
    pub(super) e2: RMom,
    pub(super) e3: CMom,
    pub(super) e4: CMom,
    pub(super) c: [C64; 20],
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct TadpoleData {
    pub(super) c0: C64,
    pub(super) c14: C64,
    pub(super) c15: C64,
    pub(super) e34: Dd,
}

/// Scalar products of one uncut propagator `j` with the expansion vectors of
/// the current cut, shared by the Laurent division and the subtractions.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Uncut {
    /// `e3.(V_j - V_0)` and its reciprocal (`2 e3k` is the division's `d0`).
    pub(super) e3k: C64,
    pub(super) ie3k: C64,
    /// `e4.(V_j - V_0)` (unscaled `e4`; `e4eff` products follow by scaling).
    pub(super) e4k: C64,
    /// Bubble `a0 = e1effl` and `a1 = e2eff` products.
    pub(super) e1k: C64,
    pub(super) e2k: Dd,
    /// Constant part of the division's `d1`.
    pub(super) d1: C64,
    /// `(V_j - V_0)² + m0² - m_j²`.
    pub(super) f: C64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct BoxRow {
    lane: usize,
    vort: CMom,
    q: [CMom; 2],
    mid: CMom,
}

/// `D = (q+V)² - m²` (Ninja's `Den` at `mu² = 0`).
#[inline(always)]
pub(super) fn den(q: &CMom, v: &RMom, m2: C64) -> C64 {
    let s = [q[0] + v[0], q[1] + v[1], q[2] + v[2], q[3] + v[3]];
    mp_cc(&s, &s) - m2
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct TriangleRow {
    lane: usize,
    e3: CMom,
    e4: CMom,
    amu: CMom,
    param: C64,
    e34: Dd,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct BubbleRow {
    lane: usize,
    e: Basis,
    /// Ninja's `e1effl = b0 e2` (before the `-V0` shift) and `e2eff = e1 + b1 e2`.
    b0: C64,
    b1: Dd,
    e1eff: CMom,
    e2eff: RMom,
    param: [C64; 3],
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct TadpoleRow {
    lane: usize,
    basis: usize,
}

/// Lane-major staging of numerator callback arguments.
#[derive(Default)]
pub(super) struct Calls {
    lanes: Vec<usize>,
    v0: Vec<CMom>,
    v1: Vec<CMom>,
    v2: Vec<CMom>,
    v3: Vec<CMom>,
    scalar: Vec<C64>,
    params: Vec<[C64; 3]>,
    out: Vec<C64>,
}

impl Calls {
    pub(super) fn clear(&mut self) {
        self.lanes.clear();
        self.v0.clear();
        self.v1.clear();
        self.v2.clear();
        self.v3.clear();
        self.scalar.clear();
        self.params.clear();
    }
    pub(super) fn prepare_out(&mut self, width: usize) {
        self.out.clear();
        self.out.resize(self.lanes.len() * width, C64::ZERO);
    }
}

#[inline(always)]
pub(super) fn sub_cr(a: &CMom, b: &RMom) -> CMom {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

#[inline(always)]
pub(super) fn scale_c(a: &CMom, s: Dd) -> CMom {
    [a[0] * s, a[1] * s, a[2] * s, a[3] * s]
}

#[inline(always)]
pub(super) fn is_zero(z: C64) -> bool {
    z.is_zero()
}

/// Polynomial division of a triangle/tadpole Laurent expansion
/// (`divpolyby<3>`) by `d0 t + d1 + (d2[0] + d2[1] mu²)/t`, given `1/d0`.
#[inline(always)]
pub(super) fn divide_t3(num: &mut [C64], nterms: usize, d0inv: C64, d1: C64, d2: [C64; 2]) {
    match nterms {
        1 => divide_t3_n::<1>(num, d0inv, d1, d2),
        2 => divide_t3_n::<2>(num, d0inv, d1, d2),
        3 => divide_t3_n::<3>(num, d0inv, d1, d2),
        4 => divide_t3_n::<4>(num, d0inv, d1, d2),
        _ => divide_t3_n::<5>(num, d0inv, d1, d2),
    }
}

/// [`divide_t3`] for a compile-time number of orders (fully unrolled).
#[inline(always)]
fn divide_t3_n<const NT: usize>(num: &mut [C64], d0inv: C64, d1: C64, d2: [C64; 2]) {
    const OFF: [usize; 6] = [0, 1, 2, 4, 6, 9];
    let num = &mut num[..OFF[NT]];
    for i in 0..NT {
        let size = 1 + i / 2;
        let o = OFF[i];
        for j in 0..size {
            num[o + j] *= d0inv;
        }
        if i + 1 < NT {
            let o1 = OFF[i + 1];
            for j in 0..size {
                let v = num[o + j];
                num[o1 + j] -= d1 * v;
            }
            if i + 2 < NT {
                let o2 = OFF[i + 2];
                for j in 0..size {
                    let v = num[o + j];
                    num[o2 + j] -= d2[0] * v;
                    num[o2 + j + 1] -= d2[1] * v;
                }
            }
        }
    }
}

/// Polynomial division of a bubble Laurent expansion (`divpolyby<2>`) by
/// `d0 t + (d1[0] + d1[1] x) + (d2[0] + d2[1] mu² + d2[2] x + d2[3] x²)/t`,
/// given `1/d0`.
#[inline(always)]
pub(super) fn divide_t2(num: &mut [C64], nterms: usize, d0inv: C64, d1: [C64; 2], d2: [C64; 4]) {
    match nterms {
        1 => divide_t2_n::<1>(num, d0inv, d1, d2),
        2 => divide_t2_n::<2>(num, d0inv, d1, d2),
        3 => divide_t2_n::<3>(num, d0inv, d1, d2),
        _ => divide_t2_n::<4>(num, d0inv, d1, d2),
    }
}

/// [`divide_t2`] for a compile-time number of orders (fully unrolled).
#[inline(always)]
fn divide_t2_n<const NT: usize>(num: &mut [C64], d0inv: C64, d1: [C64; 2], d2: [C64; 4]) {
    const OFF: [usize; 5] = [0, 1, 3, 7, 13];
    let num = &mut num[..OFF[NT]];
    for i in 0..NT {
        let size = 1 + i + i / 2 + i / 3;
        let o = OFF[i];
        for j in 0..size {
            num[o + j] *= d0inv;
        }
        if i + 1 < NT {
            let n1 = OFF[i + 1];
            let v0 = num[o];
            num[n1] -= d1[0] * v0;
            match i + 1 {
                1 => num[n1 + 1] -= d1[1] * v0,
                2 => {
                    let v1 = num[o + 1];
                    num[n1 + 2] -= d1[0] * v1;
                    num[n1 + 2] -= d1[1] * v0;
                    num[n1 + 3] -= d1[1] * v1;
                }
                _ => {
                    let (v1, v2, v3) = (num[o + 1], num[o + 2], num[o + 3]);
                    num[n1 + 1] -= d1[0] * v1;
                    num[n1 + 2] -= d1[0] * v2;
                    num[n1 + 4] -= d1[0] * v3;
                    num[n1 + 2] -= d1[1] * v0;
                    num[n1 + 3] -= d1[1] * v1;
                    num[n1 + 4] -= d1[1] * v2;
                    num[n1 + 5] -= d1[1] * v3;
                }
            }
            if i + 2 < NT {
                let n2 = OFF[i + 2];
                num[n2] -= d2[0] * v0;
                if i == 0 {
                    num[n2 + 1] -= d2[1] * v0;
                    num[n2 + 2] -= d2[2] * v0;
                    num[n2 + 3] -= d2[3] * v0;
                } else {
                    let v1 = num[o + 1];
                    num[n2 + 1] -= d2[1] * v0;
                    num[n2 + 2] -= d2[2] * v0;
                    num[n2 + 4] -= d2[3] * v0;
                    num[n2 + 2] -= d2[0] * v1;
                    num[n2 + 3] -= d2[1] * v1;
                    num[n2 + 4] -= d2[2] * v1;
                    num[n2 + 5] -= d2[3] * v1;
                }
            }
        }
    }
}

type BubbleCorrection = fn(
    &mut [C64],
    &CMom,
    &CMom,
    &[C64],
    &CMom,
    &CMom,
    &[C64; 3],
    C64,
    Dd,
    C64,
    C64,
    C64,
    &[C64; 4],
    C64,
);

/// `cuts::correcttadcoeffs` (triangle parent) for rank <= n. The cubic
/// triangle coefficients exist only for rank = n. `k0`, `k1` carry the
/// tadpole's precomputed products with the two uncut propagators.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn tadpole_from_triangle(
    divnum: &mut [C64],
    et3: &CMom,
    et4: &CMom,
    c: &[C64],
    e3: &CMom,
    k0: &Uncut,
    k1: &Uncut,
    rmn: isize,
) {
    let (e3k0, e3k1) = (k0.e3k, k1.e3k);
    let e3t3 = mp_cc(e3, et3);
    let e3t4 = mp_cc(e3, et4);
    let e3t3q = e3t3 * e3t3;
    let e3t4q = e3t4 * e3t4;
    let mut numerator = c[2] * 2. * e3k0 * e3k1 * e3t3q + c[5] * 2. * e3k0 * e3k1 * e3t4q;
    if rmn >= 0 {
        let e3t3c = e3t3q * e3t3;
        let e3t4c = e3t4q * e3t4;
        numerator -= (c[3] * e3t3c + c[6] * e3t4c) * (e3k1 * k0.f + e3k0 * k1.f);
    }
    let inverse = k0.ie3k * k0.ie3k * (k1.ie3k * k1.ie3k);
    divnum[(rmn + 1) as usize] -= numerator * 0.125 * inverse;
}

/// `cuts::correcttadcoeffs` (bubble parent) for rank <= n. The quadratic
/// bubble coefficients exist only for rank = n.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn tadpole_from_bubble(
    divnum: &mut [C64],
    eb2: &RMom,
    eb3: &CMom,
    eb4: &CMom,
    kbol: &RMom,
    b: &[C64],
    e3: &CMom,
    k: &Uncut,
    rmn: isize,
) {
    let e3b2 = mp_cr(e3, eb2);
    let e3b3 = mp_cc(e3, eb3);
    let e3b4 = mp_cc(e3, eb4);
    let e3k = k.e3k;
    let value = if rmn >= 0 {
        let kbolb2 = mp_rr(eb2, kbol);
        let abbr9 = b[2] * e3b2 * e3b2
            + b[7] * e3b2 * e3b3
            + b[4] * e3b3 * e3b3
            + b[8] * e3b2 * e3b4
            + b[6] * e3b4 * e3b4;
        -(abbr9 * k.f)
            + e3k
                * 2.
                * (b[1] * e3b2
                    + b[3] * e3b3
                    + b[5] * e3b4
                    + b[2] * 2. * e3b2 * kbolb2
                    + b[7] * e3b3 * kbolb2
                    + b[8] * e3b4 * kbolb2)
    } else {
        e3k * 2. * (b[1] * e3b2 + b[3] * e3b3 + b[5] * e3b4)
    };
    divnum[(rmn + 1) as usize] -= value * 0.25 * (k.ie3k * k.ie3k);
}

impl DdStages {
    pub(super) fn boxes<N: DdNumerator + ?Sized>(&mut self, input: &FastInput, num: &mut N) -> Result<()> {
        let n = self.topo.n;
        let lanes = self.lanes;
        let eps = self.options.infrared_eps;
        let check = Stability::new(&self.options);
        let x1 = self.topo.x1();
        let with_mu = self.topo.rank >= n;
        for b in 0..self.topo.boxes.len() {
            self.refresh_active();
            if self.active.is_empty() {
                return Ok(());
            }
            let p = self.topo.boxes[b];
            let [c1, c2, c3, c4] = p;
            self.box_rows.clear();
            self.calls.clear();
            for ai in 0..self.active.len() {
                let lane = self.active[ai];
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let k0 = sub(&v[c2], &v[c1]);
                let k1 = sub(&v[c3], &v[c2]);
                let k3 = sub(&v[c1], &v[c4]);
                if !check.relative(gram_det(&k3, &k0).abs(), gram_scale(&k3, &k0)) {
                    fail(
                        &mut self.alive,
                        &mut self.lane_status,
                        lane,
                        InstabilityTest::BoxGram,
                    );
                    continue;
                }
                let e = Basis::from_pair(&k3, &k0, eps);
                let p1e4 = mp_cr(&e.e4, &k1);
                let p1e3 = mp_cr(&e.e3, &k1);
                let mut vort = [C64::ZERO; 4];
                for mu in 0..4 {
                    vort[mu] = p1e4 * e.e3[mu] - p1e3 * e.e4[mu];
                }
                // mp2(Vort) = -2 (p1.e3) (p1.e4) e3.e4 vanishes exactly when the box
                // is degenerate (a light-like or vanishing normal): test the two
                // factors relative to their own scales.
                if !check.absolute(mp_cc(&vort, &vort).taxicab())
                    || !check.relative(p1e3.taxicab(), dot_scale(&e.e3, &k1))
                    || !check.relative(p1e4.taxicab(), dot_scale(&e.e4, &k1))
                {
                    fail(
                        &mut self.alive,
                        &mut self.lane_status,
                        lane,
                        euclidean_or(InstabilityTest::BoxNormal, &e.e3, &k3, &k0),
                    );
                    continue;
                }
                let l = box_solutions(&k3, &k0, &k1, &e, m[c1], m[c2], m[c3], m[c4]);
                let q = [sub_cr(&l[0], &v[c1]), sub_cr(&l[1], &v[c1])];
                let norm = |x: &CMom| x.iter().map(|z| z.norm_sqr()).sum::<Dd>();
                let scale = p
                    .iter()
                    .map(|&i| v[i].iter().map(|x| *x * *x).sum::<Dd>().max(m[i].abs()))
                    .fold(Dd::ZERO, Dd::max);
                self.box_lambda[b * lanes + lane] =
                    ((norm(&l[0]) + norm(&l[1])) / (scale * 2.)).to_f64().log2();
                let mut mid = [C64::ZERO; 4];
                if x1 {
                    for mu in 0..4 {
                        mid[mu] = (l[0][mu] + l[1][mu]) * 0.5 - v[c1][mu];
                    }
                }
                self.box_rows.push(BoxRow { lane, vort, q, mid });
                for qq in q {
                    self.calls.lanes.push(lane);
                    self.calls.v0.push(qq);
                    self.calls.scalar.push(C64::ZERO);
                }
            }
            if self.box_rows.is_empty() {
                continue;
            }
            self.calls.prepare_out(1);
            num.evaluate(
                &p,
                &self.calls.lanes,
                &self.calls.v0,
                &self.calls.scalar,
                &mut self.calls.out,
            )?;
            let cp = &self.topo.box_complements[b * (n - 4)..(b + 1) * (n - 4)];
            let mut keep = 0;
            for r in 0..self.box_rows.len() {
                let row = self.box_rows[r];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let (vmax, mmax) = self.bounds[lane];
                let mut values = [C64::ZERO; 2];
                let mut stable = true;
                for s in 0..2 {
                    let mut d = C64::ONE;
                    let mut singular = false;
                    let bound = mmax
                        + row.q[s]
                            .iter()
                            .map(|z| (z.taxicab() + vmax).powi(2))
                            .sum::<Dd>();
                    for &j in cp {
                        let factor = den(&row.q[s], &v[j], m[j]);
                        singular |= !check.relative_within(factor.taxicab(), bound, || {
                            den_scale(&row.q[s], &v[j], m[j])
                        });
                        d *= factor;
                    }
                    if singular || !check.absolute(d.taxicab()) {
                        stable = false;
                        break;
                    }
                    values[s] = self.calls.out[2 * r + s] / d;
                }
                if !stable {
                    fail(
                        &mut self.alive,
                        &mut self.lane_status,
                        lane,
                        InstabilityTest::BoxCutDenominator,
                    );
                    continue;
                }
                self.box_c[b * lanes + lane] = [(values[0] + values[1]) * 0.5, C64::ZERO];
                self.box_rows[keep] = row;
                keep += 1;
            }
            self.box_rows.truncate(keep);
            if !with_mu || self.box_rows.is_empty() {
                continue;
            }
            // Rational term from the mu-expansion.
            self.calls.clear();
            for row in &self.box_rows {
                self.calls.lanes.push(row.lane);
                self.calls.v0.push(row.vort);
                if x1 {
                    self.calls.v1.push(row.mid);
                }
            }
            self.calls.prepare_out(if x1 { 2 } else { 1 });
            num.mu_expansion(
                &p,
                &self.calls.lanes,
                &self.calls.v0,
                if x1 { Some(&self.calls.v1) } else { None },
                &mut self.calls.out,
            )?;
            // Rows whose geometric loss (`crate::conditioning`; at rank n
            // its cheap upper bound) exceeds the budget: (row, |n0|, |n1|,
            // the exact geometry above rank n), decided in
            // `box_mu_conditioning`.
            for r in 0..self.box_rows.len() {
                let row = self.box_rows[r];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let vort2 = mp_cc(&row.vort, &row.vort);
                let bound = 4. * self.bounds[lane].0 * taxicab_sum(&row.vort);
                if !x1 {
                    let mut d = C64::ONE;
                    let mut singular = false;
                    for &j in cp {
                        let factor = mp_cr(&row.vort, &sub(&v[j], &v[c1])) * 2.;
                        singular |= !check.relative_within(factor.taxicab(), bound, || {
                            difference_scale(&row.vort, &v[j], &v[c1])
                        });
                        d *= factor;
                    }
                    if singular || !check.absolute(d.taxicab()) {
                        fail(
                            &mut self.alive,
                            &mut self.lane_status,
                            lane,
                            InstabilityTest::BoxMuDenominator,
                        );
                        continue;
                    }
                    self.box_c[b * lanes + lane][1] = self.calls.out[r] / d / (vort2 * vort2);
                } else {
                    let mut nmu = [self.calls.out[2 * r], self.calls.out[2 * r + 1]];
                    let vsq = &self.vsq[lane * n..(lane + 1) * n];
                    let mut stable = true;
                    // Euclidean scale and the solutions' distance from the
                    // center along the normal (`crate::conditioning`).
                    for &j in cp {
                        let kk = sub(&v[j], &v[c1]);
                        let d0 = mp_cr(&row.vort, &kk) * 2.;
                        if !check.relative_within(d0.taxicab(), bound, || {
                            difference_scale(&row.vort, &v[j], &v[c1])
                        }) {
                            stable = false;
                            break;
                        }
                        let d1 =
                            C64::real(vsq[j] - vsq[c1]) - m[j] + m[c1] + mp_cr(&row.mid, &kk) * 2.;
                        nmu[0] /= d0;
                        nmu[1] -= d1 * nmu[0];
                        nmu[1] /= d0;
                    }
                    if !stable {
                        fail(
                            &mut self.alive,
                            &mut self.lane_status,
                            lane,
                            InstabilityTest::BoxMuExpansion,
                        );
                        continue;
                    }
                    self.box_c[b * lanes + lane][1] = nmu[1] / (vort2 * vort2);
                }
            }
        }
        Ok(())
    }

    pub(super) fn triangles<N: DdNumerator + ?Sized>(
        &mut self,
        input: &FastInput,
        num: &mut N,
    ) -> Result<()> {
        let n = self.topo.n;
        let lanes = self.lanes;
        let eps = self.options.infrared_eps;
        let check = Stability::new(&self.options);
        let rmn = self.topo.rmn();
        let x1 = self.topo.x1();
        let mindeg = (rmn + 3) as usize;
        let width = t3_len(mindeg);
        let nterms = mindeg + 1;
        let (lin, quad, cub) = (rmn >= -2, rmn >= -1, rmn >= 0);
        for t in 0..self.topo.triangles.len() {
            self.refresh_active();
            if self.active.is_empty() {
                return Ok(());
            }
            let p = self.topo.triangles[t];
            let [c1, c2, c3] = p;
            self.tri_rows.clear();
            for ai in 0..self.active.len() {
                let lane = self.active[ai];
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let k0 = sub(&v[c2], &v[c1]);
                let k2 = sub(&v[c1], &v[c3]);
                if !check.relative(gram_det(&k2, &k0).abs(), gram_scale(&k2, &k0)) {
                    fail(
                        &mut self.alive,
                        &mut self.lane_status,
                        lane,
                        InstabilityTest::TriangleGram,
                    );
                    continue;
                }
                let e = Basis::from_pair(&k2, &k0, eps);
                let (y1, y2, param) = triangle_solution(&k2, &k0, &e, m[c1], m[c2], m[c3]);
                let amu = sub_cr(&combine_real2(y1, &e.e1, y2, &e.e2), &v[c1]);
                self.tri_rows.push(TriangleRow {
                    lane,
                    e3: e.e3,
                    e4: e.e4,
                    amu,
                    param,
                    e34: e.mp34(),
                });
            }
            if self.tri_rows.is_empty() {
                continue;
            }
            // Both orientations share one callback: rows [e3 side..., e4 side...].
            self.calls.clear();
            for orientation in 0..2 {
                for row in &self.tri_rows {
                    let h = 0.5 / row.e34;
                    self.calls.lanes.push(row.lane);
                    self.calls.v0.push(row.amu);
                    if orientation == 0 {
                        self.calls.v1.push(row.e3);
                        self.calls.v2.push(scale_c(&row.e4, h));
                    } else {
                        self.calls.v1.push(row.e4);
                        self.calls.v2.push(scale_c(&row.e3, h));
                    }
                    self.calls.scalar.push(row.param);
                }
            }
            self.calls.prepare_out(width);
            num.t3_expansion(
                &p,
                &self.calls.lanes,
                &self.calls.v0,
                &self.calls.v1,
                &self.calls.v2,
                &self.calls.scalar,
                mindeg,
                &mut self.calls.out,
            )?;
            let rows = self.tri_rows.len();
            let cp = &self.topo.triangle_complements[t * (n - 3)..(t + 1) * (n - 3)];
            let (head, tail) = self.calls.out.split_at_mut(rows * width);
            for r in 0..rows {
                let row = &self.tri_rows[r];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let vsq = &self.vsq[lane * n..(lane + 1) * n];
                let num1 = &mut head[r * width..(r + 1) * width];
                let num2 = &mut tail[r * width..(r + 1) * width];
                let inv34 = 1. / row.e34;
                let vmax4 = 4. * self.bounds[lane].0;
                let (bound3, bound4) = (vmax4 * taxicab_sum(&row.e3), vmax4 * taxicab_sum(&row.e4));
                let mut stable = true;
                for &j in cp {
                    let pdiff = sub(&v[j], &v[c1]);
                    let e3p = mp_cr(&row.e3, &pdiff);
                    let e4p = mp_cr(&row.e4, &pdiff);
                    let (d0a, d0b) = (e3p * 2., e4p * 2.);
                    if !check.relative_within(d0a.taxicab(), bound3, || {
                        difference_scale(&row.e3, &v[j], &v[c1])
                    }) || !check.relative_within(d0b.taxicab(), bound4, || {
                        difference_scale(&row.e4, &v[j], &v[c1])
                    }) {
                        stable = false;
                        break;
                    }
                    // Only the terms the truncated division reads are formed.
                    let d1 = if nterms > 1 {
                        mp_cr(&row.amu, &pdiff) * 2. + (vsq[j] - vsq[c1]) - m[j] + m[c1]
                    } else {
                        C64::ZERO
                    };
                    let (d2a, d2b) = if nterms > 2 {
                        let (a, b) = (e4p * inv34, e3p * inv34);
                        ([row.param * a, a], [row.param * b, b])
                    } else {
                        ([C64::ZERO; 2], [C64::ZERO; 2])
                    };
                    divide_t3(num1, nterms, d0a.inv(), d1, d2a);
                    divide_t3(num2, nterms, d0b.inv(), d1, d2b);
                }
                if !stable {
                    let test = euclidean_or(
                        InstabilityTest::TriangleExpansion,
                        &row.e3,
                        &sub(&v[c1], &v[c3]),
                        &sub(&v[c2], &v[c1]),
                    );
                    fail(&mut self.alive, &mut self.lane_status, lane, test);
                    continue;
                }
                let mut c = [C64::ZERO; 15];
                if !x1 {
                    let jext3 = rmn;
                    let jext2 = rmn + 1;
                    let jext1 = rmn + 2;
                    let jext1mu2 = 3;
                    let jext0 = (rmn + 3) as usize + usize::from(cub);
                    let jext0mu2 = if cub { 5 } else { 3 };
                    c[0] = (num1[jext0] + num2[jext0]) * 0.5;
                    if lin {
                        c[4] = num1[jext1 as usize] * inv34;
                        c[1] = num2[jext1 as usize] * inv34;
                    }
                    if quad {
                        let i2 = inv34 * inv34;
                        c[5] = num1[jext2 as usize] * i2;
                        c[2] = num2[jext2 as usize] * i2;
                        c[7] = (num1[jext0mu2] + num2[jext0mu2]) * 0.5;
                    }
                    if cub {
                        let i3 = inv34 * inv34 * inv34;
                        c[6] = num1[jext3 as usize] * i3;
                        c[3] = num2[jext3 as usize] * i3;
                        c[9] = num1[jext1mu2] * inv34;
                        c[8] = num2[jext1mu2] * inv34;
                    }
                } else {
                    let i2 = inv34 * inv34;
                    let i3 = i2 * inv34;
                    let i4 = i2 * i2;
                    c[0] = (num1[6] + num2[6]) * 0.5;
                    c[4] = num1[4] * inv34;
                    c[1] = num2[4] * inv34;
                    c[5] = num1[2] * i2;
                    c[2] = num2[2] * i2;
                    c[7] = (num1[7] + num2[7]) * 0.5;
                    c[6] = num1[1] * i3;
                    c[3] = num2[1] * i3;
                    c[9] = num1[5] * inv34;
                    c[8] = num2[5] * inv34;
                    c[13] = num1[0] * i4;
                    c[12] = num2[0] * i4;
                    c[11] = num1[3] * i2;
                    c[10] = num2[3] * i2;
                    c[14] = (num1[8] + num2[8]) * 0.5;
                }
                self.tri[t * lanes + lane] = TriangleData {
                    e3: row.e3,
                    e4: row.e4,
                    c,
                };
            }
        }
        Ok(())
    }

    pub(super) fn bubbles<N: DdNumerator + ?Sized>(&mut self, input: &FastInput, num: &mut N) -> Result<()> {
        let n = self.topo.n;
        let lanes = self.lanes;
        let eps = self.options.infrared_eps;
        let check = Stability::new(&self.options);
        let rmn = self.topo.rmn();
        let x1 = self.topo.x1();
        let mindeg = (rmn + 2) as usize;
        let width = t2_len_le3(mindeg);
        let nterms = mindeg + 1;
        let (lin, quad) = (rmn >= -1, rmn >= 0);
        // Triangle parents exist only when the triangle stage ran.
        let (correct_all, correct_spurious): (Option<BubbleCorrection>, Option<BubbleCorrection>) =
            if !self.any_triangle() {
                (None, None)
            } else {
                match rmn {
                    -2 => (Some(subs::bubble_from_triangle_rm2_all), None),
                    -1 => (
                        Some(subs::bubble_from_triangle_rm1_all),
                        Some(subs::bubble_from_triangle_rm1_spurious),
                    ),
                    0 => (
                        Some(subs::bubble_from_triangle_r0_all),
                        Some(subs::bubble_from_triangle_r0_spurious),
                    ),
                    _ => (
                        Some(subs::bubble_from_triangle_x1_all),
                        Some(subs::bubble_from_triangle_x1_spurious),
                    ),
                }
            };
        for b in 0..self.topo.bubbles.len() {
            self.refresh_active();
            if self.active.is_empty() {
                return Ok(());
            }
            let p = self.topo.bubbles[b];
            let [c1, c2] = p;
            self.bub_rows.clear();
            self.calls.clear();
            for ai in 0..self.active.len() {
                let lane = self.active[ai];
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let s = self.smat[lane * n * n + c2 * n + c1];
                // Zero-initialized like a fresh Ninja CutsVector entry.
                self.bub[b * lanes + lane] = BubbleData::default();
                if is_zero(m[c1]) && is_zero(m[c2]) && s.abs() < eps {
                    continue;
                }
                let k = sub(&v[c1], &v[c2]);
                let e = Basis::from_single(&k, eps);
                let (b0, b1, param) = bubble_solution(&k, &e, m[c1], m[c2]);
                let e1effl = [b0 * e.e2[0], b0 * e.e2[1], b0 * e.e2[2], b0 * e.e2[3]];
                let e2eff = [
                    e.e1[0] + b1 * e.e2[0],
                    e.e1[1] + b1 * e.e2[1],
                    e.e1[2] + b1 * e.e2[2],
                    e.e1[3] + b1 * e.e2[3],
                ];
                let e1eff = sub_cr(&e1effl, &v[c1]);
                self.calls.lanes.push(lane);
                self.calls.v0.push(e1eff);
                self.calls.v1.push(cmom_from_real(&e2eff));
                self.calls.v2.push(e.e3);
                self.calls.v3.push(scale_c(&e.e4, 0.5 / e.mp34()));
                self.calls.params.push(param);
                self.bub_rows.push(BubbleRow {
                    lane,
                    e,
                    b0,
                    b1,
                    e1eff,
                    e2eff,
                    param,
                });
            }
            if self.bub_rows.is_empty() {
                continue;
            }
            self.calls.prepare_out(width);
            num.t2_expansion(
                &p,
                &self.calls.lanes,
                &self.calls.v0,
                &self.calls.v1,
                &self.calls.v2,
                &self.calls.v3,
                &self.calls.params,
                mindeg,
                &mut self.calls.out,
            )?;
            let cp = &self.topo.bubble_complements[b * (n - 2)..(b + 1) * (n - 2)];
            let parents = &self.topo.bubble_parents[b];
            self.second.clear();
            for r in 0..self.bub_rows.len() {
                let row = &self.bub_rows[r];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let vsq = &self.vsq[lane * n..(lane + 1) * n];
                let numexp = &mut self.calls.out[r * width..(r + 1) * width];
                let e = &row.e;
                let inv34 = 1. / e.mp34();
                let e4s = 0.5 * inv34;
                let uncut = &mut self.uncut;
                let mut stable = true;
                for &j in cp {
                    let pdiff = sub(&v[j], &v[c1]);
                    let e3k = mp_cr(&e.e3, &pdiff);
                    if !check.absolute(e3k.taxicab() * 2.) {
                        stable = false;
                        break;
                    }
                    // a0 = b0 e2 and a1 = e1 + b1 e2 with real e1, e2.
                    let e2p = mp_rr(&e.e2, &pdiff);
                    let e1k = row.b0 * e2p;
                    uncut[j] = Uncut {
                        e3k,
                        ie3k: e3k.inv(),
                        e4k: mp_cr(&e.e4, &pdiff),
                        e1k,
                        e2k: mp_rr(&e.e1, &pdiff) + row.b1 * e2p,
                        d1: (e1k - mp_rr(&v[c1], &pdiff)) * 2.
                            + (vsq[j] - vsq[c1])
                            + (-m[j] + m[c1]),
                        f: mp2(&pdiff) + (m[c1] - m[j]),
                    };
                }
                if !stable {
                    fail(
                        &mut self.alive,
                        &mut self.lane_status,
                        lane,
                        InstabilityTest::BubbleExpansion,
                    );
                    continue;
                }
                for &j in cp {
                    let u = &uncut[j];
                    let d2 = u.e4k * inv34;
                    divide_t2(
                        numexp,
                        nterms,
                        u.ie3k * 0.5,
                        [u.d1, C64::real(2. * u.e2k)],
                        [row.param[0] * d2, d2, row.param[1] * d2, row.param[2] * d2],
                    );
                }
                if let Some(all) = correct_all {
                    let e4eff = scale_c(&e.e4, e4s);
                    let np = parents.len();
                    if self.pairs.len() < (r + 1) * np {
                        self.pairs.resize((r + 1) * np, [C64::ZERO; 4]);
                    }
                    for (pi, &(t, unc)) in parents.iter().enumerate() {
                        let tri = &self.tri[t * lanes + lane];
                        let u = &uncut[unc];
                        // Shared with the second orientation: a0 = b0 e2,
                        // a1 = e1 + b1 e2 against the triangle's e3, e4.
                        let (e2t3, e2t4) = (mp_cr(&tri.e3, &e.e2), mp_cr(&tri.e4, &e.e2));
                        let pair = [
                            row.b0 * e2t3,
                            row.b0 * e2t4,
                            mp_cr(&tri.e3, &e.e1) + e2t3 * row.b1,
                            mp_cr(&tri.e4, &e.e1) + e2t4 * row.b1,
                        ];
                        self.pairs[r * np + pi] = pair;
                        all(
                            numexp,
                            &tri.e3,
                            &tri.e4,
                            &tri.c,
                            &e.e3,
                            &e4eff,
                            &row.param,
                            u.e1k,
                            u.e2k,
                            u.e3k,
                            u.ie3k,
                            u.e4k * e4s,
                            &pair,
                            u.f,
                        );
                    }
                }
                let inv12 = 1. / e.mp12;
                let data = &mut self.bub[b * lanes + lane];
                data.e2 = e.e2;
                data.e3 = e.e3;
                data.e4 = e.e4;
                let c = &mut data.c;
                if !x1 {
                    let jext2 = rmn;
                    let jext1 = rmn + 1;
                    let jext1x1 = 2;
                    let jext0 = (rmn + 2) as usize + usize::from(quad);
                    let jext0mu2 = 4;
                    let jext0x1 = if quad { 5 } else { 2 };
                    let jext0x2 = 6;
                    c[0] = numexp[jext0];
                    if lin {
                        c[1] = numexp[jext0x1] * inv12;
                        c[5] = numexp[jext1 as usize] * inv34;
                    }
                    if quad {
                        c[9] = numexp[jext0mu2];
                        c[2] = numexp[jext0x2] * (inv12 * inv12);
                        c[6] = numexp[jext2 as usize] * (inv34 * inv34);
                        c[8] = numexp[jext1x1] * (inv12 * inv34);
                    }
                } else {
                    c[0] = numexp[7];
                    c[1] = numexp[9] * inv12;
                    c[5] = numexp[3] * inv34;
                    c[9] = numexp[8];
                    c[2] = numexp[11] * (inv12 * inv12);
                    c[6] = numexp[1] * (inv34 * inv34);
                    c[8] = numexp[5] * (inv12 * inv34);
                    c[10] = numexp[10] * inv12;
                    c[12] = numexp[4] * inv34;
                    c[13] = numexp[12] * (inv12 * inv12 * inv12);
                    c[15] = numexp[0] * (inv34 * inv34 * inv34);
                    c[17] = numexp[6] * (inv34 * inv12 * inv12);
                    c[19] = numexp[2] * (inv34 * inv34 * inv12);
                }
                // Spurious coefficients are needed only as subtraction terms
                // of massive tadpoles (Ninja's spurious_terms_needed).
                if lin && (!is_zero(m[c1]) || !is_zero(m[c2])) {
                    self.second.push(r);
                }
            }
            if self.second.is_empty() {
                continue;
            }
            // Second orientation (e3 <-> e4) with reduced depth.
            let mindeg2 = mindeg - 1;
            let width2 = t2_len_le3(mindeg2);
            self.calls.clear();
            // Rows of lanes that failed in the first orientation never enter
            // `second`, so call row `s` belongs to `bub_rows[second[s]]`.
            for &r in &self.second {
                let row = &self.bub_rows[r];
                self.calls.lanes.push(row.lane);
                self.calls.v0.push(row.e1eff);
                self.calls.v1.push(cmom_from_real(&row.e2eff));
                self.calls.v2.push(row.e.e4);
                self.calls.v3.push(scale_c(&row.e.e3, 0.5 / row.e.mp34()));
                self.calls.params.push(row.param);
            }
            self.calls.prepare_out(width2);
            num.t2_expansion(
                &p,
                &self.calls.lanes,
                &self.calls.v0,
                &self.calls.v1,
                &self.calls.v2,
                &self.calls.v3,
                &self.calls.params,
                mindeg2,
                &mut self.calls.out,
            )?;
            for s in 0..self.second.len() {
                let row = &self.bub_rows[self.second[s]];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let vsq = &self.vsq[lane * n..(lane + 1) * n];
                let numexp = &mut self.calls.out[s * width2..(s + 1) * width2];
                let e = &row.e;
                let inv34 = 1. / e.mp34();
                let e3s = 0.5 * inv34;
                let uncut = &mut self.uncut;
                let mut stable = true;
                for &j in cp {
                    let pdiff = sub(&v[j], &v[c1]);
                    // Orientation e3 <-> e4: `e3k` holds e4.(V_j - V_0) here.
                    let e4k = mp_cr(&e.e4, &pdiff);
                    if !check.absolute(e4k.taxicab() * 2.) {
                        stable = false;
                        break;
                    }
                    let e2p = mp_rr(&e.e2, &pdiff);
                    let e1k = row.b0 * e2p;
                    uncut[j] = Uncut {
                        e3k: e4k,
                        ie3k: e4k.inv(),
                        e4k: mp_cr(&e.e3, &pdiff),
                        e1k,
                        e2k: mp_rr(&e.e1, &pdiff) + row.b1 * e2p,
                        d1: (e1k - mp_rr(&v[c1], &pdiff)) * 2.
                            + (vsq[j] - vsq[c1])
                            + (-m[j] + m[c1]),
                        f: mp2(&pdiff) + (m[c1] - m[j]),
                    };
                }
                if !stable {
                    fail(
                        &mut self.alive,
                        &mut self.lane_status,
                        lane,
                        InstabilityTest::BubbleExpansion,
                    );
                    continue;
                }
                for &j in cp {
                    let u = &uncut[j];
                    let d2 = u.e4k * inv34;
                    divide_t2(
                        numexp,
                        nterms - 1,
                        u.ie3k * 0.5,
                        [u.d1, C64::real(2. * u.e2k)],
                        [row.param[0] * d2, d2, row.param[1] * d2, row.param[2] * d2],
                    );
                }
                if let Some(spurious) = correct_spurious {
                    let e3eff = scale_c(&e.e3, e3s);
                    let np = parents.len();
                    let r = self.second[s];
                    for (pi, &(t, unc)) in parents.iter().enumerate() {
                        let tri = &self.tri[t * lanes + lane];
                        let u = &uncut[unc];
                        spurious(
                            numexp,
                            &tri.e3,
                            &tri.e4,
                            &tri.c,
                            &e.e4,
                            &e3eff,
                            &row.param,
                            u.e1k,
                            u.e2k,
                            u.e3k,
                            u.ie3k,
                            u.e4k * e3s,
                            &self.pairs[r * np + pi],
                            u.f,
                        );
                    }
                }
                let inv12 = 1. / e.mp12;
                let c = &mut self.bub[b * lanes + lane].c;
                if !x1 {
                    c[3] = numexp[(rmn + 1) as usize] * inv34;
                    if quad {
                        c[4] = numexp[rmn as usize] * (inv34 * inv34);
                        c[7] = numexp[2] * (inv12 * inv34);
                    }
                } else {
                    c[3] = numexp[3] * inv34;
                    c[4] = numexp[1] * (inv34 * inv34);
                    c[7] = numexp[5] * (inv12 * inv34);
                    c[11] = numexp[4] * inv34;
                    c[14] = numexp[0] * (inv34 * inv34 * inv34);
                    c[16] = numexp[6] * (inv34 * inv12 * inv12);
                    c[18] = numexp[2] * (inv34 * inv34 * inv12);
                }
            }
        }
        Ok(())
    }

    pub(super) fn tadpoles<N: DdNumerator + ?Sized>(
        &mut self,
        input: &FastInput,
        num: &mut N,
    ) -> Result<()> {
        let n = self.topo.n;
        let lanes = self.lanes;
        let eps = self.options.infrared_eps;
        let check = Stability::new(&self.options);
        let rmn = self.topo.rmn();
        let x1 = self.topo.x1();
        let mindeg = (rmn + 1) as usize;
        let width = t3_len(mindeg);
        let nterms = mindeg + 1;
        let any_triangle = self.any_triangle();
        let any_bubble = self.any_bubble();
        let orientations = if x1 { 2 } else { 1 };
        for a in 0..n {
            self.refresh_active();
            if self.active.is_empty() {
                return Ok(());
            }
            self.tad_rows.clear();
            for ai in 0..self.active.len() {
                let lane = self.active[ai];
                if is_zero(input.masses_squared[lane * n + a]) {
                    continue;
                }
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let basis = (0..2)
                    .find(|&i| {
                        let e = &self.tadpole_bases[i];
                        reference_vectors_ok(v, a, &e.e1, &e.e2, eps)
                    })
                    .unwrap_or(2);
                self.tad_rows.push(TadpoleRow { lane, basis });
            }
            if self.tad_rows.is_empty() {
                continue;
            }
            // Expansion vectors per orientation: rank <= n uses (e3, e4);
            // rank n+1 uses (e1, e2) rows followed by (e3, e4) rows.
            let vectors = |e: &Basis, orientation: usize| -> (CMom, CMom) {
                if x1 && orientation == 0 {
                    (
                        cmom_from_real(&e.e1),
                        cmom_from_real(&scale(&e.e2, 0.5 / e.mp12)),
                    )
                } else {
                    (e.e3, scale_c(&e.e4, 0.5 / e.mp34()))
                }
            };
            self.calls.clear();
            for orientation in 0..orientations {
                for row in &self.tad_rows {
                    let lane = row.lane;
                    let (ev3, ev4) = vectors(&self.tadpole_bases[row.basis], orientation);
                    let v0 = input.shifts[lane * n + a];
                    self.calls.lanes.push(lane);
                    self.calls
                        .v0
                        .push(cmom_from_real(&[-v0[0], -v0[1], -v0[2], -v0[3]]));
                    self.calls.v1.push(ev3);
                    self.calls.v2.push(ev4);
                    self.calls.scalar.push(input.masses_squared[lane * n + a]);
                }
            }
            self.calls.prepare_out(width);
            let cut = [a];
            num.t3_expansion(
                &cut,
                &self.calls.lanes,
                &self.calls.v0,
                &self.calls.v1,
                &self.calls.v2,
                &self.calls.scalar,
                mindeg,
                &mut self.calls.out,
            )?;
            let rows = self.tad_rows.len();
            let cp = &self.topo.tadpole_complements[a * (n - 1)..(a + 1) * (n - 1)];
            let tri_parents = &self.topo.tadpole_triangle_parents[a];
            let bub_parents = &self.topo.tadpole_bubble_parents[a];
            for r in 0..rows {
                let row = self.tad_rows[r];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let vsq = &self.vsq[lane * n..(lane + 1) * n];
                let e = self.tadpole_bases[row.basis];
                let param = m[a];
                let amu = [-v[a][0], -v[a][1], -v[a][2], -v[a][3]];
                let mut stable = true;
                let mut data = TadpoleData {
                    e34: -mp_rr(&e.e1, &e.e2),
                    ..TadpoleData::default()
                };
                for orientation in 0..orientations {
                    let offset = (orientation * rows + r) * width;
                    let numexp = &mut self.calls.out[offset..offset + width];
                    let (ev3, ev4) = vectors(&e, orientation);
                    let uncut = &mut self.uncut;
                    for &j in cp {
                        let pdiff = sub(&v[j], &v[a]);
                        let e3k = mp_cr(&ev3, &pdiff);
                        if !check.absolute(e3k.taxicab() * 2.) {
                            stable = false;
                            break;
                        }
                        uncut[j] = Uncut {
                            e3k,
                            ie3k: e3k.inv(),
                            e4k: mp_cr(&ev4, &pdiff),
                            e1k: C64::ZERO,
                            e2k: Dd::ZERO,
                            d1: C64::real(2. * mp_rr(&amu, &pdiff) + (vsq[j] - vsq[a])) - m[j]
                                + m[a],
                            f: mp2(&pdiff) + (m[a] - m[j]),
                        };
                    }
                    if !stable {
                        break;
                    }
                    for &j in cp {
                        let u = &uncut[j];
                        let d2 = u.e4k * 2.;
                        divide_t3(numexp, nterms, u.ie3k * 0.5, u.d1, [param * d2, d2]);
                    }
                    if any_triangle {
                        for &(t, [u0, u1]) in tri_parents {
                            let tri = &self.tri[t * lanes + lane];
                            let (k0, k1) = (&uncut[u0], &uncut[u1]);
                            if x1 {
                                subs::tadpole_from_triangle_x1(
                                    numexp, &tri.e3, &tri.e4, &tri.c, &ev3, &ev4, param, k0.e3k,
                                    k0.ie3k, k0.e4k, k0.f, k1.e3k, k1.ie3k, k1.e4k, k1.f,
                                );
                            } else {
                                tadpole_from_triangle(
                                    numexp, &tri.e3, &tri.e4, &tri.c, &ev3, k0, k1, rmn,
                                );
                            }
                        }
                    }
                    if any_bubble {
                        for &(bb, unc) in bub_parents {
                            let bub = &self.bub[bb * lanes + lane];
                            let kbol = sub(&v[self.topo.bubbles[bb][0]], &v[a]);
                            let u = &uncut[unc];
                            if x1 {
                                subs::tadpole_from_bubble_x1(
                                    numexp, &bub.e2, &bub.e3, &bub.e4, &kbol, &bub.c, &ev3, &ev4,
                                    param, u.e3k, u.ie3k, u.e4k, u.f,
                                );
                            } else {
                                tadpole_from_bubble(
                                    numexp, &bub.e2, &bub.e3, &bub.e4, &kbol, &bub.c, &ev3, u, rmn,
                                );
                            }
                        }
                    }
                    if !x1 {
                        data.c0 = numexp[(1 + rmn) as usize];
                    } else if orientation == 0 {
                        data.c0 = numexp[2];
                        data.c14 = numexp[3];
                    } else {
                        data.c15 = (numexp[3] - data.c14) * (2. / e.mp34());
                    }
                }
                if !stable {
                    fail(
                        &mut self.alive,
                        &mut self.lane_status,
                        lane,
                        InstabilityTest::TadpoleExpansion,
                    );
                    continue;
                }
                self.tad[a * lanes + lane] = data;
            }
        }
        Ok(())
    }

    /// Ninja's rank n+1 choice of the rank-3 bubble integral,
    /// `|c13 ((V2-V1).e2)³| != 0` (tested as an exact complex zero).
    #[inline]
    pub(super) fn rank3_bubble(&self, b: usize, lane: usize, v: &[RMom]) -> bool {
        let [i1, i2] = self.topo.bubbles[b];
        let data = &self.bub[b * self.lanes + lane];
        let ke2 = mp_rr(&sub(&v[i2], &v[i1]), &data.e2);
        !is_zero(data.c[13] * ke2 * ke2 * ke2)
    }
}
