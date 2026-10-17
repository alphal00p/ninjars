//! Cut bases, spinors and cut solutions of original Ninja (`basis.cc`,
//! `spinors.cc`, `cuts_utils.cc`), specialized to real external momenta and
//! complex squared masses (`ComplexMasses`).

// Index loops mirror original Ninja's component and lane loops.
#![allow(clippy::needless_range_loop)]

use super::complex::{C64, CMom, RMom, mp_cr, mp_rr};

/// `x` with the sign of `s` (Ninja's `sign`).
#[inline(always)]
fn sign(x: f64, s: f64) -> f64 {
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
pub(crate) fn scale(a: &RMom, s: f64) -> RMom {
    [a[0] * s, a[1] * s, a[2] * s, a[3] * s]
}

#[inline(always)]
pub(crate) fn mp2(a: &RMom) -> f64 {
    mp_rr(a, a)
}

/// Gram determinant `k1² k2² - (k1.k2)²`.
#[inline(always)]
pub(crate) fn gram_det(k1: &RMom, k2: &RMom) -> f64 {
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
fn signed_root(x: f64) -> C64 {
    // x < 0 ? I*sqrt(-x) : sqrt(x)
    if x < 0. {
        C64::new(0., (-x).sqrt())
    } else {
        C64::real(x.sqrt())
    }
}

impl Spinor {
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
            let ap = signed_root(p[0] + p[3]);
            Self {
                ap,
                ame: C64::new(p[1], p[2]) / ap,
                bp: ap,
                bme: C64::new(p[1], -p[2]) / ap,
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
    pub r1: f64,
    pub r2: f64,
    pub mp12: f64,
}

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
        let (mut r1, mut r2) = (0., 0.);
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
            let gamma = k1k2 + sign(1., k1k2) * (k1k2 * k1k2 - k1q * k2q).sqrt();
            r1 = k1q / gamma;
            r2 = k2q / gamma;
            let den = 1. - r1 * r2;
            e1 = sub(&e1, &scale(k2, r1));
            e1 = [e1[0] / den, e1[1] / den, e1[2] / den, e1[3] / den];
            e2 = sub(&e2, &scale(k1, r2));
            e2 = [e2[0] / den, e2[1] / den, e2[2] / den, e2[3] / den];
            mp12 = k1k2 / (1. + r1 * r2);
        }
        Self::complete(e1, e2, r1, r2, mp12, eps)
    }

    /// `Basis(k)`: `k = e1 + r1 e2` with a fixed massless reference `e2`.
    pub(crate) fn from_single(k: &RMom, eps: f64) -> Self {
        const INVSQRT3: f64 = 0.577_350_269_189_625_8; // 1./sqrt(3.)
        let k2 = [
            sign(1., k[0]),
            -sign(INVSQRT3, k[1]),
            -sign(INVSQRT3, k[2]),
            -sign(INVSQRT3, k[3]),
        ];
        let kq = mp2(k);
        let k1k2 = mp_rr(k, &k2);
        let mut e1 = *k;
        let mut r1 = 0.;
        if kq.abs() >= eps {
            r1 = 0.5 * kq / k1k2;
            e1 = sub(&e1, &scale(&k2, r1));
        }
        Self::complete(e1, k2, r1, 0., k1k2, eps)
    }

    fn complete(e1: RMom, e2: RMom, r1: f64, r2: f64, mp12: f64, eps: f64) -> Self {
        let sp1 = Spinor::new(&e1, eps);
        let sp2 = Spinor::new(&e2, eps);
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
    pub(crate) fn mp34(&self) -> f64 {
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
    refs.map(|(r1, r2)| Basis::from_pair(&r1, &r2, eps))
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
    let a1 = (m1q - m4q + mk11) / 2. / e.mp12;
    let a2 = (m2q - m1q - mk22) / 2. / e.mp12;
    let x1 = (a2 - a1 * e.r2) * beta;
    let x2 = (a1 - a2 * e.r1) * beta;
    let k23 = add(k2, k3);
    let mk33 = mp2(&k23);
    let me31 = mp_rr(&k23, &e.e1);
    let me32 = mp_rr(&k23, &e.e2);
    let me33 = mp_cr(&e.e3, &k23);
    let me34 = mp_cr(&e.e4, &k23);
    let a3 = (m3q - m1q - mk33 - x1 * (2. * me31) - x2 * (2. * me32)) / 2. / me33;
    let a4 = -(me34 / me33);
    let b1 = a3 * (-2. * e.mp12);
    let b2 = a4 * (-2. * e.mp12);
    let b0 = x1 * x2 * (2. * e.mp12) - m1q;
    let rtdel = (b1 * b1 - b0 * b2 * 4.).sqrt();
    let x41 = (-b1 + rtdel) / 2. / b2;
    let x42 = (-b1 - rtdel) / 2. / b2;
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
    let a1 = (m1q - m3q + mk11) / 2. / e.mp12;
    let a2 = (m2q - m1q - mk22) / 2. / e.mp12;
    let x1 = (a2 - a1 * e.r2) * beta;
    let x2 = (a1 - a2 * e.r1) * beta;
    let param = m1q - x1 * x2 * (2. * e.mp12);
    (x1, x2, param)
}

/// Double-cut parametrization (`CutBubble::getLoopMomentum`): returns
/// `(b0, b1, param)` with `a0 = b0 e2`, `a1 = e1 + b1 e2` (real).
pub(crate) fn bubble_solution(k: &RMom, e: &Basis, m1q: C64, m2q: C64) -> (C64, f64, [C64; 3]) {
    let kq = mp2(k);
    let e12 = 2. * e.mp12;
    let e1k = 2. * mp_rr(&e.e1, k);
    let e2k = 2. * mp_rr(&e.e2, k);
    let b0 = (-m2q + m1q + kq) / e2k;
    let b1 = -e1k / e2k;
    (b0, b1, [m1q, -(b0 * e12), C64::real(-b1 * e12)])
}
