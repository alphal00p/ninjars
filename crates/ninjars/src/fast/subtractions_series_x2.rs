//! Rank n+2 coefficient-level subtractions by series division.
//!
//! A subtraction term is the polynomial part (`t >= 0`) of a parent residue
//! written on the child cut's Laurent parametrisation and divided by the
//! parent's extra denominators (design 2.6). This module evaluates that
//! definition directly, with the machinery the engine already uses for the
//! numerator:
//!
//! 1. a parent variable `q.e/t` is a truncated jet in `u = 1/t` (the
//!    generated `tensor_jets` algebra: it is linear in the cut's slots
//!    `e3 + u a0 + u x a1 + u² S e4`, `S = P(x) + mu²`);
//! 2. the parent residue, relative to `t^R`, is accumulated in that jet basis
//!    by a Horner sweep per variable (each lower degree enters with its
//!    missing powers of `u`, and of `mu² = S - P(x)`), and converted once to
//!    Ninja's `(x, mu²)` layout (`t2_output`, `t3_output`);
//! 3. the sum is divided top-down by the uncut denominators
//!    (`divide_t2_5`, `divide_t2`, `divide_t3`): the quotient's leading orders
//!    are exactly the polynomial part.
//!
//! All parent coefficients (rank <= n+1 and the new ones) are handled
//! together. The generated routines of `subtractions_x2.rs` (original Ninja's
//! FORM routines re-indexed plus the reviewed CSE terms) compute the same
//! quantities from closed forms, several times more slowly; both are tested
//! against the series-definition fixtures of
//! `validation/rank-n2/subtraction_fixtures.json`, and this module is the one
//! the engine uses. Argument lists match the generated routines.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use super::complex::{C64, CMom, RMom, mp_cc, mp_cr, mp_rr};
use super::engine::{divide_t2, divide_t3};
use super::engine_x2::divide_t2_5;
use super::tensor::{t2_output, t2_output_k4, t3_len_const, t3_output};
use super::tensor_jets::{Jet, T2K3, T2K4, T3K3};

/// Index of `u^i x^k S^j` in a T2 jet (`tools/fast/generate_jets.py`
/// `t2_components`: by level, then `j`, then `k`).
const fn t2_jet(i: usize, k: usize, j: usize) -> usize {
    let mut index = 0;
    let mut l = 0;
    while l < i {
        index += (l / 2 + 1) * (l + 1 - l / 2);
        l += 1;
    }
    let mut jj = 0;
    while jj < j {
        index += i - 2 * jj + 1;
        jj += 1;
    }
    index + k
}

/// `pw[a - 1]` = jet of `(q/t)^a`, `a = 1..=N`, for the slots `x` of `q`.
#[inline(always)]
fn powers<P: Jet<L>, const L: usize, const N: usize>(x: &[C64; 4], pw: &mut [[C64; L]; N]) {
    P::first_set(&mut pw[0], C64::ONE, x);
    for a in 1..N {
        let (lo, hi) = pw.split_at_mut(a);
        P::mul_set(&mut hi[0], &lo[a - 1], x);
    }
}

/// `t[u^level x^k S^j] += c mu²` with `mu² = S - p0 - p1 x - p2 x²`.
#[inline(always)]
fn t2_mu<const L: usize>(t: &mut [C64; L], level: usize, c: C64, p: &[C64; 3]) {
    t[t2_jet(level, 0, 1)] += c;
    t[t2_jet(level, 0, 0)] -= c * p[0];
    t[t2_jet(level, 1, 0)] -= c * p[1];
    t[t2_jet(level, 2, 0)] -= c * p[2];
}

/// Horner sweep of one variable of the triangle residue on a bubble cut:
/// `((((c5 q + c4 u) q + u² C3) q + u³ C2) q + u⁴ C1) q` with
/// `C3 = c3 + c3m mu²`, `C2 = c2 + c2m mu²`, `C1 = c1 + c1m mu² + c1mm mu⁴`
/// and `q` the jet of `q.e/t` (slots `x`); the `u⁴` term only exists for
/// `LEVELS >= 4`. The last product is stored (`acc = false`) or
/// accumulated into `out`.
#[inline(always)]
fn bubble_horner<P: Jet<L>, const L: usize, const LEVELS: usize>(
    out: &mut [C64; L],
    acc: bool,
    c: [C64; 9],
    x: &[C64; 4],
    p: &[C64; 3],
) {
    let [c5, c4, c3, c3m, c2, c2m, c1, c1m, c1mm] = c;
    let mut h = [C64::ZERO; L];
    let mut g = [C64::ZERO; L];
    P::first_set(&mut h, c5, x);
    h[t2_jet(1, 0, 0)] += c4;
    P::mul_set(&mut g, &h, x);
    g[t2_jet(2, 0, 0)] += c3;
    t2_mu(&mut g, 2, c3m, p);
    P::mul_set(&mut h, &g, x);
    h[t2_jet(3, 0, 0)] += c2;
    t2_mu(&mut h, 3, c2m, p);
    P::mul_set(&mut g, &h, x);
    if LEVELS >= 4 {
        // mu⁴ = S² - 2 P S + P² with P = p0 + p1 x + p2 x².
        g[t2_jet(4, 0, 0)] += c1;
        t2_mu(&mut g, 4, c1m, p);
        g[t2_jet(4, 0, 2)] += c1mm;
        for k in 0..3 {
            g[t2_jet(4, k, 1)] -= c1mm * p[k] * 2.;
        }
        g[t2_jet(4, 0, 0)] += c1mm * (p[0] * p[0]);
        g[t2_jet(4, 1, 0)] += c1mm * (p[0] * p[1] * 2.);
        g[t2_jet(4, 2, 0)] += c1mm * (p[1] * p[1] + p[0] * p[2] * 2.);
        g[t2_jet(4, 3, 0)] += c1mm * (p[1] * p[2] * 2.);
        g[t2_jet(4, 4, 0)] += c1mm * (p[2] * p[2]);
    }
    if acc {
        P::mul_acc(out, &g, x);
    } else {
        P::mul_set(out, &g, x);
    }
}

/// The triangle residue (canonical rank n+2 layout: pure powers of
/// `u = q̄.et3` and `v = q̄.et4`) on a bubble cut, relative to `t^5`, as a
/// T2 jet with levels `0..=LEVELS` (4 for orientation A, 3 for B). Degree
/// `d` enters `u^(5-d)` lower; degree 0 never reaches `t >= 0`.
#[inline(always)]
fn triangle_on_bubble<P: Jet<L>, const L: usize, const LEVELS: usize>(
    c: &[C64; 21],
    xu: &[C64; 4],
    xv: &[C64; 4],
    p: &[C64; 3],
) -> [C64; L] {
    let mut t = [C64::ZERO; L];
    let cu = [c[15], c[12], c[3], c[17], c[2], c[10], c[1], c[8], c[19]];
    let cv = [c[16], c[13], c[6], c[18], c[5], c[11], c[4], c[9], c[20]];
    bubble_horner::<P, L, LEVELS>(&mut t, false, cu, xu, p);
    bubble_horner::<P, L, LEVELS>(&mut t, true, cv, xv, p);
    t
}

/// Rank n+2 triangle -> bubble subtraction, orientation A (all 22 slots).
/// Same arguments as `subtractions_x2::bubble_from_triangle_x2_all`.
#[inline(never)]
pub(super) fn bubble_from_triangle_all(
    divnum: &mut [C64],
    et3: &CMom,
    et4: &CMom,
    c: &[C64; 21],
    eb3: &CMom,
    eb4: &CMom,
    param: &[C64; 3],
    eb1k: C64,
    eb2k: f64,
    eb3k: C64,
    ieb3k: C64,
    eb4k: C64,
    eb1t: &[C64; 4],
    f: C64,
) {
    let _ = eb3k;
    let xu = [mp_cc(eb3, et3), eb1t[0], eb1t[2], mp_cc(eb4, et3)];
    let xv = [mp_cc(eb3, et4), eb1t[1], eb1t[3], mp_cc(eb4, et4)];
    let jet = triangle_on_bubble::<T2K4, 22, 4>(c, &xu, &xv, param);
    let mut n = [C64::ZERO; 22];
    t2_output_k4(&jet, param, &mut n);
    let d2 = eb4k * 2.;
    divide_t2_5(
        &mut n,
        ieb3k * 0.5,
        [eb1k * 2. + f, C64::real(2. * eb2k)],
        [param[0] * d2, d2, param[1] * d2, param[2] * d2],
    );
    for s in 0..22 {
        divnum[s] -= n[s];
    }
}

/// Rank n+2 triangle -> bubble subtraction, orientation B (e3 <-> e4): the
/// 13 slots with `t >= 1`. Same arguments as
/// `subtractions_x2::bubble_from_triangle_x2_spurious`.
#[inline(never)]
pub(super) fn bubble_from_triangle_spurious(
    divnum: &mut [C64],
    et3: &CMom,
    et4: &CMom,
    c: &[C64; 21],
    eb3: &CMom,
    eb4: &CMom,
    param: &[C64; 3],
    eb1k: C64,
    eb2k: f64,
    eb3k: C64,
    ieb3k: C64,
    eb4k: C64,
    eb1t: &[C64; 4],
    f: C64,
) {
    let _ = eb3k;
    let xu = [mp_cc(eb3, et3), eb1t[0], eb1t[2], mp_cc(eb4, et3)];
    let xv = [mp_cc(eb3, et4), eb1t[1], eb1t[3], mp_cc(eb4, et4)];
    let jet = triangle_on_bubble::<T2K3, 13, 3>(c, &xu, &xv, param);
    let mut n = [C64::ZERO; 13];
    t2_output(&jet, 3, param, &mut n);
    let d2 = eb4k * 2.;
    divide_t2(
        &mut n,
        4,
        ieb3k * 0.5,
        [eb1k * 2. + f, C64::real(2. * eb2k)],
        [param[0] * d2, d2, param[1] * d2, param[2] * d2],
    );
    for s in 0..13 {
        divnum[s] -= n[s];
    }
}

/// `t += u^shift (a g + b h)` on T3 jet levels `0..=levels`.
#[inline(always)]
fn t3_add(
    t: &mut [C64; 6],
    a: C64,
    g: &[C64; 6],
    b: C64,
    h: &[C64; 6],
    levels: usize,
    shift: usize,
) {
    for i in 0..=levels {
        for j in 0..=i / 2 {
            let s = t3_len_const(i) + j;
            t[t3_len_const(i + shift) + j] += a * g[s] + b * h[s];
        }
    }
}

/// `t += u^shift mu² (a g + b h)` on T3 jet levels `0..=levels`, `shift >= 2`,
/// with `mu² = s - P`.
#[inline(always)]
fn t3_add_mu(
    t: &mut [C64; 6],
    a: C64,
    g: &[C64; 6],
    b: C64,
    h: &[C64; 6],
    levels: usize,
    shift: usize,
    param: C64,
) {
    for i in 0..=levels {
        for j in 0..=i / 2 {
            let s = t3_len_const(i) + j;
            let v = a * g[s] + b * h[s];
            let o = t3_len_const(i + shift) + j;
            t[o + 1] += v;
            t[o] -= param * v;
        }
    }
}

/// Horner sweep of one variable of the triangle residue on a tadpole cut
/// (T3 jet, levels `0..=3`): `(((c5 q + c4 u) q + u² C3) q + u³ C2) q q`
/// with `C3 = c3 + c3m mu²`, `C2 = c2 + c2m mu²`, `mu² = s - param`.
#[inline(always)]
fn tadpole_horner(out: &mut [C64; 6], acc: bool, c: [C64; 6], x: &[C64; 4], param: C64) {
    let [c5, c4, c3, c3m, c2, c2m] = c;
    let mut h = [C64::ZERO; 6];
    let mut g = [C64::ZERO; 6];
    T3K3::first_set(&mut h, c5, x);
    h[t3_len_const(1)] += c4;
    T3K3::mul_set(&mut g, &h, x);
    g[t3_len_const(2)] += c3 - c3m * param;
    g[t3_len_const(2) + 1] += c3m;
    T3K3::mul_set(&mut h, &g, x);
    h[t3_len_const(3)] += c2 - c2m * param;
    h[t3_len_const(3) + 1] += c2m;
    T3K3::mul_set(&mut g, &h, x);
    if acc {
        T3K3::mul_acc(out, &g, x);
    } else {
        T3K3::mul_set(out, &g, x);
    }
}

/// Rank n+2 triangle -> tadpole subtraction on slots 4 (`t^0`) and 5
/// (`t^0 mu²`). Same arguments as `subtractions_x2::tadpole_from_triangle_x2`.
#[inline(never)]
pub(super) fn tadpole_from_triangle(
    divnum: &mut [C64],
    et3: &CMom,
    et4: &CMom,
    c: &[C64; 21],
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
    let _ = (e3k0, e3k1);
    let xu = [mp_cc(e3, et3), C64::ZERO, mp_cc(e4, et3), C64::ZERO];
    let xv = [mp_cc(e3, et4), C64::ZERO, mp_cc(e4, et4), C64::ZERO];
    // Relative to t^5, levels 0..=3 (the two uncut lines bring t^3 .. t^0).
    let mut t = [C64::ZERO; 6];
    let cu = [c[15], c[12], c[3], c[17], c[2], c[10]];
    let cv = [c[16], c[13], c[6], c[18], c[5], c[11]];
    tadpole_horner(&mut t, false, cu, &xu, param);
    tadpole_horner(&mut t, true, cv, &xv, param);
    let mut n = [C64::ZERO; 6];
    t3_output(&t, 3, param, &mut n);
    for (ie3k, e4k, f) in [(ie3k0, e4k0, f0), (ie3k1, e4k1, f1)] {
        let d2 = e4k * 2.;
        divide_t3(&mut n, 4, ie3k * 0.5, f, [param * d2, d2]);
    }
    divnum[4] -= n[4];
    divnum[5] -= n[5];
}

/// `g q` for a T3 jet `g` and the slots `x` of a linear `q`.
#[inline(always)]
fn t3_times(g: &[C64; 6], x: &[C64; 4]) -> [C64; 6] {
    let mut out = [C64::ZERO; 6];
    T3K3::mul_set(&mut out, g, x);
    out
}

/// The jet of `a q2 + b q3 + d q4` (slots `xw`, `xu`, `xv`).
#[inline(always)]
fn t3_linear(a: C64, xw: &[C64; 4], b: C64, xu: &[C64; 4], d: C64, xv: &[C64; 4]) -> [C64; 6] {
    let mut x = [C64::ZERO; 4];
    for k in 0..4 {
        x[k] = a * xw[k] + b * xu[k] + d * xv[k];
    }
    let mut out = [C64::ZERO; 6];
    T3K3::first_set(&mut out, C64::ONE, &x);
    out
}

/// `g += a h + b k`.
#[inline(always)]
fn t3_acc(g: &mut [C64; 6], a: C64, h: &[C64; 6], b: C64, k: &[C64; 6]) {
    for i in 0..6 {
        g[i] += a * h[i] + b * k[i];
    }
}

/// Rank n+2 bubble -> tadpole subtraction on slots 4 (`t^0`) and 5
/// (`t^0 mu²`). Same arguments as `subtractions_x2::tadpole_from_bubble_x2`.
#[inline(never)]
pub(super) fn tadpole_from_bubble(
    divnum: &mut [C64],
    eb2: &RMom,
    eb3: &CMom,
    eb4: &CMom,
    kbol: &RMom,
    c: &[C64; 35],
    e3: &CMom,
    e4: &CMom,
    param: C64,
    e3k: C64,
    ie3k: C64,
    e4k: C64,
    f: C64,
) {
    let _ = e3k;
    // q̄_b.e2 = t S2 + W + X T2/t (w); q̄_b.e3 (u) and q̄_b.e4 (v) have no
    // constant term.
    let xw = [
        mp_cr(e3, eb2),
        C64::real(mp_rr(eb2, kbol)),
        mp_cr(e4, eb2),
        C64::ZERO,
    ];
    let xu = [mp_cc(e3, eb3), C64::ZERO, mp_cc(e4, eb3), C64::ZERO];
    let xv = [mp_cc(e3, eb4), C64::ZERO, mp_cc(e4, eb4), C64::ZERO];
    let mut u = [[C64::ZERO; 6]; 4];
    let mut v = [[C64::ZERO; 6]; 4];
    powers::<T3K3, 6, 4>(&xu, &mut u);
    powers::<T3K3, 6, 4>(&xv, &mut v);
    // Degree 4: w (w (w (c20 w + c29 u + c30 v) + c27 u² + c28 v²)
    //   + c25 u³ + c26 v³) + c23 u⁴ + c24 v⁴.
    let mut g4 = t3_times(&t3_linear(c[20], &xw, c[29], &xu, c[30], &xv), &xw);
    t3_acc(&mut g4, c[27], &u[1], c[28], &v[1]);
    let mut g4 = t3_times(&g4, &xw);
    t3_acc(&mut g4, c[25], &u[2], c[26], &v[2]);
    let mut g4 = t3_times(&g4, &xw);
    t3_acc(&mut g4, c[23], &u[3], c[24], &v[3]);
    // Degree 3: w (w (c13 w + c16 u + c17 v) + c18 u² + c19 v²) + c14 u³ + c15 v³.
    let mut g3 = t3_times(&t3_linear(c[13], &xw, c[16], &xu, c[17], &xv), &xw);
    t3_acc(&mut g3, c[18], &u[1], c[19], &v[1]);
    let mut g3 = t3_times(&g3, &xw);
    t3_acc(&mut g3, c[14], &u[2], c[15], &v[2]);
    // Degree 2: w (c2 w + c7 u + c8 v) + c4 u² + c6 v²; with mu²:
    // w (c21 w + c33 u + c34 v) + c31 u² + c32 v².
    let mut g2 = t3_times(&t3_linear(c[2], &xw, c[7], &xu, c[8], &xv), &xw);
    t3_acc(&mut g2, c[4], &u[1], c[6], &v[1]);
    let mut g2m = t3_times(&t3_linear(c[21], &xw, c[33], &xu, c[34], &xv), &xw);
    t3_acc(&mut g2m, c[31], &u[1], c[32], &v[1]);
    // Degree 1: c1 w + c3 u + c5 v and, with mu², c10 w + c11 u + c12 v.
    let g1 = t3_linear(c[1], &xw, c[3], &xu, c[5], &xv);
    let g1m = t3_linear(c[10], &xw, c[11], &xu, c[12], &xv);
    // Relative to t^4, levels 0..=3 (the uncut line brings t^3 .. t^0).
    let zero = [C64::ZERO; 6];
    let mut t = g4;
    t3_add(&mut t, C64::ONE, &g3, C64::ZERO, &zero, 2, 1);
    t3_add(&mut t, C64::ONE, &g2, C64::ZERO, &zero, 1, 2);
    t3_add_mu(&mut t, C64::ONE, &g2m, C64::ZERO, &zero, 1, 2, param);
    t3_add(&mut t, C64::ONE, &g1, C64::ZERO, &zero, 0, 3);
    t3_add_mu(&mut t, C64::ONE, &g1m, C64::ZERO, &zero, 0, 3, param);
    let mut n = [C64::ZERO; 6];
    t3_output(&t, 3, param, &mut n);
    let d2 = e4k * 2.;
    divide_t3(&mut n, 4, ie3k * 0.5, f, [param * d2, d2]);
    divnum[4] -= n[4];
    divnum[5] -= n[5];
}
