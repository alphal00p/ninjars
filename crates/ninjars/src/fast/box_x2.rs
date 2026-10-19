//! Conditioned quadruple-cut geometry: always in the rank n+2 engine, and in
//! the rank <= n+1 engine with [`super::FastOptions::conditioned_cuts`].
//!
//! The box rational terms come from the integrand on the d-dimensional cut
//! `q = q̄ - V_0 + t n`, `mu² = t² n² - beta` (`beta = m_0² - q̄²`), expanded at
//! large `t`: `c6` and `c4` are its `t^(r-n+4)` and `t^(r-n+2)` coefficients
//! after the division by the uncut denominators. The expansion is exact only
//! if the cut denominators vanish identically in `t`, i.e. if the normal `n`
//! is orthogonal to the cut momenta `K_i = V_i - V_0` and `q̄.n = 0`. A
//! residual `n.K_i = delta |n||K_i|` adds terms of order `delta t` that the
//! expansion attributes to the coefficients, amplified by the numerator's
//! cancellations at high rank.
//!
//! Original Ninja's normal `(p1.e4) e3 - (p1.e3) e4` is built from the spinors
//! of the cut basis. Its residual `delta` reaches 3.2e-15 (median 9.1e-17
//! over the 573 non-vanishing boxes of eight rank n+2 cases, n = 4..7,
//! measured in Float(256) when 611f6dc was verified; the normal below stays
//! at or below 8.2e-17). On
//! `rn2-exact-mu-n7-r9-perm` lane 1 this put the box `c4` 5 to 100 times
//! further from the Float(256) value than the generic engine's (whose normal
//! comes from a linear solve), and the box stage cancels 3e5 times against
//! the triangles. An exact-arithmetic check of one box (`c4 = 1.2e5`) gave a
//! `c4` error of up to 5.5e-10 from one-ulp perturbations of the normal, and
//! below 7e-12 from the rounding of the center, of `beta` or of the division.
//!
//! The normal here is the Levi-Civita vector of the three cut momenta, from
//! their 3x3 cofactors with Kahan's difference of products (exact for the
//! small dyadic inputs of the identity cases), scaled by a power of two. The
//! center is Ninja's (the mean of its two solutions), projected onto `n⊥` and
//! refined once against `q̄.K_i = h_i = (m_i² - m_0² - K_i²)/2` with
//! compensated residuals; `beta` is compensated as well. The two
//! four-dimensional solutions are `q̄ ± sqrt(beta/n²) n`, so that `c0` uses
//! the same, consistently rounded geometry. Compensating the Gram matrix,
//! the projections or `n²` as well changed nothing measurable.
//!
//! Effect on the binary64 error against the high-precision references
//! (`ninjars case-check --engines fast --precision f64`, its scaled
//! differences) on the 426 tier-A lanes of the rank n+2 cases of
//! `validation/rank-n2/cases`, `validation/benchmark/synthetic-rank-n2` and
//! `validation/rank-n2/identity` (tier from the reference files; every lane
//! succeeds in both geometries and in the generic engine's binary64 run;
//! ratios of errors floored at 1e-16), against the spinor geometry of main
//! 0d0d7e7:
//! - geometric mean of the error ratio 0.70;
//! - 77 lanes more than 2 times better and 11 more than 2 times worse, the
//!   largest of those 5.4e-11 (`rn2-exact-q-n7-r9` lane 1, was 1.9e-11;
//!   generic binary64 2.1e-10), all within 6.9 times the generic engine's
//!   binary64 error;
//! - lanes outside `max(10 e_generic, 1e-13)` (`e_generic`: the generic
//!   engine's binary64 error): 28 -> 13;
//! - `rn2-shifted-mixed-n7-r9-mu2` lane 0: 1.7e-3 -> 1.3e-5 (generic 1.6e-4);
//!   `rn2-exact-mu-n7-r9-perm` lane 1: 6.1e-7 -> 1.8e-8.
//!
//! [`box_geometry_x2`] adds one failure condition to Ninja's checks, which
//! are kept and run first: the lane is unstable when the normal, `n²` or the
//! Gram determinant of the cut momenta is exactly zero or not finite.
//!
//! These are plain numerical kernels on binary64 numbers (error-free products
//! by `mul_add` and two-sums); no algebra is involved.

// Index loops over the four components, as in the other fast-path modules.
#![allow(clippy::needless_range_loop)]

use super::complex::{C64, CMom, RMom, mp_cr, mp_rr};

/// `a b - c d` with one rounding error (Kahan's algorithm with FMA).
#[inline(always)]
fn difference_of_products(a: f64, b: f64, c: f64, d: f64) -> f64 {
    let w = c * d;
    let e = (-c).mul_add(d, w);
    let f = a.mul_add(b, -w);
    f + e
}

/// Minkowski product `a.b` accumulated without cancellation error (error-free
/// products and two-sums; the result is rounded once more at the end).
#[inline]
fn dot_compensated(a: &RMom, b: &RMom) -> f64 {
    let products = [a[0] * b[0], -(a[1] * b[1]), -(a[2] * b[2]), -(a[3] * b[3])];
    let errors = [
        a[0].mul_add(b[0], -products[0]),
        (-a[1]).mul_add(b[1], -products[1]),
        (-a[2]).mul_add(b[2], -products[2]),
        (-a[3]).mul_add(b[3], -products[3]),
    ];
    let (mut sum, mut carry) = (0., 0.);
    for x in products {
        let t = sum + x;
        let z = t - sum;
        carry += (sum - (t - z)) + (x - z);
        sum = t;
    }
    for e in errors {
        carry += e;
    }
    sum + carry
}

/// `l.k` for a complex `l` and a real `k`, compensated.
#[inline]
fn dot_compensated_cr(l: &CMom, k: &RMom) -> C64 {
    let re: RMom = std::array::from_fn(|mu| l[mu].re);
    let im: RMom = std::array::from_fn(|mu| l[mu].im);
    C64::new(dot_compensated(&re, k), dot_compensated(&im, k))
}

/// `l.l` for a complex `l`, compensated.
#[inline]
fn square_compensated(l: &CMom) -> C64 {
    let re: RMom = std::array::from_fn(|mu| l[mu].re);
    let im: RMom = std::array::from_fn(|mu| l[mu].im);
    C64::new(
        dot_compensated(&re, &re) - dot_compensated(&im, &im),
        2. * dot_compensated(&re, &im),
    )
}

/// `det [a; b; c]` of three rows of three, by cofactors of the first row.
#[inline(always)]
fn det3(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    a[0] * difference_of_products(b[1], c[2], b[2], c[1])
        - a[1] * difference_of_products(b[0], c[2], b[2], c[0])
        + a[2] * difference_of_products(b[0], c[1], b[1], c[0])
}

/// The Minkowski normal of `k[0], k[1], k[2]` (`n.k_i = 0`), scaled by a power
/// of two to a largest component in `[1/2, 1)`; `None` if it vanishes.
fn normal(k: &[RMom; 3]) -> Option<RMom> {
    // Covariant rows: n^mu = (-1)^mu det(rows without column mu).
    let rows: [RMom; 3] = std::array::from_fn(|i| [k[i][0], -k[i][1], -k[i][2], -k[i][3]]);
    let drop = |mu: usize| -> [[f64; 3]; 3] {
        std::array::from_fn(|i| {
            let r = &rows[i];
            match mu {
                0 => [r[1], r[2], r[3]],
                1 => [r[0], r[2], r[3]],
                2 => [r[0], r[1], r[3]],
                _ => [r[0], r[1], r[2]],
            }
        })
    };
    let mut n = [0.; 4];
    for mu in 0..4 {
        let [a, b, c] = drop(mu);
        let d = det3(a, b, c);
        n[mu] = if mu.is_multiple_of(2) { d } else { -d };
    }
    let largest = n.iter().fold(0_f64, |a, x| a.max(x.abs()));
    if largest == 0. || !largest.is_finite() {
        return None;
    }
    // Exact rescaling by the power of two that maps `largest` into [1/2, 1)
    // (`2^(1022 - e)` for the biased exponent `e`; extreme exponents keep
    // the scale).
    let biased = (largest.to_bits() >> 52) & 0x7ff;
    let factor = if (1..=2044).contains(&biased) {
        f64::from_bits((2045 - biased) << 52)
    } else {
        1.
    };
    Some(n.map(|x| x * factor))
}

/// Box geometry on the cut `[V_0, V_1, V_2, V_3]` (the first is the reference).
#[derive(Clone, Copy, Debug)]
pub(super) struct BoxGeometryX2 {
    /// The normal `n` (real, stored as complex for the expansion calls).
    pub(super) normal: CMom,
    /// `q̄ - V_0`: the center in the loop-momentum frame.
    pub(super) mid: CMom,
    /// `m_0² - q̄²`, so that `mu² = t² n² - beta` on the cut.
    pub(super) beta: C64,
    /// The two four-dimensional solutions `q̄ ± sqrt(beta/n²) n - V_0`.
    pub(super) q: [CMom; 2],
}

/// The conditioned geometry of the box with shifts `v` (one per cut line, in
/// cut order), squared masses `m` and original Ninja's two solutions `l` of
/// `(l + K_i)² = m_i²` (`l = q + V_0`). `None` if the normal or the Gram
/// determinant of the cut momenta vanishes or is not finite.
pub(super) fn box_geometry_x2(v: &[RMom; 4], m: &[C64; 4], l: &[CMom; 2]) -> Option<BoxGeometryX2> {
    let sub = |a: &RMom, b: &RMom| -> RMom { [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]] };
    let k = [sub(&v[1], &v[0]), sub(&v[2], &v[0]), sub(&v[3], &v[0])];
    let n = normal(&k)?;
    let n2 = mp_rr(&n, &n);
    if n2 == 0. || !n2.is_finite() {
        return None;
    }
    let project = |c: &mut CMom| {
        let cn = mp_cr(c, &n) * (1. / n2);
        for mu in 0..4 {
            c[mu] -= cn * n[mu];
        }
    };
    let mut center: CMom = std::array::from_fn(|mu| (l[0][mu] + l[1][mu]) * 0.5);
    project(&mut center);
    // One refinement step of c.K_i = h_i inside span(K) (= n⊥): the
    // correction sum_j x_j K_j solves G x = r with the Gram matrix G.
    // G only enters the small correction: plain arithmetic suffices.
    let g: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| mp_rr(&k[i], &k[j])));
    let cofactor = |i: usize, j: usize| {
        let (r0, r1) = [(1, 2), (0, 2), (0, 1)][i];
        let (c0, c1) = [(1, 2), (0, 2), (0, 1)][j];
        let minor = g[r0][c0] * g[r1][c1] - g[r0][c1] * g[r1][c0];
        if (i + j).is_multiple_of(2) {
            minor
        } else {
            -minor
        }
    };
    let gram = g[0][0] * cofactor(0, 0) + g[0][1] * cofactor(0, 1) + g[0][2] * cofactor(0, 2);
    if gram == 0. || !gram.is_finite() {
        return None;
    }
    let inverse: [[f64; 3]; 3] =
        std::array::from_fn(|i| std::array::from_fn(|j| cofactor(j, i) / gram));
    let residual: [C64; 3] = std::array::from_fn(|i| {
        (m[i + 1] - m[0] - C64::real(dot_compensated(&k[i], &k[i]))) * 0.5
            - dot_compensated_cr(&center, &k[i])
    });
    let x: [C64; 3] = std::array::from_fn(|i| {
        residual[0] * inverse[i][0] + residual[1] * inverse[i][1] + residual[2] * inverse[i][2]
    });
    for mu in 0..4 {
        center[mu] += x[0] * k[0][mu] + x[1] * k[1][mu] + x[2] * k[2][mu];
    }
    project(&mut center);
    let beta = m[0] - square_compensated(&center);
    let s = (beta * (1. / n2)).sqrt();
    let mid: CMom = std::array::from_fn(|mu| center[mu] - v[0][mu]);
    let q = [
        std::array::from_fn(|mu| mid[mu] + s * n[mu]),
        std::array::from_fn(|mu| mid[mu] - s * n[mu]),
    ];
    Some(BoxGeometryX2 {
        normal: n.map(C64::real),
        mid,
        beta,
        q,
    })
}

#[cfg(test)]
mod tests {
    use super::super::complex::mp_cc;
    use super::*;

    fn den(q: &CMom, v: &RMom, m: C64) -> C64 {
        let s: CMom = std::array::from_fn(|mu| q[mu] + v[mu]);
        mp_cc(&s, &s) - m
    }

    /// The normal is exactly orthogonal for small integer momenta, the center
    /// satisfies the cut's linear equations and `q̄.n = 0`, and both solutions
    /// are on the four-dimensional cut.
    #[test]
    fn conditioned_box_geometry_is_on_the_cut() {
        let v = [
            [0.5, -13., 11., -2.],
            [9., -6., -10., -1.],
            [19., -3., 0., 7.],
            [-5., -11., -7., -1.],
        ];
        let m = [
            C64::new(16., -0.75),
            C64::new(25., -1.25),
            C64::new(49., -0.5),
            C64::new(9., 0.),
        ];
        // Ninja's solutions are only the starting center: perturb them.
        let k: [RMom; 3] =
            std::array::from_fn(|i| std::array::from_fn(|mu| v[i + 1][mu] - v[0][mu]));
        let n = normal(&k).unwrap();
        for ki in &k {
            assert_eq!(mp_rr(&n, ki), 0., "{n:?} {ki:?}");
        }
        let start = [[C64::new(3., 0.1); 4], [C64::new(-1., 0.2); 4]];
        let geometry = box_geometry_x2(&v, &m, &start).unwrap();
        let scale = 1e4;
        let center: CMom = std::array::from_fn(|mu| geometry.mid[mu] + v[0][mu]);
        assert!(mp_cr(&center, &n).abs() <= 1e-13 * scale);
        for q in &geometry.q {
            for i in 0..4 {
                let d = den(q, &v[i], m[i]);
                assert!(d.abs() <= 1e-12 * scale, "line {i}: {d:?}");
            }
        }
        // mu² = t² n² - beta keeps every cut denominator at zero.
        let n2 = mp_rr(&n, &n);
        for t in [0.5, 3., -7.] {
            let q: CMom = std::array::from_fn(|mu| geometry.mid[mu] + geometry.normal[mu] * t);
            let mu2 = C64::real(t * t * n2) - geometry.beta;
            for i in 0..4 {
                let d = den(&q, &v[i], m[i]) - mu2;
                assert!(d.abs() <= 1e-11 * scale, "t {t} line {i}: {d:?}");
            }
        }
    }

    /// Degenerate cut momenta have no normal.
    #[test]
    fn degenerate_boxes_have_no_geometry() {
        let v = [
            [0.; 4],
            [1., 2., 3., 4.],
            [2., 4., 6., 8.],
            [5., 1., 0., 2.],
        ];
        let m = [C64::real(1.); 4];
        let l = [[C64::ZERO; 4]; 2];
        assert!(box_geometry_x2(&v, &m, &l).is_none());
    }
}
