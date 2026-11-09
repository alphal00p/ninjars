//! Unit tests of the fast-path building blocks against naive references.

use std::collections::BTreeMap;

use super::complex::{C64, CMom, RMom, mp_cc, mp_cr, mp_rr};
use super::engine::{divide_t2, divide_t3};
use super::geometry::{
    Basis, box_solutions, bubble_solution, combine_real2, mp2, sub, tadpole_candidates,
    triangle_solution,
};
use super::numerator::{FastNumerator, t2_len, t3_len};
use super::replay::{RecordingMasters, RecordingNumerator};
use super::tensor::TensorNumerator;
use super::tensor_layout::{monomials, ncoeff, ncoeff_mu};
use super::tensor_mu::MuTensorNumerator;
use super::{
    FastAmplitude, FastInput, FastMaster, FastMasters, FastOptions, FastResult, FastStatus,
};

/// SplitMix64 generator for reproducible test data.
pub(super) struct Rng(pub(super) u64);

impl Rng {
    pub(super) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub(super) fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.
    }
    pub(super) fn c(&mut self) -> C64 {
        C64::new(self.uniform(), self.uniform())
    }
    pub(super) fn cmom(&mut self) -> CMom {
        [self.c(), self.c(), self.c(), self.c()]
    }
    pub(super) fn rmom(&mut self, scale: f64) -> RMom {
        [
            scale * self.uniform(),
            scale * self.uniform(),
            scale * self.uniform(),
            scale * self.uniform(),
        ]
    }
}

/// Sparse polynomial in `(t, x, mu²)` with signed `t` powers.
pub(super) type Poly = BTreeMap<(i32, u16, u16), C64>;

pub(super) fn poly_mul(a: &Poly, b: &Poly) -> Poly {
    let mut out = Poly::new();
    for (&(ta, xa, ma), &ca) in a {
        for (&(tb, xb, mb), &cb) in b {
            *out.entry((ta + tb, xa + xb, ma + mb)).or_default() += ca * cb;
        }
    }
    out
}

pub(super) fn poly_add_scaled(acc: &mut Poly, p: &Poly, scale: C64) {
    for (&k, &v) in p {
        *acc.entry(k).or_default() += v * scale;
    }
}

/// `N(q)` for components given as polynomials, by explicit monomials.
pub(super) fn reference_numerator(coefficients: &[C64], rank: usize, q: &[Poly; 4]) -> Poly {
    let one: Poly = [((0, 0, 0), C64::ONE)].into_iter().collect();
    let mut powers = [
        vec![one.clone()],
        vec![one.clone()],
        vec![one.clone()],
        vec![one],
    ];
    for (mu, list) in powers.iter_mut().enumerate() {
        for e in 1..=rank {
            let next = poly_mul(&list[e - 1], &q[mu]);
            list.push(next);
        }
    }
    let mut total = Poly::new();
    for (k, m) in monomials(rank).iter().enumerate() {
        let mut term = powers[0][m[0] as usize].clone();
        for mu in 1..4 {
            term = poly_mul(&term, &powers[mu][m[mu] as usize]);
        }
        poly_add_scaled(&mut total, &term, coefficients[k]);
    }
    total
}

pub(super) fn linear(terms: &[((i32, u16, u16), C64)]) -> Poly {
    let mut p = Poly::new();
    for &(k, v) in terms {
        *p.entry(k).or_default() += v;
    }
    p
}

pub(super) fn assert_close(actual: &[C64], expected: &[C64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    let scale = expected.iter().map(|z| z.abs()).fold(1e-300, f64::max);
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*a - *e).abs() <= 1e-12 * scale,
            "{context}: entry {i}: {a:?} != {e:?} (scale {scale})"
        );
    }
}

pub(super) fn get(p: &Poly, key: (i32, u16, u16)) -> C64 {
    p.get(&key).copied().unwrap_or_default()
}

#[test]
fn tensor_numerator_matches_direct_polynomial_expansions() {
    let mut rng = Rng(7);
    for rank in 0..=7 {
        let lanes = 3;
        let mut numerator = TensorNumerator::new(rank, lanes);
        for c in numerator.coefficients_mut() {
            *c = rng.c();
        }
        assert_eq!(numerator.ncoeff(), ncoeff(rank));
        for lane in 0..lanes {
            let coefficients = numerator.lane_coefficients(lane).to_vec();
            let cut = [0usize];
            // Direct evaluation.
            let q = rng.cmom();
            let mut out = [C64::ZERO];
            numerator
                .evaluate(&cut, &[lane], &[q], &[rng.c()], &mut out)
                .unwrap();
            let expected =
                reference_numerator(&coefficients, rank, &q.map(|v| linear(&[((0, 0, 0), v)])));
            assert_close(&out, &[get(&expected, (0, 0, 0))], "evaluate");

            // mu expansion, with and without v1.
            let (v0, v1) = (rng.cmom(), rng.cmom());
            let poly = reference_numerator(
                &coefficients,
                rank,
                &std::array::from_fn(|mu| linear(&[((1, 0, 0), v0[mu]), ((0, 0, 0), v1[mu])])),
            );
            let r = rank as i32;
            let mut out2 = [C64::ZERO; 2];
            numerator
                .mu_expansion(&cut, &[lane], &[v0], Some(&[v1]), &mut out2)
                .unwrap();
            assert_close(
                &out2,
                &[get(&poly, (r, 0, 0)), get(&poly, (r - 1, 0, 0))],
                "mu expansion with v1",
            );
            let mut out1 = [C64::ZERO];
            numerator
                .mu_expansion(&cut, &[lane], &[v0], None, &mut out1)
                .unwrap();
            assert_close(&out1, &[get(&poly, (r, 0, 0))], "mu expansion");

            // t3 expansions for every mindeg.
            let (a, e3, e4, param) = (rng.cmom(), rng.cmom(), rng.cmom(), rng.c());
            let poly = reference_numerator(
                &coefficients,
                rank,
                &std::array::from_fn(|mu| {
                    linear(&[
                        ((0, 0, 0), a[mu]),
                        ((1, 0, 0), e3[mu]),
                        ((-1, 0, 0), param * e4[mu]),
                        ((-1, 0, 1), e4[mu]),
                    ])
                }),
            );
            for mindeg in 0..=4 {
                let mut out = vec![C64::ZERO; t3_len(mindeg)];
                numerator
                    .t3_expansion(
                        &cut,
                        &[lane],
                        &[a],
                        &[e3],
                        &[e4],
                        &[param],
                        mindeg,
                        &mut out,
                    )
                    .unwrap();
                let mut expected = Vec::new();
                for i in 0..=mindeg {
                    for j in 0..=i / 2 {
                        expected.push(get(&poly, (r - i as i32, 0, j as u16)));
                    }
                }
                assert_close(&out, &expected, &format!("t3 rank {rank} mindeg {mindeg}"));
            }

            // t2 expansions for every mindeg.
            let (a0, a1, e3, e4) = (rng.cmom(), rng.cmom(), rng.cmom(), rng.cmom());
            let p = [rng.c(), rng.c(), rng.c()];
            let poly = reference_numerator(
                &coefficients,
                rank,
                &std::array::from_fn(|mu| {
                    linear(&[
                        ((0, 0, 0), a0[mu]),
                        ((0, 1, 0), a1[mu]),
                        ((1, 0, 0), e3[mu]),
                        ((-1, 0, 0), p[0] * e4[mu]),
                        ((-1, 1, 0), p[1] * e4[mu]),
                        ((-1, 2, 0), p[2] * e4[mu]),
                        ((-1, 0, 1), e4[mu]),
                    ])
                }),
            );
            const LAYOUT: [&[(u16, u16)]; 4] = [
                &[(0, 0)],
                &[(0, 0), (1, 0)],
                &[(0, 0), (0, 1), (1, 0), (2, 0)],
                &[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 0)],
            ];
            for mindeg in 0..=3 {
                let mut out = vec![C64::ZERO; t2_len(mindeg)];
                numerator
                    .t2_expansion(
                        &cut,
                        &[lane],
                        &[a0],
                        &[a1],
                        &[e3],
                        &[e4],
                        &[p],
                        mindeg,
                        &mut out,
                    )
                    .unwrap();
                let mut expected = Vec::new();
                for (i, order) in LAYOUT.iter().enumerate().take(mindeg + 1) {
                    for &(x, m) in order.iter() {
                        expected.push(get(&poly, (r - i as i32, x, m)));
                    }
                }
                assert_close(&out, &expected, &format!("t2 rank {rank} mindeg {mindeg}"));
            }
        }
    }
}

/// `sum_k mu^(2k) T_k(q)` with every block expanded by [`reference_numerator`].
pub(super) fn reference_mu_numerator(
    numerator: &MuTensorNumerator,
    lane: usize,
    q: &[Poly; 4],
    mu_squared: &Poly,
) -> Poly {
    let coefficients = numerator.lane_coefficients(lane);
    let one: Poly = [((0, 0, 0), C64::ONE)].into_iter().collect();
    let mut power = one;
    let mut total = Poly::new();
    for k in 0..=numerator.mu_powers() {
        let block = &coefficients[numerator.block_range(k)];
        let term = poly_mul(
            &reference_numerator(block, numerator.rank() - 2 * k, q),
            &power,
        );
        poly_add_scaled(&mut total, &term, C64::ONE);
        power = poly_mul(&power, mu_squared);
    }
    total
}

#[test]
fn mu_tensor_numerator_matches_direct_polynomial_expansions() {
    let mut rng = Rng(19);
    for rank in 0..=7 {
        for mu_powers in 0..=rank / 2 {
            let lanes = 2;
            let mut numerator = MuTensorNumerator::new(rank, mu_powers, lanes).unwrap();
            assert_eq!(numerator.ncoeff(), ncoeff_mu(rank, mu_powers));
            for c in numerator.coefficients_mut() {
                *c = rng.c();
            }
            let r = rank as i32;
            let context = |what: &str| format!("{what}: rank {rank} mu powers {mu_powers}");
            for lane in 0..lanes {
                let cut = [0usize];
                // Direct evaluation at an independent mu².
                let (q, mu2) = (rng.cmom(), rng.c());
                let mut out = [C64::ZERO];
                numerator
                    .evaluate(&cut, &[lane], &[q], &[mu2], &mut out)
                    .unwrap();
                let expected = reference_mu_numerator(
                    &numerator,
                    lane,
                    &q.map(|v| linear(&[((0, 0, 0), v)])),
                    &linear(&[((0, 0, 0), mu2)]),
                );
                assert_close(&out, &[get(&expected, (0, 0, 0))], &context("evaluate"));

                // Box mu expansion: q = t v0 + v1, mu² = t² v0².
                let (v0, v1) = (rng.cmom(), rng.cmom());
                let poly = reference_mu_numerator(
                    &numerator,
                    lane,
                    &std::array::from_fn(|mu| linear(&[((1, 0, 0), v0[mu]), ((0, 0, 0), v1[mu])])),
                    &linear(&[((2, 0, 0), mp_cc(&v0, &v0))]),
                );
                let mut out2 = [C64::ZERO; 2];
                numerator
                    .mu_expansion(&cut, &[lane], &[v0], Some(&[v1]), &mut out2)
                    .unwrap();
                assert_close(
                    &out2,
                    &[get(&poly, (r, 0, 0)), get(&poly, (r - 1, 0, 0))],
                    &context("mu expansion with v1"),
                );
                let mut out1 = [C64::ZERO];
                numerator
                    .mu_expansion(&cut, &[lane], &[v0], None, &mut out1)
                    .unwrap();
                assert_close(&out1, &[get(&poly, (r, 0, 0))], &context("mu expansion"));

                // t3 expansions: the numerator's own mu² is the cut's mu².
                let mu2_poly = linear(&[((0, 0, 1), C64::ONE)]);
                let (a, e3, e4, param) = (rng.cmom(), rng.cmom(), rng.cmom(), rng.c());
                let poly = reference_mu_numerator(
                    &numerator,
                    lane,
                    &std::array::from_fn(|mu| {
                        linear(&[
                            ((0, 0, 0), a[mu]),
                            ((1, 0, 0), e3[mu]),
                            ((-1, 0, 0), param * e4[mu]),
                            ((-1, 0, 1), e4[mu]),
                        ])
                    }),
                    &mu2_poly,
                );
                for mindeg in 0..=4 {
                    let mut out = vec![C64::ZERO; t3_len(mindeg)];
                    numerator
                        .t3_expansion(
                            &cut,
                            &[lane],
                            &[a],
                            &[e3],
                            &[e4],
                            &[param],
                            mindeg,
                            &mut out,
                        )
                        .unwrap();
                    let mut expected = Vec::new();
                    for i in 0..=mindeg {
                        for j in 0..=i / 2 {
                            expected.push(get(&poly, (r - i as i32, 0, j as u16)));
                        }
                    }
                    // Nothing beyond the layout's mu² powers.
                    for (&(t, x, m), value) in &poly {
                        if x == 0 && t > r - 5 && (r - t) >= 0 && m as i32 > (r - t) / 2 {
                            assert!(value.abs() < 1e-12, "{}", context("t3 excess mu²"));
                        }
                    }
                    assert_close(&out, &expected, &context(&format!("t3 mindeg {mindeg}")));
                }

                // t2 expansions.
                let (a0, a1, e3, e4) = (rng.cmom(), rng.cmom(), rng.cmom(), rng.cmom());
                let p = [rng.c(), rng.c(), rng.c()];
                let poly = reference_mu_numerator(
                    &numerator,
                    lane,
                    &std::array::from_fn(|mu| {
                        linear(&[
                            ((0, 0, 0), a0[mu]),
                            ((0, 1, 0), a1[mu]),
                            ((1, 0, 0), e3[mu]),
                            ((-1, 0, 0), p[0] * e4[mu]),
                            ((-1, 1, 0), p[1] * e4[mu]),
                            ((-1, 2, 0), p[2] * e4[mu]),
                            ((-1, 0, 1), e4[mu]),
                        ])
                    }),
                    &mu2_poly,
                );
                const LAYOUT: [&[(u16, u16)]; 4] = [
                    &[(0, 0)],
                    &[(0, 0), (1, 0)],
                    &[(0, 0), (0, 1), (1, 0), (2, 0)],
                    &[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 0)],
                ];
                for mindeg in 0..=3 {
                    let mut out = vec![C64::ZERO; t2_len(mindeg)];
                    numerator
                        .t2_expansion(
                            &cut,
                            &[lane],
                            &[a0],
                            &[a1],
                            &[e3],
                            &[e4],
                            &[p],
                            mindeg,
                            &mut out,
                        )
                        .unwrap();
                    let mut expected = Vec::new();
                    for (i, order) in LAYOUT.iter().enumerate().take(mindeg + 1) {
                        for &(x, m) in order.iter() {
                            expected.push(get(&poly, (r - i as i32, x, m)));
                        }
                    }
                    assert_close(&out, &expected, &context(&format!("t2 mindeg {mindeg}")));
                }
            }
        }
    }
    assert!(MuTensorNumerator::new(3, 2, 1).is_err());
}

#[test]
fn mu_free_tensor_numerators_agree_bitwise() {
    // With no mu² powers the block sweep is the plain tensor sweep.
    let mut rng = Rng(23);
    let rank = 5;
    let mut plain = TensorNumerator::new(rank, 2);
    for c in plain.coefficients_mut() {
        *c = rng.c();
    }
    let mut mu = MuTensorNumerator::new(rank, 0, 2).unwrap();
    mu.coefficients_mut().copy_from_slice(plain.coefficients());
    let lanes = [1, 0];
    let (a, e3, e4): (Vec<CMom>, Vec<CMom>, Vec<CMom>) = (
        (0..2).map(|_| rng.cmom()).collect(),
        (0..2).map(|_| rng.cmom()).collect(),
        (0..2).map(|_| rng.cmom()).collect(),
    );
    let param: Vec<C64> = (0..2).map(|_| rng.c()).collect();
    for mindeg in 0..=4 {
        let mut x = vec![C64::ZERO; 2 * t3_len(mindeg)];
        let mut y = x.clone();
        plain
            .t3_expansion(&[0], &lanes, &a, &e3, &e4, &param, mindeg, &mut x)
            .unwrap();
        mu.t3_expansion(&[0], &lanes, &a, &e3, &e4, &param, mindeg, &mut y)
            .unwrap();
        assert_eq!(x, y);
    }
}

#[test]
fn tensor_numerator_batches_rows_by_lane() {
    let mut rng = Rng(11);
    let rank = 4;
    let mut numerator = TensorNumerator::new(rank, 4);
    for c in numerator.coefficients_mut() {
        *c = rng.c();
    }
    let lanes = [3, 0, 3, 1];
    let a: Vec<CMom> = (0..4).map(|_| rng.cmom()).collect();
    let e3: Vec<CMom> = (0..4).map(|_| rng.cmom()).collect();
    let e4: Vec<CMom> = (0..4).map(|_| rng.cmom()).collect();
    let param: Vec<C64> = (0..4).map(|_| rng.c()).collect();
    let mut batched = vec![C64::ZERO; 4 * t3_len(3)];
    numerator
        .t3_expansion(&[0], &lanes, &a, &e3, &e4, &param, 3, &mut batched)
        .unwrap();
    for row in 0..4 {
        let mut single = vec![C64::ZERO; t3_len(3)];
        numerator
            .t3_expansion(
                &[0],
                &lanes[row..row + 1],
                &a[row..row + 1],
                &e3[row..row + 1],
                &e4[row..row + 1],
                &param[row..row + 1],
                3,
                &mut single,
            )
            .unwrap();
        assert_eq!(
            &batched[row * single.len()..(row + 1) * single.len()],
            &single[..]
        );
    }
    assert!(
        numerator
            .evaluate(&[0], &[4], &a[..1], &param[..1], &mut [C64::ZERO])
            .is_err()
    );
}

/// Layout `(order, x power, mu² power)` of the t3/t2 expansions.
fn layout(t2: bool, nterms: usize) -> Vec<(usize, u16, u16)> {
    let mut out = Vec::new();
    for i in 0..nterms {
        if t2 {
            let order: &[(u16, u16)] = match i {
                0 => &[(0, 0)],
                1 => &[(0, 0), (1, 0)],
                2 => &[(0, 0), (0, 1), (1, 0), (2, 0)],
                _ => &[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 0)],
            };
            out.extend(order.iter().map(|&(x, m)| (i, x, m)));
        } else {
            out.extend((0..=i / 2).map(|j| (i, 0, j as u16)));
        }
    }
    out
}

#[test]
fn polynomial_division_matches_multiplication_back() {
    let mut rng = Rng(3);
    for t2 in [false, true] {
        for nterms in 1..=(if t2 { 4 } else { 5 }) {
            let entries = layout(t2, nterms);
            let numerator: Vec<C64> = entries.iter().map(|_| rng.c()).collect();
            let d0 = rng.c();
            let mut quotient = numerator.clone();
            // D = d0 t + d1 + d2/t with the stage-specific x and mu² structure.
            let mut d = Poly::new();
            d.insert((1, 0, 0), d0);
            if t2 {
                let (d1, d2) = ([rng.c(), rng.c()], [rng.c(), rng.c(), rng.c(), rng.c()]);
                divide_t2(&mut quotient, nterms, d0.inv(), d1, d2);
                d.insert((0, 0, 0), d1[0]);
                d.insert((0, 1, 0), d1[1]);
                d.insert((-1, 0, 0), d2[0]);
                d.insert((-1, 0, 1), d2[1]);
                d.insert((-1, 1, 0), d2[2]);
                d.insert((-1, 2, 0), d2[3]);
            } else {
                let (d1, d2) = (rng.c(), [rng.c(), rng.c()]);
                divide_t3(&mut quotient, nterms, d0.inv(), d1, d2);
                d.insert((0, 0, 0), d1);
                d.insert((-1, 0, 0), d2[0]);
                d.insert((-1, 0, 1), d2[1]);
            }
            // Quotient order i multiplies t^(r-1-i); compare D*Q with N on
            // the top nterms orders (t^(r-i) <-> -i).
            let q: Poly = entries
                .iter()
                .zip(&quotient)
                .map(|(&(i, x, m), &v)| ((-1 - i as i32, x, m), v))
                .collect();
            let product = poly_mul(&d, &q);
            let expected: Vec<C64> = numerator.clone();
            let actual: Vec<C64> = entries
                .iter()
                .map(|&(i, x, m)| get(&product, (-(i as i32), x, m)))
                .collect();
            assert_close(
                &actual,
                &expected,
                &format!("division t2={t2} nterms={nterms}"),
            );
        }
    }
}

/// Two real momenta span a plane containing timelike directions (as for
/// physical kinematics), so Ninja's basis construction is real.
fn physical_pair(k1: &RMom, k2: &RMom) -> bool {
    let k12 = mp_rr(k1, k2);
    k12 * k12 - mp2(k1) * mp2(k2) > 1e-6
}

pub(super) fn random_masses(rng: &mut Rng, n: usize) -> Vec<C64> {
    (0..n)
        .map(|_| C64::new(0.5 + rng.uniform().abs(), -0.1 * rng.uniform().abs()))
        .collect()
}

fn propagator(q: &CMom, v: &RMom, m2: C64, mu2: C64) -> C64 {
    let s = [q[0] + v[0], q[1] + v[1], q[2] + v[2], q[3] + v[3]];
    mp_cc(&s, &s) - m2 - mu2
}

fn assert_zero(value: C64, scale: f64, context: &str) {
    assert!(value.abs() <= 1e-10 * scale, "{context}: {value:?}");
}

#[test]
fn bases_are_massless_and_orthogonal() {
    let mut rng = Rng(5);
    let eps = 1e-9;
    let check = |e: &Basis| {
        let c = |v: &RMom| {
            [
                C64::real(v[0]),
                C64::real(v[1]),
                C64::real(v[2]),
                C64::real(v[3]),
            ]
        };
        for v in [&c(&e.e1), &c(&e.e2), &e.e3, &e.e4] {
            assert_zero(mp_cc(v, v), 1., "massless");
        }
        for v in [&e.e3, &e.e4] {
            assert_zero(mp_cr(v, &e.e1), 1., "orthogonal e1");
            assert_zero(mp_cr(v, &e.e2), 1., "orthogonal e2");
        }
        assert_zero(mp_cc(&e.e3, &e.e4) + e.mp12, 1., "e3.e4 = -e1.e2");
        assert_zero(C64::real(mp_rr(&e.e1, &e.e2) - e.mp12), 1., "mp12");
    };
    let mut tested = 0;
    while tested < 20 {
        let (k1, k2) = (rng.rmom(3.), rng.rmom(3.));
        if !physical_pair(&k1, &k2) {
            continue;
        }
        tested += 1;
        let e = Basis::from_pair(&k1, &k2, eps);
        check(&e);
        for mu in 0..4 {
            assert!((k1[mu] - (e.e1[mu] + e.r1 * e.e2[mu])).abs() < 1e-12 * 10.);
            assert!((k2[mu] - (e.e2[mu] + e.r2 * e.e1[mu])).abs() < 1e-12 * 10.);
        }
        let e = Basis::from_single(&k1, eps);
        check(&e);
    }
    for e in tadpole_candidates(eps) {
        check(&e);
    }
    // Massless inputs take the dedicated branches.
    let lightlike = [2., 0., 0., 2.];
    check(&Basis::from_pair(&lightlike, &[3., 1., 2., 0.5], eps));
    check(&Basis::from_single(&lightlike, eps));
}

/// Orthogonal massive momenta (k1.k2 = 0 up to rounding): Ninja's
/// `k1.k2 / (1 + r1 r2)` divides noise by noise. Box (4,5,6,7) of the any-n
/// verification fixture `n8r0genericm1`, lane 0, where the fast engine was off
/// by 3e-3 (and original Ninja by a different 3e-3).
#[test]
fn orthogonal_cut_momenta_keep_the_basis_normalization() {
    let v4 = [2. / 3., -0.25, 1. / 6., 8. / 3.];
    let (v5, v7) = ([8. / 3., -0.25, 1. / 6., 5. / 3.], [2., 0., 1., 0.]);
    let (k3, k0) = (sub(&v4, &v7), sub(&v5, &v4));
    assert!(mp_rr(&k3, &k0).abs() < 1e-15, "orthogonal up to rounding");
    let e = Basis::from_pair(&k3, &k0, 1e-9);
    let exact = mp_rr(&e.e1, &e.e2);
    assert!(
        (e.mp12 - exact).abs() <= 1e-12 * exact.abs(),
        "{} vs {exact}",
        e.mp12
    );
    let e34 = mp_cc(&e.e3, &e.e4);
    assert!((e34.re + e.mp12).abs() <= 1e-12 * exact.abs() && e34.im.abs() <= 1e-12);
}

/// Light-like momenta near the `z` axis. Ninja's spinors took the axis spinor
/// within `INFRARED_EPS` of an axis (dropping the transverse momentum) and
/// divided by a cancelling `p⁰ + p³` near `-z`; a beam along `-z` reached as a
/// difference of sums of momenta has a rounding residue there, which is above
/// `INFRARED_EPS` once the energies are large. The basis relations hold to
/// rounding (relative to the vectors' components) at every angle, energy sign
/// and scale; with Ninja's spinors 205 of these 432 checks failed, by up to
/// 4.4e-4 (9.4e15 for the beam along `-z` reached as a difference at
/// `E = 4e8`).
#[test]
fn bases_near_the_z_axis_are_massless_and_orthogonal() {
    let norm = |v: &CMom| v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    let worst = |e: &Basis| {
        let real = |v: &RMom| v.map(C64::real);
        let vectors = [real(&e.e1), real(&e.e2), e.e3, e.e4];
        let mut worst = 0f64;
        for i in 0..4 {
            for j in i..4 {
                let expected = match (i, j) {
                    (0, 1) => e.mp12,
                    (2, 3) => -e.mp12,
                    _ => 0.,
                };
                let (a, b) = (&vectors[i], &vectors[j]);
                let error = (mp_cc(a, b) - expected).abs() / (norm(a) * norm(b));
                worst = worst.max(error);
            }
        }
        worst
    };
    let eps = FastOptions::default().infrared_eps;
    let mut rng = Rng(17);
    let mut failures = Vec::new();
    let mut check = |label: String, k: RMom, other: RMom| {
        for (name, e) in [
            ("single", Basis::from_single(&k, eps)),
            ("pair", Basis::from_pair(&k, &other, eps)),
            ("reversed pair", Basis::from_pair(&other, &k, eps)),
        ] {
            let error = worst(&e);
            if !(error < 1e-13) {
                failures.push(format!("{label}, {name}: {error:e}"));
            }
        }
    };
    // (label, sign of p⁰, sign of p³): near +z and -z, and their negatives.
    let axes = [
        ("+z", 1., 1.),
        ("-z", 1., -1.),
        ("-(+z)", -1., -1.),
        ("-(-z)", -1., 1.),
    ];
    let phi = 0.7_f64;
    for theta in [0_f64, 1e-9, 1e-8, 1e-6, 1e-5, 1e-3, 0.1] {
        for energy in [1e-3, 1., 500., 4e8] {
            for (axis, sign, z) in axes {
                let (s, c) = (theta.sin(), theta.cos());
                let e = sign * energy;
                let k = [e, e * s * phi.cos(), e * s * phi.sin(), z * energy * c];
                let other = rng.rmom(energy);
                check(format!("{axis} at {theta:e}, E = {energy:e}"), k, other);
            }
        }
    }
    // A beam along -z (and +z) reached as the difference of two sums.
    for energy in [1., 500., 4e5, 4e8] {
        for beam in [[energy, 0., 0., -energy], [energy, 0., 0., energy]] {
            for i in 0..4 {
                let q = rng.rmom(energy);
                let k = sub(&[q[0] + beam[0], q[1], q[2], q[3] + beam[3]], &q);
                let other = rng.rmom(energy);
                check(format!("beam {beam:?} as a difference ({i})"), k, other);
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn cut_solutions_are_on_shell() {
    let mut rng = Rng(9);
    let eps = 1e-9;
    let mut tested = 0;
    while tested < 20 {
        let v: Vec<RMom> = (0..4).map(|_| rng.rmom(2.)).collect();
        let m = random_masses(&mut rng, 4);
        // Quadruple cut (k3 = V1-V4, k0 = V2-V1, k1 = V3-V2).
        let k0 = sub(&v[1], &v[0]);
        let k1 = sub(&v[2], &v[1]);
        let k3 = sub(&v[0], &v[3]);
        let k2 = sub(&v[0], &v[2]);
        if !physical_pair(&k3, &k0) || !physical_pair(&k2, &k0) {
            continue;
        }
        tested += 1;
        let e = Basis::from_pair(&k3, &k0, eps);
        for l in box_solutions(&k3, &k0, &k1, &e, m[0], m[1], m[2], m[3]) {
            let q = [
                l[0] - v[0][0],
                l[1] - v[0][1],
                l[2] - v[0][2],
                l[3] - v[0][3],
            ];
            for i in 0..4 {
                assert_zero(propagator(&q, &v[i], m[i], C64::ZERO), 10., "box cut");
            }
        }
        // Triple cut with the (t, mu²) parametrization.
        let e = Basis::from_pair(&k2, &k0, eps);
        let (x1, x2, param) = triangle_solution(&k2, &k0, &e, m[0], m[1], m[2]);
        let a = combine_real2(x1, &e.e1, x2, &e.e2);
        let (t, mu2) = (rng.c(), rng.c());
        let s = (param + mu2) / t / (2. * e.mp34());
        let q: CMom = std::array::from_fn(|mu| a[mu] - v[0][mu] + t * e.e3[mu] + s * e.e4[mu]);
        for i in 0..3 {
            assert_zero(propagator(&q, &v[i], m[i], mu2), 10., "triangle cut");
        }
        // Double cut with the (t, x, mu²) parametrization.
        let k = sub(&v[0], &v[1]);
        let e = Basis::from_single(&k, eps);
        let (b0, b1, p) = bubble_solution(&k, &e, m[0], m[1]);
        let (t, x, mu2) = (rng.c(), rng.c(), rng.c());
        let s = (p[0] + p[1] * x + p[2] * x * x + mu2) / t / (2. * e.mp34());
        let q: CMom = std::array::from_fn(|mu| {
            b0 * e.e2[mu] - v[0][mu] + x * (e.e1[mu] + b1 * e.e2[mu]) + t * e.e3[mu] + s * e.e4[mu]
        });
        for i in 0..2 {
            assert_zero(propagator(&q, &v[i], m[i], mu2), 10., "bubble cut");
        }
        // Single cut.
        for e in tadpole_candidates(eps) {
            let (t, mu2) = (rng.c(), rng.c());
            let s = (m[0] + mu2) / t / (2. * e.mp34());
            let q: CMom = std::array::from_fn(|mu| -v[0][mu] + t * e.e3[mu] + s * e.e4[mu]);
            assert_zero(propagator(&q, &v[0], m[0], mu2), 10., "tadpole cut");
        }
        let _ = mp2(&k);
    }
}

/// Masters with unit values; rows of lane-tagged arguments can be failed or
/// poisoned through the squared scale (`mu² = 7` fails, `mu² = 9` gives NaN).
pub(super) struct UnitMasters;

impl FastMasters for UnitMasters {
    fn evaluate(
        &mut self,
        family: FastMaster,
        args: &[C64],
        out: &mut [[C64; 3]],
        ok: &mut [bool],
    ) -> crate::Result<()> {
        let width = family.outputs();
        for (row, point) in args.chunks_exact(family.arity()).enumerate() {
            let mu2 = point[point.len() - 1].re;
            ok[row] = mu2 != 7.;
            let value = if mu2 == 9. { f64::NAN } else { 1. };
            for k in 0..width {
                out[row * width + k] = [C64::real(value), C64::real(0.5), C64::ZERO];
            }
        }
        Ok(())
    }
}

pub(super) fn unit_tensor(rank: usize, lanes: usize) -> TensorNumerator {
    let mut tensor = TensorNumerator::new(rank, lanes);
    for (k, c) in tensor.coefficients_mut().iter_mut().enumerate() {
        *c = C64::new(1. / (1 + k % 7) as f64, 0.1 * (k % 3) as f64);
    }
    tensor
}

#[test]
fn lane_statuses_isolate_failures_and_keep_previous_results() {
    let n = 4;
    let rank = 3;
    // Lane 0 regular; lane 1 collinear (vanishing Gram determinant);
    // lane 2 failing masters; lane 3 non-finite masters; lane 4 regular.
    let regular: Vec<RMom> = vec![
        [0., 0., 0., 0.],
        [1.902, -0.517, -0.537, -0.876],
        [-1.599, 0.29, 0.378, 0.142],
        [-0.892, -0.823, 0.499, 0.629],
    ];
    let collinear: Vec<RMom> = vec![
        [0., 0., 0., 0.],
        [1., 0., 0., 0.],
        [2., 0., 0., 0.],
        [3., 0.5, 0., 0.],
    ];
    let lanes = 5;
    let mut shifts = Vec::new();
    for lane in 0..lanes {
        shifts.extend_from_slice(if lane == 1 { &collinear } else { &regular });
    }
    let masses = vec![C64::new(0.7, -0.05); lanes * n];
    let scales = [1., 1., 7., 9., 2.];
    let input = FastInput {
        lanes,
        shifts: &shifts,
        masses_squared: &masses,
        scales_squared: &scales,
        s_matrix: None,
    };
    let mut amplitude = FastAmplitude::new(n, rank, FastOptions::default()).unwrap();
    let sentinel = FastResult {
        laurent: [C64::real(42.); 3],
        cut_constructible: C64::real(42.),
        rational: C64::real(42.),
    };
    let mut results = vec![sentinel; lanes];
    let mut statuses = vec![FastStatus::NotEvaluated; lanes];
    let mut tensor = unit_tensor(rank, lanes);
    amplitude
        .evaluate(
            &input,
            &mut tensor,
            &mut UnitMasters,
            &mut results,
            &mut statuses,
        )
        .unwrap();
    assert_eq!(
        statuses,
        [
            FastStatus::Success,
            FastStatus::Unstable,
            FastStatus::MasterFailed,
            FastStatus::NonFinite,
            FastStatus::Success
        ]
    );
    for (lane, result) in results.iter().enumerate().take(4).skip(1) {
        assert_eq!(*result, sentinel, "failed lane {lane} must keep its result");
    }

    assert_ne!(results[0], sentinel);
    // Same kinematics and coefficients: lane 4 differs from lane 0 only by mu².
    assert_ne!(results[4], sentinel);
    let finite = results[0];
    assert!((finite.cut_constructible + finite.rational - finite.laurent[0]).abs() < 1e-12);
    // A second evaluation overwrites (does not accumulate).
    amplitude
        .evaluate(
            &input,
            &mut tensor,
            &mut UnitMasters,
            &mut results,
            &mut statuses,
        )
        .unwrap();
    assert_eq!(results[0], finite);
    // Structural errors leave everything untouched.
    let before = results.clone();
    assert!(
        amplitude
            .evaluate(
                &FastInput {
                    lanes,
                    shifts: &shifts[..lanes * n - 1],
                    ..input
                },
                &mut tensor,
                &mut UnitMasters,
                &mut results,
                &mut statuses,
            )
            .is_err()
    );
    assert_eq!(results, before);
    let mut wrong_rank = unit_tensor(rank + 1, lanes);
    assert!(
        amplitude
            .evaluate(
                &input,
                &mut wrong_rank,
                &mut UnitMasters,
                &mut results,
                &mut statuses
            )
            .is_err()
    );
}

#[test]
fn replay_reproduces_recorded_evaluations_bitwise() {
    for (n, rank) in [(1, 2), (2, 1), (3, 4), (4, 5), (5, 4), (7, 6), (8, 9)] {
        let mut rng = Rng(100 + n as u64);
        let lanes = 3;
        let shifts: Vec<RMom> = (0..lanes * n)
            .map(|i| if i % n == 0 { [0.; 4] } else { rng.rmom(2.) })
            .collect();
        let masses: Vec<C64> = (0..lanes * n)
            .map(|_| C64::new(0.5 + rng.uniform().abs(), -0.1 * rng.uniform().abs()))
            .collect();
        let scales = vec![1.5; lanes];
        let input = FastInput {
            lanes,
            shifts: &shifts,
            masses_squared: &masses,
            scales_squared: &scales,
            s_matrix: None,
        };
        let mut amplitude = FastAmplitude::new(n, rank, FastOptions::default()).unwrap();
        let mut tensor = TensorNumerator::new(rank, lanes);
        for c in tensor.coefficients_mut() {
            *c = rng.c();
        }
        let mut numerator = RecordingNumerator::new(tensor);
        let mut masters = RecordingMasters::new(UnitMasters);
        let mut results = vec![FastResult::default(); lanes];
        let mut statuses = vec![FastStatus::NotEvaluated; lanes];
        amplitude
            .evaluate(
                &input,
                &mut numerator,
                &mut masters,
                &mut results,
                &mut statuses,
            )
            .unwrap();
        let (recorded, recorded_statuses) = (results.clone(), statuses.clone());
        let mut replay_numerator = numerator.replay();
        let mut replay_masters = masters.replay();
        for _ in 0..2 {
            results.fill(FastResult::default());
            amplitude
                .evaluate(
                    &input,
                    &mut replay_numerator,
                    &mut replay_masters,
                    &mut results,
                    &mut statuses,
                )
                .unwrap();
            assert_eq!(results, recorded, "n={n} rank={rank}");
            assert_eq!(statuses, recorded_statuses);
        }
    }
}

/// `FastOptions::conditioned_cuts` chooses per cut and lane, so a batch is
/// bitwise identical to its lanes evaluated alone; rank n+2 already uses the
/// conditioned geometry and ignores the option.
#[test]
fn conditioned_cuts_are_per_lane_and_ignored_at_rank_n2() {
    let conditioned = FastOptions {
        conditioned_cuts: true,
        ..FastOptions::default()
    };
    for (n, rank) in [(3, 4), (5, 5), (5, 6), (7, 8), (5, 7)] {
        let mut rng = Rng(200 + (n * 10 + rank) as u64);
        let lanes = 4;
        // Sums of future timelike momenta: every cut spans a Minkowski plane.
        let mut shifts: Vec<RMom> = Vec::new();
        for _ in 0..lanes {
            let mut v = [0.; 4];
            for _ in 0..n {
                shifts.push(v);
                let p = rng.rmom(1.);
                v = [
                    v[0] + 2. + p[0].abs(),
                    v[1] + p[1],
                    v[2] + p[2],
                    v[3] + p[3],
                ];
            }
        }
        let masses: Vec<C64> = (0..lanes * n)
            .map(|_| C64::new(0.5 + rng.uniform().abs(), -0.1 * rng.uniform().abs()))
            .collect();
        let scales = vec![1.5; lanes];
        let coefficients: Vec<C64> = (0..lanes * ncoeff(rank)).map(|_| rng.c()).collect();
        let run = |options: FastOptions, lanes_run: &[usize]| {
            let count = lanes_run.len();
            let shifts: Vec<RMom> = lanes_run
                .iter()
                .flat_map(|&l| shifts[l * n..(l + 1) * n].iter().copied())
                .collect();
            let masses: Vec<C64> = lanes_run
                .iter()
                .flat_map(|&l| masses[l * n..(l + 1) * n].iter().copied())
                .collect();
            let mut tensor = TensorNumerator::new(rank, count);
            let size = ncoeff(rank);
            for (i, &l) in lanes_run.iter().enumerate() {
                tensor.coefficients_mut()[i * size..(i + 1) * size]
                    .copy_from_slice(&coefficients[l * size..(l + 1) * size]);
            }
            let input = FastInput {
                lanes: count,
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: &scales[..count],
                s_matrix: None,
            };
            let mut amplitude = FastAmplitude::new(n, rank, options).unwrap();
            let mut results = vec![FastResult::default(); count];
            let mut statuses = vec![FastStatus::NotEvaluated; count];
            amplitude
                .evaluate(
                    &input,
                    &mut tensor,
                    &mut UnitMasters,
                    &mut results,
                    &mut statuses,
                )
                .unwrap();
            (results, statuses)
        };
        let all: Vec<usize> = (0..lanes).collect();
        let batch = run(conditioned, &all);
        for lane in 0..lanes {
            let alone = run(conditioned, &[lane]);
            assert_eq!(
                (alone.0[0], alone.1[0]),
                (batch.0[lane], batch.1[lane]),
                "n={n} rank={rank} lane {lane}"
            );
        }
        let default = run(FastOptions::default(), &all);
        if rank == n + 2 {
            assert_eq!(default, batch, "n={n} rank={rank}");
        } else {
            assert!(batch.1.iter().all(|&s| s == FastStatus::Success));
            assert_ne!(default.0, batch.0, "n={n} rank={rank}: no cut changed");
        }
    }
}
