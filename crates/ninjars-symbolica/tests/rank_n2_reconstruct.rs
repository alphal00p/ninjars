//! V2(b): on-cut reconstruction of every rank n+2 residue of the generic
//! engine, n = 1..6.
//!
//! Random numerators of full rank n+2, with and without explicit mu² terms,
//! are reduced with complete box coefficients (`BoxCoefficientMode::Full`).
//! At fresh points of each cut S (box: `q = a + tau v_orth` with
//! `mu² = tau² v_orth² - beta`; triangle and bubble: their own Laurent
//! parametrisations at random t, x and mu²) the numerator must equal
//! `sum_{T ⊇ S} Delta_T(q, mu²) prod_{l ∉ T} D_l(q, mu²)`, with every residue
//! evaluated from its coefficients through the canonical monomial tables
//! (`rank_n2_support`). Tadpoles keep only their integrated coefficients:
//! `Delta_i = (N - sum_{T ⊋ {i}} Delta_T prod D) / prod_{l != i} D_l` is
//! expanded independently on both T3 families `q = a + t e1 + (beta + mu²)
//! e2_eff / t` and `q = a + t e3 + (beta + mu²) e4_eff / t` by a four-point
//! discrete Fourier projection (Delta_i has t powers -3..3), whose t⁰ parts
//! must be `c0 + c14 mu²` and `c0 + c14 mu² + c15 [t⁰](q̄·e3 q̄·e4)`.
//!
//! Float(256): <= 1e-60 relative to the largest term; binary64: <= 1e-9.
//! The pruned swapped bubble arithmetic is checked the same way.

mod rank_n2_support;

use std::collections::BTreeMap;

use ninjars::fast::C64;
use ninjars::{
    BoxCoefficientMode, BoxMuMode, Complex, CutOutcome, Float, FourMomentum, Numerator,
    NumeratorRequest, Real, SampleId,
};
use oneloop::NativeFloat;
use rank_n2_support::{
    Problem, Rng, ZeroMasters, box_value, bubble_value, carriers, complex, denominator, distance,
    modulus, options, reduce_with, triangle_value,
};
use symbolica::evaluate::EvaluationDomain;

/// A residue of a strict higher cut, as a function of (q, mu²).
enum Residue<T> {
    Pentagon(Complex<T>),
    Box {
        origin: FourMomentum<Complex<T>>,
        v: FourMomentum<Complex<T>>,
        c: Vec<Complex<T>>,
    },
    Triangle {
        origin: FourMomentum<Complex<T>>,
        e3: FourMomentum<Complex<T>>,
        e4: FourMomentum<Complex<T>>,
        c: Vec<Complex<T>>,
    },
    Bubble {
        origin: FourMomentum<Complex<T>>,
        e2: FourMomentum<Complex<T>>,
        e3: FourMomentum<Complex<T>>,
        e4: FourMomentum<Complex<T>>,
        c: Vec<Complex<T>>,
    },
}

impl<T: NativeFloat> Residue<T>
where
    Complex<T>: Real,
{
    fn value(&self, q: &FourMomentum<Complex<T>>, mu: &Complex<T>) -> Complex<T> {
        match self {
            Self::Pentagon(c) => c.clone() * mu,
            Self::Box { origin, v, c } => {
                let r = q.clone() + origin.clone();
                box_value(c, &r.dot(v), mu)
            }
            Self::Triangle { origin, e3, e4, c } => {
                let r = q.clone() + origin.clone();
                triangle_value(c, &r.dot(e3), &r.dot(e4), mu)
            }
            Self::Bubble {
                origin,
                e2,
                e3,
                e4,
                c,
            } => {
                let r = q.clone() + origin.clone();
                bubble_value(c, &r.dot(e2), &r.dot(e3), &r.dot(e4), mu)
            }
        }
    }
}

/// Worst relative reconstruction error per cut size `[tadpole, bubble,
/// triangle, box]` (zero where the topology has no such cut).
pub fn reconstruction<T>(problem: &Problem, prototype: T, pruned: bool) -> [f64; 4]
where
    T: NativeFloat + Into<Float>,
    Complex<T>: Real + EvaluationDomain,
{
    let p = &prototype;
    let n = problem.n;
    let mut numerator = problem.numerator(p);
    let amplitude = reduce_with(
        problem,
        &mut numerator,
        p,
        options(BoxCoefficientMode::Full, BoxMuMode::Expansion),
        &mut ZeroMasters,
        |kernels| {
            if pruned {
                kernels.with_pruned_swapped_arithmetic()
            } else {
                kernels
            }
        },
    );
    let topology = amplitude.topology().clone();
    let (shifts, masses) = carriers(problem, p);
    let lanes = problem.count();
    let shift = |lane: usize, line: usize| shifts[lane * n + line].clone();
    let mass = |lane: usize, line: usize| masses[lane * n + line].clone();
    // Residues of every cut of size >= 2, by (cut index, lane).
    let mut residues: BTreeMap<(usize, usize), Residue<T>> = BTreeMap::new();
    for outcome in amplitude.pentagons() {
        let CutOutcome::Stable(r) = outcome else {
            panic!("unstable pentagon")
        };
        residues.insert(
            (r.cut_index, r.id.lane),
            Residue::Pentagon(r.coefficient.clone()),
        );
    }
    for outcome in amplitude.boxes() {
        let CutOutcome::Stable(r) = outcome else {
            panic!("unstable box")
        };
        assert_eq!(r.coefficients.len(), 7);
        assert!(r.computed.iter().all(|&c| c));
        let first = topology.cuts()[r.cut_index].partition[0];
        residues.insert(
            (r.cut_index, r.id.lane),
            Residue::Box {
                origin: shift(r.id.lane, first),
                v: r.geometry.v_orth.clone(),
                c: r.coefficients.clone(),
            },
        );
    }
    for outcome in amplitude.triangles() {
        let CutOutcome::Stable(r) = outcome else {
            panic!("unstable triangle")
        };
        assert_eq!(r.coefficients.len(), 21);
        let first = topology.cuts()[r.cut_index].partition[0];
        residues.insert(
            (r.cut_index, r.id.lane),
            Residue::Triangle {
                origin: shift(r.id.lane, first),
                e3: r.geometry.basis.e3.clone(),
                e4: r.geometry.basis.e4.clone(),
                c: r.coefficients.clone(),
            },
        );
    }
    for outcome in amplitude.bubbles() {
        let CutOutcome::Stable(r) = outcome else {
            panic!("unstable bubble")
        };
        assert_eq!(r.coefficients.len(), 35);
        let first = topology.cuts()[r.cut_index].partition[0];
        residues.insert(
            (r.cut_index, r.id.lane),
            Residue::Bubble {
                origin: shift(r.id.lane, first),
                e2: r.geometry.basis.e2.clone(),
                e3: r.geometry.basis.e3.clone(),
                e4: r.geometry.basis.e4.clone(),
                c: r.coefficients.clone(),
            },
        );
    }
    let mut rng = Rng(0x2ec0_0000 ^ n as u64);
    let mut sample = 0usize;
    let mut evaluate_numerator = |lane: usize, q: &FourMomentum<Complex<T>>, mu: &Complex<T>| {
        sample += 1;
        let mut value = [complex(p, C64::ZERO)];
        numerator
            .evaluate_batch(
                &[NumeratorRequest {
                    id: SampleId { lane, sample },
                    q,
                    mu_squared: mu,
                    partition: &[],
                }],
                &mut value,
            )
            .unwrap();
        let [value] = value;
        value
    };
    // sum_{T ⊇ S, T != S} Delta_T prod_{l ∉ T} D_l and the largest term.
    let parents = |index: usize, lane: usize, q: &FourMomentum<Complex<T>>, mu: &Complex<T>| {
        let cut = &topology.cuts()[index];
        let mut sum = complex(p, C64::ZERO);
        let mut largest = 0_f64;
        for &parent in &cut.parents {
            let residue = &residues[&(parent, lane)];
            let mut term = residue.value(q, mu);
            for &line in &topology.cuts()[parent].complement {
                term *= &denominator(q, &shift(lane, line), &mass(lane, line), mu);
            }
            largest = largest.max(modulus(&term));
            sum += &term;
        }
        (sum, largest)
    };
    let random = |rng: &mut Rng| {
        let mut z = rng.complex();
        z.re += if z.re < 0. { -0.5 } else { 0.5 };
        complex(p, z)
    };
    let mut worst = [0_f64; 4];
    let mut cut_residual = 0_f64;
    for (index, cut) in topology.cuts().iter().enumerate() {
        let size = cut.partition.len();
        if size == 5 {
            continue;
        }
        for lane in 0..lanes {
            let on_cut = |q: &FourMomentum<Complex<T>>, mu: &Complex<T>| {
                cut.partition
                    .iter()
                    .map(|&line| {
                        modulus(&denominator(q, &shift(lane, line), &mass(lane, line), mu))
                    })
                    .fold(0., f64::max)
            };
            if size == 1 {
                let outcome = amplitude
                    .tadpoles()
                    .iter()
                    .find(|o| matches!(o, CutOutcome::Stable(r) if r.cut_index == index && r.id.lane == lane))
                    .expect("stable tadpole");
                let CutOutcome::Stable(r) = outcome else {
                    unreachable!()
                };
                let g = &r.geometry;
                let origin = shift(lane, cut.partition[0]);
                let [c0, c14, c15] = [0, 14, 15].map(|i| r.coefficients[i].clone());
                let i = complex(p, C64::new(0., 1.));
                let one = complex(p, C64::real(1.));
                let roots = [one.clone(), i.clone(), -one.clone(), -i.clone()];
                for _ in 0..2 {
                    let mu = random(&mut rng);
                    let rho = random(&mut rng);
                    for (e, e_eff, transverse) in [
                        (&g.basis.e1, &g.e2_effective, false),
                        (&g.basis.e3, &g.e4_effective, true),
                    ] {
                        let mut remainder = complex(p, C64::ZERO);
                        let mut monomial = complex(p, C64::ZERO);
                        let mut largest = 0_f64;
                        for root in &roots {
                            let t = rho.clone() * root;
                            let q = g.a.clone()
                                + e.clone() * t.clone()
                                + e_eff.clone() * ((g.beta.clone() + &mu) / &t);
                            cut_residual =
                                cut_residual.max(on_cut(&q, &mu) / (1. + modulus(&t).powi(2)));
                            let value = evaluate_numerator(lane, &q, &mu);
                            let (sum, parent_largest) = parents(index, lane, &q, &mu);
                            let mut uncut = complex(p, C64::real(1.));
                            for &line in &cut.complement {
                                uncut *=
                                    &denominator(&q, &shift(lane, line), &mass(lane, line), &mu);
                            }
                            let delta = (value.clone() - &sum) / &uncut;
                            largest = largest
                                .max(modulus(&value) / modulus(&uncut))
                                .max(parent_largest / modulus(&uncut))
                                .max(modulus(&delta));
                            remainder += &delta;
                            let r = q.clone() + origin.clone();
                            monomial += &(r.dot(&g.basis.e3) * &r.dot(&g.basis.e4));
                        }
                        let four = complex(p, C64::real(4.));
                        let remainder = remainder / &four;
                        let mut expected = c0.clone() + c14.clone() * &mu;
                        if transverse {
                            expected += c15.clone() * &(monomial / &four);
                        }
                        worst[0] = worst[0].max(distance(&remainder, &expected) / largest);
                    }
                }
                continue;
            }
            let own = &residues[&(index, lane)];
            for _ in 0..3 {
                let (q, mu) = match own {
                    Residue::Box { .. } => {
                        let CutOutcome::Stable(r) = amplitude
                            .boxes()
                            .iter()
                            .find(|o| matches!(o, CutOutcome::Stable(r) if r.cut_index == index && r.id.lane == lane))
                            .unwrap()
                        else {
                            unreachable!()
                        };
                        let g = &r.geometry;
                        let tau = random(&mut rng);
                        let q = g.a.clone() + g.v_orth.clone() * tau.clone();
                        let mu = tau.clone() * &tau * &g.v_orth_squared - &g.beta;
                        (q, mu)
                    }
                    Residue::Triangle { .. } => {
                        let CutOutcome::Stable(r) = amplitude
                            .triangles()
                            .iter()
                            .find(|o| matches!(o, CutOutcome::Stable(r) if r.cut_index == index && r.id.lane == lane))
                            .unwrap()
                        else {
                            unreachable!()
                        };
                        let g = &r.geometry;
                        let (t, mu) = (random(&mut rng), random(&mut rng));
                        let q = g.a.clone()
                            + g.basis.e3.clone() * t.clone()
                            + g.e4_effective.clone() * ((g.beta.clone() + &mu) / &t);
                        (q, mu)
                    }
                    Residue::Bubble { .. } => {
                        let CutOutcome::Stable(r) = amplitude
                            .bubbles()
                            .iter()
                            .find(|o| matches!(o, CutOutcome::Stable(r) if r.cut_index == index && r.id.lane == lane))
                            .unwrap()
                        else {
                            unreachable!()
                        };
                        let g = &r.geometry;
                        let (t, x, mu) = (random(&mut rng), random(&mut rng), random(&mut rng));
                        let [b0, b1, b2] = &g.parameters;
                        let beta = b0.clone() + b1.clone() * &x + b2.clone() * &x * &x;
                        let q = g.a0.clone()
                            + g.a1.clone() * x.clone()
                            + g.basis.e3.clone() * t.clone()
                            + g.e4_effective.clone() * ((beta + &mu) / &t);
                        (q, mu)
                    }
                    Residue::Pentagon(_) => unreachable!(),
                };
                cut_residual = cut_residual
                    .max(on_cut(&q, &mu) / (1. + q.dot(&q).re.to_f64().abs() + modulus(&mu)));
                let value = evaluate_numerator(lane, &q, &mu);
                let (sum, mut largest) = parents(index, lane, &q, &mu);
                let mut term = own.value(&q, &mu);
                for &line in &cut.complement {
                    term *= &denominator(&q, &shift(lane, line), &mass(lane, line), &mu);
                }
                largest = largest.max(modulus(&term)).max(modulus(&value));
                let total = sum + &term;
                worst[size - 1] = worst[size - 1].max(distance(&value, &total) / largest);
            }
        }
    }
    // The sampled points lie on their cuts (to the carrier's precision).
    assert!(
        cut_residual <= 1e6 * prototype_epsilon(p),
        "points off their cuts: {cut_residual:e}"
    );
    worst
}

fn prototype_epsilon<T: NativeFloat>(p: &T) -> f64 {
    2_f64.powi(-(p.get_precision() as i32))
}

fn check(n: usize, mu_terms: bool, pruned: bool) {
    let problem = Problem::new(n, n + 2, 0x4ec0 + 17 * n as u64, mu_terms, stride(n));
    let started = std::time::Instant::now();
    let float = reconstruction(&problem, Float::new(256), pruned);
    let binary64 = reconstruction(&problem, 0f64, pruned);
    eprintln!(
        "RANK_N2_RECONSTRUCT n={n} rank={} mu={mu_terms} pruned={pruned} terms={} \
         [tadpole, bubble, triangle, box] float256={float:?} binary64={binary64:?} ({:.1}s)",
        n + 2,
        problem.terms.len(),
        started.elapsed().as_secs_f64()
    );
    for (size, (f, b)) in float.iter().zip(&binary64).enumerate() {
        assert!(*f <= 1e-60, "n={n} cut size {}: Float(256) {f:e}", size + 1);
        assert!(*b <= 1e-9, "n={n} cut size {}: binary64 {b:e}", size + 1);
    }
}

/// Top-weight term stride: dense through rank 5, sparser above (Symbolica's
/// preparation of the polynomial expansions grows steeply with the terms).
fn stride(n: usize) -> usize {
    match n + 2 {
        0..=5 => 1,
        6 => 3,
        _ => 8,
    }
}

#[test]
fn rank_n2_reconstruction_n1_to_n3() {
    for n in 1..=3 {
        check(n, true, false);
        check(n, false, false);
    }
}

#[test]
fn rank_n2_reconstruction_n4() {
    check(4, true, false);
    check(4, false, false);
}

#[test]
fn rank_n2_reconstruction_n5() {
    check(5, true, false);
}

#[test]
fn rank_n2_reconstruction_n6() {
    check(6, true, false);
}

/// Pruned swapped bubble arithmetic (projection-only positive t powers of the
/// swapped orientation) reconstructs identically.
#[test]
fn rank_n2_reconstruction_pruned_swapped_bubbles() {
    for n in [2, 3, 4] {
        check(n, true, true);
    }
}
