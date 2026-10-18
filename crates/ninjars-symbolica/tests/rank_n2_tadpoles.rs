//! Rank n+2 tadpoles in the generic engine.
//!
//! The integrated tadpole layout stays Ninja's 16 entries; only c0, c14 and
//! c15 are computed, from two T3 families of six slots. The new degree-three
//! terms are odd in l = q + V and never reach t⁰.
//!
//! - Single propagator at rank 3, numerator
//!   `A + B mu² + C (l·p)² + D (l·k)³ + E mu² (l·k) + F (l·k)(l·p)²`:
//!   c0 + c14 mu² + c15 (l·e3)(l·e4) reproduces the even part on the cut, and
//!   the integral is `A A0 + B m²/2 + C p² m (A0 + m/2)/4` (the odd terms
//!   integrate to zero), with exact A0, in Float(256) to <= 1e-60 and
//!   binary64 to <= 1e-12. (The tadpole reconstruction against independent
//!   T3 expansions for n = 1..6 is in `rank_n2_reconstruct`.)
//! - Massless tadpoles and scaleless bubbles keep integrating to zero.

mod rank_n2_support;

use ninjars::fast::C64;
use ninjars::{
    AmplitudeInput, Complex, CutOutcome, EvaluationStatus, Float, FourMomentum, Numerator,
    NumeratorRequest, Real, SampleId,
};
use ninjars_symbolica::{
    PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaHighCuts,
    SymbolicaMasters,
};
use oneloop::NativeFloat;
use rank_n2_support::{complex, distance, laurent_error, modulus, parse, variables};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::rational::Rational,
    evaluate::EvaluationDomain,
};

/// Exact dyadic components.
fn momentum(v: [f64; 4]) -> FourMomentum<Atom> {
    let [a, b, c, d] = v.map(|x| Atom::num(Rational::try_from(x).unwrap()));
    FourMomentum::from_args(a, b, c, d)
}

/// Results and tadpole coefficients for the single-propagator numerator.
fn single<T>(prototype: T) -> f64
where
    T: NativeFloat + Into<Float>,
    Complex<T>: Real + EvaluationDomain,
{
    let p = &prototype;
    let shift = [0.5, -0.25, 0.75, 0.125];
    let (pv, kv) = ([1.5, 0.5, -0.25, 1.0], [0.25, 1.0, 0.5, -0.75]);
    let [a, b, c, d, e, f] = [
        C64::new(0.75, -0.5),
        C64::new(-1.25, 0.25),
        C64::new(0.5, 1.5),
        C64::new(2.0, -0.75),
        C64::new(-0.375, 1.0),
        C64::new(1.125, 0.625),
    ];
    let q = variables();
    let l = FourMomentum::from_args(q[0].clone(), q[1].clone(), q[2].clone(), q[3].clone())
        + momentum(shift);
    let (lp, lk) = (l.dot(&momentum(pv)), l.dot(&momentum(kv)));
    let names = ["a", "b", "c", "d", "e", "f"].map(parse);
    let expression = &names[0]
        + &names[1] * &q[4]
        + &names[2] * lp.pow(2)
        + &names[3] * lk.pow(3)
        + &names[4] * &q[4] * &lk
        + &names[5] * &lk * lp.pow(2);
    let mut numerator =
        PolynomialNumerator::new(expression, variables(), names.to_vec(), p.clone()).unwrap();
    assert_eq!(numerator.rank(), 3);
    let masses = [C64::new(1.5, -0.25), C64::real(0.75)];
    let lanes = masses.len();
    let values: Vec<_> = (0..lanes)
        .flat_map(|_| [a, b, c, d, e, f])
        .map(|z| complex(p, z))
        .collect();
    numerator.set_parameter_batch(&values, lanes).unwrap();
    let real = |x: f64| complex(p, C64::real(x));
    let shifts: Vec<_> = (0..lanes)
        .map(|_| FourMomentum::from(shift.map(real)))
        .collect();
    let mass: Vec<_> = masses.iter().map(|&m| complex(p, m)).collect();
    let scales: Vec<_> = (0..lanes).map(|i| real([1.0, 2.5][i])).collect();
    let mut amplitude = SymbolicaAmplitudeKernels::new(1, 3, p.clone())
        .unwrap()
        .into_amplitude(lanes)
        .unwrap();
    let mut low = SymbolicaCuts::new(p.clone()).unwrap();
    let mut high = SymbolicaHighCuts::new(p.clone()).unwrap();
    let mut masters = SymbolicaMasters::new(p.clone());
    amplitude
        .evaluate_batch_with_inputs(
            AmplitudeInput {
                shifts: &shifts,
                masses_squared: &mass,
                scales_squared: &scales,
                sampling_scales: &scales,
                s_matrix: None,
            },
            &mut numerator,
            &mut low,
            &mut high,
            &mut masters,
        )
        .unwrap();
    let mut worst = 0_f64;
    let p2 = pv[0] * pv[0] - pv[1] * pv[1] - pv[2] * pv[2] - pv[3] * pv[3];
    for lane in 0..lanes {
        assert_eq!(amplitude.statuses()[lane], EvaluationStatus::Success);
        let CutOutcome::Stable(residue) = &amplitude.tadpoles()[lane] else {
            panic!("unstable tadpole")
        };
        assert_eq!(residue.coefficients.len(), 16);
        let computed: Vec<_> = (0..16).map(|i| matches!(i, 0 | 14 | 15)).collect();
        assert_eq!(residue.computed, computed);
        // On the transverse family q = a + t e3 + (beta + mu²) e4_eff / t
        // (l = q + V), the t⁰ part of the numerator (the mean over
        // t in rho {1, i, -1, -i}; its powers are -3..3) is
        // c0 + c14 mu² + c15 (l·e3)(l·e4), with (l·e3)(l·e4) independent of t.
        let g = &residue.geometry;
        let i = complex(p, C64::new(0., 1.));
        for rho in [C64::new(0.75, 0.5), C64::new(-1.25, 0.25)] {
            for mu in [C64::new(0.5, -0.25), C64::new(-0.75, 1.0)] {
                let (rho, mu) = (complex(p, rho), complex(p, mu));
                let points: Vec<_> = [real(1.), i.clone(), real(-1.), -i.clone()]
                    .into_iter()
                    .map(|root| {
                        let t = rho.clone() * &root;
                        g.a.clone()
                            + g.basis.e3.clone() * t.clone()
                            + g.e4_effective.clone() * ((g.beta.clone() + &mu) / &t)
                    })
                    .collect();
                let mut value = vec![complex(p, C64::ZERO); points.len()];
                let requests: Vec<_> = points
                    .iter()
                    .enumerate()
                    .map(|(sample, q)| NumeratorRequest {
                        id: SampleId { lane, sample },
                        q,
                        mu_squared: &mu,
                        partition: &[0],
                    })
                    .collect();
                numerator.evaluate_batch(&requests, &mut value).unwrap();
                let constant =
                    value.iter().fold(complex(p, C64::ZERO), |sum, v| sum + v) / &real(4.);
                let r = points[0].clone() + shifts[lane].clone();
                let expected = residue.coefficients[0].clone()
                    + residue.coefficients[14].clone() * &mu
                    + residue.coefficients[15].clone()
                        * &(r.dot(&g.basis.e3) * &r.dot(&g.basis.e4));
                let scale = value.iter().map(modulus).fold(1., f64::max);
                worst = worst.max(distance(&constant, &expected) / scale);
            }
        }
        // Integrated value with the exact tadpole.
        let m = complex(p, masses[lane]);
        let mut a0 = vec![[
            complex(p, C64::ZERO),
            complex(p, C64::ZERO),
            complex(p, C64::ZERO),
        ]];
        let mut status = [ninjars::MasterStatus::Success];
        ninjars::MasterProvider::evaluate_batch(
            &mut masters,
            ninjars::MasterRequest {
                integral: ninjars::MasterIntegral::A0,
                invariants: &[],
                masses_squared: std::slice::from_ref(&m),
                scales_squared: std::slice::from_ref(&scales[lane]),
            },
            &mut a0,
            &mut status,
        )
        .unwrap();
        let half = real(0.5);
        let quarter = real(0.25);
        let (cz, az, bz) = (complex(p, c), complex(p, a), complex(p, b));
        let p2 = real(p2);
        let expected: Vec<_> = (0..3)
            .map(|k| {
                let mut value = az.clone() * &a0[0][k];
                let mut tensor = a0[0][k].clone();
                if k == 0 {
                    value = value + bz.clone() * &m * &m * &half;
                    tensor = tensor + m.clone() * &half;
                }
                value + cz.clone() * &p2 * &m * &tensor * &quarter
            })
            .collect();
        let results = &amplitude.results()[lane];
        worst = worst.max(laurent_error(results, &expected));
    }
    worst
}

#[test]
fn rank_n2_single_propagator() {
    let float = single(Float::new(256));
    let binary64 = single(0f64);
    eprintln!("RANK_N2_TADPOLE single propagator float256={float:.3e} binary64={binary64:.3e}");
    assert!(float <= 1e-60, "Float(256): {float:e}");
    assert!(binary64 <= 1e-12, "binary64: {binary64:e}");
}

/// Massless lines: scaleless tadpoles and bubbles integrate to zero, the rank
/// n+2 rational terms included (n = 2, both lines massless and lightlike
/// external momentum, so the whole integral vanishes).
#[test]
fn rank_n2_scaleless_integrals_vanish() {
    let p = Float::new(256);
    let problem = rank_n2_support::Problem::new(2, 4, 0x5ca1e, true, 1);
    let mut numerator = problem.numerator(&p);
    let real = |x: f64| complex(&p, C64::real(x));
    let lanes = problem.count();
    let shifts: Vec<_> = (0..lanes)
        .flat_map(|_| {
            [[0., 0., 0., 0.], [1.5, 0.5, 1.0, 1.0]].map(|v| FourMomentum::from(v.map(real)))
        })
        .collect();
    let masses: Vec<_> = (0..2 * lanes).map(|_| real(0.)).collect();
    let scales: Vec<_> = (0..lanes).map(|_| real(1.)).collect();
    let mut amplitude = SymbolicaAmplitudeKernels::new(2, 4, p.clone())
        .unwrap()
        .into_amplitude(lanes)
        .unwrap();
    let mut low = SymbolicaCuts::new(p.clone()).unwrap();
    let mut high = SymbolicaHighCuts::new(p.clone()).unwrap();
    let mut masters = SymbolicaMasters::new(p.clone());
    amplitude
        .evaluate_batch_with_inputs(
            AmplitudeInput {
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: &scales,
                sampling_scales: &scales,
                s_matrix: None,
            },
            &mut numerator,
            &mut low,
            &mut high,
            &mut masters,
        )
        .unwrap();
    for (lane, result) in amplitude.results().iter().enumerate() {
        assert_eq!(amplitude.statuses()[lane], EvaluationStatus::Success);
        for value in result {
            assert!(modulus(value) <= 1e-60, "lane {lane}: {value:?}");
        }
    }
}
