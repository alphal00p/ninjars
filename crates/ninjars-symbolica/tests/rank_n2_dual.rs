//! V10: dual-number derivatives of rank n+2 amplitudes in the generic engine.
//!
//! HyperDual numbers with shape `[[0,0],[1,0],[0,1],[2,0],[1,1]]` carry the
//! value, d/dx, d/dy, (d/dx)²/2 and d²/dxdy, with x shifting one squared mass
//! and y one momentum component. Dual evaluations over Float(256) are compared
//! with central finite differences of Float(512) evaluations (step 2^-80:
//! truncation O(2^-160), rounding O(2^-352)) to <= 1e-30 relative to each
//! lane's largest Laurent coefficient:
//! - n = 2..4 at rank n+2 with smooth polynomial stand-in masters (dual C0
//!   and D0 are outside the masters' derivative domain, as for rank n+1);
//! - n = 2 at rank 4 with the exact masters (`SymbolicaDualMasters`, B1111
//!   included) at Euclidean kinematics, where their derivatives are supported.

mod rank_n2_support;

use ninjars::fast::{C64, RMom};
use ninjars::{
    AmplitudeInput, Complex, EvaluationStatus, Float, FloatLike, FourMomentum, MasterProvider,
    Real, SingleFloat,
};
use ninjars_symbolica::{
    DualPolynomialNumerator, PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts,
    SymbolicaDualAmplitudeKernels, SymbolicaDualCuts, SymbolicaDualHighCuts, SymbolicaDualMasters,
    SymbolicaHighCuts, SymbolicaMasters,
};
use rank_n2_support::{Lane, PolynomialMasters, Problem, complex, distance, exact, to_c64};
use symbolica::domains::dual::HyperDual;

const MASS_LINE: usize = 1;
const SHIFT_LINE: usize = 1;

/// The problem's lanes with mass `m[MASS_LINE] + x` and
/// `V[SHIFT_LINE].px + y`, lane-major over the given (x, y) offsets.
fn shifted_inputs<T: SingleFloat>(
    lanes: &[Lane],
    offsets: &[(T, T)],
    prototype: &T,
) -> (Vec<FourMomentum<Complex<T>>>, Vec<Complex<T>>)
where
    Complex<T>: Real,
{
    let n = lanes[0].shifts.len();
    let mut shifts = Vec::new();
    let mut masses = Vec::new();
    for lane in lanes {
        for (x, y) in offsets {
            for i in 0..n {
                let v = lane.shifts[i];
                let mut momentum = FourMomentum::from(v.map(|c| complex(prototype, C64::real(c))));
                let mut mass = complex(prototype, lane.masses[i]);
                if i == SHIFT_LINE {
                    momentum.spatial.px.re += y;
                }
                if i == MASS_LINE {
                    mass.re += x;
                }
                shifts.push(momentum);
                masses.push(mass);
            }
        }
    }
    (shifts, masses)
}

/// Float evaluations of every lane at every offset (lane-major).
fn float_at<M: MasterProvider<Complex<Float>>>(
    problem: &Problem,
    lanes: &[Lane],
    offsets: &[(Float, Float)],
    bits: u32,
    masters: &mut M,
) -> Vec<[Complex<Float>; 5]> {
    let p = Float::new(bits);
    let count = lanes.len() * offsets.len();
    let mut numerator = PolynomialNumerator::new(
        problem.expression.clone(),
        rank_n2_support::variables(),
        problem.parameters.clone(),
        p.clone(),
    )
    .unwrap();
    let width = problem.parameters.len();
    let values: Vec<_> = (0..lanes.len())
        .flat_map(|l| {
            let row: Vec<_> = problem.coefficients[l * width..(l + 1) * width]
                .iter()
                .map(|&z| complex(&p, z))
                .collect();
            std::iter::repeat_n(row, offsets.len()).flatten()
        })
        .collect();
    numerator.set_parameter_batch(&values, count).unwrap();
    let (shifts, masses) = shifted_inputs(lanes, offsets, &p);
    let real = |x: f64| complex(&p, C64::real(x));
    let scales: Vec<_> = (0..count)
        .map(|i| real([1.0, 2.5, 0.7][i / offsets.len() % 3]))
        .collect();
    let sampling: Vec<_> = (0..count).map(|_| real(1.)).collect();
    let mut amplitude = SymbolicaAmplitudeKernels::new(problem.n, problem.rank, p.clone())
        .unwrap()
        .into_amplitude(count)
        .unwrap();
    let mut low = SymbolicaCuts::new(p.clone()).unwrap();
    let mut high = SymbolicaHighCuts::new(p.clone()).unwrap();
    amplitude
        .evaluate_batch_with_inputs(
            AmplitudeInput {
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: &scales,
                sampling_scales: &sampling,
                s_matrix: None,
            },
            &mut numerator,
            &mut low,
            &mut high,
            masters,
        )
        .unwrap();
    for (lane, status) in amplitude.statuses().iter().enumerate() {
        assert_eq!(*status, EvaluationStatus::Success, "lane {lane}");
    }
    amplitude.results().to_vec()
}

type D = HyperDual<Complex<Float>>;

/// Dual evaluation at x = y = 0 with unit seeds.
fn dual_at<M: MasterProvider<D>>(
    problem: &Problem,
    lanes: &[Lane],
    masters: &mut M,
    dual: &D,
) -> Vec<[D; 5]> {
    let p = Float::new(256);
    let count = lanes.len();
    let constant = |z: Complex<Float>| {
        let mut v = dual.clone();
        v.values[0] = z;
        v
    };
    let seeded = |z: Complex<Float>, slot: usize| {
        let mut v = constant(z);
        v.values[slot] = complex(&p, C64::real(1.));
        v
    };
    let source = PolynomialNumerator::new(
        problem.expression.clone(),
        rank_n2_support::variables(),
        problem.parameters.clone(),
        p.clone(),
    )
    .unwrap();
    let mut numerator = DualPolynomialNumerator::new(source, dual.clone()).unwrap();
    let width = problem.parameters.len();
    let values: Vec<_> = problem.coefficients[..count * width]
        .iter()
        .map(|&z| constant(complex(&p, z)))
        .collect();
    numerator.set_parameter_batch(&values, count).unwrap();
    let real = |x: f64| constant(complex(&p, C64::real(x)));
    let n = problem.n;
    let mut shifts: Vec<FourMomentum<D>> = Vec::new();
    let mut masses = Vec::new();
    for lane in lanes {
        for i in 0..n {
            let v = lane.shifts[i];
            let px = if i == SHIFT_LINE {
                seeded(complex(&p, C64::real(v[1])), 2)
            } else {
                real(v[1])
            };
            shifts.push(FourMomentum::from_args(
                real(v[0]),
                px,
                real(v[2]),
                real(v[3]),
            ));
            masses.push(if i == MASS_LINE {
                seeded(complex(&p, lane.masses[i]), 1)
            } else {
                constant(complex(&p, lane.masses[i]))
            });
        }
    }
    let scales: Vec<_> = (0..count).map(|i| real([1.0, 2.5, 0.7][i % 3])).collect();
    let sampling: Vec<_> = (0..count).map(|_| real(1.)).collect();
    let mut amplitude =
        SymbolicaDualAmplitudeKernels::new(problem.n, problem.rank, p.clone(), dual)
            .unwrap()
            .into_amplitude(count)
            .unwrap();
    let mut low = SymbolicaDualCuts::new(p.clone(), dual.clone()).unwrap();
    let mut high = SymbolicaDualHighCuts::new(p.clone(), dual.clone()).unwrap();
    amplitude
        .evaluate_batch_with_inputs(
            AmplitudeInput {
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: &scales,
                sampling_scales: &sampling,
                s_matrix: None,
            },
            &mut numerator,
            &mut low,
            &mut high,
            masters,
        )
        .unwrap();
    for (lane, status) in amplitude.statuses().iter().enumerate() {
        assert_eq!(*status, EvaluationStatus::Success, "lane {lane}");
    }
    amplitude.results().to_vec()
}

fn hyperdual() -> D {
    let zero = complex(&Float::new(256), C64::ZERO);
    HyperDual::from_values(
        vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]],
        vec![zero; 5],
    )
}

/// Worst relative difference of every Dual component from finite differences.
fn check<MF, MD>(
    problem: &Problem,
    lanes: &[Lane],
    float_masters: &mut MF,
    dual_masters: &mut MD,
) -> f64
where
    MF: MasterProvider<Complex<Float>>,
    MD: MasterProvider<D>,
{
    let bits = 512;
    let zero = Float::new(bits);
    let h = exact(&zero, 2_f64.powi(-80));
    let o = zero.clone();
    let offsets = vec![
        (o.clone(), o.clone()),
        (h.clone(), o.clone()),
        (-h.clone(), o.clone()),
        (o.clone(), h.clone()),
        (o.clone(), -h.clone()),
        (h.clone(), h.clone()),
        (h.clone(), -h.clone()),
        (-h.clone(), h.clone()),
        (-h.clone(), -h.clone()),
    ];
    let values = float_at(problem, lanes, &offsets, bits, float_masters);
    let dual = dual_at(problem, lanes, dual_masters, &hyperdual());
    let real = |x: Float| Complex::new(x, zero.clone());
    let two = real(exact(&zero, 2.));
    let four = real(exact(&zero, 4.));
    let first = real((h.clone() * &two.re).inv());
    let second = real((h.clone() * &h).inv());
    let mut worst = 0_f64;
    for lane in 0..lanes.len() {
        let f = |k: usize, c: usize| values[lane * offsets.len() + k][c].clone();
        for c in 0..3 {
            let center = f(0, c);
            let dx = (f(1, c) - &f(2, c)) * &first;
            let dy = (f(3, c) - &f(4, c)) * &first;
            // Taylor coefficient (d/dx)²/2.
            let dxx = (f(1, c) - &(center.clone() * &two) + &f(2, c)) * &second / &two;
            let dxy = (f(5, c) - &f(6, c) - &f(7, c) + &f(8, c)) * &second / &four;
            let expected = [center, dx, dy, dxx, dxy];
            let scale = expected
                .iter()
                .map(|z| to_c64(z).abs())
                .fold(f64::MIN_POSITIVE, f64::max);
            for (slot, e) in expected.iter().enumerate() {
                let error = distance(&dual[lane][c].values[slot], e) / scale;
                worst = worst.max(error);
            }
            // The derivative variables move the finite part.
            if c == 0 {
                assert!(to_c64(&expected[1]).abs() > 1e-8 * to_c64(&expected[0]).abs());
                assert!(to_c64(&expected[2]).abs() > 1e-8 * to_c64(&expected[0]).abs());
            }
        }
    }
    worst
}

#[test]
fn rank_n2_dual_derivatives_with_stand_in_masters() {
    for n in 2..=4 {
        let problem = Problem::new(n, n + 2, 0xd0a1 + n as u64, true, if n < 4 { 1 } else { 3 });
        let lanes = &problem.lanes[..2];
        let started = std::time::Instant::now();
        let worst = check(
            &problem,
            lanes,
            &mut PolynomialMasters,
            &mut PolynomialMasters,
        );
        eprintln!(
            "RANK_N2_DUAL n={n} rank={} stand-in masters: {worst:.3e} ({:.1}s)",
            n + 2,
            started.elapsed().as_secs_f64()
        );
        assert!(worst <= 1e-30, "n={n}: {worst:e}");
    }
}

#[test]
fn rank_n2_dual_derivatives_with_exact_masters() {
    // Euclidean bubble: spacelike external momentum, real positive masses.
    let mut problem = Problem::new(2, 4, 0xd0a2, true, 1);
    let shifts: Vec<RMom> = vec![[0., 0., 0., 0.], [0.5, 1.25, -0.75, 0.5]];
    for (lane, masses) in problem
        .lanes
        .iter_mut()
        .zip([[0.75, 1.5], [1.25, 0.5], [0.5, 0.875]])
    {
        lane.shifts = shifts.clone();
        lane.masses = masses.iter().map(|&m| C64::real(m)).collect();
    }
    let lanes = problem.lanes.clone();
    let mut float_masters = SymbolicaMasters::new(Float::new(512));
    let mut dual_masters = SymbolicaDualMasters::new(Float::new(256), &hyperdual());
    let worst = check(&problem, &lanes, &mut float_masters, &mut dual_masters);
    eprintln!("RANK_N2_DUAL n=2 rank=4 exact masters (B1111 included): {worst:.3e}");
    assert!(worst <= 1e-30, "{worst:e}");
}
