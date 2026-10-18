//! Singular and unstable points in every precision.
//!
//! - An exactly singular point (a box whose mu-expansion leading denominator
//!   `2 v_orth.(V_j - V_0)` vanishes exactly) is computed as exactly zero or as
//!   rounding noise depending on the carrier; every carrier must reject it at
//!   the same cut, not report a meaningless result.
//! - The opt-in cancellation test reports [`EvaluationStatus::Unstable`] for a
//!   binary64 lane whose cut contributions cancel by fourteen orders of
//!   magnitude, and accepts the same lane in DoubleFloat.

use ninjars::fast::{
    C64, FastAmplitude, FastInput, FastOptions, FastResult, FastStatus, GenericNumerator,
};
use ninjars::{
    AmplitudeInput, Complex, CutFailure, DoubleFloat, EvaluationStatus, Float, FloatLike,
    FourMomentum, Laurent, MasterProvider, MasterRequest, MasterStatus, Real, SingleFloat,
};
use ninjars_symbolica::{
    DualPolynomialNumerator, OneLoopNativeMasters, PolynomialNumerator, SymbolicaAmplitudeKernels,
    SymbolicaCuts, SymbolicaDualAmplitudeKernels, SymbolicaDualCuts, SymbolicaDualHighCuts,
    SymbolicaHighCuts,
};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{dual::HyperDual, float::RealLike, rational::Rational},
    evaluate::EvaluationDomain,
    parser::ParseSettings,
};

const NAMESPACE: &str = "ninjars_stability";

fn parse(text: &str) -> Atom {
    Atom::parse(text, NAMESPACE, ParseSettings::default()).unwrap()
}

fn variables() -> [Atom; 5] {
    ["q0", "q1", "q2", "q3", "mu2"].map(parse)
}

fn rational(text: &str) -> Rational {
    match text.split_once('/') {
        Some((a, b)) => {
            Rational::from(a.parse::<i64>().unwrap()) / Rational::from(b.parse::<i64>().unwrap())
        }
        None => Rational::from(text.parse::<i64>().unwrap()),
    }
}

/// Every master is one: rejected lanes never reach the masters.
struct UnitMasters;
impl<V: FloatLike> MasterProvider<V> for UnitMasters {
    fn evaluate(
        &mut self,
        request: MasterRequest<'_, V>,
        output: &mut [Laurent<V>],
    ) -> ninjars::Result<()> {
        let mut statuses = vec![MasterStatus::Success; request.rows()];
        self.evaluate_batch(request, output, &mut statuses)
    }
    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, V>,
        output: &mut [Laurent<V>],
        statuses: &mut [MasterStatus],
    ) -> ninjars::Result<()> {
        let one = request.scales_squared[0].one();
        for (out, status) in output.iter_mut().zip(statuses) {
            *out = [one.clone(), one.zero(), one.zero()];
            *status = MasterStatus::Success;
        }
        Ok(())
    }
}

/// Five propagators with `V_4 = V_1/3 - 2 V_2/7 + 3 V_3/5` exactly (`V_0 = 0`):
/// every box's normal is orthogonal to the fifth shift, so the box
/// mu-expansion leading denominators vanish. No component is a binary
/// fraction, so each carrier rounds the inputs differently.
fn singular_kinematics() -> (Vec<[Rational; 4]>, Vec<Rational>) {
    let v = |c: [&str; 4]| c.map(rational);
    let (v1, v2, v3) = (
        v(["1/3", "1/7", "2/5", "-1/11"]),
        v(["-2/9", "5/13", "1/6", "3/7"]),
        v(["4/15", "-1/3", "2/21", "5/9"]),
    );
    let (a, b, c) = (rational("1/3"), rational("-2/7"), rational("3/5"));
    let v4 = std::array::from_fn(|i| &(&(&a * &v1[i]) + &(&b * &v2[i])) + &(&c * &v3[i]));
    let zero = std::array::from_fn(|_| Rational::from(0));
    let masses = ["1/4", "1/3", "2/7", "3/10", "1/5"].map(rational).to_vec();
    (vec![zero, v1, v2, v3, v4], masses)
}

const SINGULAR_NUMERATOR: &str = "q0^5 + 2*q1^3*mu2 + q2*q3^2 + 3*q0*mu2^2 + 1";

fn singular_status<T>(prototype: T) -> EvaluationStatus
where
    T: Real + SingleFloat,
    Complex<T>: Real + EvaluationDomain,
{
    let p = &prototype;
    let (shifts, masses) = singular_kinematics();
    let value = |x: &Rational| Complex::new(p.from_rational(x), p.zero());
    let shifts: Vec<_> = shifts
        .iter()
        .map(|v| FourMomentum::from_args(value(&v[0]), value(&v[1]), value(&v[2]), value(&v[3])))
        .collect();
    let masses: Vec<_> = masses.iter().map(value).collect();
    let one = Complex::new(p.one(), p.zero());
    let mut numerator =
        PolynomialNumerator::new(parse(SINGULAR_NUMERATOR), variables(), vec![], p.clone())
            .unwrap();
    numerator.set_parameter_batch(&[], 1).unwrap();
    let mut amplitude = SymbolicaAmplitudeKernels::new(5, 5, p.clone())
        .unwrap()
        .into_amplitude(1)
        .unwrap();
    amplitude
        .evaluate_batch_with_inputs(
            AmplitudeInput {
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: std::slice::from_ref(&one),
                sampling_scales: std::slice::from_ref(&one),
                s_matrix: None,
            },
            &mut numerator,
            &mut SymbolicaCuts::new(p.clone()).unwrap(),
            &mut SymbolicaHighCuts::new(p.clone()).unwrap(),
            &mut UnitMasters,
        )
        .unwrap();
    amplitude.statuses()[0]
}

fn singular_dual_status() -> EvaluationStatus {
    let p = 0_f64;
    let zero = Complex::new(0., 0.);
    let dual = HyperDual::from_values(vec![vec![0], vec![1]], vec![zero; 2]);
    let constant = |x: &Rational| {
        let mut v = dual.clone();
        v.values[0] = Complex::new(p.from_rational(x), 0.);
        v
    };
    let (shifts, masses) = singular_kinematics();
    let shifts: Vec<_> = shifts
        .iter()
        .map(|v| {
            FourMomentum::from_args(
                constant(&v[0]),
                constant(&v[1]),
                constant(&v[2]),
                constant(&v[3]),
            )
        })
        .collect();
    // The derivative direction is a squared mass, which leaves the singular
    // geometry untouched.
    let mut masses: Vec<_> = masses.iter().map(constant).collect();
    masses[1].values[1] = Complex::new(1., 0.);
    let one = constant(&Rational::from(1));
    let source =
        PolynomialNumerator::new(parse(SINGULAR_NUMERATOR), variables(), vec![], p).unwrap();
    let mut numerator = DualPolynomialNumerator::new(source, dual.clone()).unwrap();
    numerator.set_parameter_batch(&[], 1).unwrap();
    let mut amplitude = SymbolicaDualAmplitudeKernels::new(5, 5, p, &dual)
        .unwrap()
        .into_amplitude(1)
        .unwrap();
    amplitude
        .evaluate_batch_with_inputs(
            AmplitudeInput {
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: std::slice::from_ref(&one),
                sampling_scales: std::slice::from_ref(&one),
                s_matrix: None,
            },
            &mut numerator,
            &mut SymbolicaDualCuts::new(p, dual.clone()).unwrap(),
            &mut SymbolicaDualHighCuts::new(p, dual.clone()).unwrap(),
            &mut UnitMasters,
        )
        .unwrap();
    amplitude.statuses()[0]
}

#[test]
fn exactly_singular_box_fails_at_the_same_cut_in_every_precision() {
    let statuses = [
        ("f64", singular_status(0_f64)),
        ("DoubleFloat", singular_status(DoubleFloat::from(0.))),
        ("Float(128)", singular_status(Float::new(128))),
        ("Float(256)", singular_status(Float::new(256))),
        ("Float(400)", singular_status(Float::new(400))),
        ("Dual<f64>", singular_dual_status()),
    ];
    // Cut 0 is the pentagon (not evaluated with the box mu-expansion); the
    // first box is the first cut whose leading denominator vanishes.
    for (carrier, status) in statuses {
        assert_eq!(
            status,
            EvaluationStatus::CutFailed {
                cut_index: 1,
                failure: CutFailure::SingularParameter,
            },
            "{carrier}"
        );
    }
}

/// Lane 1 of the any-n verification problem `n = 7, rank 8, light-like
/// external legs, real masses, seed 1`: dyadic inputs, exact in every carrier.
mod cancelling {
    pub const SHIFTS: [[f64; 4]; 7] = [
        [0.0, 0.0, 0.0, 0.0],
        [0.40625, -0.09375, -0.125, -0.375],
        [1.78125, 0.15625, -0.875, -1.5],
        [1.5, 0.28125, -0.84375, -1.25],
        [1.21875, 0.5, -0.96875, -1.125],
        [0.8125, 0.40625, -0.59375, -1.25],
        [2.8125, 0.40625, 1.40625, -1.25],
    ];
    pub const MASSES: [f64; 7] = [
        1.1142578125,
        1.0361328125,
        0.41845703125,
        0.606201171875,
        0.329345703125,
        0.55810546875,
        0.450927734375,
    ];
    pub const SCALE: f64 = 2.5;
    pub const SAMPLING: f64 = 4.265625;
    /// `(q0, q1, q2, q3) exponents, mu² exponent, coefficient`.
    pub const TERMS: [([u32; 4], u32, (f64, f64)); 19] = [
        ([0, 0, 0, 0], 0, (0.571044921875, 0.311279296875)),
        ([0, 0, 0, 0], 1, (-0.00927734375, -0.565673828125)),
        ([0, 0, 0, 0], 2, (0.658935546875, 0.409423828125)),
        ([0, 0, 0, 0], 4, (-0.85888671875, 0.678955078125)),
        ([0, 0, 0, 8], 0, (-0.15380859375, -0.600830078125)),
        ([0, 0, 1, 0], 0, (-0.98046875, -0.8046875)),
        ([0, 0, 1, 0], 3, (0.2705078125, 0.1005859375)),
        ([0, 1, 0, 0], 1, (0.685302734375, 0.62939453125)),
        ([0, 1, 3, 1], 0, (0.709716796875, 0.827392578125)),
        ([0, 3, 1, 2], 1, (-0.935302734375, -0.4140625)),
        ([1, 0, 1, 0], 0, (-0.384033203125, -0.145263671875)),
        ([1, 0, 1, 2], 0, (-0.51025390625, -0.587890625)),
        ([1, 1, 2, 2], 0, (-0.066162109375, -0.337158203125)),
        ([2, 0, 0, 1], 1, (-0.29638671875, -0.71630859375)),
        ([2, 1, 1, 0], 1, (0.826904296875, -0.516845703125)),
        ([2, 3, 2, 1], 0, (-0.642333984375, -0.9619140625)),
        ([3, 0, 0, 0], 0, (-0.17578125, 0.570068359375)),
        ([3, 0, 1, 3], 0, (0.959228515625, -0.241455078125)),
        ([8, 0, 0, 0], 0, (-0.388671875, 0.3544921875)),
    ];
}

fn cancelling_numerator() -> Atom {
    let q = variables();
    let exact = |x: f64| Atom::num(Rational::try_from(x).unwrap());
    let mut numerator = Atom::Zero;
    for (powers, mu, (re, im)) in cancelling::TERMS {
        let mut term = exact(re) + exact(im) * Atom::i();
        for (variable, power) in q.iter().zip(powers.into_iter().chain([mu])) {
            term *= variable.pow(i64::from(power));
        }
        numerator += term;
    }
    numerator
}

/// Binary64 native masters widened exactly into any carrier, so that every
/// carrier reduces with identical master values.
struct Widened(OneLoopNativeMasters);
impl<T: SingleFloat + RealLike> MasterProvider<Complex<T>> for Widened {
    fn evaluate(
        &mut self,
        request: MasterRequest<'_, Complex<T>>,
        output: &mut [Laurent<Complex<T>>],
    ) -> ninjars::Result<()> {
        let mut statuses = vec![MasterStatus::Success; request.rows()];
        self.evaluate_batch(request, output, &mut statuses)
    }
    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, Complex<T>>,
        output: &mut [Laurent<Complex<T>>],
        statuses: &mut [MasterStatus],
    ) -> ninjars::Result<()> {
        let round = |values: &[Complex<T>]| -> Vec<Complex<f64>> {
            values
                .iter()
                .map(|z| Complex::new(z.re.to_f64(), z.im.to_f64()))
                .collect()
        };
        let prototype = request.scales_squared[0].re.clone();
        let (invariants, masses, scales) = (
            round(request.invariants),
            round(request.masses_squared),
            round(request.scales_squared),
        );
        let mut values = vec![[Complex::new(0., 0.); 3]; request.rows()];
        self.0.evaluate_batch(
            MasterRequest {
                integral: request.integral,
                invariants: &invariants,
                masses_squared: &masses,
                scales_squared: &scales,
            },
            &mut values,
            statuses,
        )?;
        let exact = |x: f64| prototype.from_rational(&Rational::try_from(x).unwrap());
        for (out, value) in output.iter_mut().zip(&values) {
            *out = value.map(|z| Complex::new(exact(z.re), exact(z.im)));
        }
        Ok(())
    }
}

/// The cancelling lane evaluated `evaluations` times into one amplitude:
/// its status, finite part, and the binary64 cancellation measurement
/// (summed contribution moduli per Laurent coefficient and result moduli).
fn cancelling_evaluations<T>(
    prototype: T,
    stability_test: bool,
    evaluations: usize,
) -> (EvaluationStatus, Complex<T>, [f64; 3], [f64; 3])
where
    T: Real + SingleFloat + RealLike,
    Complex<T>: Real + EvaluationDomain,
{
    let p = &prototype;
    let value = |x: f64| Complex::new(p.from_rational(&Rational::try_from(x).unwrap()), p.zero());
    let shifts: Vec<_> = cancelling::SHIFTS
        .iter()
        .map(|v| FourMomentum::from_args(value(v[0]), value(v[1]), value(v[2]), value(v[3])))
        .collect();
    let masses: Vec<_> = cancelling::MASSES.into_iter().map(value).collect();
    let mut numerator =
        PolynomialNumerator::new(cancelling_numerator(), variables(), vec![], p.clone()).unwrap();
    numerator.set_parameter_batch(&[], 1).unwrap();
    let kernels = SymbolicaAmplitudeKernels::new(7, 8, p.clone()).unwrap();
    let kernels = if stability_test {
        kernels.with_stability_test()
    } else {
        kernels
    };
    let mut amplitude = kernels.into_amplitude(1).unwrap();
    for _ in 0..evaluations {
        amplitude
            .evaluate_batch_with_inputs(
                AmplitudeInput {
                    shifts: &shifts,
                    masses_squared: &masses,
                    scales_squared: &[value(cancelling::SCALE)],
                    sampling_scales: &[value(cancelling::SAMPLING)],
                    s_matrix: None,
                },
                &mut numerator,
                &mut SymbolicaCuts::new(p.clone()).unwrap(),
                &mut SymbolicaHighCuts::new(p.clone()).unwrap(),
                &mut Widened(OneLoopNativeMasters::new().unwrap()),
            )
            .unwrap();
    }
    let binary64 = |moduli: &Laurent<Complex<T>>| moduli.each_ref().map(|m| m.re.to_f64());
    let outcome = (
        amplitude.statuses()[0],
        amplitude.results()[0][0].clone(),
        binary64(&amplitude.contribution_moduli()[0]),
        binary64(&amplitude.result_moduli(0).unwrap()),
    );
    amplitude.reset();
    assert!(
        amplitude.contribution_moduli()[0]
            .iter()
            .all(|m| m.re.is_zero()),
        "reset clears the measurement"
    );
    outcome
}

fn cancelling_lane<T>(prototype: T, stability_test: bool) -> (EvaluationStatus, Complex<T>)
where
    T: Real + SingleFloat + RealLike,
    Complex<T>: Real + EvaluationDomain,
{
    let (status, value, _, _) = cancelling_evaluations(prototype, stability_test, 1);
    (status, value)
}

#[test]
fn opt_in_cancellation_test_flags_lost_binary64_digits() {
    let (status, f64_value) = cancelling_lane(0_f64, false);
    assert_eq!(status, EvaluationStatus::Success, "the test is opt-in");
    let (status, _) = cancelling_lane(0_f64, true);
    assert_eq!(status, EvaluationStatus::Unstable);
    // DoubleFloat reduces the same inputs with the same master values and keeps
    // more than half of its bits: it passes, and shows that binary64 lost all.
    let (status, reference) = cancelling_lane(DoubleFloat::from(0.), true);
    assert_eq!(status, EvaluationStatus::Success);
    let reference = Complex::new(reference.re.to_f64(), reference.im.to_f64());
    let error = (f64_value.re - reference.re).hypot(f64_value.im - reference.im)
        / reference.re.hypot(reference.im);
    assert!(error > 1., "binary64 relative error {error:e}");
}

/// The cancellation measurement is reported without the opt-in test. It
/// agrees with the test's verdict (a ratio above `2^(p/2)` in binary64, below
/// it in DoubleFloat), and it accumulates and resets with the results.
#[test]
fn cancellation_is_measured_without_the_stability_test() {
    let largest = |moduli: [f64; 3]| moduli.into_iter().fold(0., f64::max);
    let (status, _, contributions, result) = cancelling_evaluations(0_f64, false, 1);
    assert_eq!(status, EvaluationStatus::Success);
    let ratio = largest(contributions) / largest(result);
    assert!(ratio > 2_f64.powf(26.5), "binary64 ratio {ratio:e}");
    let (status, _, double_contributions, double_result) =
        cancelling_evaluations(DoubleFloat::from(0.), false, 1);
    assert_eq!(status, EvaluationStatus::Success);
    let ratio = largest(double_contributions) / largest(double_result);
    assert!(ratio < 2_f64.powi(53), "DoubleFloat ratio {ratio:e}");
    for (binary64, double) in contributions.iter().zip(&double_contributions) {
        assert!(
            (binary64 - double).abs() <= 1e-10 * double,
            "{binary64:e} {double:e}"
        );
    }
    let (_, _, twice, _) = cancelling_evaluations(0_f64, false, 2);
    for (twice, once) in twice.iter().zip(&contributions) {
        assert!(
            (twice - 2. * once).abs() <= 1e-14 * twice,
            "{twice:e} {once:e}"
        );
    }
}

/// Fast and generic binary64 engines on a constant numerator with native
/// masters: `(fast status, fast finite part, generic status, generic finite part)`.
fn binary64_engines(
    shifts: &[[f64; 4]],
    masses: &[f64],
    scale: f64,
    constant: (f64, f64),
) -> (FastStatus, Complex<f64>, EvaluationStatus, Complex<f64>) {
    let n = shifts.len();
    let numerator = || {
        let value = Atom::num(Rational::try_from(constant.0).unwrap())
            + Atom::num(Rational::try_from(constant.1).unwrap()) * Atom::i();
        let mut numerator = PolynomialNumerator::new(value, variables(), vec![], 0_f64).unwrap();
        numerator.set_parameter_batch(&[], 1).unwrap();
        numerator
    };
    let masses_c: Vec<C64> = masses.iter().map(|&m| C64::real(m)).collect();
    let mut engine = FastAmplitude::new(n, 0, FastOptions::default()).unwrap();
    let mut results = vec![FastResult::default()];
    let mut statuses = vec![FastStatus::NotEvaluated];
    engine
        .evaluate(
            &FastInput {
                lanes: 1,
                shifts,
                masses_squared: &masses_c,
                scales_squared: &[scale],
                s_matrix: None,
            },
            &mut GenericNumerator::with_rank(numerator(), 0),
            &mut OneLoopNativeMasters::new().unwrap(),
            &mut results,
            &mut statuses,
        )
        .unwrap();
    let c = |x: f64| Complex::new(x, 0.);
    let generic_shifts: Vec<_> = shifts
        .iter()
        .map(|v| FourMomentum::from_args(c(v[0]), c(v[1]), c(v[2]), c(v[3])))
        .collect();
    let generic_masses: Vec<_> = masses.iter().map(|&m| c(m)).collect();
    let mut amplitude = SymbolicaAmplitudeKernels::new(n, 0, 0_f64)
        .unwrap()
        .into_amplitude(1)
        .unwrap();
    amplitude
        .evaluate_batch_with_inputs(
            AmplitudeInput {
                shifts: &generic_shifts,
                masses_squared: &generic_masses,
                scales_squared: &[c(scale)],
                sampling_scales: &[c(1.)],
                s_matrix: None,
            },
            &mut numerator(),
            &mut SymbolicaCuts::new(0_f64).unwrap(),
            &mut SymbolicaHighCuts::new(0_f64).unwrap(),
            &mut OneLoopNativeMasters::new().unwrap(),
        )
        .unwrap();
    let fast = results[0].laurent[0];
    (
        statuses[0],
        Complex::new(fast.re, fast.im),
        amplitude.statuses()[0],
        amplitude.results()[0][0],
    )
}

/// `n8r0genericm1` lane 0 of the any-n verification: box (4,5,6,7) has
/// orthogonal massive momenta `k3.k0 = 0`, where Ninja's basis normalization
/// cancels. The fast engine was off by 3.4e-3; the box coefficient checked
/// against an independent quadruple cut is the generic engine's.
#[test]
fn fast_engine_matches_generic_engine_with_orthogonal_cut_momenta() {
    let r = |text: &str| 0_f64.from_rational(&rational(text));
    let shifts: Vec<[f64; 4]> = [
        ["0", "0", "0", "0"],
        ["-3", "-3/4", "0", "1"],
        ["-2", "-1/4", "1/3", "5/3"],
        ["-4/3", "-1/4", "-1/6", "23/12"],
        ["2/3", "-1/4", "1/6", "8/3"],
        ["8/3", "-1/4", "1/6", "5/3"],
        ["17/3", "3/4", "-5/6", "17/12"],
        ["2", "0", "1", "0"],
    ]
    .iter()
    .map(|v| v.map(r))
    .collect();
    let (fast_status, fast, generic_status, generic) =
        binary64_engines(&shifts, &[1.; 8], 1., (1., 0.));
    assert_eq!(fast_status, FastStatus::Success);
    assert_eq!(generic_status, EvaluationStatus::Success);
    let error = (fast.re - generic.re).hypot(fast.im - generic.im) / generic.re.hypot(generic.im);
    assert!(error < 1e-12, "fast vs generic relative error {error:e}");
}

/// Light-like `n = 7` lane whose box (1,2,4,5) is exactly degenerate (its
/// normal is light-like). Ninja's absolute check on `mp2(Vort)` accepted the
/// rounding noise and the fast engine returned a meaningless result; both
/// engines now reject the lane.
#[test]
fn both_engines_reject_an_exactly_degenerate_box() {
    let shifts = [
        [0.0, 0.0, 0.0, 0.0],
        [-0.5, 0.0, 0.0, 0.5],
        [1.5, 2.0, 0.0, 0.5],
        [2.75, 2.75, 0.0, -0.5],
        [3.625, 3.125, -0.75, -0.25],
        [4.625, 4.125, -0.75, -0.25],
        [5.03125, 3.96875, -0.375, -0.25],
    ];
    let masses = [
        0.0,
        0.890380859375,
        0.0,
        1.296142578125,
        0.0,
        0.666259765625,
        0.0,
    ];
    let (fast_status, _, generic_status, _) =
        binary64_engines(&shifts, &masses, 2.5, (0.55517578125, -0.8828125));
    assert_eq!(fast_status, FastStatus::Unstable);
    assert_eq!(
        generic_status,
        EvaluationStatus::CutFailed {
            cut_index: 44,
            failure: CutFailure::DegenerateGram,
        }
    );
}
