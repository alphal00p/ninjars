//! Independent boundary and numerical-domain controls for complete amplitudes.

use ninjars::{
    Complex, DoubleFloat, Error, EvaluationStatus, Float, FourMomentum, Laurent, MasterProvider,
    MasterRequest, MasterStatus, Real, Result, SingleFloat,
};
use ninjars_symbolica::{
    DualPolynomialNumerator, PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts,
    SymbolicaDualAmplitudeKernels, SymbolicaDualCuts, SymbolicaDualMasters, SymbolicaMasters,
};
use oneloop::NativeFloat;
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{dual::HyperDual, rational::Rational},
    evaluate::EvaluationDomain,
    parse,
};

fn c(value: f64) -> Complex<f64> {
    Complex::new(value, 0.)
}
fn vars() -> [Atom; 5] {
    [
        parse!("q0"),
        parse!("q1"),
        parse!("q2"),
        parse!("q3"),
        parse!("mu2"),
    ]
}
fn near(actual: Complex<f64>, expected: f64) {
    assert!(
        (actual.re - expected).abs() < 1e-9 * (1. + expected.abs()) && actual.im.abs() < 1e-9,
        "{actual} != {expected}"
    );
}

fn precise_shifted_tadpole<T: NativeFloat>(prototype: T)
where
    Complex<T>: Real + EvaluationDomain,
{
    let zero = Complex::new(prototype.zero(), prototype.zero());
    let one = Complex::new(prototype.one(), prototype.zero());
    let value = Complex::new(
        prototype.one() + prototype.from_rational(&Rational::from((1, 10)).pow(30)),
        prototype.zero(),
    );
    assert_ne!(value.re, prototype.one());
    let shift = FourMomentum::from_args(value.clone(), one.clone(), one.clone(), zero.clone());
    let mut numerator =
        PolynomialNumerator::new(parse!("a"), vars(), vec![parse!("a")], prototype.clone())
            .unwrap();
    numerator
        .set_parameter_batch(std::slice::from_ref(&value), 1)
        .unwrap();
    let mut geometry = SymbolicaCuts::new(prototype.clone()).unwrap();
    let mut masters = SymbolicaMasters::new(prototype.clone());
    let mut amplitude = SymbolicaAmplitudeKernels::new(1, 0, prototype)
        .unwrap()
        .into_amplitude(1)
        .unwrap();
    // Translation invariance and A0(m²=1,scale²=1)=[1,1,0]. The numerator
    // normalization is an independent tiny input, so a binary64 round trip fails.
    for copies in 1..=2 {
        amplitude
            .evaluate_batch(
                std::slice::from_ref(&shift),
                std::slice::from_ref(&one),
                std::slice::from_ref(&one),
                &mut numerator,
                &mut geometry,
                &mut masters,
            )
            .unwrap();
        assert_eq!(amplitude.statuses(), &[EvaluationStatus::Success]);
        let expected = if copies == 1 {
            value.clone()
        } else {
            &value + &value
        };
        assert_eq!(
            amplitude.results()[0],
            [
                expected.clone(),
                expected.clone(),
                zero.clone(),
                expected,
                zero.clone()
            ]
        );
    }
    amplitude.reset();
    assert_eq!(
        amplitude.results()[0],
        [zero.clone(), zero.clone(), zero.clone(), zero.clone(), zero]
    );
}

#[test]
fn shifted_tadpole_full_amplitudes_keep_doublefloat_and_float192_precision() {
    precise_shifted_tadpole(DoubleFloat::default());
    precise_shifted_tadpole(Float::new(192));
}

fn integrated_spacelike_bubble<T: NativeFloat + Into<Float>>(prototype: T) -> Float
where
    Complex<T>: Real + EvaluationDomain,
{
    let zero = Complex::new(prototype.zero(), prototype.zero());
    let one = Complex::new(prototype.one(), prototype.zero());
    let shifts = [
        FourMomentum::from_args(zero.clone(), zero.clone(), zero.clone(), zero.clone()),
        FourMomentum::from_args(zero.clone(), one.clone(), zero.clone(), zero.clone()),
    ];
    let mut numerator =
        PolynomialNumerator::new(parse!("1"), vars(), vec![], prototype.clone()).unwrap();
    numerator.set_parameter_batch(&[], 1).unwrap();
    let mut geometry = SymbolicaCuts::new(prototype.clone()).unwrap();
    let mut masters = SymbolicaMasters::new(prototype.clone());
    let mut amplitude = SymbolicaAmplitudeKernels::new(2, 0, prototype)
        .unwrap()
        .into_amplitude(1)
        .unwrap();
    amplitude
        .evaluate_batch(
            &shifts,
            &[one.clone(), one.clone()],
            std::slice::from_ref(&one),
            &mut numerator,
            &mut geometry,
            &mut masters,
        )
        .unwrap();
    assert_eq!(amplitude.statuses(), &[EvaluationStatus::Success]);
    let result = &amplitude.results()[0];
    assert_eq!(result[1], one);
    assert_eq!(result[2], zero);
    assert_eq!(result[0], result[3]);
    assert_eq!(result[4], zero);
    assert_eq!(result[0].im, zero.im);
    // Compare the represented numerical value at high precision. Converting
    // its exact rational value adds no information to the computed amplitude.
    let represented: Float = result[0].re.clone().into();
    Float::new(384).from_rational(&represented.to_rational())
}

#[test]
fn integrated_bubble_finite_part_gains_accuracy_with_working_precision() {
    // Independent Feynman-parameter integral:
    // B0(-1;1,1)|finite = -int_0^1 log(1+x(1-x)) dx.
    // Its closed form is evaluated directly by Symbolica, independently of
    // oneloopmaster's piecewise B0 expression and the reduction pipeline.
    let expected: Float = parse!("2-5^(1/2)*log((5^(1/2)+1)/(5^(1/2)-1))")
        .evaluate_with_prec(&std::collections::HashMap::<Atom, Float>::new(), 384)
        .unwrap();
    let results = [
        ("f64", integrated_spacelike_bubble(0.0), 14),
        (
            "DoubleFloat",
            integrated_spacelike_bubble(DoubleFloat::default()),
            29,
        ),
        ("Float192", integrated_spacelike_bubble(Float::new(192)), 52),
        ("Float256", integrated_spacelike_bubble(Float::new(256)), 70),
    ];
    let prototype = Float::new(384);
    let mut previous_error = None;
    for (name, actual, digits) in results {
        let error = (actual - &expected).norm();
        let tolerance = prototype.from_rational(&Rational::from((1, 10)).pow(digits));
        assert!(error < tolerance, "{name}: absolute finite error {error}");
        if let Some(previous) = previous_error {
            // Each precision increase must deliver a measurable improvement,
            // beyond merely passing an absolute threshold or exact pole test.
            let improvement = prototype.from_rational(&Rational::from((1, 10)).pow(12));
            assert!(
                error < previous * improvement,
                "{name}: finite accuracy did not improve"
            );
        }
        previous_error = Some(error);
    }
}

#[test]
fn derivative_only_weight_keeps_higher_mixed_components_and_failed_lane_history() {
    let shape = vec![
        vec![0, 0],
        vec![1, 0],
        vec![0, 1],
        vec![2, 0],
        vec![1, 1],
        vec![2, 1],
    ];
    let dual = |values: [f64; 6]| HyperDual::from_values(shape.clone(), values.map(c).to_vec());
    let prototype = dual([0.; 6]);
    let a = dual([0., 0., 1., 0., 0., 0.]); // a=y: primal master weight is zero.
    let mut masses = vec![dual([1., 1., 0., 0., 0., 0.]); 2]; // m²=1+x.
    let scales = vec![dual([1., 0., 0., 0., 0., 0.]); 2];
    let mut shifts: Vec<_> = [[2., -1., 3., 4.], [-2., 3., 0., 1.]]
        .into_iter()
        .map(|p| FourMomentum::from(p.map(|v| dual([v, 0., 0., 0., 0., 0.]))))
        .collect();
    shifts[0].temporal.value.values[1] = c(3.);
    shifts[1].spatial.py.values[2] = c(2.);
    let source = PolynomialNumerator::new(parse!("a"), vars(), vec![parse!("a")], 0.).unwrap();
    let mut numerator = DualPolynomialNumerator::new(source, prototype.clone()).unwrap();
    numerator.set_parameter_batch(&[a.clone(), a], 2).unwrap();
    let mut geometry = SymbolicaDualCuts::new(0., prototype.clone()).unwrap();
    let mut masters = SymbolicaDualMasters::new(0., &prototype);
    let mut amplitude = SymbolicaDualAmplitudeKernels::new(1, 0, 0., &prototype)
        .unwrap()
        .into_amplitude(2)
        .unwrap();
    // Analytic control: y*A0(1+x;1). Finite part is
    // y*(1+x)*(1-ln(1+x)) = y - x²*y/2 + O(x³*y);
    // the pole is y+x*y. Components are Taylor coefficients, not derivatives.
    let expected = [
        [0., 0., 1., 0., 0., -0.5],
        [0., 0., 1., 0., 1., 0.],
        [0.; 6],
        [0., 0., 1., 0., 0., -0.5],
        [0.; 6],
    ];
    for attempt in 1..=3 {
        if attempt == 3 {
            masses[1].values[5] = c(f64::NAN);
        }
        amplitude
            .evaluate_batch(
                &shifts,
                &masses,
                &scales,
                &mut numerator,
                &mut geometry,
                &mut masters,
            )
            .unwrap();
        assert_eq!(amplitude.statuses()[0], EvaluationStatus::Success);
        if attempt == 3 {
            assert!(matches!(
                amplitude.statuses()[1],
                EvaluationStatus::CutFailed { .. }
            ));
        } else {
            assert_eq!(amplitude.statuses()[1], EvaluationStatus::Success);
        }
        for lane in 0..2 {
            let copies = if lane == 1 && attempt == 3 {
                2
            } else {
                attempt
            };
            for (got, want) in amplitude.results()[lane].iter().zip(&expected) {
                for (got, want) in got.values.iter().zip(want) {
                    near(*got, want * f64::from(copies));
                }
            }
        }
    }
    let before: Vec<_> = amplitude
        .results()
        .iter()
        .map(|row| row.iter().map(|v| v.values.clone()).collect::<Vec<_>>())
        .collect();
    let statuses = amplitude.statuses().to_vec();
    assert!(
        amplitude
            .evaluate_batch(
                &shifts,
                &masses,
                &scales[..1],
                &mut numerator,
                &mut geometry,
                &mut masters
            )
            .is_err()
    );
    assert_eq!(amplitude.statuses(), statuses);
    assert_eq!(
        before,
        amplitude
            .results()
            .iter()
            .map(|row| row.iter().map(|v| v.values.clone()).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    );
}

struct NoMasters;
impl MasterProvider<Complex<f64>> for NoMasters {
    fn evaluate(
        &mut self,
        _: MasterRequest<'_, Complex<f64>>,
        _: &mut [Laurent<Complex<f64>>],
    ) -> Result<()> {
        panic!("an exactly zero weight requested a scalar master")
    }
    fn evaluate_batch(
        &mut self,
        _: MasterRequest<'_, Complex<f64>>,
        _: &mut [Laurent<Complex<f64>>],
        _: &mut [MasterStatus],
    ) -> Result<()> {
        panic!("an exactly zero weight requested a scalar master batch")
    }
}

#[test]
fn purely_rational_mu_squared_amplitudes_skip_all_zero_weight_masters() {
    let routing = [[0., 0., 0., 0.], [2., 1., 0., 0.], [3., 0., 1., 0.]];
    let mut geometry = SymbolicaCuts::new(0.).unwrap();
    for n in 1..=3 {
        let mut numerator = PolynomialNumerator::new(parse!("mu2"), vars(), vec![], 0.).unwrap();
        numerator.set_parameter_batch(&[], 1).unwrap();
        let mut amplitude = SymbolicaAmplitudeKernels::new(n, 2, 0.)
            .unwrap()
            .into_amplitude(1)
            .unwrap();
        let shifts: Vec<_> = routing[..n]
            .iter()
            .map(|p| FourMomentum::from(p.map(c)))
            .collect();
        amplitude
            .evaluate_batch(
                &shifts,
                &vec![c(1.); n],
                &[c(1.)],
                &mut numerator,
                &mut geometry,
                &mut NoMasters,
            )
            .unwrap();
        assert_eq!(amplitude.statuses(), &[EvaluationStatus::Success]);
        let result = &amplitude.results()[0];
        near(result[1], 0.);
        near(result[2], 0.);
        near(result[3], 0.);
        assert_eq!(result[0], result[4]);
        assert!(result[0].re != 0.);
        if n != 2 {
            near(result[0], 0.5);
        } // independent tadpole/triangle controls
    }
}

struct FailingProvider;
impl MasterProvider<Complex<f64>> for FailingProvider {
    fn evaluate(
        &mut self,
        _: MasterRequest<'_, Complex<f64>>,
        _: &mut [Laurent<Complex<f64>>],
    ) -> Result<()> {
        Err(Error::Unsupported("injected provider API failure".into()))
    }
    fn evaluate_batch(
        &mut self,
        _: MasterRequest<'_, Complex<f64>>,
        _: &mut [Laurent<Complex<f64>>],
        _: &mut [MasterStatus],
    ) -> Result<()> {
        Err(Error::Unsupported("injected provider API failure".into()))
    }
}

#[test]
fn repeated_propagators_and_late_api_errors_preserve_accumulation_and_status() {
    let mut numerator = PolynomialNumerator::new(parse!("1"), vars(), vec![], 0.).unwrap();
    numerator.set_parameter_batch(&[], 1).unwrap();
    let mut geometry = SymbolicaCuts::new(0.).unwrap();
    let mut masters = SymbolicaMasters::new(0.);
    let mut amplitude = SymbolicaAmplitudeKernels::new(2, 0, 0.)
        .unwrap()
        .into_amplitude(1)
        .unwrap();
    let shifts = [
        FourMomentum::from([c(0.); 4]),
        FourMomentum::from([c(0.), c(1.), c(0.), c(0.)]),
    ];
    let masses = [c(1.); 2];
    let scales = [c(1.)];
    amplitude
        .evaluate_batch(
            &shifts,
            &masses,
            &scales,
            &mut numerator,
            &mut geometry,
            &mut masters,
        )
        .unwrap();
    assert_eq!(amplitude.statuses(), &[EvaluationStatus::Success]);
    let before = amplitude.results().to_vec();
    let statuses = amplitude.statuses().to_vec();
    assert!(
        amplitude
            .evaluate_batch(
                &[shifts[0]; 2],
                &masses,
                &scales,
                &mut numerator,
                &mut geometry,
                &mut masters
            )
            .is_err()
    );
    assert_eq!(amplitude.results(), before);
    assert_eq!(amplitude.statuses(), statuses);
    assert!(
        amplitude
            .evaluate_batch(
                &shifts,
                &masses,
                &scales,
                &mut numerator,
                &mut geometry,
                &mut FailingProvider
            )
            .is_err()
    );
    assert_eq!(amplitude.results(), before);
    assert_eq!(amplitude.statuses(), statuses);
}
