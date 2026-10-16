use ninjars::{
    BoxCoefficientMode, BoxMuMode, BoxReductionOptions, Complex, CutFailure, DoubleFloat,
    EvaluationStatus, Float, FloatLike, FourMomentum, Real, SamplingScaleProvider,
    SamplingScaleSource as Source, SamplingScaleStatus as Status, SingleFloat,
};
use ninjars_symbolica::{
    PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaDualSamplingScales,
    SymbolicaHighCuts, SymbolicaMasters, SymbolicaSamplingScales,
};
use symbolica::{
    atom::Atom,
    domains::{
        dual::{DualNumber, DualNumberStructure, HyperDual},
        rational::Rational,
    },
    evaluate::EvaluationDomain,
    parser::ParseSettings,
};

fn c(x: f64) -> Complex<f64> {
    Complex::new(x, 0.)
}
fn p(x: f64) -> FourMomentum<Complex<f64>> {
    FourMomentum::from([c(0.), c(x), c(0.), c(0.)])
}
const SENTINEL: Status = Status::InvalidInput(CutFailure::SingularParameter);

#[test]
fn automatic_scales_are_batched_routing_invariant_homogeneous_and_transactional() {
    let cases = [
        (
            3.,
            [c(1.), c(2.)],
            9.,
            Source::Invariant {
                first: 0,
                second: 1,
            },
        ),
        (0., [c(0.); 2], 1., Source::UnitFallback),
        (
            0.25,
            [Complex::new(0., -0.5), c(0.)],
            0.5,
            Source::Mass { propagator: 0 },
        ),
        (1., [c(1.); 2], 1., Source::Mass { propagator: 0 }),
    ];
    let mut provider = SymbolicaSamplingScales::new(2, 0.).unwrap();
    for order in [&[][..], &[2][..], &[3, 0, 2, 1, 0][..], &[0, 1, 2, 3][..]] {
        let shifts: Vec<_> = order.iter().flat_map(|&i| [p(0.), p(cases[i].0)]).collect();
        let masses: Vec<_> = order.iter().flat_map(|&i| cases[i].1).collect();
        let mut scales = vec![c(-17.); order.len()];
        let mut statuses = vec![SENTINEL; order.len()];
        provider
            .evaluate_batch(&shifts, &masses, order.len(), &mut scales, &mut statuses)
            .unwrap();
        for (lane, &i) in order.iter().enumerate() {
            assert_eq!(scales[lane], c(cases[i].2));
            assert_eq!(statuses[lane], Status::Selected(cases[i].3));
        }
        let before = scales.clone();
        let status_before = statuses.clone();
        assert!(
            provider
                .evaluate_batch(
                    &shifts,
                    &masses,
                    order.len() + 1,
                    &mut scales,
                    &mut statuses
                )
                .is_err()
        );
        assert_eq!(scales, before);
        assert_eq!(statuses, status_before);
    }
    for dilation in [0.125, 1., 8.] {
        let origin = FourMomentum::from([c(13.), c(-7.), c(2.), c(3.)]);
        let shifts = [origin, origin + p(3.) * c(dilation)];
        let mut scale = [c(0.)];
        let mut status = [SENTINEL];
        provider
            .evaluate_batch(
                &shifts,
                &[c(dilation * dilation); 2],
                1,
                &mut scale,
                &mut status,
            )
            .unwrap();
        assert_eq!(scale[0], c(9. * dilation * dilation));
        assert_eq!(
            status[0],
            Status::Selected(Source::Invariant {
                first: 0,
                second: 1
            })
        );
    }
    // ABS delegates to Numerica's scaled hypot, avoiding overflow from squaring.
    let shifts = [p(0.), p(0.), p(0.), p(1e200), p(0.), p(1.)];
    let masses = [
        Complex::new(1e200, 1e200),
        c(0.),
        c(0.),
        c(0.),
        c(f64::NAN),
        c(1.),
    ];
    let mut scales = [c(-17.); 3];
    let mut statuses = [SENTINEL; 3];
    provider
        .evaluate_batch(&shifts, &masses, 3, &mut scales, &mut statuses)
        .unwrap();
    assert!((scales[0].re / 1e200 - 2_f64.sqrt()).abs() < 1e-15);
    assert_eq!(
        statuses[0],
        Status::Selected(Source::Mass { propagator: 0 })
    );
    assert_eq!(&scales[1..], &[c(-17.); 2]);
    assert_eq!(
        statuses[1],
        Status::InvalidInput(CutFailure::NonFiniteResult)
    );
    assert_eq!(
        statuses[2],
        Status::InvalidInput(CutFailure::NonFiniteInput)
    );
    assert!(SymbolicaSamplingScales::new(0, 0.).is_err());
}

fn precision<T: Real + SingleFloat>(prototype: T)
where
    Complex<T>: Real + EvaluationDomain,
{
    let zero = Complex::new(prototype.zero(), prototype.zero());
    let precise = Complex::new(
        prototype.one() + prototype.from_rational(&Rational::from((1, 10)).pow(30)),
        prototype.zero(),
    );
    let mut provider = SymbolicaSamplingScales::new(2, prototype.clone()).unwrap();
    let mut axis = FourMomentum::zero(&zero);
    axis.spatial.px = zero.one();
    let mut scales = [zero.clone()];
    let mut statuses = [SENTINEL];
    provider
        .evaluate_batch(
            &[FourMomentum::zero(&zero), axis],
            &[zero.one(), precise.clone()],
            1,
            &mut scales,
            &mut statuses,
        )
        .unwrap();
    assert_ne!(scales[0].re, prototype.one());
    assert_eq!(scales[0], precise);
    assert_eq!(
        statuses[0],
        Status::Selected(Source::Mass { propagator: 1 })
    );
    let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]];
    let dual_zero = HyperDual::from_values(shape.clone(), vec![zero.clone(); 5]);
    let seed = HyperDual::from_values(shape, vec![precise.clone(); 5]);
    let mut provider = SymbolicaDualSamplingScales::new(2, prototype, dual_zero.clone()).unwrap();
    let mut output = [dual_zero.clone()];
    provider
        .evaluate_batch(
            &vec![FourMomentum::zero(&dual_zero); 2],
            &[dual_zero.clone(), seed],
            1,
            &mut output,
            &mut statuses,
        )
        .unwrap();
    assert_eq!(output[0].values()[0], precise);
    assert!(output[0].values()[1..].iter().all(FloatLike::is_fully_zero));
    assert_eq!(output[0].get_shape(), dual_zero.get_shape());
}
#[test]
fn automatic_scales_preserve_doublefloat_and_arbitrary_precision() {
    precision(DoubleFloat::default());
    precision(Float::new(256));
}

#[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
mod static_dual {
    symbolica::create_hyperdual_from_components!(
        ScaleDual,
        [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1]]
    );
}
fn dual_control<D: Real + DualNumber<Complex<f64>>>(zero: D) {
    let mut seed = zero.zero();
    seed.values_mut()
        .copy_from_slice(&[c(1.), c(7.), c(11.), c(13.), c(17.)]);
    let mut bad = seed.clone();
    bad.values_mut()[4] = c(f64::NAN);
    let mut provider = SymbolicaDualSamplingScales::new(2, 0., zero.clone()).unwrap();
    let mut axis = FourMomentum::zero(&zero);
    axis.spatial.px = seed.clone();
    let shifts = [
        FourMomentum::zero(&zero),
        axis.clone(),
        FourMomentum::zero(&zero),
        axis,
    ];
    let masses = [seed.clone(), seed.clone(), seed.clone(), bad];
    let mut scales = vec![seed.clone(); 2];
    let mut statuses = [SENTINEL; 2];
    provider
        .evaluate_batch(&shifts, &masses, 2, &mut scales, &mut statuses)
        .unwrap();
    assert_eq!(scales[0].values(), &[c(1.), c(0.), c(0.), c(0.), c(0.)]);
    assert_eq!(scales[0].get_shape(), zero.get_shape());
    assert_eq!(scales[1].values(), seed.values());
    assert_eq!(
        statuses,
        [
            Status::Selected(Source::Mass { propagator: 0 }),
            Status::InvalidInput(CutFailure::NonFiniteInput)
        ]
    );
}
#[test]
fn higher_mixed_static_dynamic_dual_scales_are_constant_and_invalid_lanes_isolated() {
    dual_control(static_dual::ScaleDual { values: [c(0.); 5] });
    let zero = HyperDual::from_values(
        vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]],
        vec![c(0.); 5],
    );
    dual_control(zero.clone());
    let mut provider = SymbolicaDualSamplingScales::new(2, 0., zero.clone()).unwrap();
    let bad = HyperDual::from_values(vec![vec![0], vec![1]], vec![c(0.); 2]);
    let mut scales = [zero.one(), zero.one()];
    let mut statuses = [SENTINEL; 2];
    provider
        .evaluate_batch(
            &vec![FourMomentum::zero(&zero); 4],
            &[zero.one(), zero.zero(), zero.one(), bad],
            2,
            &mut scales,
            &mut statuses,
        )
        .unwrap();
    assert_eq!(
        statuses,
        [
            Status::Selected(Source::Mass { propagator: 0 }),
            Status::InvalidInput(CutFailure::DerivativeShape)
        ]
    );
    assert_eq!(scales[1].values(), zero.one().values());
}

#[test]
fn six_line_rank_seven_automatic_scale_restores_reference_accuracy_in_all_box_modes() {
    let evidence: serde_json::Value = serde_json::from_str(include_str!(
        "../../../validation/evidence/high-reduction.json"
    ))
    .unwrap();
    let cases: Vec<_> = evidence["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|case| case["rank"] == 7)
        .collect();
    assert_eq!(cases.len(), 2);
    let variables = ["q0", "q1", "q2", "q3", "mu2"]
        .map(|v| Atom::parse(v, "ninjars_sampling_test", ParseSettings::default()).unwrap());
    let expression = Atom::parse(
        cases[0]["expression"].as_str().unwrap(),
        "ninjars_sampling_test",
        ParseSettings::default(),
    )
    .unwrap();
    let mut numerator = PolynomialNumerator::new(expression, variables, vec![], 0.).unwrap();
    numerator.set_parameter_batch(&[], 2).unwrap();
    let shifts: Vec<_> = cases
        .iter()
        .flat_map(|case| {
            case["shifts"].as_array().unwrap().iter().map(|p| {
                FourMomentum::from(std::array::from_fn::<_, 4, _>(|i| {
                    c(p[i].as_f64().unwrap())
                }))
            })
        })
        .collect();
    let masses: Vec<_> = cases
        .iter()
        .flat_map(|case| {
            case["masses_squared"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| Complex::new(v[0].as_f64().unwrap(), v[1].as_f64().unwrap()))
        })
        .collect();
    let mut scales = [c(0.); 2];
    let mut statuses = [SENTINEL; 2];
    SymbolicaSamplingScales::new(6, 0.)
        .unwrap()
        .evaluate_batch(&shifts, &masses, 2, &mut scales, &mut statuses)
        .unwrap();
    assert_eq!(scales, [c(247.); 2]);
    let mut low = SymbolicaCuts::new(0.).unwrap();
    let mut high = SymbolicaHighCuts::new(0.).unwrap();
    let mut masters = SymbolicaMasters::new(0.);
    let mut automatic_error = 0_f64;
    let mut small_error = 0_f64;
    for coefficients in [BoxCoefficientMode::Integrated, BoxCoefficientMode::Full] {
        for mode in [BoxMuMode::Expansion, BoxMuMode::Sampling] {
            let mut amplitude = SymbolicaAmplitudeKernels::new(6, 7, 0.)
                .unwrap()
                .with_box_options(BoxReductionOptions {
                    coefficients,
                    mu_extraction: mode,
                })
                .into_amplitude(2)
                .unwrap();
            for (manual, scale) in [(false, scales), (true, [c(3.); 2])] {
                if manual
                    && !(coefficients == BoxCoefficientMode::Full && mode == BoxMuMode::Sampling)
                {
                    continue;
                }
                amplitude.reset();
                amplitude
                    .evaluate_batch_with_high_cuts(
                        &shifts,
                        &masses,
                        &[c(1.); 2],
                        &scale,
                        &mut numerator,
                        &mut low,
                        &mut high,
                        &mut masters,
                    )
                    .unwrap();
                for (lane, case) in cases.iter().enumerate() {
                    assert_eq!(amplitude.statuses()[lane], EvaluationStatus::Success);
                    for coefficient in 0..3 {
                        for part in 0..2 {
                            let value = amplitude.results()[lane][coefficient];
                            let actual = if part == 0 { value.re } else { value.im };
                            let expected = case["reference"]["epsilon"][coefficient][part]
                                .as_f64()
                                .unwrap();
                            let error = (actual - expected).abs() / (1. + expected.abs());
                            if manual {
                                small_error = small_error.max(error);
                            } else {
                                automatic_error = automatic_error.max(error);
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("automatic scale error {automatic_error:e}; fixed scale three error {small_error:e}");
    assert!(automatic_error < 1e-10, "{automatic_error:e}");
    assert!(automatic_error < small_error / 10.);
}
