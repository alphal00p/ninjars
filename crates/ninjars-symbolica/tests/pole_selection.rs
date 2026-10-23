//! UV/IR selection through scalar/tensor master providers and reduction engines.
use ninjars::fast::{
    C64, FastAmplitude, FastInput, FastMaster, FastMasters, FastOptions, FastResult, FastStatus,
    GenericMasters, RecordingMasters, TensorNumerator,
};
use ninjars::{
    Complex, EvaluationStatus, Float, FloatLike, FourMomentum, MasterIntegral, MasterProvider,
    MasterRequest, MasterStatus, Real,
};
use ninjars_symbolica::{
    OneLoopNativeMasters, PoleSelection, PolynomialNumerator, SymbolicaAmplitudeKernels,
    SymbolicaCuts, SymbolicaMasters,
};
use symbolica::parse;

fn c(x: f64) -> Complex<f64> {
    Complex::new(x, 0.)
}
fn scaleless<M: MasterProvider<Complex<f64>>>(
    masters: &mut M,
    integral: MasterIntegral,
) -> [Complex<f64>; 3] {
    let mut out = [[c(999.); 3]];
    masters
        .evaluate(
            MasterRequest {
                integral,
                invariants: &[c(0.)],
                masses_squared: &[c(0.), c(0.)],
                scales_squared: &[c(7.)],
            },
            &mut out,
        )
        .unwrap();
    out[0]
}
#[test]
fn native_and_symbolica_preserve_scaleless_moments_and_selector_rebinding() {
    let mut native = OneLoopNativeMasters::new().unwrap();
    let mut symbolic = SymbolicaMasters::new(0.);
    for (selection, sign) in [
        (PoleSelection::UV, 1.),
        (PoleSelection::IR, -1.),
        (PoleSelection::BOTH, 0.),
        (PoleSelection::NONE, 0.),
        (PoleSelection::UV, 1.),
    ] {
        native.set_pole_selection(selection);
        symbolic.set_pole_selection(selection);
        assert_eq!(native.pole_selection(), selection);
        assert_eq!(symbolic.pole_selection(), selection);
        for (rank, integral) in [
            MasterIntegral::B0,
            MasterIntegral::B1,
            MasterIntegral::B11,
            MasterIntegral::B111,
            MasterIntegral::B1111,
        ]
        .into_iter()
        .enumerate()
        {
            let expected = sign * (-1.0f64).powi(rank as i32) / (rank + 1) as f64;
            for value in [
                scaleless(&mut native, integral),
                scaleless(&mut symbolic, integral),
            ] {
                assert_eq!(value[0], c(0.));
                assert!((value[1].re - expected).abs() < 1e-15);
                assert_eq!(value[1].im, 0.);
                assert_eq!(value[2], c(0.));
            }
        }
    }
}

#[test]
fn mp_projection_preserves_input_precision_and_failed_rows() {
    let prototype = Float::new(256);
    let c = |x: i64| Complex::new(Float::with_val(256, x), Float::new(256));
    let mass = Float::with_val(256, 1) + Float::with_val(256, 2).pow(100).inv();
    let mut masters = SymbolicaMasters::new(prototype).with_pole_selection(PoleSelection::UV);
    let mut out = [[c(991), c(991), c(991)], [c(992), c(992), c(992)]];
    let mut statuses = [MasterStatus::Success; 2];
    masters
        .evaluate_batch(
            MasterRequest {
                integral: MasterIntegral::A0,
                invariants: &[],
                masses_squared: &[Complex::new(mass.clone(), Float::new(256)), c(4)],
                scales_squared: &[c(4), c(-1)],
            },
            &mut out,
            &mut statuses,
        )
        .unwrap();
    assert_eq!(statuses[0], MasterStatus::Success);
    assert!(matches!(statuses[1], MasterStatus::Failed(_)));
    assert_eq!(out[1], [c(992), c(992), c(992)]);
    assert!((out[0][1].re.clone() - mass).norm() < Float::with_val(256, 2).pow(200).inv());
    assert!(out[0][1].re > Float::with_val(256, 1));
    let finite = out[0][0].clone();
    masters.set_pole_selection(PoleSelection::IR);
    let mut infrared = [[c(0), c(0), c(0)]];
    masters
        .evaluate(
            MasterRequest {
                integral: MasterIntegral::A0,
                invariants: &[],
                masses_squared: &[out[0][1].clone()],
                scales_squared: &[c(4)],
            },
            &mut infrared,
        )
        .unwrap();
    assert_eq!(infrared[0][0], finite);
    assert_eq!(infrared[0][1], c(0));
    assert_eq!(infrared[0][2], c(0));
}

fn assert_fast_rejects<M: FastMasters>(mut masters: M) {
    let shifts = [[0.; 4], [2., 0., 0., 2.]];
    let masses = [C64::ZERO; 2];
    let matrix = [0.; 4];
    for rank in [0, 4] {
        let mut amplitude = FastAmplitude::new(2, rank, FastOptions::default()).unwrap();
        let mut numerator = TensorNumerator::new(rank, 1);
        numerator.coefficients_mut()[0] = C64::real(1.);
        let sentinel = FastResult {
            laurent: [C64::real(993.); 3],
            ..FastResult::default()
        };
        let mut output = [sentinel];
        let mut status = [FastStatus::NotEvaluated];
        let err = amplitude
            .evaluate(
                &FastInput {
                    lanes: 1,
                    shifts: &shifts,
                    masses_squared: &masses,
                    scales_squared: &[7.],
                    s_matrix: Some(&matrix),
                },
                &mut numerator,
                &mut masters,
                &mut output,
                &mut status,
            )
            .unwrap_err();
        assert!(err.to_string().contains("generic reduction engine"));
        assert_eq!(output, [sentinel]);
        assert_eq!(status, [FastStatus::NotEvaluated]);
    }
}
#[test]
fn fast_engine_rejects_selected_poles_before_scaleless_skip_through_all_adapters() {
    assert_fast_rejects(
        OneLoopNativeMasters::new()
            .unwrap()
            .with_pole_selection(PoleSelection::UV),
    );
    assert_fast_rejects(GenericMasters::new(
        SymbolicaMasters::new(0.).with_pole_selection(PoleSelection::IR),
    ));
    assert_fast_rejects(RecordingMasters::new(
        OneLoopNativeMasters::new()
            .unwrap()
            .with_pole_selection(PoleSelection::IR),
    ));
    // Directly recorded selected masters must keep the capability in a replay,
    // even if the provider is changed back to BOTH before constructing it.
    let mut recording = RecordingMasters::new(
        OneLoopNativeMasters::new()
            .unwrap()
            .with_pole_selection(PoleSelection::UV),
    );
    FastMasters::evaluate(
        &mut recording,
        FastMaster::A0,
        &[C64::real(4.), C64::real(7.)],
        &mut [[C64::ZERO; 3]],
        &mut [false],
    )
    .unwrap();
    recording
        .inner_mut()
        .set_pole_selection(PoleSelection::BOTH);
    assert_fast_rejects(recording.replay());
    recording.clear();
    assert!(!recording.replay().requires_scaleless_bubbles());
    let mut wrapped = GenericMasters::new(SymbolicaMasters::new(0.));
    assert!(!wrapped.requires_scaleless_bubbles());
    wrapped.inner_mut().set_pole_selection(PoleSelection::UV);
    assert!(wrapped.requires_scaleless_bubbles());
    wrapped.inner_mut().set_pole_selection(PoleSelection::NONE);
    assert!(!wrapped.requires_scaleless_bubbles());
    wrapped.inner_mut().set_pole_selection(PoleSelection::BOTH);
    assert!(!wrapped.requires_scaleless_bubbles());
    // Restoring BOTH after rejection must make a retained fast engine usable.
    let mut amplitude = FastAmplitude::new(2, 0, FastOptions::default()).unwrap();
    let mut numerator = TensorNumerator::new(0, 1);
    numerator.coefficients_mut()[0] = C64::real(1.);
    let input = FastInput {
        lanes: 1,
        shifts: &[[0.; 4], [2., 0., 0., 2.]],
        masses_squared: &[C64::ZERO; 2],
        scales_squared: &[7.],
        s_matrix: Some(&[0.; 4]),
    };
    let mut results = [FastResult::default()];
    let mut statuses = [FastStatus::NotEvaluated];
    wrapped.inner_mut().set_pole_selection(PoleSelection::IR);
    assert!(
        amplitude
            .evaluate(
                &input,
                &mut numerator,
                &mut wrapped,
                &mut results,
                &mut statuses
            )
            .is_err()
    );
    wrapped.inner_mut().set_pole_selection(PoleSelection::BOTH);
    amplitude
        .evaluate(
            &input,
            &mut numerator,
            &mut wrapped,
            &mut results,
            &mut statuses,
        )
        .unwrap();
    assert_eq!(statuses, [FastStatus::Success]);
    assert_eq!(results[0].laurent, [C64::ZERO; 3]);
}

#[test]
fn generic_reduction_retains_selected_scaleless_coefficients() {
    let vars = [
        parse!("q0"),
        parse!("q1"),
        parse!("q2"),
        parse!("q3"),
        parse!("mu2"),
    ];
    let mut numerator = PolynomialNumerator::new(parse!("q0^2"), vars, vec![], 0.).unwrap();
    numerator.set_parameter_batch(&[], 1).unwrap();
    let mut amplitude = SymbolicaAmplitudeKernels::new(2, 2, 0.)
        .unwrap()
        .into_amplitude(1)
        .unwrap();
    let mut cuts = SymbolicaCuts::new(0.).unwrap();
    let mut masters = OneLoopNativeMasters::new().unwrap();
    let shifts = [
        FourMomentum::from([c(0.); 4]),
        FourMomentum::from([c(2.), c(0.), c(0.), c(2.)]),
    ];
    for (selection, expected) in [
        (PoleSelection::UV, 4. / 3.),
        (PoleSelection::IR, -4. / 3.),
        (PoleSelection::BOTH, 0.),
    ] {
        masters.set_pole_selection(selection);
        amplitude.reset_batch(1);
        amplitude
            .evaluate_batch(
                &shifts,
                &[c(0.), c(0.)],
                &[c(7.)],
                &mut numerator,
                &mut cuts,
                &mut masters,
            )
            .unwrap();
        assert_eq!(amplitude.statuses(), &[EvaluationStatus::Success]);
        let out = &amplitude.results()[0];
        assert!(out[0].re.abs() < 1e-12);
        assert!((out[1].re - expected).abs() < 1e-12);
        assert!(out[2].re.abs() < 1e-12);
    }
}

#[test]
fn selected_failed_native_and_symbolic_rows_keep_outputs_and_recover() {
    let mut native = OneLoopNativeMasters::new()
        .unwrap()
        .with_pole_selection(PoleSelection::UV);
    let mut symbolic = SymbolicaMasters::new(0.).with_pole_selection(PoleSelection::IR);
    for masters in [
        &mut native as &mut dyn MasterProvider<Complex<f64>>,
        &mut symbolic as &mut dyn MasterProvider<Complex<f64>>,
    ] {
        let mut output = [[c(991.); 3]; 2];
        let mut status = [MasterStatus::Success; 2];
        let request = MasterRequest {
            integral: MasterIntegral::B11,
            invariants: &[c(0.), c(0.)],
            masses_squared: &[c(0.); 4],
            scales_squared: &[c(7.), c(-1.)],
        };
        masters
            .evaluate_batch(request, &mut output, &mut status)
            .unwrap();
        assert_eq!(status[0], MasterStatus::Success);
        assert!(matches!(status[1], MasterStatus::Failed(_)));
        assert_eq!(output[1], [c(991.); 3]);
        let previous = output;
        assert!(masters.evaluate(request, &mut output).is_err());
        assert_eq!(output, previous);
        masters
            .evaluate_batch(
                MasterRequest {
                    scales_squared: &[c(7.); 2],
                    ..request
                },
                &mut output,
                &mut status,
            )
            .unwrap();
        assert_eq!(status, [MasterStatus::Success; 2]);
        assert_eq!(output[0], output[1]);
    }
}

#[test]
fn arbitrary_precision_tensor_tadpole_expression_fallback_selects_poles_only() {
    let mut masters = SymbolicaMasters::new(Float::new(256));
    let c = |x: i64| Complex::new(Float::with_val(256, x), Float::new(256));
    for rank in [2, 4] {
        let mut combined = [[c(0), c(0), c(0)]];
        masters.set_pole_selection(PoleSelection::BOTH);
        masters
            .evaluate(
                MasterRequest {
                    integral: MasterIntegral::An { rank },
                    invariants: &[],
                    masses_squared: &[c(4)],
                    scales_squared: &[c(7)],
                },
                &mut combined,
            )
            .unwrap();
        for selection in [PoleSelection::UV, PoleSelection::IR, PoleSelection::NONE] {
            masters.set_pole_selection(selection);
            let mut selected = [[c(991), c(991), c(991)]];
            masters
                .evaluate(
                    MasterRequest {
                        integral: MasterIntegral::An { rank },
                        invariants: &[],
                        masses_squared: &[c(4)],
                        scales_squared: &[c(7)],
                    },
                    &mut selected,
                )
                .unwrap();
            assert_eq!(selected[0][0], combined[0][0]);
            assert_eq!(
                selected[0][1],
                if selection.uv {
                    combined[0][1].clone()
                } else {
                    c(0)
                }
            );
            assert_eq!(selected[0][2], c(0));
        }
    }
}
