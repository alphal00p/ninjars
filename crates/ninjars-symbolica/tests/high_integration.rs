//! Four-cut integration boundary controls. The controllable provider checks the
//! complete D0 request; it is not an independent numerical D0 implementation.
use ninjars::{
    Complex, Error, EvaluationStatus, FourMomentum, IntegrationStatus, Laurent, MasterFailure,
    MasterIntegral, MasterProvider, MasterRequest, MasterStatus, ResidueIntegrationRequest,
    ResidueIntegrator, Result, SampleId,
};
use ninjars_symbolica::{
    PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaDualIntegrationKernels,
    SymbolicaHighCuts, SymbolicaIntegrationKernels,
};
use symbolica::{atom::Atom, domains::dual::HyperDual, parse};

type C = Complex<f64>;
fn c(value: f64) -> C {
    Complex::new(value, 0.)
}
fn near(actual: C, expected: C) {
    assert!(
        (actual.re - expected.re).abs() < 1e-10 * (1. + expected.re.abs())
            && (actual.im - expected.im).abs() < 1e-10 * (1. + expected.im.abs()),
        "{actual} != {expected}"
    );
}
fn point(values: [f64; 4]) -> FourMomentum<C> {
    FourMomentum::from(values.map(c))
}

struct ExpectedRow {
    invariants: [C; 6],
    masses: [C; 4],
    scale: C,
}
fn expected_row(shifts: &[FourMomentum<C>], masses: &[C], p: [usize; 4], scale: C) -> ExpectedRow {
    // The six independent differences specify OneLOop order directly, without
    // using the integration kernel's generated invariant graph.
    let pairs = [
        (p[1], p[0]),
        (p[2], p[1]),
        (p[3], p[2]),
        (p[0], p[3]),
        (p[2], p[0]),
        (p[3], p[1]),
    ];
    ExpectedRow {
        invariants: pairs.map(|(a, b)| {
            let difference = shifts[a] - shifts[b];
            difference.dot(&difference)
        }),
        masses: p.map(|i| masses[i]),
        scale,
    }
}
#[derive(Clone, Copy)]
enum Response {
    Good,
    Failures,
    StructuralError,
}
struct RecordingD0 {
    expected: Vec<ExpectedRow>,
    response: Response,
    calls: Vec<Vec<C>>,
}
impl MasterProvider<C> for RecordingD0 {
    fn evaluate(&mut self, _: MasterRequest<'_, C>, _: &mut [Laurent<C>]) -> Result<()> {
        panic!("four-cut integration must preserve the batched provider boundary")
    }
    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, C>,
        output: &mut [Laurent<C>],
        statuses: &mut [MasterStatus],
    ) -> Result<()> {
        request.validate(output.len())?;
        assert_eq!(request.integral, MasterIntegral::D0);
        assert_eq!(statuses.len(), request.rows());
        self.calls.push(request.scales_squared.to_vec());
        for (row, &scale) in request.scales_squared.iter().enumerate() {
            let expected = self
                .expected
                .iter()
                .find(|r| r.scale == scale)
                .expect("unexpected scale or packed row");
            for (&actual, &expected) in request.invariants[6 * row..6 * row + 6]
                .iter()
                .zip(&expected.invariants)
            {
                near(actual, expected);
            }
            assert_eq!(
                &request.masses_squared[4 * row..4 * row + 4],
                &expected.masses
            );
        }
        if matches!(self.response, Response::StructuralError) {
            return Err(Error::Unsupported(
                "deliberate late D0 provider error".into(),
            ));
        }
        for (row, &scale) in request.scales_squared.iter().enumerate() {
            if matches!(self.response, Response::Failures) && scale.re as usize % 7 == 2 {
                statuses[row] = MasterStatus::Failed(MasterFailure::InvalidMass);
            } else {
                output[row] = [scale * 2., scale * 3., scale * 5.];
                if matches!(self.response, Response::Failures) && scale.re as usize % 7 == 4 {
                    output[row][1] = c(f64::NAN);
                }
                statuses[row] = MasterStatus::Success;
            }
        }
        Ok(())
    }
}

#[test]
fn ordered_d0_arguments_scales_and_failed_tail_rows_preserve_history() {
    let partition = [0, 2, 3, 5];
    let mut integration = ResidueIntegrator::new(
        SymbolicaIntegrationKernels::new(4, true, 0.).unwrap(),
        c(0.),
    );
    for lanes in [0, 1, 8, 33] {
        let shifts: Vec<Vec<_>> = (0..lanes)
            .map(|lane| {
                (0..6)
                    .map(|line| {
                        let t = (line + 1) as f64;
                        FourMomentum::from([
                            Complex::new(2. * t + lane as f64, t / 8.),
                            c(t * t - 2.),
                            c(3. - t),
                            c(t * (t - 1.)),
                        ])
                    })
                    .collect()
            })
            .collect();
        let masses: Vec<Vec<_>> = (0..lanes)
            .map(|lane| {
                (0..6)
                    .map(|line| Complex::new((line + lane + 1) as f64, -(line as f64) / 16.))
                    .collect()
            })
            .collect();
        let scales: Vec<_> = (0..lanes)
            .map(|lane| {
                if lane % 7 == 5 {
                    c(0.)
                } else {
                    c((lane + 1) as f64)
                }
            })
            .collect();
        let coefficients: Vec<_> = (0..lanes)
            .map(|lane| {
                let mut row = [c(0.); 6];
                row[0] = if lane % 7 == 6 {
                    c(0.)
                } else {
                    Complex::new((lane + 2) as f64, 0.5)
                };
                row[4] = Complex::new(6. * (lane + 1) as f64, -3.);
                row[5] = c(91.); // Spurious higher-rank term has no master weight.
                row
            })
            .collect();
        let requests: Vec<_> = (0..lanes)
            .rev()
            .map(|lane| ResidueIntegrationRequest {
                id: SampleId { lane, sample: 9 },
                partition: &partition,
                shifts: &shifts[lane],
                masses_squared: &masses[lane],
                coefficients: &coefficients[lane],
                basis: None,
                scale_squared: &scales[lane],
            })
            .collect();
        let expected = (0..lanes)
            .filter(|&lane| lane % 7 != 5)
            .map(|lane| expected_row(&shifts[lane], &masses[lane], partition, scales[lane]))
            .collect();
        let mut provider = RecordingD0 {
            expected,
            response: Response::Good,
            calls: Vec::new(),
        };
        let mut output = vec![[c(77.); 5]; lanes];
        let mut statuses = vec![IntegrationStatus::NonFiniteResult; lanes];
        integration
            .evaluate_batch(&requests, &mut provider, &mut output, &mut statuses)
            .unwrap();
        let history = output.clone();
        for (row, lane) in (0..lanes).rev().enumerate() {
            if lane % 7 == 5 {
                assert_eq!(statuses[row], IntegrationStatus::InvalidInput);
                assert_eq!(output[row], [c(77.); 5]);
                continue;
            }
            assert_eq!(statuses[row], IntegrationStatus::Success);
            let weight = coefficients[lane][0];
            let rational = -coefficients[lane][4] / 6.;
            for (&actual, expected) in output[row].iter().zip([
                weight * scales[lane] * 2. + rational,
                weight * scales[lane] * 3.,
                weight * scales[lane] * 5.,
                weight * scales[lane] * 2.,
                rational,
            ]) {
                near(actual, expected);
            }
        }
        let packed: Vec<_> = (0..lanes)
            .rev()
            .filter(|&lane| lane % 7 != 5 && lane % 7 != 6)
            .map(|lane| scales[lane])
            .collect();
        assert_eq!(
            provider.calls,
            if packed.is_empty() {
                vec![]
            } else {
                vec![packed.clone()]
            }
        );
        provider.calls.clear();
        provider.response = Response::Failures;
        integration
            .evaluate_batch(&requests, &mut provider, &mut output, &mut statuses)
            .unwrap();
        assert_eq!(
            provider.calls,
            if packed.is_empty() {
                vec![]
            } else {
                vec![packed]
            }
        );
        for (row, lane) in (0..lanes).rev().enumerate() {
            let expected = if lane % 7 == 5 {
                IntegrationStatus::InvalidInput
            } else if lane % 7 == 1 {
                IntegrationStatus::MasterFailed {
                    integral: MasterIntegral::D0,
                    failure: MasterFailure::InvalidMass,
                }
            } else if lane % 7 == 3 {
                IntegrationStatus::MasterFailed {
                    integral: MasterIntegral::D0,
                    failure: MasterFailure::NonFiniteResult,
                }
            } else {
                IntegrationStatus::Success
            };
            assert_eq!(statuses[row], expected);
            assert_eq!(output[row], history[row]);
        }
        if lanes > 0 {
            let saved_status = statuses.clone();
            provider.response = Response::StructuralError;
            assert!(
                integration
                    .evaluate_batch(&requests, &mut provider, &mut output, &mut statuses)
                    .is_err()
            );
            assert_eq!(output, history);
            assert_eq!(statuses, saved_status);
            let invalid = ResidueIntegrationRequest {
                partition: &[0, 2, 2, 5],
                ..requests[0]
            };
            assert!(
                integration
                    .evaluate_batch(
                        &[invalid],
                        &mut provider,
                        &mut output[..1],
                        &mut statuses[..1]
                    )
                    .is_err()
            );
            assert_eq!(output, history);
            assert_eq!(statuses, saved_status);
        }
    }
}

struct ForbiddenMasters;
impl<T> MasterProvider<T> for ForbiddenMasters {
    fn evaluate(&mut self, _: MasterRequest<'_, T>, _: &mut [Laurent<T>]) -> Result<()> {
        panic!("zero D0 weight must not be evaluated")
    }
    fn evaluate_batch(
        &mut self,
        _: MasterRequest<'_, T>,
        _: &mut [Laurent<T>],
        _: &mut [MasterStatus],
    ) -> Result<()> {
        panic!("zero D0 weight must not be evaluated")
    }
}

#[test]
fn rational_box_integrates_all_mixed_components_without_dual_d0() {
    let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]];
    let dual = |values: [C; 5]| HyperDual::from_values(shape.clone(), values.to_vec());
    let zero = dual([c(0.); 5]);
    let mut shifts: Vec<_> = (0..4)
        .map(|line| {
            FourMomentum::from(std::array::from_fn::<_, 4, _>(|axis| {
                dual([c((line + axis) as f64), c(1.), c(2.), c(3.), c(4.)])
            }))
        })
        .collect();
    let masses = vec![dual([c(1.), c(2.), c(3.), c(4.), c(5.)]); 4];
    let scale = dual([c(3.), c(1.), c(0.), c(2.), c(7.)]);
    for higher in [false, true] {
        let mut integration = ResidueIntegrator::new(
            SymbolicaDualIntegrationKernels::new(4, higher, 0., &zero).unwrap(),
            zero.clone(),
        );
        let mut coefficients = vec![vec![zero.clone(); if higher { 6 } else { 5 }]; 17];
        for (lane, row) in coefficients.iter_mut().enumerate() {
            row[4] = dual(std::array::from_fn(|component| {
                Complex::new(
                    if component == 0 {
                        0.
                    } else {
                        6. * (lane + component) as f64
                    },
                    6. * component as f64,
                )
            }));
        }
        coefficients[3][1].values[4] = c(f64::NAN); // Even a spurious component is validated.
        coefficients[7][4] = HyperDual::from_values(vec![vec![0], vec![1]], vec![c(0.); 2]);
        let requests: Vec<_> = (0..17)
            .rev()
            .map(|lane| ResidueIntegrationRequest {
                id: SampleId { lane, sample: 0 },
                partition: &[0, 1, 2, 3],
                shifts: &shifts,
                masses_squared: &masses,
                coefficients: &coefficients[lane],
                basis: None,
                scale_squared: &scale,
            })
            .collect();
        let sentinel = dual([c(77.); 5]);
        let mut output: Vec<_> = (0..17)
            .map(|_| std::array::from_fn(|_| sentinel.clone()))
            .collect();
        let mut statuses = vec![IntegrationStatus::NonFiniteResult; 17];
        integration
            .evaluate_batch(&requests, &mut ForbiddenMasters, &mut output, &mut statuses)
            .unwrap();
        for (row, lane) in (0..17).rev().enumerate() {
            if [3, 7].contains(&lane) {
                assert_eq!(statuses[row], IntegrationStatus::InvalidInput);
                for value in &output[row] {
                    assert_eq!(value.values, sentinel.values);
                }
                continue;
            }
            assert_eq!(statuses[row], IntegrationStatus::Success);
            for (component, value) in output[row].iter().enumerate() {
                for (index, &actual) in value.values.iter().enumerate() {
                    near(
                        actual,
                        if [0, 4].contains(&component) {
                            -coefficients[lane][4].values[index] / 6.
                        } else {
                            c(0.)
                        },
                    );
                }
            }
        }
    }
    // A missing basis is specific to boxes; lower-cut requests still reject it.
    let coefficients = vec![zero; 15];
    shifts.truncate(3);
    let invalid = ResidueIntegrationRequest {
        id: SampleId { lane: 0, sample: 0 },
        partition: &[0, 1, 2],
        shifts: &shifts,
        masses_squared: &masses[..3],
        coefficients: &coefficients,
        basis: None,
        scale_squared: &scale,
    };
    assert!(invalid.validate(3, 15).is_err());
}

#[test]
fn four_line_amplitude_preserves_accumulation_after_late_errors_and_lane_failures() {
    let mut numerator = PolynomialNumerator::new(
        parse!("1"),
        ["q0", "q1", "q2", "q3", "mu2"]
            .map(|name| Atom::parse(name, "high_boundary", Default::default()).unwrap()),
        vec![],
        0.,
    )
    .unwrap();
    numerator.set_parameter_batch(&[], 3).unwrap();
    let mut amplitude = SymbolicaAmplitudeKernels::new(4, 0, 0.)
        .unwrap()
        .into_amplitude(3)
        .unwrap();
    let mut low = SymbolicaCuts::new(0.).unwrap();
    let mut high = SymbolicaHighCuts::new(0.).unwrap();
    let base = [
        [0., 0., 0., 0.],
        [2., 1., 0., 0.],
        [3., 0., 1., 0.],
        [4., 0., 0., 1.],
    ]
    .map(point);
    let mut shifts: Vec<_> = (0..3).flat_map(|_| base).collect();
    let masses: Vec<_> = (0..3).flat_map(|_| [c(1.), c(2.), c(3.), c(4.)]).collect();
    let scales = [c(1.), c(2.), c(3.)];
    let mut provider = RecordingD0 {
        expected: (0..3)
            .map(|lane| {
                expected_row(
                    &base,
                    &masses[lane * 4..lane * 4 + 4],
                    [0, 1, 2, 3],
                    scales[lane],
                )
            })
            .collect(),
        response: Response::Good,
        calls: Vec::new(),
    };
    amplitude
        .evaluate_batch_with_high_cuts(
            &shifts,
            &masses,
            &scales,
            &[c(5.); 3],
            &mut numerator,
            &mut low,
            &mut high,
            &mut provider,
        )
        .unwrap();
    assert_eq!(amplitude.statuses(), &[EvaluationStatus::Success; 3]);
    let history = amplitude.results().to_vec();
    for (lane, row) in history.iter().enumerate() {
        for (&actual, expected) in row.iter().zip([
            scales[lane] * 2.,
            scales[lane] * 3.,
            scales[lane] * 5.,
            scales[lane] * 2.,
            c(0.),
        ]) {
            near(actual, expected);
        }
    }
    provider.response = Response::StructuralError;
    assert!(
        amplitude
            .evaluate_batch_with_high_cuts(
                &shifts,
                &masses,
                &scales,
                &[c(5.); 3],
                &mut numerator,
                &mut low,
                &mut high,
                &mut provider
            )
            .is_err()
    );
    assert_eq!(amplitude.results(), history);
    assert_eq!(amplitude.statuses(), &[EvaluationStatus::Success; 3]);
    provider.response = Response::Failures;
    amplitude
        .evaluate_batch_with_high_cuts(
            &shifts,
            &masses,
            &scales,
            &[c(5.); 3],
            &mut numerator,
            &mut low,
            &mut high,
            &mut provider,
        )
        .unwrap();
    for (lane, previous) in history.iter().enumerate() {
        for (&actual, expected) in amplitude.results()[lane]
            .iter()
            .zip(previous.map(|v| v * if lane == 1 { 1. } else { 2. }))
        {
            near(actual, expected);
        }
    }
    assert!(matches!(
        amplitude.statuses()[1],
        EvaluationStatus::IntegrationFailed {
            status: IntegrationStatus::MasterFailed {
                integral: MasterIntegral::D0,
                failure: MasterFailure::InvalidMass
            },
            ..
        }
    ));
    // Different masses avoid the repeated-propagator API rejection; the box
    // geometry itself is rank deficient and its lane never reaches the provider.
    shifts[1] = shifts[0];
    provider.calls.clear();
    amplitude
        .evaluate_batch_with_high_cuts(
            &shifts,
            &masses,
            &scales,
            &[c(5.); 3],
            &mut numerator,
            &mut low,
            &mut high,
            &mut provider,
        )
        .unwrap();
    assert_eq!(provider.calls, vec![vec![c(2.), c(3.)]]);
    assert!(matches!(
        amplitude.statuses()[0],
        EvaluationStatus::CutFailed { .. }
    ));
    for (lane, previous) in history.iter().enumerate() {
        for (&actual, expected) in amplitude.results()[lane]
            .iter()
            .zip(previous.map(|v| v * [2., 1., 3.][lane]))
        {
            near(actual, expected);
        }
    }
    amplitude.reset();
    assert_eq!(amplitude.results(), &[[c(0.); 5]; 3]);
    assert_eq!(amplitude.statuses(), &[EvaluationStatus::NotEvaluated; 3]);
}
