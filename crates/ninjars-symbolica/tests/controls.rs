use ninjars::{
    AmplitudeInput, Complex, CutBasis, EvaluationStatus, ExpansionKind, ExpansionRequest,
    FloatLike, FourMomentum, IntegrationStatus, Laurent, MasterIntegral, MasterProvider,
    MasterRequest, MasterStatus, Numerator, NumeratorRequest, ResidueIntegrationRequest,
    ResidueIntegrator, Result, SampleId,
};
use ninjars_symbolica::{
    PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaDualIntegrationKernels,
    SymbolicaHighCuts, SymbolicaIntegrationKernels,
};
use symbolica::{
    atom::Atom,
    domains::dual::{DualNumber, HyperDual},
    parse,
};

fn c(x: f64) -> Complex<f64> {
    Complex::new(x, 0.)
}
fn q(v: [f64; 4]) -> FourMomentum<Complex<f64>> {
    FourMomentum::from(v.map(c))
}
fn variables() -> [Atom; 5] {
    [
        parse!("q0"),
        parse!("q1"),
        parse!("q2"),
        parse!("q3"),
        parse!("mu2"),
    ]
}
fn basis() -> CutBasis<Complex<f64>> {
    CutBasis {
        e1: q([1., -1., 0., 0.]),
        e2: q([1., 1., 0., 0.]),
        e3: FourMomentum::from([c(0.), c(0.), c(1.), Complex::new(0., 1.)]),
        e4: FourMomentum::from([c(0.), c(0.), c(1.), Complex::new(0., -1.)]),
        e12: c(2.),
    }
}

#[derive(Default)]
struct InvariantMaster {
    families: Vec<MasterIntegral>,
}
impl<T: FloatLike> MasterProvider<T> for InvariantMaster {
    fn evaluate(&mut self, _: MasterRequest<'_, T>, _: &mut [Laurent<T>]) -> Result<()> {
        panic!("scalar master path")
    }
    fn evaluate_batch(
        &mut self,
        r: MasterRequest<'_, T>,
        out: &mut [Laurent<T>],
        statuses: &mut [MasterStatus],
    ) -> Result<()> {
        r.validate(out.len())?;
        self.families.push(r.integral);
        for (i, row) in out.iter_mut().enumerate() {
            let value = if r.integral.invariant_count() > 0 {
                r.invariants[i * r.integral.invariant_count()].clone()
            } else {
                r.scales_squared[i].zero()
            };
            *row = [value.clone(), value.zero(), value.zero()];
            statuses[i] = MasterStatus::Success;
        }
        Ok(())
    }
}

#[test]
fn supplied_invariants_control_both_masters_and_rational_terms() {
    let shifts = [q([0.; 4]), q([0., 1., 0., 0.])];
    let masses = [c(1.); 2];
    let mut coefficients = [c(0.); 10];
    coefficients[0] = c(2.);
    coefficients[9] = c(3.);
    let basis = basis();
    let scale = c(1.);
    let matrices = [
        [c(0.), c(-4.), c(-4.), c(0.)],
        [c(0.), c(-4.), c(-3.), c(0.)],
        [c(1.), c(-4.), c(-4.), c(0.)],
    ];
    let requests: Vec<_> = matrices
        .iter()
        .enumerate()
        .map(|(lane, matrix)| ResidueIntegrationRequest {
            id: SampleId { lane, sample: 0 },
            partition: &[0, 1],
            shifts: &shifts,
            masses_squared: &masses,
            coefficients: &coefficients,
            basis: Some(&basis),
            scale_squared: &scale,
            s_matrix: Some(matrix),
        })
        .collect();
    let mut integration = ResidueIntegrator::new(
        SymbolicaIntegrationKernels::new(2, false, 0.).unwrap(),
        c(0.),
    );
    let mut provider = InvariantMaster::default();
    let mut output = [[c(99.); 5]; 3];
    let mut statuses = [IntegrationStatus::NonFiniteResult; 3];
    integration
        .evaluate_batch(&requests, &mut provider, &mut output, &mut statuses)
        .unwrap();
    assert_eq!(
        statuses,
        [
            IntegrationStatus::Success,
            IntegrationStatus::InvalidInput,
            IntegrationStatus::InvalidInput
        ]
    );
    // c0*s + c9*(3m0²+3m1²-s)/6 = -8+5, not the momentum-derived 1.5.
    assert_eq!(output[0], [c(-3.), c(0.), c(0.), c(-8.), c(5.)]);
    assert_eq!(&output[1..], &[[c(99.); 5]; 2]);
    let saved = output;
    let saved_status = statuses;
    let malformed = ResidueIntegrationRequest {
        s_matrix: Some(&matrices[0][..3]),
        ..requests[0]
    };
    assert!(
        integration
            .evaluate_batch(
                &[malformed],
                &mut provider,
                &mut output[..1],
                &mut statuses[..1]
            )
            .is_err()
    );
    assert_eq!(output, saved);
    assert_eq!(statuses, saved_status);
}

#[test]
fn supplied_invariants_preserve_mixed_duals_and_validate_all_components() {
    let prototype = HyperDual::from_values(
        vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]],
        vec![c(0.); 5],
    );
    let lift = |x: Complex<f64>| {
        let mut d = prototype.zero();
        d.values_mut()[0] = x;
        d
    };
    let shifts = [q([0.; 4]).map(lift), q([0., 1., 0., 0.]).map(lift)];
    let b = basis();
    let basis = CutBasis {
        e1: b.e1.map(lift),
        e2: b.e2.map(lift),
        e3: b.e3.map(lift),
        e4: b.e4.map(lift),
        e12: lift(b.e12),
    };
    let masses = [prototype.from_i64(1), prototype.from_i64(1)];
    let scale = prototype.from_i64(1);
    let mut s = prototype.zero();
    s.values_mut()
        .clone_from_slice(&[c(-2.), c(3.), c(5.), c(7.), c(11.)]);
    let matrix = [prototype.zero(), s.clone(), s.clone(), prototype.zero()];
    let mut coefficients = vec![prototype.zero(); 10];
    coefficients[0] = prototype.from_i64(2);
    coefficients[9] = prototype.from_i64(3);
    let mut integration = ResidueIntegrator::new(
        SymbolicaDualIntegrationKernels::new(2, false, 0., &prototype).unwrap(),
        prototype.zero(),
    );
    let mut output = [std::array::from_fn(|_| prototype.from_i64(99))];
    let mut statuses = [IntegrationStatus::InvalidInput];
    let mut provider = InvariantMaster::default();
    let request = |matrix| ResidueIntegrationRequest {
        id: SampleId { lane: 0, sample: 0 },
        partition: &[0, 1],
        shifts: &shifts,
        masses_squared: &masses,
        coefficients: &coefficients,
        basis: Some(&basis),
        scale_squared: &scale,
        s_matrix: Some(matrix),
    };
    integration
        .evaluate_batch(
            &[request(&matrix)],
            &mut provider,
            &mut output,
            &mut statuses,
        )
        .unwrap();
    assert_eq!(statuses, [IntegrationStatus::Success]);
    assert_eq!(
        output[0][0].values(),
        &[c(0.), c(4.5), c(7.5), c(10.5), c(16.5)]
    );
    let history = output.clone();
    // Primal equality alone would miss this nonsymmetric mixed derivative.
    let mut asymmetric = matrix.clone();
    asymmetric[2].values_mut()[4] = c(12.);
    integration
        .evaluate_batch(
            &[request(&asymmetric)],
            &mut provider,
            &mut output,
            &mut statuses,
        )
        .unwrap();
    assert_eq!(statuses, [IntegrationStatus::InvalidInput]);
    for (a, b) in output[0].iter().zip(&history[0]) {
        assert_eq!(a.values(), b.values());
    }
}

struct RecordedNumerator {
    inner: PolynomialNumerator<f64>,
    partitions: Vec<usize>,
}

#[test]
fn supplied_invariants_follow_noncontiguous_triangle_and_box_partitions() {
    struct Expected(Vec<Complex<f64>>);
    impl MasterProvider<Complex<f64>> for Expected {
        fn evaluate(
            &mut self,
            _: MasterRequest<'_, Complex<f64>>,
            _: &mut [Laurent<Complex<f64>>],
        ) -> Result<()> {
            panic!("scalar master path")
        }
        fn evaluate_batch(
            &mut self,
            request: MasterRequest<'_, Complex<f64>>,
            output: &mut [Laurent<Complex<f64>>],
            statuses: &mut [MasterStatus],
        ) -> Result<()> {
            assert_eq!(request.invariants, self.0);
            output.fill([c(0.); 3]);
            statuses.fill(MasterStatus::Success);
            Ok(())
        }
    }
    let shifts: Vec<_> = (0..5).map(|i| q([0., i as f64, 0., 0.])).collect();
    let masses = [c(1.); 5];
    let matrix: Vec<_> = (0..5)
        .flat_map(|i| {
            (0..5).map(move |j| {
                if i == j {
                    c(0.)
                } else {
                    c(-((10 * i.min(j) + i.max(j) + 1) as f64))
                }
            })
        })
        .collect();
    let b = basis();
    for (partition, expected) in [
        (vec![1, 3, 4], vec![-14., -35., -15.]),
        (vec![0, 1, 3, 4], vec![-2., -14., -35., -5., -4., -15.]),
    ] {
        let size = partition.len();
        let mut coefficients = vec![c(0.); if size == 3 { 10 } else { 5 }];
        coefficients[0] = c(1.);
        let request = ResidueIntegrationRequest {
            id: SampleId { lane: 0, sample: 0 },
            partition: &partition,
            shifts: &shifts,
            masses_squared: &masses,
            coefficients: &coefficients,
            basis: if size == 3 { Some(&b) } else { None },
            scale_squared: &c(1.),
            s_matrix: Some(&matrix),
        };
        let mut integration = ResidueIntegrator::new(
            SymbolicaIntegrationKernels::new(size, false, 0.).unwrap(),
            c(0.),
        );
        let mut provider = Expected(expected.into_iter().map(c).collect());
        let mut output = [[c(99.); 5]];
        let mut statuses = [IntegrationStatus::InvalidInput];
        integration
            .evaluate_batch(&[request], &mut provider, &mut output, &mut statuses)
            .unwrap();
        assert_eq!(statuses, [IntegrationStatus::Success]);
        assert_eq!(output, [[c(0.); 5]]);
    }
}
impl Numerator<Complex<f64>> for RecordedNumerator {
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn evaluate_batch(
        &mut self,
        r: &[NumeratorRequest<'_, Complex<f64>>],
        out: &mut [Complex<f64>],
    ) -> Result<()> {
        self.partitions.extend(r.iter().map(|r| r.partition.len()));
        self.inner.evaluate_batch(r, out)
    }
    fn expansion_batch(
        &mut self,
        k: &ExpansionKind,
        r: &[ExpansionRequest<'_, Complex<f64>>],
        out: &mut [Complex<f64>],
    ) -> Result<()> {
        self.partitions.extend(r.iter().map(|r| r.partition.len()));
        self.inner.expansion_batch(k, r, out)
    }
}

#[test]
fn cut_stopping_skips_callbacks_and_preserves_accumulated_history() {
    let mut inner = PolynomialNumerator::new(parse!("mu2"), variables(), vec![], 0.).unwrap();
    inner.set_parameter_batch(&[], 1).unwrap();
    let mut numerator = RecordedNumerator {
        inner,
        partitions: Vec::new(),
    };
    let mut amplitude = SymbolicaAmplitudeKernels::new(3, 2, 0.)
        .unwrap()
        .into_amplitude(1)
        .unwrap();
    let mut low = SymbolicaCuts::new(0.).unwrap();
    let mut provider = InvariantMaster::default();
    let shifts = [q([0.; 4]), q([2., 1., 0., 0.]), q([3., 0., 1., 0.])];
    amplitude.set_cut_stop(3);
    amplitude
        .evaluate_batch(
            &shifts,
            &[c(1.); 3],
            &[c(1.)],
            &mut numerator,
            &mut low,
            &mut provider,
        )
        .unwrap();
    assert_eq!(amplitude.statuses(), &[EvaluationStatus::Success]);
    assert!((amplitude.results()[0][0].re - 0.5).abs() < 1e-12);
    assert!(numerator.partitions.iter().all(|&size| size == 3));
    assert!(amplitude.bubbles().is_empty() && amplitude.tadpoles().is_empty());
    assert!(
        provider
            .families
            .iter()
            .all(|family| *family == MasterIntegral::C0)
    );
    let history = amplitude.results().to_vec();
    numerator.partitions.clear();
    provider.families.clear();
    amplitude.set_cut_stop(5);
    amplitude
        .evaluate_batch(
            &shifts,
            &[c(1.); 3],
            &[c(1.)],
            &mut numerator,
            &mut low,
            &mut provider,
        )
        .unwrap();
    assert_eq!(amplitude.results(), history);
    assert!(numerator.partitions.is_empty() && provider.families.is_empty());
    amplitude.set_cut_stop(0);
    assert_eq!(amplitude.cut_stop(), 1);
    amplitude
        .evaluate_batch(
            &shifts,
            &[c(1.); 3],
            &[c(1.)],
            &mut numerator,
            &mut low,
            &mut provider,
        )
        .unwrap();
    for size in [1, 2, 3] {
        assert!(numerator.partitions.contains(&size));
    }
    assert!((amplitude.results()[0][0].re - 1.).abs() < 1e-9);
}

#[test]
fn borrowed_s_matrix_reaches_amplitude_and_shape_errors_preserve_history() {
    let mut numerator = PolynomialNumerator::new(parse!("1"), variables(), vec![], 0.).unwrap();
    numerator.set_parameter_batch(&[], 1).unwrap();
    let mut amplitude = SymbolicaAmplitudeKernels::new(2, 0, 0.)
        .unwrap()
        .into_amplitude(1)
        .unwrap();
    let mut low = SymbolicaCuts::new(0.).unwrap();
    let mut high = SymbolicaHighCuts::new(0.).unwrap();
    let mut provider = InvariantMaster::default();
    let shifts = [q([0.; 4]), q([0., 1., 0., 0.])];
    let masses = [c(1.); 2];
    let scales = [c(1.)];
    let matrix = [c(0.), c(-7.), c(-7.), c(0.)];
    let input = |matrix| AmplitudeInput {
        shifts: &shifts,
        masses_squared: &masses,
        scales_squared: &scales,
        sampling_scales: &[],
        s_matrix: Some(matrix),
    };
    amplitude
        .evaluate_batch_with_inputs(
            input(&matrix),
            &mut numerator,
            &mut low,
            &mut high,
            &mut provider,
        )
        .unwrap();
    assert_eq!(amplitude.statuses(), &[EvaluationStatus::Success]);
    assert_eq!(amplitude.results()[0][0], c(-7.));
    let saved = amplitude.results().to_vec();
    assert!(
        amplitude
            .evaluate_batch_with_inputs(
                input(&matrix[..3]),
                &mut numerator,
                &mut low,
                &mut high,
                &mut provider
            )
            .is_err()
    );
    assert_eq!(amplitude.results(), saved);
    assert_eq!(amplitude.statuses(), &[EvaluationStatus::Success]);
}

#[test]
fn stopping_after_pentagons_reports_failed_lanes_without_master_calls() {
    use ninjars::{BoxCoefficientMode, BoxMuMode, BoxReductionOptions, CutOutcome};
    let mut numerator = PolynomialNumerator::new(parse!("1"), variables(), vec![], 0.).unwrap();
    numerator.set_parameter_batch(&[], 2).unwrap();
    let mut amplitude = SymbolicaAmplitudeKernels::new(5, 0, 0.)
        .unwrap()
        .with_box_options(BoxReductionOptions {
            coefficients: BoxCoefficientMode::Full,
            mu_extraction: BoxMuMode::Expansion,
        })
        .into_amplitude(2)
        .unwrap();
    amplitude.set_cut_stop(5);
    let mut low = SymbolicaCuts::new(0.).unwrap();
    let mut high = SymbolicaHighCuts::new(0.).unwrap();
    let mut provider = InvariantMaster::default();
    let shifts: Vec<_> = [
        [0., 0., 0., 0.],
        [2., 1., 0., 0.],
        [3., 0., 1., 0.],
        [4., 0., 0., 1.],
        [1., 2., 3., 4.],
    ]
    .into_iter()
    .map(q)
    .chain((0..5).map(|i| q([i as f64, 0., 0., 0.])))
    .collect();
    let masses: Vec<_> = (0..2).flat_map(|_| (1..=5).map(|i| c(i as f64))).collect();
    amplitude
        .evaluate_batch_with_inputs(
            AmplitudeInput {
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: &[c(1.); 2],
                sampling_scales: &[],
                s_matrix: None,
            },
            &mut numerator,
            &mut low,
            &mut high,
            &mut provider,
        )
        .unwrap();
    assert_eq!(amplitude.statuses()[0], EvaluationStatus::Success);
    assert!(matches!(
        amplitude.statuses()[1],
        EvaluationStatus::CutFailed { cut_index: 0, .. }
    ));
    assert!(matches!(amplitude.pentagons()[0], CutOutcome::Stable(_)));
    assert!(matches!(amplitude.pentagons()[1], CutOutcome::Unstable(_)));
    assert_eq!(amplitude.results(), &[[c(0.); 5]; 2]);
    assert!(provider.families.is_empty());
    amplitude.set_cut_stop(6);
    amplitude
        .evaluate_batch_with_low_cuts(
            AmplitudeInput {
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: &[c(1.); 2],
                sampling_scales: &[],
                s_matrix: None,
            },
            &mut numerator,
            &mut low,
            &mut provider,
        )
        .unwrap();
    assert_eq!(amplitude.statuses(), &[EvaluationStatus::Success; 2]);
    assert!(amplitude.pentagons().is_empty());
    amplitude.reset_batch(0);
    numerator.set_parameter_batch(&[], 0).unwrap();
    amplitude
        .evaluate_batch(&[], &[], &[], &mut numerator, &mut low, &mut provider)
        .unwrap();
    assert!(amplitude.statuses().is_empty());
}
