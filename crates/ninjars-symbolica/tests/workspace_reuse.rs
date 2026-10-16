//! Stateful controls for reused reducer storage; all arithmetic stays in the adapters.
use ninjars::{
    BoxCoefficientMode, BoxMuMode, BoxReductionOptions, Complex, Error, EvaluationStatus,
    ExpansionKind, ExpansionRequest, FourMomentum, Laurent, MasterProvider, MasterRequest,
    MasterStatus, Numerator, NumeratorRequest, Result,
};
use ninjars_symbolica::{
    PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaHighCuts,
};
use symbolica::{atom::Atom, parse};

type C = Complex<f64>;
fn c(x: f64) -> C {
    C::new(x, 0.)
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
// This storage/transactionality test needs deterministic master values, not a
// second expensive master backend. Independent physics controls are elsewhere.
struct FixedMasters {
    fail: bool,
}
impl MasterProvider<C> for FixedMasters {
    fn evaluate(&mut self, request: MasterRequest<'_, C>, output: &mut [Laurent<C>]) -> Result<()> {
        request.validate(output.len())?;
        output.fill([c(1.), c(2.), c(3.)]);
        if self.fail {
            Err(Error::Unsupported("injected master failure".into()))
        } else {
            Ok(())
        }
    }
    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, C>,
        output: &mut [Laurent<C>],
        statuses: &mut [MasterStatus],
    ) -> Result<()> {
        self.evaluate(request, output)?;
        statuses.fill(MasterStatus::Success);
        Ok(())
    }
}
struct FaultNumerator {
    inner: PolynomialNumerator<f64>,
    calls: usize,
    fail_at: Option<usize>,
}
impl FaultNumerator {
    fn complete(&mut self, output: &mut [C]) -> Result<()> {
        self.calls += 1;
        if self.fail_at == Some(self.calls) {
            output.fill(c(f64::NAN));
            Err(Error::Unsupported(
                "injected partially written callback".into(),
            ))
        } else {
            Ok(())
        }
    }
}
impl Numerator<C> for FaultNumerator {
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn evaluate_batch(
        &mut self,
        requests: &[NumeratorRequest<'_, C>],
        output: &mut [C],
    ) -> Result<()> {
        self.inner.evaluate_batch(requests, output)?;
        self.complete(output)
    }
    fn expansion_batch(
        &mut self,
        kind: &ExpansionKind,
        requests: &[ExpansionRequest<'_, C>],
        output: &mut [C],
    ) -> Result<()> {
        self.inner.expansion_batch(kind, requests, output)?;
        self.complete(output)
    }
}
fn near(actual: &[C; 5], expected: &[C; 5], factor: f64) {
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a.re - e.re * factor).hypot(a.im - e.im * factor)
                <= 1e-9 * (1. + e.re.abs() * factor.abs() + e.im.abs() * factor.abs()),
            "{a:?} != {e:?} * {factor}"
        );
    }
}

#[test]
fn growing_shrinking_empty_and_failed_batches_reuse_all_reducer_stages() {
    let n = 5;
    let routing = [
        [0., 0., 0., 0.],
        [2., 1., 0., 0.],
        [3., 0., 1., 0.],
        [4., 0., 0., 1.],
        [5., 2., 1., -1.],
    ];
    let mut low = SymbolicaCuts::new(0.).unwrap();
    let mut high = SymbolicaHighCuts::new(0.).unwrap();
    let mut masters = FixedMasters { fail: false };
    for mu_extraction in [BoxMuMode::Expansion, BoxMuMode::Sampling] {
        let inner = PolynomialNumerator::new(
            parse!("a*((q0-q1)^6+mu2^2*q0^2)"),
            vars(),
            vec![parse!("a")],
            0.,
        )
        .unwrap();
        let mut numerator = FaultNumerator {
            inner,
            calls: 0,
            fail_at: None,
        };
        let mut amplitude = SymbolicaAmplitudeKernels::new(n, 6, 0.)
            .unwrap()
            .with_box_options(BoxReductionOptions {
                coefficients: BoxCoefficientMode::Full,
                mu_extraction,
            })
            .into_amplitude(1)
            .unwrap();
        let mut baseline = [c(0.); 5];
        for (step, lanes) in [1, 8, 1, 0, 3, 8].into_iter().enumerate() {
            let shifts: Vec<_> = (0..lanes)
                .flat_map(|_| routing.map(|p| FourMomentum::from(p.map(c))))
                .collect();
            let mut masses = vec![c(100.); n * lanes];
            let scales = vec![c(1.); lanes];
            let sampling = vec![c(100.); lanes];
            let parameters: Vec<_> = (0..lanes).map(|lane| c(1. + (lane % 2) as f64)).collect();
            numerator
                .inner
                .set_parameter_batch(&parameters, lanes)
                .unwrap();
            amplitude.reset_batch(lanes);
            numerator.calls = 0;
            let evaluate =
                |amplitude: &mut ninjars::Amplitude<C, SymbolicaAmplitudeKernels<f64>>,
                 numerator: &mut FaultNumerator,
                 low: &mut SymbolicaCuts<f64>,
                 high: &mut SymbolicaHighCuts<f64>,
                 masters: &mut FixedMasters,
                 masses: &[C]| {
                    amplitude.evaluate_batch_with_high_cuts(
                        &shifts, masses, &scales, &sampling, numerator, low, high, masters,
                    )
                };
            evaluate(
                &mut amplitude,
                &mut numerator,
                &mut low,
                &mut high,
                &mut masters,
                &masses,
            )
            .unwrap();
            assert!(
                amplitude
                    .statuses()
                    .iter()
                    .all(|s| *s == EvaluationStatus::Success)
            );
            if step == 0 {
                baseline = amplitude.results()[0];
            }
            for (lane, parameter) in parameters.iter().enumerate() {
                near(&amplitude.results()[lane], &baseline, parameter.re);
            }
            if lanes == 0 {
                continue;
            }
            // Every callback position and the late master boundary may fail after
            // writing private scratch; previously published output must survive.
            if step == 1 {
                let calls = numerator.calls;
                for fail_at in 1..=calls {
                    numerator.calls = 0;
                    numerator.fail_at = Some(fail_at);
                    let before = amplitude.results().to_vec();
                    let statuses = amplitude.statuses().to_vec();
                    assert!(
                        evaluate(
                            &mut amplitude,
                            &mut numerator,
                            &mut low,
                            &mut high,
                            &mut masters,
                            &masses
                        )
                        .is_err()
                    );
                    assert_eq!(amplitude.results(), before);
                    assert_eq!(amplitude.statuses(), statuses);
                }
                numerator.fail_at = None;
                masters.fail = true;
                let before = amplitude.results().to_vec();
                let statuses = amplitude.statuses().to_vec();
                assert!(
                    evaluate(
                        &mut amplitude,
                        &mut numerator,
                        &mut low,
                        &mut high,
                        &mut masters,
                        &masses
                    )
                    .is_err()
                );
                assert_eq!(amplitude.results(), before);
                assert_eq!(amplitude.statuses(), statuses);
                masters.fail = false;
            }
            // A numerical failure affects only its lane; a later successful call
            // must replace stale failure flags while retaining accumulated values.
            masses[0] = c(f64::NAN);
            evaluate(
                &mut amplitude,
                &mut numerator,
                &mut low,
                &mut high,
                &mut masters,
                &masses,
            )
            .unwrap();
            assert_ne!(amplitude.statuses()[0], EvaluationStatus::Success);
            near(&amplitude.results()[0], &baseline, 1.);
            for (lane, parameter) in parameters.iter().enumerate().skip(1) {
                near(&amplitude.results()[lane], &baseline, 2. * parameter.re);
            }
            masses[0] = c(100.);
            evaluate(
                &mut amplitude,
                &mut numerator,
                &mut low,
                &mut high,
                &mut masters,
                &masses,
            )
            .unwrap();
            assert!(
                amplitude
                    .statuses()
                    .iter()
                    .all(|s| *s == EvaluationStatus::Success)
            );
            for (lane, parameter) in parameters.iter().enumerate() {
                near(
                    &amplitude.results()[lane],
                    &baseline,
                    parameter.re * if lane == 0 { 2. } else { 3. },
                );
            }
        }
    }
}

#[test]
fn reused_integration_values_preserve_float_precision_and_every_dual_component() {
    use ninjars::{Float, Real};
    use ninjars_symbolica::{
        DualPolynomialNumerator, SymbolicaDualAmplitudeKernels, SymbolicaDualCuts,
        SymbolicaDualMasters,
    };
    use symbolica::domains::{dual::HyperDual, float::FloatLike};
    let shape = vec![
        vec![0, 0],
        vec![1, 0],
        vec![0, 1],
        vec![2, 0],
        vec![1, 1],
        vec![2, 1],
    ];
    let component = |bits, value| Complex::new(Float::with_val(bits, value), Float::new(3));
    let dual = |bits, values: [i64; 6]| {
        HyperDual::from_values(shape.clone(), values.map(|v| component(bits, v)).to_vec())
    };
    let prototype = dual(192, [0; 6]);
    let mut reused = SymbolicaDualAmplitudeKernels::new(1, 0, Float::new(192), &prototype)
        .unwrap()
        .into_amplitude(0)
        .unwrap();
    let mut low = SymbolicaDualCuts::new(Float::new(192), prototype.clone()).unwrap();
    let mut masters = SymbolicaDualMasters::new(Float::new(192), &prototype);
    let source =
        PolynomialNumerator::new(parse!("a"), vars(), vec![parse!("a")], Float::new(192)).unwrap();
    let mut numerator = DualPolynomialNumerator::new(source, prototype.clone()).unwrap();
    for lanes in [3, 1, 0, 2, 3] {
        let parameters: Vec<_> = (0..lanes)
            .map(|lane| dual([53, 192, 256][lane], [0, 0, 1, 0, 0, 0]))
            .collect();
        let masses: Vec<_> = (0..lanes)
            .map(|lane| dual([53, 192, 256][lane], [1, 1, 0, 0, 0, 0]))
            .collect();
        let scales = vec![dual(192, [1, 0, 0, 0, 0, 0]); lanes];
        let shifts =
            vec![FourMomentum::from(std::array::from_fn::<_, 4, _>(|_| prototype.clone())); lanes];
        numerator.set_parameter_batch(&parameters, lanes).unwrap();
        reused.reset_batch(lanes);
        reused
            .evaluate_batch(
                &shifts,
                &masses,
                &scales,
                &mut numerator,
                &mut low,
                &mut masters,
            )
            .unwrap();
        let mut fresh = SymbolicaDualAmplitudeKernels::new(1, 0, Float::new(192), &prototype)
            .unwrap()
            .into_amplitude(lanes)
            .unwrap();
        fresh
            .evaluate_batch(
                &shifts,
                &masses,
                &scales,
                &mut numerator,
                &mut low,
                &mut masters,
            )
            .unwrap();
        assert_eq!(reused.statuses(), fresh.statuses());
        assert!(
            reused
                .statuses()
                .iter()
                .all(|s| *s == EvaluationStatus::Success)
        );
        for (got, expected) in reused.results().iter().zip(fresh.results()) {
            for (got, expected) in got.iter().zip(expected) {
                for (got, expected) in got.values.iter().zip(&expected.values) {
                    assert_eq!(got.re, expected.re);
                    assert_eq!(got.im, expected.im);
                    assert_eq!(got.re.get_precision(), expected.re.get_precision());
                    assert_eq!(got.im.get_precision(), expected.im.get_precision());
                }
            }
            // Independent analytic y*A0(1+x;1) Taylor coefficient x²*y=-1/2.
            let expected = Float::with_val(192, -1) / Float::with_val(192, 2);
            assert!((got[0].values[5].re.clone() - expected).norm() < Float::with_val(192, 1e-14));
        }
    }
}
