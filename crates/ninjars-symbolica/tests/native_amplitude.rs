#![cfg(feature = "native-kernels")]

use ninjars::{
    Complex, DoubleFloat, EvaluationStatus, Float, FloatLike, FourMomentum, Numerator, Real,
    SingleFloat,
};
use ninjars_symbolica::{
    DualPolynomialNumerator, KernelBackend, KernelFamily, PolynomialNumerator,
    SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaDualAmplitudeKernels, SymbolicaDualCuts,
    SymbolicaDualMasters, SymbolicaHighCuts, SymbolicaMasters,
};
use oneloop::NativeFloat;
use symbolica::{
    atom::Atom,
    domains::{
        dual::{DualNumberStructure, HyperDual},
        rational::Rational,
    },
    evaluate::EvaluationDomain,
    parse,
};
fn vars() -> [Atom; 5] {
    [
        parse!("q0"),
        parse!("q1"),
        parse!("q2"),
        parse!("q3"),
        parse!("mu2"),
    ]
}
fn coverage(report: &[ninjars_symbolica::KernelBackendSelection], native: bool, corrections: bool) {
    assert_eq!(report.len(), 7);
    for entry in report {
        let selected = (native
            && matches!(
                entry.family,
                KernelFamily::T2Denominators | KernelFamily::T3Denominators
            ))
            || (corrections
                && matches!(
                    entry.family,
                    KernelFamily::BubbleSourceProjection | KernelFamily::TriangleBubbleCorrection
                ));
        assert_eq!(
            entry.backend,
            if selected {
                KernelBackend::NativeRust
            } else {
                KernelBackend::SymbolicaInterpreter
            }
        );
    }
}
fn scalar<T: NativeFloat + Into<Float>>(p: T, n: usize, digits: u64, catalog: bool)
where
    Complex<T>: Real + EvaluationDomain,
{
    let expression = match n {
        1 => parse!("a*q0^2"),
        2 => parse!("a*(q0-q1)^3"),
        3 => parse!("a*(q0-q1)^3+mu2"),
        _ => parse!("a*(q0-q1)^4+mu2^2"),
    };
    scalar_expression(p, n, digits, catalog, expression);
}
fn scalar_expression<T: NativeFloat + Into<Float>>(
    p: T,
    n: usize,
    digits: u64,
    catalog: bool,
    expression: Atom,
) where
    Complex<T>: Real + EvaluationDomain,
{
    let c = |x| Complex::new(p.from_i64(x), p.zero());
    let zero = c(0);
    let source =
        PolynomialNumerator::new(expression, vars(), vec![parse!("a")], p.clone()).unwrap();
    let rank = source.rank();
    let mut numerator = source;
    numerator
        .set_parameter_batch(&[c(1), c(2), c(3)], 3)
        .unwrap();
    let default = SymbolicaAmplitudeKernels::new(n, rank, p.clone()).unwrap();
    coverage(&default.kernel_backends(), false, false);
    let selected = SymbolicaAmplitudeKernels::new(n, rank, p.clone())
        .unwrap()
        .with_native_denominators();
    let selected = if catalog {
        selected.with_native_kernels()
    } else {
        selected
    };
    coverage(&selected.kernel_backends(), true, catalog);
    let mut interpreted = default.into_amplitude(3).unwrap();
    let mut native = selected.into_amplitude(3).unwrap();
    let routing = if n <= 4 {
        vec![[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    } else {
        vec![
            [0, 0, 0, 0],
            [2, 1, 0, 0],
            [4, 0, 1, 0],
            [7, 0, 0, 1],
            [11, 1, 2, 1],
            [16, 2, -1, 2],
        ]
    };
    let shifts: Vec<_> = (0..3)
        .flat_map(|_| routing[..n].iter().map(|a| FourMomentum::from(a.map(c))))
        .collect();
    let mut masses = vec![c(1); 3 * n];
    let scales = vec![c(1); 3];
    let mut cuts = SymbolicaCuts::new(p.clone()).unwrap();
    let mut high = SymbolicaHighCuts::new(p.clone()).unwrap();
    let mut masters = SymbolicaMasters::new(p.clone());
    let tolerance = Float::new(384).from_rational(&Rational::from((1, 10)).pow(digits));
    for run in 0..2 {
        if run == 1 {
            masses[n].re = p.from_i64(-1).sqrt();
        }
        for amplitude in [&mut interpreted, &mut native] {
            amplitude
                .evaluate_batch_with_high_cuts(
                    &shifts,
                    &masses,
                    &scales,
                    &scales,
                    &mut numerator,
                    &mut cuts,
                    &mut high,
                    &mut masters,
                )
                .unwrap();
        }
        assert_eq!(interpreted.statuses(), native.statuses());
        assert_eq!(native.statuses()[0], EvaluationStatus::Success);
        if run == 1 {
            assert_ne!(native.statuses()[1], EvaluationStatus::Success);
        }
        for (a, b) in native
            .results()
            .iter()
            .flatten()
            .zip(interpreted.results().iter().flatten())
        {
            for (a, b) in [(&a.re, &b.re), (&a.im, &b.im)] {
                let a: Float = a.clone().into();
                let b: Float = b.clone().into();
                assert!(
                    (a - b.clone()).norm()
                        <= tolerance.clone() * (b.norm() + Float::new(384).one())
                );
            }
        }
    }
    let before = native.results().to_vec();
    let statuses = native.statuses().to_vec();
    assert!(
        native
            .evaluate_batch_with_high_cuts(
                &shifts,
                &masses,
                &scales[..2],
                &scales,
                &mut numerator,
                &mut cuts,
                &mut high,
                &mut masters
            )
            .is_err()
    );
    assert_eq!(native.results(), before);
    assert_eq!(native.statuses(), statuses);
    native.reset();
    assert!(native.results().iter().flatten().all(|v| *v == zero));
}
#[test]
fn partial_native_amplitudes_keep_results_failure_isolation_and_reset() {
    for n in 1..=4 {
        scalar(0., n, 9, false);
    }
    scalar(DoubleFloat::default(), 2, 25, false);
    scalar(Float::new(192), 2, 45, false);
}

#[test]
fn dual_amplitude_all_components_and_unknown_shape_fallback() {
    for shape in [
        vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]],
        vec![vec![0], vec![1], vec![2], vec![3], vec![4]],
    ] {
        let d = HyperDual::from_values(shape.clone(), vec![Complex::new(0., 0.); 5]);
        let c = |v: i64| d.from_i64(v);
        let source =
            PolynomialNumerator::new(parse!("a*(q0-q1)^3"), vars(), vec![parse!("a")], 0.).unwrap();
        let mut numerator = DualPolynomialNumerator::new(source, d.clone()).unwrap();
        let mut a = c(2);
        a.values[1] = Complex::new(1., 2.);
        a.values[2] = Complex::new(2., -1.);
        numerator.set_parameter_batch(&[a.clone(), a], 2).unwrap();
        let default = SymbolicaDualAmplitudeKernels::new(2, 3, 0., &d).unwrap();
        let selected = SymbolicaDualAmplitudeKernels::new(2, 3, 0., &d)
            .unwrap()
            .with_native_denominators();
        let selected = selected.with_native_kernels();
        coverage(&selected.kernel_backends(), shape[0].len() == 2, false);
        let mut interpreted = default.into_amplitude(2).unwrap();
        let mut native = selected.into_amplitude(2).unwrap();
        let shifts: Vec<_> = (0..2)
            .flat_map(|_| [[0, 0, 0, 0], [0, 1, 0, 0]].map(|q| FourMomentum::from(q.map(c))))
            .collect();
        let mut masses = vec![c(1); 4];
        masses[0].values[1] = Complex::new(1. / 8., 0.);
        masses[1].values[2] = Complex::new(1. / 4., 0.);
        let scales = vec![c(1); 2];
        let mut geometry = SymbolicaDualCuts::new(0., d.clone()).unwrap();
        let mut masters = SymbolicaDualMasters::new(0., &d);
        for run in 0..2 {
            if run == 1 {
                masses[2].values[4].re = f64::NAN;
            }
            for amplitude in [&mut interpreted, &mut native] {
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
            }
            assert_eq!(native.statuses(), interpreted.statuses());
            assert_eq!(native.statuses()[0], EvaluationStatus::Success);
            if run == 1 {
                assert_ne!(native.statuses()[1], EvaluationStatus::Success);
            }
            for (a, b) in native
                .results()
                .iter()
                .flatten()
                .zip(interpreted.results().iter().flatten())
            {
                assert_eq!(a.get_shape(), b.get_shape());
                for (a, b) in a.values.iter().zip(&b.values) {
                    assert!((a - b).norm().re < 1e-9 * (1. + b.norm().re), "{a} != {b}");
                }
            }
        }
    }
}

#[test]
fn growing_catalog_covers_lower_quotient_degrees_and_complex_precisions() {
    for n in 1..=6 {
        scalar(0., n, 9, true);
    }
    scalar_expression(0., 3, 9, true, parse!("a*(q0-q1)^4+mu2^2"));
    scalar(DoubleFloat::default(), 3, 25, true);
    scalar(Float::new(192), 3, 45, true);
    let selection = SymbolicaAmplitudeKernels::new(3, 3, 0.)
        .unwrap()
        .with_native_kernels()
        .with_native_denominators();
    coverage(&selection.kernel_backends(), true, false);
}
