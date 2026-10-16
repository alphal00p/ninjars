#![cfg(feature = "native-kernels")]

use ninjars::{Complex, DoubleFloat, Float, FloatLike, Real, SingleFloat};
use ninjars::{
    CutFailure, CutOutcome, CutProvider, FourMomentum, SampleId, TadpoleCut, TadpoleCutRequest,
};
use ninjars_symbolica::{KernelBackend, SymbolicaCuts, SymbolicaDualCuts, TadpoleGeometryProbe};
use oneloop::NativeFloat;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
use symbolica::{
    domains::{
        dual::{DualNumber, DualNumberStructure, HyperDual},
        rational::Rational,
    },
    evaluate::EvaluationDomain,
};

#[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
mod static_dual {
    symbolica::create_hyperdual_from_components!(
        NativeDual,
        [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1]]
    );
}

struct CountingAllocator;
thread_local! { static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) }; }
fn record() {
    let _ = ALLOCATIONS.try_with(|counter| {
        if let Some(n) = counter.get() {
            counter.set(Some(n + 1));
        }
    });
}
// Count only this thread and forward every allocation to the ordinary allocator.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        record();
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
fn allocations(f: impl FnOnce()) -> usize {
    ALLOCATIONS.with(|c| assert!(c.replace(Some(0)).is_none()));
    f();
    ALLOCATIONS.with(|c| c.replace(None).unwrap())
}
fn shape() -> Vec<Vec<usize>> {
    vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]]
}
fn input<T: NativeFloat>(p: &T, index: usize) -> Complex<T> {
    Complex::new(
        p.from_rational(&Rational::from(((index % 37) as i64 - 11, 17))),
        p.from_rational(&Rational::from(((index % 13) as i64 - 5, 19))),
    )
}
fn near<T: NativeFloat + Into<Float>>(a: &Complex<T>, b: &Complex<T>, digits: i32) {
    let p = Float::new(384);
    let tolerance = p.from_rational(&Rational::from((1, 10)).pow(digits as u64));
    for (a, b) in [(&a.re, &b.re), (&a.im, &b.im)] {
        let a: Float = a.clone().into();
        let b: Float = b.clone().into();
        let a = p.from_rational(&a.to_rational());
        let b = p.from_rational(&b.to_rational());
        assert!(
            (a.clone() - &b).norm() <= tolerance.clone() * (p.one() + b.norm()),
            "{a} != {b}"
        );
    }
}
fn components<V>(cut: &TadpoleCut<V>) -> Vec<&V> {
    [
        &cut.basis.e1,
        &cut.basis.e2,
        &cut.basis.e3,
        &cut.basis.e4,
        &cut.e2_effective,
        &cut.e4_effective,
        &cut.a,
    ]
    .into_iter()
    .flat_map(FourMomentum::components)
    .chain([&cut.basis.e12, &cut.beta])
    .collect()
}

fn compare<V: Real>(
    a: &CutOutcome<TadpoleCut<V>>,
    b: &CutOutcome<TadpoleCut<V>>,
    check: &impl Fn(&V, &V),
) {
    match (a, b) {
        (CutOutcome::Stable(a), CutOutcome::Stable(b)) => {
            for (a, b) in components(a).into_iter().zip(components(b)) {
                check(a, b);
            }
        }
        (CutOutcome::Unstable(a), CutOutcome::Unstable(b)) => assert_eq!(a, b),
        _ => panic!("geometry outcomes differ"),
    }
}

fn provider_batches<V: Real>(
    mut interpreted: impl CutProvider<V>,
    mut native: impl CutProvider<V>,
    value: impl Fn(usize) -> V,
    check: impl Fn(&V, &V),
) {
    let prototype = value(0);
    let shifts: Vec<_> = (0..128)
        .map(|row| FourMomentum::from(std::array::from_fn(|k| value(4 * row + k))))
        .collect();
    let masses: Vec<_> = (0..128).map(|row| value(7 * row + 3)).collect();
    let complements: Vec<Vec<_>> = shifts
        .iter()
        .enumerate()
        .map(|(row, shift)| match row % 13 {
            2 => vec![shift.clone()],
            5 => vec![FourMomentum::from([
                prototype.nan().unwrap(),
                prototype.zero(),
                prototype.zero(),
                prototype.zero(),
            ])],
            7 => vec![],
            _ => (0..3)
                .map(|j| {
                    FourMomentum::from(std::array::from_fn(|k| value(11 * row + 5 * j + k + 251)))
                })
                .collect(),
        })
        .collect();
    let mut charts = std::collections::BTreeSet::new();
    for rows in [0, 1, 8, 32, 128, 3, 31, 127, 0, 1] {
        let order: Vec<_> = if rows % 2 == 0 {
            (0..rows).rev().collect()
        } else {
            (0..rows).collect()
        };
        let requests: Vec<_> = order
            .iter()
            .map(|&row| TadpoleCutRequest {
                id: SampleId {
                    lane: row,
                    sample: 17,
                },
                shift: &shifts[row],
                mass_squared: &masses[row],
                complementary_shifts: &complements[row],
            })
            .collect();
        let sentinel = CutOutcome::Unstable(CutFailure::NonFiniteResult);
        let mut expected = vec![sentinel.clone(); rows];
        let mut actual = expected.clone();
        interpreted.tadpole_batch(&requests, &mut expected).unwrap();
        native.tadpole_batch(&requests, &mut actual).unwrap();
        for (&row, (a, b)) in order.iter().zip(actual.iter().zip(&expected)) {
            compare(a, b, &check);
            match row % 13 {
                2 => assert!(matches!(
                    a,
                    CutOutcome::Unstable(CutFailure::DegenerateReference)
                )),
                5 => assert!(matches!(
                    a,
                    CutOutcome::Unstable(CutFailure::NonFiniteInput)
                )),
                _ => {
                    if let CutOutcome::Stable(cut) = a {
                        charts.insert(format!("{:?}", cut.basis.e1));
                    } else {
                        panic!("unexpected failure at row {row}")
                    }
                }
            }
        }
        if rows > 0 {
            let saved = actual.clone();
            assert!(
                native
                    .tadpole_batch(&requests, &mut actual[..rows - 1])
                    .is_err()
            );
            for (a, b) in actual.iter().zip(&saved) {
                compare(a, b, &check);
            }
            // A single lane chosen after the mixed batch retains its own chart.
            let mut one = [sentinel];
            native.tadpole_batch(&requests[..1], &mut one).unwrap();
            compare(&one[0], &actual[0], &check);
        }
    }
    assert!(
        charts.len() > 1,
        "input suite must exercise lane-local reference choices"
    );
}

fn ordinary<T: NativeFloat + Into<Float>>(prototype: T, digits: i32)
where
    Complex<T>: Real + EvaluationDomain,
{
    let interpreted = SymbolicaCuts::new(prototype.clone()).unwrap();
    let native = SymbolicaCuts::new(prototype.clone())
        .unwrap()
        .with_native_kernels()
        .unwrap();
    assert_eq!(
        interpreted.tadpole_probe_backend(),
        KernelBackend::SymbolicaInterpreter
    );
    assert_eq!(native.tadpole_probe_backend(), KernelBackend::NativeRust);
    provider_batches(
        interpreted,
        native,
        |i| input(&prototype, i),
        |a, b| near(a, b, digits),
    );
}
#[test]
fn ordinary_geometry_batches_precision_charts_and_failures() {
    ordinary(0., 12);
    ordinary(DoubleFloat::default(), 27);
    ordinary(Float::new(256), 65);
}

fn derivatives<T: NativeFloat + Into<Float>, D: Real + DualNumber<Complex<T>>>(
    prototype: T,
    structure: D,
    digits: i32,
    native_expected: bool,
) where
    Complex<T>: Real + EvaluationDomain,
{
    let interpreted = SymbolicaDualCuts::new(prototype.clone(), structure.clone()).unwrap();
    let native = SymbolicaDualCuts::new(prototype.clone(), structure.clone())
        .unwrap()
        .with_native_kernels()
        .unwrap();
    assert_eq!(
        native.tadpole_probe_backend(),
        if native_expected {
            KernelBackend::NativeRust
        } else {
            KernelBackend::SymbolicaInterpreter
        }
    );
    let plan = TadpoleGeometryProbe::new();
    let mut direct = plan.compile_dual(&prototype, &structure).unwrap();
    let mut selected = plan.compile_dual_native(&prototype, &structure).unwrap();
    assert_eq!(
        selected.backend(),
        if native_expected {
            KernelBackend::NativeRust
        } else {
            KernelBackend::SymbolicaInterpreter
        }
    );
    let args: Vec<_> = (0..24)
        .map(|i| {
            let mut d = structure.zero();
            for (j, v) in d.values_mut().iter_mut().enumerate() {
                *v = input(&prototype, i + 7 * j);
            }
            d
        })
        .collect();
    let mut expected = vec![structure.zero(); 84];
    let mut actual = expected.clone();
    direct.evaluate_batch(&args, &mut expected, 3).unwrap();
    selected.evaluate_batch(&args, &mut actual, 3).unwrap();
    for (a, b) in actual.iter().zip(&expected) {
        for (a, b) in a.values().iter().zip(b.values()) {
            near(a, b, digits);
        }
    }
    provider_batches(
        interpreted,
        native,
        |i| {
            let mut d = structure.zero();
            for (j, v) in d.values_mut().iter_mut().enumerate() {
                *v = input(&prototype, i + j * 7);
            }
            d
        },
        |a, b| {
            assert_eq!(a.get_shape(), b.get_shape());
            assert_eq!(a.values().len(), b.values().len());
            for (a, b) in a.values().iter().zip(b.values()) {
                near(a, b, digits);
            }
        },
    );
}
#[test]
fn full_dual_geometry_and_unknown_ordered_shape_fallback() {
    derivatives(
        0.,
        HyperDual::from_values(shape(), vec![Complex::new(0., 0.); 5]),
        12,
        true,
    );
    derivatives(
        0.,
        static_dual::NativeDual {
            values: [Complex::new(0., 0.); 5],
        },
        12,
        true,
    );
    derivatives(
        DoubleFloat::default(),
        HyperDual::from_values(
            shape(),
            vec![Complex::new(DoubleFloat::default(), DoubleFloat::default()); 5],
        ),
        27,
        true,
    );
    derivatives(
        Float::new(256),
        HyperDual::from_values(
            shape(),
            vec![Complex::new(Float::new(256), Float::new(256)); 5],
        ),
        65,
        true,
    );
    for shape in [
        vec![vec![0], vec![1], vec![2], vec![3], vec![4]],
        vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1], vec![2, 0]],
    ] {
        derivatives(
            0.,
            HyperDual::from_values(shape, vec![Complex::new(0., 0.); 5]),
            12,
            false,
        );
    }
}

#[test]
fn native_probe_dimensions_precision_and_warm_allocations() {
    let plan = TadpoleGeometryProbe::new();
    let mut interpreted = plan.compile(&Float::new(256)).unwrap();
    let mut native = plan.compile_native(&Float::new(256)).unwrap();
    for bits in [53, 128, 512] {
        let p = Float::new(bits);
        let input: Vec<_> = (0..8 * 3).map(|i| input(&p, i)).collect();
        let mut expected = vec![Complex::new(p.zero(), p.zero()); 28 * 3];
        let mut actual = expected.clone();
        interpreted
            .evaluate_batch(&input, &mut expected, 3)
            .unwrap();
        native.evaluate_batch(&input, &mut actual, 3).unwrap();
        for (a, b) in actual.iter().zip(&expected) {
            near(a, b, 12);
            assert_eq!(a.re.get_precision(), b.re.get_precision());
            assert_eq!(a.im.get_precision(), b.im.get_precision());
        }
        let before = actual.clone();
        assert!(native.evaluate_batch(&input[..23], &mut actual, 3).is_err());
        assert_eq!(actual, before);
        assert!(
            native
                .evaluate_batch(&input, &mut actual, usize::MAX)
                .is_err()
        );
        assert_eq!(actual, before);
    }
    let mut provider = SymbolicaCuts::new(0.)
        .unwrap()
        .with_native_kernels()
        .unwrap();
    let origin = FourMomentum::from([Complex::new(0., 0.); 4]);
    let complement = [FourMomentum::from([
        Complex::new(1., 0.),
        Complex::new(2., 0.),
        Complex::new(3., 0.),
        Complex::new(4., 0.),
    ])];
    let mass = Complex::new(2., 0.);
    for rows in [1, 32, 8, 1] {
        let requests: Vec<_> = (0..rows)
            .map(|lane| TadpoleCutRequest {
                id: SampleId { lane, sample: 0 },
                shift: &origin,
                mass_squared: &mass,
                complementary_shifts: &complement,
            })
            .collect();
        let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); rows];
        provider.tadpole_batch(&requests, &mut output).unwrap();
        assert!(output.iter().all(|r| matches!(r, CutOutcome::Stable(_))));
        assert_eq!(
            allocations(|| provider.tadpole_batch(&requests, &mut output).unwrap()),
            0
        );
    }
}

#[test]
fn dual_shape_and_nonprimal_failures_remain_lane_local() {
    let d = HyperDual::from_values(shape(), vec![Complex::new(0., 0.); 5]);
    let mut provider = SymbolicaDualCuts::new(0., d.clone())
        .unwrap()
        .with_native_kernels()
        .unwrap();
    let origin = FourMomentum::from(std::array::from_fn(|_| d.zero()));
    let mass = d.from_i64(2);
    let good = FourMomentum::from([1, 2, 3, 4].map(|x| d.from_i64(x)));
    let mut short = good.clone();
    short.temporal.value.values.pop();
    let mut nonfinite = good.clone();
    nonfinite.temporal.value.values[4].im = f64::NAN;
    let mut reordered = good.clone();
    reordered.temporal.value = HyperDual::from_values(
        vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1], vec![2, 0]],
        vec![Complex::new(1., 0.); 5],
    );
    let complements = [[good.clone()], [short], [nonfinite], [reordered], [good]];
    let requests: Vec<_> = complements
        .iter()
        .enumerate()
        .map(|(lane, comp)| TadpoleCutRequest {
            id: SampleId { lane, sample: 0 },
            shift: &origin,
            mass_squared: &mass,
            complementary_shifts: comp,
        })
        .collect();
    let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); 5];
    provider.tadpole_batch(&requests, &mut output).unwrap();
    assert!(matches!(output[0], CutOutcome::Stable(_)));
    assert!(matches!(output[4], CutOutcome::Stable(_)));
    assert!(matches!(
        output[1],
        CutOutcome::Unstable(CutFailure::DerivativeShape)
    ));
    assert!(matches!(
        output[2],
        CutOutcome::Unstable(CutFailure::NonFiniteInput)
    ));
    assert!(matches!(
        output[3],
        CutOutcome::Unstable(CutFailure::DerivativeShape)
    ));
    let saved = output.clone();
    assert!(provider.tadpole_batch(&requests, &mut output[..4]).is_err());
    for (a, b) in output.iter().zip(&saved) {
        compare(a, b, &|a, b| {
            for (a, b) in a.values().iter().zip(b.values()) {
                near(a, b, 12);
            }
        });
    }
    for len in [0, 4, 6] {
        let mut malformed = d.clone();
        malformed.values.resize(len, Complex::new(0., 0.));
        assert!(SymbolicaDualCuts::new(0., malformed).is_err());
    }
    fn send_sync<T: Send + Sync>() {}
    send_sync::<SymbolicaCuts<f64>>();
    send_sync::<SymbolicaDualCuts<f64, HyperDual<Complex<f64>>>>();
}

#[test]
fn integrated_amplitudes_change_only_the_selected_geometry_provider() {
    use ninjars::{EvaluationStatus, Numerator};
    use ninjars_symbolica::{
        PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaHighCuts, SymbolicaMasters,
    };
    use symbolica::parse;
    let c = |x| Complex::new(x as f64, 0.);
    for n in [1, 2, 3, 5, 6] {
        let expression = if n == 1 {
            parse!("q0^2")
        } else if n < 4 {
            parse!("(q0-q1)^3")
        } else {
            parse!("(q0-q1)^4+mu2^2")
        };
        let mut numerator = PolynomialNumerator::new(
            expression,
            [
                parse!("q0"),
                parse!("q1"),
                parse!("q2"),
                parse!("q3"),
                parse!("mu2"),
            ],
            vec![],
            0.,
        )
        .unwrap();
        numerator.set_parameter_batch(&[], 3).unwrap();
        let rank = numerator.rank();
        let factory = || {
            SymbolicaAmplitudeKernels::new(n, rank, 0.)
                .unwrap()
                .with_native_kernels()
                .into_amplitude(3)
                .unwrap()
        };
        let mut a = factory();
        let mut b = factory();
        let routing = if n <= 3 {
            vec![[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
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
            .flat_map(|_| routing[..n].iter().map(|q| FourMomentum::from(q.map(c))))
            .collect();
        let mut masses = vec![c(1); 3 * n];
        let scales = vec![c(1); 3];
        let mut interpreted = SymbolicaCuts::new(0.).unwrap();
        let mut selected = SymbolicaCuts::new(0.)
            .unwrap()
            .with_native_kernels()
            .unwrap();
        let mut high = SymbolicaHighCuts::new(0.).unwrap();
        let mut masters = SymbolicaMasters::new(0.);
        for run in 0..2 {
            if run == 1 {
                masses[n].re = f64::NAN;
            }
            a.evaluate_batch_with_high_cuts(
                &shifts,
                &masses,
                &scales,
                &scales,
                &mut numerator,
                &mut interpreted,
                &mut high,
                &mut masters,
            )
            .unwrap();
            b.evaluate_batch_with_high_cuts(
                &shifts,
                &masses,
                &scales,
                &scales,
                &mut numerator,
                &mut selected,
                &mut high,
                &mut masters,
            )
            .unwrap();
            assert_eq!(a.statuses(), b.statuses());
            assert_eq!(a.statuses()[0], EvaluationStatus::Success);
            if run == 1 {
                assert_ne!(a.statuses()[1], EvaluationStatus::Success);
            }
            for (a, b) in a
                .results()
                .iter()
                .flatten()
                .zip(b.results().iter().flatten())
            {
                near(a, b, 9);
            }
        }
    }
}

#[test]
fn integrated_mixed_dual_bubble_changes_only_geometry() {
    use ninjars::EvaluationStatus;
    use ninjars_symbolica::{
        DualPolynomialNumerator, PolynomialNumerator, SymbolicaDualAmplitudeKernels,
        SymbolicaDualMasters,
    };
    use symbolica::parse;
    let d = HyperDual::from_values(shape(), vec![Complex::new(0., 0.); 5]);
    let c = |v| d.from_i64(v);
    let source = PolynomialNumerator::new(
        parse!("a*(q0-q1)^3"),
        [
            parse!("q0"),
            parse!("q1"),
            parse!("q2"),
            parse!("q3"),
            parse!("mu2"),
        ],
        vec![parse!("a")],
        0.,
    )
    .unwrap();
    let mut numerator = DualPolynomialNumerator::new(source, d.clone()).unwrap();
    let mut coefficient = c(2);
    coefficient.values[1] = Complex::new(1., 2.);
    coefficient.values[2] = Complex::new(2., -1.);
    numerator
        .set_parameter_batch(&[coefficient.clone(), coefficient], 2)
        .unwrap();
    let mut a = SymbolicaDualAmplitudeKernels::new(2, 3, 0., &d)
        .unwrap()
        .with_native_kernels()
        .into_amplitude(2)
        .unwrap();
    let mut b = SymbolicaDualAmplitudeKernels::new(2, 3, 0., &d)
        .unwrap()
        .with_native_kernels()
        .into_amplitude(2)
        .unwrap();
    let shifts: Vec<_> = (0..2)
        .flat_map(|_| [[0, 0, 0, 0], [0, 1, 0, 0]].map(|q| FourMomentum::from(q.map(c))))
        .collect();
    let mut masses = vec![c(1); 4];
    masses[0].values[1] = Complex::new(0.125, 0.);
    masses[1].values[2] = Complex::new(0.25, 0.);
    let scales = vec![c(1); 2];
    let mut interpreted = SymbolicaDualCuts::new(0., d.clone()).unwrap();
    let mut native = SymbolicaDualCuts::new(0., d.clone())
        .unwrap()
        .with_native_kernels()
        .unwrap();
    let mut masters = SymbolicaDualMasters::new(0., &d);
    for run in 0..2 {
        if run == 1 {
            masses[2].values[4].re = f64::NAN;
        }
        a.evaluate_batch(
            &shifts,
            &masses,
            &scales,
            &mut numerator,
            &mut interpreted,
            &mut masters,
        )
        .unwrap();
        b.evaluate_batch(
            &shifts,
            &masses,
            &scales,
            &mut numerator,
            &mut native,
            &mut masters,
        )
        .unwrap();
        assert_eq!(a.statuses(), b.statuses());
        assert_eq!(a.statuses()[0], EvaluationStatus::Success);
        if run == 1 {
            assert_ne!(a.statuses()[1], EvaluationStatus::Success);
        }
        for (a, b) in a
            .results()
            .iter()
            .flatten()
            .zip(b.results().iter().flatten())
        {
            assert_eq!(a.get_shape(), b.get_shape());
            for (a, b) in a.values().iter().zip(b.values()) {
                near(a, b, 9);
            }
        }
    }
}
