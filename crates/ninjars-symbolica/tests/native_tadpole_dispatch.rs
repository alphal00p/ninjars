#![cfg(feature = "native-kernels")]

use ninjars::{Complex, DoubleFloat, Float, FloatLike, Real, SingleFloat};
use ninjars_symbolica::{KernelBackend, TadpoleSourceProjection};
use oneloop::NativeFloat;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
use symbolica::{
    domains::{
        dual::{DualNumber, HyperDual},
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
fn scalar<T: NativeFloat + Into<Float>>(prototype: T, digits: i32)
where
    Complex<T>: Real + EvaluationDomain,
{
    for source in [3, 2] {
        let plan = TadpoleSourceProjection::new(source).unwrap();
        let columns = plan.input_columns();
        let outputs = plan.output_columns();
        let mut interpreted = plan.compile(&prototype).unwrap();
        let mut native = plan.compile_native(&prototype).unwrap();
        assert_eq!(interpreted.backend(), KernelBackend::SymbolicaInterpreter);
        assert_eq!(native.backend(), KernelBackend::NativeRust);
        for rows in [0, 1, 8, 32, 128, 1024, 3, 31, 127, 0, 1] {
            let mut args: Vec<_> = (0..rows * columns).map(|i| input(&prototype, i)).collect();
            if rows % 2 == 1 {
                args.reverse();
            }
            let mut expected = vec![input(&prototype, 0); rows * outputs];
            let mut actual = expected.clone();
            interpreted
                .evaluate_batch(&args, &mut expected, rows)
                .unwrap();
            native.evaluate_batch(&args, &mut actual, rows).unwrap();
            for (a, b) in actual.iter().zip(&expected) {
                near(a, b, digits);
            }
            if rows > 0 {
                let before = actual.clone();
                assert!(
                    native
                        .evaluate_batch(&args[..args.len() - 1], &mut actual, rows)
                        .is_err()
                );
                assert_eq!(actual, before);
                assert!(
                    native
                        .evaluate_batch(&args, &mut actual, usize::MAX)
                        .is_err()
                );
                assert_eq!(actual, before);
                let len = actual.len();
                assert!(
                    native
                        .evaluate_batch(&args, &mut actual[..len - 1], rows)
                        .is_err()
                );
                assert_eq!(actual, before);
            }
        }
    }
}
#[test]
fn scalar_precision_batches_and_transactionality() {
    scalar(0., 12);
    scalar(DoubleFloat::default(), 27);
    scalar(Float::new(256), 65);
}

fn dual<T: NativeFloat + Into<Float>, D: Real + DualNumber<Complex<T>>>(
    prototype: T,
    structure: D,
    digits: i32,
) where
    Complex<T>: Real + EvaluationDomain,
{
    for source in [3, 2] {
        let plan = TadpoleSourceProjection::new(source).unwrap();
        let mut interpreted = plan.compile_dual(&prototype, &structure).unwrap();
        let mut native = plan.compile_dual_native(&prototype, &structure).unwrap();
        assert_eq!(native.backend(), KernelBackend::NativeRust);
        for rows in [0, 1, 8, 32, 128, 3, 31, 0, 1] {
            let args: Vec<_> = (0..rows * plan.input_columns())
                .map(|i| {
                    let mut d = structure.zero();
                    for (j, c) in d.values_mut().iter_mut().enumerate() {
                        *c = input(&prototype, i * 5 + j);
                    }
                    d
                })
                .collect();
            let mut expected = vec![structure.zero(); rows * plan.output_columns()];
            let mut actual = expected.clone();
            interpreted
                .evaluate_batch(&args, &mut expected, rows)
                .unwrap();
            native.evaluate_batch(&args, &mut actual, rows).unwrap();
            for (a, b) in actual.iter().zip(&expected) {
                assert_eq!(a.get_shape(), b.get_shape());
                for (a, b) in a.values().iter().zip(b.values()) {
                    near(a, b, digits);
                }
            }
        }
    }
}
#[test]
fn static_and_dynamic_higher_mixed_components() {
    for dynamic in [true, false] {
        if dynamic {
            dual(
                0.,
                HyperDual::from_values(shape(), vec![Complex::new(0., 0.); 5]),
                12,
            );
        } else {
            dual(
                0.,
                static_dual::NativeDual {
                    values: [Complex::new(0., 0.); 5],
                },
                12,
            );
        }
    }
    let p = DoubleFloat::default();
    let z = Complex::new(p.zero(), p.zero());
    dual(p, HyperDual::from_values(shape(), vec![z; 5]), 27);
    let p = Float::new(256);
    let z = Complex::new(p.zero(), p.zero());
    dual(p, HyperDual::from_values(shape(), vec![z; 5]), 65);
}
#[test]
fn exact_ordered_shape_dispatch_and_malformed_dual_transactionality() {
    let plan = TadpoleSourceProjection::new(2).unwrap();
    let zero = Complex::new(0., 0.);
    for unsupported in [
        vec![vec![0], vec![1], vec![2], vec![3], vec![4]],
        vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![2, 0], vec![1, 1]],
    ] {
        let d = HyperDual::from_values(unsupported, vec![zero; 5]);
        let mut fallback = plan.compile_dual_native(&0., &d).unwrap();
        assert_eq!(fallback.backend(), KernelBackend::SymbolicaInterpreter);
        let mut interpreter = plan.compile_dual(&0., &d).unwrap();
        let args: Vec<_> = (0..plan.input_columns())
            .map(|i| {
                let mut value = d.zero();
                for (j, component) in value.values_mut().iter_mut().enumerate() {
                    *component = input(&0., i * 5 + j);
                }
                value
            })
            .collect();
        let mut a = vec![d.zero(); plan.output_columns()];
        let mut b = a.clone();
        fallback.evaluate_batch(&args, &mut a, 1).unwrap();
        interpreter.evaluate_batch(&args, &mut b, 1).unwrap();
        for (a, b) in a.iter().zip(&b) {
            assert_eq!(a.values, b.values);
        }
    }
    let d = HyperDual::from_values(shape(), vec![zero; 5]);
    let mut native = plan.compile_dual_native(&0., &d).unwrap();
    let mut args = vec![d.from_i64(2); plan.input_columns()];
    let mut out = vec![d.from_i64(7); plan.output_columns()];
    let before: Vec<_> = out.iter().map(|v| v.values.clone()).collect();
    args[4].values.pop();
    assert!(native.evaluate_batch(&args, &mut out, 1).is_err());
    assert_eq!(
        before,
        out.iter().map(|v| v.values.clone()).collect::<Vec<_>>()
    );
    args[4] = HyperDual::from_values(
        vec![vec![0], vec![1], vec![2], vec![3], vec![4]],
        vec![zero; 5],
    );
    assert!(native.evaluate_batch(&args, &mut out, 1).is_err());
    assert_eq!(
        before,
        out.iter().map(|v| v.values.clone()).collect::<Vec<_>>()
    );
}
#[test]
fn native_warm_f64_storage_and_width_one_do_not_allocate() {
    for source in [3, 2] {
        let plan = TadpoleSourceProjection::new(source).unwrap();
        let mut native = plan.compile_native(&0.).unwrap();
        for rows in [1, 1024, 1, 8, 32, 128, 3, 31, 127, 0, 1] {
            let args: Vec<_> = (0..rows * plan.input_columns())
                .map(|i| input(&0., i))
                .collect();
            let mut output = vec![Complex::new(0., 0.); rows * plan.output_columns()];
            if rows != 1 {
                native.evaluate_batch(&args, &mut output, rows).unwrap();
            }
            let count = allocations(|| native.evaluate_batch(&args, &mut output, rows).unwrap());
            assert_eq!(count, 0, "source{source},rows{rows}");
        }
    }
}
#[test]
fn float_inputs_keep_their_own_precision() {
    let plan = TadpoleSourceProjection::new(2).unwrap();
    let working = Float::new(256);
    let mut native = plan.compile_native(&working).unwrap();
    let mut interpreted = plan.compile(&working).unwrap();
    for (bits, real_only) in [
        (53, false),
        (128, false),
        (512, false),
        (53, true),
        (128, true),
    ] {
        let args: Vec<_> = (0..3 * plan.input_columns())
            .map(|i| {
                let mut value = input(&Float::new(bits), i);
                if real_only {
                    value.im = value.im.zero();
                }
                value
            })
            .collect();
        let mut expected =
            vec![Complex::new(working.zero(), working.zero()); 3 * plan.output_columns()];
        let mut actual = expected.clone();
        interpreted.evaluate_batch(&args, &mut expected, 3).unwrap();
        native.evaluate_batch(&args, &mut actual, 3).unwrap();
        for (a, b) in actual.iter().zip(&expected) {
            near(
                a,
                b,
                if bits == 53 {
                    12
                } else if bits == 128 {
                    32
                } else {
                    65
                },
            );
            assert_eq!(a.re.get_precision(), b.re.get_precision());
            assert_eq!(a.im.get_precision(), b.im.get_precision());
        }
    }
}
#[test]
fn warm_dual_native_dispatch_adds_no_allocations_to_existing_shape_validation() {
    let d = static_dual::NativeDual {
        values: [Complex::new(0., 0.); 5],
    };
    for source in [3, 2] {
        let plan = TadpoleSourceProjection::new(source).unwrap();
        let mut native = plan.compile_dual_native(&0., &d).unwrap();
        let mut interpreted = plan.compile_dual(&0., &d).unwrap();
        for rows in [1, 32, 8, 1] {
            let args: Vec<_> = (0..rows * plan.input_columns())
                .map(|i| {
                    let mut v = d.zero();
                    for (j, c) in v.values_mut().iter_mut().enumerate() {
                        *c = input(&0., i * 5 + j);
                    }
                    v
                })
                .collect();
            let mut output = vec![d.zero(); rows * plan.output_columns()];
            native.evaluate_batch(&args, &mut output, rows).unwrap();
            interpreted
                .evaluate_batch(&args, &mut output, rows)
                .unwrap();
            let native_allocations =
                allocations(|| native.evaluate_batch(&args, &mut output, rows).unwrap());
            let interpreted_allocations = allocations(|| {
                interpreted
                    .evaluate_batch(&args, &mut output, rows)
                    .unwrap()
            });
            assert!(native_allocations <= interpreted_allocations);
            println!(
                "source={source},rows={rows},native_allocations={native_allocations},interpreted_allocations={interpreted_allocations}"
            );
        }
    }
}

#[test]
fn native_feature_preserves_send_sync_for_standard_evaluators() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<ninjars_symbolica::TadpolePlanEvaluator<f64>>();
    send_sync::<
        ninjars_symbolica::DualTadpolePlanEvaluator<f64, static_dual::NativeDual<Complex<f64>>>,
    >();
    send_sync::<ninjars_symbolica::SymbolicaAmplitudeKernels<f64>>();
}

#[test]
fn malformed_dual_prototypes_are_rejected_before_output_reconstruction() {
    let plan = TadpoleSourceProjection::new(2).unwrap();
    for length in [0, 4, 6] {
        let mut dual = HyperDual::from_values(shape(), vec![Complex::new(0., 0.); 5]);
        dual.values.resize(length, Complex::new(0., 0.));
        assert!(plan.compile_dual(&0., &dual).is_err());
        assert!(plan.compile_dual_native(&0., &dual).is_err());
        assert!(ninjars_symbolica::SymbolicaDualAmplitudeKernels::new(2, 2, 0., &dual).is_err());
    }
}

#[test]
fn geometry_probe_and_denominator_only_selection_remain_independent() {
    use ninjars_symbolica::{KernelFamily, SymbolicaDualAmplitudeKernels, TadpoleGeometryProbe};
    let d = HyperDual::from_values(shape(), vec![Complex::new(0., 0.); 5]);
    let probe = TadpoleGeometryProbe::new();
    assert_eq!(
        probe.compile(&0.).unwrap().backend(),
        KernelBackend::SymbolicaInterpreter
    );
    assert_eq!(
        probe.compile_dual(&0., &d).unwrap().backend(),
        KernelBackend::SymbolicaInterpreter
    );
    let selected = SymbolicaDualAmplitudeKernels::new(3, 3, 0., &d)
        .unwrap()
        .with_native_kernels()
        .with_native_denominators();
    for selection in selected.kernel_backends() {
        assert_eq!(
            selection.backend,
            if matches!(
                selection.family,
                KernelFamily::T2Denominators | KernelFamily::T3Denominators
            ) {
                KernelBackend::NativeRust
            } else {
                KernelBackend::SymbolicaInterpreter
            }
        );
    }
}
