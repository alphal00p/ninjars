#![cfg(feature = "native-kernels")]

use ninjars::{Complex, DoubleFloat, Float, FloatLike, Real, SingleFloat};
use ninjars_symbolica::{
    BubbleSourceProjection, KernelBackend, LaurentDivision, TriangleBubbleSubtraction,
    rust_codegen::{EmitOptions, RustEmitter, RustKernelExport},
};
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
    for degree in [None, Some(0), Some(1), Some(2), Some(3)] {
        let source = BubbleSourceProjection::new();
        let correction = degree.map(|degree| {
            TriangleBubbleSubtraction::for_quotient(
                &LaurentDivision::new(5, degree + 3, 2).unwrap(),
            )
            .unwrap()
        });
        let columns = correction.as_ref().map_or(28, |p| p.input_columns());
        let outputs = correction
            .as_ref()
            .map_or(8, |p| p.output_expansion().len());
        let mut source_interpreter = source.compile(&prototype).unwrap();
        let mut source_native = source.compile_native(&prototype).unwrap();
        let mut correction_interpreter =
            correction.as_ref().map(|p| p.compile(&prototype).unwrap());
        let mut correction_native = correction
            .as_ref()
            .map(|p| p.compile_native(&prototype).unwrap());
        assert_eq!(source_native.backend(), KernelBackend::NativeRust);
        if let Some(e) = &correction_native {
            assert_eq!(e.backend(), KernelBackend::NativeRust);
        }
        for rows in [0, 1, 8, 32, 128, 1024, 3, 31, 127, 0, 1] {
            let mut args: Vec<_> = (0..rows * columns).map(|i| input(&prototype, i)).collect();
            if rows % 2 == 1 {
                args.reverse();
            }
            if let Some(plan) = &correction {
                for row in args.chunks_exact_mut(columns) {
                    row[plan.input_layout().denominator.start] =
                        Complex::new(prototype.from_i64(3), prototype.from_i64(2));
                }
            }
            let mut expected = vec![input(&prototype, 0); rows * outputs];
            let mut actual = expected.clone();
            if let Some(e) = &mut correction_interpreter {
                e.evaluate_batch(&args, &mut expected, rows).unwrap();
            } else {
                source_interpreter
                    .evaluate_batch(&args, &mut expected, rows)
                    .unwrap();
            }
            let mut evaluate = |args: &[Complex<T>], actual: &mut [Complex<T>], rows| {
                if let Some(e) = &mut correction_native {
                    e.evaluate_batch(args, actual, rows)
                } else {
                    source_native.evaluate_batch(args, actual, rows)
                }
            };
            evaluate(&args, &mut actual, rows).unwrap();
            for (a, b) in actual.iter().zip(&expected) {
                near(a, b, digits);
            }
            if rows > 0 {
                let before = actual.clone();
                assert!(evaluate(&args[..args.len() - 1], &mut actual, rows).is_err());
                assert_eq!(actual, before);
                assert!(evaluate(&args, &mut actual, usize::MAX).is_err());
                assert_eq!(actual, before);
                let len = actual.len();
                assert!(evaluate(&args, &mut actual[..len - 1], rows).is_err());
                assert_eq!(actual, before);
            }
        }
    }
}
#[test]
fn every_semantic_degree_precision_batch_and_transactional_shape() {
    scalar(0., 12);
    scalar(DoubleFloat::default(), 27);
    scalar(Float::new(256), 65);
}

#[test]
fn empty_quotient_and_dual_corrections_keep_interpreter_fallback() {
    let empty =
        TriangleBubbleSubtraction::for_quotient(&LaurentDivision::new(5, 2, 2).unwrap()).unwrap();
    let mut empty = empty.compile_native(&0.).unwrap();
    assert_eq!(empty.backend(), KernelBackend::SymbolicaInterpreter);
    empty
        .evaluate_batch(&vec![Complex::new(2., 0.); 28], &mut [], 1)
        .unwrap();
    let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]];
    let prototype = HyperDual::from_values(shape.clone(), vec![Complex::new(0., 0.); 5]);
    let source = BubbleSourceProjection::new();
    let mut source_dual = source.compile_dual(&0., &prototype).unwrap();
    assert_eq!(source_dual.backend(), KernelBackend::SymbolicaInterpreter);
    for degree in 0..=3 {
        let selected = TriangleBubbleSubtraction::for_quotient(
            &LaurentDivision::new(5, 3 + degree, 2).unwrap(),
        )
        .unwrap();
        let full = TriangleBubbleSubtraction::new(degree == 3).unwrap();
        let mut selected_dual = selected.compile_dual(&0., &prototype).unwrap();
        let mut full_dual = full.compile_dual(&0., &prototype).unwrap();
        assert_eq!(selected_dual.backend(), KernelBackend::SymbolicaInterpreter);
        let args: Vec<_> = (0..3 * selected.input_columns())
            .map(|i| {
                HyperDual::from_values(
                    shape.clone(),
                    (0..5).map(|j| input(&0., i * 5 + j)).collect(),
                )
            })
            .collect();
        let mut actual = vec![prototype.zero(); 3 * selected.output_expansion().len()];
        let mut expected = vec![prototype.zero(); 3 * full.output_expansion().len()];
        selected_dual.evaluate_batch(&args, &mut actual, 3).unwrap();
        full_dual.evaluate_batch(&args, &mut expected, 3).unwrap();
        let ninjars::ExpansionKind::T2 { powers } = selected.output_expansion() else {
            unreachable!()
        };
        let ninjars::ExpansionKind::T2 {
            powers: full_powers,
        } = full.output_expansion()
        else {
            unreachable!()
        };
        for row in 0..3 {
            for (column, power) in powers.iter().enumerate() {
                let a = &actual[row * powers.len() + column];
                let b = &expected[row * full_powers.len()
                    + full_powers.iter().position(|p| p == power).unwrap()];
                assert_eq!(a.get_shape(), b.get_shape());
                for (a, b) in a.values().iter().zip(b.values()) {
                    near(a, b, 10);
                }
            }
        }
        let before: Vec<_> = actual.iter().map(|d| d.values.clone()).collect();
        let mut malformed = args.clone();
        malformed[2].values.pop();
        assert!(
            selected_dual
                .evaluate_batch(&malformed, &mut actual, 3)
                .is_err()
        );
        assert_eq!(
            before,
            actual.iter().map(|d| d.values.clone()).collect::<Vec<_>>()
        );
    }
    // The extracted projection's all-component oracle is independently tested
    // against shared FourMomentum arithmetic in bubble_plans.rs.
    source_dual.evaluate_batch(&[], &mut [], 0).unwrap();
}

#[test]
fn warmed_native_bubble_storage_has_no_f64_allocations() {
    for degree in [None, Some(0), Some(1), Some(2), Some(3)] {
        let source = BubbleSourceProjection::new();
        let mut source = source.compile_native(&0.).unwrap();
        let correction = degree.map(|d| {
            TriangleBubbleSubtraction::for_quotient(&LaurentDivision::new(3, d + 1, 2).unwrap())
                .unwrap()
        });
        let columns = correction.as_ref().map_or(28, |p| p.input_columns());
        let outputs = correction
            .as_ref()
            .map_or(8, |p| p.output_expansion().len());
        let mut correction = correction.map(|p| p.compile_native(&0.).unwrap());
        for rows in [1, 1024, 1, 8, 32, 128, 3, 31, 127, 0, 1] {
            let args: Vec<_> = (0..rows * columns).map(|i| input(&0., i)).collect();
            let mut output = vec![Complex::new(0., 0.); rows * outputs];
            let mut evaluate = || {
                if let Some(e) = &mut correction {
                    e.evaluate_batch(&args, &mut output, rows).unwrap()
                } else {
                    source.evaluate_batch(&args, &mut output, rows).unwrap()
                }
            };
            if rows != 1 {
                evaluate();
            }
            assert_eq!(allocations(evaluate), 0, "degree{degree:?},rows{rows}");
        }
    }
}

#[test]
fn checked_in_bubble_catalog_is_the_exact_plan_export() {
    let directory =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/native_kernels/generated");
    let mut emitter = RustEmitter::default();
    let generated = BubbleSourceProjection::new()
        .export_rust_batch(&mut emitter, EmitOptions::default())
        .unwrap();
    assert_eq!(
        std::fs::read_to_string(directory.join("bubble_source_projection.rs")).unwrap(),
        format!(
            "// Generated by generate_native_bubbles from BubbleSourceProjection.\n{}",
            generated.source
        )
    );
    for degree in 0..=3 {
        let plan = TriangleBubbleSubtraction::for_quotient(
            &LaurentDivision::new(3, degree + 1, 2).unwrap(),
        )
        .unwrap();
        let generated = plan
            .export_rust_batch(&mut emitter, EmitOptions::default())
            .unwrap();
        let suffix = if degree == 3 {
            "higher".to_owned()
        } else {
            format!("degree{degree}")
        };
        assert_eq!(
            std::fs::read_to_string(directory.join(format!("bubble_correction_{suffix}.rs")))
                .unwrap(),
            format!(
                "// Generated by generate_native_bubbles from the existing exact correction, quotient degree {degree}.\n{}",
                generated.source
            )
        );
    }
    assert_eq!(
        std::fs::read_to_string(directory.join("bubble_constants.rs")).unwrap(),
        emitter.constants_source()
    );
}

#[test]
fn tracked_precision_inputs_are_not_promoted_to_working_precision() {
    let plan =
        TriangleBubbleSubtraction::for_quotient(&LaurentDivision::new(3, 4, 2).unwrap()).unwrap();
    let working = Float::new(256);
    let mut native = plan.compile_native(&working).unwrap();
    let mut interpreted = plan.compile(&working).unwrap();
    for bits in [53, 128, 512] {
        let args: Vec<_> = (0..3 * plan.input_columns())
            .map(|i| input(&Float::new(bits), i))
            .collect();
        let mut a =
            vec![Complex::new(working.zero(), working.zero()); 3 * plan.output_expansion().len()];
        let mut b = a.clone();
        native.evaluate_batch(&args, &mut a, 3).unwrap();
        interpreted.evaluate_batch(&args, &mut b, 3).unwrap();
        for (a, b) in a.iter().zip(&b) {
            near(
                a,
                b,
                if bits == 53 {
                    11
                } else if bits == 128 {
                    31
                } else {
                    65
                },
            );
            assert_eq!(a.re.get_precision(), b.re.get_precision());
            assert_eq!(a.im.get_precision(), b.im.get_precision());
        }
    }
}
