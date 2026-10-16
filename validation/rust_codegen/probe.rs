//! Instruction-major Rust kernels against scalar and Symbolica batch controls.
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

use ninjars::{BatchShape, Result};
use ninjars_symbolica::rust_codegen::runtime::{
    BatchWorkspace, C, ConstantSpec, Context, NativeFloat, batch, primitives,
};
use ninjars_symbolica::{DenominatorExpansion, LaurentDivision, TriangleBubbleSubtraction};
use serde_json::{Value, json};
use symbolica::{
    domains::{
        dual::{DualNumber, HyperDual},
        float::{DoubleFloat, Float, Real, RealLike},
        rational::Rational,
    },
    evaluate::EvaluationDomain,
};

include!("dispatch.rs");

struct CountingAllocator;

thread_local! {
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
}

fn count_allocation() {
    let _ = ALLOCATIONS.try_with(|count| {
        if let Some(value) = count.get() {
            count.set(Some(value + 1));
        }
    });
}

// Count only this thread's calls so independent runtime workers cannot affect
// the warmed-kernel measurement. Every allocation still belongs to System.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count_allocation();
        unsafe { System.realloc(ptr, layout, size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocations(run: impl FnOnce()) -> usize {
    ALLOCATIONS.with(|count| assert!(count.replace(Some(0)).is_none()));
    run();
    ALLOCATIONS.with(|count| count.replace(None).unwrap())
}

fn scalar_rows<T: NativeFloat>(
    kernel: usize,
    dual: bool,
    input: &[C<T>],
    output: &mut [C<T>],
    rows: usize,
    context: &Context<T>,
) -> Result<()> {
    let components = if dual { 5 } else { 1 };
    let (inputs, outputs) = LAYOUTS[kernel];
    BatchShape {
        rows,
        columns: inputs * components,
    }
    .validate("compiled probe input", input.len())?;
    BatchShape {
        rows,
        columns: outputs * components,
    }
    .validate("compiled probe output", output.len())?;
    for (input, output) in input
        .chunks_exact(inputs * components)
        .zip(output.chunks_exact_mut(outputs * components))
    {
        generated(kernel, dual, input, context, output);
    }
    Ok(())
}

fn shape() -> Vec<Vec<usize>> {
    vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]]
}

fn rational<T: NativeFloat>(prototype: &T, numerator: i64, denominator: i64) -> T {
    prototype.from_rational(&Rational::from((numerator, denominator)))
}

fn compare<T: NativeFloat + RealLike>(actual: &[C<T>], expected: &[C<T>], tolerance: f64) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0_f64;
    for (actual, expected) in actual.iter().zip(expected) {
        let error = (actual.re.clone() - &expected.re)
            .norm()
            .to_f64()
            .hypot((actual.im.clone() - &expected.im).norm().to_f64());
        let scale = expected
            .re
            .norm()
            .to_f64()
            .hypot(expected.im.norm().to_f64())
            .max(1.0);
        let relative = error / scale;
        assert!(
            relative.is_finite() && relative <= tolerance,
            "native/interpreted error {relative:e} > {tolerance:e}"
        );
        maximum = maximum.max(relative);
    }
    maximum
}

fn domain<T: NativeFloat + RealLike>(
    prototype: T,
    bits: u32,
    tolerance: f64,
    name: &str,
    records: &mut Vec<Value>,
) where
    C<T>: Real + EvaluationDomain,
{
    let context = Context::<T>::new(constants::CONSTANTS, bits).unwrap();
    let zero = C::new(prototype.zero(), prototype.zero());
    let dual = HyperDual::from_values(shape(), vec![zero.clone(); 5]);
    let plans = (
        LaurentDivision::new(4, 5, 3).unwrap(),
        TriangleBubbleSubtraction::new(true).unwrap(),
        DenominatorExpansion::new(2).unwrap(),
    );
    let mut scalar = (
        plans.0.compile(&prototype).unwrap(),
        plans.1.compile(&prototype).unwrap(),
        plans.2.compile(&prototype).unwrap(),
    );
    let mut differentiated = (
        plans.0.compile_dual(&prototype, &dual).unwrap(),
        plans.1.compile_dual(&prototype, &dual).unwrap(),
        plans.2.compile_dual(&prototype, &dual).unwrap(),
    );
    // Reuse one workspace across kernel/derivative layouts and changing widths.
    let mut workspace = BatchWorkspace::new(prototype.clone());
    for kernel in 0..3 {
        let (inputs, outputs) = LAYOUTS[kernel];
        for (iteration, rows) in [0, 1, 8, 32, 128, 1024, 3, 31, 127, 0, 8, 1]
            .into_iter()
            .enumerate()
        {
            let mut input = (0..rows * inputs)
                .map(|index| {
                    let (row, col) = (index / inputs, index % inputs);
                    C::new(
                        rational(
                            &prototype,
                            ((row + 2) * (col + 1) % 17) as i64 - 8,
                            (col % 5 + 2) as i64,
                        ),
                        rational(&prototype, ((col + 3 * row) % 7) as i64 - 3, 13),
                    )
                })
                .collect::<Vec<_>>();
            let leading = match kernel {
                0 => Some(plans.0.numerator_expansion().len()),
                1 => Some(plans.1.input_layout().denominator.start),
                _ => None,
            };
            if let Some(leading) = leading {
                for row in 0..rows {
                    input[row * inputs + leading] = C::new(
                        rational(&prototype, 20 + row as i64, 10),
                        rational(&prototype, 1, 7),
                    );
                }
            }
            let mut expected = vec![zero.clone(); rows * outputs];
            match kernel {
                0 => scalar.0.evaluate_batch(&input, &mut expected, rows),
                1 => scalar.1.evaluate_batch(&input, &mut expected, rows),
                _ => scalar.2.evaluate_batch(&input, &mut expected, rows),
            }
            .unwrap();
            let mut actual = vec![zero.clone(); rows * outputs];
            scalar_rows(kernel, false, &input, &mut actual, rows, &context).unwrap();
            let scalar_error = compare(&actual, &expected, tolerance);
            let mut batched = vec![zero.clone(); actual.len()];
            generated_batch(
                kernel,
                false,
                &input,
                &mut batched,
                rows,
                &context,
                &mut workspace,
            )
            .unwrap();
            let batch_error = compare(&batched, &expected, tolerance);
            let batch_scalar_error = compare(&batched, &actual, tolerance);
            let input_dual = input
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let mut values = vec![value.clone()];
                    values.extend((1..5).map(|component| {
                        C::new(
                            rational(&prototype, ((index + component) % 5) as i64 - 2, 11),
                            rational(&prototype, ((index + 2 * component) % 7) as i64 - 3, 17),
                        )
                    }));
                    HyperDual::from_values(shape(), values)
                })
                .collect::<Vec<_>>();
            let mut expected_dual = vec![dual.clone(); rows * outputs];
            match kernel {
                0 => differentiated
                    .0
                    .evaluate_batch(&input_dual, &mut expected_dual, rows),
                1 => differentiated
                    .1
                    .evaluate_batch(&input_dual, &mut expected_dual, rows),
                _ => differentiated
                    .2
                    .evaluate_batch(&input_dual, &mut expected_dual, rows),
            }
            .unwrap();
            let flat_input = input_dual
                .iter()
                .flat_map(|value| value.values().iter().cloned())
                .collect::<Vec<_>>();
            let flat_expected = expected_dual
                .iter()
                .flat_map(|value| value.values().iter().cloned())
                .collect::<Vec<_>>();
            let mut flat_actual = vec![zero.clone(); flat_expected.len()];
            scalar_rows(kernel, true, &flat_input, &mut flat_actual, rows, &context).unwrap();
            let dual_error = compare(&flat_actual, &flat_expected, tolerance);
            let mut flat_batched = vec![zero.clone(); flat_expected.len()];
            generated_batch(
                kernel,
                true,
                &flat_input,
                &mut flat_batched,
                rows,
                &context,
                &mut workspace,
            )
            .unwrap();
            let dual_batch_error = compare(&flat_batched, &flat_expected, tolerance);
            let dual_batch_scalar_error = compare(&flat_batched, &flat_actual, tolerance);
            let mut warmed_allocations = Vec::new();
            let mut single_row_allocations = Vec::new();
            for (is_dual, input, output, expected) in [
                (false, &input, &mut batched, &expected),
                (true, &flat_input, &mut flat_batched, &flat_expected),
            ] {
                if name == "f64" {
                    let count = allocations(|| {
                        generated_batch(
                            kernel,
                            is_dual,
                            input,
                            output,
                            rows,
                            &context,
                            &mut workspace,
                        )
                        .unwrap();
                    });
                    assert_eq!(count, 0, "warmed f64 kernel allocated");
                    warmed_allocations.push(count);
                }
                if rows == 1 {
                    let mut fresh = BatchWorkspace::new(prototype.clone());
                    let count = allocations(|| {
                        generated_batch(kernel, is_dual, input, output, rows, &context, &mut fresh)
                            .unwrap();
                    });
                    assert_eq!(fresh.capacity(), 0);
                    assert_eq!(fresh.storage_len(), 0);
                    if name == "f64" {
                        assert_eq!(count, 0, "fresh single-row f64 kernel allocated");
                        single_row_allocations.push(count);
                    }
                    compare(output, expected, tolerance);
                }
                if rows > 1 {
                    let components = if is_dual { 5 } else { 1 };
                    let reversed_input = input
                        .chunks_exact(inputs * components)
                        .rev()
                        .flatten()
                        .cloned()
                        .collect::<Vec<_>>();
                    let reversed_expected = expected
                        .chunks_exact(outputs * components)
                        .rev()
                        .flatten()
                        .cloned()
                        .collect::<Vec<_>>();
                    generated_batch(
                        kernel,
                        is_dual,
                        &reversed_input,
                        output,
                        rows,
                        &context,
                        &mut workspace,
                    )
                    .unwrap();
                    compare(output, &reversed_expected, tolerance);
                }
                // Invalid calls must not mutate output or grow the workspace.
                let before = output.clone();
                let storage = (workspace.capacity(), workspace.storage_len());
                assert!(
                    generated_batch(
                        kernel,
                        is_dual,
                        input,
                        output,
                        usize::MAX,
                        &context,
                        &mut workspace,
                    )
                    .is_err()
                );
                if rows > 0 {
                    assert!(
                        generated_batch(
                            kernel,
                            is_dual,
                            &input[..input.len() - 1],
                            output,
                            rows,
                            &context,
                            &mut workspace,
                        )
                        .is_err()
                    );
                    let short = output.len() - 1;
                    assert!(
                        generated_batch(
                            kernel,
                            is_dual,
                            input,
                            &mut output[..short],
                            rows,
                            &context,
                            &mut workspace,
                        )
                        .is_err()
                    );
                } else {
                    assert!(
                        generated_batch(
                            kernel,
                            is_dual,
                            &[zero.clone()],
                            output,
                            rows,
                            &context,
                            &mut workspace,
                        )
                        .is_err()
                    );
                }
                assert_eq!(compare(output, &before, 0.), 0.);
                assert_eq!((workspace.capacity(), workspace.storage_len()), storage);
            }
            if rows > 0 {
                let before = actual.clone();
                assert!(
                    scalar_rows(
                        kernel,
                        false,
                        &input[..input.len() - 1],
                        &mut actual,
                        rows,
                        &context
                    )
                    .is_err()
                );
                assert_eq!(compare(&actual, &before, 0.), 0.);
                let before = flat_actual.clone();
                assert!(
                    scalar_rows(
                        kernel,
                        true,
                        &flat_input,
                        &mut flat_actual[..flat_expected.len() - 1],
                        rows,
                        &context
                    )
                    .is_err()
                );
                assert_eq!(compare(&flat_actual, &before, 0.), 0.);
                let malformed =
                    vec![
                        HyperDual::from_values(vec![vec![0], vec![1]], vec![zero.clone(); 2]);
                        rows * inputs
                    ];
                let before = expected_dual
                    .iter()
                    .flat_map(|value| value.values().iter().cloned())
                    .collect::<Vec<_>>();
                assert!(
                    match kernel {
                        0 => differentiated
                            .0
                            .evaluate_batch(&malformed, &mut expected_dual, rows),
                        1 => differentiated
                            .1
                            .evaluate_batch(&malformed, &mut expected_dual, rows),
                        _ => differentiated
                            .2
                            .evaluate_batch(&malformed, &mut expected_dual, rows),
                    }
                    .is_err()
                );
                let after = expected_dual
                    .iter()
                    .flat_map(|value| value.values().iter().cloned())
                    .collect::<Vec<_>>();
                assert_eq!(compare(&after, &before, 0.), 0.);
            }
            let kernel_name = ["laurent", "subtraction", "denominator"][kernel];
            records.push(json!({
                "domain":name,"bits":bits,"kernel":kernel_name,"iteration":iteration,"rows":rows,
                "scalar_components":actual.len(),"dual_components":flat_actual.len(),
                "maximum_scalar_error":scalar_error,"maximum_dual_error":dual_error,
                "maximum_batch_error":batch_error,"maximum_dual_batch_error":dual_batch_error,
                "maximum_batch_scalar_error":batch_scalar_error,
                "maximum_dual_batch_scalar_error":dual_batch_scalar_error,
                "warmed_f64_allocations":warmed_allocations,
                "fresh_single_row_f64_allocations":single_row_allocations,
                "workspace_capacity":workspace.capacity(),"workspace_cells":workspace.storage_len(),
                "permutation_checked":rows > 1,"malformed_buffers_checked":true,
                "overflow_checked":true,"tolerance":tolerance
            }));
        }
    }
}

pub fn run() {
    let mut records = Vec::new();
    domain(0_f64, 53, 1e-12, "f64", &mut records);
    domain(
        DoubleFloat::from(0.),
        106,
        1e-25,
        "DoubleFloat",
        &mut records,
    );
    domain(Float::new(256), 256, 1e-65, "Float256", &mut records);
    println!("{}",serde_json::to_string_pretty(&json!({"passed":true,"execution":"compiled instruction-major batch kernels with scalar and interpreted batch controls","production_dispatch":false,"performance_acceptance":false,"metadata":serde_json::from_str::<Value>(include_str!("metadata.json")).unwrap(),"checks":records})).unwrap());
}
