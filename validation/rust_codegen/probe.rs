//! Compiled scalar control against the existing batched Symbolica evaluators.
use ninjars::{BatchShape, Result};
use ninjars_symbolica::rust_codegen::runtime::{C, ConstantSpec, Context, NativeFloat, primitives};
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
    for kernel in 0..3 {
        let (inputs, outputs) = LAYOUTS[kernel];
        for rows in [0, 1, 3, 8] {
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
            records.push(json!({"domain":name,"bits":bits,"kernel":kernel_name,"rows":rows,"scalar_components":actual.len(),"dual_components":flat_actual.len(),"maximum_scalar_error":scalar_error,"maximum_dual_error":dual_error,"tolerance":tolerance}));
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
    println!("{}",serde_json::to_string_pretty(&json!({"passed":true,"execution":"compiled scalar row-loop control; not pipeline batching","performance_acceptance":false,"metadata":serde_json::from_str::<Value>(include_str!("metadata.json")).unwrap(),"checks":records})).unwrap());
}
