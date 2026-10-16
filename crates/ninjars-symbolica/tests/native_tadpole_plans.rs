#![cfg(feature = "rust-codegen")]

use ninjars::{Complex, DoubleFloat, Float, FloatLike, Real, SingleFloat};
use ninjars_symbolica::{
    DualTadpolePlanEvaluator, TadpoleGeometryProbe, TadpolePlanEvaluator, TadpoleSourceProjection,
    rust_codegen::{
        EmitOptions, RustEmitter, RustKernelExport,
        runtime::{BatchWorkspace, C, ConstantSpec, Context, NativeFloat, batch, primitives},
    },
};
use symbolica::{
    domains::{
        dual::{DualNumber, HyperDual},
        rational::Rational,
    },
    evaluate::EvaluationDomain,
};

#[rustfmt::skip]
#[path = "../src/native_kernels/generated/tadpole_constants.rs"]
mod constants;
#[allow(clippy::deref_addrof,clippy::identity_op,clippy::erasing_op)]
#[rustfmt::skip]
#[path = "../src/native_kernels/generated/tadpole_triangle_projection.rs"]
mod triangle;
#[allow(clippy::deref_addrof,clippy::identity_op,clippy::erasing_op)]
#[rustfmt::skip]
#[path = "../src/native_kernels/generated/tadpole_triangle_projection_dual.rs"]
mod triangle_dual;
#[allow(clippy::deref_addrof,clippy::identity_op,clippy::erasing_op)]
#[rustfmt::skip]
#[path = "../src/native_kernels/generated/tadpole_bubble_projection.rs"]
mod bubble;
#[allow(clippy::deref_addrof,clippy::identity_op,clippy::erasing_op)]
#[rustfmt::skip]
#[path = "../src/native_kernels/generated/tadpole_bubble_projection_dual.rs"]
mod bubble_dual;
#[allow(clippy::deref_addrof,clippy::identity_op,clippy::erasing_op)]
#[rustfmt::skip]
#[path = "../src/native_kernels/generated/tadpole_geometry_probe.rs"]
mod probe;
#[allow(clippy::deref_addrof,clippy::identity_op,clippy::erasing_op)]
#[rustfmt::skip]
#[path = "../src/native_kernels/generated/tadpole_geometry_probe_dual.rs"]
mod probe_dual;

fn input<T: NativeFloat>(p: &T, i: usize) -> Complex<T> {
    Complex::new(
        p.from_rational(&Rational::from(((i % 37) as i64 - 11, 17))),
        p.from_rational(&Rational::from(((i % 13) as i64 - 5, 19))),
    )
}
fn near<T: NativeFloat + Into<Float>>(a: &Complex<T>, b: &Complex<T>, digits: u64) {
    let p = Float::new(384);
    let tolerance = p.from_rational(&Rational::from((1, 10)).pow(digits));
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
fn shape() -> Vec<Vec<usize>> {
    vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]]
}
type PlanEvaluators<T> = (
    usize,
    usize,
    TadpolePlanEvaluator<T>,
    DualTadpolePlanEvaluator<T, HyperDual<Complex<T>>>,
);

fn plans<T: NativeFloat>(kind: usize, p: &T, dual: &HyperDual<Complex<T>>) -> PlanEvaluators<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    if kind == 2 {
        let plan = TadpoleGeometryProbe::new();
        (
            plan.input_columns(),
            plan.output_columns(),
            plan.compile(p).unwrap(),
            plan.compile_dual(p, dual).unwrap(),
        )
    } else {
        let plan = TadpoleSourceProjection::new(3 - kind).unwrap();
        (
            plan.input_columns(),
            plan.output_columns(),
            plan.compile(p).unwrap(),
            plan.compile_dual(p, dual).unwrap(),
        )
    }
}
fn evaluate<T: NativeFloat>(
    kind: usize,
    dual: bool,
    input: &[C<T>],
    output: &mut [C<T>],
    rows: usize,
    context: &Context<T>,
    workspace: &mut BatchWorkspace<T>,
) -> Result<(), String> {
    match (kind, dual) {
        (0, false) => triangle::evaluate_batch(input, output, rows, context, workspace),
        (0, true) => triangle_dual::evaluate_batch(input, output, rows, context, workspace),
        (1, false) => bubble::evaluate_batch(input, output, rows, context, workspace),
        (1, true) => bubble_dual::evaluate_batch(input, output, rows, context, workspace),
        (2, false) => probe::evaluate_batch(input, output, rows, context, workspace),
        (2, true) => probe_dual::evaluate_batch(input, output, rows, context, workspace),
        _ => unreachable!(),
    }
}
fn parity<T: NativeFloat + Into<Float>>(p: T, digits: u64)
where
    Complex<T>: Real + EvaluationDomain,
{
    let context = Context::new(constants::CONSTANTS, p.get_precision()).unwrap();
    let mut workspace = BatchWorkspace::new(p.clone());
    let dual = HyperDual::from_values(shape(), vec![input(&p, 0).zero(); 5]);
    for kind in 0..3 {
        let (columns, width, mut scalar, mut dual_evaluator) = plans(kind, &p, &dual);
        for rows in [0, 1, 8, 32, 128, 1024, 3, 31, 0, 1] {
            for is_dual in [false, true] {
                let components = if is_dual { 5 } else { 1 };
                let mut args: Vec<_> = (0..rows * columns * components)
                    .map(|i| input(&p, i))
                    .collect();
                if rows % 2 == 1 {
                    args.reverse();
                }
                let mut expected = vec![input(&p, 0); rows * width * components];
                if is_dual {
                    let args: Vec<_> = args
                        .as_chunks::<5>()
                        .0
                        .iter()
                        .map(|values| HyperDual::from_values(shape(), values.to_vec()))
                        .collect();
                    let mut values = vec![dual.zero(); rows * width];
                    dual_evaluator
                        .evaluate_batch(&args, &mut values, rows)
                        .unwrap();
                    for (target, value) in expected.as_chunks_mut::<5>().0.iter_mut().zip(&values) {
                        target.clone_from_slice(value.values());
                    }
                } else {
                    scalar.evaluate_batch(&args, &mut expected, rows).unwrap();
                }
                let mut actual = expected.clone();
                evaluate(
                    kind,
                    is_dual,
                    &args,
                    &mut actual,
                    rows,
                    &context,
                    &mut workspace,
                )
                .unwrap();
                for (a, b) in actual.iter().zip(&expected) {
                    near(a, b, digits);
                }
                if rows > 0 {
                    let before = actual.clone();
                    assert!(
                        evaluate(
                            kind,
                            is_dual,
                            &args[..args.len() - 1],
                            &mut actual,
                            rows,
                            &context,
                            &mut workspace
                        )
                        .is_err()
                    );
                    assert_eq!(actual, before);
                    assert!(
                        evaluate(
                            kind,
                            is_dual,
                            &args,
                            &mut actual,
                            usize::MAX,
                            &context,
                            &mut workspace
                        )
                        .is_err()
                    );
                    assert_eq!(actual, before);
                    let len = actual.len();
                    assert!(
                        evaluate(
                            kind,
                            is_dual,
                            &args,
                            &mut actual[..len - 1],
                            rows,
                            &context,
                            &mut workspace
                        )
                        .is_err()
                    );
                    assert_eq!(actual, before);
                }
            }
        }
    }
}

#[test]
fn emitted_plans_match_interpreter_for_precisions_batches_and_every_dual_component() {
    parity(0., 12);
    parity(DoubleFloat::default(), 27);
    parity(Float::new(256), 65);
}

#[test]
fn generated_catalog_and_constant_indices_regenerate_exactly() {
    let directory =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/native_kernels/generated");
    let mut emitter = RustEmitter::default();
    let structure = HyperDual::<Complex<Rational>>::new(shape());
    for (kind, name) in ["triangle_projection", "bubble_projection", "geometry_probe"]
        .into_iter()
        .enumerate()
    {
        for dual in [false, true] {
            let generated = if kind == 2 {
                let plan = TadpoleGeometryProbe::new();
                if dual {
                    plan.export_rust_batch_dual(&structure, &mut emitter, EmitOptions::default())
                } else {
                    plan.export_rust_batch(&mut emitter, EmitOptions::default())
                }
            } else {
                let plan = TadpoleSourceProjection::new(3 - kind).unwrap();
                if dual {
                    plan.export_rust_batch_dual(&structure, &mut emitter, EmitOptions::default())
                } else {
                    plan.export_rust_batch(&mut emitter, EmitOptions::default())
                }
            }
            .unwrap();
            let source = format!(
                "// Generated by generate_native_tadpoles; exact tadpole plan {name}, Dual={dual}.\n{}",
                generated.source
            );
            let filename = format!("tadpole_{name}{}.rs", if dual { "_dual" } else { "" });
            assert_eq!(
                std::fs::read_to_string(directory.join(filename)).unwrap(),
                source
            );
        }
    }
    assert_eq!(
        std::fs::read_to_string(directory.join("tadpole_constants.rs")).unwrap(),
        emitter.constants_source()
    );
}

#[test]
fn float_tracked_precision_and_malformed_duals_are_preserved() {
    let p = Float::new(256);
    let context = Context::new(constants::CONSTANTS, 256).unwrap();
    let mut workspace = BatchWorkspace::new(p.clone());
    let dual = HyperDual::from_values(shape(), vec![input(&p, 0).zero(); 5]);
    for kind in 0..3 {
        let (columns, width, mut scalar, mut dual_evaluator) = plans(kind, &p, &dual);
        for bits in [53, 128, 512] {
            let args: Vec<_> = (0..3 * columns)
                .map(|i| input(&Float::new(bits), i))
                .collect();
            let mut expected = vec![input(&p, 0); 3 * width];
            let mut actual = expected.clone();
            scalar.evaluate_batch(&args, &mut expected, 3).unwrap();
            evaluate(kind, false, &args, &mut actual, 3, &context, &mut workspace).unwrap();
            for (a, b) in actual.iter().zip(&expected) {
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
        let mut args = vec![dual.clone(); columns];
        let mut actual = vec![dual.clone(); width];
        let before: Vec<_> = actual.iter().map(|v| v.values().to_vec()).collect();
        args[columns - 1].values.pop();
        assert!(
            dual_evaluator
                .evaluate_batch(&args, &mut actual, 1)
                .is_err()
        );
        assert_eq!(
            before,
            actual
                .iter()
                .map(|v| v.values().to_vec())
                .collect::<Vec<_>>()
        );
        let reordered = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![2, 0], vec![1, 1]];
        args[columns - 1] = HyperDual::from_values(reordered, dual.values().to_vec());
        assert!(
            dual_evaluator
                .evaluate_batch(&args, &mut actual, 1)
                .is_err()
        );
        assert_eq!(
            before,
            actual
                .iter()
                .map(|v| v.values().to_vec())
                .collect::<Vec<_>>()
        );
    }
}
