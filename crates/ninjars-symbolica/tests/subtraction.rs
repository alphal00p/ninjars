use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use ninjars::{Complex, ExpansionKind, FloatLike};
use ninjars_symbolica::{LaurentDivision, TriangleBubbleSubtraction};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::dual::{DualNumber, HyperDual},
    parse, symbol,
};

fn c(x: f64) -> Complex<f64> {
    Complex::new(x, 0.)
}

fn simple_row(plan: &TriangleBubbleSubtraction, cubic: f64, leading: f64) -> Vec<Complex<f64>> {
    let layout = plan.input_layout();
    let mut row = vec![c(0.); plan.input_columns()];
    row[layout.residue.start + 3] = c(cubic);
    row[layout.projection_e3.start + 2] = c(1.);
    row[layout.denominator.start] = c(leading);
    row[layout.denominator.start + 1] = c(5.);
    row
}

#[test]
fn subtraction_layout_scalar_batch_permutation_and_failed_shape() {
    for higher in [false, true] {
        let plan = TriangleBubbleSubtraction::new(higher).unwrap();
        let division = LaurentDivision::new(3, if higher { 4 } else { 3 }, 2).unwrap();
        assert_eq!(plan.output_expansion(), division.quotient_expansion());
        assert_eq!(
            plan.denominator_expansion(),
            division.denominator_expansion()
        );
        let mut evaluator = plan.compile(&0.).unwrap();
        let mut inputs = Vec::new();
        for lane in [2, 0, 1] {
            inputs.extend(simple_row(&plan, (lane + 1) as f64, 3.));
        }
        let width = plan.output_expansion().len();
        let mut outputs = vec![c(0.); 3 * width];
        evaluator.evaluate_batch(&inputs, &mut outputs, 3).unwrap();
        let ExpansionKind::T2 { powers } = plan.output_expansion() else {
            unreachable!()
        };
        for (row, lane) in outputs.chunks_exact(width).zip([2, 0, 1]) {
            for (value, &(t, x, mu)) in row.iter().zip(powers) {
                // Exact polynomial part of a*t³/(b*t+5).
                let expected = if x == 0 && mu == 0 && t <= 2 {
                    (lane + 1) as f64 * (-5_f64).powi(2 - t) / 3_f64.powi(3 - t)
                } else {
                    0.
                };
                assert!((value.re - expected).abs() < 1e-12);
                assert!(value.im.abs() < 1e-12);
            }
        }
        let mut scalar = vec![c(0.); width];
        evaluator
            .evaluate_batch(&inputs[..plan.input_columns()], &mut scalar, 1)
            .unwrap();
        assert_eq!(scalar, outputs[..width]);
        evaluator.evaluate_batch(&[], &mut [], 0).unwrap();
        let before = scalar.clone();
        assert!(
            evaluator
                .evaluate_batch(&inputs[..plan.input_columns() - 1], &mut scalar, 1)
                .is_err()
        );
        assert_eq!(before, scalar);
    }
}

#[test]
fn higher_and_mixed_dual_subtraction_matches_symbolic_derivatives() {
    let plan = TriangleBubbleSubtraction::new(true).unwrap();
    let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![0, 2], vec![1, 1]];
    let zero = HyperDual::from_values(shape.clone(), vec![c(0.); shape.len()]);
    let mut evaluator = plan.compile_dual(&0., &zero).unwrap();
    let layout = plan.input_layout();
    let mut inputs = vec![zero.clone(); plan.input_columns()];
    for (value, primal) in inputs.iter_mut().zip(simple_row(&plan, 2., 3.)) {
        value.values_mut()[0] = primal;
    }
    inputs[layout.residue.start + 3].values_mut()[1] = c(1.);
    inputs[layout.denominator.start].values_mut()[2] = c(1.);
    let mut result = vec![zero.clone(); plan.output_expansion().len()];
    evaluator.evaluate_batch(&inputs, &mut result, 1).unwrap();
    let ExpansionKind::T2 { powers } = plan.output_expansion() else {
        unreachable!()
    };
    for (value, &(t, x, mu)) in result.iter().zip(powers) {
        let expression = if x == 0 && mu == 0 && t <= 2 {
            parse!("subtraction_test::a") * Atom::num(-5).pow(2 - t)
                / parse!("subtraction_test::b").pow(3 - t)
        } else {
            Atom::Zero
        };
        let bindings = HashMap::from([
            (parse!("subtraction_test::a"), c(2.)),
            (parse!("subtraction_test::b"), c(3.)),
        ]);
        for (component, orders) in value.values().iter().zip(&shape) {
            let mut derivative = expression.clone();
            for _ in 0..orders[0] {
                derivative = derivative.derivative(symbol!("subtraction_test::a"));
            }
            for _ in 0..orders[1] {
                derivative = derivative.derivative(symbol!("subtraction_test::b"));
            }
            if orders[1] == 2 {
                derivative /= Atom::num(2);
            }
            let expected = derivative.evaluate(&bindings).unwrap();
            assert!((component.re - expected.re).abs() < 1e-12);
            assert!(component.im.abs() < 1e-12);
        }
    }
    inputs[0] = HyperDual::from_values(vec![vec![0], vec![1]], vec![c(0.), c(0.)]);
    let before: Vec<_> = result.iter().map(|x| x.values().to_vec()).collect();
    assert!(evaluator.evaluate_batch(&inputs, &mut result, 1).is_err());
    assert_eq!(
        before,
        result
            .iter()
            .map(|x| x.values().to_vec())
            .collect::<Vec<_>>()
    );
    assert!(zero.is_fully_zero());
}

/// Compare every coefficient with unchanged original Ninja generated code.
#[test]
#[ignore = "requires C++ compiler and validation/reference/build.py artifacts"]
fn triangle_bubble_matches_original_subtractions_with_offset_origins() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let reference = std::env::var_os("NINJARS_REFERENCE_BUILD")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("target/reference"));
    let executable =
        std::env::temp_dir().join(format!("ninjars-subtraction-{}", std::process::id()));
    let output = Command::new(std::env::var_os("CXX").unwrap_or_else(|| "c++".into()))
        .arg("-std=c++11")
        .arg("-O2")
        .arg(format!(
            "-I{}",
            reference.join("ninja-source/src").display()
        ))
        .arg(format!("-I{}", reference.join("ninja-build/src").display()))
        .arg(root.join("validation/reference/subtraction_probe.cc"))
        .arg(reference.join("ninja-build/src/.libs/libninja.a"))
        .arg("-o")
        .arg(&executable)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = Command::new(&executable).output().unwrap();
    fs::remove_file(&executable).unwrap();
    assert!(output.status.success());
    let data = String::from_utf8(output.stdout).unwrap();
    let mut tokens = data.split_whitespace();
    let mut cases = 0;
    while let Some(higher) = tokens.next() {
        let plan = TriangleBubbleSubtraction::new(higher == "1").unwrap();
        let columns: usize = tokens.next().unwrap().parse().unwrap();
        let count: usize = tokens.next().unwrap().parse().unwrap();
        assert_eq!(columns, plan.input_columns());
        assert_eq!(count, plan.output_expansion().len());
        let mut read = || {
            Complex::new(
                tokens.next().unwrap().parse::<f64>().unwrap(),
                tokens.next().unwrap().parse::<f64>().unwrap(),
            )
        };
        let inputs: Vec<_> = (0..columns).map(|_| read()).collect();
        let expected: Vec<_> = (0..count).map(|_| read()).collect();
        let mut got = vec![c(0.); count];
        plan.compile(&0.)
            .unwrap()
            .evaluate_batch(&inputs, &mut got, 1)
            .unwrap();
        for (value, want) in got.iter().zip(&expected) {
            assert!(
                (value.re - want.re).abs() <= 1e-10 * (1. + want.re.abs()),
                "case {cases}: {value} versus {want}"
            );
            assert!(
                (value.im - want.im).abs() <= 1e-10 * (1. + want.im.abs()),
                "case {cases}: {value} versus {want}"
            );
        }
        cases += 1;
    }
    assert_eq!(cases, 6);
}

#[test]
fn tadpole_subtractions_preserve_offsets_and_full_layout() {
    use ninjars_symbolica::TadpoleSubtraction;
    assert!(TadpoleSubtraction::new(1, false).is_err());
    assert!(TadpoleSubtraction::new(4, true).is_err());
    for source in [2, 3] {
        for higher in [false, true] {
            let plan = TadpoleSubtraction::new(source, higher).unwrap();
            let division = LaurentDivision::new(source, source + usize::from(higher), 1).unwrap();
            assert_eq!(plan.output_expansion(), division.quotient_expansion());
            assert_eq!(
                plan.denominator_expansion(),
                division.denominator_expansion()
            );
            let layout = plan.input_layout();
            let mut row = vec![c(0.); plan.input_columns()];
            let residue_index = match (source, higher) {
                (3, true) => 12,
                (3, false) => 3,
                (2, true) => 13,
                (2, false) => 2,
                _ => unreachable!(),
            };
            row[residue_index] = c(3.);
            row[layout.projections[0].start] = c(1.);
            row[layout.projections[0].start + 1] = c(1.);
            for d in &layout.denominators {
                row[d.start] = c(1.);
                row[d.start + 1] = c(1.);
            }
            let mut evaluator = plan.compile(&0.).unwrap();
            let mut out = vec![c(0.); plan.output_expansion().len()];
            evaluator.evaluate_batch(&row, &mut out, 1).unwrap();
            // Exact cancellation leaves3*(t+1), or3*(t+1)² at higher rank.
            assert_eq!(
                out,
                if higher {
                    vec![c(3.), c(6.), c(3.), c(0.)]
                } else {
                    vec![c(3.), c(3.)]
                }
            );
            let mut batch = Vec::new();
            batch.extend_from_slice(&row);
            batch.extend_from_slice(&row);
            let mut batched = vec![c(0.); 2 * out.len()];
            evaluator.evaluate_batch(&batch, &mut batched, 2).unwrap();
            assert_eq!(&batched[..out.len()], out);
            assert_eq!(&batched[out.len()..], out);
            evaluator.evaluate_batch(&[], &mut [], 0).unwrap();
        }
    }
}

#[test]
fn tadpole_dual_retains_mu_terms_and_mixed_parameter_derivatives() {
    use ninjars_symbolica::TadpoleSubtraction;
    let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![0, 2], vec![1, 1]];
    let zero = HyperDual::from_values(shape.clone(), vec![c(0.); shape.len()]);
    for source in [2, 3] {
        let plan = TadpoleSubtraction::new(source, true).unwrap();
        let layout = plan.input_layout();
        let mut inputs = vec![zero.clone(); plan.input_columns()];
        let coef = 10; // mu²*E3² triangle, mu²*E2 bubble.
        inputs[coef].values_mut()[0] = c(2.);
        inputs[coef].values_mut()[1] = c(1.);
        inputs[layout.projections[0].start + 1].values_mut()[0] = c(1.);
        for d in &layout.denominators {
            inputs[d.start].values_mut()[0] = c(1.);
        }
        inputs[layout.denominators[0].start].values_mut()[0] = c(3.);
        inputs[layout.denominators[0].start].values_mut()[2] = c(1.);
        let mut evaluator = plan.compile_dual(&0., &zero).unwrap();
        let mut result = vec![zero.clone(); plan.output_expansion().len()];
        evaluator.evaluate_batch(&inputs, &mut result, 1).unwrap();
        let last = result.last().unwrap();
        let expression = parse!("subtraction_test::a/subtraction_test::b");
        let bindings = HashMap::from([
            (parse!("subtraction_test::a"), c(2.)),
            (parse!("subtraction_test::b"), c(3.)),
        ]);
        for (value, orders) in last.values().iter().zip(&shape) {
            let mut derivative = expression.clone();
            for _ in 0..orders[0] {
                derivative = derivative.derivative(symbol!("subtraction_test::a"));
            }
            for _ in 0..orders[1] {
                derivative = derivative.derivative(symbol!("subtraction_test::b"));
            }
            if orders[1] == 2 {
                derivative /= Atom::num(2);
            }
            let expected = derivative.evaluate(&bindings).unwrap();
            assert!((value.re - expected.re).abs() < 1e-12);
            assert!(value.im.abs() < 1e-12);
        }
        assert!(
            result[..result.len() - 1]
                .iter()
                .all(FloatLike::is_fully_zero)
        );
    }
}

#[test]
#[ignore = "requires C++ compiler and validation/reference/build.py artifacts"]
fn tadpole_coefficients_match_original_generated_subtractions() {
    use ninjars_symbolica::TadpoleSubtraction;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let reference = std::env::var_os("NINJARS_REFERENCE_BUILD")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("target/reference"));
    let executable = std::env::temp_dir().join(format!(
        "ninjars-tadpole-subtraction-{}",
        std::process::id()
    ));
    let output = Command::new(std::env::var_os("CXX").unwrap_or_else(|| "c++".into()))
        .arg("-std=c++11")
        .arg("-O2")
        .arg(format!(
            "-I{}",
            reference.join("ninja-source/src").display()
        ))
        .arg(format!("-I{}", reference.join("ninja-build/src").display()))
        .arg(root.join("validation/reference/tadpole_subtraction_probe.cc"))
        .arg(reference.join("ninja-build/src/.libs/libninja.a"))
        .arg("-o")
        .arg(&executable)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = Command::new(&executable).output().unwrap();
    fs::remove_file(&executable).unwrap();
    assert!(output.status.success());
    let data = String::from_utf8(output.stdout).unwrap();
    let mut tokens = data.split_whitespace();
    let mut cases = 0;
    while let Some(source) = tokens.next() {
        let source = source.parse().unwrap();
        let higher = tokens.next().unwrap() == "1";
        let plan = TadpoleSubtraction::new(source, higher).unwrap();
        let columns: usize = tokens.next().unwrap().parse().unwrap();
        let count: usize = tokens.next().unwrap().parse().unwrap();
        assert_eq!(columns, plan.input_columns());
        assert_eq!(count, plan.output_expansion().len());
        let mut read = || {
            Complex::new(
                tokens.next().unwrap().parse::<f64>().unwrap(),
                tokens.next().unwrap().parse::<f64>().unwrap(),
            )
        };
        let inputs: Vec<_> = (0..columns).map(|_| read()).collect();
        let expected: Vec<_> = (0..count).map(|_| read()).collect();
        let mut got = vec![c(0.); count];
        plan.compile(&0.)
            .unwrap()
            .evaluate_batch(&inputs, &mut got, 1)
            .unwrap();
        for (value, want) in got.iter().zip(&expected) {
            assert!(
                (value.re - want.re).abs() <= 1e-10 * (1. + want.re.abs()),
                "case {cases}: {value} versus {want}"
            );
            assert!(
                (value.im - want.im).abs() <= 1e-10 * (1. + want.im.abs()),
                "case {cases}: {value} versus {want}"
            );
        }
        cases += 1;
    }
    assert_eq!(cases, 8);
}
