use std::collections::HashMap;

use ninjars::{Complex, ExpansionKind, FloatLike, Real};
use ninjars_symbolica::LaurentDivision;
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        atom::AtomField,
        dual::{DualNumber, HyperDual},
        float::{Float, SingleFloat},
    },
    parse, symbol,
};

fn c(x: f64) -> Complex<f64> {
    Complex::new(x, 0.)
}

fn coefficients(expression: &Atom, kind: &ExpansionKind) -> Vec<Complex<f64>> {
    let vars = [
        parse!("division_test::t"),
        parse!("division_test::x"),
        parse!("division_test::mu"),
    ];
    let polynomial = expression.expand().to_polynomial_in_vars_with_field::<i32>(
        &vars,
        &AtomField {
            statistical_zero_test: false,
            ..AtomField::new()
        },
    );
    let powers: Vec<_> = match kind {
        ExpansionKind::T3 { powers } => powers.iter().map(|&(t, m)| [t, 0, i32::from(m)]).collect(),
        ExpansionKind::T2 { powers } => powers
            .iter()
            .map(|&(t, x, m)| [t, i32::from(x), i32::from(m)])
            .collect(),
        _ => unreachable!(),
    };
    powers
        .iter()
        .map(|p| {
            polynomial
                .coefficient(p)
                .unwrap_or(Atom::Zero)
                .evaluate(&HashMap::<Atom, Complex<f64>>::new())
                .unwrap()
        })
        .collect()
}

#[test]
fn division_reconstructs_known_quotients_through_nine_lines_and_rank_n_plus_one() {
    let t = parse!("division_test::t");
    let x = parse!("division_test::x");
    let mu = parse!("division_test::mu");
    for cut in 1..=3 {
        for n in cut..=9 {
            let plan = LaurentDivision::new(n, n + 1, cut).unwrap();
            let mut evaluator = plan.compile(&0_f64).unwrap();
            let degree = cut + 1;
            let quotient = t.pow(degree as i32) + &t * 3 + &mu * 5 + 7;
            let quotient = if cut == 2 {
                quotient + &x * 11 + &x * &mu * 13 + x.pow(3) * 17
            } else {
                quotient
            };
            let mut input = Vec::new();
            for lane in 0..3 {
                let expected = &quotient * Atom::num(lane + 1);
                let denominators: Vec<_> = (0..n - cut)
                    .map(|j| {
                        let base = &t * Atom::num(j + 2) + Atom::num(3 + lane) + (&mu + 5) / &t;
                        if cut == 2 {
                            base + &x * 2 + (x.pow(2) + &x * 3) / &t
                        } else {
                            base
                        }
                    })
                    .collect();
                let numerator = denominators.iter().fold(expected, |p, d| p * d) + t.pow(-1);
                input.extend(coefficients(&numerator, plan.numerator_expansion()));
                for d in &denominators {
                    input.extend(coefficients(d, plan.denominator_expansion()));
                }
            }
            let mut output = vec![c(0.); 3 * plan.quotient_expansion().len()];
            evaluator.evaluate_batch(&input, &mut output, 3).unwrap();
            let expected = coefficients(&quotient, plan.quotient_expansion());
            for (lane, row) in output.chunks_exact(expected.len()).enumerate() {
                for (got, want) in row.iter().zip(&expected) {
                    assert!(
                        (got.re - want.re * (lane + 1) as f64).abs() < 1e-9,
                        "n={n} cut={cut} {got} {want}"
                    );
                    assert!(got.im.abs() < 1e-12);
                }
            }
            let mut scalar = vec![c(0.); expected.len()];
            evaluator
                .evaluate_batch(&input[..plan.input_columns()], &mut scalar, 1)
                .unwrap();
            assert_eq!(scalar, &output[..expected.len()]);
            evaluator.evaluate_batch(&[], &mut [], 0).unwrap();
            let saved = scalar.clone();
            assert!(
                evaluator
                    .evaluate_batch(&input[..plan.input_columns() - 1], &mut scalar, 1)
                    .is_err()
            );
            assert_eq!(saved, scalar);
        }
    }
}

#[test]
fn low_rank_and_invalid_layouts_are_explicit() {
    let plan = LaurentDivision::new(6, 0, 2).unwrap();
    assert!(plan.quotient_expansion().is_empty());
    let mut evaluator = plan.compile(&0_f64).unwrap();
    evaluator
        .evaluate_batch(&vec![c(1.); plan.input_columns()], &mut [], 1)
        .unwrap();
    assert!(LaurentDivision::new(0, 0, 1).is_err());
    assert!(LaurentDivision::new(3, 6, 3).is_err());
    assert!(LaurentDivision::new(3, 3, 4).is_err());
}

#[test]
fn mixed_dual_quotient_matches_symbolic_derivatives() {
    let plan = LaurentDivision::new(4, 4, 3).unwrap();
    let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]];
    let prototype = HyperDual::from_values(shape.clone(), vec![c(0.); shape.len()]);
    let mut evaluator = plan.compile_dual(&0., &prototype).unwrap();
    let mut inputs = vec![prototype.clone(); plan.input_columns()];
    inputs[0].values_mut()[0] = c(2.);
    inputs[0].values_mut()[1] = c(1.);
    let denominator = plan.numerator_expansion().len();
    inputs[denominator].values_mut()[0] = c(3.);
    inputs[denominator].values_mut()[2] = c(1.);
    inputs[denominator + 1].values_mut()[0] = c(5.);
    let mut result = vec![prototype.clone(); plan.quotient_expansion().len()];
    evaluator.evaluate_batch(&inputs, &mut result, 1).unwrap();
    let ExpansionKind::T3 { powers } = plan.quotient_expansion() else {
        unreachable!()
    };
    for (value, &(power, mu)) in result.iter().zip(powers) {
        let expression = if mu != 0 {
            Atom::Zero
        } else {
            parse!("division_test::a") * Atom::num(-5).pow(3 - power)
                / parse!("division_test::b").pow(4 - power)
        };
        let bindings = HashMap::from([
            (parse!("division_test::a"), c(2.)),
            (parse!("division_test::b"), c(3.)),
        ]);
        for (component, orders) in value.values().iter().zip(&shape) {
            let mut derivative = expression.clone();
            for _ in 0..orders[0] {
                derivative = derivative.derivative(symbol!("division_test::a"));
            }
            for _ in 0..orders[1] {
                derivative = derivative.derivative(symbol!("division_test::b"));
            }
            if orders[0] == 2 {
                derivative /= Atom::num(2);
            }
            let expected = derivative.evaluate(&bindings).unwrap();
            assert!((component.re - expected.re).abs() < 1e-12);
            assert!(component.im.abs() < 1e-12);
        }
    }
    // Equal primal values do not make a mismatched derivative shape valid.
    inputs[0] = HyperDual::from_values(vec![vec![0], vec![1]], vec![c(2.), c(1.)]);
    let before: Vec<_> = result.iter().map(|v| v.values().to_vec()).collect();
    assert!(evaluator.evaluate_batch(&inputs, &mut result, 1).is_err());
    assert_eq!(
        before,
        result
            .iter()
            .map(|v| v.values().to_vec())
            .collect::<Vec<_>>()
    );
}

#[test]
fn arbitrary_precision_keeps_small_quotient_corrections() {
    let plan = LaurentDivision::new(4, 1, 3).unwrap();
    let prototype = Float::new(192);
    let mut evaluator = plan.compile(&prototype).unwrap();
    let numerator = prototype.one()
        + prototype.from_rational(&symbolica::domains::rational::Rational::from((1, 10)).pow(30));
    let mut inputs = vec![Complex::new(prototype.zero(), prototype.zero()); plan.input_columns()];
    inputs[0].re = numerator.clone();
    inputs[1].re = prototype.from_i64(3);
    let mut out = vec![Complex::new(prototype.zero(), prototype.zero()); 1];
    evaluator.evaluate_batch(&inputs, &mut out, 1).unwrap();
    let expected = numerator / prototype.from_i64(3);
    let tolerance =
        prototype.from_rational(&symbolica::domains::rational::Rational::from((1, 10)).pow(55));
    assert!((out[0].re.clone() - expected).norm() < tolerance);
    assert_ne!(out[0].re, prototype.one() / prototype.from_i64(3));
}
