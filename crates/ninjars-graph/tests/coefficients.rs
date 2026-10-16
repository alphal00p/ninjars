use std::collections::HashMap;

use ninjars_graph::{CoefficientError, PolynomialCoefficientPlan};
use symbolica::{
    atom::{Atom, AtomCore, Symbol},
    domains::float::{Complex, DoubleFloat, Float, Real, RealLike, SingleFloat},
    evaluate::EvaluationDomain,
    parse,
};

fn variables() -> [Atom; 5] {
    [
        parse!("q0"),
        parse!("q1"),
        parse!("q2"),
        parse!("q3"),
        parse!("mu2"),
    ]
}
fn c(value: f64) -> Complex<f64> {
    Complex::new(value, 0.)
}
fn plan(expression: Atom, parameters: Vec<Atom>) -> PolynomialCoefficientPlan {
    PolynomialCoefficientPlan::new(expression, variables(), parameters).unwrap()
}
fn close(actual: Complex<f64>, expected: Complex<f64>) {
    assert!(
        (actual - expected).norm().re < 2e-12 * (1. + expected.norm().re),
        "{actual} != {expected}"
    );
}

#[test]
fn shared_polynomial_reconstructs_exactly_and_counts_factor_once() {
    let expression = parse!("3*(a+b)^4*(q0^2+q1*q2)+7*(a-b)*mu2^2+5");
    let plan = plan(expression.clone(), vec![parse!("a"), parse!("b")]);
    assert_eq!(plan.rank(), 4);
    assert_eq!(plan.polynomial().nterms(), 4);
    assert_eq!(plan.coefficient_expressions().len(), 3);
    let restored = plan
        .coefficient_symbols()
        .iter()
        .zip(plan.coefficient_expressions())
        .fold(plan.template().clone(), |value, (symbol, head)| {
            value.replace(symbol.clone()).with(head.clone())
        });
    assert!((restored - expression).expand().is_zero());
    // External powers remain in a coefficient; the preparation never expands
    // the physical coefficient into a second monomial representation.
    assert!(
        plan.coefficient_expressions()
            .iter()
            .any(|head| head.contains(parse!("(a+b)^4")))
    );
}

#[test]
fn factoring_preserves_exact_numerator_and_merges_equivalent_heads() {
    let expression = parse!("(a^2-2*a*b+b^2)*q0+(a-b)^2*q1+(a-b)^2*mu2^2");
    let original = plan(expression.clone(), vec![parse!("a"), parse!("b")]);
    assert_eq!(original.coefficient_expressions().len(), 2);
    let factored = original.factor_coefficients().unwrap();
    assert_eq!(factored.rank(), 4);
    assert_eq!(factored.coefficient_expressions().len(), 1);
    let restored = factored
        .coefficient_symbols()
        .iter()
        .zip(factored.coefficient_expressions())
        .fold(factored.template().clone(), |value, (symbol, head)| {
            value.replace(symbol.clone()).with(head.clone())
        });
    assert!((restored - expression).expand().is_zero());
    let zero = plan(Atom::Zero, vec![]).factor_coefficients().unwrap();
    assert!(zero.coefficient_expressions().is_empty());
    assert!(zero.template().is_zero());
}

#[test]
fn factored_coefficients_resolve_cancellation_against_high_precision() {
    let original = plan(parse!("(a^2-2*a*b+b^2)*q0"), vec![parse!("a"), parse!("b")]);
    let mut oracle = original.compile(Float::new(256)).unwrap();
    let factored = original.factor_coefficients().unwrap();
    let mut interpreter = factored.compile(0.).unwrap();
    #[cfg(feature = "numerator-jit")]
    let mut jit = factored.compile_symjit_o2().unwrap();
    // All integers are exactly representable in binary64. The expanded head
    // subtracts terms of order 2^54 to obtain one, zero or a complex square.
    let a = 134_217_728.;
    let rows = [c(a + 1.), c(a), c(a), c(a), Complex::new(a, 1.), c(a)];
    let high_input = rows
        .iter()
        .map(|v| Complex::new(Float::with_val(256, v.re), Float::with_val(256, v.im)))
        .collect::<Vec<_>>();
    let mut high_output = vec![Complex::new(Float::new(256), Float::new(256)); 3];
    oracle
        .evaluate_batch(&high_input, &mut high_output, 3)
        .unwrap();
    let expected = high_output
        .iter()
        .map(|v| Complex::new(v.re.to_f64(), v.im.to_f64()))
        .collect::<Vec<_>>();
    let mut actual = vec![c(0.); 3];
    interpreter.evaluate_batch(&rows, &mut actual, 3).unwrap();
    assert_eq!(actual, expected);
    #[cfg(feature = "numerator-jit")]
    {
        jit.evaluate_batch(&rows, &mut actual, 3).unwrap();
        assert_eq!(actual, expected);
    }
}

#[test]
fn rejects_loop_nonpolynomials_and_ambiguous_bindings() {
    for expression in [parse!("q0^-1"), parse!("sin(q2)"), parse!("sqrt(mu2)")] {
        assert!(matches!(
            PolynomialCoefficientPlan::new(expression, variables(), vec![]),
            Err(CoefficientError::NotPolynomial)
        ));
    }
    assert!(matches!(
        PolynomialCoefficientPlan::new(parse!("q0"), variables(), vec![parse!("q0")]),
        Err(CoefficientError::Variables)
    ));
    let mut vars = variables();
    vars[3] = parse!("2");
    assert!(matches!(
        PolynomialCoefficientPlan::new(parse!("q0"), vars, vec![]),
        Err(CoefficientError::Variables)
    ));
}

#[test]
fn interpreter_binding_is_transactional_and_preserves_lane_order() {
    let plan = plan(
        parse!("q0/a+(a+b)*q1^2+3*mu2"),
        vec![parse!("a"), parse!("b")],
    );
    let mut evaluator = plan.compile(0.).unwrap();
    let input = [c(2.), c(3.), Complex::new(1., 1.), c(-2.), c(-4.), c(4.)];
    let columns = plan.coefficient_expressions().len();
    let mut output = vec![c(77.); 3 * columns];
    evaluator.evaluate_batch(&input, &mut output, 3).unwrap();
    for (lane, row) in input.as_chunks::<2>().0.iter().enumerate() {
        let bindings = HashMap::from([(parse!("a"), row[0]), (parse!("b"), row[1])]);
        for (column, head) in plan.coefficient_expressions().iter().enumerate() {
            close(
                output[lane * columns + column],
                head.evaluate_with_prec(&bindings, 53).unwrap(),
            );
        }
    }
    let mut permuted = vec![c(0.); output.len()];
    let input_permuted = [input[4], input[5], input[0], input[1], input[2], input[3]];
    evaluator
        .evaluate_batch(&input_permuted, &mut permuted, 3)
        .unwrap();
    for (row, old) in [2, 0, 1].into_iter().enumerate() {
        assert_eq!(
            &permuted[row * columns..(row + 1) * columns],
            &output[old * columns..(old + 1) * columns]
        );
    }
    evaluator.evaluate_batch(&[], &mut [], 0).unwrap();
    let unchanged = output.clone();
    assert!(matches!(
        evaluator.evaluate_batch(&input[..5], &mut output, 3),
        Err(CoefficientError::Shape { .. })
    ));
    assert_eq!(output, unchanged);
    let mut bad = input;
    bad[2] = c(f64::NAN);
    assert!(matches!(
        evaluator.evaluate_batch(&bad, &mut output, 3),
        Err(CoefficientError::NonFiniteInput { row: 1, column: 0 })
    ));
    assert_eq!(output, unchanged);
    bad[2] = c(0.);
    assert!(matches!(
        evaluator.evaluate_batch(&bad, &mut output, 3),
        Err(CoefficientError::NonFiniteCoefficient { row: 1, .. })
    ));
    assert_eq!(output, unchanged);
    assert!(matches!(
        evaluator.evaluate_batch(&[], &mut [], usize::MAX),
        Err(CoefficientError::Overflow)
    ));
    evaluator.evaluate_batch(&input, &mut output, 3).unwrap();
    assert_eq!(output, unchanged);
}

fn precision_control<T: Real + RealLike + SingleFloat>(prototype: T, denominator: &str)
where
    Complex<T>: Real + EvaluationDomain,
{
    let delta = parse!(denominator);
    let expression = parse!("q0") * (parse!("a") + Atom::one() + &delta);
    let plan = plan(expression, vec![parse!("a")]);
    let mut evaluator = plan.compile(prototype.clone()).unwrap();
    let mut output = vec![Complex::new(prototype.zero(), prototype.zero()); 1];
    evaluator
        .evaluate_batch(
            &[Complex::new(prototype.from_i64(-1), prototype.zero())],
            &mut output,
            1,
        )
        .unwrap();
    let expected: Complex<T> = delta
        .evaluate_with_prec(
            &HashMap::<Atom, Complex<T>>::new(),
            prototype.get_precision(),
        )
        .unwrap();
    assert!(
        (output[0].re.clone() / expected.re - prototype.one())
            .to_f64()
            .abs()
            < 1e-7
    );
    assert!(output[0].im.is_zero());
}

#[test]
fn generic_interpreter_preserves_precision_beyond_binary64() {
    precision_control(DoubleFloat::from(0.), "1/10000000000000000000000000");
    precision_control(
        Float::new(256),
        "1/1000000000000000000000000000000000000000000000000000000000000",
    );
}

#[test]
fn zero_and_registered_constant_only_heads_have_strict_batch_shapes() {
    let zero = plan(Atom::Zero, vec![parse!("a")]);
    assert!(zero.coefficient_expressions().is_empty());
    assert!(zero.exact_evaluator().unwrap().is_none());
    let mut evaluator = zero.compile(0.).unwrap();
    evaluator
        .evaluate_batch(&[c(2.), c(3.)], &mut [], 2)
        .unwrap();
    assert!(evaluator.evaluate_batch(&[], &mut [], 2).is_err());
    let constant = plan(Atom::var(Symbol::PI) * 3, vec![]);
    let mut evaluator = constant.compile(0.).unwrap();
    let mut output = [c(0.); 3];
    evaluator.evaluate_batch(&[], &mut output, 3).unwrap();
    for value in output {
        close(value, c(3. * std::f64::consts::PI));
    }
}

#[cfg(feature = "numerator-jit")]
#[test]
fn symjit_o2_multihead_matches_interpreter_for_complex_lanes_and_tails() {
    let plan = plan(
        parse!("3*(a+b)^4*(q0^2+q1*q2)+(sin(a)+cos(b))*q3+(a-b)*mu2^2+5"),
        vec![parse!("a"), parse!("b")],
    );
    let mut interpreted = plan.compile(0.).unwrap();
    let mut jit = plan.compile_symjit_o2().unwrap();
    let input = [c(2.), c(3.), Complex::new(1., 1.), c(-2.), c(-4.), c(4.)];
    for indices in [vec![], vec![1], vec![2, 0, 1], vec![2, 1, 1, 0, 2]] {
        let input = indices
            .iter()
            .flat_map(|i| input[2 * i..2 * i + 2].iter().copied())
            .collect::<Vec<_>>();
        let mut expected = vec![c(0.); indices.len() * plan.coefficient_expressions().len()];
        let mut actual = expected.clone();
        interpreted
            .evaluate_batch(&input, &mut expected, indices.len())
            .unwrap();
        jit.evaluate_batch(&input, &mut actual, indices.len())
            .unwrap();
        for (a, e) in actual.into_iter().zip(expected) {
            close(a, e);
        }
    }
}

#[cfg(feature = "numerator-jit")]
#[test]
fn symjit_constant_zero_and_failed_binding_controls() {
    for expression in [Atom::Zero, Atom::var(Symbol::PI) * 3, parse!("7")] {
        let plan = plan(expression.clone(), vec![]);
        let mut jit = plan.compile_symjit_o2().unwrap();
        let mut output = vec![c(0.); 3 * plan.coefficient_expressions().len()];
        jit.evaluate_batch(&[], &mut output, 3).unwrap();
        if !output.is_empty() {
            for value in output {
                close(
                    value,
                    expression
                        .evaluate_with_prec(&HashMap::<Atom, Complex<f64>>::new(), 53)
                        .unwrap(),
                );
            }
        }
        jit.evaluate_batch(&[], &mut [], 0).unwrap();
    }
    let plan = plan(parse!("q0/a"), vec![parse!("a")]);
    let mut jit = plan.compile_symjit_o2().unwrap();
    let mut output = [c(13.); 3];
    assert!(matches!(
        jit.evaluate_batch(&[c(2.), c(0.), c(3.)], &mut output, 3),
        Err(CoefficientError::NonFiniteCoefficient { row: 1, column: 0 })
    ));
    assert_eq!(output, [c(13.); 3]);
    assert!(matches!(
        jit.evaluate_batch(&[c(f64::INFINITY); 3], &mut output, 3),
        Err(CoefficientError::NonFiniteInput { row: 0, column: 0 })
    ));
    assert_eq!(output, [c(13.); 3]);
    assert!(jit.evaluate_batch(&[], &mut output, 3).is_err());
    jit.evaluate_batch(&[c(2.), c(4.), c(8.)], &mut output, 3)
        .unwrap();
    assert_eq!(output, [c(0.5), c(0.25), c(0.125)]);
}

#[cfg(feature = "numerator-jit")]
#[test]
fn symjit_preserves_lazy_coefficient_branches_across_reactivated_lanes() {
    let conditional = Symbol::IF.call((parse!("condition"), parse!("1/a"), Atom::num(7)));
    let expression = parse!("q0") * &conditional + parse!("q1^2") * (conditional + parse!("b"));
    for factored in [false, true] {
        let plan = plan(
            expression.clone(),
            vec![parse!("condition"), parse!("a"), parse!("b")],
        );
        let plan = if factored {
            plan.factor_coefficients().unwrap()
        } else {
            plan
        };
        let mut interpreted = plan.compile(0.).unwrap();
        let mut jit = plan.compile_symjit_o2().unwrap();
        let columns = plan.coefficient_expressions().len();
        for input in [
            vec![
                c(0.),
                c(0.),
                c(3.),
                c(1.),
                c(2.),
                c(-1.),
                c(0.),
                c(0.),
                c(4.),
            ],
            vec![
                c(1.),
                Complex::new(1., 1.),
                c(3.),
                c(0.),
                c(0.),
                c(-1.),
                c(1.),
                c(4.),
                c(4.),
            ],
        ] {
            let mut expected = vec![c(0.); 3 * columns];
            let mut actual = expected.clone();
            interpreted
                .evaluate_batch(&input, &mut expected, 3)
                .unwrap();
            jit.evaluate_batch(&input, &mut actual, 3).unwrap();
            for (actual, expected) in actual.into_iter().zip(expected) {
                close(actual, expected);
            }
        }
    }
}
