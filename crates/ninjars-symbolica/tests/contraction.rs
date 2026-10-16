use ninjars::{Complex, DoubleFloat, Float, MasterProvider, MasterRequest, Real, SingleFloat};
use ninjars_symbolica::{MasterContraction, SymbolicaMasters};
use symbolica::{
    domains::{dual::HyperDual, rational::Rational},
    evaluate::EvaluationDomain,
};

fn c(value: f64) -> Complex<f64> {
    Complex::new(value, 0.)
}
fn close(actual: Complex<f64>, expected: f64) {
    assert!(
        (actual.re - expected).abs() < 1e-9 * (1. + expected.abs()),
        "{actual} != {expected}"
    );
    assert!(
        actual.im.abs() < 1e-9 * (1. + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn original_indices_and_higher_rank_rational_weights() {
    for (cut, higher, expected) in [
        (1, false, vec![2., 0.]),
        (1, true, vec![19.5, 134.75]),
        (2, false, vec![2., -21., 245., 91. / 3.]),
        (2, true, vec![2., -21., 245., -4459., 210.]),
        (3, false, vec![2., 1.5]),
        (3, true, vec![2., 13. / 12.]),
    ] {
        let plan = MasterContraction::new(cut, higher).unwrap();
        let layout = plan.input_layout();
        let mut input = vec![c(0.); plan.input_columns()];
        input[0] = c(2.);
        match cut {
            1 => {
                input[layout.masses_squared.start] = c(7.);
                input[layout.e12.unwrap()] = c(-2.);
                if higher {
                    assert_eq!(layout.coefficients.len(), 16);
                    input[9] = c(12345.); // original reserved entry
                    input[14] = c(3.);
                    input[15] = c(5.);
                }
            }
            2 => {
                input[1] = c(3.);
                input[2] = c(5.);
                input[9] = c(7.);
                input[layout.invariants.start] = c(-2.);
                input[layout.masses_squared.clone()].copy_from_slice(&[c(3.), c(5.)]);
                input[layout.ke2.unwrap()] = c(-7.);
                if higher {
                    input[10] = c(11.);
                    input[13] = c(13.);
                }
            }
            _ => {
                input[7] = c(3.);
                input[layout.invariants.clone()].copy_from_slice(&[c(3.), c(-1.), c(8.)]);
                input[layout.masses_squared.clone()].fill(c(1.));
                if higher {
                    input[14] = c(5.);
                }
            }
        }
        let mut evaluator = plan.compile(&0.).unwrap();
        let mut output = vec![c(0.); plan.weight_columns()];
        evaluator.weights_batch(&input, &mut output, 1).unwrap();
        for (actual, expected) in output.into_iter().zip(expected) {
            close(actual, expected);
        }
    }
    assert!(MasterContraction::new(0, false).is_err());
    assert!(MasterContraction::new(5, true).is_err());
}

#[test]
fn bubble_null_moments_match_independent_ninja_oneloop_values() {
    // Original Ninja tensor runner, V0=0,V1=(0,1,0,0), m0²=m1²=mu_R²=1,
    // numerators (q0-q1)^r for r=1,2,3. Here e2=(1,1,0,0), so K.e2=-1.
    // Compare the total Laurent coefficients: the original reducer's choice
    // of residue basis can split rational terms differently from direct B11/B111.
    let finite = [
        -0.076_022_352_410_010_04,
        -0.045_807_580_991_126_25,
        -0.030_700_195_280_005_67,
    ];
    let plan = MasterContraction::new(2, true).unwrap();
    let layout = plan.input_layout();
    let mut input = vec![c(0.); plan.input_columns() * 3];
    for (row, index) in input.chunks_exact_mut(plan.input_columns()).zip([1, 2, 13]) {
        row[index] = c(1.);
        row[layout.invariants.start] = c(-1.);
        row[layout.masses_squared.clone()].fill(c(1.));
        row[layout.ke2.unwrap()] = c(-1.);
    }
    let mut evaluator = plan.compile(&0.).unwrap();
    let mut weights = vec![c(0.); plan.weight_columns() * 3];
    evaluator.weights_batch(&input, &mut weights, 3).unwrap();
    let mut provider = SymbolicaMasters::new(0.);
    let mut masters = vec![[c(0.); 3]; plan.masters().len() * 3];
    for (index, &integral) in plan.masters().iter().enumerate() {
        let mut values = [[c(0.); 3]];
        provider
            .evaluate(
                MasterRequest {
                    integral,
                    invariants: &[c(-1.)],
                    masses_squared: &[c(1.), c(1.)],
                    scales_squared: &[c(1.)],
                },
                &mut values,
            )
            .unwrap();
        for row in 0..3 {
            masters[row * plan.masters().len() + index] = values[0];
        }
    }
    let mut output = vec![c(0.); 15];
    evaluator
        .contract_batch(&weights, &masters, &mut output, 3)
        .unwrap();
    for (index, row) in output.as_chunks::<5>().0.iter().enumerate() {
        close(row[0], finite[index]);
        close(row[1], 1. / (index as f64 + 2.));
        close(row[2], 0.);
        close(row[3], finite[index]);
        close(row[4], 0.);
    }
}

#[test]
fn explicit_mu_triangle_terms_only_change_the_finite_rational_coefficient() {
    // Independently recorded by validation/reference/test_generated_runner.py:
    // mu² triangle ->1/2; mu⁴ triangle at s=[3,-1,8],m²=[1,1,1] ->-1/12.
    let plan = MasterContraction::new(3, true).unwrap();
    let layout = plan.input_layout();
    let mut input = vec![c(0.); plan.input_columns() * 2];
    for (row, coefficient) in input.chunks_exact_mut(plan.input_columns()).zip([7, 14]) {
        row[coefficient] = c(1.);
        row[layout.invariants.clone()].copy_from_slice(&[c(3.), c(-1.), c(8.)]);
        row[layout.masses_squared.clone()].fill(c(1.));
    }
    let mut evaluator = plan.compile(&0.).unwrap();
    let mut weights = vec![c(0.); plan.weight_columns() * 2];
    evaluator.weights_batch(&input, &mut weights, 2).unwrap();
    let mut output = vec![c(0.); 10];
    evaluator
        .contract_batch(&weights, &[[c(17.); 3]; 2], &mut output, 2)
        .unwrap();
    for (row, rational) in output.as_chunks::<5>().0.iter().zip([0.5, -1. / 12.]) {
        for (actual, expected) in row.iter().zip([rational, 0., 0., 0., rational]) {
            close(*actual, expected);
        }
    }
}

fn precise_weights<T: Real + SingleFloat>(prototype: T)
where
    Complex<T>: Real + EvaluationDomain,
{
    let plan = MasterContraction::new(3, false).unwrap();
    let mut evaluator = plan.compile(&prototype).unwrap();
    let zero = Complex::new(prototype.zero(), prototype.zero());
    let mut input = vec![zero.clone(); plan.input_columns()];
    let tiny = prototype.from_rational(&Rational::from((1, 10)).pow(30));
    let value = prototype.one() + tiny;
    input[0] = Complex::new(value.clone(), prototype.zero());
    input[7] = Complex::new(
        prototype.from_rational(&Rational::from((2, 3))),
        prototype.zero(),
    );
    let mut output = vec![zero; plan.weight_columns()];
    evaluator.weights_batch(&input, &mut output, 1).unwrap();
    assert_eq!(output[0].re, value);
    assert!(output[0].re != prototype.one());
    assert_eq!(
        output[1].re,
        prototype.from_rational(&Rational::from((1, 3)))
    );
}

#[test]
fn exact_constants_and_inputs_preserve_doublefloat_and_float_precision() {
    precise_weights(DoubleFloat::default());
    precise_weights(Float::new(192));
}

#[test]
fn dual_weights_and_master_products_keep_higher_and_mixed_components() {
    let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]];
    let dual = |values: [f64; 5]| HyperDual::from_values(shape.clone(), values.map(c).to_vec());
    let zero = dual([0.; 5]);
    let plan = MasterContraction::new(1, true).unwrap();
    let layout = plan.input_layout();
    let mut evaluator = plan.compile_dual(&0., &zero).unwrap();
    let mut input = vec![zero.clone(); plan.input_columns()];
    input[15] = dual([1., 0., 1., 0., 0.]);
    input[layout.masses_squared.start] = dual([3., 1., 0., 0., 0.]);
    input[layout.e12.unwrap()] = dual([2., 0., 0., 0., 0.]);
    let mut weights = vec![zero.clone(); plan.weight_columns()];
    evaluator.weights_batch(&input, &mut weights, 1).unwrap();
    let masters = [[
        dual([2., 1., 0., 0., 0.]),
        dual([3., 0., 0., 0., 0.]),
        dual([5., 0., 0., 0., 0.]),
    ]];
    let mut output = vec![zero; 5];
    evaluator
        .contract_batch(&weights, &masters, &mut output, 1)
        .unwrap();
    let expected = [
        [-5.25, -4., -5.25, -0.75, -4.],
        [-4.5, -1.5, -4.5, 0., -1.5],
        [-7.5, -2.5, -7.5, 0., -2.5],
        [-3., -2.5, -3., -0.5, -2.5],
        [-2.25, -1.5, -2.25, -0.25, -1.5],
    ];
    for (actual, expected) in output.iter().zip(expected) {
        for (actual, expected) in actual.values.iter().zip(expected) {
            close(*actual, expected);
        }
    }
}

#[test]
fn empty_batches_and_invalid_shapes_preserve_the_output() {
    let plan = MasterContraction::new(1, false).unwrap();
    let mut evaluator = plan.compile(&0.).unwrap();
    evaluator.weights_batch(&[], &mut [], 0).unwrap();
    evaluator.contract_batch(&[], &[], &mut [], 0).unwrap();
    let mut output = [c(17.); 5];
    assert!(
        evaluator
            .contract_batch(&[c(1.), c(0.)], &[], &mut output, 1)
            .is_err()
    );
    assert_eq!(output, [c(17.); 5]);
    assert!(evaluator.weights_batch(&[], &mut output, 1).is_err());
    assert_eq!(output, [c(17.); 5]);
}
