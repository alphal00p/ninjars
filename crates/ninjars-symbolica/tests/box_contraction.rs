use ninjars::{Complex, MasterIntegral};
use ninjars_symbolica::MasterContraction;
use symbolica::domains::dual::HyperDual;

fn c(value: f64) -> Complex<f64> {
    Complex::new(value, 0.)
}
fn near(actual: Complex<f64>, expected: Complex<f64>) {
    assert!(
        (actual.re - expected.re).abs() < 1e-12 * (1. + expected.re.abs())
            && (actual.im - expected.im).abs() < 1e-12 * (1. + expected.im.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn original_box_indices_and_batched_rational_contraction() {
    for higher in [false, true] {
        let plan = MasterContraction::new(4, higher).unwrap();
        let count = if higher { 6 } else { 5 };
        let layout = plan.input_layout();
        assert_eq!(layout.coefficients, 0..count);
        assert_eq!(layout.invariants, count..count + 6);
        assert_eq!(layout.masses_squared, count + 6..count + 10);
        assert_eq!(layout.e12, None);
        assert_eq!(layout.ke2, None);
        assert_eq!(plan.input_columns(), count + 10);
        assert_eq!(plan.masters(), &[MasterIntegral::D0]);
        let mut evaluator = plan.compile(&0.).unwrap();
        for lanes in [0, 1, 7, 33] {
            let mut input = vec![Complex::new(71., 29.); lanes * plan.input_columns()];
            let mut masters = Vec::new();
            for (row, source) in input
                .chunks_exact_mut(plan.input_columns())
                .zip((0..lanes).rev())
            {
                row[0] = Complex::new((source + 1) as f64, 0.25);
                row[4] = Complex::new(6. * (source + 2) as f64, -3.);
                masters.push([c(2.), c(3.), c(5.)]);
            }
            let mut weights = vec![c(99.); lanes * 2];
            evaluator
                .weights_batch(&input, &mut weights, lanes)
                .unwrap();
            let mut output = vec![c(99.); lanes * 5];
            evaluator
                .contract_batch(&weights, &masters, &mut output, lanes)
                .unwrap();
            for (row, source) in output.as_chunks::<5>().0.iter().zip((0..lanes).rev()) {
                let coefficient = Complex::new((source + 1) as f64, 0.25);
                let rational = Complex::new(-((source + 2) as f64), 0.5);
                for (actual, expected) in row.iter().zip([
                    coefficient * 2. + rational,
                    coefficient * 3.,
                    coefficient * 5.,
                    coefficient * 2.,
                    rational,
                ]) {
                    near(*actual, expected);
                }
            }
            if lanes > 0 {
                let saved = output.clone();
                assert!(
                    evaluator
                        .contract_batch(&weights, &masters[..lanes - 1], &mut output, lanes)
                        .is_err()
                );
                assert_eq!(output, saved);
            }
        }
    }
}

#[test]
fn box_contraction_keeps_higher_and_mixed_dual_components() {
    let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]];
    let dual = |values: [f64; 5]| HyperDual::from_values(shape.clone(), values.map(c).to_vec());
    let zero = dual([0.; 5]);
    let plan = MasterContraction::new(4, true).unwrap();
    let mut evaluator = plan.compile_dual(&0., &zero).unwrap();
    let mut input = vec![dual([101.; 5]); plan.input_columns()];
    input[0] = dual([2., 3., 5., 7., 11.]);
    input[4] = dual([6., 12., 18., 24., 30.]);
    let mut weights = vec![zero.clone(); 2];
    evaluator.weights_batch(&input, &mut weights, 1).unwrap();
    let masters = [[
        dual([13., 17., 19., 23., 29.]),
        dual([31., 37., 41., 43., 47.]),
        zero.clone(),
    ]];
    let mut output = vec![zero; 5];
    evaluator
        .contract_batch(&weights, &masters, &mut output, 1)
        .unwrap();
    for (actual, expected) in output.iter().zip([
        [25., 71., 100., 184., 338.],
        [62., 167., 237., 414., 743.],
        [0.; 5],
        [26., 73., 103., 188., 343.],
        [-1., -2., -3., -4., -5.],
    ]) {
        for (actual, expected) in actual.values.iter().zip(expected) {
            near(*actual, c(expected));
        }
    }
}
