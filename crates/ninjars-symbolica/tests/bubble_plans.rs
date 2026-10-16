use ninjars::{Complex, ExpansionKind, FourMomentum, Real};
use ninjars_symbolica::{BubbleSourceProjection, LaurentDivision, TriangleBubbleSubtraction};
use symbolica::domains::dual::{DualNumber, DualNumberStructure, HyperDual};

fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}
fn row(seed: usize, columns: usize) -> Vec<Complex<f64>> {
    (0..columns)
        .map(|i| {
            c(
                ((i + seed) % 11) as f64 - 4.,
                ((3 * i + seed) % 7) as f64 - 2.,
            )
        })
        .collect()
}
fn momentum<T: Clone>(row: &[T], offset: usize) -> FourMomentum<T> {
    FourMomentum::from_args(
        row[offset].clone(),
        row[offset + 1].clone(),
        row[offset + 2].clone(),
        row[offset + 3].clone(),
    )
}
fn projection_control<T: Real>(row: &[T]) -> Vec<T> {
    let mut values = Vec::new();
    for axis in [20, 24] {
        let e = momentum(row, axis);
        values.push((momentum(row, 0) + momentum(row, 16)).dot(&e));
        for offset in [4, 8, 12] {
            values.push(momentum(row, offset).dot(&e));
        }
    }
    values
}

#[test]
fn source_origin_projection_uses_shared_momentum_and_all_dual_components() {
    let plan = BubbleSourceProjection::new();
    let mut evaluator = plan.compile(&0.).unwrap();
    let mut inputs = Vec::new();
    for seed in [3, 0, 2] {
        inputs.extend(row(seed, 28));
    }
    let mut output = vec![c(0., 0.); 24];
    evaluator.evaluate_batch(&inputs, &mut output, 3).unwrap();
    for (input, actual) in inputs
        .as_chunks::<28>()
        .0
        .iter()
        .zip(output.as_chunks::<8>().0.iter())
    {
        assert_eq!(actual.as_slice(), projection_control(input));
    }
    let before = output.clone();
    assert!(
        evaluator
            .evaluate_batch(&inputs[..83], &mut output, 3)
            .is_err()
    );
    assert_eq!(before, output);
    evaluator.evaluate_batch(&[], &mut [], 0).unwrap();
    let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]];
    let prototype = HyperDual::from_values(shape.clone(), vec![c(0., 0.); 5]);
    let mut dual = plan.compile_dual(&0., &prototype).unwrap();
    let input: Vec<_> = (0..28)
        .map(|i| HyperDual::from_values(shape.clone(), row(i, 5)))
        .collect();
    let mut actual = vec![prototype.clone(); 8];
    dual.evaluate_batch(&input, &mut actual, 1).unwrap();
    for (actual, expected) in actual.iter().zip(projection_control(&input)) {
        assert_eq!(actual.get_shape(), expected.get_shape());
        assert_eq!(actual.values(), expected.values());
    }
}

#[test]
fn exact_power_subset_agrees_with_full_correction_in_every_degree_class() {
    for n in [3, 5] {
        for degree in 0..=3 {
            let division = LaurentDivision::new(n, n - 2 + degree, 2).unwrap();
            let selected = TriangleBubbleSubtraction::for_quotient(&division).unwrap();
            let full = TriangleBubbleSubtraction::new(degree == 3).unwrap();
            assert_eq!(selected.output_expansion(), division.quotient_expansion());
            let ExpansionKind::T2 {
                powers: selected_powers,
            } = selected.output_expansion()
            else {
                unreachable!()
            };
            let ExpansionKind::T2 {
                powers: full_powers,
            } = full.output_expansion()
            else {
                unreachable!()
            };
            let mut input = Vec::new();
            for seed in [2, 1, 5] {
                let mut values = row(seed, full.input_columns());
                values[full.input_layout().denominator.start] = c(3., 2.);
                input.extend(values);
            }
            let mut expected = vec![c(0., 0.); 3 * full_powers.len()];
            let mut actual = vec![c(0., 0.); 3 * selected_powers.len()];
            full.compile(&0.)
                .unwrap()
                .evaluate_batch(&input, &mut expected, 3)
                .unwrap();
            selected
                .compile(&0.)
                .unwrap()
                .evaluate_batch(&input, &mut actual, 3)
                .unwrap();
            for lane in 0..3 {
                for (column, power) in selected_powers.iter().enumerate() {
                    let original = expected[lane * full_powers.len()
                        + full_powers.iter().position(|p| p == power).unwrap()];
                    let value = actual[lane * selected_powers.len() + column];
                    assert!((value.re - original.re).abs() < 1e-11 * (1. + original.re.abs()));
                    assert!((value.im - original.im).abs() < 1e-11 * (1. + original.im.abs()));
                }
            }
        }
    }
    assert!(
        TriangleBubbleSubtraction::for_quotient(&LaurentDivision::new(3, 3, 3).unwrap()).is_err()
    );
    let empty =
        TriangleBubbleSubtraction::for_quotient(&LaurentDivision::new(5, 2, 2).unwrap()).unwrap();
    assert!(empty.output_expansion().is_empty());
    empty
        .compile(&0.)
        .unwrap()
        .evaluate_batch(&vec![c(1., 0.); empty.input_columns()], &mut [], 1)
        .unwrap();
}
