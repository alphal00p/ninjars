use std::collections::HashMap;

use ninjars::{Complex, FloatLike, FourMomentum, Real};
use ninjars_symbolica::{TadpoleGeometryProbe, TadpoleSourceProjection};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::dual::{DualNumber, HyperDual},
};

fn value(i: usize) -> Complex<f64> {
    Complex::new((i % 19) as f64 / 4. - 2., (i % 13) as f64 / 8. - 1.)
}
fn momentum<V: Clone>(row: &[V], start: usize) -> FourMomentum<V> {
    FourMomentum::from_args(
        row[start].clone(),
        row[start + 1].clone(),
        row[start + 2].clone(),
        row[start + 3].clone(),
    )
}
fn source_control<V: Real>(row: &[V]) -> Vec<V> {
    let a = momentum(row, 0) + momentum(row, 12);
    (16..row.len())
        .step_by(4)
        .flat_map(|offset| {
            let axis = momentum(row, offset);
            [
                a.dot(&axis),
                momentum(row, 4).dot(&axis),
                momentum(row, 8).dot(&axis),
            ]
        })
        .collect()
}
fn near(a: &Complex<f64>, b: &Complex<f64>) {
    assert!(
        (a.re - b.re).abs() < 1e-12 * (1. + b.re.abs()),
        "{a} != {b}"
    );
    assert!(
        (a.im - b.im).abs() < 1e-12 * (1. + b.im.abs()),
        "{a} != {b}"
    );
}

// Independently assemble the historical reference choices using FeynKit's
// public Weyl API, then contract their numerical values with shared momenta.
fn reference_axes() -> Vec<FourMomentum<Complex<f64>>> {
    let mut axes = Vec::new();
    for (a, b, c, d) in [
        (1, 2, -2, 1),
        (2, -1, 1, 3),
        (-1, 3, 2, 1),
        (3, 1, -1, -2),
        (1, -3, 3, -2),
        (2, 3, -3, 1),
    ] {
        let angle1 = [Atom::num(1), Atom::num(a) + Atom::i() * b];
        let square1 = [Atom::num(1), Atom::num(a) - Atom::i() * b];
        let angle2 = [Atom::num(1), Atom::num(c) + Atom::i() * d];
        let square2 = [Atom::num(1), Atom::num(c) - Atom::i() * d];
        for (angle, square) in [
            (&angle1, &square1),
            (&angle2, &square2),
            (&angle1, &square2),
            (&angle2, &square1),
        ] {
            let exact = FourMomentum::from_weyl_spinors(angle, square);
            let p = exact.components().map(|component| {
                component
                    .evaluate(&HashMap::<Atom, Complex<f64>>::new())
                    .unwrap()
            });
            axes.push(FourMomentum::from_args(p[0], p[1], p[2], p[3]));
        }
    }
    axes
}

#[test]
fn source_projection_preserves_parent_origins_order_and_mixed_duals() {
    let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]];
    let prototype = HyperDual::from_values(shape.clone(), vec![value(0).zero(); 5]);
    for source in [3, 2] {
        let plan = TadpoleSourceProjection::new(source).unwrap();
        assert_eq!(plan.source_size(), source);
        let mut evaluator = plan.compile(&0.).unwrap();
        let mut dual_evaluator = plan.compile_dual(&0., &prototype).unwrap();
        for rows in [0, 1, 8, 3, 1] {
            let input: Vec<_> = (0..rows * plan.input_columns()).map(value).collect();
            let mut actual = vec![value(0).zero(); rows * plan.output_columns()];
            evaluator.evaluate_batch(&input, &mut actual, rows).unwrap();
            let expected: Vec<_> = input
                .chunks_exact(plan.input_columns())
                .flat_map(source_control)
                .collect();
            for (a, b) in actual.iter().zip(&expected) {
                near(a, b);
            }
            let dual_input: Vec<_> = (0..input.len())
                .map(|i| {
                    HyperDual::from_values(
                        shape.clone(),
                        (0..5).map(|j| value(i * 5 + j)).collect(),
                    )
                })
                .collect();
            let mut dual_actual = vec![prototype.zero(); rows * plan.output_columns()];
            dual_evaluator
                .evaluate_batch(&dual_input, &mut dual_actual, rows)
                .unwrap();
            let dual_expected: Vec<_> = dual_input
                .chunks_exact(plan.input_columns())
                .flat_map(source_control)
                .collect();
            for (a, b) in dual_actual.iter().zip(&dual_expected) {
                for (a, b) in a.values().iter().zip(b.values()) {
                    near(a, b);
                }
            }
        }
    }
    for invalid in [0, 1, 4, usize::MAX] {
        assert!(TadpoleSourceProjection::new(invalid).is_err());
    }
}

#[test]
fn geometry_probe_preserves_all_six_reference_candidates_and_derivatives() {
    let plan = TadpoleGeometryProbe::new();
    assert_eq!((plan.input_columns(), plan.output_columns()), (8, 28));
    let axes = reference_axes();
    let mut evaluator = plan.compile(&0.).unwrap();
    let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]];
    let prototype = HyperDual::from_values(shape.clone(), vec![value(0).zero(); 5]);
    let mut dual_evaluator = plan.compile_dual(&0., &prototype).unwrap();
    for rows in [0, 1, 8, 3, 1] {
        let input: Vec<_> = (0..rows * 8).map(value).collect();
        let mut actual = vec![value(0); rows * 28];
        evaluator.evaluate_batch(&input, &mut actual, rows).unwrap();
        for (row, actual) in input
            .as_chunks::<8>()
            .0
            .iter()
            .zip(actual.as_chunks::<28>().0)
        {
            let difference = momentum(row, 4) - momentum(row, 0);
            let expected: Vec<_> = difference
                .components()
                .into_iter()
                .cloned()
                .chain(axes.iter().map(|axis| difference.dot(axis)))
                .collect();
            for (a, b) in actual.iter().zip(&expected) {
                near(a, b);
            }
        }
        let input: Vec<_> = (0..rows * 8)
            .map(|i| {
                HyperDual::from_values(shape.clone(), (0..5).map(|j| value(i * 5 + j)).collect())
            })
            .collect();
        let mut actual = vec![prototype.zero(); rows * 28];
        dual_evaluator
            .evaluate_batch(&input, &mut actual, rows)
            .unwrap();
        // The probe is linear: each Taylor component obeys the same independent
        // numerical difference/contraction, including the higher/mixed entries.
        for (row, actual) in input
            .as_chunks::<8>()
            .0
            .iter()
            .zip(actual.as_chunks::<28>().0)
        {
            for component in 0..5 {
                let scalar: Vec<_> = row.iter().map(|v| v.values()[component]).collect();
                let difference = momentum(&scalar, 4) - momentum(&scalar, 0);
                let expected: Vec<_> = difference
                    .components()
                    .into_iter()
                    .cloned()
                    .chain(axes.iter().map(|axis| difference.dot(axis)))
                    .collect();
                for (a, b) in actual.iter().zip(&expected) {
                    near(&a.values()[component], b);
                }
            }
        }
    }
}
