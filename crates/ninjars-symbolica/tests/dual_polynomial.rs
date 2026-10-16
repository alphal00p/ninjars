// Symbolica's generated Dual division implementations multiply by an inverse.
#![allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]

use ninjars::{
    Complex, ExpansionKind, ExpansionRequest, FourMomentum, Numerator, NumeratorRequest, SampleId,
};
use ninjars_symbolica::{DualPolynomialNumerator, PolynomialNumerator};
use symbolica::{
    create_hyperdual_from_components,
    domains::{
        dual::{DualNumber, HyperDual},
        float::{FloatLike, Real},
    },
    parse,
};

create_hyperdual_from_components!(MixedDual, [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1]]);

fn c(value: f64) -> Complex<f64> {
    Complex::new(value, 0.)
}

fn check_carrier<D: Real + DualNumber<Complex<f64>>>(mut prototype: D) {
    let source = PolynomialNumerator::new(
        parse!("a*q0^3+mu2*q1"),
        [
            parse!("q0"),
            parse!("q1"),
            parse!("q2"),
            parse!("q3"),
            parse!("mu2"),
        ],
        vec![parse!("a")],
        0.,
    )
    .unwrap();
    let mut numerator = DualPolynomialNumerator::new(source, prototype.clone()).unwrap();
    let zero = prototype.zero();
    let mut alpha = zero.clone();
    alpha
        .values_mut()
        .clone_from_slice(&[c(3.), c(0.), c(1.), c(0.), c(0.)]);
    numerator
        .set_parameter_batch(&[alpha.clone(), alpha], 2)
        .unwrap();
    let mut q0 = zero.clone();
    q0.values_mut()
        .clone_from_slice(&[c(2.), c(1.), c(0.), c(0.), c(0.)]);
    let q = FourMomentum::from_args(q0, zero.clone(), zero.clone(), zero.clone());
    let requests: Vec<_> = [1, 0]
        .into_iter()
        .map(|lane| NumeratorRequest {
            id: SampleId { lane, sample: 0 },
            q: &q,
            mu_squared: &zero,
            partition: &[0, 1],
        })
        .collect();
    let mut output = vec![zero.clone(); 2];
    numerator.evaluate_batch(&requests, &mut output).unwrap();
    for value in &output {
        // Symbolica stores Taylor coefficients: the x² slot is d_xx/2!.
        assert_eq!(value.values(), &[c(24.), c(36.), c(8.), c(18.), c(12.)]);
    }
    numerator.evaluate_batch(&[], &mut []).unwrap();
    let vectors = [q];
    let requests = [ExpansionRequest {
        id: SampleId { lane: 1, sample: 0 },
        partition: &[0, 1],
        vectors: &vectors,
        parameters: &[],
    }];
    numerator
        .expansion_batch(
            &ExpansionKind::Mu { powers: vec![3, 2] },
            &requests,
            &mut output,
        )
        .unwrap();
    assert_eq!(output[0].values(), &[c(24.), c(36.), c(8.), c(18.), c(12.)]);
    assert!(output[1].values().iter().all(FloatLike::is_fully_zero));
    // A derivative-only perturbation must be propagated, despite equal primals.
    prototype
        .values_mut()
        .clone_from_slice(&[c(3.), c(4.), c(1.), c(0.), c(0.)]);
    numerator.set_parameter_batch(&[prototype], 1).unwrap();
    let requests = [ExpansionRequest {
        id: SampleId { lane: 0, sample: 0 },
        partition: &[0, 1],
        vectors: &vectors,
        parameters: &[],
    }];
    numerator
        .expansion_batch(
            &ExpansionKind::Mu { powers: vec![3] },
            &requests,
            &mut output[..1],
        )
        .unwrap();
    assert_eq!(output[0].values(), &[c(24.), c(68.), c(8.), c(66.), c(12.)]);
}

#[test]
fn static_and_dynamic_duals_preserve_higher_mixed_components() {
    check_carrier(MixedDual { values: [c(0.); 5] });
    check_carrier(HyperDual::from_values(
        vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]],
        vec![c(0.); 5],
    ));
}

#[test]
fn same_length_different_dual_shapes_are_rejected() {
    let a = HyperDual::from_values(vec![vec![0, 0], vec![1, 0], vec![0, 1]], vec![c(0.); 3]);
    let b = HyperDual::from_values(vec![vec![0, 0], vec![0, 1], vec![1, 0]], vec![c(0.); 3]);
    let source = PolynomialNumerator::new(
        parse!("q0+a"),
        [
            parse!("q0"),
            parse!("q1"),
            parse!("q2"),
            parse!("q3"),
            parse!("mu2"),
        ],
        vec![parse!("a")],
        0.,
    )
    .unwrap();
    let mut numerator = DualPolynomialNumerator::new(source, a).unwrap();
    assert!(numerator.set_parameter_batch(&[b], 1).is_err());
}
