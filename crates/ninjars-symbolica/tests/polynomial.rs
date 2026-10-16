use ninjars::{
    Complex, ExpansionKind, ExpansionRequest, FourMomentum, Numerator, NumeratorRequest, SampleId,
};
use ninjars_symbolica::PolynomialNumerator;
use symbolica::domains::float::{DoubleFloat, Float, Real, RealLike, SingleFloat};
use symbolica::evaluate::EvaluationDomain;
use symbolica::{atom::Atom, parse};

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
fn vector(values: [f64; 4]) -> FourMomentum<Complex<f64>> {
    FourMomentum::from_args(c(values[0]), c(values[1]), c(values[2]), c(values[3]))
}

#[test]
fn direct_batch_preserves_lane_order_and_rejects_nonpolynomials() {
    let mut numerator = PolynomialNumerator::new(
        parse!("a*q0^2+(q1+2*q2)/3+mu2^2"),
        variables(),
        vec![parse!("a")],
        0.,
    )
    .unwrap();
    assert_eq!(numerator.rank(), 4);
    numerator.set_parameter_batch(&[c(2.), c(5.)], 2).unwrap();
    let q = vector([3., 1., 4., 2.]);
    let mu2 = c(2.);
    let requests: Vec<_> = [1, 0, 1]
        .into_iter()
        .enumerate()
        .map(|(sample, lane)| NumeratorRequest {
            id: SampleId { lane, sample },
            q: &q,
            mu_squared: &mu2,
            partition: &[0, 1],
        })
        .collect();
    let mut out = [c(0.); 3];
    numerator.evaluate_batch(&requests, &mut out).unwrap();
    assert_eq!(out, [c(52.), c(25.), c(52.)]);
    numerator.evaluate_batch(&[], &mut []).unwrap();
    let invalid = [NumeratorRequest {
        id: SampleId { lane: 2, sample: 0 },
        q: &q,
        mu_squared: &mu2,
        partition: &[],
    }];
    let mut unchanged = [c(17.)];
    assert!(numerator.evaluate_batch(&invalid, &mut unchanged).is_err());
    assert_eq!(unchanged, [c(17.)]);
    for expression in [parse!("1/q0"), parse!("sin(q1)"), parse!("sqrt(mu2)")] {
        assert!(PolynomialNumerator::new(expression, variables(), vec![], 0.).is_err());
    }
}

#[test]
fn triangle_laurent_coefficients_include_negative_powers_and_mu_weights() {
    let mut numerator = PolynomialNumerator::new(
        parse!("a*q0^2+mu2*q1+mu2^2"),
        variables(),
        vec![parse!("a")],
        0.,
    )
    .unwrap();
    numerator.set_parameter_batch(&[c(1.), c(2.)], 2).unwrap();
    let vectors = [
        vector([1., 0., 0., 0.]),
        vector([2., 1., 0., 0.]),
        vector([3., 4., 0., 0.]),
    ];
    let parameters = [c(5.)];
    let requests: Vec<_> = [1, 0]
        .into_iter()
        .map(|lane| ExpansionRequest {
            id: SampleId { lane, sample: 0 },
            partition: &[0, 1, 2],
            vectors: &vectors,
            parameters: &parameters,
        })
        .collect();
    let kind = ExpansionKind::T3 {
        powers: vec![
            (2, 0),
            (1, 0),
            (1, 1),
            (0, 0),
            (0, 1),
            (0, 2),
            (-1, 0),
            (-1, 1),
            (-1, 2),
            (-2, 0),
            (-2, 1),
            (-2, 2),
        ],
    };
    let mut out = vec![c(0.); 24];
    numerator
        .expansion_batch(&kind, &requests, &mut out)
        .unwrap();
    for (lane, alpha) in [2., 1.].into_iter().enumerate() {
        let expected = [
            4. * alpha,
            4. * alpha,
            1.,
            61. * alpha,
            12. * alpha,
            1.,
            30. * alpha,
            6. * alpha + 20.,
            4.,
            225. * alpha,
            90. * alpha,
            9. * alpha,
        ];
        for (actual, expected) in out[lane * 12..(lane + 1) * 12].iter().zip(expected) {
            assert_eq!(*actual, c(expected));
        }
    }
    let mut repeated = [c(0.); 12];
    numerator
        .expansion_batch(&kind, &requests[..1], &mut repeated)
        .unwrap();
    assert_eq!(repeated, out[..12]);
}

#[test]
fn mu_expansion_uses_bilinear_metric_and_optional_offset() {
    let mut numerator =
        PolynomialNumerator::new(parse!("q0^2+mu2"), variables(), vec![], 0.).unwrap();
    numerator.set_parameter_batch(&[], 1).unwrap();
    let vectors = [vector([2., 1., 0., 0.]), vector([3., 0., 0., 0.])];
    let requests = [
        ExpansionRequest {
            id: SampleId { lane: 0, sample: 0 },
            partition: &[],
            vectors: &vectors,
            parameters: &[],
        },
        ExpansionRequest {
            id: SampleId { lane: 0, sample: 1 },
            partition: &[],
            vectors: &vectors[..1],
            parameters: &[],
        },
    ];
    let kind = ExpansionKind::Mu {
        powers: vec![2, 1, 0],
    };
    let mut out = [c(0.); 6];
    numerator
        .expansion_batch(&kind, &requests, &mut out)
        .unwrap();
    assert_eq!(out, [c(7.), c(12.), c(9.), c(7.), c(0.), c(0.)]);
}

#[test]
fn bubble_laurent_coefficients_reconstruct_direct_numerator() {
    let mut numerator =
        PolynomialNumerator::new(parse!("q0^2+mu2*q1+mu2^2"), variables(), vec![], 0.).unwrap();
    numerator.set_parameter_batch(&[], 1).unwrap();
    let vectors = [
        vector([1., 1., 0., 0.]),
        vector([2., 3., 0., 0.]),
        vector([3., 2., 0., 0.]),
        vector([4., 1., 0., 0.]),
    ];
    let parameters = [c(2.), c(3.), c(5.)];
    let request = [ExpansionRequest {
        id: SampleId { lane: 0, sample: 0 },
        partition: &[0, 1],
        vectors: &vectors,
        parameters: &parameters,
    }];
    let powers: Vec<_> = (-2..=2)
        .flat_map(|t| (0..=4).flat_map(move |x| (0..=2).map(move |mu| (t, x, mu))))
        .collect();
    let kind = ExpansionKind::T2 {
        powers: powers.clone(),
    };
    let mut out = vec![c(0.); powers.len()];
    numerator
        .expansion_batch(&kind, &request, &mut out)
        .unwrap();
    for (t, x, mu) in [(2.0_f64, 3.0_f64, 1.0_f64), (-1., 2., 3.)] {
        let reconstructed: f64 = powers
            .iter()
            .zip(&out)
            .map(|(&(pt, px, pm), value)| {
                value.re * t.powi(pt) * x.powi(i32::from(px)) * mu.powi(i32::from(pm))
            })
            .sum();
        let q0 = 1. + 2. * x + 3. * t + 4. * (2. + 3. * x + 5. * x * x + mu) / t;
        let q1 = 1. + 3. * x + 2. * t + (2. + 3. * x + 5. * x * x + mu) / t;
        assert!((reconstructed - (q0 * q0 + mu * q1 + mu * mu)).abs() < 1e-9);
    }
}

fn precision_probe<T: Real + RealLike + SingleFloat>(
    prototype: T,
    expression: Atom,
    expected: f64,
    relative: f64,
) where
    Complex<T>: Real + EvaluationDomain,
{
    let mut numerator =
        PolynomialNumerator::new(expression, variables(), vec![], prototype.clone()).unwrap();
    numerator.set_parameter_batch(&[], 2).unwrap();
    let zero = Complex::new(prototype.zero(), prototype.zero());
    let q = FourMomentum::from_args(
        Complex::new(prototype.from_i64(-1), prototype.zero()),
        zero.clone(),
        zero.clone(),
        zero.clone(),
    );
    let requests: Vec<_> = (0..2)
        .map(|lane| NumeratorRequest {
            id: SampleId { lane, sample: 0 },
            q: &q,
            mu_squared: &zero,
            partition: &[],
        })
        .collect();
    let mut output = vec![zero.clone(); 2];
    numerator.evaluate_batch(&requests, &mut output).unwrap();
    for value in output {
        assert!(value.im.is_fully_zero());
        assert!(
            (value.re.to_f64() / expected - 1.).abs() < relative,
            "{value}"
        );
    }
}

#[test]
fn coefficient_conversion_preserves_precision_beyond_f64() {
    precision_probe(
        DoubleFloat::from(0.),
        parse!("q0+1+1/10000000000000000000000000"),
        1e-25,
        1e-7,
    );
    precision_probe(
        Float::new(256),
        parse!("q0+1+1/1000000000000000000000000000000000000000000000000000000000000"),
        1e-60,
        1e-14,
    );
}
