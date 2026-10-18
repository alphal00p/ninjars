//! Independent full-expansion controls for the coefficient-preserving path.
use super::*;
use crate::DualPolynomialNumerator;
use symbolica::{
    domains::{
        dual::{DualNumber, HyperDual},
        float::FloatLike,
    },
    id::{Pattern, Replacement},
    parse,
};

fn variables() -> [Atom; 5] {
    [
        parse!("factoring_test::q0"),
        parse!("factoring_test::q1"),
        parse!("factoring_test::q2"),
        parse!("factoring_test::q3"),
        parse!("factoring_test::mu2"),
    ]
}
fn parameters() -> Vec<Atom> {
    vec![parse!("factoring_test::a"), parse!("factoring_test::b")]
}
fn expression() -> Atom {
    parse!(
        "(factoring_test::a+factoring_test::b)^6*(factoring_test::q0+factoring_test::q1)^4+(factoring_test::a-factoring_test::b)^3*factoring_test::q2^5+(factoring_test::a*factoring_test::b+1)^3*factoring_test::mu2*factoring_test::q3^2+(factoring_test::a+2*factoring_test::b)^4*factoring_test::mu2^2"
    ) + Atom::i()
        * parse!("(factoring_test::a+3*factoring_test::b)*factoring_test::q1*factoring_test::mu2")
        / 7
        + parse!("(factoring_test::a-factoring_test::b)*factoring_test::q3^2") / 11
}
fn vectors() -> [FourMomentum<Atom>; 4] {
    [[1, 2, 3, 4], [2, 1, -1, 0], [1, -1, 2, 1], [2, 1, 1, -1]].map(|v| {
        FourMomentum::from_args(
            Atom::num(v[0]),
            Atom::num(v[1]),
            Atom::num(v[2]),
            Atom::num(v[3]),
        )
    })
}
fn scalars() -> [Atom; 3] {
    [Atom::num(2), Atom::num(-3), Atom::one()]
}

/// The deliberately full Atom expansion is an independent small oracle. All
/// polynomial operations and coefficient extraction still belong to Symbolica.
fn oracle(kind: &ExpansionKind) -> (Vec<Atom>, Vec<Atom>) {
    let t = parse!("factoring_oracle::t");
    let x = parse!("factoring_oracle::x");
    let mu = parse!("factoring_oracle::mu");
    let [a, b, c, d] = vectors();
    let [p0, p1, p2] = scalars();
    let (q, mu_value, input) = match kind {
        ExpansionKind::Mu { .. } => {
            let value = &t * &t * a.dot(&a);
            let input = a
                .components()
                .into_iter()
                .chain(b.components())
                .cloned()
                .collect::<Vec<_>>();
            (a * t.clone() + b, value, input)
        }
        // Rank n+2 boxes: the exact d-dimensional cut, mu² = t² a² - beta.
        ExpansionKind::MuShifted { .. } => {
            let value = &t * &t * a.dot(&a) - &p0;
            let input = a
                .components()
                .into_iter()
                .chain(b.components())
                .cloned()
                .chain([p0.clone()])
                .collect::<Vec<_>>();
            (a * t.clone() + b, value, input)
        }
        ExpansionKind::T3 { .. } => {
            let input = a
                .components()
                .into_iter()
                .chain(b.components())
                .chain(c.components())
                .cloned()
                .chain([p0.clone()])
                .collect::<Vec<_>>();
            (a + b * t.clone() + c * ((p0 + &mu) / &t), mu.clone(), input)
        }
        ExpansionKind::T2 { .. } => {
            let input = a
                .components()
                .into_iter()
                .chain(b.components())
                .chain(c.components())
                .chain(d.components())
                .cloned()
                .chain([p0.clone(), p1.clone(), p2.clone()])
                .collect::<Vec<_>>();
            (
                a + b * x.clone() + c * t.clone() + d * ((p0 + p1 * &x + p2 * x.pow(2) + &mu) / &t),
                mu.clone(),
                input,
            )
        }
    };
    let replacements = variables()
        .iter()
        .zip(q.components().into_iter().chain([&mu_value]))
        .map(|(a, b)| Replacement::new(Pattern::Literal(a.clone()), Pattern::Literal(b.clone())))
        .collect::<Vec<_>>();
    let full = expression().replace_multiple(&replacements).expand();
    let polynomial = full.to_polynomial_in_vars_with_field::<i32>(&[t, x, mu], &exact_field());
    let powers = match kind {
        ExpansionKind::Mu { powers } | ExpansionKind::MuShifted { powers } => {
            powers.iter().map(|&t| [t, 0, 0]).collect::<Vec<_>>()
        }
        ExpansionKind::T3 { powers } => powers.iter().map(|&(t, m)| [t, 0, i32::from(m)]).collect(),
        ExpansionKind::T2 { powers } => powers
            .iter()
            .map(|&(t, x, m)| [t, i32::from(x), i32::from(m)])
            .collect(),
    };
    (
        powers
            .iter()
            .map(|p| polynomial.coefficient(p).unwrap_or(Atom::Zero))
            .collect(),
        input.into_iter().chain(parameters()).collect(),
    )
}

#[test]
fn coefficient_preserving_callbacks_equal_full_expansion_exactly() {
    let numerator =
        PolynomialNumerator::new(expression(), variables(), parameters(), 0_f64).unwrap();
    assert_eq!(numerator.rank(), 5);
    let kinds = [
        ExpansionKind::Mu {
            powers: (0..=5).collect(),
        },
        ExpansionKind::MuShifted {
            powers: (0..=5).rev().collect(),
        },
        ExpansionKind::T3 {
            powers: (-5..=5)
                .flat_map(|t| (0..=5).map(move |m| (t, m)))
                .collect(),
        },
        ExpansionKind::T2 {
            powers: (-5..=5)
                .flat_map(|t| (0..=10).flat_map(move |x| (0..=5).map(move |m| (t, x, m))))
                .collect(),
        },
    ];
    for kind in kinds {
        let (expected, inputs) = oracle(&kind);
        let mut graph = numerator
            .exact_expansion(&kind)
            .unwrap()
            .map_to_ring(&exact_field())
            .unwrap();
        let mut actual = vec![Atom::Zero; expected.len()];
        graph.evaluate_in_ring(&inputs, &mut actual, &exact_field());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).expand().is_zero(),
                "{kind:?} coefficient {index}"
            );
        }
    }
}

#[test]
fn factored_external_coefficients_preserve_mixed_dual_lanes() {
    let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]];
    let zero = Complex::new(0_f64, 0.);
    let prototype = HyperDual::from_values(shape, vec![zero; 5]);
    let source = PolynomialNumerator::new(expression(), variables(), parameters(), 0_f64).unwrap();
    let mut numerator = DualPolynomialNumerator::new(source, prototype.clone()).unwrap();
    let constant = |value: f64| {
        let mut v = prototype.zero();
        v.values_mut()[0] = Complex::new(value, 0.);
        v
    };
    let points = [(2., 3.), (2., -2.), (-1., 1.)];
    let values = points
        .iter()
        .flat_map(|&(a, b)| {
            let mut aa = constant(a);
            aa.values_mut()[1] = Complex::new(1., 0.);
            let mut bb = constant(b);
            bb.values_mut()[2] = Complex::new(1., 0.);
            [aa, bb]
        })
        .collect::<Vec<_>>();
    numerator.set_parameter_batch(&values, 3).unwrap();
    let vector_values = [
        [1., 2., 3., 4.],
        [2., 1., -1., 0.],
        [1., -1., 2., 1.],
        [2., 1., 1., -1.],
    ]
    .map(|v| {
        FourMomentum::from_args(
            constant(v[0]),
            constant(v[1]),
            constant(v[2]),
            constant(v[3]),
        )
    });
    let scalar_values = [constant(2.), constant(-3.), constant(1.)];
    let kind = ExpansionKind::T2 {
        powers: vec![
            (5, 0, 0),
            (4, 1, 0),
            (3, 2, 0),
            (0, 0, 2),
            (-3, 0, 3),
            (-5, 10, 0),
            (0, 3, 0),
        ],
    };
    let lanes = [2, 0, 1, 2];
    let requests = lanes.map(|lane| ExpansionRequest {
        id: ninjars::SampleId { lane, sample: 0 },
        partition: &[0, 1],
        vectors: &vector_values,
        parameters: &scalar_values,
    });
    let mut output = vec![prototype.clone(); requests.len() * kind.len()];
    numerator
        .expansion_batch(&kind, &requests, &mut output)
        .unwrap();
    let (coefficients, _) = oracle(&kind);
    for (row, lane) in lanes.into_iter().enumerate() {
        let bindings = parameters()
            .into_iter()
            .zip([
                Complex::new(points[lane].0, 0.),
                Complex::new(points[lane].1, 0.),
            ])
            .collect::<HashMap<_, _>>();
        for (index, coefficient) in coefficients.iter().enumerate() {
            let a = symbol!("factoring_test::a");
            let b = symbol!("factoring_test::b");
            let expected = [
                coefficient.clone(),
                coefficient.derivative(a),
                coefficient.derivative(b),
                coefficient.derivative(a).derivative(a) / 2,
                coefficient.derivative(a).derivative(b),
            ];
            for (component, expected) in expected.into_iter().enumerate() {
                let expected = expected.evaluate_with_prec(&bindings, 53).unwrap();
                let actual = &output[row * kind.len() + index].values()[component];
                let error = (actual.re - expected.re).hypot(actual.im - expected.im)
                    / 1_f64.max(expected.re.hypot(expected.im));
                assert!(
                    error <= 1e-9,
                    "lane{lane} coefficient{index} component{component} error{error:e}"
                );
            }
        }
    }
    numerator.expansion_batch(&kind, &[], &mut []).unwrap();
}
