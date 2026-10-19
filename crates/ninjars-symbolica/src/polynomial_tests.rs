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

/// The previous preparation: substitute the cut parameterization into the
/// whole numerator, expand in t, x and mu², then extract coefficients. Its
/// atoms define the evaluators, so the truncated expansion must reproduce
/// them exactly (equal atoms give equal evaluators and bitwise results).
fn full_expansion(
    numerator: &PolynomialNumerator<f64>,
    kind: &ExpansionKind,
    symbols: &[Atom],
) -> Vec<Atom> {
    let (t, x, mu2) = (&symbols[0], &symbols[1], &symbols[2]);
    let vector = |index: usize| {
        FourMomentum::from_args(
            symbols[3 + 4 * index].clone(),
            symbols[4 + 4 * index].clone(),
            symbols[5 + 4 * index].clone(),
            symbols[6 + 4 * index].clone(),
        )
    };
    let (q, mu) = match kind {
        ExpansionKind::Mu { .. } => {
            let a = vector(0);
            let mu = t.pow(2) * a.dot(&a);
            (a * t.clone() + vector(1), mu)
        }
        ExpansionKind::MuShifted { .. } => {
            let a = vector(0);
            let mu = t.pow(2) * a.dot(&a) - &symbols[19];
            (a * t.clone() + vector(1), mu)
        }
        ExpansionKind::T3 { .. } => (
            vector(0) + vector(1) * t.clone() + vector(2) * ((&symbols[19] + mu2) / t),
            mu2.clone(),
        ),
        ExpansionKind::T2 { .. } => (
            vector(0)
                + vector(1) * x.clone()
                + vector(2) * t.clone()
                + vector(3)
                    * ((&symbols[19] + &symbols[20] * x + &symbols[21] * x.pow(2) + mu2) / t),
            mu2.clone(),
        ),
    };
    let replacements: Vec<_> = numerator
        .variables()
        .iter()
        .zip(q.components().into_iter().chain([&mu]))
        .map(|(from, to)| {
            Replacement::new(Pattern::Literal(from.clone()), Pattern::Literal(to.clone()))
        })
        .collect();
    let polynomial = numerator.polynomial();
    let protected = polynomial.map_coeff(
        |coefficient| {
            let index = polynomial
                .coefficients
                .iter()
                .position(|candidate| candidate == coefficient)
                .unwrap();
            symbols[22 + index].clone()
        },
        exact_field(),
    );
    let substituted = protected
        .flatten(false)
        .replace_multiple(&replacements)
        .expand_in(t)
        .expand_in(x)
        .expand_in(mu2);
    let expanded = substituted.to_polynomial_in_vars_with_field::<i32>(
        &[t.clone(), x.clone(), mu2.clone()],
        &exact_field(),
    );
    let powers: Vec<[i32; 3]> = match kind {
        ExpansionKind::Mu { powers } | ExpansionKind::MuShifted { powers } => {
            powers.iter().map(|&a| [a, 0, 0]).collect()
        }
        ExpansionKind::T3 { powers } => powers.iter().map(|&(a, c)| [a, 0, i32::from(c)]).collect(),
        ExpansionKind::T2 { powers } => powers
            .iter()
            .map(|&(a, b, c)| [a, i32::from(b), i32::from(c)])
            .collect(),
    };
    powers
        .iter()
        .map(|power| expanded.coefficient(power).unwrap_or(Atom::Zero))
        .collect()
}

/// Expansion kinds requested by the reduction (Laurent divisions of every cut
/// size for several propagator counts, rank n+2 included, and the box mu
/// expansions, the shifted one of rank n+2 boxes included) plus wide
/// ranges with negative powers.
fn requested_kinds(rank: usize) -> Vec<ExpansionKind> {
    let r = rank as i32;
    let mut kinds = vec![
        ExpansionKind::Mu { powers: vec![r] },
        ExpansionKind::Mu {
            powers: vec![r, r - 1],
        },
        ExpansionKind::MuShifted {
            powers: (0..=r).rev().collect(),
        },
        ExpansionKind::T3 {
            powers: (-3..=r)
                .flat_map(|t| (0..=3).map(move |m| (t, m)))
                .collect(),
        },
        ExpansionKind::T2 {
            powers: (-2..=r)
                .flat_map(|t| (0..=4).flat_map(move |x| (0..=2).map(move |m| (t, x, m))))
                .collect(),
        },
    ];
    for propagators in rank.saturating_sub(2).max(1)..=rank + 4 {
        for cut in 1..=3.min(propagators) {
            if let Ok(division) = crate::LaurentDivision::new(propagators, rank, cut) {
                kinds.push(division.numerator_expansion().clone());
            }
        }
    }
    kinds
}

#[test]
fn truncated_expansion_reproduces_full_expansion_atoms() {
    // Dense through rank 4 in q and mu², with repeated, rational, imaginary
    // and parameter-dependent coefficients.
    let q = variables();
    let [a, b] = [parse!("factoring_test::a"), parse!("factoring_test::b")];
    let coefficients = [
        Atom::num(1),
        a.clone(),
        Atom::num(1) / 3,
        &a * &b - Atom::i(),
        Atom::num(1),
        (&a + &b).pow(2),
    ];
    let mut dense = Atom::Zero;
    let mut count = 0;
    for e0 in 0..=4u32 {
        for e1 in 0..=4 - e0 {
            for e2 in 0..=4 - e0 - e1 {
                for e3 in 0..=4 - e0 - e1 - e2 {
                    for m in 0..=(4 - e0 - e1 - e2 - e3) / 2 {
                        let monomial = q[0].pow(e0 as i64)
                            * q[1].pow(e1 as i64)
                            * q[2].pow(e2 as i64)
                            * q[3].pow(e3 as i64)
                            * q[4].pow(m as i64);
                        dense += &coefficients[count % coefficients.len()] * monomial;
                        count += 1;
                    }
                }
            }
        }
    }
    let sparse_rank7 = parse!(
        "factoring_test::a*factoring_test::q0^3*factoring_test::q1^2*factoring_test::q3^2+factoring_test::q2^7+3*factoring_test::mu2^2*factoring_test::q0*factoring_test::q3^2+factoring_test::b*factoring_test::mu2^3*factoring_test::q1+factoring_test::q0*factoring_test::q1*factoring_test::q2*factoring_test::q3+factoring_test::a"
    );
    let numerators = [
        (expression(), 5),
        (dense, 4),
        (sparse_rank7, 7),
        (Atom::num(3) / 2, 0),
        (
            parse!("factoring_test::a*factoring_test::q1+factoring_test::mu2"),
            2,
        ),
    ];
    for (expression, rank) in numerators {
        let numerator =
            PolynomialNumerator::new(expression, variables(), parameters(), 0_f64).unwrap();
        assert_eq!(numerator.rank(), rank);
        let symbols = numerator.fresh_symbols();
        for kind in requested_kinds(rank) {
            assert_eq!(
                numerator.laurent_coefficients(&kind, &symbols),
                full_expansion(&numerator, &kind, &symbols),
                "rank {rank} {kind:?}"
            );
        }
    }
}

/// A sparse rank-12 numerator took about 900 s to prepare: the full expansion
/// of a rank-12 double-cut substitution has ~10^5 terms, of which the
/// reduction keeps a handful of leading powers.
#[test]
fn rank_twelve_expansions_prepare_quickly() {
    let [q0, q1, q2, q3, mu] = variables();
    let [a, b] = [parse!("factoring_test::a"), parse!("factoring_test::b")];
    let monomials = [
        q0.pow(12),
        q3.pow(12),
        q1.pow(4) * q2.pow(5) * q3.pow(3),
        q0.pow(3) * q1.pow(3) * q2.pow(3) * q3.pow(3),
        q0.pow(2) * q1.pow(2) * mu.pow(4),
        q2.pow(7) * q3.pow(1) * mu.pow(2),
        q0.pow(1) * q1.pow(5) * q2.pow(1) * q3.pow(2) * mu.pow(1),
        mu.pow(6),
        q0.pow(5) * q2.pow(4),
        q1.pow(3) * q3.pow(2) * mu.pow(2),
        q0.pow(1) * q1.pow(1) * q2.pow(1) * q3.pow(1),
        q2.pow(2) * mu.pow(1),
        Atom::num(1),
    ];
    let expression = monomials
        .into_iter()
        .enumerate()
        .fold(Atom::Zero, |sum, (k, monomial)| {
            let coefficient = if k % 2 == 0 {
                &a + Atom::num(k as i64)
            } else {
                &b * Atom::num(k as i64)
            };
            sum + coefficient * monomial
        });
    let numerator = PolynomialNumerator::new(expression, variables(), parameters(), 0_f64).unwrap();
    assert_eq!(numerator.rank(), 12);
    let start = std::time::Instant::now();
    numerator
        .exact_expansion(&ExpansionKind::Mu {
            powers: vec![12, 11],
        })
        .unwrap();
    for propagators in [11, 12] {
        for cut in 1..=3 {
            let division = crate::LaurentDivision::new(propagators, 12, cut).unwrap();
            numerator
                .exact_expansion(division.numerator_expansion())
                .unwrap();
        }
    }
    let elapsed = start.elapsed();
    assert!(elapsed.as_secs() < 20, "{elapsed:?}");
}
