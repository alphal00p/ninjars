//! Independent ultraviolet-pole control; all differentiation is Symbolica's.
use std::{collections::HashMap, fs};

use serde_json::{Value, json};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        atom::AtomField,
        float::{Complex, Float},
    },
    parse,
    parser::ParseSettings,
    symbol,
};

// Four-dimensional +--- Laplacian. For a quartic numerator, rotational
// averaging gives pole = Delta_q^2 N / [8 D(D+2)] = Delta_q^2 N / 192.
// Equivalently, differentiating oneloopmaster's A0000 pole (m²)^3/24
// three times and dividing by 3! gives the metric-pair coefficient 1/24;
// contracting the symmetric quartic tensor gives Delta_q^2 N / 8.
fn laplacian(expression: &Atom) -> Atom {
    let q = [
        symbol!("ninjars::q0"),
        symbol!("ninjars::q1"),
        symbol!("ninjars::q2"),
        symbol!("ninjars::q3"),
    ];
    expression.derivative(q[0]).derivative(q[0])
        - expression.derivative(q[1]).derivative(q[1])
        - expression.derivative(q[2]).derivative(q[2])
        - expression.derivative(q[3]).derivative(q[3])
}

fn within_rank(numerator: &Atom, maximum_rank: usize) -> bool {
    let variables = [
        parse!("ninjars::q0"),
        parse!("ninjars::q1"),
        parse!("ninjars::q2"),
        parse!("ninjars::q3"),
    ];
    let polynomial = numerator.to_polynomial_in_vars_with_field::<u16>(
        &variables,
        &AtomField {
            statistical_zero_test: false,
            ..AtomField::new()
        },
    );
    (&polynomial).into_iter().all(|term| {
        term.exponents
            .iter()
            .map(|&power| usize::from(power))
            .sum::<usize>()
            <= maximum_rank
            && variables.iter().all(|v| !term.coefficient.contains(v))
    })
}

fn main() {
    // Harmonic higher-degree terms have zero Laplacians but lie outside these
    // pole formulas. Degree validation must reject them before differentiation.
    for (maximum_rank, numerator) in [
        (2, parse!("(ninjars::q0+ninjars::q1)^3")),
        (4, parse!("(ninjars::q0+ninjars::q1)^5")),
    ] {
        assert!(laplacian(&numerator).expand().is_zero());
        assert!(!within_rank(&numerator, maximum_rank));
    }
    for (numerator, expected) in [
        (
            parse!("(ninjars::q0^2-ninjars::q1^2-ninjars::q2^2-ninjars::q3^2)^2"),
            Atom::one(),
        ),
        (parse!("ninjars::q0^4"), Atom::one() / 8),
        (parse!("ninjars::q0^2*ninjars::q1^2"), -Atom::one() / 24),
    ] {
        assert!(
            (laplacian(&laplacian(&numerator)) / 192 - expected)
                .expand()
                .is_zero()
        );
    }
    for (numerator, expected) in [
        (
            parse!("ninjars::q0^2-ninjars::q1^2-ninjars::q2^2-ninjars::q3^2"),
            Atom::one(),
        ),
        (parse!("ninjars::q0^2"), Atom::one() / 4),
        (parse!("ninjars::q1^2"), -Atom::one() / 4),
    ] {
        assert!((laplacian(&numerator) / 8 - expected).expand().is_zero());
    }
    let files = std::env::args().skip(1).collect::<Vec<_>>();
    assert_eq!(
        files.len(),
        50,
        "eight triangles, eighteen boxes and twenty-four pentagons required"
    );
    let mut rows = Vec::new();
    let mut sums = std::array::from_fn::<_, 8, _>(|_| Atom::Zero);
    for (orientation, file) in files.iter().enumerate() {
        let document: Value = serde_json::from_slice(&fs::read(file).unwrap()).unwrap();
        let propagators = document["propagator_edges"].as_array().unwrap().len();
        assert_eq!(
            propagators,
            if orientation < 8 {
                3
            } else if orientation < 26 {
                4
            } else {
                5
            }
        );
        let samples = document["samples"].as_array().unwrap();
        assert_eq!(samples.len(), 8);
        for (lane, sample) in samples.iter().enumerate() {
            let masses = sample["mass_squared"].as_array().unwrap();
            assert_eq!(masses.len(), propagators);
            assert!(
                masses.iter().all(|mass| mass.as_str() == Some("29929")),
                "fixed massive-top fixture is required"
            );
            let parse = |name: &str| {
                Atom::parse(
                    sample[name].as_str().unwrap(),
                    "ninjars",
                    ParseSettings::default(),
                )
                .unwrap()
            };
            let factor = parse("overall_factor");
            for q in [
                symbol!("ninjars::q0"),
                symbol!("ninjars::q1"),
                symbol!("ninjars::q2"),
                symbol!("ninjars::q3"),
                symbol!("ninjars::mu2"),
            ] {
                assert!(!factor.contains_symbol(q));
            }
            let numerator = parse("numerator") * factor;
            assert!(!numerator.contains_symbol(symbol!("ninjars::mu2")));
            let maximum_rank = if propagators == 3 { 2 } else { 4 };
            assert!(
                within_rank(&numerator, maximum_rank),
                "numerator degree exceeds the independent pole formula's domain"
            );
            let pole = if propagators == 3 {
                laplacian(&numerator) / 8
            } else if propagators == 4 {
                laplacian(&laplacian(&numerator)) / 192
            } else {
                // Massive pentagons with rank <=4 have negative superficial
                // UV degree (4+4-2*5=-2), and no massless internal IR poles.
                Atom::Zero
            }
            .expand();
            let value: Complex<Float> = pole
                .evaluate_with_prec(&HashMap::<Atom, Complex<Float>>::new(), 256)
                .expect("the differentiated pole must be fully bound and loop independent");
            sums[lane] += &pole;
            rows.push(json!({
                "orientation": orientation,
                "lane": lane,
                "exact": pole.to_canonical_string(),
                "value": [value.re.to_string(), value.im.to_string()],
            }));
        }
    }
    let exact_sums = sums.map(|sum| sum.expand());
    assert!(exact_sums.iter().all(AtomCore::is_zero));
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "rows": rows,
            "analytic_controls": ["box: (q^2)^2 -> 1", "box: q0^4 -> 1/8", "box: q0^2*q1^2 -> -1/24", "triangle: q^2 -> 1", "triangle: q0^2 -> 1/4", "triangle: q1^2 -> -1/4", "massive rank<=4 pentagon: superficial UV degree<=-2 -> 0"],
            "degree_rejection_controls": ["harmonic (q0+q1)^3 rejected for triangle", "harmonic (q0+q1)^5 rejected for box/pentagon"],
            "exact_sum_zero": exact_sums.map(|sum| sum.is_zero()),
        }))
        .unwrap()
    );
}
