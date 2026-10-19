use std::collections::HashMap;

use ninjars::{
    Complex, DoubleFloat, Float, FourMomentum, Laurent, MasterFailure, MasterIntegral,
    MasterProvider, MasterRequest, MasterStatus,
};
use ninjars_symbolica::SymbolicaMasters;
use oneloop::NativeFloat;
use serde_json::Value;
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        float::{Real, RealLike},
        rational::Rational,
    },
    evaluate::EvaluationDomain,
    parser::ParseSettings,
};

fn evidence() -> Value {
    serde_json::from_str(include_str!(
        "../../../validation/evidence/box-masters.json"
    ))
    .unwrap()
}
type Inputs<T> = (Vec<Complex<T>>, Vec<Complex<T>>, Vec<Complex<T>>);
fn exact<T: NativeFloat>(number: &Value, prototype: &T) -> Complex<T>
where
    Complex<T>: EvaluationDomain,
{
    Atom::parse(
        number
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| number.to_string()),
        "ninjars_box_master_tests",
        ParseSettings::default(),
    )
    .unwrap()
    .evaluate_with_prec(
        &HashMap::<Atom, Complex<T>>::new(),
        prototype.get_precision(),
    )
    .unwrap()
}
fn inputs<T: NativeFloat>(indices: &[usize], prototype: &T) -> Inputs<T>
where
    Complex<T>: EvaluationDomain,
{
    let data = evidence();
    let mut invariants = Vec::new();
    let mut masses = Vec::new();
    let mut scales = Vec::new();
    for &index in indices {
        let case = &data["cases"][index];
        invariants.extend(
            case["invariants"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| exact(v, prototype)),
        );
        masses.extend(
            case["masses_squared"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| Complex::new(exact(&v[0], prototype).re, exact(&v[1], prototype).re)),
        );
        scales.push(exact(&case["scale_squared"], prototype));
    }
    (invariants, masses, scales)
}
fn close(actual: Complex<f64>, expected: Complex<f64>) {
    assert!(
        (actual.re - expected.re).abs() <= 1e-9 * (1. + expected.re.abs())
            && (actual.im - expected.im).abs() <= 1e-9 * (1. + expected.im.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn d0_order_and_mixed_branch_batches_match_unchanged_ninja() {
    let data = evidence();
    assert_eq!(
        data["invariant_order"],
        serde_json::json!(["s21", "s32", "s43", "s14", "s31", "s42"])
    );
    // Reuse the canonical momentum dot product as an independent routing check.
    for case in data["cases"].as_array().unwrap() {
        if let Some(shifts) = case["shifts"].as_array() {
            let shifts: Vec<_> = shifts
                .iter()
                .map(|v| {
                    FourMomentum::from_args(
                        v[0].as_f64().unwrap(),
                        v[1].as_f64().unwrap(),
                        v[2].as_f64().unwrap(),
                        v[3].as_f64().unwrap(),
                    )
                })
                .collect();
            for (index, (a, b)) in [(1, 0), (2, 1), (3, 2), (0, 3), (2, 0), (3, 1)]
                .into_iter()
                .enumerate()
            {
                let delta = shifts[a] - shifts[b];
                assert_eq!(
                    delta.dot(&delta),
                    case["invariants"][index].as_f64().unwrap()
                );
            }
        }
    }
    let mut provider = SymbolicaMasters::new(0.);
    for indices in [
        vec![],
        vec![4],
        (0..6).collect(),
        (0..31).rev().map(|i| i % 6).collect(),
    ] {
        let (invariants, masses, scales) = inputs(&indices, &0.);
        let mut output = vec![[Complex::new(77., 0.); 3]; indices.len()];
        let mut statuses =
            vec![MasterStatus::Failed(MasterFailure::NonFiniteResult); indices.len()];
        provider
            .evaluate_batch(
                MasterRequest {
                    integral: MasterIntegral::D0,
                    invariants: &invariants,
                    masses_squared: &masses,
                    scales_squared: &scales,
                },
                &mut output,
                &mut statuses,
            )
            .unwrap();
        assert!(
            statuses
                .iter()
                .all(|status| *status == MasterStatus::Success)
        );
        for (row, &index) in output.iter().zip(&indices) {
            for (actual, expected) in row
                .iter()
                .zip(data["cases"][index]["epsilon"].as_array().unwrap())
            {
                close(
                    *actual,
                    Complex::new(expected[0].as_f64().unwrap(), expected[1].as_f64().unwrap()),
                );
            }
        }
    }
    // A domain failure in the middle of different active branches must retain
    // its old output without selecting that row's invalid continuation.
    let (invariants, mut masses, scales) = inputs(&[0, 1, 5], &0.);
    masses[4].im = 0.1;
    let mut output = [[Complex::new(77., 0.); 3]; 3];
    let mut statuses = [MasterStatus::Success; 3];
    provider
        .evaluate_batch(
            MasterRequest {
                integral: MasterIntegral::D0,
                invariants: &invariants,
                masses_squared: &masses,
                scales_squared: &scales,
            },
            &mut output,
            &mut statuses,
        )
        .unwrap();
    assert_eq!(
        statuses,
        [
            MasterStatus::Success,
            MasterStatus::Failed(MasterFailure::InvalidMass),
            MasterStatus::Success
        ]
    );
    assert_eq!(output[1], [Complex::new(77., 0.); 3]);
    for (row, index) in [(0, 0), (2, 5)] {
        for (actual, expected) in output[row]
            .iter()
            .zip(data["cases"][index]["epsilon"].as_array().unwrap())
        {
            close(
                *actual,
                Complex::new(expected[0].as_f64().unwrap(), expected[1].as_f64().unwrap()),
            );
        }
    }
}

fn evaluate<T: NativeFloat>(prototype: T) -> Vec<Laurent<Complex<T>>>
where
    Complex<T>: EvaluationDomain,
{
    let (invariants, masses, scales) = inputs(&[0, 1], &prototype);
    let zero = Complex::new(prototype.zero(), prototype.zero());
    let mut output = vec![std::array::from_fn(|_| zero.clone()); 2];
    SymbolicaMasters::new(prototype)
        .evaluate(
            MasterRequest {
                integral: MasterIntegral::D0,
                invariants: &invariants,
                masses_squared: &masses,
                scales_squared: &scales,
            },
            &mut output,
        )
        .unwrap();
    output
}

fn relative_difference(a: &Float, b: &Float) -> Float {
    (a.clone() - b).norm() / (Float::with_val(320, 1) + b.norm())
}

#[test]
fn d0_double_float_and_arbitrary_precision_converge_beyond_binary64() {
    let binary64 = evaluate(0.);
    let doubled = evaluate(DoubleFloat::default());
    let lower = evaluate(Float::new(160));
    let higher = evaluate(Float::new(256));
    let mut saw_binary64_roundoff = false;
    for row in 0..2 {
        for component in 0..3 {
            for (ordinary, doubled, lower, higher) in [
                (
                    &binary64[row][component].re,
                    &doubled[row][component].re,
                    &lower[row][component].re,
                    &higher[row][component].re,
                ),
                (
                    &binary64[row][component].im,
                    &doubled[row][component].im,
                    &lower[row][component].im,
                    &higher[row][component].im,
                ),
            ] {
                let ordinary = Float::with_val(320, *ordinary);
                let doubled = Float::from(*doubled);
                let e64 = relative_difference(&ordinary, higher);
                let e106 = relative_difference(&doubled, higher);
                let e160 = relative_difference(lower, higher);
                assert!(e64.to_f64() < 1e-9, "binary64 error {e64}");
                assert!(
                    e106 < Float::parse("1e-25", Some(320)).unwrap(),
                    "DoubleFloat error {e106}"
                );
                assert!(
                    e160 < Float::parse("1e-42", Some(320)).unwrap(),
                    "Float160/256 error {e160}"
                );
                if e64.to_f64() > 1e-20 {
                    saw_binary64_roundoff = true;
                    assert!(e106 < e64);
                }
            }
        }
    }
    assert!(
        saw_binary64_roundoff,
        "fixture must distinguish the numerical carriers"
    );
}

/// Equal-mass four-mass D0 (`m² = mu² = 1`) from n12r12genericm1 lane 2,
/// `[p1², p2², p3², p4², s12, s23]`. DoubleFloat once returned the real part
/// +0.37226 instead of -0.063421 in one relabelling and at the rounded
/// invariants the reduction forms from its momenta: its non-commutative
/// product left an imaginary residue in `1 - z conj(z)`, and an exact sign
/// test inverted a dilogarithm argument (oneloopmaster
/// `tests/native_double_float_d0.rs`).
const EQUAL_MASS_BOX: [(i64, i64); 6] = [
    (2171, 144),
    (211, 16),
    (-5, 4),
    (-587, 72),
    (-311, 72),
    (27, 16),
];
/// The rounded DoubleFloat invariants of the reduction (exact decimal values).
const EQUAL_MASS_BOX_ROUNDED: [&str; 6] = [
    "1.507638888888888888888888888888891080169e1",
    "1.318750000000000000000000000000001232595e1",
    "-1.249999999999999999999999999999981511073e0",
    "-8.152777777777777777777777777777821603384e0",
    "-4.319444444444444444444444444444430748943e0",
    "1.687499999999999999999999999999996918512e0",
];
/// The finite part from native Float and the interpreter at 256 bits.
const EQUAL_MASS_BOX_FINITE: [&str; 2] = [
    "-6.342090417471351697878784160261216715137e-2",
    "2.110820888749008107801505116588042554555e-2",
];

#[test]
fn d0_double_float_equal_mass_box_matches_float_in_every_labelling() {
    let exact = EQUAL_MASS_BOX.map(|(n, d)| DoubleFloat::from(&Rational::from((n, d))));
    let rounded =
        EQUAL_MASS_BOX_ROUNDED.map(|v| Float::parse(v, Some(256)).unwrap().to_double_float());
    // The box symmetries: cyclic shifts and the reflection of the propagators.
    let shift = |p: &[DoubleFloat; 6]| [1, 2, 3, 0, 5, 4].map(|i| p[i]);
    let reflect = |p: &[DoubleFloat; 6]| [2, 1, 0, 3, 5, 4].map(|i| p[i]);
    let mut invariants = Vec::new();
    for start in [exact, rounded] {
        let mut current = start;
        for _ in 0..4 {
            for labelling in [current, reflect(&current)] {
                invariants.extend(labelling.map(|p| Complex::new(p, DoubleFloat::default())));
            }
            current = shift(&current);
        }
    }
    let rows = invariants.len() / 6;
    let one = Complex::new(DoubleFloat::from(1.), DoubleFloat::default());
    let zero = Complex::new(DoubleFloat::default(), DoubleFloat::default());
    let mut output = vec![[zero; 3]; rows];
    SymbolicaMasters::new(DoubleFloat::default())
        .evaluate(
            MasterRequest {
                integral: MasterIntegral::D0,
                invariants: &invariants,
                masses_squared: &vec![one; 4 * rows],
                scales_squared: &vec![one; rows],
            },
            &mut output,
        )
        .unwrap();
    let expected = EQUAL_MASS_BOX_FINITE.map(|v| Float::parse(v, Some(320)).unwrap());
    let bound = Float::parse("1e-29", Some(320)).unwrap();
    for (row, value) in output.iter().enumerate() {
        for (actual, expected) in [(&value[0].re, &expected[0]), (&value[0].im, &expected[1])] {
            let error = relative_difference(&Float::from(*actual), expected);
            assert!(
                error < bound,
                "row {row}: {actual:?} vs {expected} ({error})"
            );
        }
        assert!(value[1] == zero && value[2] == zero, "row {row}: poles");
    }
}
