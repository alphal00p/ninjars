//! A soft box in all 24 labellings of its propagators, through every master
//! provider and carrier.
//!
//! oneloopmaster's D0 expressions (OneLOop's `d0`) find a soft line's poles
//! on the four external invariants only: a massless line with its two
//! neighbours' invariants on their mass shells. A labelling that puts such an
//! on-shell pair, with a massive partner, on a channel (`s12` or `s23`)
//! reaches the finite formulas and misses the pole. oneloopmaster's numerical
//! entry points (`NativeEvaluator`, behind `OneLoopNativeMasters` and the
//! fixed-precision carriers of `SymbolicaMasters`) evaluate such a point in a
//! labelling of the same box that keeps the pair off the channels
//! (`oneloop::regular_d0_labelling`). `SymbolicaMasters` with `Float` carriers
//! evaluates the symbolic expressions, so it relabels each D0 row itself
//! (without that, at oneloopmaster 8ec256fc, `Float(128)` puts labelling 0132
//! 73 times its largest coefficient away from labelling 0123).
//!
//! The box is pyAmpliCol's `e+ e- -> t t~ h` box with lines `(m, m, 0, m)`,
//! `m² = 29929` (`MT = 173`) and `mu² = 1e6`, in exact rationals. The 8
//! dihedral ("cyclic") labellings were right at oneloopmaster 09b2c74; the
//! other 16 were finite without the soft pole in binary64 and NaN in
//! DoubleFloat and `Float(128)`. References:
//! - COLLIER 1.2.8 COLI in binary64 (`SetMuUV2_cll(1e6)`, `SetMuIR2_cll(1e6)`,
//!   cache off): the finite part with `SetDeltaIR_cll(0,0)` and the pole
//!   coefficients as differences with `DeltaIR1 = 1` and `DeltaIR2 = 1`, the
//!   same for all 24 labellings;
//! - oneloopmaster 09b2c74 in `Float(128)` on the cyclic labellings, 38
//!   digits (it agrees with COLLIER to 2e-16 and with OneLOop in quadruple
//!   precision to its own working precision).

use ninjars::fast::{C64, FastMaster, FastMasters};
use ninjars::{
    Complex, DoubleFloat, Float, Laurent, MasterIntegral, MasterProvider, MasterRequest,
    MasterStatus,
};
use ninjars_symbolica::{OneLoopNativeMasters, SymbolicaMasters};
use oneloop::NativeFloat;
use symbolica::domains::{
    float::{Real, RealLike},
    rational::Rational,
};

/// `(labelling, cyclic, [mu², p1², p2², p3², p4², s12, s23, m1², m2², m3², m4²])`
/// with real squared masses: the labelling `p` puts the propagators in the
/// order `(q_p0, q_p1, q_p2, q_p3)` of the base row.
const ROWS: [(&str, bool, [&str; 11]); 24] = {
    const M: &str = "29929";
    const H: &str = "15625";
    const A: &str = "1299773111931/1562500";
    const B: &str = "976955119487/3125000";
    const S: &str = "1756052/5";
    const U: &str = "1000000";
    const Z: &str = "0";
    [
        ("0123", true, [U, H, M, M, A, S, B, M, M, Z, M]),
        ("0132", false, [U, H, B, M, S, A, M, M, M, M, Z]),
        ("0213", false, [U, S, M, B, A, H, M, M, Z, M, M]),
        ("0231", false, [U, S, M, B, H, A, M, M, Z, M, M]),
        ("0312", false, [U, A, B, M, S, H, M, M, M, M, Z]),
        ("0321", true, [U, A, M, M, H, S, B, M, M, Z, M]),
        ("1023", false, [U, H, S, M, B, M, A, M, M, Z, M]),
        ("1032", true, [U, H, A, M, M, B, S, M, M, M, Z]),
        ("1203", false, [U, M, S, A, B, H, M, M, Z, M, M]),
        ("1230", true, [U, M, M, A, H, B, S, M, Z, M, M]),
        ("1302", false, [U, B, A, S, M, H, M, M, M, M, Z]),
        ("1320", false, [U, B, M, S, H, M, A, M, M, Z, M]),
        ("2013", false, [U, S, H, B, M, M, A, Z, M, M, M]),
        ("2031", false, [U, S, A, B, M, M, H, Z, M, M, M]),
        ("2103", true, [U, M, H, A, M, S, B, Z, M, M, M]),
        ("2130", false, [U, M, B, A, S, M, H, Z, M, M, M]),
        ("2301", true, [U, M, A, H, M, S, B, Z, M, M, M]),
        ("2310", false, [U, M, B, H, S, M, A, Z, M, M, M]),
        ("3012", true, [U, A, H, M, M, B, S, M, M, M, Z]),
        ("3021", false, [U, A, S, M, B, M, H, M, M, Z, M]),
        ("3102", false, [U, B, H, S, M, A, M, M, M, M, Z]),
        ("3120", false, [U, B, M, S, A, M, H, M, M, Z, M]),
        ("3201", false, [U, M, S, H, B, A, M, M, Z, M, M]),
        ("3210", true, [U, M, M, H, A, B, S, M, Z, M, M]),
    ]
};

/// COLLIER's `[finite, 1/eps, 1/eps²]`, (re, im).
const COLLIER: [(f64, f64); 3] = [
    (-2.0688070109990276e-10, 4.1246220036002915e-11),
    (-2.6860952042523525e-11, 3.9817595477722426e-11),
    (0., 0.),
];

/// oneloopmaster 09b2c74, `Float(128)`, cyclic labellings.
const PRECISE: [(&str, &str); 3] = [
    (
        "-2.0688070109990272212997780991677850542e-10",
        "4.124622003600292727844102532702935962e-11",
    ),
    (
        "-2.6860952042523523990297763214306021163e-11",
        "3.9817595477722419566002502278666577728e-11",
    ),
    ("0", "0"),
];

/// Bits of the comparison arithmetic.
const BITS: u32 = 512;

fn rational(text: &str) -> Rational {
    match text.split_once('/') {
        Some((numerator, denominator)) => Rational::from((
            numerator.parse::<i64>().unwrap(),
            denominator.parse::<i64>().unwrap(),
        )),
        None => Rational::from((text.parse::<i64>().unwrap(), 1)),
    }
}

/// The 24 rows in the carrier of `prototype`: `[p², s12, s23]`, the squared
/// masses and `mu²`, each concatenated over the rows.
fn requests<T: NativeFloat>(prototype: &T) -> [Vec<Complex<T>>; 3] {
    let value =
        |text: &str| Complex::new(prototype.from_rational(&rational(text)), prototype.zero());
    let mut invariants = Vec::new();
    let mut masses = Vec::new();
    let mut scales = Vec::new();
    for (_, _, row) in &ROWS {
        scales.push(value(row[0]));
        invariants.extend(row[1..7].iter().map(|text| value(text)));
        masses.extend(row[7..11].iter().map(|text| value(text)));
    }
    [invariants, masses, scales]
}

fn evaluate<T, M>(provider: &mut M, prototype: &T) -> Vec<Laurent<Complex<T>>>
where
    T: NativeFloat,
    M: MasterProvider<Complex<T>>,
{
    let [invariants, masses, scales] = requests(prototype);
    let zero = || Complex::new(prototype.zero(), prototype.zero());
    let mut batch = vec![[zero(), zero(), zero()]; ROWS.len()];
    let mut statuses = vec![MasterStatus::Success; ROWS.len()];
    provider
        .evaluate_batch(
            MasterRequest {
                integral: MasterIntegral::D0,
                invariants: &invariants,
                masses_squared: &masses,
                scales_squared: &scales,
            },
            &mut batch,
            &mut statuses,
        )
        .unwrap();
    assert!(
        statuses
            .iter()
            .all(|status| *status == MasterStatus::Success),
        "{statuses:?}"
    );
    // One row at a time through `evaluate`: the same bits.
    for (row, batched) in batch.iter().enumerate() {
        let mut single = [[zero(), zero(), zero()]];
        provider
            .evaluate(
                MasterRequest {
                    integral: MasterIntegral::D0,
                    invariants: &invariants[6 * row..6 * row + 6],
                    masses_squared: &masses[4 * row..4 * row + 4],
                    scales_squared: &scales[row..row + 1],
                },
                &mut single,
            )
            .unwrap();
        assert!(
            single[0] == *batched,
            "labelling {}: single {:?} != batched {:?}",
            ROWS[row].0,
            single[0],
            batched
        );
    }
    batch
}

fn from_f64(z: Complex<f64>) -> Complex<Float> {
    Complex::new(Float::with_val(BITS, z.re), Float::with_val(BITS, z.im))
}

fn from_double(z: &Complex<DoubleFloat>) -> Complex<Float> {
    // The leading binary64 and the exact remainder.
    let float = |x: &DoubleFloat| {
        let high = x.to_f64();
        let low = (*x - DoubleFloat::from(high)).to_f64();
        Float::with_val(BITS, high) + Float::with_val(BITS, low)
    };
    Complex::new(float(&z.re), float(&z.im))
}

fn modulus(z: &Complex<Float>) -> f64 {
    (z.re.clone() * &z.re + z.im.clone() * &z.im)
        .sqrt()
        .to_f64()
}

/// Largest `|value - reference|` over the three coefficients, over the
/// largest `|reference|`.
fn error(value: &[Complex<Float>; 3], reference: &[Complex<Float>; 3]) -> f64 {
    let scale = reference.iter().map(modulus).fold(0., f64::max);
    value
        .iter()
        .zip(reference)
        .map(|(x, r)| modulus(&Complex::new(x.re.clone() - &r.re, x.im.clone() - &r.im)))
        .fold(0., f64::max)
        / scale
}

fn collier() -> [Complex<Float>; 3] {
    COLLIER.map(|(re, im)| from_f64(Complex::new(re, im)))
}

fn reference() -> [Complex<Float>; 3] {
    let parse = |text: &str| Float::parse(text, Some(BITS)).unwrap();
    PRECISE.map(|(re, im)| Complex::new(parse(re), parse(im)))
}

/// Every labelling against COLLIER and, for the extended carriers, against
/// the 38-digit reference; the soft pole is present in every labelling
/// (COLLIER's simple pole is 8.3e-2 of the box's largest coefficient).
fn check(name: &str, values: &[[Complex<Float>; 3]], collier_tolerance: f64, tolerance: f64) {
    let (collier, reference) = (collier(), reference());
    // The provider's own value in the labelling of the base row (cyclic).
    let base = &values[0];
    let mut worst = (0f64, 0f64);
    for ((labelling, cyclic, _), value) in ROWS.iter().zip(values) {
        let against_collier = error(value, &collier);
        let against_reference = error(value, &reference);
        let against_base = error(value, base);
        eprintln!(
            "{name} {labelling} {}: COLLIER {against_collier:.2e}, Float(128) reference \
             {against_reference:.2e}, labelling 0123 {against_base:.2e}",
            if *cyclic { "cyclic" } else { "non-cyclic" }
        );
        assert!(
            against_base <= tolerance,
            "{name}, labelling {labelling}: {against_base:e} from labelling 0123"
        );
        assert!(
            against_collier <= collier_tolerance,
            "{name}, labelling {labelling}: {against_collier:e} from COLLIER"
        );
        assert!(
            against_reference <= tolerance,
            "{name}, labelling {labelling}: {against_reference:e} from the Float(128) reference"
        );
        worst = (worst.0.max(against_collier), worst.1.max(against_reference));
    }
    eprintln!(
        "{name}: worst {:.2e} from COLLIER, {:.2e} from the Float(128) reference",
        worst.0, worst.1
    );
}

#[test]
fn binary64_providers_find_the_soft_pole_in_every_labelling() {
    let mut native = OneLoopNativeMasters::new().unwrap();
    let native_values = evaluate(&mut native, &0f64);
    // The same rows through the fast engine's interface: the same bits.
    let [invariants, masses, scales] = requests(&0f64);
    let args: Vec<C64> = (0..ROWS.len())
        .flat_map(|row| {
            invariants[6 * row..6 * row + 6]
                .iter()
                .chain(&masses[4 * row..4 * row + 4])
                .chain(&scales[row..row + 1])
                .map(|z| C64::new(z.re, z.im))
                .collect::<Vec<_>>()
        })
        .collect();
    let mut fast = vec![[C64::ZERO; 3]; ROWS.len()];
    let mut ok = vec![false; ROWS.len()];
    FastMasters::evaluate(&mut native, FastMaster::D0, &args, &mut fast, &mut ok).unwrap();
    assert!(ok.iter().all(|&ok| ok));
    for ((labelling, _, _), (fast, value)) in ROWS.iter().zip(fast.iter().zip(&native_values)) {
        for (f, v) in fast.iter().zip(value) {
            assert!(
                f.re.to_bits() == v.re.to_bits() && f.im.to_bits() == v.im.to_bits(),
                "labelling {labelling}: FastMasters {f:?} != MasterProvider {v:?}"
            );
        }
    }
    // `SymbolicaMasters<f64>` goes through the same native entry point.
    let symbolica_values = evaluate(&mut SymbolicaMasters::new(0f64), &0f64);
    for ((labelling, _, _), (s, v)) in ROWS.iter().zip(symbolica_values.iter().zip(&native_values))
    {
        assert!(
            s == v,
            "labelling {labelling}: SymbolicaMasters<f64> {s:?} != OneLoopNativeMasters {v:?}"
        );
    }
    let values: Vec<_> = native_values
        .iter()
        .map(|value| value.map(from_f64))
        .collect();
    check("binary64", &values, 1e-13, 1e-13);
}

#[test]
fn double_float_finds_the_soft_pole_in_every_labelling() {
    let prototype = DoubleFloat::from(0.);
    let values = evaluate(&mut SymbolicaMasters::new(prototype), &prototype);
    let values: Vec<_> = values
        .iter()
        .map(|value| [0, 1, 2].map(|k| from_double(&value[k])))
        .collect();
    check("DoubleFloat", &values, 1e-13, 1e-26);
}

#[test]
fn float_finds_the_soft_pole_in_every_labelling() {
    for bits in [128, 256] {
        let prototype = Float::new(bits);
        let values = evaluate(&mut SymbolicaMasters::new(prototype.clone()), &prototype);
        check(&format!("Float({bits})"), &values, 1e-13, 1e-30);
    }
}
