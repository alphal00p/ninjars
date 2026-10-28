//! The on-shell threshold of `OneLoopNativeMasters`
//! (`with_onshell_threshold`): by default original Ninja's absolute `1e-9`,
//! which also moves genuine scales below it (a squared mass of `1e-12`, a keV
//! mass in GeV, becomes massless); `None` evaluates every row at its given
//! arguments, as oneloopmaster's exact masters (300-bit `SymbolicaMasters`,
//! which has no threshold for `Float`) do.

use ninjars::fast::{C64, FastMaster, FastMasters};
use ninjars::{Complex, Float, MasterIntegral, MasterProvider, MasterRequest};
use ninjars_symbolica::{OneLoopNativeMasters, SymbolicaMasters};

/// A keV mass squared in GeV².
const LIGHT: f64 = 1e-12;
const MU2: f64 = 1e4;
const BITS: u32 = 300;

fn native(threshold: Option<f64>) -> OneLoopNativeMasters {
    OneLoopNativeMasters::new()
        .unwrap()
        .with_onshell_threshold(threshold)
}

/// Every Laurent triple of `rows` (row-major, `family.arity()` per row)
/// through the batched `FastMasters` interface.
fn fast(masters: &mut OneLoopNativeMasters, family: FastMaster, rows: &[C64]) -> Vec<[C64; 3]> {
    let count = rows.len() / family.arity();
    let mut out = vec![[C64::ZERO; 3]; count * family.outputs()];
    let mut ok = vec![false; count];
    FastMasters::evaluate(masters, family, rows, &mut out, &mut ok).unwrap();
    assert!(ok.iter().all(|&ok| ok), "{family:?} {rows:?}");
    out
}

/// One row through the generic `MasterProvider` interface.
fn provider<M: MasterProvider<Complex<T>>, T>(
    masters: &mut M,
    integral: MasterIntegral,
    row: &[C64],
    convert: impl Fn(C64) -> Complex<T>,
    zero: Complex<T>,
) -> [Complex<T>; 3]
where
    T: Clone,
{
    let (invariants, masses) = match integral {
        MasterIntegral::A0 => (0, 1),
        MasterIntegral::C0 => (3, 3),
        MasterIntegral::D0 => (6, 4),
        _ => (1, 2),
    };
    let values: Vec<Complex<T>> = row.iter().map(|&z| convert(z)).collect();
    let mut out = vec![[zero.clone(), zero.clone(), zero]];
    masters
        .evaluate(
            MasterRequest {
                integral,
                invariants: &values[..invariants],
                masses_squared: &values[invariants..invariants + masses],
                scales_squared: &values[invariants + masses..],
            },
            &mut out,
        )
        .unwrap();
    out.pop().unwrap()
}

/// oneloopmaster's exact master in 300-bit arithmetic, rounded to binary64.
fn exact(integral: MasterIntegral, row: &[C64]) -> [C64; 3] {
    let mut masters = SymbolicaMasters::new(Float::new(BITS));
    let cf = |z: C64| Complex::new(Float::with_val(BITS, z.re), Float::with_val(BITS, z.im));
    provider(&mut masters, integral, row, cf, cf(C64::ZERO)).map(|v| {
        let f = v.to_f64();
        C64::new(f.re, f.im)
    })
}

fn bits(values: &[[C64; 3]]) -> Vec<[u64; 6]> {
    values
        .iter()
        .map(|v| {
            [
                v[0].re.to_bits(),
                v[0].im.to_bits(),
                v[1].re.to_bits(),
                v[1].im.to_bits(),
                v[2].re.to_bits(),
                v[2].im.to_bits(),
            ]
        })
        .collect()
}

/// Largest deviation of the Laurent coefficients relative to the row's
/// largest coefficient.
fn relative(value: &[C64; 3], reference: &[C64; 3]) -> f64 {
    let scale = reference.iter().map(|z| z.abs()).fold(1e-300, f64::max);
    (0..3)
        .map(|k| (value[k] - reference[k]).abs())
        .fold(0., f64::max)
        / scale
}

fn real(values: &[f64]) -> Vec<C64> {
    values.iter().map(|&x| C64::real(x)).collect()
}

/// Rows with a light squared mass and, for each, the same row with every
/// light mass set to zero: what the default threshold evaluates instead.
fn light_rows() -> Vec<(FastMaster, MasterIntegral, Vec<C64>, Vec<C64>)> {
    let m = LIGHT;
    vec![
        (
            FastMaster::A0,
            MasterIntegral::A0,
            real(&[m, MU2]),
            real(&[0., MU2]),
        ),
        // B0(0; m², m²) = 1/eps - log(m²/mu²): scaleless when massless.
        (
            FastMaster::BubbleRank2,
            MasterIntegral::B0,
            real(&[0., m, m, MU2]),
            real(&[0., 0., 0., MU2]),
        ),
        // A collinear-singular triangle: log²(m²) when massive, 1/eps² when
        // massless.
        (
            FastMaster::C0,
            MasterIntegral::C0,
            real(&[0., 0., -MU2, m, m, m, MU2]),
            real(&[0., 0., -MU2, 0., 0., 0., MU2]),
        ),
        // A box with two light lines between massless legs.
        (
            FastMaster::D0,
            MasterIntegral::D0,
            real(&[0., 0., 0., 0., -MU2, -2. * MU2, m, m, 0., 0., MU2]),
            real(&[0., 0., 0., 0., -MU2, -2. * MU2, 0., 0., 0., 0., MU2]),
        ),
    ]
}

#[test]
fn the_default_threshold_is_ninjas_and_snaps_light_masses() {
    assert_eq!(
        OneLoopNativeMasters::new().unwrap().onshell_threshold(),
        Some(1e-9)
    );
    let mut snapped = native(Some(1e-9));
    let mut default = OneLoopNativeMasters::new().unwrap();
    for (family, _, row, massless) in light_rows() {
        let value = fast(&mut default, family, &row);
        assert_eq!(bits(&value), bits(&fast(&mut snapped, family, &row)));
        // Below the threshold the light row is the massless one, bit for bit.
        assert_eq!(
            bits(&value),
            bits(&fast(&mut default, family, &massless)),
            "{family:?}"
        );
    }
}

#[test]
fn without_a_threshold_light_masses_match_the_exact_masters() {
    let mut masters = native(None);
    assert_eq!(masters.onshell_threshold(), None);
    let mut snapped = OneLoopNativeMasters::new().unwrap();
    for (family, integral, row, _) in light_rows() {
        // The first coefficient of a bubble family is B0.
        let value = fast(&mut masters, family, &row)[0];
        let reference = exact(integral, &row);
        let error = relative(&value, &reference);
        assert!(
            error <= 1e-12,
            "{family:?} {row:?}: native {value:?} exact {reference:?} ({error:e})"
        );
        let wrong = fast(&mut snapped, family, &row)[0];
        assert!(
            relative(&wrong, &reference) > 1e-3,
            "{family:?}: the default threshold no longer changes {row:?}"
        );
    }
}

#[test]
fn interfaces_agree_bitwise_without_a_threshold() {
    let mut batched = native(None);
    let mut generic = native(None);
    for (family, integral, row, _) in light_rows() {
        let value = fast(&mut batched, family, &row)[0];
        let through = provider(
            &mut generic,
            integral,
            &row,
            C64::to_symbolica,
            Complex::new(0., 0.),
        )
        .map(C64::from_symbolica);
        assert_eq!(bits(&[value]), bits(&[through]), "{family:?}");
    }
    // Two rows in one batched call equal their single-row calls.
    let rows = light_rows();
    let (_, _, light, massless) = &rows[2];
    let pair = [light.as_slice(), massless.as_slice()].concat();
    let together = fast(&mut batched, FastMaster::C0, &pair);
    assert_eq!(
        bits(&together[..1]),
        bits(&fast(&mut batched, FastMaster::C0, light))
    );
    assert_eq!(
        bits(&together[1..]),
        bits(&fast(&mut batched, FastMaster::C0, massless))
    );
}

#[test]
fn a_chosen_threshold_snaps_only_below_it() {
    let mut fine = native(Some(1e-15));
    let mut none = native(None);
    for (family, _, row, massless) in light_rows() {
        assert_eq!(
            bits(&fast(&mut fine, family, &row)),
            bits(&fast(&mut none, family, &row)),
            "{family:?}: 1e-12 is above 1e-15"
        );
        let tiny: Vec<C64> = row
            .iter()
            .zip(&massless)
            .map(|(&z, &zero)| if z == zero { z } else { C64::real(1e-16) })
            .collect();
        assert_eq!(
            bits(&fast(&mut fine, family, &tiny)),
            bits(&fast(&mut none, family, &massless)),
            "{family:?}: 1e-16 is below 1e-15"
        );
    }
}

#[test]
fn near_shell_invariants_are_kept_without_a_threshold() {
    // C0(m², 0, s; 0, m², m²)-type row of a massive leg, its invariant
    // 4e-10 off the shell of the mass 29929 (inside the default threshold).
    let mass = 29929.;
    let near = mass + 4e-10;
    assert_ne!(near, mass);
    let on = real(&[mass, 0., -MU2, 0., mass, mass, MU2]);
    let off = real(&[near, 0., -MU2, 0., mass, mass, MU2]);
    let mut default = OneLoopNativeMasters::new().unwrap();
    let mut none = native(None);
    assert_eq!(
        bits(&fast(&mut default, FastMaster::C0, &off)),
        bits(&fast(&mut default, FastMaster::C0, &on))
    );
    assert_ne!(
        bits(&fast(&mut none, FastMaster::C0, &off)),
        bits(&fast(&mut none, FastMaster::C0, &on))
    );
    // A row the threshold leaves unchanged (a zero mass and an exactly
    // on-shell invariant) is bitwise equal under both settings.
    let unchanged = real(&[100., 15625., -MU2, mass, 15625., 0., MU2]);
    assert_eq!(
        bits(&fast(&mut default, FastMaster::C0, &unchanged)),
        bits(&fast(&mut none, FastMaster::C0, &unchanged))
    );
    // `set_onshell_threshold` restores the default on the same provider.
    none.set_onshell_threshold(Some(1e-9));
    assert_eq!(
        bits(&fast(&mut none, FastMaster::C0, &off)),
        bits(&fast(&mut default, FastMaster::C0, &on))
    );
}
