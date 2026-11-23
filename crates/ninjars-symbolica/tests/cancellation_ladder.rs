//! The generic engine's cancellation ladder (`cancellation_ladder`,
//! `docs/cancellation-ladder.md`) on pyAmpliCol `g g > g g g` lanes with a
//! degenerate two-mass triangle (`validation/conditioning/triangle-gram`),
//! made consistent at 512 bits and rounded to each base carrier, against
//! their 512-bit reduction.

#[path = "common/conditioning.rs"]
mod conditioning;
#[path = "common/ladder.rs"]
mod ladder;

use conditioning::{Error, parse_fixture};
use ladder::{Lane, consistent, run};
use ninjars::{Complex, DoubleFloat, Float};
use ninjars_symbolica::cancellation_ladder::LadderOutcome;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../validation/conditioning/triangle-gram");
const REFERENCE_BITS: u32 = 512;
const DEGENERATE: [&str; 3] = ["ggg-308-n3-r3", "ggg-308-n4-r4", "ggg-9117-n3-r3"];

fn lane(name: &str) -> Lane {
    let text = std::fs::read_to_string(format!("{DIR}/{name}.json")).expect("fixture");
    consistent(&parse_fixture(&text).expect("fixture shape"), REFERENCE_BITS)
}

fn truth(lane: &Lane) -> Complex<f64> {
    run(lane, Float::with_val(REFERENCE_BITS, 0), false, None)
        .expect("reference")
        .value
}

fn relative(a: Complex<f64>, b: Complex<f64>) -> f64 {
    (a.re - b.re).hypot(a.im - b.im) / b.re.hypot(b.im)
}

/// Double-double base: the degenerate lanes lose about 70 of 106 bits (the
/// generic engine refuses them `Unstable`); the ladder serves them from one
/// MPFR rung within 1e-20 of the reference (the base pass is 1e-14 .. 1e-9
/// off).
#[test]
fn double_double_lanes_climb_to_mpfr() -> Result<(), Error> {
    for name in DEGENERATE {
        let lane = lane(name);
        let reference = truth(&lane);
        let out = run(&lane, DoubleFloat::from(0.0), true, None)?;
        let rescued = out.ladder.expect("flagged");
        assert_eq!(rescued.outcome, LadderOutcome::Accepted, "{name}");
        assert_eq!(rescued.rungs.len(), 1, "{name}: {:?}", rescued.rungs);
        assert!(rescued.rungs[0] > 106 && rescued.rungs[0] % 64 == 0, "{name}");
        assert!(rescued.projected, "{name}");
        let before = relative(out.value, reference);
        let after = relative(rescued.value, reference);
        assert!(after < 1e-20, "{name}: {after:e}");
        assert!(before > 1e3 * after.max(1e-30), "{name}: {before:e} -> {after:e}");
    }
    Ok(())
}

/// binary64 base: the same lanes climb to double-double when
/// `L + log2(1/target) + m` fits in 106 bits, else to MPFR; served within the
/// default target (`2^-26.5`).
#[test]
fn binary64_lanes_take_the_first_sufficient_rung() -> Result<(), Error> {
    for name in DEGENERATE {
        let lane = lane(name);
        let reference = truth(&lane);
        let out = run(&lane, 0f64, true, None)?;
        let rescued = out.ladder.expect("flagged");
        assert_eq!(rescued.outcome, LadderOutcome::Accepted, "{name}");
        let expected = if out.lost_bits + 26.5 + 8. <= 106. { 106 } else { 128 };
        assert!(rescued.rungs[0] == expected || out.lost_bits > 80., "{name}: L {} rungs {:?}", out.lost_bits, rescued.rungs);
        let after = relative(rescued.value, reference);
        assert!(after < 2f64.powf(-26.5), "{name}: {after:e}");
    }
    Ok(())
}

/// MPFR base (200 bits): the lanes keep about 130 bits and are not flagged
/// at the default target; a caller's target of 1e-50 flags them and the
/// rule picks `L + 166 + 8` bits (256).
#[test]
fn mpfr_base_follows_the_callers_target() -> Result<(), Error> {
    let name = DEGENERATE[0];
    let lane = lane(name);
    let reference = truth(&lane);
    let quiet = run(&lane, Float::with_val(200, 0), true, None)?;
    assert!(quiet.ladder.is_none(), "default target: not flagged");
    let out = run(&lane, Float::with_val(200, 0), true, Some(1e-50))?;
    let rescued = out.ladder.expect("flagged at 1e-50");
    assert_eq!(rescued.outcome, LadderOutcome::Accepted);
    assert_eq!(rescued.rungs, vec![256]);
    assert!(relative(rescued.value, reference) < 1e-15, "f64 comparison floor");
    Ok(())
}

/// A loose caller target re-reduces nothing: the degenerate double-double
/// lane, refused `Unstable` by the base pass (more than half its bits lost),
/// is served at its base value because its estimate meets the target. A
/// regular lane is never flagged.
#[test]
fn loose_targets_and_regular_lanes_are_left_alone() -> Result<(), Error> {
    let degenerate = lane(DEGENERATE[0]);
    let out = run(&degenerate, DoubleFloat::from(0.0), true, Some(1e-5))?;
    assert_eq!(out.status, "Unstable");
    let served = out.ladder.expect("record");
    assert_eq!(served.outcome, LadderOutcome::WithinTarget);
    assert!(served.rungs.is_empty());
    assert_eq!(served.value, out.value);
    let control = lane("ggg-0-n5-r5-control");
    let out = run(&control, DoubleFloat::from(0.0), true, None)?;
    assert!(out.ladder.is_none(), "regular lane flagged (L = {})", out.lost_bits);
    assert_eq!(out.status, "Success");
    Ok(())
}
