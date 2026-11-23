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
        // One rung, or two when the first (128 bits) keeps too few tracked
        // bits; MPFR widths only.
        assert!(rescued.rungs.len() <= 2, "{name}: {:?}", rescued.rungs);
        assert!(rescued.rungs.iter().all(|&b| b > 106 && b % 64 == 0), "{name}");
        assert!(rescued.projected, "{name}");
        let before = relative(out.value, reference);
        let after = relative(rescued.value, reference);
        assert!(after < 1e-20, "{name}: {after:e}");
        assert!(before > 1e3 * after.max(1e-30), "{name}: {before:e} -> {after:e}");
    }
    Ok(())
}

/// binary64 base (the generic engine's routed lanes): every lane the ladder
/// serves is within its target. (These lanes, fed binary64-rounded inputs,
/// come out as noise as large as their terms, so the measurement may not
/// flag them: a documented limit of a measured trigger, not of the rung.)
#[test]
fn binary64_lanes_are_served_within_their_target() -> Result<(), Error> {
    for name in DEGENERATE {
        let lane = lane(name);
        let reference = truth(&lane);
        let out = run(&lane, 0f64, true, None)?;
        if let Some(rescued) = out.ladder {
            if rescued.outcome == LadderOutcome::Accepted {
                let after = relative(rescued.value, reference);
                assert!(after < 256. * 2f64.powf(-26.5), "{name}: {after:e}");
            }
        }
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
    assert_eq!(rescued.rungs[0], 256, "{:?}", rescued.rungs);
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

/// No served value is more wrong than its own estimate says, at every rung
/// width the rule produces from a double-double base (first rungs 128, 192
/// and 256 bits). A 128-bit MPFR rung of a lane that loses about 70 bits
/// keeps far fewer bits than `p' - L'` (numerica rounds every intermediate to
/// its tracked precision): its estimate must say so, and the ladder then
/// climbs instead of serving it.
#[test]
fn served_values_meet_their_estimates_at_every_rung_width() -> Result<(), Error> {
    for name in DEGENERATE {
        let lane = lane(name);
        let reference = truth(&lane);
        let lost = run(&lane, DoubleFloat::from(0.0), true, None)?.lost_bits;
        // Targets whose first rung is 128, 192 and 256 bits for this lane.
        for (target, first) in [
            (2f64.powf(lost - 110.), 128),
            (2f64.powi(-100), 192),
            (2f64.powi(-150), 256),
        ] {
            let out = run(&lane, DoubleFloat::from(0.0), true, Some(target))?;
            let rescued = out.ladder.expect("flagged");
            assert_eq!(rescued.rungs[0], first, "{name} {target:e}: L {lost}");
            assert_eq!(rescued.outcome, LadderOutcome::Accepted, "{name} {target:e}: {:?}", rescued.rungs);
            let estimate = rescued.estimate.expect("measured");
            assert!(estimate <= target, "{name} {target:e}: {estimate:e}");
            let error = relative(rescued.value, reference);
            // f64 comparison floor 1e-15; a factor 2^8 for the growth the
            // estimate leaves out.
            assert!(error <= 256. * estimate + 1e-15, "{name} {target:e}: error {error:e}, estimate {estimate:e}, rungs {:?}", rescued.rungs);
        }
    }
    Ok(())
}
