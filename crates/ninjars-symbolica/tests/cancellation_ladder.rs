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
use ladder::{Lane, consistent, run, run_inputs, run_rule};
use ninjars::{Complex, DoubleFloat, Float};
use ninjars_symbolica::cancellation_ladder::{LadderOutcome, LadderRule};

const DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../validation/conditioning/triangle-gram"
);
const REFERENCE_BITS: u32 = 512;
const DEGENERATE: [&str; 3] = ["ggg-308-n3-r3", "ggg-308-n4-r4", "ggg-9117-n3-r3"];

fn lane(name: &str) -> Lane {
    let text = std::fs::read_to_string(format!("{DIR}/{name}.json")).expect("fixture");
    consistent(
        &parse_fixture(&text).expect("fixture shape"),
        REFERENCE_BITS,
    )
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
        // One MPFR rung and its confirming reduction (64 bits wider), or a
        // second pair after a climb; MPFR widths only.
        assert!(rescued.rungs.len() <= 4, "{name}: {:?}", rescued.rungs);
        assert!(
            rescued.rungs.iter().all(|&b| b > 106 && b % 64 == 0),
            "{name}"
        );
        assert!(rescued.projected, "{name}");
        let before = relative(out.value, reference);
        let after = relative(rescued.value, reference);
        assert!(after < 1e-20, "{name}: {after:e}");
        assert!(
            before > 1e3 * after.max(1e-30),
            "{name}: {before:e} -> {after:e}"
        );
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

/// MPFR base (200 bits): the lanes keep about 100 bits (numerica's tracked
/// precision, which the measurement counts); at a caller's target of 1e-50
/// they are flagged, the rule picks its first rung from the measured loss,
/// and the served value is confirmed by a reduction 64 bits wider.
#[test]
fn mpfr_base_follows_the_callers_target() -> Result<(), Error> {
    let name = DEGENERATE[0];
    let lane = lane(name);
    let reference = truth(&lane);
    let out = run(&lane, Float::with_val(200, 0), true, Some(1e-50))?;
    let rescued = out.ladder.expect("flagged at 1e-50");
    assert_eq!(rescued.outcome, LadderOutcome::Accepted);
    let rule = LadderRule {
        target: Some(1e-50),
        ..LadderRule::default()
    };
    assert_eq!(
        Some(rescued.rungs[0]),
        rule.rung(200, out.lost_bits).map(|r| r.bits()),
        "{:?}",
        rescued.rungs
    );
    assert_eq!(rescued.rungs[1], rescued.rungs[0] + 64);
    assert!(
        relative(rescued.value, reference) < 1e-15,
        "f64 comparison floor"
    );
    Ok(())
}

/// MPFR base at 128 bits: the degenerate lanes keep only about 25
/// tracked bits although their moduli say about 70 bits are lost; the
/// measurement counts the tracked loss, so at a lane target of 1e-11 they are
/// flagged and re-reduced, never served at their base value.
#[test]
fn mpfr_base_tracked_loss_is_measured() -> Result<(), Error> {
    for name in DEGENERATE {
        let lane = lane(name);
        let reference = truth(&lane);
        let out = run(&lane, Float::with_val(128, 0), true, Some(1e-11))?;
        assert!(out.lost_bits > 128. - 60., "{name}: L {}", out.lost_bits);
        let rescued = out.ladder.expect("flagged");
        assert_ne!(rescued.outcome, LadderOutcome::WithinTarget, "{name}");
        if rescued.outcome == LadderOutcome::Accepted {
            let error = relative(rescued.value, reference);
            assert!(error <= 1e-11 + 1e-15, "{name}: {error:e}");
        }
    }
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
    assert!(
        out.ladder.is_none(),
        "regular lane flagged (L = {})",
        out.lost_bits
    );
    assert_eq!(out.status, "Success");
    Ok(())
}

/// No served value is more wrong than its own estimate says, at every rung
/// width: 128 bits (the rule without its MPFR tracking margin, where a lane
/// losing about 70 bits keeps far fewer bits than `p' - L'` because numerica
/// rounds every intermediate to its tracked precision: the estimate must say
/// so, and the lane climbs), and the default rule's first rungs at targets
/// `2^-60`, `2^-100`, `2^-150` (192 to 320 bits).
#[test]
fn served_values_meet_their_estimates_at_every_rung_width() -> Result<(), Error> {
    for name in DEGENERATE {
        let lane = lane(name);
        let reference = truth(&lane);
        let lost = run(&lane, DoubleFloat::from(0.0), true, None)?.lost_bits;
        let untracked = LadderRule {
            target: Some(2f64.powf(lost - 110.)),
            mpfr_tracking: 0.,
            ..LadderRule::default()
        };
        let mut cases = vec![(untracked, 128)];
        for exponent in [-60, -100, -150] {
            let rule = LadderRule {
                target: Some(2f64.powi(exponent)),
                ..LadderRule::default()
            };
            let first = rule.rung(106, lost).expect("rung").bits();
            cases.push((rule, first));
        }
        for (rule, first) in cases {
            let target = rule.target.unwrap();
            let out = run_rule(&lane, DoubleFloat::from(0.0), true, rule)?;
            let rescued = out.ladder.expect("flagged");
            assert_eq!(rescued.rungs[0], first, "{name} {target:e}: L {lost}");
            let floor = 4. * 2f64.powi(-106);
            if target < floor {
                // A value returned in double-double cannot meet a target below
                // four of its units: typed Missed, the best value carried
                // with its clamped estimate.
                assert_eq!(rescued.outcome, LadderOutcome::Missed, "{name} {target:e}");
                let (best, estimate, _) = rescued.best.expect("best");
                assert!(estimate >= floor, "{name} {target:e}: {estimate:e}");
                assert!(relative(best, reference) <= 256. * estimate + 1e-15);
                continue;
            }
            assert_eq!(
                rescued.outcome,
                LadderOutcome::Accepted,
                "{name} {target:e}: {:?}",
                rescued.rungs
            );
            let estimate = rescued.estimate.expect("measured");
            // The reported estimate is that of the value returned in
            // double-double: never below four of its units, and
            // within the requested target.
            assert!(estimate >= floor, "{name} {target:e}: {estimate:e}");
            assert!(estimate <= target, "{name} {target:e}: {estimate:e}");
            let error = relative(rescued.value, reference);
            // f64 comparison floor 1e-15; a factor 2^8 for the growth the
            // estimate leaves out.
            assert!(
                error <= 256. * estimate + 1e-15,
                "{name} {target:e}: error {error:e}, estimate {estimate:e}, rungs {:?}",
                rescued.rungs
            );
        }
    }
    Ok(())
}

/// A lane no allowed rung can serve is refused, typed, with no value, and
/// carries its best value: here the first rung is sized without the MPFR
/// tracking margin (256 bits) for a target of 1e-50 it cannot meet, and the
/// next rung would exceed the cap, so the best is the confirmed 256/320-bit
/// reduction, far better than the base pass.
#[test]
fn refused_lanes_carry_their_best_value() -> Result<(), Error> {
    let name = DEGENERATE[0];
    let lane = lane(name);
    let reference = truth(&lane);
    let rule = LadderRule {
        target: Some(1e-50),
        max_bits: Some(256),
        mpfr_tracking: 0.,
        ..LadderRule::default()
    };
    let out = run_rule(&lane, DoubleFloat::from(0.0), true, rule)?;
    let rescued = out.ladder.expect("flagged");
    assert_eq!(
        rescued.outcome,
        LadderOutcome::Missed,
        "{:?}",
        rescued.rungs
    );
    assert!(rescued.value.re.is_nan(), "a refused lane serves no value");
    let (best, estimate, bits) = rescued.best.expect("best value");
    assert!(bits > 106, "the best value is a rung's ({bits} bits)");
    let error = relative(best, reference);
    assert!(
        error <= 256. * estimate + 1e-15,
        "best {error:e} vs its estimate {estimate:e}"
    );
    assert!(
        error < relative(out.value, reference).max(1e-15),
        "better than the base pass"
    );
    Ok(())
}

/// Audit m10: a lane the double-double base pass refuses `Unstable` is served
/// unconfirmed (`WithinTarget`) only when its estimate meets the target with
/// the 8-bit margin; between the two it is re-reduced.
#[test]
fn refused_double_double_lanes_need_the_margin() -> Result<(), Error> {
    let lane = lane(DEGENERATE[0]);
    let reference = truth(&lane);
    let lost = run(&lane, DoubleFloat::from(0.0), true, Some(1e-5))?.lost_bits;
    let estimate = 2f64.powf(lost - 106.);
    let out = run(&lane, DoubleFloat::from(0.0), true, Some(estimate * 16.))?;
    assert_eq!(out.status, "Unstable");
    let rescued = out.ladder.expect("flagged");
    assert_eq!(
        rescued.outcome,
        LadderOutcome::Accepted,
        "{:?}",
        rescued.rungs
    );
    assert!(relative(rescued.value, reference) <= estimate * 16. + 1e-15);
    let loose = run(&lane, DoubleFloat::from(0.0), true, Some(estimate * 512.))?;
    assert_eq!(
        loose.ladder.expect("record").outcome,
        LadderOutcome::WithinTarget
    );
    Ok(())
}

/// A served lane is within its reported estimate
/// on EVERY Laurent coefficient (poles included), relative to the largest
/// coefficient (f64 comparison floor).
#[test]
fn served_poles_meet_the_estimate_too() -> Result<(), Error> {
    for name in DEGENERATE {
        let lane = lane(name);
        let reference = run(&lane, Float::with_val(REFERENCE_BITS, 0), false, None)?.laurent;
        let out = run(&lane, DoubleFloat::from(0.0), true, Some(1e-23))?;
        let rescued = out.ladder.expect("flagged");
        assert_eq!(rescued.outcome, LadderOutcome::Accepted, "{name}");
        let served = rescued.laurent.expect("served");
        let estimate = rescued.estimate.expect("estimate");
        let scale = reference
            .iter()
            .map(|z| z.re.hypot(z.im))
            .fold(0., f64::max);
        for k in 0..3 {
            let error =
                (served[k].re - reference[k].re).hypot(served[k].im - reference[k].im) / scale;
            assert!(
                error <= 256. * estimate + 1e-15,
                "{name} order {k}: {error:e} vs {estimate:e}"
            );
        }
    }
    Ok(())
}

/// A rung cannot create digits its inputs do not have: an MPFR point known
/// to only 20 bits is left to the base pass (value and tracked precision as
/// without the ladder: no record), so a caller's tracked-precision test still
/// refuses it; the same point at full precision is rescued (target 1e-60).
#[test]
fn inputs_of_limited_precision_bound_the_ladder() -> Result<(), Error> {
    let lane = lane(DEGENERATE[0]);
    let rule = LadderRule {
        target: Some(1e-60),
        ..LadderRule::default()
    };
    let coarse = run_inputs(&lane, Float::with_val(256, 0), true, rule, Some(20))?;
    assert!(coarse.ladder.is_none(), "a 20-bit point is not rescued");
    let full = run_inputs(&lane, Float::with_val(256, 0), true, rule, None)?;
    assert!(full.ladder.is_some(), "the full-precision point is");
    Ok(())
}

/// An order-one lane whose terms are 2^360 times
/// larger (a two-line equal-mass bubble, numerator `2^360 (v² + 2 q.v) + i`:
/// the real piece is odd under `q -> -q - v` and integrates to zero, the `i`
/// piece is the scalar bubble, finite part `-1.24460560505 i`). It is never
/// `BelowResolution` below the top rung: with the default cap (512 bits from
/// a double-double base) it climbs until its 360-bit loss fits and comes back
/// `Accepted` within its estimate; capped at 320 bits it comes back a typed
/// refusal (no value) with a FINITE absolute bound and the term moduli.
#[test]
fn huge_terms_climb_to_the_top_rung_or_are_refused_with_finite_bounds() -> Result<(), Error> {
    use ninjars::fast::tensor_layout::monomials;
    let bits = 768;
    let f = |x: f64| Float::with_val(bits, x);
    let c = |x: Float| Complex::new(x, f(0.));
    let big = f(2f64.powi(360));
    let v = [2.0, 0.75, 0.25, -0.5];
    let v2 = v[0] * v[0] - v[1] * v[1] - v[2] * v[2] - v[3] * v[3];
    let coefficients = monomials(1)
        .iter()
        .map(|alpha| match alpha.iter().position(|&p| p == 1) {
            None => Complex::new(big.clone() * &f(v2), f(1.)),
            Some(mu) => c(big.clone() * &f(2. * if mu == 0 { v[0] } else { -v[mu] })),
        })
        .collect();
    let lane = Lane {
        n: 2,
        rank: 1,
        shifts: vec![[f(0.), f(0.), f(0.), f(0.)], v.map(f)],
        masses: vec![c(f(4.)), c(f(4.))],
        s: vec![c(f(0.)), c(f(v2)), c(f(v2)), c(f(0.))],
        scale: c(f(1.)),
        coefficients,
    };
    let truth = run(&lane, Float::with_val(bits, 0), false, None)?.value;
    assert!(
        (truth.im + 1.24460560505).abs() < 1e-9 && truth.re.abs() < 1e-9,
        "{truth:?}"
    );
    let out = run(&lane, DoubleFloat::from(0.0), true, Some(1e-8))?;
    let rescued = out.ladder.expect("flagged");
    assert_eq!(
        rescued.outcome,
        LadderOutcome::Accepted,
        "{:?}",
        rescued.rungs
    );
    let estimate = rescued.estimate.expect("estimate");
    assert!(estimate <= 1e-8 && estimate.is_finite());
    assert!(
        relative(rescued.value, truth) <= 256. * estimate + 1e-15,
        "{:?} vs {truth:?}",
        rescued.value
    );
    let capped = LadderRule {
        target: Some(1e-8),
        max_bits: Some(320),
        ..LadderRule::default()
    };
    let out = run_rule(&lane, DoubleFloat::from(0.0), true, capped)?;
    let rescued = out.ladder.expect("flagged");
    assert!(
        matches!(
            rescued.outcome,
            LadderOutcome::BelowResolution | LadderOutcome::Missed
        ),
        "{:?} {:?}",
        rescued.outcome,
        rescued.rungs
    );
    assert!(rescued.value.re.is_nan(), "no value served");
    let bound = rescued.absolute_bound.expect("bound");
    let moduli = rescued.term_moduli.expect("moduli");
    assert!(
        bound
            .iter()
            .chain(&moduli)
            .all(|x| x.is_finite() && *x >= 0.),
        "{bound:?} {moduli:?}"
    );
    assert!(rescued.estimate.is_some_and(f64::is_finite));
    assert!(rescued.best.is_some_and(|(_, e, _)| e.is_finite()));
    Ok(())
}

/// An equal-mass bubble `N = 2^k (1 + 2 q0) + i` (shifts 0 and
/// (1, 0, 0, 0), `m² = 4`, `μ² = 1`), the numerator scaled by `2^scale`.
fn scaled_bubble(k: i32, scale: i32) -> Lane {
    use ninjars::fast::tensor_layout::monomials;
    let bits = 2048;
    let f = |x: f64| Float::with_val(bits, x);
    let c = |x: Float| Complex::new(x, f(0.));
    let two = |e: i32| {
        let mut x = f(1.);
        for _ in 0..e.unsigned_abs() {
            x = if e > 0 { x * &f(2.) } else { x / &f(2.) };
        }
        x
    };
    let (big, unit) = (two(k + scale), two(scale));
    let coefficients = monomials(1)
        .iter()
        .map(|alpha| match alpha.iter().position(|&p| p == 1) {
            None => Complex::new(big.clone(), unit.clone()),
            Some(0) => c(big.clone() * &f(2.)),
            Some(_) => c(f(0.)),
        })
        .collect();
    Lane {
        n: 2,
        rank: 1,
        shifts: vec![[f(0.), f(0.), f(0.), f(0.)], [f(1.), f(0.), f(0.), f(0.)]],
        masses: vec![c(f(4.)), c(f(4.))],
        s: vec![c(f(0.)), c(f(1.)), c(f(1.)), c(f(0.))],
        scale: c(f(1.)),
        coefficients,
    }
}

/// With the numerator scaled by 2^-1074 the
/// binary64 value of the lane is subnormal (`-1.3435 i 2^-1074` cannot be
/// represented within 1e-8): it must not be `Accepted` (it was, with
/// estimate 4.4e-16 and a 25.6 % error); the narrowing error is formed as a
/// ratio in MPFR, so the lane is a typed refusal. With k = 4 the lane does
/// not cancel at all: the base trigger counts the subnormal spacing, so the
/// lane is not served as a plain success either.
#[test]
fn subnormal_values_are_not_accepted() -> Result<(), Error> {
    for k in [4, 80, 160] {
        let lane = scaled_bubble(k, -1074);
        let out = run(&lane, 0f64, true, Some(1e-8))?;
        let rescued = out.ladder.expect("flagged");
        assert_ne!(
            rescued.outcome,
            LadderOutcome::Accepted,
            "k = {k}: {:?}",
            rescued.rungs
        );
        assert!(rescued.value.re.is_nan(), "k = {k}: no value served");
        assert!(
            rescued.best.is_some_and(|(_, e, _)| e > 1e-8),
            "k = {k}: honest best estimate"
        );
    }
    Ok(())
}

/// Scale invariance: the ladder's decisions (outcome, rungs) are unchanged
/// when the numerator is multiplied by 2^1000 (nothing overflows in
/// binary64: coefficients up to 2^1017), and the value scales with it.
#[test]
fn decisions_are_scale_invariant() -> Result<(), Error> {
    let plain = run(&scaled_bubble(16, 0), 0f64, true, Some(1e-14))?;
    let scaled = run(&scaled_bubble(16, 1000), 0f64, true, Some(1e-14))?;
    let (a, b) = (
        plain.ladder.expect("flagged"),
        scaled.ladder.expect("flagged"),
    );
    assert_eq!(a.outcome, b.outcome);
    assert_eq!(a.rungs, b.rungs);
    assert_eq!(a.outcome, LadderOutcome::Accepted);
    let back = Complex::new(b.value.re * 2f64.powi(-1000), b.value.im * 2f64.powi(-1000));
    assert!(
        relative(back, a.value) <= 1e-14,
        "{back:?} vs {:?}",
        a.value
    );
    assert!((plain.lost_bits - scaled.lost_bits).abs() < 1e-9);
    Ok(())
}
