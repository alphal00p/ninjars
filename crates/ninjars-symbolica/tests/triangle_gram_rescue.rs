//! The fast engine's double-double re-reduction of cancelling lanes
//! (`FastOptions::cancellation_rescue`, `docs/triangle-gram-conditioning.md`)
//! on pyAmpliCol `g g > g g g` lanes with a degenerate two-mass triangle
//! (`validation/conditioning/triangle-gram`), its regular control, and its
//! double-double masters against OneLOop's binary64 ones.

#[path = "common/conditioning.rs"]
mod conditioning;

use conditioning::{Error, fast, parse_fixture};
use ninjars::fast::dd::Dd;
use ninjars::fast::dd::Cd;
use ninjars::fast::{C64, FastMaster, FastMasters, FastOptions, take_rescued_lanes};
use ninjars_symbolica::OneLoopNativeMasters;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../validation/conditioning/triangle-gram");

fn fixture(name: &str) -> conditioning::Fixture {
    let text = std::fs::read_to_string(format!("{DIR}/{name}.json")).expect("fixture");
    parse_fixture(&text).expect("fixture shape")
}

/// The rescue on (NinjaRS's default is off) at pyAmpliCol's lane target
/// (`reduction_target / 1000` = 1e-8).
fn rescue() -> FastOptions {
    FastOptions {
        cancellation_rescue: true,
        rescue_target: Some(1e-8),
        ..FastOptions::default()
    }
}

fn without_rescue() -> FastOptions {
    FastOptions {
        cancellation_rescue: false,
        ..FastOptions::default()
    }
}

/// Lanes whose binary64 reduction loses every digit (relative errors 1e3 to
/// 1e6) come back within 1e-8 of the reference, each through one
/// double-double re-reduction.
#[test]
fn degenerate_triangle_lanes_are_rescued() -> Result<(), Error> {
    for name in ["ggg-308-n3-r3", "ggg-308-n4-r4", "ggg-9117-n3-r3"] {
        let f = fixture(name);
        let r = f.references[0].as_ref().expect("reference");
        let reference = C64::new(r[0].parse()?, r[1].parse()?);
        take_rescued_lanes();
        let served = fast(&f, rescue())?[0].laurent.expect("success")[0];
        assert_eq!(take_rescued_lanes(), 1, "{name}");
        let binary64 = fast(&f, without_rescue())?[0].laurent.expect("success")[0];
        assert_eq!(take_rescued_lanes(), 0, "{name}");
        let error = (served - reference).abs() / reference.abs();
        let before = (binary64 - reference).abs() / reference.abs();
        assert!(error < 1e-8, "{name}: served {served:?}, reference {reference:?}, error {error:e}");
        assert!(before > 1e-2, "{name}: the binary64 reduction no longer fails ({before:e})");
    }
    Ok(())
}

/// A regular lane is not re-reduced: its value is the binary64 reduction's,
/// bit for bit.
#[test]
fn regular_lanes_are_untouched() -> Result<(), Error> {
    let f = fixture("ggg-0-n5-r5-control");
    take_rescued_lanes();
    let on = fast(&f, rescue())?[0].laurent.expect("success");
    let off = fast(&f, without_rescue())?[0].laurent.expect("success");
    assert_eq!(take_rescued_lanes(), 0);
    assert_eq!(on, off);
    Ok(())
}

fn close(a: C64, b: C64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance * b.abs().max(1e-300) || (a - b).abs() < 1e-300
}

/// `OneLoopNativeMasters::evaluate_rescue` (oneloopmaster's native masters in
/// double-double) agrees with its binary64 `evaluate` on massless, massive,
/// complex-mass and mixed rows of every family it provides; rank-four bubbles
/// are not provided.
#[test]
fn double_double_masters_match_the_binary64_ones() -> Result<(), Error> {
    let mut masters = OneLoopNativeMasters::new()?.with_onshell_threshold(None);
    let mu2 = C64::real(8315.251344);
    let (z, t, w) = (C64::ZERO, C64::real(29929.), C64::new(29929., -238.32134));
    let r = C64::real;
    let rows: Vec<(FastMaster, Vec<C64>)> = vec![
        (FastMaster::A0, vec![t, mu2]),
        (FastMaster::A0, vec![w, mu2]),
        (FastMaster::BubbleRank2, vec![r(-1.9342957622827255e5), z, z, mu2]),
        (FastMaster::BubbleRank3, vec![r(3.1e5), z, z, mu2]),
        (FastMaster::BubbleRank3, vec![r(4.0e5), t, t, mu2]),
        (FastMaster::BubbleRank3, vec![r(-2.0e5), t, z, mu2]),
        (FastMaster::BubbleRank3, vec![r(6000.8), z, t, mu2]),
        (FastMaster::BubbleRank3, vec![r(5.0e5), w, r(6400.), mu2]),
        (FastMaster::C0, vec![z, r(-1.9342957622827255e5), r(-1.9340112758684525e5), z, z, z, mu2]),
        (FastMaster::C0, vec![z, r(3.0e5), r(2.9e5), z, z, z, mu2]),
        (FastMaster::C0, vec![r(1.0e5), z, r(-3.0e4), z, z, z, mu2]),
        (FastMaster::C0, vec![z, r(6000.8), r(6066.3), z, t, z, mu2]),
        (FastMaster::C0, vec![r(-1.9e5), t, r(-1.8e5), z, t, z, mu2]),
        (FastMaster::C0, vec![r(2.0e5), r(1.5e5), r(3.0e5), w, w, w, mu2]),
        (FastMaster::D0, vec![z, z, z, z, r(4.0e5), r(-1.5e5), z, z, z, z, mu2]),
        (FastMaster::D0, vec![z, z, t, t, r(5.0e5), r(-2.0e5), z, z, t, t, mu2]),
        (FastMaster::D0, vec![z, z, r(1.6e4), r(1.6e4), r(4.5e5), r(-1.2e5), w, w, w, w, mu2]),
    ];
    for (family, args) in rows {
        let width = family.outputs();
        let mut out = vec![[z; 3]; width];
        let mut ok = vec![false];
        masters.evaluate(family, &args, &mut out, &mut ok)?;
        let dargs: Vec<Cd> = args.iter().map(|&x| Cd::from_c64(x)).collect();
        let mut dout = vec![[Cd::ZERO; 3]; width];
        let mut dok = vec![false];
        assert!(masters.evaluate_rescue(family, &dargs, &mut dout, &mut dok)?, "{family:?}");
        assert!(ok[0] && dok[0], "{family:?} {args:?}");
        for j in 0..width {
            for o in 0..3 {
                assert!(
                    close(dout[j][o].to_c64(), out[j][o], 1e-11),
                    "{family:?} {args:?} [{j}][{o}]: {:?} vs {:?}",
                    dout[j][o].to_c64(),
                    out[j][o]
                );
            }
        }
    }
    let mut dout = vec![[Cd::ZERO; 3]; 5];
    let row = [Cd::real(Dd::from(1e5)), Cd::ZERO, Cd::ZERO, Cd::from_c64(mu2)];
    assert!(!masters.evaluate_rescue(FastMaster::BubbleRank4, &row, &mut dout, &mut [false])?);
    Ok(())
}

/// The box level: loop-induced `g g > a a g` pentagon lanes 1e-9 rad out of
/// the planar configuration, which the binary64 box mu-expansion screen
/// refuses, are re-reduced in double-double (boxes included) and served within
/// 1e-9 of the fixture's references.
#[test]
fn box_mu_screened_lanes_are_cured() -> Result<(), Error> {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../validation/conditioning/box-mu/gg_aag-planar-1e-9.json"
    ))?;
    let f = parse_fixture(&text)?;
    let off = fast(&f, without_rescue())?;
    let on = fast(&f, rescue())?;
    for (lane, (a, b)) in off.iter().zip(&on).enumerate() {
        assert!(!a.success, "lane {lane}: the binary64 screen no longer refuses it");
        let r = f.references[lane].as_ref().expect("reference");
        let reference = C64::new(r[0].parse()?, r[1].parse()?);
        let served = b.laurent.expect("served")[0];
        let error = (served - reference).abs() / reference.abs();
        assert!(error < 1e-9, "lane {lane}: {served:?} against {reference:?} ({error:e})");
    }
    Ok(())
}

/// `FastAmplitude::evaluate_refined` (`docs/triangle-gram-conditioning.md`,
/// section 7): with the low parts of the exactly consistent kinematics the
/// rescued lanes are served within 1e-8 of their references; with all-zero
/// low parts (a refinement that claims the binary64 shifts are exact, their
/// light-like legs carrying the rounding of their components as a mass)
/// NinjaRS repairs nothing and the rescue stays at the binary64 level; the
/// binary64 pass is the same either way (checked with the rescue off).
#[test]
fn refined_kinematics_drive_the_rescue() -> Result<(), Error> {
    use ninjars::fast::{
        FastAmplitude, FastInput, FastRefinement, FastResult, FastStatus, RMom, TensorNumerator,
    };
    use ninjars_symbolica::OneLoopNativeMasters;
    for name in ["ggg-308-n3-r3", "ggg-308-n4-r4"] {
        let text = std::fs::read_to_string(format!("{DIR}/{name}.json"))?;
        let f = parse_fixture(&text)?;
        let raw: serde_json::Value = serde_json::from_str(&text)?;
        let num = |v: &serde_json::Value| -> f64 { v.as_str().unwrap().parse().unwrap() };
        let n = f.n;
        let shifts: Vec<RMom> = f.shifts.iter().map(|v| v.clone().map(|x| x.parse().unwrap())).collect();
        let masses: Vec<C64> = f.masses_squared.iter().map(|m| C64::new(m[0].parse().unwrap(), m[1].parse().unwrap())).collect();
        let s_matrix: Vec<f64> = f.s_matrix.iter().map(|x| x.parse().unwrap()).collect();
        let tensor: Vec<C64> = f.tensors[0].iter().map(|z| C64::new(z[0].parse().unwrap(), z[1].parse().unwrap())).collect();
        let lo_shifts: Vec<RMom> = raw["refinement"]["shifts"].as_array().unwrap().iter()
            .map(|v| std::array::from_fn(|mu| num(&v[mu]))).collect();
        let lo_s: Vec<f64> = raw["refinement"]["s_matrix"].as_array().unwrap().iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(num).collect::<Vec<_>>()).collect();
        let zeros_shifts = vec![[0.; 4]; n];
        let zeros_s = vec![0.; n * n];
        let r = f.references[0].as_ref().unwrap();
        let reference = C64::new(r[0].parse()?, r[1].parse()?);
        let scales = [f.mu_squared.parse::<f64>()?];
        let input = FastInput { lanes: 1, shifts: &shifts, masses_squared: &masses, scales_squared: &scales, s_matrix: Some(&s_matrix) };
        let run = |options: FastOptions, refinement: Option<&FastRefinement>| -> Result<C64, Error> {
            let mut amp = FastAmplitude::new(n, f.rank, options)?;
            let mut numerator = TensorNumerator::new(f.rank, 0);
            numerator.set_coefficients(1, &tensor)?;
            let mut masters = OneLoopNativeMasters::new()?;
            let mut results = [FastResult::default()];
            let mut statuses = [FastStatus::NotEvaluated];
            amp.evaluate_refined(&input, refinement, &mut numerator, &mut masters, &mut results, &mut statuses)?;
            assert_eq!(statuses[0], FastStatus::Success);
            Ok(results[0].laurent[0])
        };
        let consistent = FastRefinement { shifts: &lo_shifts, s_matrix: Some(&lo_s), masses_squared: None };
        let zero = FastRefinement { shifts: &zeros_shifts, s_matrix: Some(&zeros_s), masses_squared: None };
        let refined = run(rescue(), Some(&consistent))?;
        let unrepaired = run(rescue(), Some(&zero))?;
        let err = |v: C64| (v - reference).abs() / reference.abs();
        assert!(err(refined) < 1e-8, "{name}: refined {refined:?} error {:e}", err(refined));
        // An inconsistent refinement (zero low parts for inexact binary64
        // momenta) is not trusted: the lane takes the projection.
        assert!(err(unrepaired) < 1e-8, "{name}: an inconsistent refinement was trusted ({:e})", err(unrepaired));
        // Without an invariant matrix the projection serves the lane too.
        let nos = FastInput { s_matrix: None, ..input };
        let mut amp = FastAmplitude::new(n, f.rank, rescue())?;
        let mut numerator = TensorNumerator::new(f.rank, 0);
        numerator.set_coefficients(1, &tensor)?;
        let mut masters = OneLoopNativeMasters::new()?;
        let mut results = [FastResult::default()];
        let mut statuses = [FastStatus::NotEvaluated];
        amp.evaluate(&nos, &mut numerator, &mut masters, &mut results, &mut statuses)?;
        assert_eq!(statuses[0], FastStatus::Success, "{name} without s_matrix");
        assert!(err(results[0].laurent[0]) < 1e-8, "{name} without s_matrix: {:e}", err(results[0].laurent[0]));
        let off = FastOptions { cancellation_rescue: false, ..rescue() };
        assert_eq!(run(off, Some(&consistent))?, run(off, None)?, "{name}: the binary64 pass read the refinement");
    }
    Ok(())
}

/// The target accuracy (`FastOptions::rescue_target`) decides: at the default
/// the dumped lane is re-reduced and served (one `Rescue` record, 106 bits,
/// served, its estimate within the target); a target the binary64 lane already
/// meets (an infinite one) re-reduces nothing; a target the re-reduction cannot meet refuses
/// the lane, typed (`Unstable`, `InstabilityTest::CancellationRescue`).
#[test]
fn the_target_accuracy_decides_and_misses_are_typed() -> Result<(), Error> {
    use ninjars::fast::{DEFAULT_RESCUE_TARGET, InstabilityTest, take_instabilities, take_rescues};
    let f = fixture("ggg-308-n3-r3");
    let with = |target: Option<f64>| FastOptions {
        rescue_target: target,
        ..rescue()
    };
    let served = fast(&f, with(None))?;
    let records = take_rescues();
    assert!(served[0].success);
    assert_eq!(records.len(), 1);
    let (lane, rescue) = records[0];
    assert_eq!((lane, rescue.precision_bits, rescue.served), (0, 106, true));
    assert!(rescue.estimate <= DEFAULT_RESCUE_TARGET, "{rescue:?}");
    fast(&f, with(Some(f64::INFINITY)))?;
    assert!(take_rescues().is_empty());
    let refused = fast(&f, with(Some(1e-30)))?;
    assert!(!refused[0].success, "{}", refused[0].status);
    let records = take_rescues();
    assert_eq!(records.len(), 1);
    assert!(!records[0].1.served);
    let instabilities = take_instabilities();
    assert!(
        instabilities
            .iter()
            .any(|(lane, i)| *lane == 0 && i.test == InstabilityTest::CancellationRescue),
        "{instabilities:?}"
    );
    Ok(())
}

/// Beyond double-double: lanes whose box is too degenerate for the
/// re-reduction (`validation/conditioning/box-gram`, collinear final state
/// tilted by 1e-6, rank n+1), which binary64 serves 1e15 off as successes,
/// are refused, typed, so that a caller's own rescue can act.
#[test]
fn lanes_beyond_double_double_are_refused_typed() -> Result<(), Error> {
    use ninjars::fast::{InstabilityTest, take_instabilities};
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../validation/conditioning/box-gram/collinear-ext-n4-r5-m0-t1e-6.json"
    ))?;
    let f = parse_fixture(&text)?;
    let off = fast(&f, without_rescue())?;
    assert!(off.iter().all(|o| o.success));
    let on = fast(&f, rescue())?;
    let records = take_instabilities();
    let rescues = ninjars::fast::take_rescues();
    for (lane, o) in on.iter().enumerate() {
        assert!(!o.success, "lane {lane}: {}", o.status);
        assert!(
            records
                .iter()
                .any(|(l, i)| *l == lane && i.test == InstabilityTest::CancellationRescue),
            "{records:?}"
        );
        // The refused lane's slot carries the best available value, the
        // record says which carrier and its estimate.
        let (_, r) = rescues.iter().find(|(l, _)| *l == lane).expect("record");
        assert!(!r.served && r.outcome == ninjars::fast::RescueOutcome::Missed, "{r:?}");
        assert!(r.held_bits == 53 || r.held_bits == 106, "{r:?}");
        assert!(r.held_estimate.is_finite() && r.held_estimate > 0., "{r:?}");
        let binary64 = off[lane].laurent.expect("served")[0];
        if r.held_bits == 53 {
            assert_eq!(o.slot[0], binary64, "lane {lane}");
        } else {
            assert!(r.held_estimate == r.estimate && r.absolute.is_finite(), "{r:?}");
        }
        assert!(o.slot.iter().all(|z| z.re.is_finite() && z.im.is_finite()) && o.slot[0] != C64::ZERO);
    }
    Ok(())
}

/// A lane whose re-reduction's own estimate misses the target is refused,
/// typed, even when its value is below binary64's resolution of its own terms
/// (no absolute scale of the point is known): the dumped lane of point 9117
/// at the default target 2^-30 (estimate about 6e-9). Its slot keeps the
/// re-reduced value, the better one, with that carrier and estimate in the
/// records; the binary64 value is 1e5 off.
#[test]
fn lanes_missing_the_target_are_refused_with_the_better_value() -> Result<(), Error> {
    use ninjars::fast::{InstabilityTest, RescueOutcome, take_instabilities, take_rescues};
    let f = fixture("ggg-9117-n3-r3");
    let r = f.references[0].as_ref().expect("reference");
    let reference = C64::new(r[0].parse()?, r[1].parse()?);
    let off = fast(&f, without_rescue())?;
    let strict = FastOptions { rescue_target: None, ..rescue() };
    let refused = fast(&f, strict)?;
    let (_, rec) = take_rescues().into_iter().find(|(l, _)| *l == 0).expect("record");
    let records = take_instabilities();
    assert!(!refused[0].success && rec.outcome == RescueOutcome::Missed && !rec.served, "{rec:?}");
    assert!(rec.estimate > ninjars::fast::DEFAULT_RESCUE_TARGET, "{rec:?}");
    assert_eq!(rec.held_bits, 106, "{rec:?}");
    let held = refused[0].slot[0];
    assert!((held - reference).abs() < 1e-8 * reference.abs(), "{held:?}");
    let binary64 = off[0].laurent.expect("served")[0];
    assert!((binary64 - reference).abs() > 1e-2 * reference.abs());
    let record = records
        .iter()
        .find(|(l, i)| *l == 0 && i.test == InstabilityTest::CancellationRescue)
        .expect("typed");
    assert_eq!(record.1.value, rec.held_estimate);
    Ok(())
}

/// A lane whose binary64 value is garbage from inside the stages (audit2's
/// `a2-n3-d1e-05-T-mixed-r4`: time-like, δ = 3e-5, top mass on alternate
/// lines, rank n+1; binary64 8e19 off as a success): the term-moduli estimate
/// divides by that garbage and stays small, the value-independent
/// triangle-Gram term `2^-53/δ²` flags it at pyAmpliCol's target, and the
/// re-reduction, beyond double-double here, refuses it typed.
#[test]
fn garbage_lanes_are_flagged_by_the_value_independent_term() -> Result<(), Error> {
    use ninjars::fast::{RescueOutcome, take_rescues};
    let f = fixture("audit2-n3-T-mixed-r4");
    let r = f.references[0].as_ref().expect("reference");
    let reference = C64::new(r[0].parse()?, r[1].parse()?);
    let off = fast(&f, without_rescue())?;
    let binary64 = off[0].laurent.expect("served")[0];
    assert!((binary64 - reference).abs() > 1e10 * reference.abs(), "{binary64:?}");
    let on = fast(&f, rescue())?;
    let (_, rec) = take_rescues().into_iter().find(|(l, _)| *l == 0).expect("flagged");
    assert!(!on[0].success, "{}", on[0].status);
    assert!(!rec.served && matches!(rec.outcome, RescueOutcome::Missed | RescueOutcome::Failed), "{rec:?}");
    Ok(())
}
