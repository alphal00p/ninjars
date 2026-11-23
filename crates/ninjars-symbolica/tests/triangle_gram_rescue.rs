//! The fast engine's double-double re-reduction of cancelling lanes
//! (`FastOptions::cancellation_rescue`, `docs/triangle-gram-conditioning.md`)
//! on pyAmpliCol `g g > g g g` lanes with a degenerate two-mass triangle
//! (`validation/conditioning/triangle-gram`), its regular control, and its
//! double-double masters against OneLOop's binary64 ones.

#[path = "common/conditioning.rs"]
mod conditioning;

use conditioning::{Error, fast, parse_fixture};
use ninjars::fast::dd::Dd;
use ninjars::fast::{
    C64, FastMaster, FastMasters, FastOptions, bubble_tensors, massless_b0, massless_c0,
    take_rescued_lanes,
};
use ninjars_symbolica::OneLoopNativeMasters;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../validation/conditioning/triangle-gram");

fn fixture(name: &str) -> conditioning::Fixture {
    let text = std::fs::read_to_string(format!("{DIR}/{name}.json")).expect("fixture");
    parse_fixture(&text).expect("fixture shape")
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
        let served = fast(&f, FastOptions::default())?[0].laurent.expect("success")[0];
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
    let on = fast(&f, FastOptions::default())?[0].laurent.expect("success");
    let off = fast(&f, without_rescue())?[0].laurent.expect("success");
    assert_eq!(take_rescued_lanes(), 0);
    assert_eq!(on, off);
    Ok(())
}

fn close(a: C64, b: C64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance * b.abs().max(1e-300) || (a - b).abs() < 1e-300
}

/// The double-double massless `B0`, `C0` (one and two masses, space- and
/// time-like) and the Passarino-Veltman bubble tensors agree with OneLOop's
/// binary64 values (OneLOop's own two-mass `C0` loses digits as the masses
/// approach each other: 3e-13 at the degenerate point of the first case).
#[test]
fn double_double_masters_match_oneloop() -> Result<(), Error> {
    let mut masters = OneLoopNativeMasters::new()?;
    let mu2 = C64::real(8315.251344);
    let z = C64::ZERO;
    for (p1, p2, p3) in [
        (0., -1.9342957622827255e5, -1.9340112758684525e5),
        (0., 3.0e5, 2.9e5),
        (0., -1.0e5, 2.0e5),
        (0., 0., -4.0e4),
        (0., 0., 5.0e4),
        (1.0e5, 0., -3.0e4),
    ] {
        let mut out = vec![[z; 3]; 1];
        let mut ok = vec![false];
        let args = [C64::real(p1), C64::real(p2), C64::real(p3), z, z, z, mu2];
        masters.evaluate(FastMaster::C0, &args, &mut out, &mut ok)?;
        let dd = massless_c0(
            [Dd::from(p1), Dd::from(p2), Dd::from(p3)],
            [p1 == 0., p2 == 0., p3 == 0.],
            mu2.re,
        )
        .expect("one or two masses");
        for o in 0..3 {
            assert!(close(dd[o].to_c64(), out[0][o], 1e-12), "C0 ({p1}, {p2}, {p3}) order {o}");
        }
    }
    for (p2, m0, m1) in [
        (-1.9342957622827255e5, 0., 0.),
        (3.1e5, 0., 0.),
        (4.0e5, 173. * 173., 173. * 173.),
        (-2.0e5, 173. * 173., 0.),
        (9.0e4, 0., 91. * 91.),
    ] {
        let (m0, m1) = (C64::real(m0), C64::real(m1));
        let mut out = vec![[z; 3]; 4];
        let mut ok = vec![false];
        masters.evaluate(FastMaster::BubbleRank3, &[C64::real(p2), m0, m1, mu2], &mut out, &mut ok)?;
        let a0 = |m: C64, masters: &mut OneLoopNativeMasters| -> [C64; 3] {
            if m.is_zero() {
                return [z; 3];
            }
            let mut o = vec![[z; 3]; 1];
            let mut k = vec![false];
            masters.evaluate(FastMaster::A0, &[m, mu2], &mut o, &mut k).expect("A0");
            o[0]
        };
        let (a0m0, a0m1) = (a0(m0, &mut masters), a0(m1, &mut masters));
        let b0dd = (m0.is_zero() && m1.is_zero()).then(|| massless_b0(Dd::from(p2), mu2.re));
        if let Some(b0) = &b0dd {
            for o in 0..3 {
                assert!(close(b0[o].to_c64(), out[0][o], 1e-14), "B0 {p2} order {o}");
            }
        }
        let tensors = bubble_tensors(Dd::from(p2), ninjars::fast::dd::Cd::from_c64(m0), ninjars::fast::dd::Cd::from_c64(m1), &out[0], &a0m0, &a0m1, b0dd)
            .expect("well-conditioned reduction");
        for (j, tensor) in tensors.iter().enumerate() {
            for o in 0..3 {
                assert!(
                    close(tensor[o].to_c64(), out[j][o], 1e-13),
                    "B tensor {j} at ({p2}, {m0:?}, {m1:?}) order {o}: {:?} vs {:?}",
                    tensor[o].to_c64(),
                    out[j][o]
                );
            }
        }
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
        let refined = run(FastOptions::default(), Some(&consistent))?;
        let unrepaired = run(FastOptions::default(), Some(&zero))?;
        let err = |v: C64| (v - reference).abs() / reference.abs();
        assert!(err(refined) < 1e-8, "{name}: refined {refined:?} error {:e}", err(refined));
        assert!(err(unrepaired) > 1e-3, "{name}: an unrepaired refinement was repaired ({:e})", err(unrepaired));
        let off = FastOptions { cancellation_rescue: false, ..FastOptions::default() };
        assert_eq!(run(off, Some(&consistent))?, run(off, None)?, "{name}: the binary64 pass read the refinement");
    }
    Ok(())
}
