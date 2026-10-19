//! V5 (design section 7): rank padding.
//!
//! Every rank <= n+1 bench case with an original-Ninja result
//! (`validation/rank-n2/identity/padding-manifest.json`: the Ninja reference
//! cases, the benchmark and synthetic cases and the rank-n2 control twins) is
//! declared at a higher rank, i.e. its numerator gets zero top coefficients,
//! and must give the same integrals:
//! - fast engine (`TensorNumerator` padded): within `max(1e-11, 10 noise)` of
//!   original Ninja with Ninja's own master values where the Ninja result
//!   lists its master calls (as `tests/fast_ninja_reference.rs`), and of the
//!   unpadded fast engine with identical native masters everywhere. `noise`
//!   is the lane's binary64 noise, the change of the unpadded result when the
//!   propagators are listed in reverse order (as `tests/fast_parity.rs`):
//!   padding changes the rounding of ill-conditioned lanes (small-p^2 edge
//!   cases) at that level. At rank n+2 the engine also chooses its own
//!   bubble references and tadpole bases (`fast/bases_x2.rs`), so where the
//!   unpadded result (or Ninja's) is less accurate than that tolerance, the
//!   padded one must instead be within ten times its error of the exact
//!   reduction (generic engine in Float(256) with exact masters);
//! - generic engine in Float(256): within 1e-50 of the unpadded result. At
//!   rank n+2 the engine conditions its bubble references and tadpole bases
//!   for the rank n+2 depth (`cuts.rs::wide_bubble_reference`), so padded
//!   and unpadded lanes round differently; as for the fast engine, where the
//!   unpadded result is less accurate than 1e-50, the padded one must
//!   instead be within ten times its error of the exact reduction (Float(512)).
//!   This comparison takes the invariants of the shifts: on a lane whose
//!   `s_matrix` is not exactly those invariants (snapped `phys` and `syn`
//!   S-matrices), the integral is defined only up to how a code combines the
//!   two (`validation/rank-n2/README.md`, "Input ambiguity"), and two bubble
//!   references are two such combinations.
//!
//! Declared rank n+2 exercises the rank n+2 paths with rank <= n+1
//! numerators. Declared rank n+1 for the rank <= n cases exercises the rank
//! n+1 paths with zero top coefficients.

mod common;
mod rank_n2_reference_support;

use common::ninja_reference::{NinjaLane, NinjaMasters, parse_ninja};
use ninjars::fast::{C64, FastMaster, FastMasters, FastStatus};
use ninjars_symbolica::OneLoopNativeMasters;
use rank_n2_reference_support::{
    fast_noise,
    harness::{BenchCase, Engines, fast_values, run_fast, scaled_difference},
    rank_n2, root,
};
use serde_json::Value;
use symbolica::domains::float::RealLike;

const FAST_TOLERANCE: f64 = 1e-11;
const FLOAT_TOLERANCE: f64 = 1e-50;
const BITS: u32 = 256;
/// The exact reduction of a lane whose Float(256) result is less accurate
/// than `FLOAT_TOLERANCE`.
const EXACT_BITS: u32 = 512;

struct Entry {
    case: BenchCase,
    ninja: Option<Vec<NinjaLane>>,
}

/// The manifest's cases with `rank <= max_excess + n`, and original Ninja's
/// lanes where its master calls are recorded.
fn entries(max_excess: i64) -> Vec<Entry> {
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(rank_n2().join("identity/padding-manifest.json")).unwrap(),
    )
    .unwrap();
    let mut out = Vec::new();
    for row in manifest["cases"].as_array().unwrap() {
        let case = BenchCase::load(&root().join(row["case"].as_str().unwrap())).unwrap();
        if case.rank as i64 - case.n as i64 > max_excess {
            continue;
        }
        let ninja_path = root().join(row["ninja"].as_str().unwrap());
        let text = std::fs::read_to_string(&ninja_path).unwrap();
        let masters_path = ninja_path.with_extension("masters.json");
        let masters = std::fs::read_to_string(masters_path).ok();
        let ninja = parse_ninja(&text, masters.as_deref()).ok();
        out.push(Entry { case, ninja });
    }
    assert!(!out.is_empty());
    out
}

/// Original Ninja's master values where a request matches one of its calls,
/// the native masters otherwise (counted).
struct NinjaFirst {
    ninja: NinjaMasters,
    native: OneLoopNativeMasters,
    fallbacks: usize,
}

impl FastMasters for NinjaFirst {
    fn evaluate(
        &mut self,
        family: FastMaster,
        args: &[C64],
        out: &mut [[C64; 3]],
        ok: &mut [bool],
    ) -> ninjars::Result<()> {
        self.ninja.evaluate(family, args, out, ok)?;
        let (arity, width) = (family.arity(), family.outputs());
        for row in 0..ok.len() {
            if !ok[row] {
                self.fallbacks += 1;
                let mut row_ok = [false];
                self.native.evaluate(
                    family,
                    &args[row * arity..(row + 1) * arity],
                    &mut out[row * width..(row + 1) * width],
                    &mut row_ok,
                )?;
                ok[row] = row_ok[0];
            }
        }
        Ok(())
    }
}

/// Laurent coefficients of the exact reduction of every lane of `case`
/// (generic engine in Float(256) with exact masters; `None` where it does
/// not succeed), computed on first use.
fn exact_laurent<'a>(
    cache: &'a mut Option<Vec<Option<[C64; 3]>>>,
    engines: &mut Engines,
    case: &BenchCase,
) -> &'a [Option<[C64; 3]>] {
    cache.get_or_insert_with(|| {
        let (engine, masters) = engines.float(BITS).unwrap();
        engine
            .evaluate(case, case.rank, masters)
            .unwrap()
            .iter()
            .map(|lane| {
                (lane.status == ninjars::EvaluationStatus::Success).then(|| {
                    std::array::from_fn(|k| {
                        C64::new(lane.values[k].re.to_f64(), lane.values[k].im.to_f64())
                    })
                })
            })
            .collect()
    })
}

/// Fast engine, padded to `declared(case)`: against original Ninja with its
/// master values, and against the unpadded engine with native masters.
fn fast_padding(entries: &[Entry], declared: impl Fn(&BenchCase) -> usize) {
    let mut native = OneLoopNativeMasters::new().unwrap();
    let mut engines = Engines::default();
    let (mut worst_ninja, mut worst_self, mut lanes, mut fallbacks) = (0_f64, 0_f64, 0, 0);
    let (mut within_noise, mut exact_checked, mut worst_exact_ratio) = (0, 0, 0_f64);
    for entry in entries {
        let case = &entry.case;
        let mut exact = None;
        // Where `d` exceeds `tolerance`: the padded result `a` must be within
        // ten times the reference's (`b`) error of the exact reduction.
        let mut check_exact = |lane: usize, a: &[C64], b: &[C64], scale: f64, label: &str| {
            let e = exact_laurent(&mut exact, &mut engines, case)[lane]
                .unwrap_or_else(|| panic!("{label}: no exact reduction"));
            let error = |v: &[C64]| (0..3).map(|k| (v[k] - e[k]).abs()).fold(0., f64::max) / scale;
            let (padded, reference) = (error(a), error(b));
            assert!(
                padded <= FAST_TOLERANCE.max(10. * reference),
                "{label}: against the exact reduction {padded:e}, the reference {reference:e}"
            );
            padded / reference.max(1e-300)
        };
        let rank = declared(case);
        let name = format!("{} declared rank {rank}", case.name);
        let original = run_fast(case, case.rank, case.lanes.len(), &mut native).unwrap();
        let padded = run_fast(case, rank, case.lanes.len(), &mut native).unwrap();
        let noise = fast_noise(case, case.rank, &mut native);
        let tolerance = |lane: usize| FAST_TOLERANCE.max(10. * noise[lane]);
        #[allow(clippy::needless_range_loop)]
        for lane in 0..case.lanes.len() {
            assert_eq!(
                padded.statuses[lane], original.statuses[lane],
                "{name} lane {lane}"
            );
            if original.statuses[lane] != FastStatus::Success {
                continue;
            }
            let a = fast_values(&padded.results[lane]);
            let b = fast_values(&original.results[lane]);
            let scale = b.iter().map(|z| z.abs()).fold(0., f64::max).max(1e-300);
            let d = (0..3).map(|k| (a[k] - b[k]).abs()).fold(0., f64::max) / scale;
            if d > tolerance(lane) {
                let label = format!(
                    "{name} lane {lane}: padded vs unpadded {d:e} (noise {:e})",
                    noise[lane]
                );
                let ratio = check_exact(lane, &a, &b, scale, &label);
                worst_exact_ratio = worst_exact_ratio.max(ratio);
                exact_checked += 1;
            } else {
                within_noise += usize::from(d > FAST_TOLERANCE);
            }
            worst_self = worst_self.max(d);
            lanes += 1;
        }
        let Some(ninja) = &entry.ninja else { continue };
        let mut masters = NinjaFirst {
            ninja: NinjaMasters::new(ninja),
            native: OneLoopNativeMasters::new().unwrap(),
            fallbacks: 0,
        };
        let run = run_fast(case, rank, case.lanes.len(), &mut masters).unwrap();
        let (results, statuses) = (run.results, run.statuses);
        fallbacks += masters.fallbacks;
        for (lane, reference) in ninja.iter().enumerate() {
            if reference.status != 0 || statuses[lane] != FastStatus::Success {
                continue;
            }
            let got = fast_values(&results[lane]);
            let scale = reference
                .values
                .iter()
                .map(|z| z.abs())
                .fold(0., f64::max)
                .max(1e-300);
            let d = (0..3)
                .map(|k| (got[k] - reference.values[k]).abs())
                .fold(0., f64::max)
                / scale;
            if d > tolerance(lane) {
                let label = format!(
                    "{name} lane {lane}: vs original Ninja {d:e} (noise {:e})",
                    noise[lane]
                );
                let ratio = check_exact(lane, &got, &reference.values, scale, &label);
                worst_exact_ratio = worst_exact_ratio.max(ratio);
                exact_checked += 1;
            }
            worst_ninja = worst_ninja.max(d);
        }
    }
    eprintln!(
        "V5 fast: {} cases, {lanes} lanes; max scaled difference padded vs unpadded {worst_self:e} ({within_noise} above 1e-11, within 10x noise), vs original Ninja {worst_ninja:e}; {exact_checked} comparisons beyond that checked against the exact reduction (max error ratio to the reference's {worst_exact_ratio:.2}); {fallbacks} master requests not among Ninja's calls",
        entries.len()
    );
}

/// Generic engine in Float(256), padded to `declared(case)`, against the
/// unpadded result: within `FLOAT_TOLERANCE`, or, where the unpadded result
/// is itself less accurate than that (ill-conditioned lanes, against the
/// exact reduction in Float(`EXACT_BITS`)), within ten times its error of
/// the exact reduction. With `shift_invariants` every lane takes the
/// invariants of its shifts instead of its `s_matrix` (module docs).
fn generic_padding(
    entries: &[Entry],
    declared: impl Fn(&BenchCase) -> usize,
    shift_invariants: bool,
) {
    let mut engines = Engines::default();
    let (mut worst, mut lanes, mut exact_checked, mut worst_ratio) = (0_f64, 0, 0, 0_f64);
    for entry in entries {
        let case = if shift_invariants {
            let mut case = entry.case.clone();
            for lane in &mut case.lanes {
                lane.s_matrix = None;
            }
            case
        } else {
            entry.case.clone()
        };
        let rank = declared(&case);
        let (original, padded) = {
            let (engine, masters) = engines.float(BITS).unwrap();
            (
                engine.evaluate(&case, case.rank, masters).unwrap(),
                engine.evaluate(&case, rank, masters).unwrap(),
            )
        };
        let mut exact = None;
        for (lane, (a, b)) in padded.iter().zip(&original).enumerate() {
            let label = format!("{} declared rank {rank} lane {lane}", case.name);
            assert_eq!(a.status, b.status, "{label}");
            if b.status != ninjars::EvaluationStatus::Success {
                continue;
            }
            let scale = b.values[..3]
                .iter()
                .map(|z| z.re.to_f64().hypot(z.im.to_f64()))
                .fold(0., f64::max)
                .max(1e-300);
            let laurent = |values: &[ninjars::Complex<ninjars::Float>]| -> [ninjars::Complex<ninjars::Float>; 3] {
                std::array::from_fn(|k| {
                    let mut z = values[k].clone();
                    z.re.set_prec(BITS);
                    z.im.set_prec(BITS);
                    z
                })
            };
            let d = scaled_difference(&a.values, &laurent(&b.values), scale);
            worst = worst.max(d);
            lanes += 1;
            if d <= FLOAT_TOLERANCE {
                continue;
            }
            let exact = exact.get_or_insert_with(|| {
                let (engine, masters) = engines.float(EXACT_BITS).unwrap();
                engine.evaluate(&case, case.rank, masters).unwrap()
            });
            assert_eq!(exact[lane].status, b.status, "{label} (exact)");
            let reference = laurent(&exact[lane].values);
            let unpadded = scaled_difference(&b.values, &reference, scale);
            let error = scaled_difference(&a.values, &reference, scale);
            assert!(
                unpadded > FLOAT_TOLERANCE && error <= 10. * unpadded,
                "{label}: padded {d:e} from the unpadded result; against the exact reduction \
                 padded {error:e}, unpadded {unpadded:e}"
            );
            exact_checked += 1;
            worst_ratio = worst_ratio.max(error / unpadded);
        }
    }
    eprintln!(
        "V5 generic Float(256): {} cases, {lanes} lanes; max scaled difference {worst:e}; \
         {exact_checked} lanes above {FLOAT_TOLERANCE:e} checked against Float({EXACT_BITS}) \
         (max error ratio to the unpadded result's {worst_ratio:.2})",
        entries.len()
    );
}

/// Rank <= n cases declared at rank n+1 (zero top coefficients through the
/// rank n+1 paths), fast engine.
#[test]
fn fast_padding_to_rank_n_plus_one() {
    fast_padding(&entries(0), |case| case.n + 1);
}

/// The same with the generic engine in Float(256), n <= 5 (the n = 6 cases
/// run with the rank n+2 padding).
#[test]
fn generic_padding_to_rank_n_plus_one() {
    let entries: Vec<Entry> = entries(0).into_iter().filter(|e| e.case.n <= 5).collect();
    generic_padding(&entries, |case| case.n + 1, false);
}

/// V5, fast engine: every rank <= n+1 case declared at rank n+2.
#[test]
fn fast_padding_to_rank_n_plus_two() {
    fast_padding(&entries(1), |case| case.n + 2);
}

/// V5, generic engine in Float(256): every rank <= n+1 case declared at
/// rank n+2.
#[test]
fn generic_padding_to_rank_n_plus_two() {
    generic_padding(&entries(1), |case| case.n + 2, true);
}
