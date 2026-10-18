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
//!   cases) at that level;
//! - generic engine in Float(256): within 1e-50 of the unpadded result.
//!
//! Declared rank n+2 exercises the rank n+2 paths with rank <= n+1
//! numerators and runs at integration. Declared rank n+1 for the rank <= n
//! cases runs now (the rank n+1 paths with zero top coefficients).

mod common;
mod rank_n2_support;

use common::ninja_reference::{NinjaLane, NinjaMasters, parse_ninja};
use ninjars::fast::{C64, FastMaster, FastMasters, FastStatus};
use ninjars_symbolica::OneLoopNativeMasters;
use rank_n2_support::{
    fast_noise,
    harness::{BenchCase, Engines, fast_values, run_fast, scaled_difference},
    rank_n2, root,
};
use serde_json::Value;

const FAST_TOLERANCE: f64 = 1e-11;
const FLOAT_TOLERANCE: f64 = 1e-50;
const BITS: u32 = 256;

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

/// Fast engine, padded to `declared(case)`: against original Ninja with its
/// master values, and against the unpadded engine with native masters.
fn fast_padding(entries: &[Entry], declared: impl Fn(&BenchCase) -> usize) {
    let mut native = OneLoopNativeMasters::new().unwrap();
    let (mut worst_ninja, mut worst_self, mut lanes, mut fallbacks) = (0_f64, 0_f64, 0, 0);
    let mut within_noise = 0;
    for entry in entries {
        let case = &entry.case;
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
            assert!(
                d <= tolerance(lane),
                "{name} lane {lane}: padded vs unpadded {d:e} (noise {:e})",
                noise[lane]
            );
            within_noise += usize::from(d > FAST_TOLERANCE);
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
            assert!(
                d <= tolerance(lane),
                "{name} lane {lane}: vs original Ninja {d:e} (noise {:e})",
                noise[lane]
            );
            worst_ninja = worst_ninja.max(d);
        }
    }
    eprintln!(
        "V5 fast: {} cases, {lanes} lanes; max scaled difference padded vs unpadded {worst_self:e} ({within_noise} above 1e-11, within 10x noise), vs original Ninja {worst_ninja:e}; {fallbacks} master requests not among Ninja's calls",
        entries.len()
    );
}

/// Generic engine in Float(256), padded to `declared(case)`, against the
/// unpadded result.
fn generic_padding(entries: &[Entry], declared: impl Fn(&BenchCase) -> usize) {
    let mut engines = Engines::default();
    let (engine, masters) = engines.float(BITS).unwrap();
    let (mut worst, mut lanes) = (0_f64, 0);
    for entry in entries {
        let case = &entry.case;
        let rank = declared(case);
        let original = engine.evaluate(case, case.rank, masters).unwrap();
        let padded = engine.evaluate(case, rank, masters).unwrap();
        for (lane, (a, b)) in padded.iter().zip(&original).enumerate() {
            let label = format!("{} declared rank {rank} lane {lane}", case.name);
            assert_eq!(a.status, b.status, "{label}");
            if b.status != ninjars::EvaluationStatus::Success {
                continue;
            }
            let scale = b.values[..3]
                .iter()
                .map(|z| {
                    use symbolica::domains::float::RealLike;
                    z.re.to_f64().hypot(z.im.to_f64())
                })
                .fold(0., f64::max)
                .max(1e-300);
            let expected = [
                b.values[0].clone(),
                b.values[1].clone(),
                b.values[2].clone(),
            ];
            let d = scaled_difference(&a.values, &expected, scale);
            assert!(d <= FLOAT_TOLERANCE, "{label}: {d:e}");
            worst = worst.max(d);
            lanes += 1;
        }
    }
    eprintln!(
        "V5 generic Float(256): {} cases, {lanes} lanes; max scaled difference {worst:e}",
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
/// run at integration with the rank n+2 padding).
#[test]
fn generic_padding_to_rank_n_plus_one() {
    let entries: Vec<Entry> = entries(0).into_iter().filter(|e| e.case.n <= 5).collect();
    generic_padding(&entries, |case| case.n + 1);
}

/// V5, fast engine: every rank <= n+1 case declared at rank n+2.
#[test]
#[ignore = "enable at integration"]
fn fast_padding_to_rank_n_plus_two() {
    fast_padding(&entries(1), |case| case.n + 2);
}

/// V5, generic engine in Float(256): every rank <= n+1 case declared at
/// rank n+2.
#[test]
#[ignore = "enable at integration"]
fn generic_padding_to_rank_n_plus_two() {
    generic_padding(&entries(1), |case| case.n + 2);
}
