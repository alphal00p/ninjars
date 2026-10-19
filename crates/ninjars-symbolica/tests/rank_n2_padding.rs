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
//! - generic engine in Float(256): within 1e-50 of the unpadded result.
//!
//! Declared rank n+2 exercises the rank n+2 paths with rank <= n+1
//! numerators: the fast engine runs them through `AtDeclaredRank`, because by
//! default it reduces every lane declared at rank n+2 at the lane's own rank
//! (`fast/declared_rank.rs`). That default must equal the fast engine with
//! the numerator declared at that rank bitwise. Declared rank n+1 for the
//! rank <= n cases exercises the rank n+1 paths with zero top coefficients.
//!
//! Every lane of these cases has its case's rank, so they never split a
//! batch. `fast_mixed_rank_batches_reduce_each_lane_at_its_own_rank` builds
//! batches declared at rank n+2 from lanes of ranks n-1 to n+2 (plain and
//! with `mu²` powers, some with a duplicated propagator, which fails), and
//! checks that the split by rank and the scatter reproduce every lane's
//! own-rank single-lane result bitwise.

mod common;
mod rank_n2_reference_support;

use common::ninja_reference::{NinjaLane, NinjaMasters, parse_ninja};
use ninjars::fast::{
    AtDeclaredRank, C64, FastAmplitude, FastInput, FastMaster, FastMasters, FastNumerator,
    FastOptions, FastResult, FastStatus, MuTensorNumerator, RMom, TensorNumerator,
    tensor_layout::{mu_block_offset, ncoeff, ncoeff_mu},
};
use ninjars_symbolica::OneLoopNativeMasters;
use rank_n2_reference_support::{
    fast_noise,
    harness::{BenchCase, BenchLane, Engines, FastRun, fast_values, run_fast, scaled_difference},
    rank_n2, root,
};
use serde_json::Value;
use symbolica::domains::float::RealLike;

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

/// `run_fast` (one batch of the case's lanes) with the numerator wrapped in
/// `AtDeclaredRank`: the fast engine reduces it at the declared rank even
/// where its top coefficients vanish.
fn run_fast_at_declared_rank<M: FastMasters>(
    case: &BenchCase,
    declared_rank: usize,
    masters: &mut M,
) -> FastRun {
    let width = case.lanes.len();
    let (n, lanes) = (case.n, &case.lanes);
    let shifts: Vec<RMom> = lanes
        .iter()
        .flat_map(|l| l.shifts.iter().copied())
        .collect();
    let masses: Vec<C64> = lanes
        .iter()
        .flat_map(|l| l.masses.iter().copied())
        .collect();
    let scales: Vec<f64> = lanes.iter().map(|l| l.mu2).collect();
    let s_matrix: Option<Vec<f64>> = lanes
        .iter()
        .map(|l| l.s_matrix.clone())
        .collect::<Option<Vec<_>>>()
        .map(|rows| rows.concat());
    let input = FastInput {
        lanes: width,
        shifts: &shifts,
        masses_squared: &masses,
        scales_squared: &scales,
        s_matrix: s_matrix.as_deref(),
    };
    let mut amplitude = FastAmplitude::new(n, declared_rank, FastOptions::default()).unwrap();
    let mut results = vec![FastResult::default(); width];
    let mut statuses = vec![FastStatus::NotEvaluated; width];
    let size = ncoeff_mu(declared_rank, case.mu_powers);
    let coefficients: Vec<C64> = (0..width)
        .flat_map(|lane| case.padded_tensor(lane, declared_rank))
        .collect();
    assert_eq!(coefficients.len(), width * size);
    if case.mu_powers == 0 {
        let mut tensor = TensorNumerator::new(declared_rank, width);
        tensor.set_coefficients(width, &coefficients).unwrap();
        let mut numerator = AtDeclaredRank(&mut tensor);
        amplitude
            .evaluate(&input, &mut numerator, masters, &mut results, &mut statuses)
            .unwrap();
    } else {
        let mut tensor = MuTensorNumerator::new(declared_rank, case.mu_powers, width).unwrap();
        tensor.set_coefficients(width, &coefficients).unwrap();
        let mut numerator = AtDeclaredRank(&mut tensor);
        amplitude
            .evaluate(&input, &mut numerator, masters, &mut results, &mut statuses)
            .unwrap();
    }
    FastRun {
        ids: lanes.iter().map(|l| l.id.clone()).collect(),
        lanes: (0..width).collect(),
        results,
        statuses,
    }
}

/// The bits of the five values of a lane.
fn bits(result: &FastResult) -> [(u64, u64); 5] {
    fast_values(result).map(|z| (z.re.to_bits(), z.im.to_bits()))
}

/// The rank of every lane of `case` (`lane_rank` of the case's own
/// numerator type).
fn lane_ranks(case: &BenchCase) -> Vec<usize> {
    let lanes = case.lanes.len();
    let coefficients: Vec<C64> = case.lanes.iter().flat_map(|l| l.tensor.clone()).collect();
    let mut numerator: Box<dyn FastNumerator> = if case.mu_powers == 0 {
        let mut tensor = TensorNumerator::new(case.rank, lanes);
        tensor.set_coefficients(lanes, &coefficients).unwrap();
        Box::new(tensor)
    } else {
        let mut tensor = MuTensorNumerator::new(case.rank, case.mu_powers, lanes).unwrap();
        tensor.set_coefficients(lanes, &coefficients).unwrap();
        Box::new(tensor)
    };
    (0..lanes)
        .map(|lane| numerator.lane_rank(lane).unwrap())
        .collect()
}

/// Lane `lane` of `case` alone, declared at its rank `rank <= case.rank`
/// (the coefficients above `rank` are zero).
fn single_at_rank(case: &BenchCase, lane: usize, rank: usize) -> BenchCase {
    let mu_powers = case.mu_powers.min(rank / 2);
    let source = &case.lanes[lane].tensor;
    let mut tensor = vec![C64::ZERO; ncoeff_mu(rank, mu_powers)];
    for k in 0..=mu_powers {
        let (from, to, size) = (
            mu_block_offset(case.rank, k),
            mu_block_offset(rank, k),
            ncoeff(rank - 2 * k),
        );
        tensor[to..to + size].copy_from_slice(&source[from..from + size]);
    }
    let mut single = case.clone();
    single.rank = rank;
    single.mu_powers = mu_powers;
    single.lanes = vec![case.lanes[lane].clone()];
    single.lanes[0].tensor = tensor;
    single
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

/// Fast engine, padded to `declared(case)` and reduced at that rank
/// (`AtDeclaredRank`): against original Ninja with its master values, and
/// against the unpadded engine with native masters. By default (without
/// `AtDeclaredRank`) a numerator padded to rank n+2 must reduce every lane
/// bitwise as declared at the lane's own rank.
fn fast_padding(entries: &[Entry], declared: impl Fn(&BenchCase) -> usize) {
    let mut native = OneLoopNativeMasters::new().unwrap();
    let mut engines = Engines::default();
    let (mut worst_ninja, mut worst_self, mut lanes, mut fallbacks) = (0_f64, 0_f64, 0, 0);
    let (mut within_noise, mut exact_checked, mut worst_exact_ratio) = (0, 0, 0_f64);
    let mut lowered_bitwise = 0;
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
        let padded = run_fast_at_declared_rank(case, rank, &mut native);
        if rank == case.n + 2 {
            let lowered = run_fast(case, rank, case.lanes.len(), &mut native).unwrap();
            for (lane, lane_rank) in lane_ranks(case).into_iter().enumerate() {
                let single = single_at_rank(case, lane, lane_rank);
                let want = run_fast(&single, lane_rank, 1, &mut native).unwrap();
                let label = format!("{name} lane {lane} at its rank {lane_rank}");
                assert_eq!(lowered.statuses[lane], want.statuses[0], "{label}");
                assert_eq!(
                    bits(&lowered.results[lane]),
                    bits(&want.results[0]),
                    "{label}"
                );
            }
            lowered_bitwise += case.lanes.len();
        }
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
        let run = run_fast_at_declared_rank(case, rank, &mut masters);
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
        "V5 fast: {} cases, {lanes} lanes; max scaled difference padded vs unpadded {worst_self:e} ({within_noise} above 1e-11, within 10x noise), vs original Ninja {worst_ninja:e}; {exact_checked} comparisons beyond that checked against the exact reduction (max error ratio to the reference's {worst_exact_ratio:.2}); {fallbacks} master requests not among Ninja's calls; {lowered_bitwise} lanes reduced at their own rank, bitwise, by default",
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
/// run with the rank n+2 padding).
#[test]
fn generic_padding_to_rank_n_plus_one() {
    let entries: Vec<Entry> = entries(0).into_iter().filter(|e| e.case.n <= 5).collect();
    generic_padding(&entries, |case| case.n + 1);
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
    generic_padding(&entries(1), |case| case.n + 2);
}

// ------------------------------------------------------- mixed-rank batches

/// Sources of a mixed-rank batch (`validation/rank-n2`, one `n`, ranks n+2
/// down to n-1): lanes 0 and 1 of each, the second with a duplicated
/// propagator where marked `true`.
const MIXED_PLAIN: [(&str, bool); 4] = [
    ("cases/rn2-int-complex-n3-r5", true),
    ("controls/cases/rn2-int-complex-n3-r4", false),
    ("controls/cases/rn2-int-complex-n3-r3", true),
    ("controls/cases/rn2-int-complex-n3-r2", false),
];

/// The same with `mu²` powers: rank n+2 lanes with `mu²`, `mu⁴` and `mu⁶`
/// terms, rank n+1 lanes with `mu²` terms and plain lanes of every rank, in
/// a numerator with three `mu²` powers.
const MIXED_MU: [(&str, bool); 6] = [
    ("cases/rn2-int-complex-n4-r6-muall", false),
    ("cases/rn2-int-complex-n4-r6", true),
    ("controls/cases/rn2-int-complex-n4-r5-mu2", true),
    ("controls/cases/rn2-int-complex-n4-r5", false),
    ("controls/cases/rn2-int-complex-n4-r4", false),
    ("controls/cases/rn2-int-complex-n4-r3", true),
];

/// Lane `lane` of `case` laid out at rank `rank` with `mu_powers` powers of
/// `mu²` (zero new coefficients and blocks).
fn relaid(case: &BenchCase, lane: usize, rank: usize, mu_powers: usize) -> BenchLane {
    assert!(rank >= case.rank && mu_powers >= case.mu_powers);
    let source = &case.lanes[lane];
    let mut tensor = vec![C64::ZERO; ncoeff_mu(rank, mu_powers)];
    for k in 0..=case.mu_powers {
        let (from, to, size) = (
            mu_block_offset(case.rank, k),
            mu_block_offset(rank, k),
            ncoeff(case.rank - 2 * k),
        );
        tensor[to..to + size].copy_from_slice(&source.tensor[from..from + size]);
    }
    BenchLane {
        tensor,
        ..source.clone()
    }
}

/// `lane` with its second propagator replaced by its first (S-matrix
/// included): every cut through both is singular.
fn with_duplicated_propagator(lane: &BenchLane, n: usize) -> BenchLane {
    let mut out = lane.clone();
    out.id = format!("{}-duplicated", lane.id);
    out.shifts[1] = out.shifts[0];
    out.masses[1] = out.masses[0];
    if let Some(s) = &mut out.s_matrix {
        let old = s.clone();
        let line = |i: usize| if i == 1 { 0 } else { i };
        for i in 0..n {
            for j in 0..n {
                s[i * n + j] = old[line(i) * n + line(j)];
            }
        }
    }
    out
}

/// The case of rank n+2 with `mu_powers` whose lanes are lanes 0 and 1 of
/// `sources` in turn (all of lane 0 first), and the rank of every lane's
/// source.
fn mixed_case(name: &str, sources: &[(&str, bool)], mu_powers: usize) -> (BenchCase, Vec<usize>) {
    let sources: Vec<(BenchCase, bool)> = sources
        .iter()
        .map(|&(path, duplicated)| {
            let case = BenchCase::load(&rank_n2().join(format!("{path}.json"))).unwrap();
            (case, duplicated)
        })
        .collect();
    let n = sources[0].0.n;
    let rank = n + 2;
    let (mut lanes, mut ranks) = (Vec::new(), Vec::new());
    for index in 0..2 {
        for (source, duplicated) in &sources {
            assert_eq!(source.n, n, "{}", source.name);
            let lane = relaid(source, index, rank, mu_powers);
            lanes.push(if index == 1 && *duplicated {
                with_duplicated_propagator(&lane, n)
            } else {
                lane
            });
            ranks.push(source.rank);
        }
    }
    let case = BenchCase {
        name: name.to_string(),
        path: Default::default(),
        n,
        rank,
        mu_powers,
        lanes,
    };
    (case, ranks)
}

/// A result no evaluation produces, which a failed lane must keep.
fn sentinel(lane: usize) -> FastResult {
    let z = C64::new(lane as f64 + 0.25, -1.);
    FastResult {
        laurent: [z; 3],
        cut_constructible: z,
        rational: z,
    }
}

/// Lanes `order` of `case` as one batch declared at `case.rank` (the
/// numerator types and S-matrix rule of `run_fast`), every result preset to
/// [`sentinel`].
fn run_lanes(
    amplitude: &mut FastAmplitude,
    case: &BenchCase,
    order: &[usize],
    masters: &mut OneLoopNativeMasters,
) -> (Vec<FastResult>, Vec<FastStatus>) {
    let width = order.len();
    let lanes: Vec<&BenchLane> = order.iter().map(|&lane| &case.lanes[lane]).collect();
    let shifts: Vec<RMom> = lanes
        .iter()
        .flat_map(|l| l.shifts.iter().copied())
        .collect();
    let masses: Vec<C64> = lanes
        .iter()
        .flat_map(|l| l.masses.iter().copied())
        .collect();
    let scales: Vec<f64> = lanes.iter().map(|l| l.mu2).collect();
    let s_matrix: Option<Vec<f64>> = lanes
        .iter()
        .map(|l| l.s_matrix.clone())
        .collect::<Option<Vec<_>>>()
        .map(|rows| rows.concat());
    let input = FastInput {
        lanes: width,
        shifts: &shifts,
        masses_squared: &masses,
        scales_squared: &scales,
        s_matrix: s_matrix.as_deref(),
    };
    let coefficients: Vec<C64> = lanes
        .iter()
        .flat_map(|l| l.tensor.iter().copied())
        .collect();
    let mut results: Vec<FastResult> = (0..width).map(sentinel).collect();
    let mut statuses = vec![FastStatus::NotEvaluated; width];
    if case.mu_powers == 0 {
        let mut tensor = TensorNumerator::new(case.rank, width);
        tensor.set_coefficients(width, &coefficients).unwrap();
        amplitude
            .evaluate(&input, &mut tensor, masters, &mut results, &mut statuses)
            .unwrap();
    } else {
        let mut tensor = MuTensorNumerator::new(case.rank, case.mu_powers, width).unwrap();
        tensor.set_coefficients(width, &coefficients).unwrap();
        amplitude
            .evaluate(&input, &mut tensor, masters, &mut results, &mut statuses)
            .unwrap();
    }
    (results, statuses)
}

/// Batches declared at rank n+2 whose lanes have ranks n-1 to n+2, plain and
/// with `mu²` powers, some failing (duplicated propagator), with and without
/// S-matrices: every lane, alone and in natural, reversed and cyclic batches
/// of up to 37 lanes, must be bitwise its own-rank single-lane result
/// (`run_fast` declared at the lane's rank, `mu²` powers up to half that
/// rank), and a failed lane must keep its previous result. One amplitude
/// serves every batch of a case, so the lower-rank engines are reused across
/// widths. Without the dispatch of `fast/declared_rank.rs` (every lane
/// through the rank n+2 engine) this fails.
#[test]
fn fast_mixed_rank_batches_reduce_each_lane_at_its_own_rank() {
    let mut native = OneLoopNativeMasters::new().unwrap();
    let (mut checked, mut failed, mut batches) = (0, 0, 0);
    for (name, sources, mu_powers) in [
        ("mixed-plain-n3", &MIXED_PLAIN[..], 0),
        ("mixed-mu-n4", &MIXED_MU[..], 3),
    ] {
        let (full, ranks) = mixed_case(name, sources, mu_powers);
        let n = full.n;
        // The lanes have the ranks of their sources, n-1 to n+2.
        assert_eq!(lane_ranks(&full), ranks, "{name}");
        assert!((n - 1..=n + 2).all(|rank| ranks.contains(&rank)), "{name}");
        let mut without_s_matrix = full.clone();
        for lane in &mut without_s_matrix.lanes {
            lane.s_matrix = None;
        }
        for case in [full, without_s_matrix] {
            let s_matrix = case.lanes[0].s_matrix.is_some();
            let count = case.lanes.len();
            let own: Vec<(FastResult, FastStatus)> = (0..count)
                .map(|lane| {
                    let single = single_at_rank(&case, lane, ranks[lane]);
                    let run = run_fast(&single, ranks[lane], 1, &mut native).unwrap();
                    (run.results[0], run.statuses[0])
                })
                .collect();
            for (lane, (_, status)) in own.iter().enumerate() {
                let expected = if case.lanes[lane].id.ends_with("-duplicated") {
                    FastStatus::Unstable
                } else {
                    FastStatus::Success
                };
                assert_eq!(
                    *status, expected,
                    "{name} lane {lane} ({}): own-rank status",
                    case.lanes[lane].id
                );
            }
            let mut amplitude = FastAmplitude::new(n, n + 2, FastOptions::default()).unwrap();
            let mut orders: Vec<Vec<usize>> = (0..count).map(|lane| vec![lane]).collect();
            orders.push((0..count).collect());
            orders.push((0..count).rev().collect());
            orders.push((0..37).map(|i| (2 * i + i / count) % count).collect());
            for order in &orders {
                let (results, statuses) = run_lanes(&mut amplitude, &case, order, &mut native);
                for (position, &lane) in order.iter().enumerate() {
                    let (want, want_status) = own[lane];
                    let label = format!(
                        "{name} (S-matrix {s_matrix}) lane {lane} ({}, rank {}) at position {position} of {}",
                        case.lanes[lane].id,
                        ranks[lane],
                        order.len()
                    );
                    assert_eq!(statuses[position], want_status, "{label}");
                    if want_status == FastStatus::Success {
                        assert_eq!(bits(&results[position]), bits(&want), "{label}");
                    } else {
                        assert_eq!(
                            results[position],
                            sentinel(position),
                            "{label}: a failed lane keeps its previous result"
                        );
                        failed += 1;
                    }
                    checked += 1;
                }
                batches += 1;
            }
        }
    }
    eprintln!(
        "FAST_MIXED_RANK {batches} batches, {checked} lanes bitwise equal to their own-rank single-lane runs ({failed} of them failed lanes keeping their previous result)"
    );
}
