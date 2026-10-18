//! The binary64 fast engine at numerator rank n+2 (`fast/engine_x2.rs`).
//!
//! 1. **Rank padding (design V5).** Every rank <= n+1 case with an
//!    original-Ninja result (`validation/fast/ninja_reference/cases`,
//!    `validation/benchmark/{cases,synthetic}`) is declared rank n+2 (zero top
//!    tensor blocks) and reduced by the rank n+2 engine with Ninja's own
//!    OneLOop master values. Where the rank n+2 engine asks for a bubble
//!    family Ninja did not call (rank 3 or 4: the coefficients of `B111` and
//!    `B1111` are then rounding-level), or a family of a stage Ninja skips at
//!    low rank, the missing entries come from the exact oneloopmaster
//!    expressions (`SymbolicaMasters` in `Float(256)`). Every lane must
//!    agree with Ninja within the tolerance of `fast_ninja_reference.rs`,
//!    statuses must match, and batched evaluations must reproduce the width-1
//!    results bitwise. Where the validated rank n+1 engine itself, on the same
//!    numerator declared rank n+1, already differs from Ninja by more than
//!    that tolerance, the lane must stay within ten times its deviation:
//!    this happens for ill-conditioned synthetic points (`syn-n5-r6`,
//!    `syn-n6-r5`, `syn-n6-r7`, `syn-gghhh-ggHHH-n5-r5`: 2e-9 to 5e-6 for
//!    the rank n+1 engine) and near-lightlike bubble momenta below rank n+1
//!    (`edge-small-p2`, p² = ±0.01 at p = 200), where rank n+1 arithmetic
//!    (the tadpole `(e1, e2)` orientation) differs from Ninja's lower-rank
//!    one by up to 6e-8.
//! 2. **High-precision references.** The rank n+2 cases of the review's
//!    60-digit references (`tools/rank_n2/review/hpref`) and, when present,
//!    the whole validation suite (`validation/rank-n2/reference`, tier A
//!    and B lanes) are reduced with oneloopmaster's native masters plus the
//!    exact `B1111` in `Float(256)` (the native B1111 family comes with S4).
//!    Their inputs live in
//!    `validation/rank-n2/cases` (`NINJARS_RANK_N2_SUITE` overrides the
//!    directory); without them part 2 is skipped with a note.

mod common;

use common::ninja_reference::{MasterCall, NinjaLane, parse_case, parse_ninja, values};
use ninjars::fast::{
    C64, FastAmplitude, FastInput, FastMaster, FastMasters, FastNumerator, FastOptions, FastResult,
    FastStatus, MuTensorNumerator, RMom, TensorNumerator,
    tensor_layout::{ncoeff, ncoeff_mu},
};
use ninjars::{Complex, Float, MasterIntegral, MasterProvider, MasterRequest};
use ninjars_symbolica::{OneLoopNativeMasters, SymbolicaMasters};
use serde_json::Value;
use std::path::{Path, PathBuf};
use symbolica::domains::{
    float::{RealLike, SingleFloat},
    rational::Rational,
};

/// Same tolerance as `fast_ninja_reference.rs` (scaled, no absolute floor).
const NINJA_TOLERANCE: f64 = 1e-11;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn json_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "json")
                && !p.to_string_lossy().ends_with(".masters.json")
        })
        .collect();
    files.sort();
    files
}

fn same(a: C64, b: C64) -> bool {
    let scale = a.abs().max(b.abs());
    (a - b).abs() <= 1e-13 * scale
}

fn is_bubble(family: FastMaster) -> bool {
    matches!(
        family,
        FastMaster::BubbleRank2 | FastMaster::BubbleRank3 | FastMaster::BubbleRank4
    )
}

/// Exact oneloopmaster expressions in `Float(256)`, rounded to binary64:
/// every fast family, in the fast output layout.
struct FloatMasters {
    masters: SymbolicaMasters<Float>,
    prototype: Float,
}

impl FloatMasters {
    fn new() -> Self {
        let prototype = Float::new(256);
        Self {
            masters: SymbolicaMasters::new(prototype.clone()),
            prototype,
        }
    }

    fn convert(&self, z: C64) -> Complex<Float> {
        let real = |x: f64| {
            self.prototype
                .from_rational(&Rational::try_from(x).unwrap())
        };
        Complex::new(real(z.re), real(z.im))
    }

    /// Outputs `first..first + out.len()` of one row of `family` (fast
    /// argument and output layout).
    fn evaluate_row(
        &mut self,
        family: FastMaster,
        args: &[C64],
        first: usize,
        out: &mut [[C64; 3]],
    ) {
        let (invariants, masses) = match family {
            FastMaster::A0 => (0, 1),
            FastMaster::C0 => (3, 3),
            FastMaster::D0 => (6, 4),
            _ => (1, 2),
        };
        let integrals: &[MasterIntegral] = match family {
            FastMaster::A0 => &[MasterIntegral::A0],
            FastMaster::C0 => &[MasterIntegral::C0],
            FastMaster::D0 => &[MasterIntegral::D0],
            _ => &[
                MasterIntegral::B0,
                MasterIntegral::B1,
                MasterIntegral::B11,
                MasterIntegral::B111,
                MasterIntegral::B1111,
            ],
        };
        let inv: Vec<_> = args[..invariants]
            .iter()
            .map(|&z| self.convert(z))
            .collect();
        let mass: Vec<_> = args[invariants..invariants + masses]
            .iter()
            .map(|&z| self.convert(z))
            .collect();
        let scale = [self.convert(args[args.len() - 1])];
        for (k, &integral) in integrals[first..first + out.len()].iter().enumerate() {
            let zero = Complex::new(self.prototype.clone(), self.prototype.clone());
            let mut output = vec![[zero.clone(), zero.clone(), zero]];
            self.masters
                .evaluate(
                    MasterRequest {
                        integral,
                        invariants: &inv,
                        masses_squared: &mass,
                        scales_squared: &scale,
                    },
                    &mut output,
                )
                .unwrap();
            out[k] = std::array::from_fn(|e| {
                C64::new(output[0][e].re.to_f64(), output[0][e].im.to_f64())
            });
        }
    }
}

/// Original Ninja's master values; entries Ninja never requested from the
/// exact oneloopmaster expressions ([`FloatMasters`]).
struct PaddedMasters {
    calls: Vec<MasterCall>,
    exact: FloatMasters,
    /// Rows Ninja never requested, per family (filled exactly).
    filled: std::collections::BTreeMap<FastMaster, usize>,
    completed: usize,
}

impl PaddedMasters {
    fn new(ninja: &[NinjaLane]) -> Self {
        Self {
            calls: ninja.iter().flat_map(|l| l.calls.iter().cloned()).collect(),
            exact: FloatMasters::new(),
            filled: Default::default(),
            completed: 0,
        }
    }
}

impl FastMasters for PaddedMasters {
    fn evaluate(
        &mut self,
        family: FastMaster,
        args: &[C64],
        out: &mut [[C64; 3]],
        ok: &mut [bool],
    ) -> ninjars::Result<()> {
        let (arity, width) = (family.arity(), family.outputs());
        for (row, a) in args.chunks_exact(arity).enumerate() {
            let matches = |c: &&MasterCall| {
                (c.family == family || (is_bubble(c.family) && is_bubble(family)))
                    && c.args.len() == a.len()
                    && c.args.iter().zip(a).all(|(&x, &y)| same(x, y))
            };
            // Prefer the call of the same family, then the widest bubble call.
            let found = self
                .calls
                .iter()
                .filter(matches)
                .max_by_key(|c| (c.family == family, c.values.len()));
            let have = match found {
                Some(call) => {
                    let have = call.values.len().min(width);
                    out[row * width..row * width + have].copy_from_slice(&call.values[..have]);
                    have
                }
                // Ninja skips the stages its rank does not reach (triangles
                // below n-3, bubbles below n-2, tadpoles below n-1); the
                // padded reduction's coefficients there are rounding-level.
                None => {
                    *self.filled.entry(family).or_default() += 1;
                    0
                }
            };
            ok[row] = true;
            if have < width {
                self.exact.evaluate_row(
                    family,
                    a,
                    have,
                    &mut out[row * width + have..(row + 1) * width],
                );
                self.completed += 1;
            }
        }
        Ok(())
    }
}

fn padded_tensor(
    case: &common::ninja_reference::Case,
    points: &[usize],
    rank: usize,
) -> TensorNumerator {
    let mut tensor = TensorNumerator::new(rank, points.len());
    for (lane, &index) in points.iter().enumerate() {
        let source = &case.points[index].tensor;
        tensor.lane_coefficients_mut(lane)[..source.len()].copy_from_slice(source);
    }
    tensor
}

fn batch_input(
    case: &common::ninja_reference::Case,
    points: &[usize],
) -> common::ninja_reference::Batch {
    common::ninja_reference::Batch::new(case, points)
}

#[test]
fn fast_rank_n2_padding_reproduces_original_ninja() {
    let root = root();
    let mut pairs: Vec<(PathBuf, PathBuf)> = Vec::new();
    let reference = root.join("validation/fast/ninja_reference");
    for case in json_files(&reference.join("cases")) {
        let ninja = reference.join("ninja").join(case.file_name().unwrap());
        pairs.push((case, ninja));
    }
    for dir in [
        "validation/benchmark/cases",
        "validation/benchmark/synthetic",
    ] {
        for case in json_files(&root.join(dir)) {
            let ninja = root
                .join("validation/benchmark/ninja")
                .join(case.file_name().unwrap());
            if ninja.exists() {
                pairs.push((case, ninja));
            }
        }
    }
    assert!(pairs.len() >= 60, "missing padding cases ({})", pairs.len());
    let (mut lanes, mut worst, mut completed, mut unstable) = (0, 0f64, 0, 0);
    let (mut excepted, mut worst_excepted, mut worst_ratio) = (0, 0f64, 0f64);
    for (case_path, ninja_path) in &pairs {
        let case = parse_case(&std::fs::read_to_string(case_path).unwrap()).unwrap();
        let masters_path = ninja_path.with_extension("masters.json");
        let masters_text = masters_path
            .exists()
            .then(|| std::fs::read_to_string(&masters_path).unwrap());
        let ninja = parse_ninja(
            &std::fs::read_to_string(ninja_path).unwrap(),
            masters_text.as_deref(),
        )
        .unwrap();
        assert!(case.rank <= case.n + 1);
        let declared = case.n + 2;
        let name = case.name.clone();
        let mut masters = PaddedMasters::new(&ninja);
        let mut amplitude = FastAmplitude::new(case.n, declared, FastOptions::default()).unwrap();
        // The validated rank n+1 engine on the same numerator padded to n+1.
        let mut x1 = FastAmplitude::new(case.n, case.n + 1, FastOptions::default()).unwrap();
        let mut single = Vec::new();
        for (index, reference) in ninja.iter().enumerate() {
            let batch = batch_input(&case, &[index]);
            let mut tensor = padded_tensor(&case, &[index], declared);
            let mut results = [FastResult::default()];
            let mut statuses = [FastStatus::NotEvaluated];
            amplitude
                .evaluate(
                    &batch.input(),
                    &mut tensor,
                    &mut masters,
                    &mut results,
                    &mut statuses,
                )
                .unwrap();
            let expected = match reference.status {
                0 => FastStatus::Success,
                2 => FastStatus::Unstable,
                other => panic!("{name}: Ninja status {other}"),
            };
            assert_eq!(statuses[0], expected, "{name} {}", reference.id);
            if expected == FastStatus::Success {
                let want = reference.values;
                let scale = want.iter().map(|z| z.abs()).fold(0., f64::max).max(1e-300);
                let scaled = |r: &FastResult| {
                    values(r)
                        .iter()
                        .zip(&want)
                        .map(|(a, b)| (*a - *b).abs() / scale)
                        .fold(0., f64::max)
                };
                let error = scaled(&results[0]);
                let mut tolerance = NINJA_TOLERANCE;
                let mut tensor = padded_tensor(&case, &[index], case.n + 1);
                let mut x1_results = [FastResult::default()];
                x1.evaluate(
                    &batch.input(),
                    &mut tensor,
                    &mut masters,
                    &mut x1_results,
                    &mut statuses,
                )
                .unwrap();
                assert_eq!(statuses[0], expected, "{name} {} at n+1", reference.id);
                let x1_error = scaled(&x1_results[0]);
                if x1_error > NINJA_TOLERANCE {
                    tolerance = 10. * x1_error;
                    excepted += 1;
                    worst_excepted = worst_excepted.max(error);
                    worst_ratio = worst_ratio.max(error / x1_error);
                }
                assert!(
                    error <= tolerance,
                    "{name} {} declared rank {declared}: scaled error {error:e} (tolerance {tolerance:e})",
                    reference.id
                );
                if tolerance == NINJA_TOLERANCE {
                    worst = worst.max(error);
                }
            } else {
                unstable += 1;
            }
            lanes += 1;
            single.push((results[0], statuses[0]));
        }
        // Only families of stages Ninja skipped at this rank were filled.
        for (&family, &rows) in &masters.filled {
            let skipped = match family {
                FastMaster::C0 => case.rank + 3 < case.n,
                FastMaster::A0 => case.rank + 1 < case.n,
                FastMaster::D0 => false,
                _ => case.rank + 2 < case.n,
            };
            assert!(skipped, "{name}: {rows} unmatched {family:?} requests");
        }
        completed += masters.completed;
        // Batches (natural, reversed, cyclic and wide) reproduce width 1 bitwise.
        let count = ninja.len();
        let natural: Vec<usize> = (0..count).collect();
        let reversed: Vec<usize> = (0..count).rev().collect();
        let cyclic: Vec<usize> = (0..37).map(|i| (2 * i + i / count) % count).collect();
        for points in [natural, reversed, cyclic] {
            let batch = batch_input(&case, &points);
            let mut tensor = padded_tensor(&case, &points, declared);
            let mut results = vec![FastResult::default(); points.len()];
            let mut statuses = vec![FastStatus::NotEvaluated; points.len()];
            amplitude
                .evaluate(
                    &batch.input(),
                    &mut tensor,
                    &mut masters,
                    &mut results,
                    &mut statuses,
                )
                .unwrap();
            for (lane, &index) in points.iter().enumerate() {
                let (want, want_status) = single[index];
                assert_eq!(statuses[lane], want_status, "{name} batch lane {lane}");
                if want_status == FastStatus::Success {
                    assert_eq!(results[lane], want, "{name} batch lane {lane}");
                }
            }
        }
    }
    assert!(unstable >= 8, "the edge cases must include unstable points");
    eprintln!(
        "FAST_RANK_N2_PADDING cases={} lanes={lanes} unstable={unstable} exact_fills={completed} max_scaled_error={worst:e}; lanes where the rank n+1 engine itself deviates (> 1e-11): {excepted}, their max scaled error {worst_excepted:e}, max ratio to the rank n+1 deviation {worst_ratio:.2}",
        pairs.len()
    );
}

// ------------------------------------------------- high-precision references

/// oneloopmaster's native masters, with `B1111` of the rank-four bubble
/// family from the exact expressions (`SymbolicaMasters`).
struct ExactB1111Masters {
    native: OneLoopNativeMasters,
    exact: FloatMasters,
    scratch: Vec<[C64; 3]>,
}

impl FastMasters for ExactB1111Masters {
    fn evaluate(
        &mut self,
        family: FastMaster,
        args: &[C64],
        out: &mut [[C64; 3]],
        ok: &mut [bool],
    ) -> ninjars::Result<()> {
        if family != FastMaster::BubbleRank4 {
            return FastMasters::evaluate(&mut self.native, family, args, out, ok);
        }
        let rows = ok.len();
        self.scratch.clear();
        self.scratch.resize(rows * 4, [C64::ZERO; 3]);
        FastMasters::evaluate(
            &mut self.native,
            FastMaster::BubbleRank3,
            args,
            &mut self.scratch,
            ok,
        )?;
        for (row, a) in args.as_chunks::<4>().0.iter().enumerate() {
            out[row * 5..row * 5 + 4].copy_from_slice(&self.scratch[row * 4..(row + 1) * 4]);
            self.exact.evaluate_row(
                FastMaster::BubbleRank4,
                a,
                4,
                &mut out[row * 5 + 4..row * 5 + 5],
            );
            ok[row] &= out[row * 5 + 4].iter().all(|z| z.is_finite());
        }
        Ok(())
    }
}

/// A rank n+2 suite case (`ninjars-bench-case-v1`, or `-v2` with `mu²`
/// blocks).
struct SuiteCase {
    name: String,
    n: usize,
    rank: usize,
    mu_powers: Option<usize>,
    shifts: Vec<RMom>,
    masses: Vec<C64>,
    scales: Vec<f64>,
    smat: Vec<f64>,
    tensors: Vec<Vec<C64>>,
}

fn c64(value: &Value) -> C64 {
    C64::new(value[0].as_f64().unwrap(), value[1].as_f64().unwrap())
}

fn parse_suite_case(text: &str) -> SuiteCase {
    let value: Value = serde_json::from_str(text).unwrap();
    let n = value["n"].as_u64().unwrap() as usize;
    let rank = value["rank"].as_u64().unwrap() as usize;
    let mu_powers = match value["format"].as_str().unwrap() {
        "ninjars-bench-case-v1" => None,
        "ninjars-bench-case-v2" => Some(value["mu_powers"].as_u64().unwrap() as usize),
        other => panic!("format {other}"),
    };
    let width = mu_powers.map_or(ncoeff(rank), |k| ncoeff_mu(rank, k));
    let mut case = SuiteCase {
        name: value["name"].as_str().unwrap().to_string(),
        n,
        rank,
        mu_powers,
        shifts: Vec::new(),
        masses: Vec::new(),
        scales: Vec::new(),
        smat: Vec::new(),
        tensors: Vec::new(),
    };
    for lane in value["lanes"].as_array().unwrap() {
        for p in lane["shifts"].as_array().unwrap() {
            case.shifts
                .push(std::array::from_fn(|k| p[k].as_f64().unwrap()));
        }
        case.masses
            .extend(lane["masses_squared"].as_array().unwrap().iter().map(c64));
        case.scales.push(lane["mu_squared"].as_f64().unwrap());
        for row in lane["s_matrix"].as_array().unwrap() {
            case.smat
                .extend(row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()));
        }
        let tensor: Vec<C64> = lane["tensor"].as_array().unwrap().iter().map(c64).collect();
        assert_eq!(tensor.len(), width, "{}", case.name);
        case.tensors.push(tensor);
    }
    case
}

/// Fast rank n+2 results of every lane (one batch).
fn reduce_suite_case(
    case: &SuiteCase,
    masters: &mut ExactB1111Masters,
) -> (Vec<FastResult>, Vec<FastStatus>) {
    let lanes = case.scales.len();
    let input = FastInput {
        lanes,
        shifts: &case.shifts,
        masses_squared: &case.masses,
        scales_squared: &case.scales,
        s_matrix: Some(&case.smat),
    };
    let mut amplitude = FastAmplitude::new(case.n, case.rank, FastOptions::default()).unwrap();
    let mut results = vec![FastResult::default(); lanes];
    let mut statuses = vec![FastStatus::NotEvaluated; lanes];
    let mut numerator: Box<dyn FastNumerator> = match case.mu_powers {
        None => {
            let mut tensor = TensorNumerator::new(case.rank, lanes);
            for (lane, t) in case.tensors.iter().enumerate() {
                tensor.lane_coefficients_mut(lane).copy_from_slice(t);
            }
            Box::new(tensor)
        }
        Some(k) => {
            let mut tensor = MuTensorNumerator::new(case.rank, k, lanes).unwrap();
            for (lane, t) in case.tensors.iter().enumerate() {
                tensor.lane_coefficients_mut(lane).copy_from_slice(t);
            }
            Box::new(tensor)
        }
    };
    amplitude
        .evaluate(
            &input,
            numerator.as_mut(),
            masters,
            &mut results,
            &mut statuses,
        )
        .unwrap();
    (results, statuses)
}

fn laurent_error(result: &FastResult, reference: &[C64; 3]) -> f64 {
    let scale = reference.iter().map(|z| z.abs()).fold(0., f64::max);
    let scale = if scale == 0. { 1. } else { scale };
    result
        .laurent
        .iter()
        .zip(reference)
        .map(|(a, b)| (*a - *b).abs() / scale)
        .fold(0., f64::max)
}

fn suite_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("NINJARS_RANK_N2_SUITE")
        .map(PathBuf::from)
        .unwrap_or_else(|| root().join("validation/rank-n2"));
    dir.join("cases").is_dir().then_some(dir)
}

fn exact_masters() -> ExactB1111Masters {
    ExactB1111Masters {
        native: OneLoopNativeMasters::new().unwrap(),
        exact: FloatMasters::new(),
        scratch: Vec::new(),
    }
}

/// Per-lane scaled errors of a suite case (`<dir>/cases`, `<dir>/reference`)
/// reduced by the fast engine, or `None` if either file is missing.
fn suite_errors(dir: &Path, name: &str, masters: &mut ExactB1111Masters) -> Option<Vec<f64>> {
    let case_path = dir.join("cases").join(format!("{name}.json"));
    let reference_path = dir.join("reference").join(format!("{name}.json"));
    if !case_path.exists() || !reference_path.exists() {
        return None;
    }
    let case = parse_suite_case(&std::fs::read_to_string(&case_path).unwrap());
    let reference: Value =
        serde_json::from_str(&std::fs::read_to_string(&reference_path).unwrap()).unwrap();
    let (results, statuses) = reduce_suite_case(&case, masters);
    Some(
        reference["lanes"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(lane, entry)| {
                let want = [
                    c64(&entry["eps0"]),
                    c64(&entry["eps_m1"]),
                    c64(&entry["eps_m2"]),
                ];
                if statuses[lane] == FastStatus::Success {
                    laurent_error(&results[lane], &want)
                } else {
                    f64::INFINITY
                }
            })
            .collect(),
    )
}

/// The rank n+1 control twin of a rank n+2 suite case (same lanes, the
/// numerator one rank lower: `rn2-int-complex-n5-r7` -> `...-n5-r6`), reduced
/// by the validated rank n+1 engine: its per-lane error measures how far
/// binary64 can go on these kinematics.
fn twin_errors(
    suite: &Path,
    name: &str,
    n: usize,
    masters: &mut ExactB1111Masters,
) -> Option<Vec<f64>> {
    let twin = name.replacen(
        &format!("-n{n}-r{}", n + 2),
        &format!("-n{n}-r{}", n + 1),
        1,
    );
    suite_errors(&suite.join("controls"), &twin, masters)
}

/// Rank n+2 costs digits on top of rank n+1: two more Laurent orders per cut
/// and the new terms (B1111, mu⁴ and mu² (q.e2)² rationals) that cancel
/// against the rest by up to 1e5 in this suite. Observed on the review's
/// references: at most 4e2 times the rank n+1 twin's error.
const TWIN_FACTOR: f64 = 1e3;

fn twin_tolerance(twin: Option<f64>) -> f64 {
    twin.map_or(1e-9, |t| 1e-11f64.max(TWIN_FACTOR * t))
}

#[test]
fn fast_rank_n2_matches_high_precision_references() {
    let Some(suite) = suite_dir() else {
        eprintln!(
            "FAST_RANK_N2_HPREF skipped: no validation/rank-n2/cases (set NINJARS_RANK_N2_SUITE)"
        );
        return;
    };
    let mut masters = exact_masters();
    // (1) The review's 60-digit references, against the rank n+1 twins.
    let hpref = root().join("tools/rank_n2/review/hpref");
    let (mut errors, mut ratios) = (Vec::new(), Vec::new());
    for file in json_files(&hpref) {
        let reference: Value =
            serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        // `a0test*` are the review's harness self-checks, not suite cases.
        let Some(name) = reference["case"].as_str().filter(|n| n.starts_with("rn2-")) else {
            continue;
        };
        let case_path = suite.join("cases").join(format!("{name}.json"));
        let case = parse_suite_case(&std::fs::read_to_string(&case_path).unwrap());
        assert_eq!(case.rank, case.n + 2);
        let (results, statuses) = reduce_suite_case(&case, &mut masters);
        let twins = twin_errors(&suite, name, case.n, &mut masters);
        for (lane, entry) in reference["lanes"].as_array().unwrap().iter().enumerate() {
            assert_eq!(statuses[lane], FastStatus::Success, "{name} lane {lane}");
            let want = [
                c64(&entry["eps0"]),
                c64(&entry["eps_m1"]),
                c64(&entry["eps_m2"]),
            ];
            let error = laurent_error(&results[lane], &want);
            let twin = twins.as_ref().map(|t| t[lane]);
            let tolerance = twin_tolerance(twin);
            eprintln!(
                "FAST_RANK_N2_HPREF {name} lane {lane}: scaled error {error:.2e} (rank n+1 twin {:.2e})",
                twin.unwrap_or(f64::NAN)
            );
            assert!(
                error <= tolerance,
                "{name} lane {lane}: scaled error {error:e} > {tolerance:e}"
            );
            errors.push(error);
            if let Some(t) = twin {
                ratios.push(error / t.max(1e-15));
            }
        }
    }
    assert!(errors.len() >= 36, "hpref lanes checked: {}", errors.len());
    errors.sort_by(f64::total_cmp);
    ratios.sort_by(f64::total_cmp);
    eprintln!(
        "FAST_RANK_N2_HPREF lanes={} median_scaled_error={:.1e} max={:.1e}; ratio to the rank n+1 twin: median {:.1} max {:.1}",
        errors.len(),
        errors[errors.len() / 2],
        errors[errors.len() - 1],
        ratios[ratios.len() / 2],
        ratios[ratios.len() - 1]
    );
    // (2) The whole suite (report only: the per-lane binary64 tolerances of
    // design V4f are checked in tests/rank_n2_reference.rs).
    if !suite.join("reference").is_dir() {
        return;
    }
    let mut by_tier: [Vec<(f64, f64)>; 3] = Default::default();
    for case_path in json_files(&suite.join("cases")) {
        let name = case_path.file_stem().unwrap().to_string_lossy().to_string();
        let case = parse_suite_case(&std::fs::read_to_string(&case_path).unwrap());
        if case.rank != case.n + 2 {
            continue;
        }
        let Some(lane_errors) = suite_errors(&suite, &name, &mut masters) else {
            continue;
        };
        let twins = twin_errors(&suite, &name, case.n, &mut masters);
        let reference: Value = serde_json::from_str(
            &std::fs::read_to_string(suite.join("reference").join(format!("{name}.json"))).unwrap(),
        )
        .unwrap();
        for (lane, entry) in reference["lanes"].as_array().unwrap().iter().enumerate() {
            let tier = match entry["tier"].as_str().unwrap_or("C") {
                "A" => 0,
                "B" => 1,
                _ => 2,
            };
            assert!(lane_errors[lane].is_finite(), "{name} lane {lane} failed");
            let twin = twins.as_ref().map_or(f64::NAN, |t| t[lane]);
            // Ratios against twins below 1e-15 say nothing (rounding level).
            let ratio = if twin.is_nan() {
                f64::NAN
            } else {
                lane_errors[lane] / twin.max(1e-15)
            };
            if ratio > TWIN_FACTOR {
                eprintln!(
                    "FAST_RANK_N2_SUITE {name} lane {lane} (tier {}): scaled error {:.1e}, rank n+1 twin {twin:.1e}",
                    ["A", "B", "C"][tier],
                    lane_errors[lane]
                );
            }
            by_tier[tier].push((lane_errors[lane], ratio));
        }
    }
    for (tier, rows) in by_tier.iter_mut().enumerate() {
        if rows.is_empty() {
            continue;
        }
        let mut e: Vec<f64> = rows.iter().map(|r| r.0).collect();
        let mut r: Vec<f64> = rows.iter().map(|r| r.1).filter(|x| x.is_finite()).collect();
        e.sort_by(f64::total_cmp);
        r.sort_by(f64::total_cmp);
        eprintln!(
            "FAST_RANK_N2_SUITE tier {} lanes={} scaled error median {:.1e} p90 {:.1e} max {:.1e}; ratio to the rank n+1 twin median {:.1} max {:.1} ({} with twins)",
            ["A", "B", "C"][tier],
            e.len(),
            e[e.len() / 2],
            e[e.len() * 9 / 10],
            e[e.len() - 1],
            r.get(r.len() / 2).copied().unwrap_or(f64::NAN),
            r.last().copied().unwrap_or(f64::NAN),
            r.len()
        );
    }
}

#[test]
fn exact_b1111_masters_extend_the_native_rank3_family_bitwise() {
    // The composite provider's B0..B111 are the native rank-3 values.
    let mut masters = exact_masters();
    let args = [
        C64::real(3.7),
        C64::new(1.2, -0.1),
        C64::new(0.4, -0.02),
        C64::real(2.),
    ];
    let mut rank3 = [[C64::ZERO; 3]; 4];
    let mut rank4 = [[C64::ZERO; 3]; 5];
    let (mut ok3, mut ok4) = ([false], [false]);
    masters
        .evaluate(FastMaster::BubbleRank3, &args, &mut rank3, &mut ok3)
        .unwrap();
    masters
        .evaluate(FastMaster::BubbleRank4, &args, &mut rank4, &mut ok4)
        .unwrap();
    assert!(ok3[0] && ok4[0]);
    assert_eq!(&rank4[..4], &rank3[..]);
}
