//! Shared driver of the rank n+2 validation tests (design section 7, S5b).
//!
//! The bench-case model, reference files, tolerance policy, the
//! precision-generic tensor numerator and both engine runners are those of
//! `ninjars case-check` (`crates/ninjars-cli/src/case_check/harness.rs`),
//! compiled here as well so that the tests and the command agree exactly.
//! The data live in `validation/rank-n2` (see its README).

#![allow(dead_code)]

#[path = "../../../ninjars-cli/src/case_check/harness.rs"]
pub mod harness;

use std::{
    fmt::Write as _,
    path::{Path, PathBuf},
    time::Instant,
};

use harness::{BenchCase, CheckedLane, Engines, Policy, Precision, Reference, Verdict};
use ninjars_symbolica::OneLoopNativeMasters;

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn rank_n2() -> PathBuf {
    root().join("validation/rank-n2")
}

/// Sorted `*.json` files of a directory whose file name satisfies `keep`.
pub fn cases_in(dir: &Path, keep: impl Fn(&str) -> bool) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "json")
                && keep(p.file_name().unwrap().to_str().unwrap())
        })
        .collect();
    files.sort();
    files
}

/// `n` of a case file name `...-n<n>-r<rank>...` (`syn-n4-r5`, `rn2-int-real-n4-r6-mu2`).
pub fn name_n(name: &str) -> Option<usize> {
    name.split('-')
        .find_map(|part| part.strip_prefix('n').and_then(|n| n.parse().ok()))
}

/// The suite's rank <= n+1 controls: the twins of `controls/cases` and the
/// synthetic benchmark cases `syn-n{2..6}-r{n,n+1}` whose references are in
/// `controls/reference`.
pub fn controls(max_n: usize) -> Vec<PathBuf> {
    let mut files = cases_in(&rank_n2().join("controls/cases"), |name| {
        name_n(name).is_some_and(|n| n <= max_n)
    });
    for n in 2..=max_n.min(6) {
        for rank in [n, n + 1] {
            files.push(root().join(format!(
                "validation/benchmark/synthetic/syn-n{n}-r{rank}.json"
            )));
        }
    }
    files
}

/// The tolerance policy with the known failures and the twins' original-Ninja
/// errors of `validation/rank-n2`.
pub fn policy() -> Policy {
    let mut policy = Policy::default();
    policy
        .load_known_failures(&rank_n2().join("known-failures.json"))
        .unwrap();
    for summary in ["controls", "synthetic-twins", "n7-controls"] {
        let path = rank_n2().join(format!("summary/{summary}.json"));
        if path.exists() {
            policy.load_twins(&path).unwrap();
        }
    }
    policy
}

/// Every lane of a set of cases with one engine run.
pub struct SetReport {
    pub label: String,
    pub lanes: Vec<(String, CheckedLane)>,
    pub errors: Vec<(String, String)>,
    pub seconds: f64,
}

impl SetReport {
    fn new(label: String) -> Self {
        Self {
            label,
            lanes: Vec::new(),
            errors: Vec::new(),
            seconds: 0.,
        }
    }

    pub fn count(&self, verdict: Verdict) -> usize {
        self.lanes
            .iter()
            .filter(|(_, l)| l.judgement.verdict == verdict)
            .count()
    }

    /// Largest scaled difference over passing or failing lanes.
    pub fn worst(&self) -> f64 {
        self.lanes
            .iter()
            .filter(|(_, l)| matches!(l.judgement.verdict, Verdict::Pass | Verdict::Fail))
            .filter_map(|(_, l)| l.difference)
            .fold(0., f64::max)
    }

    pub fn summary(&self) -> String {
        let mut text = format!(
            "{}: {} lanes in {:.1}s: pass {}, fail {}, report-only {}, known-failure {}, unchecked {}, case errors {}; max checked difference {:.2e}",
            self.label,
            self.lanes.len(),
            self.seconds,
            self.count(Verdict::Pass),
            self.count(Verdict::Fail),
            self.count(Verdict::ReportOnly),
            self.count(Verdict::KnownFailure),
            self.count(Verdict::Unchecked),
            self.errors.len(),
            self.worst()
        );
        for (case, lane) in &self.lanes {
            if !matches!(lane.judgement.verdict, Verdict::Pass | Verdict::Unchecked) {
                let _ = write!(
                    text,
                    "\n  {} {case} {}: {:?} difference {:?} tolerance {:?} ({})",
                    lane.judgement.verdict.label(),
                    lane.id,
                    lane.status,
                    lane.difference,
                    lane.judgement.tolerance,
                    lane.judgement
                        .reason
                        .as_deref()
                        .unwrap_or("")
                        .chars()
                        .take(110)
                        .collect::<String>()
                );
            }
        }
        for (case, error) in &self.errors {
            let _ = write!(text, "\n  error {case}: {error}");
        }
        text
    }

    /// No failing lane, no case error, and every case lane checked.
    pub fn assert_clean(&self, expect_checked: bool) {
        eprintln!("{}", self.summary());
        assert!(
            self.errors.is_empty() && self.count(Verdict::Fail) == 0,
            "{}",
            self.summary()
        );
        if expect_checked {
            assert_eq!(self.count(Verdict::Unchecked), 0, "{}", self.summary());
        }
    }
}

/// Which engine run [`check_set`] performs.
#[derive(Clone, Copy, Debug)]
pub enum Run {
    Fast,
    Generic(Precision),
}

/// Reduce every case (declared at `declare(case)`) and judge it against
/// `<reference_dir>/<case>.json`.
pub fn check_set(
    files: &[PathBuf],
    reference_dir: &Path,
    run: Run,
    declare: impl Fn(&BenchCase) -> usize,
    policy: &Policy,
) -> SetReport {
    let label = match run {
        Run::Fast => "fast/f64".to_string(),
        Run::Generic(p) => format!("generic/{}", p.label()),
    };
    let mut report = SetReport::new(label);
    let started = Instant::now();
    let mut engines = Engines::default();
    let mut masters = OneLoopNativeMasters::new().unwrap();
    for file in files {
        let case = BenchCase::load(file).unwrap();
        let reference = Reference::find(reference_dir, &case.name).unwrap();
        let declared = declare(&case);
        let lanes = match run {
            Run::Fast => harness::check_fast(
                &case,
                declared,
                case.lanes.len(),
                reference.as_ref(),
                policy,
                &mut masters,
            )
            .map(|check| {
                assert!(check.repeats_bitwise, "{}", case.name);
                check.lanes
            }),
            Run::Generic(precision) => {
                engines.check(precision, &case, declared, reference.as_ref(), policy)
            }
        };
        match lanes {
            Ok(lanes) => report
                .lanes
                .extend(lanes.into_iter().map(|l| (case.name.clone(), l))),
            Err(error) => report.errors.push((case.name.clone(), error.to_string())),
        }
    }
    report.seconds = started.elapsed().as_secs_f64();
    report
}

/// Laurent coefficients of a decimal lane as `[re, im]` strings.
pub fn laurent(lane: &CheckedLane) -> &[[String; 2]] {
    &lane.values[..3]
}

/// The case with every numerator truncated to rank `rank` (<= the case
/// rank): each block `T_k` keeps its monomials of degree <= `rank - 2k`, and
/// blocks with `2k > rank` are dropped. Truncation commutes with propagator
/// relabelling, so the relabelling identities of a case hold for it too.
pub fn truncated(case: &BenchCase, rank: usize) -> BenchCase {
    use ninjars::fast::tensor_layout::{mu_block_offset, ncoeff};
    assert!(rank <= case.rank);
    let mu_powers = case.mu_powers.min(rank / 2);
    let mut out = case.clone();
    out.rank = rank;
    out.mu_powers = mu_powers;
    for lane in &mut out.lanes {
        let mut tensor = Vec::new();
        for k in 0..=mu_powers {
            let from = mu_block_offset(case.rank, k);
            tensor.extend_from_slice(&lane.tensor[from..from + ncoeff(rank - 2 * k)]);
        }
        lane.tensor = tensor;
    }
    out
}

/// Exact coefficients of `N(q - P)` from those of `N(q)` (one dense block of
/// rank `rank`, any integer `P`), in exact integer arithmetic on the dyadic
/// inputs. Panics if an input or output is not exactly representable.
pub fn shifted_block(
    block: &[ninjars::fast::C64],
    rank: usize,
    p: [i64; 4],
) -> Vec<ninjars::fast::C64> {
    use ninjars::fast::{
        C64,
        tensor_layout::{monomial_index, monomials, ncoeff},
    };
    const SCALE: i32 = 60;
    let to_int = |x: f64| -> i128 {
        let scaled = x * 2_f64.powi(SCALE);
        assert!(
            scaled.fract() == 0. && scaled.abs() < 2_f64.powi(120),
            "{x} is not dyadic"
        );
        scaled as i128
    };
    let to_f64 = |x: i128| -> f64 {
        let value = x as f64;
        assert_eq!(
            value as i128, x,
            "shifted coefficient not exact in binary64"
        );
        value * 2_f64.powi(-SCALE)
    };
    let binomial = |n: u32, k: u32| -> i128 {
        (0..k).fold(1, |acc, i| acc * (n - i) as i128 / (i + 1) as i128)
    };
    assert_eq!(block.len(), ncoeff(rank));
    let mut re = vec![0_i128; block.len()];
    let mut im = vec![0_i128; block.len()];
    for (m, c) in monomials(rank).into_iter().zip(block) {
        let (cr, ci) = (to_int(c.re), to_int(c.im));
        if cr == 0 && ci == 0 {
            continue;
        }
        // prod_mu (q_mu - P_mu)^(m_mu) = sum_j prod_mu C(m_mu, j_mu) (-P_mu)^(m_mu - j_mu) q^j
        for j0 in 0..=m[0] {
            for j1 in 0..=m[1] {
                for j2 in 0..=m[2] {
                    for j3 in 0..=m[3] {
                        let j = [j0, j1, j2, j3];
                        let mut factor: i128 = 1;
                        for mu in 0..4 {
                            factor *= binomial(m[mu], j[mu]) * (-p[mu] as i128).pow(m[mu] - j[mu]);
                        }
                        let index = monomial_index(j);
                        re[index] += factor * cr;
                        im[index] += factor * ci;
                    }
                }
            }
        }
    }
    re.into_iter()
        .zip(im)
        .map(|(r, i)| C64::new(to_f64(r), to_f64(i)))
        .collect()
}

/// `case` with every lane's shifts `V_i - P_lane` and numerator
/// `N(q - P_lane)` (exact): the same integrals.
pub fn shifted(case: &BenchCase, p: &[[i64; 4]]) -> BenchCase {
    use ninjars::fast::tensor_layout::{mu_block_offset, ncoeff};
    let mut out = case.clone();
    out.name = format!("{}-shifted", case.name);
    for (lane, shift) in out.lanes.iter_mut().zip(p) {
        for v in &mut lane.shifts {
            for mu in 0..4 {
                v[mu] -= shift[mu] as f64;
            }
        }
        let mut tensor = Vec::new();
        for k in 0..=case.mu_powers {
            let from = mu_block_offset(case.rank, k);
            let size = ncoeff(case.rank - 2 * k);
            tensor.extend(shifted_block(
                &lane.tensor[from..from + size],
                case.rank - 2 * k,
                *shift,
            ));
        }
        lane.tensor = tensor;
    }
    out
}

/// `case` with the propagators of every lane listed in reverse order (the
/// same integrals).
pub fn reversed(case: &BenchCase) -> BenchCase {
    let n = case.n;
    let mut out = case.clone();
    for lane in &mut out.lanes {
        lane.shifts.reverse();
        lane.masses.reverse();
        if let Some(s) = &mut lane.s_matrix {
            let old = s.clone();
            for i in 0..n {
                for j in 0..n {
                    s[i * n + j] = old[(n - 1 - i) * n + (n - 1 - j)];
                }
            }
        }
    }
    out
}

/// Binary64 noise of every lane: the scaled difference of the fast engine's
/// Laurent coefficients when the propagators are listed in reverse order
/// (as `tests/fast_parity.rs`). Lanes that fail in either order get 0.
pub fn fast_noise(
    case: &BenchCase,
    declared_rank: usize,
    masters: &mut OneLoopNativeMasters,
) -> Vec<f64> {
    use harness::{fast_values, run_fast};
    use ninjars::fast::FastStatus;
    let count = case.lanes.len();
    let a = run_fast(case, declared_rank, count, masters).unwrap();
    let b = run_fast(&reversed(case), declared_rank, count, masters).unwrap();
    (0..count)
        .map(|lane| {
            if a.statuses[lane] != FastStatus::Success || b.statuses[lane] != FastStatus::Success {
                return 0.;
            }
            let (x, y) = (fast_values(&a.results[lane]), fast_values(&b.results[lane]));
            let scale = x[..3]
                .iter()
                .map(|z| z.abs())
                .fold(0., f64::max)
                .max(1e-300);
            (0..3).map(|k| (x[k] - y[k]).abs()).fold(0., f64::max) / scale
        })
        .collect()
}
