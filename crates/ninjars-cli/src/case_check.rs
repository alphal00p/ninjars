//! `ninjars case-check`: reduce bench cases ("ninjars-bench-case-v1", and
//! "-v2" with explicit `mu^2` blocks) with the binary64 fast engine and/or
//! the generic engine at a chosen precision, compare every lane with a
//! reference file and its uncertainty, and report per lane `pass`, `fail`,
//! `report-only`, `known-failure` or `unchecked`.
//!
//! - Fast engine: `TensorNumerator` (`MuTensorNumerator` for v2 cases) and
//!   oneloopmaster's native masters, the case's lanes repeated cyclically up
//!   to `--width` exactly as `case-bench` batches them; repeated lanes must be
//!   bitwise identical. The report's `case_bench_reference_check` is
//!   `case-bench`'s own reference check of that batch.
//! - Generic engine (`SymbolicaAmplitudeKernels`): `f64` with native masters,
//!   `double` (DoubleFloat) and `float:<bits>` with oneloopmaster's exact
//!   expressions (`SymbolicaMasters`). The numerator is evaluated and expanded
//!   numerically from the dense tensor ([`harness::TensorGenericNumerator`]),
//!   so dense high-rank numerators need no Symbolica preparation.
//! - `--declare-rank R` (or `n+K`) runs every numerator declared at a higher
//!   rank with zero top coefficients (design V5).
//! - References: `<reference-dir>/<case>.json`, either the high-precision
//!   `ninjars-reference-collier-v1` files of `validation/rank-n2` (tier,
//!   uncertainty, condition) or original-Ninja results; tolerances in
//!   [`harness::Policy`].
//! - `--out-dir` writes one result file per case with 45-digit
//!   `lanes[].laurent_decimal` (`ninjars-generic-float-run-v1`), the
//!   `GENERIC_RUNNER` interface of `validation/rank-n2/harness/run_suite.sh`:
//!   `ninjars case-check --bits 256 --out-dir DIR CASE.json...`.

pub mod harness;

use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use clap::Args as ClapArgs;
use ninjars_symbolica::OneLoopNativeMasters;
use serde_json::{Value, json};

use harness::{
    BenchCase, CheckedLane, Engines, Policy, Precision, Reference, ReferenceKind, Verdict,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(ClapArgs)]
pub struct Args {
    /// "ninjars-bench-case-v1" or "-v2" files.
    pub cases: Vec<PathBuf>,
    /// Engines to run: fast, generic.
    #[arg(long, value_delimiter = ',', default_value = "fast")]
    pub engines: Vec<String>,
    /// Generic-engine precisions: f64 (native masters), double, float:<bits>
    /// (exact oneloopmaster masters).
    #[arg(long, value_delimiter = ',', default_value = "float:256")]
    pub precision: Vec<String>,
    /// Shorthand for `--engines generic --precision float:<BITS>`.
    #[arg(long)]
    pub bits: Option<u32>,
    /// Declared numerator rank: an integer or `n+K` (zero-padded top
    /// coefficients; default the case rank).
    #[arg(long)]
    pub declare_rank: Option<String>,
    /// Fast-engine batch width, at least the case's lane count (default: the
    /// lane count); lanes repeat cyclically.
    #[arg(long)]
    pub width: Option<usize>,
    /// Directory of reference files `<case>.json` (high-precision
    /// `ninjars-reference-collier-v1` or original-Ninja results).
    #[arg(long)]
    pub reference_dir: Option<PathBuf>,
    /// `ninjars-known-failures-v1` file: listed lanes that fail are reported
    /// as `known-failure`.
    #[arg(long)]
    pub known_failures: Option<PathBuf>,
    /// Binary64 tolerance floor: `factor max(1, cond)` (scaled to the unit
    /// roundoff for other fixed precisions).
    #[arg(long, default_value = "1e-14")]
    pub binary64_factor: f64,
    /// Fast engine binary64 budget in units of original Ninja's error on the
    /// lane (or its rank twins).
    #[arg(long, default_value = "30")]
    pub fast_ninja_factor: f64,
    /// The same for the generic engine in fixed precision.
    #[arg(long, default_value = "1000")]
    pub generic_ninja_factor: f64,
    /// `assemble.py` summaries with original Ninja's errors on rank twins
    /// (`validation/rank-n2/summary/{controls,synthetic-twins}.json`), for the
    /// binary64 budget of lanes without their own Ninja result.
    #[arg(long)]
    pub twins: Vec<PathBuf>,
    /// Scaled tolerance against original-Ninja references.
    #[arg(long, default_value = "1e-8")]
    pub ninja_tolerance: f64,
    /// Write per-case result files (`lanes[].laurent_decimal`); needs exactly
    /// one engine run.
    #[arg(long)]
    pub out_dir: Option<PathBuf>,
    /// Write the JSON report here instead of stdout.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Exit successfully even when lanes fail.
    #[arg(long)]
    pub no_fail: bool,
}

#[derive(Clone, Copy)]
enum Run {
    Fast,
    Generic(Precision),
}

impl Run {
    fn engine(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Generic(_) => "generic",
        }
    }
    fn precision(self) -> Precision {
        match self {
            Self::Fast => Precision::F64,
            Self::Generic(p) => p,
        }
    }
    fn label(self) -> String {
        format!("{}/{}", self.engine(), self.precision().label())
    }
    fn masters(self) -> &'static str {
        match self {
            Self::Fast | Self::Generic(Precision::F64) => {
                "oneloopmaster native (OneLoopNativeMasters)"
            }
            Self::Generic(_) => "oneloopmaster exact expressions (SymbolicaMasters)",
        }
    }
}

fn runs(args: &Args) -> Result<Vec<Run>> {
    let precisions: Vec<Precision> = match args.bits {
        Some(bits) => vec![Precision::Float(bits)],
        None => args
            .precision
            .iter()
            .map(|p| Precision::parse(p))
            .collect::<harness::Result<_>>()
            .map_err(|e| e.to_string())?,
    };
    let engines: Vec<String> = if args.bits.is_some() && args.engines == ["fast"] {
        vec!["generic".into()]
    } else {
        args.engines.clone()
    };
    let mut out = Vec::new();
    for engine in &engines {
        match engine.as_str() {
            "fast" => out.push(Run::Fast),
            "generic" => out.extend(precisions.iter().map(|&p| Run::Generic(p))),
            other => return Err(format!("unknown engine {other:?} (fast, generic)").into()),
        }
    }
    if out.is_empty() {
        return Err("no engine selected".into());
    }
    Ok(out)
}

fn declared_rank(spec: &Option<String>, case: &BenchCase) -> Result<usize> {
    let rank = match spec.as_deref() {
        None => case.rank,
        Some(text) => match text.strip_prefix("n+") {
            Some(k) => case.n + k.parse::<usize>()?,
            None if text == "n" => case.n,
            None => text.parse()?,
        },
    };
    if rank < case.rank {
        return Err(format!(
            "{}: declared rank {rank} below the case rank {}",
            case.name, case.rank
        )
        .into());
    }
    Ok(rank)
}

/// The JSON report of one checked lane.
fn lane_report(lane: &CheckedLane) -> Value {
    let names = ["eps0", "eps_m1", "eps_m2", "cut_constructible", "rational"];
    let values: serde_json::Map<String, Value> = names
        .iter()
        .zip(&lane.values)
        .map(|(name, v)| (name.to_string(), json!(v)))
        .collect();
    let mut report = json!({
        "id": lane.id,
        "status": lane.status,
        "values": values,
        "scaled_difference": lane.difference,
        "tolerance": lane.judgement.tolerance,
        "verdict": lane.judgement.verdict.label(),
    });
    if let Some(reason) = &lane.judgement.reason {
        report["reason"] = json!(reason);
    }
    if let Some((r, kind)) = &lane.reference {
        report["reference"] = match kind {
            ReferenceKind::HighPrecision => json!({
                "tier": r.tier,
                "uncertainty": r.uncertainty,
                "uncertainty_origin": r.uncertainty_origin,
                "condition": r.condition,
                "ninja_error": r.ninja_error,
                "scale": r.scale,
            }),
            ReferenceKind::Ninja => json!({"ninja_status": r.ninja_status, "scale": r.scale}),
        };
    }
    report
}

struct RunOutcome {
    /// Extra run fields (fast engine: width, bitwise repeats, case-bench check).
    extra: Value,
    lanes: Vec<CheckedLane>,
}

fn fast_case(
    args: &Args,
    case: &BenchCase,
    declared: usize,
    reference: Option<&Reference>,
    policy: &Policy,
    masters: &mut OneLoopNativeMasters,
) -> Result<RunOutcome> {
    let count = case.lanes.len();
    let width = args.width.unwrap_or(count);
    let checked = harness::check_fast(case, declared, width, reference, policy, masters)
        .map_err(|e| e.to_string())?;
    let check = crate::case_bench::reference_check(
        &args.reference_dir,
        &case.name,
        count,
        &checked.run.ids,
        &checked.run.results,
        &checked.run.statuses,
    )?;
    Ok(RunOutcome {
        extra: json!({
            "width": width,
            "repeats_bitwise": checked.repeats_bitwise,
            "case_bench_reference_check": check,
        }),
        lanes: checked.lanes,
    })
}

fn counts(verdicts: &[(Verdict, Option<f64>)]) -> Value {
    let mut by_verdict = BTreeMap::<&str, usize>::new();
    let mut worst_checked = 0_f64;
    for &(verdict, difference) in verdicts {
        *by_verdict.entry(verdict.label()).or_default() += 1;
        if matches!(verdict, Verdict::Pass | Verdict::Fail)
            && let Some(d) = difference
        {
            worst_checked = worst_checked.max(d);
        }
    }
    json!({ "verdicts": by_verdict, "max_scaled_difference_pass_or_fail": worst_checked })
}

fn write_results(
    dir: &Path,
    case: &BenchCase,
    run: Run,
    declared: usize,
    outcome: &RunOutcome,
    seconds: f64,
) -> Result<()> {
    fs::create_dir_all(dir)?;
    let keys = ["eps0", "eps_m1", "eps_m2"];
    let lanes: Vec<Value> = case
        .lanes
        .iter()
        .zip(&outcome.lanes)
        .map(|(lane, result)| {
            if result.succeeded {
                let values = &result.values;
                json!({
                    "id": lane.id,
                    "laurent_decimal": keys.iter().zip(values).map(|(k, v)| (k.to_string(), json!(v))).collect::<serde_json::Map<_, _>>(),
                    "cut_constructible": values[3],
                    "rational": values[4],
                })
            } else {
                json!({ "id": lane.id, "error": result.status })
            }
        })
        .collect();
    let bits = match run.precision() {
        Precision::F64 => 53,
        Precision::Double => 106,
        Precision::Float(bits) => bits,
    };
    let document = json!({
        "format": "ninjars-generic-float-run-v1",
        "case": case.name,
        "bits": bits,
        "engine": format!("NinjaRS {} engine ({}), masters: {}", run.engine(), run.precision().label(), run.masters()),
        "declared_rank": declared,
        "seconds": seconds,
        "lanes": lanes,
    });
    fs::write(
        dir.join(format!("{}.json", case.name)),
        serde_json::to_string_pretty(&document)? + "\n",
    )?;
    Ok(())
}

pub fn run(args: Args) -> Result<()> {
    let runs = runs(&args)?;
    if args.out_dir.is_some() && runs.len() != 1 {
        return Err("--out-dir needs exactly one engine run".into());
    }
    if args.cases.is_empty() {
        return Err("no case files given".into());
    }
    let mut policy = Policy {
        binary64_factor: args.binary64_factor,
        fast_ninja_factor: args.fast_ninja_factor,
        generic_ninja_factor: args.generic_ninja_factor,
        ninja_tolerance: args.ninja_tolerance,
        ..Policy::default()
    };
    if let Some(path) = &args.known_failures {
        policy
            .load_known_failures(path)
            .map_err(|e| e.to_string())?;
    }
    for path in &args.twins {
        policy.load_twins(path).map_err(|e| e.to_string())?;
    }
    let mut fast_masters = None;
    let mut generic = Engines::default();
    let mut reports = Vec::new();
    let mut totals = BTreeMap::<String, Vec<(Verdict, Option<f64>)>>::new();
    let mut errors = 0;
    for path in &args.cases {
        let case = BenchCase::load(path).map_err(|e| e.to_string())?;
        let declared = declared_rank(&args.declare_rank, &case)?;
        let reference = match &args.reference_dir {
            Some(dir) => Reference::find(dir, &case.name).map_err(|e| e.to_string())?,
            None => None,
        };
        let mut case_runs = Vec::new();
        for &run in &runs {
            let started = Instant::now();
            let outcome = match run {
                Run::Fast => {
                    if fast_masters.is_none() {
                        fast_masters = Some(OneLoopNativeMasters::new()?);
                    }
                    fast_case(
                        &args,
                        &case,
                        declared,
                        reference.as_ref(),
                        &policy,
                        fast_masters.as_mut().unwrap(),
                    )
                }
                Run::Generic(precision) => generic
                    .check(precision, &case, declared, reference.as_ref(), &policy)
                    .map(|lanes| RunOutcome {
                        extra: json!({}),
                        lanes,
                    })
                    .map_err(|e| e.to_string().into()),
            };
            let seconds = started.elapsed().as_secs_f64();
            let mut entry = json!({
                "engine": run.engine(),
                "precision": run.precision().label(),
                "masters": run.masters(),
                "seconds": seconds,
            });
            let verdicts = match outcome {
                Ok(outcome) => {
                    if let Some(dir) = &args.out_dir {
                        write_results(dir, &case, run, declared, &outcome, seconds)?;
                    }
                    for (key, value) in outcome.extra.as_object().unwrap() {
                        entry[key] = value.clone();
                    }
                    entry["lanes"] = outcome.lanes.iter().map(lane_report).collect();
                    outcome
                        .lanes
                        .iter()
                        .map(|lane| (lane.judgement.verdict, lane.difference))
                        .collect()
                }
                Err(error) => {
                    errors += 1;
                    entry["error"] = json!(error.to_string());
                    vec![(Verdict::Error, None); case.lanes.len()]
                }
            };
            entry["summary"] = counts(&verdicts);
            eprintln!(
                "case-check {} {} (declared rank {declared}): {} in {seconds:.1}s",
                case.name,
                run.label(),
                entry["summary"]["verdicts"]
            );
            totals.entry(run.label()).or_default().extend(verdicts);
            case_runs.push(entry);
        }
        reports.push(json!({
            "case": case.name,
            "file": path,
            "n": case.n,
            "rank": case.rank,
            "declared_rank": declared,
            "mu_powers": case.mu_powers,
            "lanes": case.lanes.len(),
            "reference": reference.as_ref().map(|r| json!({"path": r.path, "format": r.format})),
            "runs": case_runs,
        }));
    }
    let failures: usize = totals
        .values()
        .flatten()
        .filter(|(v, _)| matches!(v, Verdict::Fail | Verdict::Error))
        .count();
    let report = json!({
        "schema_version": 1,
        "operation": "case-check",
        "command": std::env::args().collect::<Vec<_>>(),
        "policy": policy.describe(),
        "summary": totals.iter().map(|(k, v)| (k.clone(), counts(v))).collect::<serde_json::Map<_, _>>(),
        "failed_lanes": failures,
        "case_errors": errors,
        "cases": reports,
    });
    let text = serde_json::to_string_pretty(&report)? + "\n";
    match &args.output {
        Some(path) => fs::write(path, text)?,
        None => print!("{text}"),
    }
    if failures > 0 && !args.no_fail {
        return Err(format!("{failures} lanes failed or could not be evaluated").into());
    }
    Ok(())
}
