//! Warmed Rust measurements. Reference performance acceptance is a separate gate.

use std::{
    error::Error,
    fs,
    hint::black_box,
    num::NonZeroUsize,
    path::PathBuf,
    time::{Duration, Instant},
};

use clap::{Args as ClapArgs, ValueEnum};
use ninjars::{Complex, fast::FastStatus};
use serde_json::{Value, json};
use symbolica::{
    domains::float::{DoubleFloat, Float},
    evaluate::EvaluationDomain,
};

use crate::{
    fast_runtime::{self, FastBatch, FastPool, FastSession},
    numerator_coefficients::CoefficientScalar,
    prepare::{self, PreparedDocument},
    reduce::{self, Engine, MuExtraction, NumeratorEvaluator, Precision, ReductionKernels},
    runtime::{CallTiming, DiagnosticTimings, EvaluationSession, NumericalBatch, NumericalPool},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, ValueEnum)]
pub enum Workload {
    Repeated,
    Changing,
    Both,
}

#[derive(ClapArgs)]
pub struct Args {
    /// DOT graph with --samples, or a schema-version-1 JSON fixture.
    pub input: PathBuf,
    #[arg(long)]
    pub samples: Option<PathBuf>,
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long)]
    pub feynman_rules: bool,
    /// R2 from the numerator: `off` (four-dimensional loop momentum, R1 only)
    /// or `fdh` (K.K -> q.q - mu^2 in FDH; the mu^2 terms give R2).
    #[arg(long, value_enum, default_value = "off")]
    pub r2: prepare::R2,
    /// Reduction engine; `fast` requires --precision f64 and the default box mode.
    #[arg(long, value_enum, default_value = "generic")]
    pub engine: Engine,
    #[arg(long, value_enum, default_value = "f64")]
    pub precision: Precision,
    /// Numerator preparation provider; coefficient evaluation stays inside every timed iteration.
    #[arg(long, value_enum, default_value = "symbolica")]
    pub numerator_evaluator: NumeratorEvaluator,
    /// Reduction execution, independently of the numerator preparation provider.
    #[arg(long, value_enum, default_value = "symbolica")]
    pub reduction_kernels: ReductionKernels,
    /// Bound complete native kernel rows per call; omission keeps whole-batch execution.
    #[arg(long)]
    pub native_block_rows: Option<NonZeroUsize>,
    #[arg(long)]
    pub bits: Option<u32>,
    /// Phase-space batch widths; the default 1025-point round includes tails.
    #[arg(long, value_delimiter = ',', default_value = "1,8,32,128,1024")]
    pub batch_sizes: Vec<NonZeroUsize>,
    /// Prepared points per iteration, cycling the fixture pool for changing inputs.
    #[arg(long, default_value = "1025")]
    pub points: NonZeroUsize,
    #[arg(long, value_enum, default_value = "both")]
    pub workload: Workload,
    /// Complete untimed passes over every full and tail batch, at least one.
    #[arg(long, default_value = "1")]
    pub warmup: NonZeroUsize,
    /// Separate elapsed-time samples; raw samples are retained in the report.
    #[arg(long, default_value = "5")]
    pub rounds: NonZeroUsize,
    /// Complete point-pool passes within each timing sample.
    #[arg(long, default_value = "1")]
    pub iterations: NonZeroUsize,
    #[arg(long, default_value = "1")]
    pub cut_stop: usize,
    #[arg(long, value_enum, default_value = "expansion")]
    pub mu_extraction: MuExtraction,
    #[arg(long)]
    pub full_box_coefficients: bool,
    #[arg(long)]
    pub mu_sampling_scale: Option<String>,
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

impl Args {
    fn reduction(&self) -> reduce::Args {
        reduce::Args {
            input: self.input.clone(),
            samples: self.samples.clone(),
            model: self.model.clone(),
            feynman_rules: self.feynman_rules,
            r2: self.r2,
            engine: self.engine,
            precision: self.precision,
            numerator_evaluator: self.numerator_evaluator,
            reduction_kernels: self.reduction_kernels,
            native_block_rows: self.native_block_rows,
            bits: self.bits,
            batch_size: self.batch_sizes[0],
            cut_stop: self.cut_stop,
            diagnostics: false,
            mu_extraction: self.mu_extraction,
            full_box_coefficients: self.full_box_coefficients,
            mu_sampling_scale: self.mu_sampling_scale.clone(),
            output: None,
        }
    }
}

fn timing(timing: &CallTiming) -> Value {
    json!({"nanoseconds":timing.elapsed.as_secs_f64()*1e9,"calls":timing.calls,"rows":timing.rows})
}

fn stage_accounting(total: Duration, diagnostic: &DiagnosticTimings) -> Result<Value> {
    let coefficients = diagnostic.numerator_coefficients.elapsed;
    let callbacks =
        diagnostic.numerator_evaluation.elapsed + diagnostic.numerator_expansion.elapsed;
    let masters = diagnostic.masters.elapsed;
    let residual = total
        .checked_sub(coefficients + callbacks + masters)
        .ok_or("diagnostic stage intervals exceed their containing elapsed interval")?;
    Ok(json!({
        "total": total.as_nanos(),
        "numerator_coefficients": coefficients.as_nanos(),
        "numerator_callbacks": callbacks.as_nanos(),
        "actual_master_provider": masters.as_nanos(),
        "residual_reduction": residual.as_nanos(),
        "reduction_excluding_masters_and_coefficients": (residual + callbacks).as_nanos(),
    }))
}

fn quantile(sorted: &[f64], fraction: f64) -> f64 {
    let position = fraction * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
}

/// One reusable reduction session as the benchmark drives it.
trait Measured {
    type Batch;
    /// Diagnostic scope of the master timer.
    const MASTER_SCOPE: &'static str;
    /// Diagnostic scope of the residual reduction time.
    const RESIDUAL_SCOPE: &'static str;
    fn pack(&self, indices: &[usize]) -> Result<Self::Batch>;
    fn lanes(batch: &Self::Batch) -> usize;
    fn evaluate(&mut self, batch: &mut Self::Batch) -> Result<()>;
    fn evaluate_diagnostic(
        &mut self,
        batch: &mut Self::Batch,
        timings: &mut DiagnosticTimings,
    ) -> Result<()>;
    /// `None` for a successful lane, otherwise its failure kind.
    fn failure(&self, batch: &Self::Batch, lane: usize) -> Option<String>;
    /// Keep the last results observable inside timed loops.
    fn observe(&self);
    /// Encoded components and their precision bits of a successful lane.
    fn representative(&self, batch: &Self::Batch, lane: usize) -> Option<(Value, Value)>;
}

struct Generic<T: CoefficientScalar>
where
    Complex<T>: EvaluationDomain,
{
    pool: NumericalPool<T>,
    session: EvaluationSession<T>,
    digits: usize,
}

impl<T: CoefficientScalar> Measured for Generic<T>
where
    Complex<T>: EvaluationDomain,
{
    type Batch = NumericalBatch<T>;
    const MASTER_SCOPE: &'static str = "Actual direct SymbolicaMasters execution, including provider validation and conversion; no numerical result cache or shared frozen-master lookup";
    const RESIDUAL_SCOPE: &'static str = "scale selection, parameter binding, reset, geometry, reduction, contraction, bookkeeping and timer overhead";
    fn pack(&self, indices: &[usize]) -> Result<Self::Batch> {
        self.pool.batch(indices)
    }
    fn lanes(batch: &Self::Batch) -> usize {
        batch.lanes
    }
    fn evaluate(&mut self, batch: &mut Self::Batch) -> Result<()> {
        self.session.evaluate(batch)
    }
    fn evaluate_diagnostic(
        &mut self,
        batch: &mut Self::Batch,
        timings: &mut DiagnosticTimings,
    ) -> Result<()> {
        self.session.evaluate_diagnostic(batch, timings)
    }
    fn failure(&self, batch: &Self::Batch, lane: usize) -> Option<String> {
        if self.session.successful(batch, lane) {
            None
        } else if let Some(failure) = self.session.scale_failure(batch, lane) {
            Some(format!("sampling_scale:{failure:?}"))
        } else {
            Some(format!("{:?}", self.session.amplitude.statuses()[lane]))
        }
    }
    fn observe(&self) {
        black_box(self.session.amplitude.results());
        black_box(self.session.amplitude.statuses());
    }
    fn representative(&self, batch: &Self::Batch, lane: usize) -> Option<(Value, Value)> {
        let digits = self.digits;
        self.session.successful(batch, lane).then(|| {
            let result = &self.session.amplitude.results()[lane];
            (
                json!(result.each_ref().map(|value| {
                    [
                        format!("{:.*e}", digits, &value.re),
                        format!("{:.*e}", digits, &value.im),
                    ]
                })),
                json!(
                    result
                        .each_ref()
                        .map(|value| [value.re.get_precision(), value.im.get_precision()])
                ),
            )
        })
    }
}

struct Fast {
    pool: FastPool,
    session: FastSession,
}

impl Measured for Fast {
    type Batch = FastBatch;
    const MASTER_SCOPE: &'static str = "Actual oneloopmaster native binary64 execution (OneLoopNativeMasters), one batched call per family, including on-shell thresholding; no numerical result cache";
    const RESIDUAL_SCOPE: &'static str = "parameter binding, reset, geometry, reduction, subtractions, contraction, bookkeeping and timer overhead";
    fn pack(&self, indices: &[usize]) -> Result<Self::Batch> {
        self.pool.batch(indices)
    }
    fn lanes(batch: &Self::Batch) -> usize {
        batch.lanes
    }
    fn evaluate(&mut self, batch: &mut Self::Batch) -> Result<()> {
        self.session.evaluate(batch)
    }
    fn evaluate_diagnostic(
        &mut self,
        batch: &mut Self::Batch,
        timings: &mut DiagnosticTimings,
    ) -> Result<()> {
        self.session.evaluate_diagnostic(batch, timings)
    }
    fn failure(&self, _: &Self::Batch, lane: usize) -> Option<String> {
        let status = self.session.status(lane);
        (status != FastStatus::Success).then(|| format!("{status:?}"))
    }
    fn observe(&self) {
        black_box(self.session.results());
        black_box(self.session.statuses());
    }
    fn representative(&self, _: &Self::Batch, lane: usize) -> Option<(Value, Value)> {
        (self.session.status(lane) == FastStatus::Success).then(|| {
            (
                json!(self.session.components(lane).map(fast_runtime::encode)),
                json!(fast_runtime::COMPONENT_BITS),
            )
        })
    }
}

/// Engine-specific report fields.
struct EngineReport {
    precision: Value,
    rank: usize,
    cut_stop: usize,
    numerator_provider: Value,
    reduction_kernels: Value,
    box_reduction: Value,
    measurement_scope: &'static str,
    numerical_binding_ns: f64,
    initial_session_ns: f64,
}

fn timed_points(prepared: &PreparedDocument, args: &Args) -> Result<usize> {
    if prepared.samples.is_empty() {
        return Err("benchmark requires at least one prepared sample".into());
    }
    if !matches!(args.workload, Workload::Repeated)
        && (prepared.samples.len() < 2 || args.points.get() < 2)
    {
        return Err("changing workloads require at least two fixture samples and two points; use --workload repeated for one point".into());
    }
    let timed_points = args
        .points
        .get()
        .checked_mul(args.iterations.get())
        .ok_or("timed point count overflow")?;
    timed_points
        .checked_mul(args.rounds.get())
        .ok_or("total timed point count overflow")?;
    Ok(timed_points)
}

fn cases<M: Measured>(
    measured: &mut M,
    prepared: &PreparedDocument,
    args: &Args,
    timed_points: usize,
    success: &mut bool,
) -> Result<Vec<Value>> {
    let workloads = match args.workload {
        Workload::Repeated => vec![false],
        Workload::Changing => vec![true],
        Workload::Both => vec![false, true],
    };
    let mut cases = Vec::new();
    for changing in workloads {
        let indices: Vec<_> = (0..args.points.get())
            .map(|point| {
                if changing {
                    point % prepared.samples.len()
                } else {
                    0
                }
            })
            .collect();
        for width in &args.batch_sizes {
            let packing_start = Instant::now();
            let mut batches = indices
                .chunks(width.get())
                .map(|rows| measured.pack(rows))
                .collect::<Result<Vec<_>>>()?;
            let packing_ns = packing_start.elapsed().as_secs_f64() * 1e9;
            let warmup_start = Instant::now();
            let mut warmup_failed_lanes = 0;
            for pass in 0..args.warmup.get() {
                for batch in &mut batches {
                    measured.evaluate(batch)?;
                    if pass + 1 == args.warmup.get() {
                        warmup_failed_lanes += (0..M::lanes(batch))
                            .filter(|lane| measured.failure(batch, *lane).is_some())
                            .count();
                    }
                }
            }
            let warmup_ns = warmup_start.elapsed().as_secs_f64() * 1e9;
            let mut samples_ns = Vec::with_capacity(args.rounds.get());
            for _ in 0..args.rounds.get() {
                let start = Instant::now();
                for _ in 0..args.iterations.get() {
                    for batch in &mut batches {
                        measured.evaluate(batch)?;
                        measured.observe();
                    }
                }
                samples_ns.push(start.elapsed().as_secs_f64() * 1e9);
            }
            // A separate pass measures attribution. Its timer/counter overhead
            // cannot enter the primary measurements above.
            let mut diagnostic = DiagnosticTimings::default();
            let mut diagnostic_elapsed = Duration::ZERO;
            let mut failed_lanes = 0;
            let mut failure_kinds = std::collections::BTreeMap::<String, usize>::new();
            for batch in &mut batches {
                let start = Instant::now();
                measured.evaluate_diagnostic(batch, &mut diagnostic)?;
                diagnostic_elapsed += start.elapsed();
                for lane in 0..M::lanes(batch) {
                    if let Some(kind) = measured.failure(batch, lane) {
                        failed_lanes += 1;
                        *failure_kinds.entry(kind).or_default() += 1;
                    }
                }
            }
            *success &= failed_lanes == 0 && warmup_failed_lanes == 0;
            let stages = stage_accounting(diagnostic_elapsed, &diagnostic)?;
            let mut sorted = samples_ns.clone();
            sorted.sort_by(f64::total_cmp);
            let median = quantile(&sorted, 0.5);
            let last = batches.last().unwrap();
            let lane = M::lanes(last) - 1;
            let (representative, representative_bits) = measured
                .representative(last, lane)
                .unwrap_or((Value::Null, Value::Null));
            cases.push(json!({"workload":if changing {"changing"} else {"repeated"},"batch_size":width.get(),
                "points_per_iteration":args.points.get(),"full_batches":args.points.get()/width.get(),"tail_lanes":args.points.get()%width.get(),
                "pool_policy":if changing {"cyclic fixture rows in fixture order"} else {"fixture row zero repeated"},
                "distinct_source_rows":if changing {args.points.get().min(prepared.samples.len())} else {1},
                "packing_ns":packing_ns,"warmup_ns":warmup_ns,"warmup_passes":args.warmup.get(),
                "timed_rounds":args.rounds.get(),"iterations_per_round":args.iterations.get(),"timed_points_per_round":timed_points,
                "round_ns":samples_ns,"median_ns":median,"min_ns":sorted[0],"max_ns":sorted[sorted.len()-1],
                "p10_ns":quantile(&sorted,0.1),"p90_ns":quantile(&sorted,0.9),
                "median_ns_per_point":median/timed_points as f64,"median_points_per_second":timed_points as f64*1e9/median,
                "uncertainty_policy":"raw round samples and observed quantiles; no confidence-interval claim",
                "warmup_final_pass_failed_lanes":warmup_failed_lanes,
                "diagnostic":{"total_ns":diagnostic_elapsed.as_secs_f64()*1e9,"points":args.points.get(),
                    "numerator_coefficients":timing(&diagnostic.numerator_coefficients),
                    "numerator_evaluation":timing(&diagnostic.numerator_evaluation),"numerator_expansion":timing(&diagnostic.numerator_expansion),
                    "masters":timing(&diagnostic.masters),"residual_reduction_ns":stages["residual_reduction"],
                    "same_pass_nanoseconds":stages,"additivity_checked":true,
                    "master_scope":M::MASTER_SCOPE,
                    "residual_scope":M::RESIDUAL_SCOPE,
                    "successful_lanes":args.points.get()-failed_lanes,"failed_lanes":failed_lanes,"failure_kinds":failure_kinds},
                "precision_rescue":{"enabled":false,"attempts":0,"successful":0},
                "representative_source_lane":indices[indices.len()-1],"representative_result":representative,
                "representative_component_precision_bits":representative_bits}));
        }
    }
    Ok(cases)
}

fn report(
    prepared: &PreparedDocument,
    args: &Args,
    engine: EngineReport,
    cases: Vec<Value>,
    success: bool,
    preparation_ns: f64,
) -> Value {
    json!({"schema_version":1,"operation":"bench","engine":args.engine.name(),"success":success,"performance_acceptance":false,
        "acceptance_limitations":["Rust-only measurements; original Ninja no-result-cache baseline is not measured", "shared-master reduction comparison is not measured"],
        "cache_policy":{"compiled_evaluators":"retained and warmed","numerical_master_results":"not cached","reference":"not run"},
        "measurement_scope":engine.measurement_scope,
        "diagnostic_scope":"separate instrumented pass; excluded from primary round timings",
        "precision":engine.precision,
        "build_profile":if cfg!(debug_assertions) {"debug"} else {"release"},"thread_policy":"one reduction worker; no benchmark parallelism",
        "inputs":prepared.document,"source_pool_rows":prepared.samples.len(),"propagators":prepared.propagators,"rank":engine.rank,"cut_stop":engine.cut_stop,
        "numerator_provider":engine.numerator_provider,
        "reduction_kernels":engine.reduction_kernels,
        "requested_cut_stop":args.cut_stop,
        "box_reduction":engine.box_reduction,
        "host":{"os":std::env::consts::OS,"architecture":std::env::consts::ARCH,
            "available_parallelism":std::thread::available_parallelism().ok().map(NonZeroUsize::get)},
        "setup":{"graph_preparation_ns":preparation_ns,"numerical_binding_ns":engine.numerical_binding_ns,"initial_session_ns":engine.initial_session_ns,
            "lazy_compilation_policy":"first uses and capacity growth are included in per-case warmup, outside primary timing"},
        "cases":cases})
}

fn evaluate<T: CoefficientScalar>(
    prepared: &PreparedDocument,
    args: &Args,
    prototype: T,
    preparation_ns: f64,
) -> Result<Value>
where
    Complex<T>: EvaluationDomain,
{
    let timed_points = timed_points(prepared, args)?;
    let binding_start = Instant::now();
    let pool = NumericalPool::new(prepared, &prototype)?;
    let numerical_binding_ns = binding_start.elapsed().as_secs_f64() * 1e9;
    let reduction = args.reduction();
    let session_start = Instant::now();
    let session = EvaluationSession::new(prepared, &reduction, prototype.clone())?;
    let initial_session_ns = session_start.elapsed().as_secs_f64() * 1e9;
    let mut measured = Generic {
        pool,
        session,
        digits: (u64::from(prototype.get_precision()) * 30103 / 100000 + 3) as usize,
    };
    let mut success = true;
    let cases = cases(&mut measured, prepared, args, timed_points, &mut success)?;
    let session = &measured.session;
    let engine = EngineReport {
        precision: json!({"carrier":match args.precision {Precision::F64=>"f64",Precision::DoubleFloat=>"double-float",Precision::Float=>"float"},"bits":prototype.get_precision()}),
        rank: session.rank,
        cut_stop: session.amplitude.cut_stop(),
        numerator_provider: session.numerator_provider_info(),
        reduction_kernels: session.kernel_provider_info(),
        box_reduction: json!({"mu_extraction":match args.mu_extraction {MuExtraction::Expansion=>"expansion",MuExtraction::Sampling=>"sampling"},
            "coefficient_mode":if args.full_box_coefficients {"full"}else{"integrated"},"mu_sampling_scale":args.mu_sampling_scale}),
        measurement_scope: "per-sample numerator coefficient evaluation when selected, numeric parameter rebinding, automatic scales, reset, all cut geometry/callbacks/reduction/masters/integration; no graph preparation, exact numeric parsing, batch packing or JSON",
        numerical_binding_ns,
        initial_session_ns,
    };
    Ok(report(
        prepared,
        args,
        engine,
        cases,
        success,
        preparation_ns,
    ))
}

/// `--engine fast`: the same measurements of the binary64 fast engine.
fn evaluate_fast(prepared: &PreparedDocument, args: &Args, preparation_ns: f64) -> Result<Value> {
    let timed_points = timed_points(prepared, args)?;
    let binding_start = Instant::now();
    let pool = FastPool::new(prepared)?;
    let numerical_binding_ns = binding_start.elapsed().as_secs_f64() * 1e9;
    let session_start = Instant::now();
    let session = FastSession::new(prepared, args.numerator_evaluator, args.cut_stop)?;
    let initial_session_ns = session_start.elapsed().as_secs_f64() * 1e9;
    let mut measured = Fast { pool, session };
    let mut success = true;
    let cases = cases(&mut measured, prepared, args, timed_points, &mut success)?;
    let session = &measured.session;
    let engine = EngineReport {
        precision: json!({"carrier":"f64","bits":f64::MANTISSA_DIGITS}),
        rank: session.rank(),
        cut_stop: session.cut_stop(),
        numerator_provider: session.numerator_provider_info(),
        reduction_kernels: session.kernel_provider_info(),
        box_reduction: json!({"mu_extraction":"expansion","coefficient_mode":"integrated","mu_sampling_scale":null}),
        measurement_scope: "per-sample numerator coefficient evaluation and tensor scatter (or numeric parameter rebinding), reset, all cut geometry/callbacks/reduction/native masters/integration; no graph preparation, exact numeric parsing, batch packing or JSON",
        numerical_binding_ns,
        initial_session_ns,
    };
    Ok(report(
        prepared,
        args,
        engine,
        cases,
        success,
        preparation_ns,
    ))
}

pub(crate) fn run(args: Args) -> Result<()> {
    args.reduction().validate()?;
    if args.batch_sizes.is_empty() {
        return Err("at least one batch size is required".into());
    }
    for (position, width) in args.batch_sizes.iter().enumerate() {
        if args.batch_sizes[..position].contains(width) {
            return Err("batch sizes must be distinct".into());
        }
    }
    let start = Instant::now();
    let prepared = prepare::prepare(&prepare::Args {
        input: args.input.clone(),
        samples: args.samples.clone(),
        model: args.model.clone(),
        feynman_rules: args.feynman_rules,
        r2: args.r2,
        output: None,
        #[cfg(feature = "reference-codegen")]
        reference_dir: None,
        #[cfg(feature = "reference-codegen")]
        reference_bench_dir: None,
    })?;
    let preparation_ns = start.elapsed().as_secs_f64() * 1e9;
    let report = match (args.engine, args.precision) {
        (Engine::Fast, _) => evaluate_fast(&prepared, &args, preparation_ns)?,
        (Engine::Generic, Precision::F64) => evaluate(&prepared, &args, 0_f64, preparation_ns)?,
        (Engine::Generic, Precision::DoubleFloat) => {
            evaluate(&prepared, &args, DoubleFloat::default(), preparation_ns)?
        }
        (Engine::Generic, Precision::Float) => evaluate(
            &prepared,
            &args,
            Float::new(args.bits.unwrap_or(256)),
            preparation_ns,
        )?,
    };
    let output = serde_json::to_string_pretty(&report)? + "\n";
    if let Some(path) = args.output {
        fs::write(path, output)?;
    } else {
        print!("{output}");
    }
    if report["success"] != true {
        return Err(
            "benchmark contains failed numerical lanes; inspect diagnostic statistics".into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod accounting_tests {
    use super::*;

    #[test]
    fn overlapping_stage_intervals_are_rejected() {
        let mut diagnostic = DiagnosticTimings::default();
        diagnostic.masters.elapsed = Duration::from_nanos(7);
        diagnostic.numerator_expansion.elapsed = Duration::from_nanos(4);
        assert!(stage_accounting(Duration::from_nanos(10), &diagnostic).is_err());
    }
}
