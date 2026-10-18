//! Standalone orchestration over graph preparation and the batched reduction API.

use crate::{
    fast_runtime::{self, FastPool, FastSession},
    numerator_coefficients::CoefficientScalar,
    prepare::{self, PreparedDocument, PreparedSample},
    runtime::{EvaluationSession, NumericalPool},
};
use clap::{Args as ClapArgs, ValueEnum};
use ninjars::{
    Complex, EvaluationStatus, SamplingScaleSource, SamplingScaleStatus, fast::FastStatus,
};
use serde_json::{Value, json};
use std::{error::Error, fs, num::NonZeroUsize, path::PathBuf};
use symbolica::{
    atom::AtomCore,
    domains::float::{DoubleFloat, Float, RealLike},
    evaluate::EvaluationDomain,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Engine {
    /// Symbolica-generic reduction in every precision, with every box mode.
    Generic,
    /// Binary64 port of original Ninja (`ninjars::fast`) with oneloopmaster's native masters.
    Fast,
}

impl Engine {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Generic => "generic",
            Self::Fast => "fast",
        }
    }
}

/// Options shared by `reduce` and `bench` that select the reduction engine.
pub(crate) struct EngineOptions<'a> {
    pub engine: Engine,
    pub precision: Precision,
    pub bits: Option<u32>,
    pub reduction_kernels: ReductionKernels,
    pub native_block_rows: Option<NonZeroUsize>,
    pub mu_extraction: MuExtraction,
    pub full_box_coefficients: bool,
    pub mu_sampling_scale: Option<&'a str>,
}

impl EngineOptions<'_> {
    /// Reject generic-engine options under `--engine fast` before any preparation.
    pub(crate) fn validate(&self) -> Result<()> {
        if self.engine == Engine::Generic {
            return Ok(());
        }
        let generic_only = |option: String| -> Result<()> {
            Err(format!("--engine fast does not support {option}; use --engine generic").into())
        };
        if !matches!(self.precision, Precision::F64) || self.bits.is_some() {
            return Err("--engine fast reduces in binary64 and requires --precision f64 (without --bits); use --engine generic for double-float or float".into());
        }
        if matches!(self.mu_extraction, MuExtraction::Sampling) {
            return generic_only(
                "--mu-extraction sampling (it uses original Ninja's box mu-expansion)".into(),
            );
        }
        if self.full_box_coefficients {
            return generic_only("--full-box-coefficients".into());
        }
        if self.mu_sampling_scale.is_some() {
            return generic_only("--mu-sampling-scale (it never samples mu^2)".into());
        }
        if !matches!(self.reduction_kernels, ReductionKernels::Symbolica) {
            return generic_only(format!(
                "--reduction-kernels {} (generic-engine kernels)",
                self.reduction_kernels.name()
            ));
        }
        if self.native_block_rows.is_some() {
            return generic_only("--native-block-rows (generic-engine kernels)".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, ValueEnum)]
pub enum Precision {
    F64,
    #[value(alias = "dd")]
    DoubleFloat,
    Float,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum MuExtraction {
    Expansion,
    Sampling,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum ReductionKernels {
    Symbolica,
    /// Execute generated Rust for T2/T3 denominator expansions only.
    NativeDenominators,
    /// Execute the available partial native catalog, including bubble corrections.
    Native,
    /// Guarded n=5/r=4 corrections with interpreted generic kernels and alternates.
    InterpretedRank,
    /// Guarded n=5/r=4 native alternates with the generic native catalog as fallback.
    NativeRank,
}

impl ReductionKernels {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Symbolica => "symbolica",
            Self::NativeDenominators => "native-denominators",
            Self::Native => "native",
            Self::InterpretedRank => "interpreted-rank",
            Self::NativeRank => "native-rank",
        }
    }

    pub(crate) fn validate(self, native_block_rows: Option<NonZeroUsize>) -> Result<()> {
        if matches!(
            self,
            Self::NativeDenominators | Self::Native | Self::NativeRank
        ) && !cfg!(feature = "native-kernels")
        {
            return Err(format!(
                "--reduction-kernels {} requires a build with --features native-kernels",
                self.name()
            )
            .into());
        }
        if native_block_rows.is_some() && matches!(self, Self::Symbolica | Self::InterpretedRank) {
            return Err("--native-block-rows requires a native reduction-kernel selection".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, ValueEnum)]
pub enum NumeratorEvaluator {
    Symbolica,
    SymjitO2,
    /// Factor external coefficients exactly with Symbolica before SymJIT O2.
    SymjitO2Factored,
}

impl NumeratorEvaluator {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Symbolica => "symbolica",
            Self::SymjitO2 => "symjit-o2",
            Self::SymjitO2Factored => "symjit-o2-factored",
        }
    }

    pub(crate) fn validate(self, precision: Precision) -> Result<()> {
        if matches!(self, Self::SymjitO2 | Self::SymjitO2Factored) {
            if !matches!(precision, Precision::F64) {
                return Err(format!("--numerator-evaluator {} requires --precision f64; use symbolica to preserve higher precision", self.name()).into());
            }
            if !cfg!(feature = "numerator-jit") {
                return Err(format!(
                    "--numerator-evaluator {} requires a build with --features numerator-jit",
                    self.name()
                )
                .into());
            }
        }
        Ok(())
    }
}

#[derive(ClapArgs)]
pub struct Args {
    /// DOT graph with --samples, or a schema-version-1 JSON fixture.
    pub input: PathBuf,
    #[arg(long)]
    pub samples: Option<PathBuf>,
    #[command(flatten)]
    pub graph: prepare::GraphOptions,
    /// Reduction engine. `fast` requires --precision f64 and the default box
    /// mu-expansion; it uses Ninja's tensor numerator (with explicit mu^2
    /// blocks when the numerator has mu^2 terms, e.g. --r2 fdh) and
    /// oneloopmaster's native binary64 masters.
    #[arg(long, value_enum, default_value = "generic")]
    pub engine: Engine,
    /// Numerical carrier; all input expressions are evaluated directly at this precision.
    #[arg(long, value_enum, default_value = "f64")]
    pub precision: Precision,
    /// Evaluate the physical numerator directly, or compile its coefficient heads once with SymJIT O2.
    #[arg(long, value_enum, default_value = "symbolica")]
    pub numerator_evaluator: NumeratorEvaluator,
    /// Reduction execution, independently of the numerator preparation provider.
    #[arg(long, value_enum, default_value = "symbolica")]
    pub reduction_kernels: ReductionKernels,
    /// Bound complete native kernel rows per call, independently of phase-space batch size.
    /// Omission keeps whole-batch execution; requires a native reduction selection.
    #[arg(long)]
    pub native_block_rows: Option<NonZeroUsize>,
    /// Working precision in bits for --precision float (default 256).
    #[arg(long)]
    pub bits: Option<u32>,
    /// Number of phase-space samples passed together through the entire reduction.
    #[arg(long, default_value = "32")]
    pub batch_size: NonZeroUsize,
    /// Stop after this cut size, assuming lower cuts contribute zero; 0/1 keep all cuts.
    #[arg(long, default_value = "1")]
    pub cut_stop: usize,
    /// Include cut counts and the common parameterized numerator.
    #[arg(long)]
    pub diagnostics: bool,
    /// Box rational extraction through the mu callback or finite-mu samples.
    #[arg(long, value_enum, default_value = "expansion")]
    pub mu_extraction: MuExtraction,
    /// Extract all box residue coefficients for reconstruction diagnostics.
    #[arg(long)]
    pub full_box_coefficients: bool,
    /// Override the automatic mu² sampling scale with an exact nonzero expression.
    /// This auxiliary scale is independent of the squared renormalization scale.
    #[arg(long)]
    pub mu_sampling_scale: Option<String>,
    /// Report a lane as `unstable` when cancellation between its cut contributions
    /// leaves fewer than half of the carrier's significant bits (generic engine).
    /// Exactly vanishing amplitudes (Ward identities, zero helicity amplitudes)
    /// fail it in every precision.
    #[arg(long)]
    pub stability_test: bool,
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

impl Args {
    pub(crate) fn engine_options(&self) -> EngineOptions<'_> {
        EngineOptions {
            engine: self.engine,
            precision: self.precision,
            bits: self.bits,
            reduction_kernels: self.reduction_kernels,
            native_block_rows: self.native_block_rows,
            mu_extraction: self.mu_extraction,
            full_box_coefficients: self.full_box_coefficients,
            mu_sampling_scale: self.mu_sampling_scale.as_deref(),
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        self.engine_options().validate()?;
        self.reduction_kernels.validate(self.native_block_rows)?;
        self.numerator_evaluator.validate(self.precision)?;
        if self.bits.is_some() && !matches!(self.precision, Precision::Float) {
            return Err("--bits applies only to --precision float".into());
        }
        if self.bits.is_some_and(|bits| bits < 2) {
            return Err("--bits must be at least 2".into());
        }
        Ok(())
    }
}

/// Document fields common to both engines.
fn finish_document(prepared: &mut PreparedDocument, args: &Args, rows: Vec<Value>, success: bool) {
    prepared.document["operation"] = "reduce".into();
    prepared.document["engine"] = args.engine.name().into();
    prepared.document["batch_size"] = args.batch_size.get().into();
    prepared.document["requested_cut_stop"] = args.cut_stop.into();
    prepared.document["success"] = success.into();
    prepared.document["result_order"] = json!([
        "finite",
        "single_pole",
        "double_pole",
        "cut_constructible_finite",
        "rational"
    ]);
    prepared.document["number_format"] = "complex_decimal_strings_re_im".into();
    prepared.document["results"] = rows.into();
    if args.diagnostics {
        prepared.document["numerator_template"] =
            prepared.numerator_template.to_canonical_string().into();
        prepared.document["numerator_parameters"] = json!(
            prepared
                .parameters
                .iter()
                .map(AtomCore::to_canonical_string)
                .collect::<Vec<_>>()
        );
    }
}

fn status_json(status: EvaluationStatus) -> Value {
    match status {
        EvaluationStatus::Success => json!({"kind":"success"}),
        EvaluationStatus::NotEvaluated => json!({"kind":"not_evaluated"}),
        EvaluationStatus::CutFailed { cut_index, failure } => {
            json!({"kind":"cut_failed","cut_index":cut_index,"reason":format!("{failure:?}")})
        }
        EvaluationStatus::IntegrationFailed { cut_index, status } => {
            json!({"kind":"integration_failed","cut_index":cut_index,"reason":format!("{status:?}")})
        }
        EvaluationStatus::NonFiniteResult => json!({"kind":"nonfinite_result"}),
        EvaluationStatus::RepeatedPropagator { first, second } => {
            json!({"kind":"repeated_propagator","propagators":[first,second],"reason":"raised or repeated loop propagators are not supported"})
        }
        EvaluationStatus::Unstable => json!({"kind":"unstable"}),
    }
}

/// Cancellation between one lane's cut contributions, in binary64: the moduli
/// `|Re| + |Im|` of the contributions summed per Laurent coefficient
/// (`Amplitude::contribution_moduli`), their largest over the result's
/// largest modulus (null for a vanishing result), and the rounding error of
/// that sum alone, `2^-p` times the moduli (`p` mantissa bits, also reported
/// as `precision_bits`: the binary64 estimate underflows to zero once
/// `2^-p` times a modulus leaves the binary64 range, e.g. above 1074 bits).
fn cancellation_json<T: RealLike>(
    contributions: &ninjars::Laurent<Complex<T>>,
    result: &ninjars::Laurent<Complex<T>>,
    bits: u32,
) -> Value {
    let contributions = contributions.each_ref().map(|modulus| modulus.re.to_f64());
    let largest = |values: [f64; 3]| values.into_iter().fold(0., f64::max);
    let result = largest(result.each_ref().map(|modulus| modulus.re.to_f64()));
    let epsilon = (-f64::from(bits)).exp2();
    json!({
        "contribution_moduli": contributions,
        "ratio": (result > 0.).then(|| largest(contributions) / result),
        "rounding_estimate": contributions.map(|modulus| modulus * epsilon),
        "precision_bits": bits,
    })
}

fn evaluate<T: CoefficientScalar>(
    prepared: &mut PreparedDocument,
    args: &Args,
    prototype: T,
) -> Result<bool>
where
    Complex<T>: EvaluationDomain,
{
    let pool = NumericalPool::new(prepared, &prototype)?;
    let mut session = EvaluationSession::new(prepared, args, prototype.clone())?;
    let n = prepared.propagators;
    let rank = session.rank;
    let box_options = session.box_options;
    let boxes_active = session.boxes_active;
    // Display enough digits for a round trip, including DoubleFloat's low word.
    let digits = (u64::from(prototype.get_precision()) * 30103 / 100000 + 3) as usize;
    let encode = |value: &Complex<T>| {
        [
            format!("{:.*e}", digits, &value.re),
            format!("{:.*e}", digits, &value.im),
        ]
    };
    let mut rows = Vec::new();
    let mut success = true;
    for (chunk_index, chunk) in prepared.samples.chunks(args.batch_size.get()).enumerate() {
        let first = chunk_index * args.batch_size.get();
        let indices = (first..first + chunk.len()).collect::<Vec<_>>();
        let mut batch = pool.batch(&indices)?;
        session.evaluate(&mut batch)?;
        let amplitude = &session.amplitude;
        let sampling_scales = &batch.sampling_scales;
        let sampling_statuses = &batch.sampling_statuses;
        for (local, sample) in chunk.iter().enumerate() {
            let status = amplitude.statuses()[local];
            let scale_failure = if boxes_active && args.mu_sampling_scale.is_none() {
                match sampling_statuses[local] {
                    SamplingScaleStatus::InvalidInput(failure) => Some(failure),
                    _ => None,
                }
            } else {
                None
            };
            let valid = status == EvaluationStatus::Success && scale_failure.is_none();
            success &= valid;
            let mut row = json!({"lane":sample.lane,"status":if let Some(failure)=scale_failure {json!({"kind":"sampling_scale_failed","reason":format!("{failure:?}")})} else {status_json(status)},"values":if valid {Some(amplitude.results()[local].each_ref().map(encode))} else {None}});
            row["s_matrix_source"] = if sample.s_matrix.is_some() {
                "supplied"
            } else {
                "momenta"
            }
            .into();
            row["mu_sampling_scale"] = if let Some(expression) = &args.mu_sampling_scale {
                json!({"mode":"manual","expression":expression,"value":encode(&sampling_scales[local])})
            } else {
                match sampling_statuses[local] {
                    SamplingScaleStatus::Selected(source) => {
                        let source = match source {
                            SamplingScaleSource::Mass { propagator } => {
                                json!({"kind":"mass_squared","propagator":propagator})
                            }
                            SamplingScaleSource::Invariant { first, second } => {
                                json!({"kind":"invariant","propagators":[first,second]})
                            }
                            SamplingScaleSource::UnitFallback => json!({"kind":"unit_fallback"}),
                        };
                        json!({"mode":"automatic","value":encode(&sampling_scales[local]),"source":source})
                    }
                    SamplingScaleStatus::InvalidInput(failure) if boxes_active => {
                        json!({"mode":"automatic","value":null,"reason":format!("{failure:?}")})
                    }
                    SamplingScaleStatus::InvalidInput(_) => {
                        json!({"mode":"automatic","value":null})
                    }
                }
            };
            row["mu_sampling_scale"]["used"] =
                json!(boxes_active && box_options.sample_count(n, rank) > 1);
            // Float tracks significant bits separately after cancellation. Report
            // these unchanged; nominal working bits do not promise result bits.
            row["component_precision_bits"] = if valid {
                json!(
                    amplitude.results()[local]
                        .each_ref()
                        .map(|value| [value.re.get_precision(), value.im.get_precision()])
                )
            } else {
                Value::Null
            };
            // Measured whether or not --stability-test is set.
            row["cancellation"] = match amplitude.result_moduli(local) {
                Some(result) if valid => cancellation_json(
                    &amplitude.contribution_moduli()[local],
                    &result,
                    prototype.get_precision(),
                ),
                _ => Value::Null,
            };
            if args.diagnostics {
                row["cut_counts"] = json!({"pentagons":amplitude.pentagons().len()/chunk.len(),"boxes":amplitude.boxes().len()/chunk.len(),"triangles":amplitude.triangles().len()/chunk.len(),"bubbles":amplitude.bubbles().len()/chunk.len(),"tadpoles":amplitude.tadpoles().len()/chunk.len()});
            }
            rows.push(row);
        }
    }
    prepared.document["precision"] = json!({"carrier":match args.precision {Precision::F64=>"f64",Precision::DoubleFloat=>"double-float",Precision::Float=>"float"},"bits":prototype.get_precision()});
    prepared.document["cut_stop"] = session.amplitude.cut_stop().into();
    prepared.document["rank"] = rank.into();
    prepared.document["numerator_provider"] = session.numerator_provider_info();
    prepared.document["reduction_kernels"] = session.kernel_provider_info();
    prepared.document["box_reduction"] = json!({"mu_extraction":match args.mu_extraction {MuExtraction::Expansion=>"expansion",MuExtraction::Sampling=>"sampling"},"coefficient_mode":if args.full_box_coefficients {"full"}else{"integrated"},"mu_sampling_scale":args.mu_sampling_scale});
    prepared.document["box_reduction"]["mu_sampling_scale_mode"] =
        if args.mu_sampling_scale.is_some() {
            "manual"
        } else {
            "automatic"
        }
        .into();
    finish_document(prepared, args, rows, success);
    Ok(success)
}

/// `--engine fast`: the same result schema from the binary64 fast engine.
fn evaluate_fast(prepared: &mut PreparedDocument, args: &Args) -> Result<bool> {
    let pool = FastPool::new(prepared)?;
    let batches = (0..prepared.samples.len())
        .collect::<Vec<_>>()
        .chunks(args.batch_size.get())
        .map(|indices| pool.batch(indices))
        .collect::<Result<Vec<_>>>()?;
    let mut session = FastSession::new(prepared, args.numerator_evaluator, args.cut_stop)?;
    let mut rows = Vec::new();
    let mut success = true;
    let mut lanes = prepared.samples.iter();
    for batch in &batches {
        session.evaluate(batch)?;
        for local in 0..batch.lanes {
            let sample = lanes
                .next()
                .ok_or("fast batch exceeds the prepared samples")?;
            // The fast engine has no repeated-propagator check; report such lanes
            // exactly as the generic engine does.
            let repeated = repeated_propagators(sample);
            let status = session.status(local);
            let valid = status == FastStatus::Success && repeated.is_none();
            success &= valid;
            let values = valid.then(|| session.components(local).map(fast_runtime::encode));
            let status = match repeated {
                Some((first, second)) => {
                    status_json(EvaluationStatus::RepeatedPropagator { first, second })
                }
                None => fast_runtime::status_json(status),
            };
            let mut row = json!({"lane":sample.lane,"status":status,"values":values});
            row["s_matrix_source"] = if sample.s_matrix.is_some() {
                "supplied"
            } else {
                "momenta"
            }
            .into();
            row["mu_sampling_scale"] = json!({"mode":"none","value":null,"used":false});
            row["component_precision_bits"] = if valid {
                json!(fast_runtime::COMPONENT_BITS)
            } else {
                Value::Null
            };
            if args.diagnostics {
                // The fast engine applies Ninja's per-lane skip rules inside each
                // stage and does not count the cuts it evaluates.
                row["cut_counts"] = Value::Null;
            }
            rows.push(row);
        }
    }
    prepared.document["precision"] = json!({"carrier":"f64","bits":f64::MANTISSA_DIGITS});
    prepared.document["cut_stop"] = session.cut_stop().into();
    prepared.document["rank"] = session.rank().into();
    prepared.document["numerator_provider"] = session.numerator_provider_info();
    prepared.document["reduction_kernels"] = session.kernel_provider_info();
    prepared.document["box_reduction"] = json!({"mu_extraction":"expansion","coefficient_mode":"integrated","mu_sampling_scale":null,"mu_sampling_scale_mode":"none"});
    finish_document(prepared, args, rows, success);
    Ok(success)
}

pub(crate) fn evaluate_prepared(prepared: &mut PreparedDocument, args: &Args) -> Result<bool> {
    args.validate()?;
    if prepared.samples.is_empty() {
        return Err(
            "reduction needs at least one phase-space sample; pass a JSON fixture \
                    with samples, or a DOT graph with --samples <fixture>"
                .into(),
        );
    }
    match (args.engine, args.precision) {
        (Engine::Fast, _) => evaluate_fast(prepared, args),
        (Engine::Generic, Precision::F64) => evaluate(prepared, args, 0_f64),
        (Engine::Generic, Precision::DoubleFloat) => {
            evaluate(prepared, args, DoubleFloat::default())
        }
        (Engine::Generic, Precision::Float) => {
            evaluate(prepared, args, Float::new(args.bits.unwrap_or(256)))
        }
    }
}

pub fn run(args: Args) -> Result<()> {
    args.validate()?;
    let mut prepared = prepare::prepare(&prepare::Args::for_input(
        args.input.clone(),
        args.samples.clone(),
        args.graph.clone(),
    ))?;
    let success = evaluate_prepared(&mut prepared, &args)?;
    let output = serde_json::to_string_pretty(&prepared.document)? + "\n";
    if let Some(path) = args.output {
        fs::write(path, output)?;
    } else {
        print!("{output}");
    }
    if !success {
        let failures = prepared.document["results"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|row| row["status"]["kind"] != "success")
            .map(|row| match row["status"]["reason"].as_str() {
                Some(reason) => format!(
                    "lane {}: {} ({reason})",
                    row["lane"],
                    row["status"]["kind"].as_str().unwrap_or("failed")
                ),
                None => format!(
                    "lane {}: {}",
                    row["lane"],
                    row["status"]["kind"].as_str().unwrap_or("failed")
                ),
            })
            .collect::<Vec<_>>();
        let shown = failures.len().min(3);
        let more = if failures.len() > shown {
            format!(" and {} more", failures.len() - shown)
        } else {
            String::new()
        };
        return Err(format!(
            "one or more samples failed reduction ({}{more}); inspect per-lane statuses in the JSON output",
            failures[..shown].join("; ")
        )
        .into());
    }
    Ok(())
}

/// The first pair of exactly coincident propagators (equal shifts and squared
/// masses) of a prepared sample.
fn repeated_propagators(sample: &PreparedSample) -> Option<(usize, usize)> {
    let n = sample.shifts.len();
    let equal = |a: &symbolica::atom::Atom, b: &symbolica::atom::Atom| (a - b).together().is_zero();
    (0..n)
        .flat_map(|first| (first + 1..n).map(move |second| (first, second)))
        .find(|&(first, second)| {
            equal(&sample.mass_squared[first], &sample.mass_squared[second])
                && sample.shifts[first]
                    .components()
                    .into_iter()
                    .zip(sample.shifts[second].components())
                    .all(|(a, b)| equal(a, b))
        })
}
