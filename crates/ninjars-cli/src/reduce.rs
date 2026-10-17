//! Standalone orchestration over graph preparation and the batched reduction API.

use crate::{
    numerator_coefficients::CoefficientScalar,
    prepare::{self, PreparedDocument},
    runtime::{EvaluationSession, NumericalPool},
};
use clap::{Args as ClapArgs, ValueEnum};
use ninjars::{Complex, EvaluationStatus, SamplingScaleSource, SamplingScaleStatus};
use serde_json::{Value, json};
use std::{error::Error, fs, num::NonZeroUsize, path::PathBuf};
use symbolica::{
    atom::AtomCore,
    domains::float::{DoubleFloat, Float},
    evaluate::EvaluationDomain,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

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

    pub(crate) fn validate(self) -> Result<()> {
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
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long)]
    pub feynman_rules: bool,
    /// Numerical carrier; all input expressions are evaluated directly at this precision.
    #[arg(long, value_enum, default_value = "f64")]
    pub precision: Precision,
    /// Evaluate the physical numerator directly, or compile its coefficient heads once with SymJIT O2.
    #[arg(long, value_enum, default_value = "symbolica")]
    pub numerator_evaluator: NumeratorEvaluator,
    /// Reduction execution, independently of the numerator preparation provider.
    #[arg(long, value_enum, default_value = "symbolica")]
    pub reduction_kernels: ReductionKernels,
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
    #[arg(short, long)]
    pub output: Option<PathBuf>,
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
    }
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
            if args.diagnostics {
                row["cut_counts"] = json!({"pentagons":amplitude.pentagons().len()/chunk.len(),"boxes":amplitude.boxes().len()/chunk.len(),"triangles":amplitude.triangles().len()/chunk.len(),"bubbles":amplitude.bubbles().len()/chunk.len(),"tadpoles":amplitude.tadpoles().len()/chunk.len()});
            }
            rows.push(row);
        }
    }
    prepared.document["operation"] = "reduce".into();
    prepared.document["precision"] = json!({"carrier":match args.precision {Precision::F64=>"f64",Precision::DoubleFloat=>"double-float",Precision::Float=>"float"},"bits":prototype.get_precision()});
    prepared.document["batch_size"] = args.batch_size.get().into();
    prepared.document["cut_stop"] = session.amplitude.cut_stop().into();
    prepared.document["requested_cut_stop"] = args.cut_stop.into();
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
    Ok(success)
}

pub(crate) fn evaluate_prepared(prepared: &mut PreparedDocument, args: &Args) -> Result<bool> {
    args.reduction_kernels.validate()?;
    args.numerator_evaluator.validate(args.precision)?;
    if args.bits.is_some() && !matches!(args.precision, Precision::Float) {
        return Err("--bits applies only to --precision float".into());
    }
    if args.bits.is_some_and(|bits| bits < 2) {
        return Err("--bits must be at least 2".into());
    }
    match args.precision {
        Precision::F64 => evaluate(prepared, args, 0_f64),
        Precision::DoubleFloat => evaluate(prepared, args, DoubleFloat::default()),
        Precision::Float => evaluate(prepared, args, Float::new(args.bits.unwrap_or(256))),
    }
}

pub fn run(args: Args) -> Result<()> {
    args.reduction_kernels.validate()?;
    args.numerator_evaluator.validate(args.precision)?;
    let mut prepared = prepare::prepare(&prepare::Args {
        input: args.input.clone(),
        samples: args.samples.clone(),
        model: args.model.clone(),
        feynman_rules: args.feynman_rules,
        output: None,
        #[cfg(feature = "reference-codegen")]
        reference_dir: None,
        #[cfg(feature = "reference-codegen")]
        reference_bench_dir: None,
    })?;
    let success = evaluate_prepared(&mut prepared, &args)?;
    let output = serde_json::to_string_pretty(&prepared.document)? + "\n";
    if let Some(path) = args.output {
        fs::write(path, output)?;
    } else {
        print!("{output}");
    }
    if !success {
        return Err(
            "one or more samples failed reduction; inspect per-lane statuses in the JSON output"
                .into(),
        );
    }
    Ok(())
}
