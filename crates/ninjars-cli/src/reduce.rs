//! Standalone orchestration over graph preparation and the batched reduction API.

use std::{collections::HashMap, error::Error, fs, num::NonZeroUsize, path::PathBuf};

use clap::{Args as ClapArgs, ValueEnum};
use ninjars::{
    AmplitudeInput, BoxCoefficientMode, BoxMuMode, BoxReductionOptions, Complex, EvaluationStatus,
    FloatLike, FourMomentum, Numerator, SamplingScaleProvider, SamplingScaleSource,
    SamplingScaleStatus,
};
use ninjars_symbolica::{
    PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaHighCuts,
    SymbolicaMasters, SymbolicaSamplingScales,
};
use oneloop::NativeFloat;
use serde_json::{Value, json};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::float::{DoubleFloat, Float},
    evaluate::EvaluationDomain,
    parser::ParseSettings,
    symbol,
};

use crate::prepare::{self, PreparedDocument};

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

fn numerical<T: NativeFloat>(expression: &Atom, prototype: &T, label: &str) -> Result<Complex<T>>
where
    Complex<T>: EvaluationDomain,
{
    let value = expression
        .evaluate_with_prec(
            &HashMap::<Atom, Complex<T>>::new(),
            prototype.get_precision(),
        )
        .map_err(|error| format!("{label} must be fully bound: {error}"))?;
    if !value.re.is_finite() || !value.im.is_finite() {
        return Err(format!("{label} is nonfinite").into());
    }
    Ok(value)
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

fn evaluate<T: NativeFloat>(
    prepared: &mut PreparedDocument,
    args: &Args,
    prototype: T,
) -> Result<bool>
where
    Complex<T>: EvaluationDomain,
{
    let variables =
        ["q0", "q1", "q2", "q3", "mu2"].map(|name| Atom::var(symbol!(format!("ninjars::{name}"))));
    let mut numerator = PolynomialNumerator::new(
        prepared.numerator_template.clone(),
        variables,
        prepared.parameters.clone(),
        prototype.clone(),
    )?;
    let n = prepared.propagators;
    let rank = numerator.rank();
    let box_options = BoxReductionOptions {
        coefficients: if args.full_box_coefficients {
            BoxCoefficientMode::Full
        } else {
            BoxCoefficientMode::Integrated
        },
        mu_extraction: match args.mu_extraction {
            MuExtraction::Expansion => BoxMuMode::Expansion,
            MuExtraction::Sampling => BoxMuMode::Sampling,
        },
    };
    let mut amplitude = SymbolicaAmplitudeKernels::new(n, rank, prototype.clone())?
        .with_box_options(box_options)
        .into_amplitude(0)?;
    amplitude.set_cut_stop(args.cut_stop);
    let boxes_active = n >= 4 && amplitude.cut_stop() <= 4;
    let mut geometry = SymbolicaCuts::new(prototype.clone())?;
    let mut high_geometry = if n >= 4 {
        Some(SymbolicaHighCuts::new(prototype.clone())?)
    } else {
        None
    };
    let mut masters = SymbolicaMasters::new(prototype.clone());
    let scale = numerical(&prepared.scale_squared, &prototype, "scale_squared")?;
    let sampling_scale = args
        .mu_sampling_scale
        .as_ref()
        .map(|expression| -> Result<Complex<T>> {
            let value = numerical(
                &Atom::parse(expression, "ninjars", ParseSettings::default())?,
                &prototype,
                "mu_sampling_scale",
            )?;
            if value.is_fully_zero() {
                return Err("mu_sampling_scale must be nonzero".into());
            }
            Ok(value)
        })
        .transpose()?;
    let mut auto_scales = if sampling_scale.is_none() {
        Some(SymbolicaSamplingScales::new(n, prototype.clone())?)
    } else {
        None
    };
    if !scale.im.is_zero() || scale.re <= prototype.zero() {
        return Err("scale_squared must be positive and real".into());
    }
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
    // Make fallback evaluation independent of chunk boundaries: a missing
    // matrix in a mixed fixture follows the same exact path at every batch size.
    let has_s_matrix = prepared
        .samples
        .iter()
        .any(|sample| sample.s_matrix.is_some());
    for chunk in prepared.samples.chunks(args.batch_size.get()) {
        let mut shifts = Vec::with_capacity(chunk.len() * n);
        let mut masses = Vec::with_capacity(chunk.len() * n);
        let mut parameters = Vec::with_capacity(chunk.len() * prepared.parameters.len());
        let mut s_matrix = has_s_matrix.then(Vec::new);
        for sample in chunk {
            for (index, shift) in sample.shifts.iter().enumerate() {
                let values = shift
                    .components()
                    .into_iter()
                    .enumerate()
                    .map(|(component, atom)| {
                        numerical(
                            atom,
                            &prototype,
                            &format!("sample {} shift {index} component {component}", sample.lane),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                let mut values = values.into_iter();
                shifts.push(FourMomentum::from_args(
                    values.next().unwrap(),
                    values.next().unwrap(),
                    values.next().unwrap(),
                    values.next().unwrap(),
                ));
            }
            masses.extend(
                sample
                    .mass_squared
                    .iter()
                    .enumerate()
                    .map(|(index, atom)| {
                        numerical(
                            atom,
                            &prototype,
                            &format!("sample {} mass_squared {index}", sample.lane),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
            if let Some(matrix) = s_matrix.as_mut() {
                let derived;
                let exact = if let Some(supplied) = &sample.s_matrix {
                    supplied
                } else {
                    // Shared momenta and Symbolica derive omitted matrices, so
                    // mixed supplied/omitted lanes retain one batched call.
                    derived = sample
                        .shifts
                        .iter()
                        .flat_map(|first| {
                            sample.shifts.iter().map(move |second| {
                                let difference = second.clone() - first.clone();
                                difference.dot(&difference)
                            })
                        })
                        .collect::<Vec<_>>();
                    &derived
                };
                matrix.extend(
                    exact
                        .iter()
                        .enumerate()
                        .map(|(index, atom)| {
                            numerical(
                                atom,
                                &prototype,
                                &format!("sample {} s_matrix entry {index}", sample.lane),
                            )
                        })
                        .collect::<Result<Vec<_>>>()?,
                );
            }
            parameters.extend(
                sample
                    .parameters
                    .iter()
                    .enumerate()
                    .map(|(index, atom)| {
                        numerical(
                            atom,
                            &prototype,
                            &format!("sample {} external component {index}", sample.lane),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        numerator.set_parameter_batch(&parameters, chunk.len())?;
        let mut sampling_scales =
            vec![sampling_scale.clone().unwrap_or_else(|| scale.zero()); chunk.len()];
        let mut sampling_statuses =
            vec![SamplingScaleStatus::Selected(SamplingScaleSource::UnitFallback); chunk.len()];
        if let Some(provider) = auto_scales.as_mut() {
            provider.evaluate_batch(
                &shifts,
                &masses,
                chunk.len(),
                &mut sampling_scales,
                &mut sampling_statuses,
            )?;
        }
        amplitude.reset_batch(chunk.len());
        let scales_squared = vec![scale.clone(); chunk.len()];
        let input = AmplitudeInput {
            shifts: &shifts,
            masses_squared: &masses,
            scales_squared: &scales_squared,
            sampling_scales: &sampling_scales,
            s_matrix: s_matrix.as_deref(),
        };
        if let Some(high_geometry) = high_geometry.as_mut() {
            amplitude.evaluate_batch_with_inputs(
                input,
                &mut numerator,
                &mut geometry,
                high_geometry,
                &mut masters,
            )?;
        } else {
            amplitude.evaluate_batch_with_low_cuts(
                input,
                &mut numerator,
                &mut geometry,
                &mut masters,
            )?;
        }
        for (local, sample) in chunk.iter().enumerate() {
            let status = amplitude.statuses()[local];
            let scale_failure = if boxes_active && sampling_scale.is_none() {
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
    prepared.document["cut_stop"] = amplitude.cut_stop().into();
    prepared.document["requested_cut_stop"] = args.cut_stop.into();
    prepared.document["rank"] = rank.into();
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
    let mut prepared = prepare::prepare(&prepare::Args {
        input: args.input.clone(),
        samples: args.samples.clone(),
        model: args.model.clone(),
        feynman_rules: args.feynman_rules,
        output: None,
        #[cfg(feature = "reference-codegen")]
        reference_dir: None,
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
