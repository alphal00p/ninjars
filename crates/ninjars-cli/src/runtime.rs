//! Shared CLI execution state; symbolic preparation and presentation stay outside it.

use std::{
    collections::HashMap,
    error::Error,
    time::{Duration, Instant},
};

use ninjars::{
    Amplitude, AmplitudeInput, BoxCoefficientMode, BoxMuMode, BoxReductionOptions, Complex,
    EvaluationStatus, ExpansionKind, ExpansionRequest, FloatLike, FourMomentum, Laurent,
    MasterProvider, MasterRequest, MasterStatus, Numerator, NumeratorRequest,
    SamplingScaleProvider, SamplingScaleSource, SamplingScaleStatus,
};
use ninjars_symbolica::{
    PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaHighCuts,
    SymbolicaMasters, SymbolicaSamplingScales,
};
use oneloop::NativeFloat;
use symbolica::{
    atom::{Atom, AtomCore},
    evaluate::EvaluationDomain,
    parser::ParseSettings,
    symbol,
};

use crate::{
    numerator_coefficients::{CoefficientBatch, CoefficientScalar},
    prepare::PreparedDocument,
    reduce::{Args, MuExtraction, NumeratorEvaluator, ReductionKernels},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type NumericalAmplitude<T> = Amplitude<Complex<T>, SymbolicaAmplitudeKernels<T>>;

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

/// Flat prepared numerical buffers, using shared FeynKit momenta.
pub(crate) struct NumericalPool<T> {
    n: usize,
    parameter_count: usize,
    lanes: usize,
    shifts: Vec<FourMomentum<Complex<T>>>,
    masses: Vec<Complex<T>>,
    parameters: Vec<Complex<T>>,
    s_matrix: Option<Vec<Complex<T>>>,
    scale: Complex<T>,
}

pub(crate) struct NumericalBatch<T> {
    pub lanes: usize,
    shifts: Vec<FourMomentum<Complex<T>>>,
    masses: Vec<Complex<T>>,
    parameters: Vec<Complex<T>>,
    s_matrix: Option<Vec<Complex<T>>>,
    scales: Vec<Complex<T>>,
    pub sampling_scales: Vec<Complex<T>>,
    pub sampling_statuses: Vec<SamplingScaleStatus>,
}

impl<T> NumericalBatch<T> {
    pub(crate) fn input(&self) -> AmplitudeInput<'_, Complex<T>> {
        AmplitudeInput {
            shifts: &self.shifts,
            masses_squared: &self.masses,
            scales_squared: &self.scales,
            sampling_scales: &self.sampling_scales,
            s_matrix: self.s_matrix.as_deref(),
        }
    }

    #[cfg(feature = "reference-codegen")]
    pub(crate) fn parameters(&self) -> &[Complex<T>] {
        &self.parameters
    }
}

impl<T: NativeFloat> NumericalPool<T>
where
    Complex<T>: EvaluationDomain,
{
    pub(crate) fn new(prepared: &PreparedDocument, prototype: &T) -> Result<Self> {
        let n = prepared.propagators;
        let matrix_entries = n.checked_mul(n).ok_or("S-matrix size overflow")?;
        let scale = numerical(&prepared.scale_squared, prototype, "scale_squared")?;
        if !scale.im.is_zero() || scale.re <= prototype.zero() {
            return Err("scale_squared must be positive and real".into());
        }
        let mut pool = Self {
            n,
            parameter_count: prepared.parameters.len(),
            lanes: prepared.samples.len(),
            shifts: Vec::new(),
            masses: Vec::new(),
            parameters: Vec::new(),
            s_matrix: prepared
                .samples
                .iter()
                .any(|s| s.s_matrix.is_some())
                .then(Vec::new),
            scale,
        };
        for sample in &prepared.samples {
            if sample.shifts.len() != n
                || sample.mass_squared.len() != n
                || sample.parameters.len() != prepared.parameters.len()
                || sample
                    .s_matrix
                    .as_ref()
                    .is_some_and(|matrix| matrix.len() != matrix_entries)
            {
                return Err(format!(
                    "prepared sample {} has an incompatible numerical shape",
                    sample.lane
                )
                .into());
            }
            for (index, shift) in sample.shifts.iter().enumerate() {
                let components = shift
                    .components()
                    .into_iter()
                    .enumerate()
                    .map(|(component, atom)| {
                        numerical(
                            atom,
                            prototype,
                            &format!("sample {} shift {index} component {component}", sample.lane),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                let mut components = components.into_iter();
                pool.shifts.push(FourMomentum::from_args(
                    components.next().unwrap(),
                    components.next().unwrap(),
                    components.next().unwrap(),
                    components.next().unwrap(),
                ));
            }
            pool.masses.extend(
                sample
                    .mass_squared
                    .iter()
                    .enumerate()
                    .map(|(index, atom)| {
                        numerical(
                            atom,
                            prototype,
                            &format!("sample {} mass_squared {index}", sample.lane),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
            pool.parameters.extend(
                sample
                    .parameters
                    .iter()
                    .enumerate()
                    .map(|(index, atom)| {
                        numerical(
                            atom,
                            prototype,
                            &format!("sample {} external component {index}", sample.lane),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
            if let Some(matrix) = pool.s_matrix.as_mut() {
                let derived;
                let exact = if let Some(supplied) = &sample.s_matrix {
                    supplied
                } else {
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
                                prototype,
                                &format!("sample {} s_matrix entry {index}", sample.lane),
                            )
                        })
                        .collect::<Result<Vec<_>>>()?,
                );
            }
        }
        Ok(pool)
    }

    /// Pack selected rows outside steady-state timing; repeated indices are intentional.
    pub(crate) fn batch(&self, indices: &[usize]) -> Result<NumericalBatch<T>> {
        if indices.iter().any(|index| *index >= self.lanes) {
            return Err("sample-pool index is out of bounds".into());
        }
        let capacity = |columns: usize| {
            indices
                .len()
                .checked_mul(columns)
                .ok_or("numerical batch size overflow")
        };
        let mut batch = NumericalBatch {
            lanes: indices.len(),
            shifts: Vec::with_capacity(capacity(self.n)?),
            masses: Vec::with_capacity(capacity(self.n)?),
            parameters: Vec::with_capacity(capacity(self.parameter_count)?),
            s_matrix: self.s_matrix.as_ref().map(|_| Vec::new()),
            scales: vec![self.scale.clone(); indices.len()],
            sampling_scales: vec![self.scale.zero(); indices.len()],
            sampling_statuses: vec![
                SamplingScaleStatus::Selected(SamplingScaleSource::UnitFallback);
                indices.len()
            ],
        };
        for &index in indices {
            batch
                .shifts
                .extend_from_slice(&self.shifts[index * self.n..(index + 1) * self.n]);
            batch
                .masses
                .extend_from_slice(&self.masses[index * self.n..(index + 1) * self.n]);
            batch.parameters.extend_from_slice(
                &self.parameters[index * self.parameter_count..(index + 1) * self.parameter_count],
            );
            if let (Some(source), Some(target)) = (&self.s_matrix, &mut batch.s_matrix) {
                let width = self.n * self.n;
                target.extend_from_slice(&source[index * width..(index + 1) * width]);
            }
        }
        Ok(batch)
    }
}

#[derive(Default)]
pub(crate) struct CallTiming {
    pub elapsed: Duration,
    pub calls: usize,
    pub rows: usize,
}
impl CallTiming {
    fn record(&mut self, start: Instant, rows: usize) {
        self.elapsed += start.elapsed();
        self.calls += 1;
        self.rows += rows;
    }
}

#[derive(Default)]
pub(crate) struct DiagnosticTimings {
    pub numerator_coefficients: CallTiming,
    pub numerator_evaluation: CallTiming,
    pub numerator_expansion: CallTiming,
    pub masters: CallTiming,
}

struct TimedNumerator<'a, N> {
    inner: &'a mut N,
    evaluation: &'a mut CallTiming,
    expansion: &'a mut CallTiming,
}
impl<T, N: Numerator<T>> Numerator<T> for TimedNumerator<'_, N> {
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn evaluate_batch(
        &mut self,
        requests: &[NumeratorRequest<'_, T>],
        output: &mut [T],
    ) -> ninjars::Result<()> {
        let start = Instant::now();
        let result = self.inner.evaluate_batch(requests, output);
        self.evaluation.record(start, requests.len());
        result
    }
    fn expansion_batch(
        &mut self,
        kind: &ExpansionKind,
        requests: &[ExpansionRequest<'_, T>],
        output: &mut [T],
    ) -> ninjars::Result<()> {
        let start = Instant::now();
        let result = self.inner.expansion_batch(kind, requests, output);
        self.expansion.record(start, requests.len());
        result
    }
}

struct TimedMasters<'a, M> {
    inner: &'a mut M,
    timing: &'a mut CallTiming,
}
impl<T, M: MasterProvider<T>> MasterProvider<T> for TimedMasters<'_, M> {
    fn evaluate(
        &mut self,
        request: MasterRequest<'_, T>,
        output: &mut [Laurent<T>],
    ) -> ninjars::Result<()> {
        let rows = request.rows();
        let start = Instant::now();
        let result = self.inner.evaluate(request, output);
        self.timing.record(start, rows);
        result
    }
    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, T>,
        output: &mut [Laurent<T>],
        statuses: &mut [MasterStatus],
    ) -> ninjars::Result<()> {
        let rows = request.rows();
        let start = Instant::now();
        let result = self.inner.evaluate_batch(request, output, statuses);
        self.timing.record(start, rows);
        result
    }
}

pub(crate) struct EvaluationSession<T: NativeFloat>
where
    Complex<T>: EvaluationDomain,
{
    pub amplitude: NumericalAmplitude<T>,
    numerator: PolynomialNumerator<T>,
    coefficients: Option<Box<dyn CoefficientBatch<T>>>,
    numerator_info: serde_json::Value,
    kernel_info: serde_json::Value,
    geometry: SymbolicaCuts<T>,
    high_geometry: Option<SymbolicaHighCuts<T>>,
    masters: SymbolicaMasters<T>,
    auto_scales: Option<SymbolicaSamplingScales<T>>,
    manual_scale: Option<Complex<T>>,
    pub box_options: BoxReductionOptions,
    pub boxes_active: bool,
    pub rank: usize,
}

fn execute<T: NativeFloat, N: Numerator<Complex<T>>, M: MasterProvider<Complex<T>>>(
    amplitude: &mut NumericalAmplitude<T>,
    numerator: &mut N,
    geometry: &mut SymbolicaCuts<T>,
    high_geometry: &mut Option<SymbolicaHighCuts<T>>,
    masters: &mut M,
    batch: &NumericalBatch<T>,
) -> ninjars::Result<()>
where
    Complex<T>: EvaluationDomain,
{
    let input = batch.input();
    if let Some(high_geometry) = high_geometry.as_mut() {
        amplitude.evaluate_batch_with_inputs(input, numerator, geometry, high_geometry, masters)
    } else {
        amplitude.evaluate_batch_with_low_cuts(input, numerator, geometry, masters)
    }
}

impl<T: CoefficientScalar> EvaluationSession<T>
where
    Complex<T>: EvaluationDomain,
{
    pub(crate) fn new(prepared: &PreparedDocument, args: &Args, prototype: T) -> Result<Self> {
        args.reduction_kernels.validate()?;
        let variables = ["q0", "q1", "q2", "q3", "mu2"]
            .map(|name| Atom::var(symbol!(format!("ninjars::{name}"))));
        let (expression, parameters, coefficients, numerator_info) = match args.numerator_evaluator
        {
            NumeratorEvaluator::Symbolica => (
                prepared.numerator_template.clone(),
                prepared.parameters.clone(),
                None,
                serde_json::json!({"evaluator":args.numerator_evaluator.name(), "coefficient_outputs":null,
                        "coefficient_preparation_ns":0, "coefficient_compilation_ns":0,
                        "coefficient_policy":"external coefficients evaluated within existing cut callbacks"}),
            ),
            NumeratorEvaluator::SymjitO2 | NumeratorEvaluator::SymjitO2Factored => {
                let start = Instant::now();
                let plan = ninjars_graph::PolynomialCoefficientPlan::new(
                    prepared.numerator_template.clone(),
                    variables.clone(),
                    prepared.parameters.clone(),
                )?;
                let factoring = matches!(
                    args.numerator_evaluator,
                    NumeratorEvaluator::SymjitO2Factored
                );
                let factor_start = Instant::now();
                let plan = if factoring {
                    plan.factor_coefficients()?
                } else {
                    plan
                };
                let factoring_ns = if factoring {
                    factor_start.elapsed().as_secs_f64() * 1e9
                } else {
                    0.
                };
                let preparation_ns = start.elapsed().as_secs_f64() * 1e9;
                let start = Instant::now();
                let evaluator = T::compile_symjit_coefficients(&plan)?;
                let compilation_ns = start.elapsed().as_secs_f64() * 1e9;
                let info = serde_json::json!({"evaluator":args.numerator_evaluator.name(),
                        "external_inputs":prepared.parameters.len(),
                        "coefficient_outputs":plan.coefficient_expressions().len(),
                        "coefficient_preparation_ns":preparation_ns,
                        "coefficient_factoring":if factoring { "symbolica_exact" } else { "none" },
                        "coefficients_factored":factoring,
                        "coefficient_factoring_ns":factoring_ns,
                        "coefficient_compilation_ns":compilation_ns,
                        "coefficient_error_policy":"reject_binding_batch_before_reduction",
                        "optimization_level":2,
                        "coefficient_policy":"all heads evaluated once per sample in every batch; compiled evaluator retained; coefficient values not cached"});
                (
                    plan.template().clone(),
                    plan.coefficient_symbols().to_vec(),
                    Some(evaluator),
                    info,
                )
            }
        };
        let numerator =
            PolynomialNumerator::new(expression, variables, parameters, prototype.clone())?;
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
        let kernels = SymbolicaAmplitudeKernels::new(n, rank, prototype.clone())?
            .with_box_options(box_options);
        #[cfg(feature = "native-kernels")]
        let kernels = match args.reduction_kernels {
            ReductionKernels::Symbolica => kernels,
            ReductionKernels::NativeDenominators => kernels.with_native_denominators(),
            ReductionKernels::Native => kernels.with_native_kernels(),
        };
        let denominator_backend = match args.reduction_kernels {
            ReductionKernels::Symbolica => "symbolica-interpreter",
            ReductionKernels::NativeDenominators | ReductionKernels::Native => "native-rust",
        };
        let correction_backend = match args.reduction_kernels {
            ReductionKernels::Native => "native-rust",
            _ => "symbolica-interpreter",
        };
        let kernel_info = serde_json::json!({
            "selection": args.reduction_kernels.name(),
            "t2_denominators": denominator_backend,
            "t3_denominators": denominator_backend,
            "bubble_source_projection": correction_backend,
            "triangle_bubble_correction": correction_backend,
            "tadpole_source_projection": correction_backend,
            "other_reduction": "symbolica-interpreter",
            "integration": "symbolica-interpreter",
            "accumulation": "symbolica-interpreter",
            "scope": "Partial generated reduction catalog; numerator callbacks, cut geometry and masters retain their separate providers. Unused cut stages need not execute."
        });
        let mut amplitude = kernels.into_amplitude(0)?;
        amplitude.set_cut_stop(args.cut_stop);
        let manual_scale = args
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
        Ok(Self {
            boxes_active: n >= 4 && amplitude.cut_stop() <= 4,
            amplitude,
            numerator,
            coefficients,
            numerator_info,
            kernel_info,
            geometry: SymbolicaCuts::new(prototype.clone())?,
            high_geometry: if n >= 4 {
                Some(SymbolicaHighCuts::new(prototype.clone())?)
            } else {
                None
            },
            masters: SymbolicaMasters::new(prototype.clone()),
            auto_scales: if manual_scale.is_none() {
                Some(SymbolicaSamplingScales::new(n, prototype)?)
            } else {
                None
            },
            manual_scale,
            box_options,
            rank,
        })
    }

    pub(crate) fn numerator_provider_info(&self) -> serde_json::Value {
        self.numerator_info.clone()
    }

    pub(crate) fn kernel_provider_info(&self) -> serde_json::Value {
        self.kernel_info.clone()
    }

    fn bind_numerator(&mut self, batch: &NumericalBatch<T>) -> Result<()> {
        if let Some(evaluator) = &mut self.coefficients {
            let coefficients = evaluator.evaluate_batch(&batch.parameters, batch.lanes)?;
            self.numerator
                .set_parameter_batch(coefficients, batch.lanes)?;
        } else {
            self.numerator
                .set_parameter_batch(&batch.parameters, batch.lanes)?;
        }
        Ok(())
    }

    fn begin_batch(
        &mut self,
        batch: &mut NumericalBatch<T>,
        timing: Option<&mut CallTiming>,
    ) -> Result<()> {
        if let Some(timing) = timing.filter(|_| self.coefficients.is_some()) {
            let start = Instant::now();
            self.bind_numerator(batch)?;
            timing.record(start, batch.lanes);
        } else {
            self.bind_numerator(batch)?;
        }
        if let Some(scale) = &self.manual_scale {
            batch.sampling_scales.fill(scale.clone());
        } else if let Some(provider) = self.auto_scales.as_mut() {
            provider.evaluate_batch(
                &batch.shifts,
                &batch.masses,
                batch.lanes,
                &mut batch.sampling_scales,
                &mut batch.sampling_statuses,
            )?;
        }
        self.amplitude.reset_batch(batch.lanes);
        Ok(())
    }

    /// Rebind numeric parameters, select scales, reset and reduce without parsing or formatting.
    pub(crate) fn evaluate(&mut self, batch: &mut NumericalBatch<T>) -> Result<()> {
        self.begin_batch(batch, None)?;
        execute(
            &mut self.amplitude,
            &mut self.numerator,
            &mut self.geometry,
            &mut self.high_geometry,
            &mut self.masters,
            batch,
        )?;
        Ok(())
    }

    /// Separate diagnostic path; the primary timing path contains no timer wrappers.
    pub(crate) fn evaluate_diagnostic(
        &mut self,
        batch: &mut NumericalBatch<T>,
        timings: &mut DiagnosticTimings,
    ) -> Result<()> {
        self.begin_batch(batch, Some(&mut timings.numerator_coefficients))?;
        let mut numerator = TimedNumerator {
            inner: &mut self.numerator,
            evaluation: &mut timings.numerator_evaluation,
            expansion: &mut timings.numerator_expansion,
        };
        let mut masters = TimedMasters {
            inner: &mut self.masters,
            timing: &mut timings.masters,
        };
        execute(
            &mut self.amplitude,
            &mut numerator,
            &mut self.geometry,
            &mut self.high_geometry,
            &mut masters,
            batch,
        )?;
        Ok(())
    }

    pub(crate) fn scale_failure(
        &self,
        batch: &NumericalBatch<T>,
        lane: usize,
    ) -> Option<ninjars::CutFailure> {
        if self.boxes_active
            && self.manual_scale.is_none()
            && let SamplingScaleStatus::InvalidInput(failure) = batch.sampling_statuses[lane]
        {
            return Some(failure);
        }
        None
    }

    pub(crate) fn successful(&self, batch: &NumericalBatch<T>, lane: usize) -> bool {
        self.amplitude.statuses()[lane] == EvaluationStatus::Success
            && self.scale_failure(batch, lane).is_none()
    }
}
