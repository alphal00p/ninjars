//! `--engine fast`: the binary64 `ninjars::fast` engine behind `reduce` and `bench`.
//!
//! Inputs are bound exactly as for the generic engine (`NumericalPool<f64>`,
//! including the exact S-matrix rule) and repacked once into the fast
//! engine's real-momentum layout; complex external kinematics are rejected.
//! The numerator is original Ninja's dense `TensorNumerator`, filled per batch
//! from the coefficient heads of a `TensorCoefficientPlan` (Symbolica's
//! interpreter or SymJIT O2, following `--numerator-evaluator`). A prepared
//! numerator with explicit `mu^2` terms has no tensor layout; it runs as the
//! existing `PolynomialNumerator<f64>` behind `GenericNumerator`. Masters are
//! oneloopmaster's native binary64 code (`OneLoopNativeMasters`).

use std::{error::Error, time::Instant};

use ninjars::fast::{
    C64, CMom, FastAmplitude, FastInput, FastMaster, FastMasters, FastNumerator, FastOptions,
    FastResult, FastStatus, GenericNumerator, RMom, TensorNumerator,
};
use ninjars_graph::{CoefficientError, PolynomialCoefficientPlan, TensorCoefficientPlan};
use ninjars_symbolica::{OneLoopNativeMasters, PolynomialNumerator};
use serde_json::{Value, json};
use symbolica::domains::float::Complex;

use crate::{
    numerator_coefficients::{CoefficientBatch, CoefficientScalar, InterpretedCoefficients},
    prepare::PreparedDocument,
    reduce::NumeratorEvaluator,
    runtime::{CallTiming, DiagnosticTimings, NumericalPool, loop_variables},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Decimal digits of the generic engine's binary64 output (`53 * log10(2) + 3`).
const DIGITS: usize = 18;

/// Working and result precision of every binary64 component.
pub(crate) const COMPONENT_BITS: [[u32; 2]; 5] = [[f64::MANTISSA_DIGITS; 2]; 5];

/// Lane-major binary64 inputs of every prepared sample.
pub(crate) struct FastPool {
    n: usize,
    lanes: usize,
    parameter_count: usize,
    shifts: Vec<RMom>,
    masses: Vec<C64>,
    parameters: Vec<Complex<f64>>,
    s_matrix: Option<Vec<f64>>,
    scale: f64,
}

/// One packed batch of [`FastPool`] rows.
pub(crate) struct FastBatch {
    pub lanes: usize,
    shifts: Vec<RMom>,
    masses: Vec<C64>,
    scales: Vec<f64>,
    s_matrix: Option<Vec<f64>>,
    parameters: Vec<Complex<f64>>,
}

impl FastBatch {
    fn input(&self) -> FastInput<'_> {
        FastInput {
            lanes: self.lanes,
            shifts: &self.shifts,
            masses_squared: &self.masses,
            scales_squared: &self.scales,
            s_matrix: self.s_matrix.as_deref(),
        }
    }
}

fn real(value: &Complex<f64>, label: impl FnOnce() -> String) -> Result<f64> {
    if value.im != 0. {
        return Err(format!(
            "--engine fast requires real external kinematics, but {} = {}{:+}i; use --engine generic",
            label(),
            value.re,
            value.im
        )
        .into());
    }
    Ok(value.re)
}

impl FastPool {
    pub(crate) fn new(prepared: &PreparedDocument) -> Result<Self> {
        let pool = NumericalPool::new(prepared, &0_f64)?;
        let n = pool.n;
        let lane = |row: usize| prepared.samples[row].lane;
        let shifts = pool
            .shifts
            .iter()
            .enumerate()
            .map(|(index, shift)| {
                let components = shift.components();
                let mut momentum = [0.; 4];
                for (component, (target, value)) in momentum.iter_mut().zip(components).enumerate()
                {
                    *target = real(value, || {
                        format!(
                            "sample {} shift {} component {component}",
                            lane(index / n),
                            index % n
                        )
                    })?;
                }
                Ok(momentum)
            })
            .collect::<Result<Vec<_>>>()?;
        let s_matrix = pool
            .s_matrix
            .as_ref()
            .map(|matrix| {
                matrix
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        real(value, || {
                            format!(
                                "sample {} s_matrix entry {}",
                                lane(index / (n * n)),
                                index % (n * n)
                            )
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        Ok(Self {
            n,
            lanes: pool.lanes,
            parameter_count: pool.parameter_count,
            shifts,
            masses: pool
                .masses
                .iter()
                .copied()
                .map(C64::from_symbolica)
                .collect(),
            parameters: pool.parameters,
            s_matrix,
            // NumericalPool checked that the squared scale is positive and real.
            scale: pool.scale.re,
        })
    }

    /// Pack selected rows outside steady-state timing; repeated indices are intentional.
    pub(crate) fn batch(&self, indices: &[usize]) -> Result<FastBatch> {
        if indices.iter().any(|index| *index >= self.lanes) {
            return Err("sample-pool index is out of bounds".into());
        }
        let (n, width) = (self.n, self.parameter_count);
        let rows = |stride: usize| indices.iter().map(move |&i| i * stride..(i + 1) * stride);
        Ok(FastBatch {
            lanes: indices.len(),
            shifts: rows(n)
                .flat_map(|r| self.shifts[r].iter().copied())
                .collect(),
            masses: rows(n)
                .flat_map(|r| self.masses[r].iter().copied())
                .collect(),
            scales: vec![self.scale; indices.len()],
            s_matrix: self.s_matrix.as_ref().map(|matrix| {
                rows(n * n)
                    .flat_map(|r| matrix[r].iter().copied())
                    .collect()
            }),
            parameters: rows(width)
                .flat_map(|r| self.parameters[r].iter().copied())
                .collect(),
        })
    }
}

/// The numerator provider and its per-batch coefficient binding.
enum Binding {
    /// Ninja's dense tensor numerator from evaluated coefficient heads.
    Tensor {
        plan: TensorCoefficientPlan,
        heads: Box<dyn CoefficientBatch<f64>>,
        values: Vec<C64>,
        tensor: TensorNumerator,
    },
    /// Explicit mu^2 dependence: the generic polynomial numerator.
    Polynomial {
        numerator: Box<GenericNumerator<PolynomialNumerator<f64>>>,
        coefficients: Option<Box<dyn CoefficientBatch<f64>>>,
    },
}

impl Binding {
    fn evaluates_coefficients(&self) -> bool {
        match self {
            Self::Tensor { .. } => true,
            Self::Polynomial { coefficients, .. } => coefficients.is_some(),
        }
    }

    fn bind(&mut self, batch: &FastBatch) -> Result<()> {
        match self {
            Self::Tensor {
                plan,
                heads,
                values,
                tensor,
            } => {
                let rows = heads.evaluate_batch(&batch.parameters, batch.lanes)?;
                values.clear();
                values.extend(rows.iter().copied().map(C64::from_symbolica));
                tensor.set_lanes(batch.lanes);
                plan.scatter(values, tensor.coefficients_mut(), batch.lanes, &C64::ZERO)?;
            }
            Self::Polynomial {
                numerator,
                coefficients,
            } => {
                let values = match coefficients {
                    Some(evaluator) => evaluator.evaluate_batch(&batch.parameters, batch.lanes)?,
                    None => batch.parameters.as_slice(),
                };
                numerator
                    .inner_mut()
                    .set_parameter_batch(values, batch.lanes)?;
            }
        }
        Ok(())
    }
}

/// Reusable fast-engine state: reduction workspaces, numerator and masters.
pub(crate) struct FastSession {
    amplitude: FastAmplitude,
    numerator: Binding,
    masters: OneLoopNativeMasters,
    numerator_info: Value,
    results: Vec<FastResult>,
    statuses: Vec<FastStatus>,
}

impl FastSession {
    pub(crate) fn new(
        prepared: &PreparedDocument,
        evaluator: NumeratorEvaluator,
        cut_stop: usize,
    ) -> Result<Self> {
        let start = Instant::now();
        let plan = PolynomialCoefficientPlan::new(
            prepared.numerator_template.clone(),
            loop_variables(),
            prepared.parameters.clone(),
        )?;
        let factoring = matches!(evaluator, NumeratorEvaluator::SymjitO2Factored);
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
        let tensor_plan = match TensorCoefficientPlan::new(&plan) {
            Ok(tensor_plan) => Some(tensor_plan),
            Err(CoefficientError::ExplicitMuSquared { .. }) => None,
            Err(error) => return Err(error.into()),
        };
        let preparation_ns = start.elapsed().as_secs_f64() * 1e9;
        let jit = !matches!(evaluator, NumeratorEvaluator::Symbolica);
        let mut compilation_ns = 0.;
        let mut compile = |jit: bool| -> Result<Box<dyn CoefficientBatch<f64>>> {
            let start = Instant::now();
            let heads: Box<dyn CoefficientBatch<f64>> = if jit {
                f64::compile_symjit_coefficients(&plan)?
            } else {
                Box::new(InterpretedCoefficients::new(&plan)?)
            };
            compilation_ns = start.elapsed().as_secs_f64() * 1e9;
            Ok(heads)
        };
        let mut info = json!({
            "evaluator": evaluator.name(),
            "external_inputs": prepared.parameters.len(),
            "coefficient_preparation_ns": preparation_ns,
            "coefficient_factoring": if factoring { "symbolica_exact" } else { "none" },
            "coefficients_factored": factoring,
            "coefficient_factoring_ns": factoring_ns,
        });
        let numerator = if let Some(tensor_plan) = tensor_plan {
            let heads = compile(jit)?;
            info["interface"] = "tensor".into();
            info["tensor"] = json!({
                "rank": tensor_plan.rank(),
                "ncoeff": tensor_plan.ncoeff(),
                "nonzero_slots": tensor_plan.entries().len(),
                "ordering": "original Ninja/MadLoop tensor_layout (ninjars::fast::tensor_layout)",
            });
            info["coefficient_outputs"] = tensor_plan.head_count().into();
            info["coefficient_policy"] = "all heads evaluated once per sample in every batch and scattered into Ninja's dense tensor; coefficient values not cached".into();
            Binding::Tensor {
                values: Vec::new(),
                tensor: TensorNumerator::new(tensor_plan.rank(), 0),
                plan: tensor_plan,
                heads,
            }
        } else {
            // Ninja's tensor numerator has no mu^2 slots; keep the polynomial
            // numerator and its existing coefficient policy.
            let (expression, parameters, coefficients) = if jit {
                (
                    plan.template().clone(),
                    plan.coefficient_symbols().to_vec(),
                    Some(compile(true)?),
                )
            } else {
                (
                    prepared.numerator_template.clone(),
                    prepared.parameters.clone(),
                    None,
                )
            };
            info["interface"] = "generic-polynomial".into();
            info["interface_reason"] = "explicit mu^2 dependence".into();
            info["coefficient_outputs"] = if jit {
                plan.coefficient_expressions().len().into()
            } else {
                Value::Null
            };
            info["coefficient_policy"] = if jit {
                "all heads evaluated once per sample in every batch; coefficient values not cached"
            } else {
                "external coefficients evaluated within existing cut callbacks"
            }
            .into();
            Binding::Polynomial {
                numerator: Box::new(GenericNumerator::new(PolynomialNumerator::new(
                    expression,
                    loop_variables(),
                    parameters,
                    0_f64,
                )?)),
                coefficients,
            }
        };
        info["coefficient_compilation_ns"] = compilation_ns.into();
        if jit {
            info["optimization_level"] = 2.into();
            info["coefficient_error_policy"] = "reject_binding_batch_before_reduction".into();
        }
        let rank = match &numerator {
            Binding::Tensor { tensor, .. } => tensor.rank(),
            Binding::Polynomial { numerator, .. } => numerator.rank(),
        };
        let amplitude = FastAmplitude::new(
            prepared.propagators,
            rank,
            FastOptions {
                cut_stop: cut_stop.max(1),
                ..FastOptions::default()
            },
        )?;
        Ok(Self {
            amplitude,
            numerator,
            masters: OneLoopNativeMasters::new()?,
            numerator_info: info,
            results: Vec::new(),
            statuses: Vec::new(),
        })
    }

    pub(crate) fn rank(&self) -> usize {
        self.amplitude.rank()
    }

    pub(crate) fn cut_stop(&self) -> usize {
        self.amplitude.options().cut_stop
    }

    pub(crate) fn numerator_provider_info(&self) -> Value {
        self.numerator_info.clone()
    }

    pub(crate) fn kernel_provider_info(&self) -> Value {
        let options = self.amplitude.options();
        json!({
            "selection": "fast",
            "reduction": "ninjars::fast binary64 port of original Ninja's Amplitude::evaluate (ComplexMasses, box mu-expansion, stability checks)",
            "masters": "oneloopmaster native binary64 (OneLoopNativeMasters)",
            "infrared_eps": options.infrared_eps,
            "stability_threshold": options.stability_threshold,
        })
    }

    fn reset(&mut self, lanes: usize) {
        self.results.clear();
        self.results.resize(lanes, FastResult::default());
        self.statuses.clear();
        self.statuses.resize(lanes, FastStatus::NotEvaluated);
    }

    /// Bind coefficients, reset and reduce one packed batch.
    pub(crate) fn evaluate(&mut self, batch: &FastBatch) -> Result<()> {
        self.numerator.bind(batch)?;
        self.reset(batch.lanes);
        let input = batch.input();
        let (results, statuses) = (&mut self.results, &mut self.statuses);
        match &mut self.numerator {
            Binding::Tensor { tensor, .. } => {
                self.amplitude
                    .evaluate(&input, tensor, &mut self.masters, results, statuses)?
            }
            Binding::Polynomial { numerator, .. } => self.amplitude.evaluate(
                &input,
                &mut **numerator,
                &mut self.masters,
                results,
                statuses,
            )?,
        }
        Ok(())
    }

    /// Separate diagnostic path; the primary timing path contains no timer wrappers.
    pub(crate) fn evaluate_diagnostic(
        &mut self,
        batch: &FastBatch,
        timings: &mut DiagnosticTimings,
    ) -> Result<()> {
        if self.numerator.evaluates_coefficients() {
            let start = Instant::now();
            self.numerator.bind(batch)?;
            timings.numerator_coefficients.record(start, batch.lanes);
        } else {
            self.numerator.bind(batch)?;
        }
        self.reset(batch.lanes);
        let input = batch.input();
        let (results, statuses) = (&mut self.results, &mut self.statuses);
        let mut masters = TimedMasters {
            inner: &mut self.masters,
            timing: &mut timings.masters,
        };
        let (evaluation, expansion) = (
            &mut timings.numerator_evaluation,
            &mut timings.numerator_expansion,
        );
        match &mut self.numerator {
            Binding::Tensor { tensor, .. } => {
                let mut numerator = TimedNumerator {
                    inner: tensor,
                    evaluation,
                    expansion,
                };
                self.amplitude
                    .evaluate(&input, &mut numerator, &mut masters, results, statuses)?
            }
            Binding::Polynomial { numerator, .. } => {
                let mut numerator = TimedNumerator {
                    inner: &mut **numerator,
                    evaluation,
                    expansion,
                };
                self.amplitude
                    .evaluate(&input, &mut numerator, &mut masters, results, statuses)?
            }
        }
        Ok(())
    }

    pub(crate) fn status(&self, lane: usize) -> FastStatus {
        self.statuses[lane]
    }

    pub(crate) fn results(&self) -> &[FastResult] {
        &self.results
    }

    pub(crate) fn statuses(&self) -> &[FastStatus] {
        &self.statuses
    }

    /// `[finite, single pole, double pole, cut-constructible finite, rational]`.
    pub(crate) fn components(&self, lane: usize) -> [C64; 5] {
        let result = &self.results[lane];
        let [finite, single, double] = result.laurent;
        [
            finite,
            single,
            double,
            result.cut_constructible,
            result.rational,
        ]
    }
}

/// The generic engine's decimal encoding of a binary64 complex value.
pub(crate) fn encode(value: C64) -> [String; 2] {
    [
        format!("{:.*e}", DIGITS, value.re),
        format!("{:.*e}", DIGITS, value.im),
    ]
}

/// Map fast-engine lane statuses onto the generic status kinds. The fast
/// engine does not report which cut failed.
pub(crate) fn status_json(status: FastStatus) -> Value {
    match status {
        FastStatus::Success => json!({"kind":"success"}),
        FastStatus::NotEvaluated => json!({"kind":"not_evaluated"}),
        FastStatus::Unstable => {
            json!({"kind":"cut_failed","cut_index":null,"reason":"Unstable"})
        }
        FastStatus::MasterFailed => {
            json!({"kind":"integration_failed","cut_index":null,"reason":"MasterFailed"})
        }
        FastStatus::NonFinite => json!({"kind":"nonfinite_result"}),
    }
}

struct TimedNumerator<'a, N: ?Sized> {
    inner: &'a mut N,
    evaluation: &'a mut CallTiming,
    expansion: &'a mut CallTiming,
}

impl<N: FastNumerator + ?Sized> FastNumerator for TimedNumerator<'_, N> {
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    fn evaluate(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        q: &[CMom],
        mu_squared: &[C64],
        out: &mut [C64],
    ) -> ninjars::Result<()> {
        let start = Instant::now();
        let result = self.inner.evaluate(cut, lanes, q, mu_squared, out);
        self.evaluation.record(start, lanes.len());
        result
    }

    fn mu_expansion(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        v0: &[CMom],
        v1: Option<&[CMom]>,
        out: &mut [C64],
    ) -> ninjars::Result<()> {
        let start = Instant::now();
        let result = self.inner.mu_expansion(cut, lanes, v0, v1, out);
        self.expansion.record(start, lanes.len());
        result
    }

    fn t3_expansion(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        a: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[C64],
        mindeg: usize,
        out: &mut [C64],
    ) -> ninjars::Result<()> {
        let start = Instant::now();
        let result = self
            .inner
            .t3_expansion(cut, lanes, a, e3, e4, param, mindeg, out);
        self.expansion.record(start, lanes.len());
        result
    }

    fn t2_expansion(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        a0: &[CMom],
        a1: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[[C64; 3]],
        mindeg: usize,
        out: &mut [C64],
    ) -> ninjars::Result<()> {
        let start = Instant::now();
        let result = self
            .inner
            .t2_expansion(cut, lanes, a0, a1, e3, e4, param, mindeg, out);
        self.expansion.record(start, lanes.len());
        result
    }
}

struct TimedMasters<'a, M> {
    inner: &'a mut M,
    timing: &'a mut CallTiming,
}

impl<M: FastMasters> FastMasters for TimedMasters<'_, M> {
    fn evaluate(
        &mut self,
        family: FastMaster,
        args: &[C64],
        out: &mut [[C64; 3]],
        ok: &mut [bool],
    ) -> ninjars::Result<()> {
        let start = Instant::now();
        let result = self.inner.evaluate(family, args, out, ok);
        self.timing.record(start, ok.len());
        result
    }
}
