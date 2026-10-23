//! `--engine fast`: the binary64 `ninjars::fast` engine behind `reduce` and `bench`.
//!
//! Inputs are bound exactly as for the generic engine (`NumericalPool<f64>`,
//! including the exact S-matrix rule) and repacked once into the fast
//! engine's real-momentum layout; complex external kinematics are rejected.
//! The numerator is original Ninja's dense `TensorNumerator`, filled per batch
//! from the coefficient heads of a `TensorCoefficientPlan` (Symbolica's
//! interpreter or SymJIT O2, following `--numerator-evaluator`). A prepared
//! numerator with explicit `mu^2` terms (`--r2 fdh`, or explicit fixture
//! terms) uses the same heads in the `mu^2` block layout of
//! `MuTensorNumerator` (`N = sum_k mu^(2k) T_k(q)`); a `mu^2`-free numerator
//! keeps the plain `TensorNumerator`. Masters are oneloopmaster's native
//! binary64 code (`OneLoopNativeMasters`).

use std::{error::Error, time::Instant};

use ninjars::fast::{
    C64, CMom, FastAmplitude, FastInput, FastMaster, FastMasters, FastNumerator, FastOptions,
    FastResult, FastStatus, MuTensorNumerator, RMom, TensorNumerator,
};
use ninjars_graph::{PolynomialCoefficientPlan, TensorCoefficientPlan};
use ninjars_symbolica::OneLoopNativeMasters;
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

/// Batched dense numerator filled from tensor coefficients: Ninja's plain
/// tensor, or the tensor with explicit `mu^2` blocks (`--r2 fdh`). Benchmarks
/// monomorphize over it, so the `mu^2`-free path is exactly `TensorNumerator`.
pub(crate) trait DenseNumerator: FastNumerator + Sized {
    fn with_lanes(rank: usize, mu_powers: usize, lanes: usize) -> Result<Self>;
    fn coefficients(&self) -> &[C64];
    fn coefficients_mut(&mut self) -> &mut [C64];
}

impl DenseNumerator for TensorNumerator {
    fn with_lanes(rank: usize, _mu_powers: usize, lanes: usize) -> Result<Self> {
        Ok(TensorNumerator::new(rank, lanes))
    }
    fn coefficients(&self) -> &[C64] {
        TensorNumerator::coefficients(self)
    }
    fn coefficients_mut(&mut self) -> &mut [C64] {
        TensorNumerator::coefficients_mut(self)
    }
}

impl DenseNumerator for MuTensorNumerator {
    fn with_lanes(rank: usize, mu_powers: usize, lanes: usize) -> Result<Self> {
        Ok(MuTensorNumerator::new(rank, mu_powers, lanes)?)
    }
    fn coefficients(&self) -> &[C64] {
        MuTensorNumerator::coefficients(self)
    }
    fn coefficients_mut(&mut self) -> &mut [C64] {
        MuTensorNumerator::coefficients_mut(self)
    }
}

/// The dense numerator filled from the coefficient heads.
enum Tensor {
    /// Ninja's tensor numerator (no explicit mu^2 terms).
    Plain(TensorNumerator),
    /// Explicit mu^2 blocks, `sum_k mu^(2k) T_k(q)`.
    Mu(MuTensorNumerator),
}

/// The numerator provider and its per-batch coefficient binding.
struct Binding {
    plan: TensorCoefficientPlan,
    heads: Box<dyn CoefficientBatch<f64>>,
    values: Vec<C64>,
    tensor: Tensor,
}

impl Binding {
    fn bind(&mut self, batch: &FastBatch) -> Result<()> {
        let rows = self.heads.evaluate_batch(&batch.parameters, batch.lanes)?;
        self.values.clear();
        self.values
            .extend(rows.iter().copied().map(C64::from_symbolica));
        let coefficients = match &mut self.tensor {
            Tensor::Plain(tensor) => {
                tensor.set_lanes(batch.lanes);
                tensor.coefficients_mut()
            }
            Tensor::Mu(tensor) => {
                tensor.set_lanes(batch.lanes);
                tensor.coefficients_mut()
            }
        };
        self.plan
            .scatter(&self.values, coefficients, batch.lanes, &C64::ZERO)?;
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
        // Ninja's plain tensor layout unless the numerator has mu^2 terms.
        let tensor_plan = TensorCoefficientPlan::with_mu_squared(&plan, None)?;
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
        let heads = compile(jit)?;
        let mu_powers = tensor_plan.mu_powers();
        info["interface"] = if mu_powers == 0 {
            "tensor"
        } else {
            "mu-tensor"
        }
        .into();
        info["tensor"] = json!({
            "rank": tensor_plan.rank(),
            "ncoeff": tensor_plan.ncoeff(),
            "nonzero_slots": tensor_plan.entries().len(),
            "ordering": "original Ninja/MadLoop tensor_layout (ninjars::fast::tensor_layout)",
        });
        if mu_powers > 0 {
            info["tensor"]["mu_powers"] = mu_powers.into();
            info["tensor"]["layout"] = "blocks k = 0..=mu_powers of mu^(2k) T_k(q), T_k of rank (rank - 2k) (ninjars::fast::MuTensorNumerator)".into();
        }
        info["coefficient_outputs"] = tensor_plan.head_count().into();
        info["coefficient_policy"] = "all heads evaluated once per sample in every batch and scattered into Ninja's dense tensor; coefficient values not cached".into();
        let tensor = if mu_powers == 0 {
            Tensor::Plain(TensorNumerator::new(tensor_plan.rank(), 0))
        } else {
            Tensor::Mu(MuTensorNumerator::new(tensor_plan.rank(), mu_powers, 0)?)
        };
        let numerator = Binding {
            values: Vec::new(),
            tensor,
            plan: tensor_plan,
            heads,
        };
        info["coefficient_compilation_ns"] = compilation_ns.into();
        if jit {
            info["optimization_level"] = 2.into();
            info["coefficient_error_policy"] = "reject_binding_batch_before_reduction".into();
        }
        let rank = numerator.plan.rank();
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
        match &mut self.numerator.tensor {
            Tensor::Plain(tensor) => {
                self.amplitude
                    .evaluate(&input, tensor, &mut self.masters, results, statuses)?
            }
            Tensor::Mu(tensor) => {
                self.amplitude
                    .evaluate(&input, tensor, &mut self.masters, results, statuses)?
            }
        }
        Ok(())
    }

    /// Separate diagnostic path; the primary timing path contains no timer wrappers.
    pub(crate) fn evaluate_diagnostic(
        &mut self,
        batch: &FastBatch,
        timings: &mut DiagnosticTimings,
    ) -> Result<()> {
        let start = Instant::now();
        self.numerator.bind(batch)?;
        timings.numerator_coefficients.record(start, batch.lanes);
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
        match &mut self.numerator.tensor {
            Tensor::Plain(tensor) => {
                let mut numerator = TimedNumerator {
                    inner: tensor,
                    evaluation,
                    expansion,
                };
                self.amplitude
                    .evaluate(&input, &mut numerator, &mut masters, results, statuses)?
            }
            Tensor::Mu(tensor) => {
                let mut numerator = TimedNumerator {
                    inner: tensor,
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

    fn mu_expansion_x2(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        v0: &[CMom],
        v1: &[CMom],
        beta: &[C64],
        out: &mut [C64],
    ) -> ninjars::Result<()> {
        let start = Instant::now();
        let result = self.inner.mu_expansion_x2(cut, lanes, v0, v1, beta, out);
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

    fn t3_expansion_x2(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        a: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[C64],
        out: &mut [C64],
    ) -> ninjars::Result<()> {
        let start = Instant::now();
        let result = self
            .inner
            .t3_expansion_x2(cut, lanes, a, e3, e4, param, out);
        self.expansion.record(start, lanes.len());
        result
    }

    fn t2_expansion_x2(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        a0: &[CMom],
        a1: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[[C64; 3]],
        out: &mut [C64],
    ) -> ninjars::Result<()> {
        let start = Instant::now();
        let result = self
            .inner
            .t2_expansion_x2(cut, lanes, a0, a1, e3, e4, param, out);
        self.expansion.record(start, lanes.len());
        result
    }

    fn lane_rank(&mut self, lane: usize) -> Option<usize> {
        self.inner.lane_rank(lane)
    }

    fn lower_rank(&mut self, lanes: &[usize], rank: usize) -> ninjars::Result<()> {
        self.inner.lower_rank(lanes, rank)
    }

    fn restore_rank(&mut self) {
        self.inner.restore_rank();
    }
}

struct TimedMasters<'a, M> {
    inner: &'a mut M,
    timing: &'a mut CallTiming,
}

impl<M: FastMasters> FastMasters for TimedMasters<'_, M> {
    fn requires_scaleless_bubbles(&self) -> bool {
        self.inner.requires_scaleless_bubbles()
    }

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
