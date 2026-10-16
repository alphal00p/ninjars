//! Numerical physical preparation, independent of reduction and master integrals.
//!
//! Static graph contraction, routing and model binding use the existing shared
//! APIs. Propagators and coefficient heads use Symbolica batch evaluators.
//! FeynKit currently exposes scalar wavefunction construction: those calls are
//! per lane. This is an explicit remaining arithmetic-batching gap.
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::atomic::{AtomicU64, Ordering},
};

use feynkit_graph::{EdgeId, ExternalState};
use feynkit_kinematics::{FourMomentum, Helicity, KinematicScalar, WavefunctionKind};
use symbolica::{
    atom::{Atom, AtomCore, Symbol},
    domains::float::{Complex, DoubleFloat, Float, FloatLike, Real},
    evaluate::{BatchExpressionEvaluator, EvaluationDomain},
    id::{Pattern, Replacement},
    symbol,
};
use thiserror::Error;

use crate::{CoefficientEvaluator, PolynomialCoefficientPlan, PreparedIntegrand, symbols};

static NEXT_PLAN: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Error)]
pub enum PhysicalInputError {
    #[error("physical preparation: {0}")]
    Preparation(String),
    #[error("{buffer} has {actual} entries; expected {expected}")]
    Shape {
        buffer: &'static str,
        actual: usize,
        expected: usize,
    },
    #[error("physical batch dimensions overflow")]
    Overflow,
    #[error("physical sample {row}: {reason}")]
    Sample { row: usize, reason: String },
}
type Result<T> = std::result::Result<T, PhysicalInputError>;
fn error(value: impl std::fmt::Display) -> PhysicalInputError {
    PhysicalInputError::Preparation(value.to_string())
}
fn sample_error(row: usize, value: impl std::fmt::Display) -> PhysicalInputError {
    PhysicalInputError::Sample {
        row,
        reason: value.to_string(),
    }
}
fn entries(rows: usize, columns: usize) -> Result<usize> {
    rows.checked_mul(columns)
        .ok_or(PhysicalInputError::Overflow)
}
fn shape(buffer: &'static str, actual: usize, rows: usize, columns: usize) -> Result<()> {
    let expected = entries(rows, columns)?;
    if actual != expected {
        return Err(PhysicalInputError::Shape {
            buffer,
            actual,
            expected,
        });
    }
    Ok(())
}

mod supported {
    pub trait Scalar {}
    impl Scalar for f64 {}
    impl Scalar for symbolica::domains::float::DoubleFloat {}
    impl Scalar for symbolica::domains::float::Float {}
}

/// Supported ordinary real carriers. Dual preparation needs a separate flattened
/// evaluator and all-component validation; primal-only finiteness is insufficient.
pub trait PhysicalScalar: KinematicScalar + supported::Scalar {}
impl PhysicalScalar for f64 {}
impl PhysicalScalar for DoubleFloat {}
impl PhysicalScalar for Float {}

/// Shared physical-leg metadata in the caller's dense physical index order.
#[derive(Clone, Debug)]
pub struct PhysicalExternalLeg {
    pub index: usize,
    pub edge: EdgeId,
    pub pdg: i64,
    pub state: ExternalState,
    pub wavefunction: WavefunctionKind,
}

/// Exact common graph/model plan. No phase-space point is specialized at setup.
pub struct PhysicalInputPlan {
    coefficients: PolynomialCoefficientPlan,
    propagation: Vec<Atom>,
    parameters: Vec<Atom>,
    model_parameters: Vec<Symbol>,
    external_legs: Vec<PhysicalExternalLeg>,
    momentum_sources: Vec<usize>,
    vector_sources: Vec<(usize, WavefunctionKind)>,
    propagator_edges: Vec<EdgeId>,
}

impl PhysicalInputPlan {
    /// `graph` must already contain its intended external/color projector.
    /// Its overall factor is included exactly once. Free model symbols are
    /// supplied in the caller's order; fixed bindings remain exact `Atom`s.
    /// Unoverridden model defaults retain their model-provided precision.
    /// Supported external particles are scalars and vectors, not fermions.
    pub fn new(
        graph: &PreparedIntegrand,
        loop_variables: [Atom; 5],
        model_parameters: Vec<Symbol>,
        fixed_bindings: BTreeMap<Symbol, Atom>,
        factor_coefficients: bool,
    ) -> Result<Self> {
        let model = graph.diagram.model();
        let known_scalars: BTreeSet<_> = model
            .parameters()
            .iter()
            .map(|parameter| symbol!(format!("UFO::{}", parameter.name)))
            .chain(
                model
                    .couplings()
                    .iter()
                    .map(|coupling| symbol!(format!("UFO::{}", coupling.name))),
            )
            .collect();
        for parameter in model_parameters.iter().chain(fixed_bindings.keys()) {
            if !known_scalars.contains(parameter) {
                return Err(error(format!(
                    "unknown model scalar {}",
                    Atom::var(*parameter)
                )));
            }
        }
        for (i, parameter) in model_parameters.iter().enumerate() {
            if model_parameters[..i].contains(parameter) || fixed_bindings.contains_key(parameter) {
                return Err(error(
                    "model inputs must be distinct and disjoint from fixed bindings",
                ));
            }
        }
        let namespace = NEXT_PLAN.fetch_add(1, Ordering::Relaxed);
        let variable = |name: &str| {
            Atom::var(symbol!(format!(
                "ninjars_physical_input::plan{namespace}_{name}"
            )))
        };
        let model_inputs: Vec<_> = (0..model_parameters.len())
            .map(|i| variable(&format!("model{i}")))
            .collect();
        let mut overrides = fixed_bindings;
        overrides.extend(
            model_parameters
                .iter()
                .copied()
                .zip(model_inputs.iter().cloned()),
        );
        let bindings: Vec<_> = model
            .scalar_bindings(None, &overrides)
            .map_err(error)?
            .into_iter()
            .map(|(symbol, value)| {
                Replacement::new(Pattern::Literal(Atom::var(symbol)), Pattern::Literal(value))
            })
            .collect();
        let mut external_legs = Vec::new();
        let mut external_masses = Vec::new();
        let mut momentum_sources = Vec::new();
        let mut vector_sources = Vec::new();
        let mut carriers = Vec::new();
        let mut vectors = Vec::new();
        for (position, edge) in graph
            .diagram
            .loop_momentum_basis()
            .external_edges
            .iter()
            .enumerate()
        {
            let data = graph
                .diagram
                .edges()
                .find(|(id, _, _)| id == edge)
                .ok_or_else(|| error("missing external edge"))?
                .2;
            let leg = data
                .external
                .as_ref()
                .ok_or_else(|| error("missing external metadata"))?;
            let particle = model.particle_by_id(data.particle).map_err(error)?;
            let kind = match particle.spin {
                1 => WavefunctionKind::Scalar,
                3 => {
                    let candidates: Vec<_> = [
                        (symbols::epsilon(), WavefunctionKind::Epsilon),
                        (symbols::epsilonbar(), WavefunctionKind::EpsilonBar),
                    ]
                    .into_iter()
                    .filter(|(head, _)| {
                        graph
                            .numerator
                            .pattern_match(
                                &head
                                    .call((edge.0, variable("representation_")))
                                    .to_pattern(),
                                None,
                                None,
                            )
                            .next()
                            .is_some()
                    })
                    .collect();
                    if candidates.len() != 1 {
                        return Err(error(
                            "one vector wavefunction carrier is required per external vector",
                        ));
                    }
                    vectors.push(candidates[0].0.call(edge.0));
                    vector_sources.push((leg.index, candidates[0].1));
                    candidates[0].1
                }
                _ => {
                    return Err(error(format!(
                        "external PDG{} has unsupported spin{}; only scalar/vector preparation is implemented",
                        particle.pdg_code, particle.spin
                    )));
                }
            };
            external_legs.push(PhysicalExternalLeg {
                index: leg.index,
                edge: *edge,
                pdg: particle.pdg_code,
                state: leg.state,
                wavefunction: kind,
            });
            let mass = model.parameter_by_id(particle.mass).map_err(error)?;
            external_masses.push((
                leg.index,
                Atom::var(symbol!(format!("UFO::{}", mass.name))).pow(2),
            ));
            let momentum = symbols::external_momentum().call(position);
            if graph.family.external_momenta().contains(&momentum) {
                carriers.push(momentum);
                momentum_sources.push(leg.index);
            }
        }
        external_legs.sort_by_key(|leg| leg.index);
        if external_legs.is_empty()
            || external_legs
                .iter()
                .enumerate()
                .any(|(index, leg)| index != leg.index)
        {
            return Err(error(
                "physical external indices must be unique and dense from zero",
            ));
        }
        external_masses.sort_by_key(|(index, _)| *index);
        carriers.extend(vectors);
        let mut parameters: Vec<_> = (0..4 * carriers.len())
            .map(|i| variable(&format!("component{i}")))
            .collect();
        let symbolic: BTreeMap<_, _> = carriers
            .into_iter()
            .zip(
                parameters
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .cloned()
                    .map(FourMomentum::from),
            )
            .collect();
        parameters.extend(model_inputs);
        let loop_components = FourMomentum::from([
            loop_variables[0].clone(),
            loop_variables[1].clone(),
            loop_variables[2].clone(),
            loop_variables[3].clone(),
        ]);
        let kinematics = graph
            .component_kinematics(&loop_components, &symbolic)
            .map_err(error)?;
        let numerator = kinematics
            .apply(&(&graph.numerator * &graph.overall_factor))
            .replace_multiple(&bindings);
        let coefficients =
            PolynomialCoefficientPlan::new(numerator, loop_variables, parameters.clone())
                .map_err(error)?;
        let coefficients = if factor_coefficients {
            coefficients.factor_coefficients().map_err(error)?
        } else {
            coefficients
        };
        let shifts = graph.shifts(&symbolic).map_err(error)?;
        let mut propagation: Vec<_> = shifts
            .iter()
            .flat_map(|p| p.components().map(Clone::clone))
            .chain(graph.squared_masses())
            .collect();
        for left in &shifts {
            for right in &shifts {
                let delta = left.clone() - right.clone();
                propagation.push(delta.dot(&delta));
            }
        }
        propagation.extend(external_masses.into_iter().map(|(_, mass)| mass));
        for expression in &mut propagation {
            *expression = expression.replace_multiple(&bindings);
        }
        Ok(Self {
            coefficients,
            propagation,
            parameters,
            model_parameters,
            external_legs,
            momentum_sources,
            vector_sources,
            propagator_edges: graph.propagator_edges.clone(),
        })
    }
    pub fn coefficients(&self) -> &PolynomialCoefficientPlan {
        &self.coefficients
    }
    pub fn external_legs(&self) -> &[PhysicalExternalLeg] {
        &self.external_legs
    }
    pub fn model_parameters(&self) -> &[Symbol] {
        &self.model_parameters
    }
    pub fn propagator_edges(&self) -> &[EdgeId] {
        &self.propagator_edges
    }

    /// Compile once at working precision. The caller supplies a positive relative
    /// tolerance for represented input conservation and mass-shell residuals.
    pub fn compile<T: PhysicalScalar>(
        &self,
        prototype: T,
        relative_tolerance: T,
    ) -> Result<PhysicalInputEvaluator<T>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        if !relative_tolerance.is_finite() || relative_tolerance <= prototype.zero() {
            return Err(error(
                "physical input tolerance must be finite and positive",
            ));
        }
        let propagation = Atom::evaluator_multiple(&self.propagation, &self.parameters)
            .cores(1)
            .build()
            .map_err(error)?
            .map_coeff_with_prec(
                &|v| {
                    Complex::new(
                        prototype.from_rational(&v.re),
                        prototype.from_rational(&v.im),
                    )
                },
                prototype.get_precision(),
            )
            .into_batch_evaluator();
        Ok(PhysicalInputEvaluator {
            coefficients: self
                .coefficients
                .compile(prototype.clone())
                .map_err(error)?,
            propagation,
            external_legs: self.external_legs.clone(),
            momentum_sources: self.momentum_sources.clone(),
            vector_sources: self.vector_sources.clone(),
            model_count: self.model_parameters.len(),
            propagators: self.propagator_edges.len(),
            input_columns: self.parameters.len(),
            output_columns: self.propagation.len(),
            coefficient_columns: self.coefficients.coefficient_expressions().len(),
            prototype,
            relative_tolerance,
            inputs: Vec::new(),
            propagation_values: Vec::new(),
            staging: PhysicalInputValues::default(),
            committed: PhysicalInputValues::default(),
        })
    }
}

/// Row-major physical data; all momenta have positive physical energies.
pub struct PhysicalInputBatch<'a, T> {
    pub lanes: usize,
    /// Row-major, in `PhysicalInputPlan::external_legs()` order.
    pub momenta: &'a [FourMomentum<T>],
    /// Row-major, in the same physical-leg order as `momenta`.
    pub helicities: &'a [Helicity],
    /// Row-major, in `PhysicalInputPlan::model_parameters()` order.
    pub model_parameters: &'a [Complex<T>],
    /// One positive squared renormalization scale per row.
    pub scales_squared: &'a [T],
}

/// Prepared buffers retain the shared momentum and numerical representations.
#[derive(Clone, Debug, PartialEq)]
pub struct PhysicalInputValues<T: FloatLike> {
    pub lanes: usize,
    /// Row-major, in `PhysicalInputPlan::propagator_edges()` order.
    pub shifts: Vec<FourMomentum<Complex<T>>>,
    /// Row-major, in the same propagator order as `shifts`.
    pub masses_squared: Vec<Complex<T>>,
    /// Row-major batches of row-major n×n `(V_i - V_j)^2` matrices.
    pub s_matrix: Vec<Complex<T>>,
    /// Row-major, in the coefficient plan's `coefficient_symbols()` order.
    pub coefficients: Vec<Complex<T>>,
    /// One squared renormalization scale per row.
    pub scales_squared: Vec<Complex<T>>,
}
impl<T: FloatLike> Default for PhysicalInputValues<T> {
    fn default() -> Self {
        Self {
            lanes: 0,
            shifts: Vec::new(),
            masses_squared: Vec::new(),
            s_matrix: Vec::new(),
            coefficients: Vec::new(),
            scales_squared: Vec::new(),
        }
    }
}

pub struct PhysicalInputEvaluator<T: FloatLike> {
    coefficients: CoefficientEvaluator<T>,
    propagation: BatchExpressionEvaluator<Complex<T>>,
    external_legs: Vec<PhysicalExternalLeg>,
    momentum_sources: Vec<usize>,
    vector_sources: Vec<(usize, WavefunctionKind)>,
    model_count: usize,
    propagators: usize,
    input_columns: usize,
    output_columns: usize,
    coefficient_columns: usize,
    prototype: T,
    relative_tolerance: T,
    inputs: Vec<Complex<T>>,
    propagation_values: Vec<Complex<T>>,
    staging: PhysicalInputValues<T>,
    committed: PhysicalInputValues<T>,
}

impl<T: PhysicalScalar> PhysicalInputEvaluator<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    /// Last successfully prepared batch. Errors leave it unchanged.
    pub fn values(&self) -> &PhysicalInputValues<T> {
        &self.committed
    }
    /// Shape and domain errors reject the whole batch transactionally. Propagator
    /// and coefficient arithmetic each execute one shared batched evaluator call.
    pub fn evaluate_batch(
        &mut self,
        input: PhysicalInputBatch<'_, T>,
    ) -> Result<&PhysicalInputValues<T>> {
        let rows = input.lanes;
        let legs = self.external_legs.len();
        let n = self.propagators;
        shape("physical momenta", input.momenta.len(), rows, legs)?;
        shape("physical helicities", input.helicities.len(), rows, legs)?;
        shape(
            "physical model parameters",
            input.model_parameters.len(),
            rows,
            self.model_count,
        )?;
        shape("physical scales", input.scales_squared.len(), rows, 1)?;
        let zero = self.prototype.zero();
        let complex_zero = Complex::new(zero.clone(), zero.clone());
        self.inputs
            .resize(entries(rows, self.input_columns)?, complex_zero.clone());
        self.inputs.fill(complex_zero.clone());
        for row in 0..rows {
            let momenta = &input.momenta[row * legs..(row + 1) * legs];
            if momenta
                .iter()
                .flat_map(FourMomentum::components)
                .any(|v| !v.is_finite())
                || !input.scales_squared[row].is_finite()
                || input.scales_squared[row] <= zero
                || momenta.iter().any(|p| p.temporal.value <= zero)
            {
                return Err(sample_error(
                    row,
                    "nonfinite momentum or nonpositive energy/scale",
                ));
            }
            let model =
                &input.model_parameters[row * self.model_count..(row + 1) * self.model_count];
            if model.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
                return Err(sample_error(row, "nonfinite model input"));
            }
            let mut conservation =
                FourMomentum::from([zero.clone(), zero.clone(), zero.clone(), zero.clone()]);
            let mut energy_scale = self.prototype.one();
            for (momentum, leg) in momenta.iter().zip(&self.external_legs) {
                conservation = if leg.state == ExternalState::Incoming {
                    conservation + momentum.clone()
                } else {
                    conservation - momentum.clone()
                };
                if momentum.temporal.value > energy_scale {
                    energy_scale = momentum.temporal.value.clone();
                }
            }
            if conservation.components().iter().any(|v| {
                !v.is_finite() || v.norm() > self.relative_tolerance.clone() * energy_scale.clone()
            }) {
                return Err(sample_error(row, "momentum conservation failed"));
            }
            let cells = &mut self.inputs[row * self.input_columns..(row + 1) * self.input_columns];
            for (slot, &leg) in self.momentum_sources.iter().enumerate() {
                for (j, value) in momenta[leg].components().iter().enumerate() {
                    cells[4 * slot + j] = Complex::new((*value).clone(), zero.clone());
                }
            }
            cells[self.input_columns - self.model_count..].clone_from_slice(model);
        }
        self.propagation_values
            .resize(entries(rows, self.output_columns)?, complex_zero.clone());
        if rows > 0 {
            self.propagation
                .evaluate_batch(&self.inputs, &mut self.propagation_values, rows)
                .map_err(error)?;
        }
        for row in 0..rows {
            let values = &self.propagation_values
                [row * self.output_columns..(row + 1) * self.output_columns];
            if values
                .iter()
                .any(|z| !z.re.is_finite() || !z.im.is_finite())
            {
                return Err(sample_error(row, "nonfinite propagated mass or invariant"));
            }
            let momenta = &input.momenta[row * legs..(row + 1) * legs];
            let mut energy_scale = self.prototype.one();
            for p in momenta {
                if p.temporal.value > energy_scale {
                    energy_scale = p.temporal.value.clone();
                }
            }
            let tolerance = self.relative_tolerance.clone() * energy_scale.clone() * energy_scale;
            if !tolerance.is_finite() {
                return Err(sample_error(row, "mass-shell scale overflow"));
            }
            for (leg, p) in momenta.iter().enumerate() {
                let mass = &values[5 * n + n * n + leg];
                let residual = p.mass_squared() - mass.re.clone();
                if !mass.im.is_zero()
                    || mass.re < zero
                    || !residual.is_finite()
                    || residual.norm() > tolerance
                {
                    return Err(sample_error(
                        row,
                        format!("mass-shell check failed for leg{leg}"),
                    ));
                }
            }
            // Reuse owning-crate wavefunctions; no local helicity/spinor algebra.
            for leg in &self.external_legs {
                if leg.wavefunction == WavefunctionKind::Scalar {
                    momenta[leg.index]
                        .wavefunction(leg.wavefunction, input.helicities[row * legs + leg.index])
                        .map_err(|e| sample_error(row, e))?;
                }
            }
            for (index, &(leg, kind)) in self.vector_sources.iter().enumerate() {
                let wave = momenta[leg]
                    .wavefunction(kind, input.helicities[row * legs + leg])
                    .map_err(|e| sample_error(row, e))?;
                let offset = row * self.input_columns + 4 * (self.momentum_sources.len() + index);
                self.inputs[offset..offset + 4].clone_from_slice(wave.components());
            }
        }
        self.staging
            .coefficients
            .resize(entries(rows, self.coefficient_columns)?, complex_zero);
        self.coefficients
            .evaluate_batch(&self.inputs, &mut self.staging.coefficients, rows)
            .map_err(error)?;
        self.staging.shifts.clear();
        self.staging.masses_squared.clear();
        self.staging.s_matrix.clear();
        self.staging.scales_squared.clear();
        for row in 0..rows {
            let values = &self.propagation_values
                [row * self.output_columns..(row + 1) * self.output_columns];
            self.staging.shifts.extend(
                values[..4 * n]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .cloned()
                    .map(FourMomentum::from),
            );
            self.staging
                .masses_squared
                .extend_from_slice(&values[4 * n..5 * n]);
            self.staging
                .s_matrix
                .extend_from_slice(&values[5 * n..5 * n + n * n]);
            self.staging.scales_squared.push(Complex::new(
                input.scales_squared[row].clone(),
                zero.clone(),
            ));
        }
        self.staging.lanes = rows;
        std::mem::swap(&mut self.staging, &mut self.committed);
        Ok(&self.committed)
    }
}
