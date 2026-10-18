//! Per-point physical numerator preparation in Ninja's dense tensor layout.
//!
//! Setup (graph import, Feynman rules, contraction, polynomial preparation,
//! coefficient planning/factoring and compilation) happens once in
//! [`PhysicalTensorNumerator::new`], which records the duration of every
//! stage. Each [`PhysicalTensorNumerator::prepare`] call then starts from raw
//! binary64 inputs per phase-space point — the physical four-momenta of all
//! external legs in MadGraph order (incoming then outgoing, positive energies,
//! `(E, px, py, pz)`), one helicity per leg and one row of free model
//! parameters — computes the external wavefunctions, evaluates the multi-output
//! coefficient evaluator and scatters the heads into `lanes x ncoeff`
//! tensor coefficients. Warmed calls do not allocate.
//!
//! With [`R2Scheme::Fdh`] the loop momentum is bound d-dimensionally
//! (`K.K -> q.q - mu^2`), so the tensor carries explicit `mu^2` powers in the
//! block layout of `ninjars::fast::MuTensorNumerator`
//! ([`TensorCoefficientPlan::with_mu_squared`], [`PhysicalTensorNumerator::mu_powers`]);
//! [`R2Scheme::Off`] keeps Ninja's plain tensor unchanged.
//!
//! External vector wavefunctions are FeynKit's (GammaLoop's HELAS/MadGraph
//! convention): an incoming vector of helicity `h` is `vxxxxx(p, m, h, -1)`,
//! an outgoing one `vxxxxx(p, m, h, +1)`, i.e. the conjugate polarization
//! (for massless vectors FeynKit normalizes by `|p|` where HELAS uses `E`; the
//! two agree to rounding on shell). They are recomputed here without
//! allocation, bit-for-bit identical to `FourMomentum::wavefunction`.

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
    time::{Duration, Instant},
};

use feynkit_graph::ExternalState;
use feynkit_kinematics::{FourMomentum, Kinematics};
use feynkit_model::Model;
use symbolica::{
    atom::{Atom, AtomCore, AtomView, Symbol},
    domains::float::{Complex, SingleFloat},
    evaluate::ExpressionEvaluator,
    id::{Pattern, Replacement},
    parser::ParseSettings,
    symbol,
};
use thiserror::Error;

use crate::{
    CoefficientError, CoefficientEvaluator, NumeratorMode, PolynomialCoefficientPlan,
    PreparationError, PreparationTimings, PreparedIntegrand, R2Scheme, TensorCoefficientPlan,
    prepare_dot_timed, symbols,
};

#[derive(Debug, Error)]
pub enum PhysicalNumeratorError {
    #[error(transparent)]
    Preparation(#[from] Box<PreparationError>),
    #[error(transparent)]
    Coefficient(#[from] CoefficientError),
    #[error("model: {0}")]
    Model(String),
    #[error("external leg {leg} has spin {spin}; only scalar and vector legs are supported")]
    UnsupportedLeg { leg: usize, spin: i64 },
    #[error("external legs must be numbered 0..{0} exactly once")]
    LegNumbering(usize),
    #[error("vector leg {leg} must have exactly one prepared polarization carrier; found {found}")]
    Carrier { leg: usize, found: usize },
    #[error("{buffer} has {actual} entries; expected {expected}")]
    Shape {
        buffer: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("row {row}, leg {leg}: helicity {helicity} is not allowed for this state")]
    Helicity {
        row: usize,
        leg: usize,
        helicity: i8,
    },
    #[error("row {row}: nonfinite physical input")]
    NonFinite { row: usize },
    #[error("symbolic preparation: {0}")]
    Symbolic(String),
}

type Result<T> = std::result::Result<T, PhysicalNumeratorError>;

impl From<PreparationError> for PhysicalNumeratorError {
    fn from(error: PreparationError) -> Self {
        Self::Preparation(Box::new(error))
    }
}

/// Coefficient-head execution backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CoefficientBackend {
    /// Symbolica's binary64 batch interpreter.
    Interpreter,
    /// SymJIT O2 (requires the `numerator-jit` feature).
    #[default]
    SymJitO2,
}

/// Setup description of a physical tensor numerator.
#[derive(Clone)]
pub struct PhysicalNumeratorSpec {
    pub model: Arc<Model>,
    /// FeynKit DOT source of one one-loop graph with its projector.
    pub dot: String,
    pub numerator_mode: NumeratorMode,
    /// Apply Idenso's fundamental-color dimension invariants after contraction.
    pub color_dimension_invariants: bool,
    /// UFO external parameters supplied numerically per phase-space point,
    /// in the column order of the `model` rows (e.g. `["MT", "ymt", "MH"]`).
    pub runtime_parameters: Vec<String>,
    /// UFO parameters fixed exactly at setup (e.g. zero widths).
    pub fixed_parameters: BTreeMap<String, Atom>,
    /// Factor every coefficient head exactly with Symbolica before compiling.
    pub factor_coefficients: bool,
    /// Tensor rank; defaults to the numerator degree.
    pub tensor_rank: Option<usize>,
    pub backend: CoefficientBackend,
    /// Loop-momentum dimension of the binding: `Fdh` adds the `mu^2` terms (R2).
    pub r2: R2Scheme,
}

/// External-state treatment of one physical leg.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LegState {
    Scalar,
    /// Incoming vector, polarization `epsilon`.
    Epsilon,
    /// Outgoing vector, polarization `epsilon*`.
    EpsilonBar,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalLeg {
    /// Physical (MadGraph) leg index; position in every momentum/helicity row.
    pub index: usize,
    pub pdg: i64,
    pub incoming: bool,
    pub state: LegState,
}

/// Wall-clock durations of every one-time setup stage.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GenerationTimings {
    pub preparation: PreparationTimings,
    /// Fundamental-color invariants (zero when disabled).
    pub color: Duration,
    /// Component kinematics, model bindings and expansion into the loop polynomial.
    pub polynomial: Duration,
    /// Coefficient-head extraction and template ([`PolynomialCoefficientPlan::new`]).
    pub planning: Duration,
    /// Exact Symbolica factoring of the heads (zero when disabled).
    pub factoring: Duration,
    /// Static tensor scatter ([`TensorCoefficientPlan`]).
    pub tensor_plan: Duration,
    /// Symbolica's shared multi-output evaluator (graph optimization) and its
    /// binary64 coefficient mapping; the complete interpreter setup for
    /// [`CoefficientBackend::Interpreter`].
    pub evaluator: Duration,
    /// SymJIT O2 machine-code compilation (zero for the interpreter).
    pub compilation: Duration,
    /// Shift and mass evaluators for [`PhysicalTensorNumerator::kinematics`].
    pub kinematics: Duration,
}

impl GenerationTimings {
    /// Sum of all stages.
    pub fn total(&self) -> Duration {
        let p = &self.preparation;
        p.import
            + p.feynman_rules
            + p.propagators
            + p.contraction
            + self.color
            + self.polynomial
            + self.planning
            + self.factoring
            + self.tensor_plan
            + self.evaluator
            + self.compilation
            + self.kinematics
    }
}

enum Coefficients {
    Interpreter(CoefficientEvaluator<f64>),
    #[cfg(feature = "numerator-jit")]
    Jit(crate::SymJitCoefficientEvaluator),
}

impl Coefficients {
    fn evaluate(
        &mut self,
        input: &[Complex<f64>],
        output: &mut [Complex<f64>],
        rows: usize,
    ) -> std::result::Result<(), CoefficientError> {
        match self {
            Self::Interpreter(evaluator) => evaluator.evaluate_batch(input, output, rows),
            #[cfg(feature = "numerator-jit")]
            Self::Jit(evaluator) => evaluator.evaluate_batch(input, output, rows),
        }
    }
}

/// Reusable per-point tensor-numerator preparation for one physical graph.
pub struct PhysicalTensorNumerator {
    legs: Vec<PhysicalLeg>,
    /// Leg of each independent external momentum, in parameter order.
    sources: Vec<usize>,
    /// Vector legs carrying a polarization parameter, in parameter order.
    waves: Vec<(usize, bool)>,
    runtime_parameters: Vec<String>,
    plan: PolynomialCoefficientPlan,
    tensor: TensorCoefficientPlan,
    coefficients: Coefficients,
    backend: CoefficientBackend,
    r2: R2Scheme,
    propagators: usize,
    /// Row-major `propagators x sources`: `V_i = sum_j c_ij p_{sources[j]}`.
    shift_coefficients: Vec<f64>,
    masses: ExpressionEvaluator<Complex<f64>>,
    model_input: Vec<Complex<f64>>,
    overall_factor: Atom,
    timings: GenerationTimings,
    input: Vec<Complex<f64>>,
    heads: Vec<Complex<f64>>,
}

fn symbolic(error: impl std::fmt::Display) -> PhysicalNumeratorError {
    PhysicalNumeratorError::Symbolic(error.to_string())
}

fn ufo(name: &str) -> Result<Symbol> {
    let atom = Atom::parse(name, "UFO", ParseSettings::default()).map_err(symbolic)?;
    match atom.as_view() {
        AtomView::Var(symbol) => Ok(symbol.get_symbol()),
        _ => Err(PhysicalNumeratorError::Model(format!(
            "parameter name {name:?} is not a symbol"
        ))),
    }
}

fn shape(buffer: &'static str, rows: usize, columns: usize, actual: usize) -> Result<()> {
    let expected = rows
        .checked_mul(columns)
        .ok_or(PhysicalNumeratorError::Shape {
            buffer,
            expected: usize::MAX,
            actual,
        })?;
    if expected != actual {
        return Err(PhysicalNumeratorError::Shape {
            buffer,
            expected,
            actual,
        });
    }
    Ok(())
}

/// FeynKit's vector wavefunction without allocation; `bar` conjugates.
///
/// Identical floating-point operations to `FourMomentum::wavefunction` with
/// `WavefunctionKind::{Epsilon, EpsilonBar}`. Returns `None` for helicities
/// the state does not admit (longitudinal massless/at-rest vectors).
pub fn vector_wavefunction(
    momentum: &[f64; 4],
    helicity: i8,
    bar: bool,
) -> Option<[Complex<f64>; 4]> {
    let p = FourMomentum::from(*momentum);
    let mut components = match helicity {
        0 => {
            let mass2 = p.mass_squared();
            let norm = p.spatial.norm();
            if mass2 <= 0. || norm == 0. {
                return None;
            }
            let mass = mass2.sqrt();
            let scale = p.temporal.value / (mass * norm);
            [
                norm / mass,
                p.spatial.px * scale,
                p.spatial.py * scale,
                p.spatial.pz * scale,
            ]
            .map(|x| Complex::new(x, 0.))
        }
        1 | -1 => {
            let [a, b] = p.transverse_basis();
            let scale = 1. / 2_f64.sqrt();
            std::array::from_fn(|i| {
                Complex::new(
                    (if helicity == 1 { -a[i] } else { a[i] }) * scale,
                    -b[i] * scale,
                )
            })
        }
        _ => return None,
    };
    if bar {
        for component in &mut components {
            component.im = -component.im;
        }
    }
    Some(components)
}

impl PhysicalTensorNumerator {
    pub fn new(spec: &PhysicalNumeratorSpec) -> Result<Self> {
        let mut timings = GenerationTimings::default();
        let (prepared, preparation) = prepare_dot_timed(
            spec.model.clone(),
            &spec.dot,
            &Kinematics::new(),
            spec.numerator_mode,
        )?;
        timings.preparation = preparation;
        let start = Instant::now();
        let prepared = if spec.color_dimension_invariants {
            prepared.with_color_dimension_invariants()?
        } else {
            prepared
        };
        timings.color = start.elapsed();

        let start = Instant::now();
        let model = &spec.model;
        for name in spec
            .runtime_parameters
            .iter()
            .chain(spec.fixed_parameters.keys())
        {
            if !model.parameters().iter().any(|p| &p.name == name) {
                return Err(PhysicalNumeratorError::Model(format!(
                    "unknown model parameter {name:?}"
                )));
            }
        }
        let model_symbols: Vec<Atom> = spec
            .runtime_parameters
            .iter()
            .map(|name| Atom::var(symbol!(format!("ninjars_physical::model_{name}"))))
            .collect();
        let mut overrides = BTreeMap::new();
        for (name, value) in &spec.fixed_parameters {
            overrides.insert(ufo(name)?, value.clone());
        }
        for (name, value) in spec.runtime_parameters.iter().zip(&model_symbols) {
            if overrides.insert(ufo(name)?, value.clone()).is_some() {
                return Err(PhysicalNumeratorError::Model(format!(
                    "parameter {name:?} is both fixed and runtime, or repeated"
                )));
            }
        }
        let replacements: Vec<_> = model
            .scalar_bindings(None, &overrides)
            .map_err(|e| PhysicalNumeratorError::Model(e.to_string()))?
            .into_iter()
            .map(|(key, value)| {
                Replacement::new(Pattern::Literal(Atom::var(key)), Pattern::Literal(value))
            })
            .collect();

        let (legs, carriers, sources, waves) = Self::legs(&prepared)?;
        let mut generation = 0;
        let mut parameters = loop {
            let candidates: Vec<_> = (0..4 * carriers.len())
                .map(|i| Atom::var(symbol!(format!("ninjars_physical::p{generation}_{i}"))))
                .collect();
            if candidates.iter().all(|parameter| {
                !prepared.numerator.contains(parameter)
                    && !prepared.overall_factor.contains(parameter)
            }) {
                break candidates;
            }
            generation += 1;
        };
        let symbolic_external: BTreeMap<_, _> = carriers
            .iter()
            .cloned()
            .zip(
                parameters
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .cloned()
                    .map(FourMomentum::from),
            )
            .collect();
        parameters.extend(model_symbols.iter().cloned());
        let variables = ["q0", "q1", "q2", "q3", "mu2"]
            .map(|name| Atom::var(symbol!(format!("ninjars::{name}"))));
        let loop_components = FourMomentum::from_args(
            variables[0].clone(),
            variables[1].clone(),
            variables[2].clone(),
            variables[3].clone(),
        );
        let kinematics = prepared.dimensional_kinematics(
            &loop_components,
            spec.r2.mu_squared(&variables[4]),
            &symbolic_external,
        )?;
        let expression = kinematics
            .apply(&(&prepared.numerator * &prepared.overall_factor))
            .replace_multiple(&replacements)
            .expand();
        let overall_factor = prepared.overall_factor.replace_multiple(&replacements);
        timings.polynomial = start.elapsed();

        let start = Instant::now();
        let plan = PolynomialCoefficientPlan::new(expression, variables, parameters)?;
        timings.planning = start.elapsed();
        let plan = if spec.factor_coefficients {
            let start = Instant::now();
            let plan = plan.factor_coefficients()?;
            timings.factoring = start.elapsed();
            plan
        } else {
            plan
        };
        let start = Instant::now();
        // Identical to `TensorCoefficientPlan::with_rank` without mu^2 terms.
        let tensor = TensorCoefficientPlan::with_mu_squared(&plan, spec.tensor_rank)?;
        timings.tensor_plan = start.elapsed();

        let start = Instant::now();
        let coefficients = match spec.backend {
            CoefficientBackend::Interpreter => {
                let evaluator = plan.compile(0.)?;
                timings.evaluator = start.elapsed();
                Coefficients::Interpreter(evaluator)
            }
            #[cfg(feature = "numerator-jit")]
            CoefficientBackend::SymJitO2 => {
                let evaluator = plan.compile_symjit_o2()?;
                let total = start.elapsed();
                timings.evaluator = evaluator.compilation_timings().evaluator.min(total);
                timings.compilation = total - timings.evaluator;
                Coefficients::Jit(evaluator)
            }
            #[cfg(not(feature = "numerator-jit"))]
            CoefficientBackend::SymJitO2 => {
                return Err(CoefficientError::Jit(
                    "SymJIT O2 requires the numerator-jit feature".into(),
                )
                .into());
            }
        };

        let start = Instant::now();
        let loop_name = &prepared.family.loop_momenta()[0];
        let mut shift_coefficients = vec![0.; prepared.quadratics.len() * sources.len()];
        for (row, quadratic) in prepared.quadratics.iter().enumerate() {
            for (momentum, coefficient) in prepared
                .family
                .kinematics()
                .linear_terms(quadratic.momentum())
                .map_err(PreparationError::from)?
            {
                if momentum == *loop_name {
                    continue;
                }
                let column = carriers[..sources.len()]
                    .iter()
                    .position(|carrier| *carrier == momentum)
                    .ok_or_else(|| PreparationError::MissingMomentum(momentum.clone()))?;
                let value: Complex<f64> = coefficient
                    .evaluate_with_prec(&HashMap::<Atom, Complex<f64>>::new(), 53)
                    .map_err(symbolic)?;
                if value.im != 0. || !value.re.is_finite() {
                    return Err(symbolic("shift coefficient is not a finite real number"));
                }
                shift_coefficients[row * sources.len() + column] += value.re;
            }
        }
        let masses: Vec<_> = prepared
            .squared_masses()
            .iter()
            .map(|mass| mass.replace_multiple(&replacements))
            .collect();
        let masses = Atom::evaluator_multiple(&masses, &model_symbols)
            .cores(1)
            .build()
            .map_err(symbolic)?
            .map_coeff_with_prec(
                &|v| Complex::new(0_f64.from_rational(&v.re), 0_f64.from_rational(&v.im)),
                53,
            );
        timings.kinematics = start.elapsed();

        Ok(Self {
            legs,
            sources,
            waves,
            runtime_parameters: spec.runtime_parameters.clone(),
            plan,
            tensor,
            coefficients,
            backend: spec.backend,
            r2: spec.r2,
            propagators: prepared.quadratics.len(),
            shift_coefficients,
            masses,
            model_input: Vec::new(),
            overall_factor,
            timings,
            input: Vec::new(),
            heads: Vec::new(),
        })
    }

    /// Physical legs, the independent external-momentum carriers (first) and
    /// polarization carriers, their legs and vector conjugation flags.
    #[allow(clippy::type_complexity)]
    fn legs(
        prepared: &PreparedIntegrand,
    ) -> Result<(Vec<PhysicalLeg>, Vec<Atom>, Vec<usize>, Vec<(usize, bool)>)> {
        let model = prepared.diagram.model();
        let basis = prepared.diagram.loop_momentum_basis();
        let count = basis.external_edges.len();
        let mut legs: Vec<Option<PhysicalLeg>> = vec![None; count];
        let mut momentum_carriers = Vec::new();
        let mut sources = Vec::new();
        let mut wave_carriers = BTreeMap::new();
        for (position, edge) in basis.external_edges.iter().enumerate() {
            let (_, _, data) = prepared
                .diagram
                .edges()
                .find(|(id, _, _)| id == edge)
                .ok_or(PhysicalNumeratorError::LegNumbering(count))?;
            let external = data
                .external
                .as_ref()
                .ok_or(PhysicalNumeratorError::LegNumbering(count))?;
            let leg = external.index;
            if leg >= count || legs[leg].is_some() {
                return Err(PhysicalNumeratorError::LegNumbering(count));
            }
            let particle = model
                .particle_by_id(data.particle)
                .map_err(|e| PhysicalNumeratorError::Model(e.to_string()))?;
            let incoming = external.state == ExternalState::Incoming;
            let momentum = symbols::external_momentum().call(position);
            if prepared.family.external_momenta().contains(&momentum) {
                momentum_carriers.push(momentum);
                sources.push(leg);
            }
            let state = match particle.spin {
                1 => LegState::Scalar,
                3 => {
                    let found: Vec<_> = [
                        (symbols::epsilon(), LegState::Epsilon),
                        (symbols::epsilonbar(), LegState::EpsilonBar),
                    ]
                    .into_iter()
                    .filter(|(head, _)| {
                        let pattern = head
                            .call((edge.0, Atom::var(symbol!("ninjars_physical::rep_"))))
                            .to_pattern();
                        prepared
                            .numerator
                            .pattern_match(&pattern, None, None)
                            .next()
                            .is_some()
                    })
                    .map(|(head, state)| (head.call(edge.0), state))
                    .collect();
                    if found.len() != 1 {
                        return Err(PhysicalNumeratorError::Carrier {
                            leg,
                            found: found.len(),
                        });
                    }
                    let (carrier, state) = found.into_iter().next().unwrap();
                    wave_carriers.insert(leg, (carrier, state == LegState::EpsilonBar));
                    state
                }
                spin => return Err(PhysicalNumeratorError::UnsupportedLeg { leg, spin }),
            };
            legs[leg] = Some(PhysicalLeg {
                index: leg,
                pdg: particle.pdg_code,
                incoming,
                state,
            });
        }
        let legs = legs
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(PhysicalNumeratorError::LegNumbering(count))?;
        let mut carriers = momentum_carriers;
        let mut waves = Vec::new();
        for (leg, (carrier, bar)) in wave_carriers {
            carriers.push(carrier);
            waves.push((leg, bar));
        }
        Ok((legs, carriers, sources, waves))
    }

    /// Physical legs in row order.
    pub fn legs_info(&self) -> &[PhysicalLeg] {
        &self.legs
    }
    pub fn runtime_parameters(&self) -> &[String] {
        &self.runtime_parameters
    }
    pub fn propagators(&self) -> usize {
        self.propagators
    }
    pub fn rank(&self) -> usize {
        self.tensor.rank()
    }
    /// Highest explicit `mu^2` power; zero for Ninja's plain tensor layout.
    pub fn mu_powers(&self) -> usize {
        self.tensor.mu_powers()
    }
    /// Coefficients per lane, including any `mu^2` blocks.
    pub fn ncoeff(&self) -> usize {
        self.tensor.ncoeff()
    }
    pub fn r2(&self) -> R2Scheme {
        self.r2
    }
    pub fn head_count(&self) -> usize {
        self.tensor.head_count()
    }
    pub fn backend(&self) -> CoefficientBackend {
        self.backend
    }
    pub fn coefficient_plan(&self) -> &PolynomialCoefficientPlan {
        &self.plan
    }
    pub fn tensor_plan(&self) -> &TensorCoefficientPlan {
        &self.tensor
    }
    /// Loop-independent graph factor, already included in the tensor.
    pub fn overall_factor(&self) -> &Atom {
        &self.overall_factor
    }
    pub fn generation_timings(&self) -> &GenerationTimings {
        &self.timings
    }
    /// Evaluator input columns per row (momenta, polarizations, model).
    pub fn input_columns(&self) -> usize {
        self.plan.parameters().len()
    }

    /// Fill `tensor` (`rows x ncoeff`, row-major) for `rows` phase-space
    /// points. `momenta` holds `rows x legs` physical four-momenta, `helicities`
    /// `rows x legs` integers and `model` `rows x runtime_parameters` values.
    /// The whole batch is transactional: on error `tensor` is unchanged.
    pub fn prepare(
        &mut self,
        momenta: &[[f64; 4]],
        helicities: &[i8],
        model: &[f64],
        rows: usize,
        tensor: &mut [Complex<f64>],
    ) -> Result<()> {
        let legs = self.legs.len();
        shape("physical momenta", rows, legs, momenta.len())?;
        shape("physical helicities", rows, legs, helicities.len())?;
        shape(
            "physical model parameters",
            rows,
            self.runtime_parameters.len(),
            model.len(),
        )?;
        shape(
            "tensor coefficients",
            rows,
            self.tensor.ncoeff(),
            tensor.len(),
        )?;
        self.input.clear();
        for row in 0..rows {
            let p = &momenta[row * legs..(row + 1) * legs];
            let h = &helicities[row * legs..(row + 1) * legs];
            let m = &model
                [row * self.runtime_parameters.len()..(row + 1) * self.runtime_parameters.len()];
            if p.iter().flatten().chain(m).any(|x| !x.is_finite()) {
                return Err(PhysicalNumeratorError::NonFinite { row });
            }
            for leg in &self.legs {
                if leg.state == LegState::Scalar && h[leg.index] != 0 {
                    return Err(PhysicalNumeratorError::Helicity {
                        row,
                        leg: leg.index,
                        helicity: h[leg.index],
                    });
                }
            }
            for &leg in &self.sources {
                self.input
                    .extend(p[leg].iter().map(|&x| Complex::new(x, 0.)));
            }
            for &(leg, bar) in &self.waves {
                let wave = vector_wavefunction(&p[leg], h[leg], bar).ok_or(
                    PhysicalNumeratorError::Helicity {
                        row,
                        leg,
                        helicity: h[leg],
                    },
                )?;
                self.input.extend_from_slice(&wave);
            }
            self.input.extend(m.iter().map(|&x| Complex::new(x, 0.)));
        }
        self.heads
            .resize(rows * self.tensor.head_count(), Complex::new(0., 0.));
        self.coefficients
            .evaluate(&self.input, &mut self.heads, rows)?;
        self.tensor
            .scatter(&self.heads, tensor, rows, &Complex::new(0., 0.))?;
        Ok(())
    }

    /// Propagator data for `rows` points: real shifts `V_i` (`rows x n`),
    /// squared masses (`rows x n`) and, optionally, the `rows x n x n`
    /// symmetric matrix `s_ij = (V_i - V_j)^2`. Following MadLoop's
    /// `BUILD_KINEMATIC_MATRIX` (as passed to Ninja, i.e. with the squared
    /// masses added back), an entry within relative `on_shell_threshold`
    /// of `m_i^2` or `m_j^2`, or of zero relative to the squared mean
    /// energy, is set to that exact value (MadLoop's default `OSThres` is 1e-8).
    #[allow(clippy::too_many_arguments)]
    pub fn kinematics(
        &mut self,
        momenta: &[[f64; 4]],
        model: &[f64],
        rows: usize,
        shifts: &mut [[f64; 4]],
        masses_squared: &mut [Complex<f64>],
        s_matrix: Option<&mut [f64]>,
        on_shell_threshold: f64,
    ) -> Result<()> {
        let (legs, n, columns) = (
            self.legs.len(),
            self.propagators,
            self.runtime_parameters.len(),
        );
        shape("physical momenta", rows, legs, momenta.len())?;
        shape("physical model parameters", rows, columns, model.len())?;
        shape("shifts", rows, n, shifts.len())?;
        shape("squared masses", rows, n, masses_squared.len())?;
        if let Some(s) = &s_matrix {
            shape("s matrix", rows, n * n, s.len())?;
        }
        for row in 0..rows {
            if momenta[row * legs..(row + 1) * legs]
                .iter()
                .flatten()
                .chain(&model[row * columns..(row + 1) * columns])
                .any(|x| !x.is_finite())
            {
                return Err(PhysicalNumeratorError::NonFinite { row });
            }
        }
        for row in 0..rows {
            let p = &momenta[row * legs..(row + 1) * legs];
            for (i, shift) in shifts[row * n..(row + 1) * n].iter_mut().enumerate() {
                *shift = [0.; 4];
                for (j, &leg) in self.sources.iter().enumerate() {
                    let c = self.shift_coefficients[i * self.sources.len() + j];
                    if c != 0. {
                        for (s, x) in shift.iter_mut().zip(&p[leg]) {
                            *s += c * x;
                        }
                    }
                }
            }
            self.model_input.clear();
            self.model_input.extend(
                model[row * columns..(row + 1) * columns]
                    .iter()
                    .map(|&x| Complex::new(x, 0.)),
            );
            self.masses.evaluate(
                &self.model_input,
                &mut masses_squared[row * n..(row + 1) * n],
            );
        }
        if let Some(s_matrix) = s_matrix {
            let threshold = on_shell_threshold;
            for row in 0..rows {
                let v = &shifts[row * n..(row + 1) * n];
                let m = &masses_squared[row * n..(row + 1) * n];
                let s = &mut s_matrix[row * n * n..(row + 1) * n * n];
                for i in 0..n {
                    s[i * n + i] = 0.;
                    for j in i + 1..n {
                        let d: [f64; 4] = std::array::from_fn(|k| v[i][k] - v[j][k]);
                        let difference = d[0] * d[0] - d[1] * d[1] - d[2] * d[2] - d[3] * d[3];
                        let mut value = difference;
                        // |difference - m^2| < threshold |m^2|, squared to avoid hypot.
                        for mass in [m[i], m[j]] {
                            let norm = mass.re * mass.re + mass.im * mass.im;
                            let offset = difference - mass.re;
                            if norm != 0.
                                && offset * offset + mass.im * mass.im
                                    < threshold * threshold * norm
                            {
                                value = mass.re;
                            }
                        }
                        let reference = (0.5 * (v[i][0].abs() + v[j][0].abs())).powi(2);
                        if reference != 0. && difference.abs() < threshold * reference {
                            value = 0.;
                        }
                        s[i * n + j] = value;
                        s[j * n + i] = value;
                    }
                }
            }
        }
        Ok(())
    }
}
