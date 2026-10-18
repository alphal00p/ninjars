use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use feynkit_generator::{FeynmanDiagramRulesExt, GenerationError};
use feynkit_graph::{
    DiagramError, EdgeId, FeynmanDiagram, IntegralFamily, IntegralFamilyError, QuadraticMomentum,
    symbols,
};
use feynkit_kinematics::{Kinematics, SymbolicKinematicsError};
use feynkit_model::{Model, ModelError};
use idenso::tensor::inference::TensorInferenceError;
use spenso::structure::StructureError;
use symbolica::{
    atom::{Atom, AtomCore},
    symbol,
};
use thiserror::Error;

use crate::{NumeratorMode, connected_components, contract_numerator_with_color};

#[derive(Debug, Error)]
pub enum PreparationError {
    #[error(transparent)]
    Diagram(#[from] DiagramError),
    #[error(transparent)]
    Generation(#[from] GenerationError),
    #[error(transparent)]
    Family(#[from] IntegralFamilyError),
    #[error(transparent)]
    Kinematics(#[from] SymbolicKinematicsError),
    #[error(transparent)]
    Tensor(#[from] TensorInferenceError),
    #[error(transparent)]
    Structure(#[from] StructureError),
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error("one-loop reduction needs exactly one loop; graph contains {0}")]
    LoopCount(usize),
    #[error("graph is disconnected: it has {0} connected components; reduce each one separately")]
    Disconnected(usize),
    #[error(
        "numerator retains open indices {0}; supply an external-state or color projector \
         (the DOT `projector` attribute, written in the half-edge indices listed here), \
         or use --feynman-rules for the model's default external states"
    )]
    OpenNumeratorIndices(String),
    #[error("loop propagator at edge {0:?} is not a single quadratic momentum")]
    NonQuadratic(EdgeId),
    #[error("loop propagators at edges {0:?} and {1:?} are repeated")]
    RepeatedPropagator(EdgeId, EdgeId),
    #[error("loop-independent propagator at edge {0:?} is identically zero")]
    ZeroBridge(EdgeId),
    #[error("no components supplied for routed external momentum {0}")]
    MissingMomentum(Atom),
    #[error("the FDH (R2) loop-momentum binding is not exact for this numerator: it contains {0}")]
    UnsupportedFdh(&'static str),
    #[error("complex-mass propagator at edge {0:?} has no mass-squared term to shift")]
    ComplexMass(EdgeId),
    #[error("invalid graph topology: {0}")]
    Topology(String),
    #[error("spinor chain: {0}")]
    SpinorChain(String),
}

/// Choices made while preparing an integrand, beyond the numerator source.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreparationOptions {
    pub numerator: NumeratorMode,
    /// Replace `m²` by `m² − i·m·Γ` in every propagator denominator whose
    /// particle has a model width, loop and tree alike. Numerators and
    /// couplings keep their real masses: this is not the full complex-mass
    /// scheme, only its propagators.
    pub complex_mass_propagators: bool,
}

impl From<NumeratorMode> for PreparationOptions {
    fn from(numerator: NumeratorMode) -> Self {
        Self {
            numerator,
            ..Self::default()
        }
    }
}

/// Prepared scalar numerator and ordered propagators in their shared types.
///
/// `numerator` includes local factors, the graph numerator prefactor and the
/// external projector. Multiply `overall_factor` exactly once after reduction;
/// it includes graph weights, loop-independent inverse propagators, and the
/// normalization of each quadratic denominator. `family` retains the original
/// denominator normalization; `quadratics` are its exact decompositions.
/// `color_projector` is the automatic projector included in `numerator`, if
/// one open color pair had to be closed.
#[derive(Clone, Debug)]
pub struct PreparedIntegrand {
    pub diagram: FeynmanDiagram,
    pub family: IntegralFamily,
    pub numerator: Atom,
    pub overall_factor: Atom,
    pub propagator_edges: Vec<EdgeId>,
    pub quadratics: Vec<QuadraticMomentum>,
    pub color_projector: Option<Atom>,
}

/// Import either FeynKit's compact or stable DOT dialect without a second parser.
pub fn prepare_dot(
    model: impl Into<Arc<Model>>,
    dot: &str,
    kinematics: &Kinematics,
    options: impl Into<PreparationOptions>,
) -> Result<PreparedIntegrand, PreparationError> {
    prepare_diagram(FeynmanDiagram::from_dot(model, dot)?, kinematics, options)
}

/// Wall-clock durations of the setup stages of one graph preparation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PreparationTimings {
    /// FeynKit DOT import into a `FeynmanDiagram`.
    pub import: Duration,
    /// Model Feynman rules and the default external-state projector; zero in
    /// [`NumeratorMode::Explicit`], where the DOT carries its local factors.
    pub feynman_rules: Duration,
    /// Integral family, ordered propagators and exact quadratic decompositions.
    pub propagators: Duration,
    /// Routing, tensor contraction and algebra simplification of the numerator.
    pub contraction: Duration,
}

/// [`prepare_dot`] with per-stage wall-clock timings; the result is identical.
pub fn prepare_dot_timed(
    model: impl Into<Arc<Model>>,
    dot: &str,
    kinematics: &Kinematics,
    options: impl Into<PreparationOptions>,
) -> Result<(PreparedIntegrand, PreparationTimings), PreparationError> {
    let start = Instant::now();
    let diagram = FeynmanDiagram::from_dot(model, dot)?;
    let import = start.elapsed();
    let (prepared, mut timings) = prepare_diagram_timed(diagram, kinematics, options)?;
    timings.import = import;
    Ok((prepared, timings))
}

pub fn prepare_diagram(
    diagram: FeynmanDiagram,
    kinematics: &Kinematics,
    options: impl Into<PreparationOptions>,
) -> Result<PreparedIntegrand, PreparationError> {
    prepare_diagram_timed(diagram, kinematics, options).map(|(prepared, _)| prepared)
}

/// [`prepare_diagram`] with per-stage wall-clock timings (`import` is zero).
pub fn prepare_diagram_timed(
    mut diagram: FeynmanDiagram,
    kinematics: &Kinematics,
    options: impl Into<PreparationOptions>,
) -> Result<(PreparedIntegrand, PreparationTimings), PreparationError> {
    let options = options.into();
    let mut timings = PreparationTimings::default();
    let components = connected_components(&diagram);
    if components > 1 {
        return Err(PreparationError::Disconnected(components));
    }
    if diagram.loop_count() != 1 {
        return Err(PreparationError::LoopCount(diagram.loop_count()));
    }
    let start = Instant::now();
    if options.numerator == NumeratorMode::FeynmanRules {
        diagram = diagram.apply_feynman_rules()?;
        if diagram.projector().is_one() {
            let projector = diagram.external_state_projector()?;
            diagram = diagram.with_projector(projector);
        }
    }
    timings.feynman_rules = start.elapsed();
    let start = Instant::now();
    let mut ordered = diagram.ordered_propagators(kinematics)?;
    if options.complex_mass_propagators {
        for (edge, denominator) in &mut ordered {
            *denominator = complex_mass_denominator(&diagram, *edge, denominator)?;
        }
    }
    // The same family as `FeynmanDiagram::propagator_family`, from the
    // propagators above rather than a second extraction.
    let basis = diagram.loop_momentum_basis();
    let full_family = IntegralFamily::new(
        (0..basis.loop_edges.len())
            .map(|i| symbols::loop_momentum().call(i))
            .collect(),
        basis
            .external_edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| !basis.dependent_externals.contains(edge))
            .map(|(i, _)| symbols::external_momentum().call(i))
            .collect(),
        ordered
            .iter()
            .map(|(_, denominator)| denominator.clone())
            .collect(),
        kinematics,
    )?;
    let mut overall_factor =
        feynkit_graph::expressions::evaluate_overall_factor(diagram.overall_factor().as_view());
    let mut propagator_edges = Vec::new();
    let mut quadratics = Vec::new();
    let mut denominators = Vec::new();
    let mut normalized = Vec::<Atom>::new();
    for (index, (edge, denominator)) in ordered.into_iter().enumerate() {
        let Some(quadratic) = full_family.quadratic_denominator(index)? else {
            // A bridge has no loop scalar products. Symbolica's exact tree
            // containment check covers the family's explicit dot notation.
            let loop_momenta = full_family.loop_momenta();
            let depends_on_loop = loop_momenta.iter().any(|p| denominator.contains(p));
            if depends_on_loop {
                return Err(PreparationError::NonQuadratic(edge));
            }
            if denominator.is_zero() {
                return Err(PreparationError::ZeroBridge(edge));
            }
            overall_factor /= denominator;
            continue;
        };
        let normalized_denominator = (&denominator / quadratic.scale()).together();
        for (previous, previous_edge) in normalized.iter().zip(&propagator_edges) {
            if (&normalized_denominator - previous).together().is_zero() {
                return Err(PreparationError::RepeatedPropagator(*previous_edge, edge));
            }
        }
        // Verify the shared decomposition can be resolved in the routed basis.
        full_family
            .kinematics()
            .linear_terms(quadratic.momentum())?;
        overall_factor /= quadratic.scale();
        normalized.push(normalized_denominator);
        propagator_edges.push(edge);
        denominators.push(denominator);
        quadratics.push(quadratic);
    }
    let family = IntegralFamily::new(
        full_family.loop_momenta().to_vec(),
        full_family.external_momenta().to_vec(),
        denominators,
        full_family.kinematics(),
    )?;
    timings.propagators = start.elapsed();
    let start = Instant::now();
    let (numerator, color_projector) =
        contract_numerator_with_color(&diagram, family.kinematics())?;
    timings.contraction = start.elapsed();
    Ok((
        PreparedIntegrand {
            diagram,
            family,
            numerator,
            overall_factor: overall_factor.together(),
            propagator_edges,
            quadratics,
            color_projector,
        },
        timings,
    ))
}

/// Mass and width symbols of an edge's particle, `None` without a width.
pub fn edge_width(
    diagram: &FeynmanDiagram,
    edge: EdgeId,
) -> Result<Option<(Atom, Atom)>, PreparationError> {
    let model = diagram.model();
    let (_, _, data) = diagram
        .edges()
        .find(|(candidate, _, _)| *candidate == edge)
        .ok_or_else(|| PreparationError::Topology(format!("edge {} does not exist", edge.0)))?;
    let particle = model.particle_by_id(data.particle)?;
    let width = model.parameter_by_id(particle.width)?;
    if width.name == "ZERO" || particle.is_massless(model) {
        return Ok(None);
    }
    Ok(Some((
        particle.symbolic_mass(model),
        Atom::var(symbol!(&format!("UFO::{}", width.name))),
    )))
}

fn complex_mass_denominator(
    diagram: &FeynmanDiagram,
    edge: EdgeId,
    denominator: &Atom,
) -> Result<Atom, PreparationError> {
    let Some((mass, width)) = edge_width(diagram, edge)? else {
        return Ok(denominator.clone());
    };
    let squared = mass.pow(2);
    let shifted = denominator
        .replace(squared.to_pattern())
        .with((&squared - Atom::i() * &mass * &width).to_pattern());
    if shifted == *denominator {
        return Err(PreparationError::ComplexMass(edge));
    }
    Ok(shifted)
}
