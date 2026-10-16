use std::sync::Arc;

use feynkit_generator::{FeynmanDiagramRulesExt, GenerationError};
use feynkit_graph::{
    DiagramError, EdgeId, FeynmanDiagram, IntegralFamily, IntegralFamilyError, QuadraticMomentum,
};
use feynkit_kinematics::{Kinematics, SymbolicKinematicsError};
use feynkit_model::Model;
use idenso::tensor::inference::TensorInferenceError;
use symbolica::atom::{Atom, AtomCore};
use thiserror::Error;

use crate::{NumeratorMode, contract_numerator};

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
    #[error("one-loop reduction needs exactly one loop; graph contains {0}")]
    LoopCount(usize),
    #[error("numerator retains open indices; supply an external-state/color projector")]
    OpenNumeratorIndices,
    #[error("loop propagator at edge {0:?} is not a single quadratic momentum")]
    NonQuadratic(EdgeId),
    #[error("loop propagators at edges {0:?} and {1:?} are repeated")]
    RepeatedPropagator(EdgeId, EdgeId),
    #[error("loop-independent propagator at edge {0:?} is identically zero")]
    ZeroBridge(EdgeId),
    #[error("no components supplied for routed external momentum {0}")]
    MissingMomentum(Atom),
}

/// Prepared scalar numerator and ordered propagators in their shared types.
///
/// `numerator` includes local factors, the graph numerator prefactor and the
/// external projector. Multiply `overall_factor` exactly once after reduction;
/// it includes graph weights, loop-independent inverse propagators, and the
/// normalization of each quadratic denominator. `family` retains the original
/// denominator normalization; `quadratics` are its exact decompositions.
#[derive(Clone, Debug)]
pub struct PreparedIntegrand {
    pub diagram: FeynmanDiagram,
    pub family: IntegralFamily,
    pub numerator: Atom,
    pub overall_factor: Atom,
    pub propagator_edges: Vec<EdgeId>,
    pub quadratics: Vec<QuadraticMomentum>,
}

/// Import either FeynKit's compact or stable DOT dialect without a second parser.
pub fn prepare_dot(
    model: impl Into<Arc<Model>>,
    dot: &str,
    kinematics: &Kinematics,
    mode: NumeratorMode,
) -> Result<PreparedIntegrand, PreparationError> {
    prepare_diagram(FeynmanDiagram::from_dot(model, dot)?, kinematics, mode)
}

pub fn prepare_diagram(
    mut diagram: FeynmanDiagram,
    kinematics: &Kinematics,
    mode: NumeratorMode,
) -> Result<PreparedIntegrand, PreparationError> {
    if diagram.loop_count() != 1 {
        return Err(PreparationError::LoopCount(diagram.loop_count()));
    }
    if mode == NumeratorMode::FeynmanRules {
        diagram = diagram.apply_feynman_rules()?;
        if diagram.projector().is_one() {
            let projector = diagram.external_state_projector()?;
            diagram = diagram.with_projector(projector);
        }
    }
    let full_family = diagram.propagator_family(kinematics)?;
    let ordered = diagram.ordered_propagators(kinematics)?;
    let mut overall_factor = diagram.overall_factor().clone();
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
    let numerator = contract_numerator(&diagram, family.kinematics())?;
    Ok(PreparedIntegrand {
        diagram,
        family,
        numerator,
        overall_factor: overall_factor.together(),
        propagator_edges,
        quadratics,
    })
}
