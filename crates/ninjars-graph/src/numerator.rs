use feynkit_graph::FeynmanDiagram;
use feynkit_kinematics::Kinematics;
use idenso::tensor::{AlgebraSettings, SymbolicTensor, contract::ContractSettings};
use spenso::structure::partial::PartialStructureExt;
use symbolica::atom::Atom;

use crate::PreparationError;

/// Whether local DOT numerator fragments or the model's Feynman rules are used.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum NumeratorMode {
    /// Preserve explicit graph, vertex and edge `num` attributes.
    #[default]
    Explicit,
    /// Replace local factors with model rules and supply external wavefunctions
    /// when the stored projector is one. Retain any nontrivial supplied projector.
    FeynmanRules,
}

/// Route and contract the numerator, including its prefactor and projector once.
///
/// The diagram's overall factor is returned separately by [`crate::prepare_dot`].
/// Open Lorentz, spinor or color slots must be closed by a supplied projector.
pub fn contract_numerator(
    diagram: &FeynmanDiagram,
    kinematics: &Kinematics,
) -> Result<Atom, PreparationError> {
    let numerator = diagram.numerator() * diagram.numerator_prefactor() * diagram.projector();
    let routed = diagram.loop_momentum_basis().route_expression(&numerator);
    let contracted = SymbolicTensor::infer(routed)?
        .contract(ContractSettings::default())?
        .simplify_algebra(&AlgebraSettings::hep())?
        .to_dots()?;
    if !contracted.structure().logical_slots().is_empty() {
        return Err(PreparationError::OpenNumeratorIndices);
    }
    Ok(kinematics.apply(contracted.expression()))
}
