use feynkit_graph::{DiagramEndpoint, DiagramHalfEdge, FeynmanDiagram};
use feynkit_kinematics::Kinematics;
use idenso::{
    representations::{ColorAdjoint, ColorFundamental, ColorSextet},
    tensor::{AlgebraSettings, SymbolicTensor, contract::ContractSettings},
};
use spenso::{
    network::library::symbolic::ETS,
    structure::{
        partial::{PartialStructure, PartialStructureExt},
        representation::{LibraryRep, RepName},
        slot::{DualSlotTo, IsAbstractSlot},
    },
};
use symbolica::atom::{Atom, AtomCore, AtomView, FunctionBuilder};

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
/// Open Lorentz, spinor or color slots must be closed by a supplied projector,
/// except for one open color pair; see [`contract_numerator_with_color`].
pub fn contract_numerator(
    diagram: &FeynmanDiagram,
    kinematics: &Kinematics,
) -> Result<Atom, PreparationError> {
    contract_numerator_with_color(diagram, kinematics).map(|(numerator, _)| numerator)
}

/// [`contract_numerator`], also returning any automatic color projector.
///
/// When the contracted numerator leaves exactly two open color slots that can
/// be joined and that belong to two distinct colored external legs, i.e. two
/// external gluons or a quark–antiquark pair, it is projected onto the
/// coefficient of their invariant δ, i.e. multiplied by δ/N with N the
/// representation dimension (δab/8, δij/3). This is the projector of the
/// shipped physical examples. Any other open slot, including a dangling index
/// of a hand-written numerator, is an error that names the slots and their
/// external legs.
pub fn contract_numerator_with_color(
    diagram: &FeynmanDiagram,
    kinematics: &Kinematics,
) -> Result<(Atom, Option<Atom>), PreparationError> {
    let numerator = diagram.numerator() * diagram.numerator_prefactor() * diagram.projector();
    let routed = diagram.loop_momentum_basis().route_expression(&numerator);
    let contracted = contract(routed.clone())?;
    let slots = contracted.structure().slots()?;
    if slots.is_empty() {
        return Ok((kinematics.apply(contracted.expression()), None));
    }
    let projector = match slots.as_slice() {
        [first, second]
            if is_color(first.rep_name())
                && first.dual().rep() == second.rep()
                && external_color_pair(diagram, &first.to_atom(), &second.to_atom()) =>
        {
            let delta = FunctionBuilder::new(ETS.metric)
                .add_arg(first.dual().to_atom())
                .add_arg(second.dual().to_atom())
                .finish();
            Some(delta / first.dim().to_symbolic())
        }
        _ => None,
    };
    let Some(projector) = projector else {
        return Err(PreparationError::OpenNumeratorIndices(
            slots
                .iter()
                .map(|slot| describe_slot(diagram, &slot.to_atom()))
                .collect::<Vec<_>>()
                .join(", "),
        ));
    };
    // Contract once more from the routed numerator, exactly as for a projector
    // supplied with the graph, so color algebra sees the closed structure.
    let projected = contract(routed * &projector)?;
    if !projected.structure().slots()?.is_empty() {
        return Err(PreparationError::OpenNumeratorIndices(
            "the automatic color projector left open slots".into(),
        ));
    }
    Ok((kinematics.apply(projected.expression()), Some(projector)))
}

fn contract(expression: Atom) -> Result<SymbolicTensor<PartialStructure>, PreparationError> {
    Ok(SymbolicTensor::infer(expression)?
        .contract(ContractSettings::default())?
        .simplify_algebra(&AlgebraSettings::hep())?
        .to_dots()?)
}

fn is_color(rep: LibraryRep) -> bool {
    [
        ColorAdjoint {}.into(),
        ColorFundamental {}.into(),
        ColorFundamental {}.dual().into(),
        ColorSextet {}.into(),
        ColorSextet {}.dual().into(),
    ]
    .contains(&rep)
}

/// An external leg owning the half-edge of an open slot.
struct SlotLeg {
    index: usize,
    hedge: usize,
    particle: String,
    colored: bool,
}

/// The external legs whose vertex half-edge an open slot refers to.
fn slot_legs(diagram: &FeynmanDiagram, slot: &Atom) -> Vec<SlotLeg> {
    let mut hedges = Vec::new();
    slot.visitor(&mut |part| {
        if let AtomView::Fun(function) = part
            && function.get_symbol() == feynkit_graph::symbols::hedge_index()
            && let Some(hedge) = function
                .iter()
                .next()
                .and_then(|arg| usize::try_from(arg).ok())
        {
            hedges.push(hedge);
        }
        true
    });
    diagram
        .edges()
        .filter_map(|(edge, endpoints, data)| {
            let external = data.external.as_ref()?;
            let endpoint = if endpoints.source.is_some() {
                DiagramEndpoint::Source
            } else {
                DiagramEndpoint::Target
            };
            let hedge = diagram.half_edge_id(DiagramHalfEdge { edge, endpoint })?.0;
            if !hedges.contains(&hedge) {
                return None;
            }
            let particle = diagram.model().particle_by_id(data.particle).ok();
            Some(SlotLeg {
                index: external.index,
                hedge,
                particle: particle.map_or("?".into(), |particle| particle.name.clone()),
                colored: particle.is_some_and(|particle| particle.is_qcd_charged()),
            })
        })
        .collect()
}

/// Whether two open color slots belong to two distinct colored external legs.
fn external_color_pair(diagram: &FeynmanDiagram, first: &Atom, second: &Atom) -> bool {
    match (
        slot_legs(diagram, first).as_slice(),
        slot_legs(diagram, second).as_slice(),
    ) {
        ([a], [b]) => a.index != b.index && a.colored && b.colored,
        _ => false,
    }
}

/// Name an open slot together with the external leg owning its half-edge.
fn describe_slot(diagram: &FeynmanDiagram, slot: &Atom) -> String {
    let legs = slot_legs(diagram, slot)
        .into_iter()
        .map(|leg| {
            format!(
                "external leg {} ({}, half-edge {})",
                leg.index, leg.particle, leg.hedge
            )
        })
        .collect::<Vec<_>>();
    let slot = slot.to_canonical_string();
    if legs.is_empty() {
        slot
    } else {
        format!("{slot} at {}", legs.join(" and "))
    }
}
