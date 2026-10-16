//! Physical graph preparation using the existing FeynKit and Idenso APIs.

use std::sync::Arc;

use feynkit_amplitude::{Amplitude, AmplitudeError, ColorSum, ColorSumError};
use feynkit_generator::{
    GenerationError, GenerationFilter, GenerationOptions, NumeratorGrouping, Process,
};
use feynkit_kinematics::Kinematics;
use feynkit_model::{Model, ModelError};
use idenso::{
    color::ColorSimplifySettings,
    tensor::{AlgebraSettings, SymbolicTensor},
};
use spenso::structure::dimension::Dimension;
use thiserror::Error;

use crate::{NumeratorMode, PreparationError, PreparedIntegrand, prepare_diagram};

#[derive(Debug, Error)]
pub enum PhysicalPreparationError {
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error(transparent)]
    Generation(#[from] GenerationError),
    #[error(transparent)]
    Amplitude(#[from] AmplitudeError),
    #[error(transparent)]
    Color(#[from] ColorSumError),
    #[error(transparent)]
    Preparation(#[from] Box<PreparationError>),
    #[error("physical four-dimensional numerator preparation requires dimension four")]
    Dimension,
    #[error("expected one top-quark {0} vertex; found {1}")]
    Vertex(&'static str, usize),
    #[error("expected two top-quark Higgs-to-gluon triangle orientations; found {0}")]
    Orientations(usize),
    #[error("generated Higgs-to-gluon graph has incompatible external states")]
    ExternalStates,
}

/// Generate and prepare the two oriented top-quark `H -> g g` triangles.
///
/// The numerator uses four-dimensional model Feynman rules from the outset.
/// The external gluon color tensor is projected with `delta_ab / 8`, so the
/// result is the coefficient of `delta_ab`, with no spin sum or helicity
/// average. The graph sign, numerator prefactor and color projector are each
/// retained exactly once through [`PreparedIntegrand`]. No R2 or UV graphs are
/// generated. The reduction engine supplies R1 through its denominators.
///
/// External `epsilonbar(edge)` carriers remain symbolic. Bind their components
/// with the shared wavefunction API and [`PreparedIntegrand::component_numerator`]
/// before reduction. The graph's external leg labels and routing remain
/// available in its existing FeynKit metadata. Couplings and masses also remain
/// symbolic; use [`Model::scalar_bindings`] with exact overrides, including zero
/// widths for the physical reference convention. The top Yukawa parameter
/// `ymt` and propagator mass `MT` are independent model inputs.
pub fn prepare_top_higgs_gluons(
    model: impl Into<Arc<Model>>,
    kinematics: &Kinematics,
) -> Result<Vec<PreparedIntegrand>, PhysicalPreparationError> {
    if kinematics.dimension() != Dimension::Concrete(4) {
        return Err(PhysicalPreparationError::Dimension);
    }
    let model = model.into();
    let top = model.particle_id_by_pdg(6)?;
    let antitop = model.particle_id_by_pdg(-6)?;
    let higgs = model.particle_id_by_pdg(25)?;
    let gluon = model.particle_id_by_pdg(21)?;
    let mut vertices = Vec::new();
    for (name, third) in [("Yukawa", higgs), ("gluon", gluon)] {
        let mut required = vec![top, antitop, third];
        required.sort();
        let selected: Vec<_> = model
            .vertex_rules()
            .iter()
            .filter(|vertex| {
                let mut particles = vertex.particles.clone();
                particles.sort();
                particles == required
            })
            .map(|vertex| vertex.name.clone().into())
            .collect();
        if selected.len() != 1 {
            return Err(PhysicalPreparationError::Vertex(name, selected.len()));
        }
        vertices.extend(selected);
    }
    let options = GenerationOptions::default()
        .with_loop_count(1, 1)
        .expect("the fixed one-loop range is valid")
        .threads(1)
        .max_vertices(3)
        .numerator_grouping(NumeratorGrouping::None)
        .with_graph_filter(GenerationFilter::MaxNumberOfBridges(0))
        .with_graph_filter(GenerationFilter::VertexAllow(vertices));
    let process = Process::new(
        [model.particle_by_id(higgs)?.name.clone()],
        [
            model.particle_by_id(gluon)?.name.clone(),
            model.particle_by_id(gluon)?.name.clone(),
        ],
    );
    let generated = process.generate_diagrams(model.clone(), &options)?;
    if generated.diagrams.len() != 2 {
        return Err(PhysicalPreparationError::Orientations(
            generated.diagrams.len(),
        ));
    }
    let color = ColorSum::new(model.particle_by_id(gluon)?)?.averaged(true);
    generated
        .diagrams
        .into_iter()
        .map(|diagram| {
            // Reuse FeynKit's physical-leg port identification. Its complete
            // amplitude expression is not used as a reduction numerator.
            let amplitude = Amplitude::from_diagram(diagram.clone())?;
            let gluons: Vec<_> = amplitude
                .legs()
                .iter()
                .filter(|leg| leg.particle == gluon)
                .collect();
            if gluons.len() != 2 {
                return Err(PhysicalPreparationError::ExternalStates);
            }
            let projector = diagram.projector()
                * color.expression([gluons[0].index_atom(), gluons[1].index_atom()]);
            let mut prepared = prepare_diagram(
                diagram.with_projector(projector),
                kinematics,
                NumeratorMode::Explicit,
            )
            .map_err(Box::new)?;
            // Use the model's SU(3) fundamental normalization, including T_R=1/2.
            // The general preparation API deliberately retains symbolic invariants.
            prepared.numerator = SymbolicTensor::infer(prepared.numerator)
                .and_then(|tensor| {
                    tensor.simplify_algebra(&AlgebraSettings {
                        color: Some(
                            ColorSimplifySettings::default().with_cof_dimension_invariants(),
                        ),
                        ..AlgebraSettings::default()
                    })
                })
                .map_err(PreparationError::from)
                .map_err(Box::new)?
                .into_expression();
            Ok(prepared)
        })
        .collect()
}
