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
    #[error("expected six oriented top-quark four-photon boxes; found {0}")]
    PhotonBoxes(usize),
    #[error("generated Higgs-to-gluon graph has incompatible external states")]
    ExternalStates,
    #[error(
        "expected a gluon loop, a ghost loop and one loop per quark flavor ({expected} graphs); found {found}"
    )]
    SelfEnergies { expected: usize, found: usize },
}

/// Prepare the six oriented top-quark boxes for `gamma gamma -> gamma gamma`.
///
/// Only the model's top-antitop-photon vertex is admitted. FeynKit supplies the
/// graph sign, four-dimensional Feynman rules, momentum routing and the four
/// external vector carriers. No external-color projector, spin sum, helicity
/// average or identical-particle phase-space factor is applied. Idenso resolves
/// the closed fundamental-color trace using its existing dimension identities.
/// Thus the fermion-loop sign and internal color multiplicity occur exactly
/// once, respectively in `overall_factor` and the contracted numerator.
///
/// Incoming `epsilon(edge)` and outgoing `epsilonbar(edge)` carriers remain
/// symbolic. Bind their components through the shared component interface.
/// Parameters also remain exact symbolic model inputs. This construction has
/// no explicit mu-squared numerator, R2 vertex or UV counterterm; R1 is supplied
/// by the reduction of the denominators. Graph ordering is generator metadata,
/// not a promised correspondence to another program's diagram numbering.
pub fn prepare_top_four_photons(
    model: impl Into<Arc<Model>>,
    kinematics: &Kinematics,
) -> Result<Vec<PreparedIntegrand>, PhysicalPreparationError> {
    if kinematics.dimension() != Dimension::Concrete(4) {
        return Err(PhysicalPreparationError::Dimension);
    }
    let model = model.into();
    let top = model.particle_id_by_pdg(6)?;
    let antitop = model.particle_id_by_pdg(-6)?;
    let photon = model.particle_id_by_pdg(22)?;
    let mut required = vec![top, antitop, photon];
    required.sort();
    let vertices: Vec<_> = model
        .vertex_rules()
        .iter()
        .filter(|vertex| {
            let mut particles = vertex.particles.clone();
            particles.sort();
            particles == required
        })
        .map(|vertex| vertex.name.clone().into())
        .collect();
    if vertices.len() != 1 {
        return Err(PhysicalPreparationError::Vertex("photon", vertices.len()));
    }
    let options = GenerationOptions::default()
        .with_loop_count(1, 1)
        .expect("the fixed one-loop range is valid")
        .threads(1)
        .max_vertices(4)
        .numerator_grouping(NumeratorGrouping::None)
        .with_graph_filter(GenerationFilter::MaxNumberOfBridges(0))
        .with_graph_filter(GenerationFilter::VertexAllow(vertices));
    let name = model.particle_by_id(photon)?.name.clone();
    let process = Process::new([name.clone(), name.clone()], [name.clone(), name]);
    let generated = process.generate_diagrams(model, &options)?;
    if generated.diagrams.len() != 6 {
        return Err(PhysicalPreparationError::PhotonBoxes(
            generated.diagrams.len(),
        ));
    }
    generated
        .diagrams
        .into_iter()
        .map(|diagram| {
            prepare_diagram(diagram, kinematics, NumeratorMode::Explicit)
                .and_then(PreparedIntegrand::with_color_dimension_invariants)
                .map_err(Box::new)
                .map_err(PhysicalPreparationError::from)
        })
        .collect()
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
            prepare_diagram(
                diagram.with_projector(projector),
                kinematics,
                NumeratorMode::Explicit,
            )
            .and_then(PreparedIntegrand::with_color_dimension_invariants)
            .map_err(Box::new)
            .map_err(PhysicalPreparationError::from)
        })
        .collect()
}

/// Generate and prepare the one-loop QCD gluon self-energy `g -> g`.
///
/// Admits the model's ghost-ghost-gluon, three- and four-gluon vertices and
/// the quark-gluon vertex of every flavor in `quark_pdgs` (positive PDG codes).
/// FeynKit supplies the gluon-loop symmetry factor `1/2`, the ghost- and
/// quark-loop signs `-1`, Feynman-gauge rules and routing; the massless
/// four-gluon tadpole (a self-loop, zero in dimensional regularization) is not
/// generated. As for `H -> g g`, the external color tensor is projected with
/// `delta_ab / 8` (the coefficient of `delta_ab`), there is no spin sum, and
/// incoming `epsilon(edge)` / outgoing `epsilonbar(edge)` carriers remain
/// symbolic. The sum of all graphs is transverse, `p_mu Pi^{mu nu} = 0`, for
/// a d-dimensional loop momentum (FDH: [`crate::R2Scheme::Fdh`]); with a
/// four-dimensional numerator the missing `mu^2` terms (R2) break it.
pub fn prepare_gluon_self_energy(
    model: impl Into<Arc<Model>>,
    kinematics: &Kinematics,
    quark_pdgs: &[i64],
) -> Result<Vec<PreparedIntegrand>, PhysicalPreparationError> {
    if kinematics.dimension() != Dimension::Concrete(4) {
        return Err(PhysicalPreparationError::Dimension);
    }
    let model = model.into();
    let gluon = model.particle_id_by_pdg(21)?;
    let name = model.particle_by_id(gluon)?.name.clone();
    let ghost = model.particle_id_by_pdg(9000005)?;
    let antighost = model.particle_id_by_pdg(-9000005)?;
    let mut allowed: Vec<Vec<_>> = vec![
        vec![ghost, antighost, gluon],
        vec![gluon; 3],
        vec![gluon; 4],
    ];
    for &pdg in quark_pdgs {
        allowed.push(vec![
            model.particle_id_by_pdg(pdg)?,
            model.particle_id_by_pdg(-pdg)?,
            gluon,
        ]);
    }
    let mut vertices = Vec::new();
    for required in &mut allowed {
        required.sort();
        let selected: Vec<_> = model
            .vertex_rules()
            .iter()
            .filter(|vertex| {
                let mut particles = vertex.particles.clone();
                particles.sort();
                particles == *required
            })
            .map(|vertex| vertex.name.clone().into())
            .collect();
        if selected.len() != 1 {
            return Err(PhysicalPreparationError::Vertex("QCD", selected.len()));
        }
        vertices.extend(selected);
    }
    let options = GenerationOptions::default()
        .with_loop_count(1, 1)
        .expect("the fixed one-loop range is valid")
        .threads(1)
        .numerator_grouping(NumeratorGrouping::None)
        .with_graph_filter(GenerationFilter::MaxNumberOfBridges(0))
        .with_graph_filter(GenerationFilter::VertexAllow(vertices));
    let process = Process::new([name.clone()], [name]);
    let generated = process.generate_diagrams(model.clone(), &options)?;
    let expected = 2 + quark_pdgs.len();
    if generated.diagrams.len() != expected {
        return Err(PhysicalPreparationError::SelfEnergies {
            expected,
            found: generated.diagrams.len(),
        });
    }
    let color = ColorSum::new(model.particle_by_id(gluon)?)?.averaged(true);
    generated
        .diagrams
        .into_iter()
        .map(|diagram| {
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
            prepare_diagram(
                diagram.with_projector(projector),
                kinematics,
                NumeratorMode::Explicit,
            )
            .and_then(PreparedIntegrand::with_color_dimension_invariants)
            .map_err(Box::new)
            .map_err(PhysicalPreparationError::from)
        })
        .collect()
}

impl PreparedIntegrand {
    /// Resolve supported fundamental-color invariants using Idenso's existing
    /// dimension formulas. This is an explicit preparation policy; the default
    /// graph preparation keeps arbitrary color invariants symbolic.
    pub fn with_color_dimension_invariants(mut self) -> Result<Self, PreparationError> {
        self.numerator = SymbolicTensor::infer(self.numerator)?
            .simplify_algebra(&AlgebraSettings {
                color: Some(ColorSimplifySettings::default().with_cof_dimension_invariants()),
                ..AlgebraSettings::default()
            })?
            .into_expression();
        Ok(self)
    }
}
