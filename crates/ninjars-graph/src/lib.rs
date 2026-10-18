//! Model-aware input preparation, independent of any integrand reduction engine.
//!
//! DOT parsing, momentum routing, propagator construction and tensor algebra
//! belong to FeynKit, Idenso and Symbolica. This crate only composes those APIs.
//! The only item used from the `ninjars` crate is the tensor-numerator monomial
//! ordering shared by original Ninja and MadLoop (`fast::tensor_layout`).

mod coefficients;
mod components;
mod graph_factors;
mod numerator;
mod physical;
mod physical_numerator;
mod prepare;
mod spinors;
mod tensor;

pub use components::R2Scheme;
pub use feynkit_graph::symbols;
pub use feynkit_graph::{EdgeId, ExternalState, FeynmanDiagram, IntegralFamily, QuadraticMomentum};
pub use feynkit_kinematics::{FourMomentum, Kinematics};
pub use feynkit_model::Model;
pub use graph_factors::{GraphFactors, connected_components, graph_factors};
pub use numerator::{NumeratorMode, contract_numerator, contract_numerator_with_color};
pub use physical::{
    PhysicalPreparationError, prepare_gluon_self_energy, prepare_top_four_photons,
    prepare_top_higgs_gluons,
};
pub use physical_numerator::{
    CoefficientBackend, GenerationTimings, LegState, PhysicalLeg, PhysicalNumeratorError,
    PhysicalNumeratorSpec, PhysicalTensorNumerator, vector_wavefunction,
};
pub use prepare::{
    PreparationError, PreparationOptions, PreparationTimings, PreparedIntegrand, edge_width,
    prepare_diagram, prepare_diagram_timed, prepare_dot, prepare_dot_timed,
};
pub use spinors::{expand_spinor_chains, has_spinor_chains};

pub use coefficients::{CoefficientError, CoefficientEvaluator, PolynomialCoefficientPlan};
#[cfg(feature = "numerator-jit")]
pub use coefficients::{SymJitCoefficientEvaluator, SymJitCompilationTimings};
pub use tensor::{TensorCoefficientPlan, TensorEntry};
