//! Model-aware input preparation, independent of any integrand reduction engine.
//!
//! DOT parsing, momentum routing, propagator construction and tensor algebra
//! belong to FeynKit, Idenso and Symbolica. This crate only composes those APIs.
//! The only item used from the `ninjars` crate is the tensor-numerator monomial
//! ordering shared by original Ninja and MadLoop (`fast::tensor_layout`).

mod coefficients;
mod components;
mod numerator;
mod physical;
mod prepare;
mod tensor;

pub use feynkit_graph::symbols;
pub use feynkit_graph::{EdgeId, FeynmanDiagram, IntegralFamily, QuadraticMomentum};
pub use feynkit_kinematics::{FourMomentum, Kinematics};
pub use feynkit_model::Model;
pub use numerator::{NumeratorMode, contract_numerator};
pub use physical::{PhysicalPreparationError, prepare_top_four_photons, prepare_top_higgs_gluons};
pub use prepare::{PreparationError, PreparedIntegrand, prepare_diagram, prepare_dot};

#[cfg(feature = "numerator-jit")]
pub use coefficients::SymJitCoefficientEvaluator;
pub use coefficients::{CoefficientError, CoefficientEvaluator, PolynomialCoefficientPlan};
pub use tensor::{TensorCoefficientPlan, TensorEntry};
