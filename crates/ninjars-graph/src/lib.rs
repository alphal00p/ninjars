//! Model-aware input preparation, independent of any integrand reduction engine.
//!
//! DOT parsing, momentum routing, propagator construction and tensor algebra
//! belong to FeynKit, Idenso and Symbolica. This crate only composes those APIs.

mod components;
mod numerator;
mod prepare;

pub use feynkit_graph::symbols;
pub use feynkit_graph::{EdgeId, FeynmanDiagram, IntegralFamily, QuadraticMomentum};
pub use feynkit_kinematics::{FourMomentum, Kinematics};
pub use feynkit_model::Model;
pub use numerator::{NumeratorMode, contract_numerator};
pub use prepare::{PreparationError, PreparedIntegrand, prepare_diagram, prepare_dot};
