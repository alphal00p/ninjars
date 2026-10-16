//! Symbolica-backed numerator and master-integral adapters for NinjaRS.
#![forbid(unsafe_code)]

mod dual_polynomial;
mod masters;
mod polynomial;
#[cfg(feature = "reference-codegen")]
mod reference;

pub use dual_polynomial::DualPolynomialNumerator;
pub use masters::{SymbolicaDualMasters, SymbolicaMasters};
pub use polynomial::PolynomialNumerator;
