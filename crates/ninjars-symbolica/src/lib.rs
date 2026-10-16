//! Symbolica-backed numerator and master-integral adapters for NinjaRS.
#![forbid(unsafe_code)]

mod contraction;
mod cuts;
mod denominators;
mod dual_polynomial;
mod kernel;
mod laurent;
mod masters;
mod polynomial;
mod projection;
#[cfg(feature = "reference-codegen")]
mod reference;
mod subtraction;
mod triangle_kernels;

pub use contraction::{
    DualMasterContractionEvaluator, MasterContraction, MasterContractionEvaluator,
    MasterContractionInputLayout,
};
pub use cuts::{SymbolicaCuts, SymbolicaDualCuts};
pub use denominators::{DenominatorEvaluator, DenominatorExpansion, DualDenominatorEvaluator};
pub use dual_polynomial::DualPolynomialNumerator;
pub use laurent::{DualLaurentDivisionEvaluator, LaurentDivision, LaurentDivisionEvaluator};
pub use masters::{SymbolicaDualMasters, SymbolicaMasters};
pub use polynomial::PolynomialNumerator;
pub use projection::{
    DualResidueProjectionEvaluator, ResidueProjection, ResidueProjectionEvaluator,
};
pub use subtraction::{
    DualTadpoleSubtractionEvaluator, TadpoleSubtraction, TadpoleSubtractionEvaluator,
    TadpoleSubtractionInputLayout,
};
pub use subtraction::{
    DualTriangleBubbleEvaluator, TriangleBubbleEvaluator, TriangleBubbleInputLayout,
    TriangleBubbleSubtraction,
};
pub use triangle_kernels::{SymbolicaDualTriangleKernels, SymbolicaTriangleKernels};
