//! Symbolica-backed numerator and master-integral adapters for NinjaRS.
#![forbid(unsafe_code)]

mod amplitude;
mod box_kernels;
mod bubble_kernels;
mod contraction;
mod cuts;
mod denominators;
mod dual_polynomial;
mod high_cuts;
mod integration;
mod kernel;
mod laurent;
mod masters;
mod pentagon_kernels;
mod polynomial;
mod projection;
#[cfg(feature = "reference-codegen")]
mod reference;
mod subtraction;
mod tadpole_kernels;
mod triangle_kernels;

pub use amplitude::{SymbolicaAmplitudeKernels, SymbolicaDualAmplitudeKernels};
pub use box_kernels::{SymbolicaBoxKernels, SymbolicaDualBoxKernels};
pub use bubble_kernels::{SymbolicaBubbleKernels, SymbolicaDualBubbleKernels};
pub use contraction::{
    DualMasterContractionEvaluator, MasterContraction, MasterContractionEvaluator,
    MasterContractionInputLayout,
};
pub use cuts::{SymbolicaCuts, SymbolicaDualCuts};
pub use denominators::{DenominatorEvaluator, DenominatorExpansion, DualDenominatorEvaluator};
pub use dual_polynomial::DualPolynomialNumerator;
pub use high_cuts::{SymbolicaDualHighCuts, SymbolicaHighCuts};
pub use integration::{SymbolicaDualIntegrationKernels, SymbolicaIntegrationKernels};
pub use laurent::{DualLaurentDivisionEvaluator, LaurentDivision, LaurentDivisionEvaluator};
pub use masters::{SymbolicaDualMasters, SymbolicaMasters};
pub use pentagon_kernels::{SymbolicaDualPentagonKernels, SymbolicaPentagonKernels};
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
pub use tadpole_kernels::{SymbolicaDualTadpoleKernels, SymbolicaTadpoleKernels};
pub use triangle_kernels::{SymbolicaDualTriangleKernels, SymbolicaTriangleKernels};
