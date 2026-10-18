//! Symbolica-backed numerator and master-integral adapters for NinjaRS.
#![forbid(unsafe_code)]

#[cfg(all(test, feature = "rust-codegen"))]
mod algorithm_transpose_probe;
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
mod kinematics;
mod laurent;
mod masters;
mod oneloop_native;
pub use oneloop_native::OneLoopNativeMasters;
#[cfg(feature = "native-kernels")]
mod native_kernels;
#[cfg(feature = "native-kernels")]
pub use native_kernels::{KernelBackend, KernelBackendSelection, KernelFamily};
#[cfg(feature = "native-kernels")]
pub use oneloop::rust_codegen::runtime::BatchExecutionPolicy;
mod pentagon_kernels;
#[cfg(test)]
mod plan_fingerprints_tests;
mod polynomial;
mod projection;
mod rank_specialization;
#[cfg(feature = "reference-codegen")]
mod reference;
#[cfg(feature = "rust-codegen")]
pub mod rust_codegen;
mod subtraction;
mod tadpole_kernels;
mod tadpole_plans;
mod triangle_kernels;
mod value_checks;

pub use amplitude::{SymbolicaAmplitudeKernels, SymbolicaDualAmplitudeKernels};
pub use box_kernels::{SymbolicaBoxKernels, SymbolicaDualBoxKernels};
pub use bubble_kernels::{
    PrunedSwappedArithmetic, SymbolicaBubbleKernels, SymbolicaDualBubbleKernels,
};
pub use contraction::{
    DualMasterContractionEvaluator, MasterContraction, MasterContractionEvaluator,
    MasterContractionInputLayout,
};
pub use cuts::{SymbolicaCuts, SymbolicaDualCuts};
pub use denominators::{DenominatorEvaluator, DenominatorExpansion, DualDenominatorEvaluator};
pub use dual_polynomial::DualPolynomialNumerator;
pub use high_cuts::{SymbolicaDualHighCuts, SymbolicaHighCuts};
pub use integration::{SymbolicaDualIntegrationKernels, SymbolicaIntegrationKernels};
pub use kinematics::{SymbolicaDualSamplingScales, SymbolicaSamplingScales};
pub use laurent::{DualLaurentDivisionEvaluator, LaurentDivision, LaurentDivisionEvaluator};
pub use masters::{SymbolicaDualMasters, SymbolicaMasters};
pub use pentagon_kernels::{SymbolicaDualPentagonKernels, SymbolicaPentagonKernels};
pub use polynomial::PolynomialNumerator;
pub use projection::{
    BubbleSourceProjection, BubbleSourceProjectionEvaluator, DualBubbleSourceProjectionEvaluator,
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
pub use tadpole_plans::{
    DualTadpolePlanEvaluator, TadpoleGeometryProbe, TadpolePlanEvaluator, TadpoleSourceProjection,
};
pub use triangle_kernels::{SymbolicaDualTriangleKernels, SymbolicaTriangleKernels};
