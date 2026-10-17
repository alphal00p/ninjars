//! NinjaRS: one-loop integrand reduction using shared FeynKit physics types.
//!
//! All numerical data can carry Symbolica precision and derivative domains.
//! Numerator preparation, graph import, and model algebra live in separate crates.

#![forbid(unsafe_code)]

mod amplitude;
mod batch;
mod boxes;
mod bubbles;
mod cuts;
pub mod fast;
mod high_cuts;
mod integration;
mod kinematics;
mod masters;
mod numerator;
mod pentagons;
mod residue_storage;
mod tadpoles;
mod topology;
mod triangles;

pub use amplitude::{Amplitude, AmplitudeInput, AmplitudeKernels, EvaluationStatus};
pub use batch::{BatchShape, SampleId};
pub use boxes::{
    BoxCoefficientMode, BoxDivisionRequest, BoxMuDivisionRequest, BoxMuMode, BoxProjectionRequest,
    BoxReducer, BoxReductionKernels, BoxReductionOptions, BoxResidue,
};
pub use bubbles::{
    BubbleDivisionRequest, BubbleReducer, BubbleReductionKernels, BubbleResidue,
    BubbleTriangleCorrectionRequest,
};
pub use cuts::{
    BubbleCut, BubbleCutRequest, CutBasis, CutFailure, CutOutcome, CutProvider, TadpoleCut,
    TadpoleCutRequest, TriangleCut, TriangleCutRequest,
};
pub use cuts::{BubbleSampleRequest, CutSampler, TriangleSampleRequest};
pub use feynkit_kinematics::{Energy, FourMomentum, MomentumSignature, ThreeMomentum};
pub use high_cuts::{
    BoxCut, BoxCutRequest, BoxSampleRequest, HighCutProvider, PentagonCut, PentagonCutRequest,
};
pub use integration::{
    IntegratedContribution, IntegrationKernels, IntegrationStatus, ResidueIntegrationRequest,
    ResidueIntegrator,
};
pub use kinematics::{SamplingScaleProvider, SamplingScaleSource, SamplingScaleStatus};
pub use masters::{MasterFailure, MasterIntegral, MasterProvider, MasterRequest, MasterStatus};
pub use numerator::{ExpansionKind, ExpansionRequest, Numerator, NumeratorRequest};
pub use pentagons::{
    PentagonDivisionRequest, PentagonReducer, PentagonReductionKernels, PentagonResidue,
};
pub use symbolica::domains::float::{Complex, DoubleFloat, Float, FloatLike, Real, SingleFloat};
pub use tadpoles::{
    IntegratedTadpoleResidue, TadpoleCorrectionRequest, TadpoleDivisionRequest, TadpoleReducer,
    TadpoleReductionKernels, TadpoleSource,
};
pub use topology::{Cut, Topology};
pub use triangles::{
    TriangleDivisionRequest, TriangleReducer, TriangleReductionKernels, TriangleResidue,
};

/// Laurent coefficients in Ninja/OneLOop order: finite, single pole, double pole.
pub type Laurent<T> = [T; 3];

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{name} has {actual} entries, expected {expected}")]
    Shape {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("a topology must have at least one propagator")]
    NoPropagators,
    #[error("rank {rank} exceeds {propagators} propagators + 1")]
    Rank { rank: usize, propagators: usize },
    #[error("invalid phase-space lane {lane}; batch has {lanes} lanes")]
    Lane { lane: usize, lanes: usize },
    #[error("invalid expansion: {0}")]
    Expansion(String),
    #[error("symbolic preparation failed: {0}")]
    Symbolic(String),
    #[error("unsupported operation: {0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, Error>;
