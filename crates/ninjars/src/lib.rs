//! NinjaRS: one-loop integrand reduction using shared FeynKit physics types.
//!
//! All numerical data can carry Symbolica precision and derivative domains.
//! Numerator preparation, graph import, and model algebra live in separate crates.

#![forbid(unsafe_code)]

mod batch;
mod cuts;
mod masters;
mod numerator;
mod topology;
mod triangles;

pub use batch::{BatchShape, SampleId};
pub use cuts::{
    BubbleCut, BubbleCutRequest, CutBasis, CutFailure, CutOutcome, CutProvider, TadpoleCut,
    TadpoleCutRequest, TriangleCut, TriangleCutRequest,
};
pub use cuts::{BubbleSampleRequest, CutSampler, TriangleSampleRequest};
pub use feynkit_kinematics::{Energy, FourMomentum, MomentumSignature, ThreeMomentum};
pub use masters::{MasterIntegral, MasterProvider, MasterRequest};
pub use numerator::{ExpansionKind, ExpansionRequest, Numerator, NumeratorRequest};
pub use symbolica::domains::float::{Complex, DoubleFloat, Float, FloatLike, Real, SingleFloat};
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
