//! Batched binary64 fast path for Ninja's integrand reduction.
//!
//! The generic [`crate::Amplitude`] carries arbitrary Symbolica numerical
//! domains (DoubleFloat, arbitrary precision, Dual numbers) through
//! Symbolica-generated kernels. This module is a monomorphic specialization of
//! the same Laurent reduction for ordinary binary64 inputs with real external
//! kinematics and real or complex squared masses. It follows original Ninja's
//! algorithm (Mastrolia, Mirabella, Peraro, arXiv:1203.0291, 1403.1229) cut by
//! cut, while keeping every phase-space lane of a batch together so numerator
//! and master providers receive compact batched requests.
//!
//! Results use Ninja/OneLOop Laurent order `[finite, 1/eps, 1/eps²]`.

pub mod adapters;
pub mod complex;
mod engine;
mod geometry;
pub mod masters;
pub mod numerator;
pub mod replay;
mod subtractions;
pub mod tensor;
mod tensor_jets;
pub mod tensor_layout;
#[cfg(test)]
mod tests;

pub use adapters::{GenericMasters, GenericNumerator};
pub use complex::{C64, CMom, RMom, mp_cc, mp_cr, mp_rr, to_complex};
pub use engine::{FastAmplitude, FastOptions};
pub use masters::{FastMaster, FastMasters};
pub use numerator::{FastNumerator, t2_len, t3_len};
pub use replay::{RecordingMasters, RecordingNumerator, ReplayMasters, ReplayNumerator};
pub use tensor::TensorNumerator;

/// Lane-major numerical inputs of a fast-path evaluation.
///
/// For `lanes` phase-space points and `n` propagators
/// `D_i = (q + V_i)² - m_i² - mu²`:
/// - `shifts[lane * n + i]` is the real momentum `V_i`;
/// - `masses_squared[lane * n + i]` is `m_i²` (imaginary part ≤ 0);
/// - `scales_squared[lane]` is the positive squared renormalization scale;
/// - `s_matrix`, if given, holds `(V_i - V_j)²` at `lane * n * n + i * n + j`
///   (exact invariants that rounded momenta might not reproduce).
#[derive(Clone, Copy, Debug)]
pub struct FastInput<'a> {
    pub lanes: usize,
    pub shifts: &'a [RMom],
    pub masses_squared: &'a [C64],
    pub scales_squared: &'a [f64],
    pub s_matrix: Option<&'a [f64]>,
}

/// Per-lane outcome of a fast-path evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FastStatus {
    NotEvaluated,
    Success,
    /// A cut failed Ninja's stability checks (near-singular Gram determinant,
    /// vanishing uncut denominator expansion, ...). Results are unchanged.
    Unstable,
    /// A master integral failed for this lane. Results are unchanged.
    MasterFailed,
    /// The accumulated result is not finite. Results are unchanged.
    NonFinite,
}

/// Integrated contribution of one lane.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FastResult {
    /// `[finite, 1/eps, 1/eps²]`, including the rational part in `finite`.
    pub laurent: [C64; 3],
    /// Finite cut-constructible part (without the rational term).
    pub cut_constructible: C64,
    /// Finite rational part.
    pub rational: C64,
}
