//! Lane-local selection of an auxiliary scale for finite-mu box sampling.

use crate::{CutFailure, FourMomentum, Result};

/// First maximum in propagator order, followed by lexicographic invariant order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingScaleSource {
    Mass {
        propagator: usize,
    },
    Invariant {
        first: usize,
        second: usize,
    },
    /// Every mass and pairwise invariant has zero primal magnitude.
    UnitFallback,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingScaleStatus {
    Selected(SamplingScaleSource),
    InvalidInput(CutFailure),
}

/// Select max(|m_i²|, |(V_i-V_j)²|), or one when every candidate is zero.
///
/// Shifts and squared masses are lane-major. The scale is auxiliary and distinct
/// from the squared renormalization scale. Selection uses primal values; Dual
/// outputs retain their shape with all derivative components zero. Invalid
/// numerical lanes leave their scales unchanged. Structural errors leave all
/// caller buffers unchanged. No accuracy guarantee follows from scale selection.
pub trait SamplingScaleProvider<T> {
    fn propagators(&self) -> usize;
    fn evaluate_batch(
        &mut self,
        shifts: &[FourMomentum<T>],
        masses_squared: &[T],
        lanes: usize,
        scales: &mut [T],
        statuses: &mut [SamplingScaleStatus],
    ) -> Result<()>;
}
