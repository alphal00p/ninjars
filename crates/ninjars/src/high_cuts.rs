//! Four- and five-cut geometry over the shared momentum type.

use crate::{CutOutcome, FourMomentum, Result, SampleId};

#[derive(Clone, Debug)]
pub struct BoxCut<T> {
    /// Global loop-momentum center, including -V[first].
    /// (a+V[first]).v_orth = 0.
    pub a: FourMomentum<T>,
    /// Orthogonal to every cut-shift difference. Its normalization is explicit
    /// and need not match the reference Ninja spinor chart.
    pub v_orth: FourMomentum<T>,
    pub v_orth_squared: T,
    /// m[first]^2 - (a+V[first])^2.
    pub beta: T,
}

#[derive(Clone, Debug)]
pub struct PentagonCut<T> {
    pub q: FourMomentum<T>,
    pub mu_squared: T,
}

pub struct BoxCutRequest<'a, T> {
    pub id: SampleId,
    pub shifts: [&'a FourMomentum<T>; 4],
    pub masses_squared: [&'a T; 4],
}

pub struct PentagonCutRequest<'a, T> {
    pub id: SampleId,
    pub shifts: [&'a FourMomentum<T>; 5],
    pub masses_squared: [&'a T; 5],
}

pub struct BoxSampleRequest<'a, T> {
    pub id: SampleId,
    pub cut: &'a BoxCut<T>,
    pub mu_squared: &'a T,
}

/// Independent extension of the two/three-cut provider API.
pub trait HighCutProvider<T> {
    fn box_batch(
        &mut self,
        requests: &[BoxCutRequest<'_, T>],
        output: &mut [CutOutcome<BoxCut<T>>],
    ) -> Result<()>;
    fn pentagon_batch(
        &mut self,
        requests: &[PentagonCutRequest<'_, T>],
        output: &mut [CutOutcome<PentagonCut<T>>],
    ) -> Result<()>;
    /// Return q+ then q-, where q±=a±sqrt((beta+mu²)/v_orth²)*v_orth.
    /// Coincident roots are reported as SingularParameter, including for Duals.
    fn sample_box_batch(
        &mut self,
        requests: &[BoxSampleRequest<'_, T>],
        output: &mut [CutOutcome<[FourMomentum<T>; 2]>],
    ) -> Result<()>;
}
