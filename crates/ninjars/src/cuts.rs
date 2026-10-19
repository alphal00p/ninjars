//! Reduction-specific cut geometry over shared FeynKit momenta.

use crate::{FourMomentum, RankClass, Result, SampleId};

#[derive(Clone, Debug)]
pub struct CutBasis<T> {
    pub e1: FourMomentum<T>,
    pub e2: FourMomentum<T>,
    pub e3: FourMomentum<T>,
    pub e4: FourMomentum<T>,
    /// Bilinear Minkowski product e1.e2; e3.e4 = -e12.
    pub e12: T,
}

#[derive(Clone, Debug)]
pub struct TriangleCut<T> {
    pub basis: CutBasis<T>,
    /// Global loop momentum offset, including -V[first cut propagator].
    pub a: FourMomentum<T>,
    /// e4 / (2 e3.e4), so e3.e4_effective = 1/2.
    pub e4_effective: FourMomentum<T>,
    /// q = a + t*e3 + (beta + mu_squared)*e4_effective/t.
    pub beta: T,
}

#[derive(Clone, Debug)]
pub struct BubbleCut<T> {
    pub basis: CutBasis<T>,
    pub a0: FourMomentum<T>,
    pub a1: FourMomentum<T>,
    /// e4 / (2 e3.e4), so e3.e4_effective = 1/2.
    pub e4_effective: FourMomentum<T>,
    /// q = a0 + x*a1 + t*e3 +
    /// (parameters[0] + x*parameters[1] + x^2*parameters[2] + mu_squared)*e4_effective/t.
    pub parameters: [T; 3],
}

#[derive(Clone, Debug)]
pub struct TadpoleCut<T> {
    pub basis: CutBasis<T>,
    /// Global loop momentum offset -V[cut propagator].
    pub a: FourMomentum<T>,
    pub beta: T,
    /// e2 / (2 e1.e2), for the longitudinal T3 parametrization.
    pub e2_effective: FourMomentum<T>,
    /// e4 / (2 e3.e4), for the transverse T3 parametrization.
    pub e4_effective: FourMomentum<T>,
}

pub struct TadpoleCutRequest<'a, T> {
    pub id: SampleId,
    pub shift: &'a FourMomentum<T>,
    pub mass_squared: &'a T,
    /// Uncut propagator shifts used to select a well-conditioned reference.
    pub complementary_shifts: &'a [FourMomentum<T>],
    /// Rank class of the reduction requesting the cut. The Laurent division
    /// divides by powers of each uncut line's leading coefficient that grow
    /// with the class (up to the fifth at rank n+2), so providers may condition
    /// the basis for it.
    pub rank_class: RankClass,
}

/// Propagators are in cut partition order: D_i=(q+shifts[i])²-masses_squared[i]-mu².
pub struct TriangleCutRequest<'a, T> {
    pub id: SampleId,
    pub shifts: [&'a FourMomentum<T>; 3],
    pub masses_squared: [&'a T; 3],
}

/// Propagators are in cut partition order: D_i=(q+shifts[i])²-masses_squared[i]-mu².
pub struct BubbleCutRequest<'a, T> {
    pub id: SampleId,
    pub shifts: [&'a FourMomentum<T>; 2],
    pub masses_squared: [&'a T; 2],
    /// Uncut propagator shifts. The leading large-t coefficient of an uncut
    /// denominator is 2 e3.(V_uncut - V_first), which vanishes when that
    /// difference lies in the plane of the cut momentum and the auxiliary
    /// reference; providers use these shifts to choose a reference avoiding it.
    pub complementary_shifts: &'a [FourMomentum<T>],
    /// Rank class of the reduction requesting the cut. The Laurent division
    /// divides by powers of each uncut line's leading coefficient that grow
    /// with the class (up to the fifth at rank n+2), so providers may condition
    /// the reference for it.
    pub rank_class: RankClass,
}

pub struct TriangleSampleRequest<'a, T> {
    pub id: SampleId,
    pub cut: &'a TriangleCut<T>,
    pub t: &'a T,
    pub mu_squared: &'a T,
}

pub struct BubbleSampleRequest<'a, T> {
    pub id: SampleId,
    pub cut: &'a BubbleCut<T>,
    pub x: &'a T,
    pub t: &'a T,
    pub mu_squared: &'a T,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CutFailure {
    NonFiniteInput,
    DerivativeShape,
    DegenerateGram,
    DegenerateReference,
    NonFiniteResult,
    SingularParameter,
}

#[derive(Clone, Debug)]
pub enum CutOutcome<G> {
    Stable(G),
    Unstable(CutFailure),
}

pub trait CutProvider<T> {
    fn tadpole_batch(
        &mut self,
        requests: &[TadpoleCutRequest<'_, T>],
        output: &mut [CutOutcome<TadpoleCut<T>>],
    ) -> Result<()>;
    fn triangle_batch(
        &mut self,
        requests: &[TriangleCutRequest<'_, T>],
        output: &mut [CutOutcome<TriangleCut<T>>],
    ) -> Result<()>;
    fn bubble_batch(
        &mut self,
        requests: &[BubbleCutRequest<'_, T>],
        output: &mut [CutOutcome<BubbleCut<T>>],
    ) -> Result<()>;
}

pub trait CutSampler<T> {
    fn sample_triangle_batch(
        &mut self,
        requests: &[TriangleSampleRequest<'_, T>],
        output: &mut [CutOutcome<FourMomentum<T>>],
    ) -> Result<()>;
    fn sample_bubble_batch(
        &mut self,
        requests: &[BubbleSampleRequest<'_, T>],
        output: &mut [CutOutcome<FourMomentum<T>>],
    ) -> Result<()>;
}
