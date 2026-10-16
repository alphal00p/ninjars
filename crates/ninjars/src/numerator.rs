use crate::{FourMomentum, Result, SampleId};

/// A direct numerator evaluation, carrying phase-space and cut identity.
pub struct NumeratorRequest<'a, T> {
    pub id: SampleId,
    pub q: &'a FourMomentum<T>,
    pub mu_squared: &'a T,
    pub partition: &'a [usize],
}

/// Requested raw numerator Laurent coefficients. T3 tuples are `(t, mu²)`
/// powers; T2 tuples are `(t, x, mu²)` powers.
/// The output follows this explicit order; no implicit rank-dependent indexing.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExpansionKind {
    Mu { powers: Vec<i32> },
    T3 { powers: Vec<(i32, u16)> },
    T2 { powers: Vec<(i32, u16, u16)> },
}

impl ExpansionKind {
    pub fn len(&self) -> usize {
        match self {
            Self::Mu { powers } => powers.len(),
            Self::T3 { powers } => powers.len(),
            Self::T2 { powers } => powers.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Parameters for Ninja's three Laurent numerator callbacks.
///
/// Mu: q = t*v0 + v1, mu² = t²*v0² (vectors [v0,v1], or [v0] for v1=0).
/// T3: q = a + t*e3 + (beta+mu²)*e4/t (vectors [a,e3,e4]).
/// T2: q = a0+x*a1+t*e3+(b0+b1*x+b2*x²+mu²)*e4/t.
pub struct ExpansionRequest<'a, T> {
    pub id: SampleId,
    pub partition: &'a [usize],
    pub vectors: &'a [FourMomentum<T>],
    pub parameters: &'a [T],
}

/// Ninja's caller-supplied numerator interface, batched at every entry point.
///
/// Implementations must use each request's lane to select external coefficients.
/// Output is request-major; expansion columns follow `ExpansionKind` exactly.
/// A callback cache must include lane, partition, expansion kind and sample,
/// together with the current inputs/parameter generation. `SampleId` alone is
/// not globally unique across reduction stages or successive evaluations.
pub trait Numerator<T> {
    fn rank(&self) -> usize;
    fn evaluate_batch(
        &mut self,
        requests: &[NumeratorRequest<'_, T>],
        output: &mut [T],
    ) -> Result<()>;
    fn expansion_batch(
        &mut self,
        kind: &ExpansionKind,
        requests: &[ExpansionRequest<'_, T>],
        output: &mut [T],
    ) -> Result<()>;
}
