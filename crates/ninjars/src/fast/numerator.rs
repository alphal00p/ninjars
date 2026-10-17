//! Batched binary64 numerator callbacks with original Ninja output layouts.
//!
//! The fast engine processes one cut at a time for every active phase-space
//! lane, so each callback receives a compact batch of rows sharing one cut
//! partition. Row `i` belongs to phase-space lane `lanes[i]`; implementations
//! use that lane to select their per-point coefficients. Outputs are row-major
//! with the per-row width given by the layout helpers below.

use super::complex::{C64, CMom};
use crate::Result;

/// Number of coefficients Ninja's `t3Expansion` returns for `mindeg`.
///
/// For `q = a + t e3 + (param + mu²)/t e4` the terms are ordered by decreasing
/// power of `t`: `t^r`, `t^(r-1)`, `t^(r-2)`, `t^(r-2) mu²`, `t^(r-3)`,
/// `t^(r-3) mu²`, `t^(r-4)`, `t^(r-4) mu²`, `t^(r-4) mu⁴`. Order `i` has
/// `1 + i/2` entries.
pub const fn t3_len(mindeg: usize) -> usize {
    let mut total = 0;
    let mut i = 0;
    while i <= mindeg {
        total += 1 + i / 2;
        i += 1;
    }
    total
}

/// Number of coefficients Ninja's `t2Expansion` returns for `mindeg`.
///
/// For `q = a0 + x a1 + t e3 + (p0 + x p1 + x² p2 + mu²)/t e4`, order `i`
/// (power `t^(r-i)`) has `1 + i + i/2 + i/3` entries:
/// `[1]`, `[1, x]`, `[1, mu², x, x²]`, `[1, mu², x, x mu², x², x³]`.
pub const fn t2_len(mindeg: usize) -> usize {
    let mut total = 0;
    let mut i = 0;
    while i <= mindeg {
        total += 1 + i + i / 2 + i / 3;
        i += 1;
    }
    total
}

/// Batched numerator interface used by [`super::FastAmplitude`].
///
/// All methods receive `rows = lanes.len()` requests. `cut` is the sorted cut
/// partition (original propagator indices) shared by every row of the call.
/// Implementations must not assume any relation between consecutive calls.
pub trait FastNumerator {
    /// Maximal total degree in the loop momentum, with `mu²` counting as two.
    fn rank(&self) -> usize;

    /// Direct evaluation `N(q, mu²)`; `out[i]` for row `i`.
    fn evaluate(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        q: &[CMom],
        mu_squared: &[C64],
        out: &mut [C64],
    ) -> Result<()>;

    /// Ninja's `muExpansion`: `q = t v0 + v1`, `mu² = t² v0²`, returning the
    /// leading coefficients in `t`. Without `v1` (rank ≤ n) one coefficient,
    /// `t^r`, is returned per row; with `v1` (rank n+1) two, `t^r, t^(r-1)`.
    fn mu_expansion(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        v0: &[CMom],
        v1: Option<&[CMom]>,
        out: &mut [C64],
    ) -> Result<()>;

    /// Ninja's `t3Expansion`; `t3_len(mindeg)` outputs per row.
    #[allow(clippy::too_many_arguments)]
    fn t3_expansion(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        a: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[C64],
        mindeg: usize,
        out: &mut [C64],
    ) -> Result<()>;

    /// Ninja's `t2Expansion`; `t2_len(mindeg)` outputs per row.
    #[allow(clippy::too_many_arguments)]
    fn t2_expansion(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        a0: &[CMom],
        a1: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[[C64; 3]],
        mindeg: usize,
        out: &mut [C64],
    ) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_match_original_ninja() {
        assert_eq!((0..5).map(t3_len).collect::<Vec<_>>(), [1, 2, 4, 6, 9]);
        assert_eq!((0..4).map(t2_len).collect::<Vec<_>>(), [1, 3, 7, 13]);
    }
}
