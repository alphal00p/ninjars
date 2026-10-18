//! Batched binary64 numerator callbacks with original Ninja output layouts.
//!
//! The fast engine processes one cut at a time for every active phase-space
//! lane, so each callback receives a compact batch of rows sharing one cut
//! partition. Row `i` belongs to phase-space lane `lanes[i]`; implementations
//! use that lane to select their per-point coefficients. Outputs are row-major
//! with the per-row width given by the layout helpers below.

use super::complex::{C64, CMom};
use crate::{Error, Result};

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
/// (power `t^(r-i)`) lists `x^a mu^(2j)` for `a + 2j <= i`, ordered by `a`
/// then `j`: `(i/2 + 1)(i + 1 - i/2)` entries, i.e.
/// `[1]`, `[1, x]`, `[1, mu², x, x²]`, `[1, mu², x, x mu², x², x³]` and, at
/// rank n+2, `[1, mu², mu⁴, x, x mu², x², x² mu², x³, x⁴]`. (Original
/// Ninja's `SizeHelper<2>`, `1 + i + i/2 + i/3`, agrees only for `i <= 3`.)
pub const fn t2_len(mindeg: usize) -> usize {
    let mut total = 0;
    let mut i = 0;
    while i <= mindeg {
        total += (i / 2 + 1) * (i + 1 - i / 2);
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

    /// The rank n+2 box expansion on the exact d-dimensional cut
    /// ([`crate::ExpansionKind::MuShifted`]): `q = t v0 + v1`,
    /// `mu² = t² v0² - beta`, returning the coefficients of `t^r`, `t^(r-1)`
    /// and `t^(r-2)` (three per row, in that order).
    ///
    /// The default reports [`Error::Unsupported`], so numerators written
    /// before rank n+2 keep compiling; the engines call it only at rank n+2.
    #[allow(clippy::too_many_arguments)]
    fn mu_expansion_x2(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        v0: &[CMom],
        v1: &[CMom],
        beta: &[C64],
        out: &mut [C64],
    ) -> Result<()> {
        let _ = (cut, lanes, v0, v1, beta, out);
        Err(Error::Unsupported(
            "numerator lacks the rank n+2 mu-expansion".into(),
        ))
    }

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

    /// Rank n+2 depths: T3 mindeg 5 (12 slots), T2 mindeg 4 (22 slots,
    /// level sizes 1, 2, 4, 6, 9), and Ninja's `SizeHelper<2>` still for
    /// every existing depth.
    #[test]
    fn rank_n_plus_two_layouts() {
        assert_eq!(t3_len(5), 12);
        assert_eq!((0..5).map(t2_len).collect::<Vec<_>>(), [1, 3, 7, 13, 22]);
        const LEVEL_FOUR: usize = t2_len(4) - t2_len(3);
        assert_eq!(LEVEL_FOUR, 9);
        for i in 0..=3 {
            let ninja = 1 + i + i / 2 + i / 3;
            let level = t2_len(i) - if i == 0 { 0 } else { t2_len(i - 1) };
            assert_eq!(level, ninja);
        }
        // Each level counts the pairs (a, j) with a + 2j <= i.
        for i in 0..=8 {
            let pairs = (0..=i)
                .flat_map(|a| (0..=i / 2).map(move |j| (a, j)))
                .filter(|&(a, j)| a + 2 * j <= i)
                .count();
            let level = t2_len(i) - if i == 0 { 0 } else { t2_len(i - 1) };
            assert_eq!(level, pairs);
        }
    }
}
