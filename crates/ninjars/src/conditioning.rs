//! The conditioning bound of the box mu-expansion (`docs/box-mu-conditioning.md`).
//!
//! At rank `r = n + k` (`k` = 0, 1, 2) a box cut of an `n`-point lane gets its
//! rational coefficients (`c4`, and `c6` at rank n+2) from the leading terms
//! of `N(q)/Π_j D_j` along `q = t v + a`, where every uncut line is
//! `D_j = d0_j t + d1_j` with `d0_j = 2 v.(V_j - V_first)` (`v` the box normal).
//! Dividing the numerator's leading coefficient `n0` by `d0_j` (`k + 1` times
//! for the term `n0 d1^k / d0^(k+1)`) is exact algebra, but `d0_j` vanishes
//! with the Gram determinant of the `n`-point lane (a near-planar 2 -> 3 point
//! makes every box normal of a pentagon nearly orthogonal to the fifth line),
//! while the rational part of the lane stays finite: the boxes' large `c4`
//! cancel between the boxes of the lane. Rounding then decides the result.
//!
//! First-order rounding of the division: `d0_j = 2 v.(V_j - V_first)` is
//! computed from the box normal `v`, whose components carry a relative error
//! of order `u`, and two shifts, so its absolute error is `δ ≲ u S_j` with the
//! Euclidean scale `S_j = 2 |v| |V_j - V_first|` (`|.|` the Euclidean norm of
//! the components; the dot product's own rounding is bounded by the same
//! expression, Cauchy-Schwarz). With `R_j = S_j / |d0_j|` the relative error
//! of the leading term is `u Σ_j R_j` and its size relative to a regular box's
//! (`|d0_j| ~ S_j`, `R_j` of order 1 at regular kinematics) is
//! `ν Π_j R_j ρ^k`, where `ν = |n0| / N̄ <= 1` compares the numerator's
//! leading coefficient along `v` with its scale along the box's own
//! directions (`N̄`), and `ρ = Σ_j |d1_j| / (|d0_j| t)` compares the term
//! `n0 d1^k / d0^(k+1)` with the regular `n_k / d0` (`n_k ~ n0 t^k`, `t` the
//! distance of the box's four-dimensional cut solutions from its center along
//! `v`, `t² = |beta / v.v|`; `ρ` grows like `R` at a degenerate lane and is of
//! order 1 at regular ones, taken at least 1). The rounding of `n0` itself
//! (and of the subleading terms) adds `u Π_j R_j ρ^k` relative. The predicted
//! loss in bits relative to the regular box is therefore
//!
//! ```text
//! L = log2(Π_j R_j) + k log2(max(ρ, 1)) + log2(ν Σ_j R_j + 1),
//! ```
//!
//! for one uncut line at a degenerate lane (`ρ ~ R`) `L = (k + 1) log2 R +
//! log2(ν R + 1)`: about `(k + 2) log2 R` for a generic numerator and
//! `(k + 1) log2 R` when the leading coefficient vanishes with `d0` (`ν ~ 1/R`, e.g. a numerator even
//! under the reflection through the event plane, as helicity-summed parity-even
//! interferences are at planar points). A lane is refused (Unstable /
//! `SingularParameter`) when `L` exceeds the carrier's precision `p` minus
//! [`BOX_MU_MARGIN_BITS`]. The test is independent of the noise tests
//! (`FastOptions::stability_threshold` / `relative_tolerance`, the generic
//! engine's `Negligible`), which stay as Ninja designed them.
//!
//! At rank n (`k = 0`) the engines test a cheap upper bound first: `R_j <= 3
//! S'/|d0_j|₁` with the taxicab scale `S' = 4 max|V| Σ_μ |v_μ|₁` (`S_j <= 2
//! S'`, `|d0| >= |d0|₁ / sqrt(2)`), which regular rows pass with one division
//! and two products per uncut line; above rank n they form `R_j` and `ρ`
//! directly. A row over the budget gets its `ν`: the numerator's leading coefficient along three
//! directions of the box's own hyperplane (pair sums of its shift
//! differences, scaled to the normal's taxicab norm), `N̄` the largest modulus.

/// The margin `M` in bits: a lane is refused when a box's predicted loss `L`
/// exceeds `p - M`. Derived from the target and the calibration
/// (`docs/box-mu-conditioning.md`, section 3): a served lane must keep the
/// lax target `1e-5` (`log2(1e5) = 16.6` bits), and over the calibration
/// lanes (loop-induced and Born-interfered 2 -> 3 lanes at near-planar
/// points; binary64 fast engine, double-double and MPFR at 107 and 128 bits)
/// the measured loss stays at least `3.4` bits below `L` (the smallest slack,
/// double-double), so `M = ceil(16.6 - 3.4) = 14`.
pub const BOX_MU_MARGIN_BITS: u32 = 14;

/// Running product and sum of `R_j = S_j / |d0_j|` over a box's uncut lines,
/// and at rank above n the expansion ratio `ρ = Σ_j |d1_j| / (|d0_j| t)`.
#[derive(Clone, Copy, Debug)]
pub struct BoxMuConditioning {
    /// `Π_j R_j`.
    pub product: f64,
    /// `Σ_j R_j`.
    pub sum: f64,
    /// `ρ`: `Σ_j |d1_j| / (|d0_j| t)`, `t` the distance of the box's
    /// four-dimensional cut solutions from its center along the normal (in
    /// units of the normal). Read only at rank above n.
    pub expansion: f64,
}

impl BoxMuConditioning {
    /// A box without uncut lines.
    pub const ONE: Self = Self {
        product: 1.,
        sum: 0.,
        expansion: 0.,
    };

    /// Adds an uncut line with leading coefficient magnitude `d0` and scale
    /// `scale` (`S_j`, or an upper bound of it).
    #[inline(always)]
    pub fn line(&mut self, scale: f64, d0: f64) {
        let r = scale / d0;
        self.product *= r;
        self.sum += r;
    }

    /// Adds an uncut line's expansion ratio `|d1| / (|d0| t)` (rank above n).
    #[inline(always)]
    pub fn expansion(&mut self, d1: f64, d0: f64, t: f64) {
        self.expansion += d1 / (d0 * t);
    }

    /// Whether the predicted loss at rank `n + excess` with numerator ratio
    /// `nu` exceeds `precision - BOX_MU_MARGIN_BITS` bits. NaN exceeds.
    #[inline(always)]
    pub fn exceeds(self, excess: u32, nu: f64, precision: u32) -> bool {
        let budget = precision.saturating_sub(BOX_MU_MARGIN_BITS) as i32;
        let amplification =
            self.product * self.expansion.max(1.).powi(excess as i32) * (nu * self.sum + 1.);
        amplification.is_nan() || amplification > 2f64.powi(budget)
    }
}

/// `ν = |n0| / N̄` capped at 1: 0 when the leading coefficient vanishes
/// exactly (the numerator has no term of the box's top degree), 1 when only
/// the reference vanishes or is not finite.
#[inline]
pub fn box_mu_numerator_ratio(leading: f64, reference: f64) -> f64 {
    if leading == 0. {
        0.
    } else if reference > 0. && reference.is_finite() {
        (leading / reference).min(1.)
    } else {
        1.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One uncut line at a degenerate lane: `ρ = R`.
    fn line(r: f64) -> BoxMuConditioning {
        let mut c = BoxMuConditioning::ONE;
        c.line(r, 1.);
        c.expansion(r, 1., 1.);
        c
    }

    /// One uncut line: `L = (k + 1) log2 R + log2(ν R + 1)` against
    /// `p - BOX_MU_MARGIN_BITS`.
    #[test]
    fn one_line_follows_the_loss_model() {
        let budget = (53 - BOX_MU_MARGIN_BITS) as f64;
        // Generic numerator: about 2 log2 R at rank n.
        let edge = 2f64.powf(budget / 2.);
        assert!(!line(edge * 0.7).exceeds(0, 1., 53));
        assert!(line(edge * 1.5).exceeds(0, 1., 53));
        // Rank n+1 and n+2 lose one more division each.
        assert!(line(2f64.powf(budget / 3.) * 1.5).exceeds(1, 1., 53));
        assert!(!line(2f64.powf(budget / 3.) * 0.7).exceeds(1, 1., 53));
        assert!(line(2f64.powf(budget / 4.) * 1.5).exceeds(2, 1., 53));
        assert!(!line(2f64.powf(budget / 4.) * 0.7).exceeds(2, 1., 53));
        // A leading coefficient vanishing with d0 (nu ~ 1/R): one division.
        let r = edge * 4.;
        assert!(line(r).exceeds(0, 1., 53));
        assert!(!line(r).exceeds(0, 1. / r, 53));
        // The carrier's precision sets the budget.
        assert!(!line(edge * 4.).exceeds(0, 1., 106));
    }

    #[test]
    fn products_sums_and_nan() {
        let mut c = BoxMuConditioning::ONE;
        c.line(8., 2.);
        c.line(6., 3.);
        assert_eq!((c.product, c.sum), (8., 6.));
        // Above rank n a regular expansion ratio (rho < 1) costs nothing.
        let mut regular = BoxMuConditioning::ONE;
        regular.line(4., 1.);
        regular.expansion(0.5, 1., 1.);
        assert_eq!(regular.exceeds(2, 1., 53), regular.exceeds(0, 1., 53));
        assert!(line(f64::NAN).exceeds(0, 1., 53));
        assert!(line(f64::INFINITY).exceeds(0, 1., 53));
        assert!(!BoxMuConditioning::ONE.exceeds(2, 1., 53));
        assert_eq!(box_mu_numerator_ratio(1., 4.), 0.25);
        assert_eq!(box_mu_numerator_ratio(8., 4.), 1.);
        assert_eq!(box_mu_numerator_ratio(1., 0.), 1.);
        assert_eq!(box_mu_numerator_ratio(0., 0.), 0.);
        assert_eq!(box_mu_numerator_ratio(1., f64::NAN), 1.);
    }
}
