//! Measured-cancellation rescue of the precision-generic engine, with the
//! rescue precision chosen from the base precision (the "ladder").
//!
//! The generic [`Amplitude`] measures, per lane, the moduli of every term
//! contracted into its cut contributions when its kernels opt in
//! ([`SymbolicaAmplitudeKernels::with_stability_test`]; each master integral
//! times its weight, each bubble tensor master separately, each rational
//! term). With `M = Σ_cuts Σ_terms |t_0|` and `V = Σ_cuts c_0` (finite part,
//! taxicab moduli) the lane lost `L = log2(M / |V|)` bits to cancellation:
//! rounding leaves about `γ 2^-p M` of absolute error in a carrier of `p`
//! bits, i.e. `p - L - log2 γ` correct bits. The loss is a property of the
//! lane, the same in every carrier; only `p` changes. This is the criterion
//! of the binary64 fast engine's rescue (`FastOptions::cancellation_rescue`,
//! `docs/triangle-gram-conditioning.md`) written in bits relative to the
//! carrier.
//!
//! **The rule** ([`LadderRule`], `docs/cancellation-ladder.md`), driven by one
//! number, the target relative accuracy `τ` of a lane (the caller's; by
//! default [`default_target`], half the base carrier's bits). A lane whose
//! estimate `ε = u_p M / |V| = 2^(L - p)` exceeds `τ` is reduced again in
//! the smallest available carrier ([`Rung`]: double-double, 106 bits, above
//! a smaller base; MPFR at a multiple of 64 bits from 128) whose estimate
//! meets `τ` with a margin: `p' >= L + log2(1/τ) + m`, `m = 8` bits
//! (`ceil(log2 γ) + 1`, `γ <= 50` the growth of the rounding through the
//! stages measured on the degenerate lanes, one bit for measuring `L` from
//! the base value itself). It is reduced from kinematics made consistent in
//! that carrier (below) and from the base numerator coefficients lifted
//! exactly, and measured again ([`LadderOutcome`]): served when its own
//! estimate `2^(L' - p')` meets `τ` (`Accepted`), or when its value is below
//! the base carrier's resolution of its terms (`|V'| <= 2^-p M'`, a lane
//! that vanishes up to rounding: `BelowResolution`, its absolute error far
//! below the base pass's); otherwise the next rung (`L' + log2(1/τ) + m`),
//! at most [`LadderRule::max_rungs`] and [`LadderRule::max_bits`], after
//! which the lane is reported `Missed` and served no value
//! ([`LaneRescue::value`] `None`) for the caller's own rescue.
//!
//! **Kinematics.** Lifting the base shifts exactly keeps their inconsistency
//! with the exact invariants (a massless leg's `p² = O(2^-p E²)` against an
//! exactly zero invariant), which the lane amplifies like its rounding. The
//! ladder projects the shifts, in the rung's precision, onto exact shells of
//! every pair whose invariant is exactly zero or exactly a real squared
//! mass of the lane (minimal-norm Gauss-Newton), and takes every other
//! invariant from the projected shifts; the point moves by `O(2^-p)`
//! relative. A caller with higher-precision kinematics passes them as low
//! parts ([`LadderRefinement`]); they are projected the same way (a move
//! below their own precision when they are consistent already). Real shifts
//! only: a lane with complex shifts is lifted as given.

use std::cmp::Ordering;

use ninjars::{
    Amplitude, AmplitudeInput, Complex, DoubleFloat, Float, FourMomentum, Laurent, Numerator, Real, Result,
    SamplingScaleProvider, SamplingScaleSource, SamplingScaleStatus, EvaluationStatus,
};
use oneloop::NativeFloat;
use symbolica::{
    atom::Atom,
    domains::float::{RealLike, Scalar},
    evaluate::EvaluationDomain,
};

use crate::{
    PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaHighCuts,
    SymbolicaMasters, SymbolicaSamplingScales,
};

/// Bits of a double-double carrier.
pub const DOUBLE_DOUBLE_BITS: u32 = 106;
/// MPFR rungs are multiples of one limb: padding to the limb costs nothing.
pub const MPFR_LIMB_BITS: u32 = 64;

/// A carrier the ladder reduces a lane in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Rung {
    DoubleDouble,
    /// MPFR at these bits (a multiple of [`MPFR_LIMB_BITS`]).
    Arbitrary(u32),
}

impl Rung {
    pub fn bits(self) -> u32 {
        match self {
            Self::DoubleDouble => DOUBLE_DOUBLE_BITS,
            Self::Arbitrary(bits) => bits,
        }
    }
}

/// The ladder's rule (module docs): when a lane is rescued, in which carrier,
/// and when a rescued value is accepted. Everything is driven by one number,
/// the target relative accuracy of a lane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LadderRule {
    /// The target relative accuracy of every lane (the caller's, reduced by
    /// its own safety factor), or `None`: [`default_target`] of the base
    /// carrier.
    pub target: Option<f64>,
    /// `m`: bits of the rescue carrier above `L + log2(1 / target)`.
    pub margin_bits: f64,
    /// Rungs tried before a lane is reported [`LadderOutcome::Missed`].
    pub max_rungs: usize,
    /// Highest rung precision (`None`: `max(4 p, 512)`).
    pub max_bits: Option<u32>,
    /// Project the rung's shifts onto their exact shells (module docs).
    /// `false` takes the given (or refined) kinematics as they are.
    pub project: bool,
    /// Extra bits of an MPFR rung per bit lost: numerica's `Float` rounds
    /// every intermediate to its tracked precision, which a cancelling
    /// reduction consumes beyond `L` (measured: about `0.65 L` at 128 bits for
    /// `L = 64`). With 1, the first MPFR rung usually suffices; the
    /// re-measurement guarantees the value either way.
    pub mpfr_tracking: f64,
}

/// The default target of a base carrier of `bits` bits: half its bits,
/// `2^-(p/2)` (`1.1e-8` in binary64, `1.4e-16` in double-double), the level
/// below which the generic engine's own stability test refuses a lane
/// ([`ninjars::EvaluationStatus::Unstable`]).
pub fn default_target(bits: u32) -> f64 {
    2f64.powf(-f64::from(bits) / 2.)
}

/// The measured error estimate `u_p M / |V| = 2^(L - p)` of a lane that lost
/// `lost` bits in a carrier of `bits` bits.
pub fn estimate(bits: u32, lost: f64) -> f64 {
    2f64.powf(lost - f64::from(bits))
}

impl Default for LadderRule {
    fn default() -> Self {
        Self {
            target: None,
            margin_bits: 8.,
            max_rungs: 3,
            max_bits: None,
            project: true,
            mpfr_tracking: 1.,
        }
    }
}

impl LadderRule {
    pub fn target(&self, base_bits: u32) -> f64 {
        self.target.unwrap_or_else(|| default_target(base_bits))
    }

    /// Whether a lane of a `base_bits` carrier that lost `lost` bits misses
    /// the target: `2^(L - p) > target`.
    pub fn flags(&self, base_bits: u32, lost: f64) -> bool {
        estimate(base_bits, lost) > self.target(base_bits)
    }

    pub fn max_bits(&self, base_bits: u32) -> u32 {
        self.max_bits.unwrap_or((4 * base_bits).max(512))
    }

    /// The smallest carrier above `base_bits` whose estimate meets the target
    /// with the margin, `p' >= L + log2(1 / target) + m` (an MPFR rung
    /// [`Self::mpfr_tracking`]` * L` more), or `None` above [`Self::max_bits`].
    pub fn rung(&self, base_bits: u32, lost: f64) -> Option<Rung> {
        let need = (lost.max(0.) - self.target(base_bits).log2() + self.margin_bits).ceil();
        if !need.is_finite() {
            return None;
        }
        let need = (need.max(0.) as u32).max(base_bits + 1);
        let rung = if need <= DOUBLE_DOUBLE_BITS {
            Rung::DoubleDouble
        } else {
            let tracked = (f64::from(need) + self.mpfr_tracking * lost.max(0.)).ceil();
            if !tracked.is_finite() || tracked > f64::from(u32::MAX) {
                return None;
            }
            let bits = (tracked as u32).max(2 * MPFR_LIMB_BITS);
            Rung::Arbitrary(bits.div_ceil(MPFR_LIMB_BITS) * MPFR_LIMB_BITS)
        };
        (rung.bits() <= self.max_bits(base_bits)).then_some(rung)
    }
}

/// `L = log2(M / |V|)` from the finite-part sum `value` and term moduli
/// `moduli` (module docs): 0 without terms, `inf` for an exactly zero sum of
/// non-zero terms.
pub fn lost_bits(value_modulus: f64, moduli: f64) -> f64 {
    if !(moduli > 0.) {
        0.
    } else if value_modulus > 0. {
        (moduli / value_modulus).log2().max(0.)
    } else {
        f64::INFINITY
    }
}

/// A base carrier of the ladder and its conversions through MPFR.
pub trait LadderCarrier: NativeFloat {
    /// This value exactly (a double-double's two parts summed at `bits`).
    fn to_float(&self, bits: u32) -> Float;
    /// `value` rounded to the carrier of `prototype`.
    fn from_float(value: &Float, prototype: &Self) -> Self;
    /// State that this value is accurate to `bits` significant bits: MPFR's
    /// tracked precision is lowered to it (so that a caller's tracked
    /// precision test sees the ladder's estimate); fixed carriers ignore it.
    fn limit_precision(&mut self, bits: u32) {
        let _ = bits;
    }
    /// Whether the carrier tracks (and rounds to) significant bits, so that
    /// its estimates are numerica's, not a fixed unit roundoff (MPFR).
    fn tracks_precision() -> bool {
        false
    }
}

impl LadderCarrier for f64 {
    fn to_float(&self, bits: u32) -> Float {
        Float::with_val(bits, *self)
    }
    fn from_float(value: &Float, _: &Self) -> Self {
        symbolica::domains::float::RealLike::to_f64(value)
    }
}

impl LadderCarrier for DoubleFloat {
    fn to_float(&self, bits: u32) -> Float {
        let inner = self.into_inner();
        Float::with_val(bits, inner.hi()) + &Float::with_val(bits, inner.lo())
    }
    fn from_float(value: &Float, _: &Self) -> Self {
        value.to_double_float()
    }
}

impl LadderCarrier for Float {
    fn to_float(&self, bits: u32) -> Float {
        let mut value = self.clone();
        value.set_prec(bits.max(value.prec()));
        value
    }
    fn from_float(value: &Float, prototype: &Self) -> Self {
        let mut value = value.clone();
        value.set_prec(prototype.prec());
        value
    }
    fn limit_precision(&mut self, bits: u32) {
        if bits < self.prec() {
            self.set_prec(bits.max(1));
        }
    }
    fn tracks_precision() -> bool {
        true
    }
}

fn exact(value: &Float, bits: u32) -> Float {
    let mut value = value.clone();
    value.set_prec(bits);
    value
}

fn complex_to_float<T: LadderCarrier>(value: &Complex<T>, bits: u32) -> Complex<Float>
where
    Complex<T>: Real + EvaluationDomain,
{
    Complex::new(value.re.to_float(bits), value.im.to_float(bits))
}

fn taxicab<T: NativeFloat>(value: &Complex<T>) -> f64 {
    value.re.to_f64().abs() + value.im.to_f64().abs()
}

/// The measured loss of one lane of `amplitude` (built with
/// [`SymbolicaAmplitudeKernels::with_stability_test`]): `(L, M_0, Σ c)` with
/// `Σ c` the lane's Laurent coefficients summed over its cuts, `M_0` the
/// finite part's term moduli and `L` the larger of the finite part's loss
/// and the loss over all Laurent orders (largest term moduli against the
/// largest coefficient, the measure of [`ninjars::EvaluationStatus::Unstable`]),
/// and at least the bits the tracked precision of an MPFR result says are
/// gone (`working_bits` minus [`crate::tracked_bits`]). `None` when nothing
/// was measured.
pub fn lane_loss<T: NativeFloat>(
    amplitude: &Amplitude<Complex<T>, SymbolicaAmplitudeKernels<T>>,
    lane: usize,
    working_bits: u32,
) -> Option<(f64, f64, Laurent<Complex<T>>)>
where
    Complex<T>: Real + EvaluationDomain,
{
    let lanes = amplitude.lanes();
    let cuts = amplitude.topology().cuts().len();
    let moduli = amplitude.term_moduli();
    let contributions = amplitude.debug_contributions();
    if lane >= lanes || moduli.len() < cuts * lanes || contributions.len() < cuts * lanes {
        return None;
    }
    let zero = contributions[lane][0].zero();
    let mut sum: Laurent<Complex<T>> = std::array::from_fn(|_| zero.clone());
    let mut terms = [0.; 3];
    for cut in 0..cuts {
        let row = cut * lanes + lane;
        for (total, value) in sum.iter_mut().zip(&contributions[row]) {
            *total += value;
        }
        for (k, term) in terms.iter_mut().enumerate() {
            *term += moduli[row][k].re.to_f64();
        }
    }
    // The finite part's loss, and the generic engine's own measure over the
    // Laurent orders (largest terms against the largest coefficient), which
    // also sees a cancellation of the poles.
    let finite = lost_bits(taxicab(&sum[0]), terms[0]);
    let largest = sum.iter().map(taxicab).fold(0., f64::max);
    let orders = lost_bits(largest, terms.iter().copied().fold(0., f64::max));
    // MPFR (numerica's `Float`) tracks the significant bits of every
    // intermediate and rounds to them: the result keeps at most its tracked
    // bits, whatever the moduli say (fixed carriers report their own bits).
    let tracked = crate::tracked_bits(&sum[0]).map_or(0., |t| f64::from(working_bits) - t);
    Some((finite.max(orders).max(tracked), terms[0], sum))
}

/// Higher-precision kinematics of rescued lanes, as low parts in the base
/// carrier (CONTRACT.md, LADDER): the refined value of every entry is the
/// base input plus its low part, summed exactly in the rung's precision (no
/// normalisation of the pair is assumed). Asked only for lanes the ladder
/// rescues. The refined point must be the higher-precision cure of the base
/// point (on-shell legs, momentum conservation, invariants consistent with
/// the shifts); the ladder projects it onto its shells in the rung's
/// precision either way (module docs), which moves a consistent point by
/// less than its own precision.
pub trait LadderRefinement<T> {
    /// Fill one lane's low parts (zero-initialised: `n` shifts of four real
    /// components, `n²` invariants `(V_i - V_j)²` row-major, `n` squared
    /// masses) and return true, or return false: no refinement for this lane
    /// (its base inputs are lifted and projected).
    fn lane(
        &self,
        lane: usize,
        shifts: &mut [[Complex<T>; 4]],
        s_matrix: &mut [Complex<T>],
        masses: &mut [Complex<T>],
    ) -> bool;
}

/// Lane-major low parts as slices ([`LadderRefinement`]): `shifts` `n` per
/// lane, `s_matrix` `n²` per lane (`None`: the invariants of the refined
/// shifts), `masses_squared` `n` per lane (`None`: exact in the base carrier).
pub struct SliceRefinement<'a, T> {
    pub n: usize,
    pub shifts: &'a [FourMomentum<Complex<T>>],
    pub s_matrix: Option<&'a [Complex<T>]>,
    pub masses_squared: Option<&'a [Complex<T>]>,
}

impl<T: NativeFloat> LadderRefinement<T> for SliceRefinement<'_, T>
where
    Complex<T>: Real + EvaluationDomain,
{
    fn lane(
        &self,
        lane: usize,
        shifts: &mut [[Complex<T>; 4]],
        s_matrix: &mut [Complex<T>],
        masses: &mut [Complex<T>],
    ) -> bool {
        let n = self.n;
        for (i, v) in shifts.iter_mut().enumerate() {
            *v = self.shifts[lane * n + i].components().map(Clone::clone);
        }
        if let Some(s) = self.s_matrix {
            s_matrix.clone_from_slice(&s[lane * n * n..(lane + 1) * n * n]);
        }
        if let Some(m) = self.masses_squared {
            masses.clone_from_slice(&m[lane * n..(lane + 1) * n]);
        }
        true
    }
}

/// How the ladder ended for one lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LadderOutcome {
    /// The base pass refused the lane ([`EvaluationStatus::Unstable`]: more
    /// than half the carrier's bits lost) but its estimate meets the target:
    /// its base value is served, nothing is re-reduced.
    WithinTarget,
    /// The last rung's own estimate meets the target; its value is served.
    Accepted,
    /// The last rung's value lies below the base carrier's resolution of
    /// the lane's terms (`|V'| <= 2^-p M'`): the lane vanishes up to rounding,
    /// so only an absolute accuracy could be stated. Refused, typed (no
    /// value; its rung value is [`LaneRescue::best`]): the caller gave no
    /// absolute budget.
    BelowResolution,
    /// Every rung allowed missed the target: no value is served, the lane is
    /// refused `Unstable` for the caller's own rescue.
    Missed,
    /// A rung failed (a cut, a master, a non-finite value) or no rung was
    /// available: refused as `Missed`.
    Failed,
}

/// The ladder's record of one lane it re-reduced.
pub struct LaneRescue<T> {
    pub lane: usize,
    /// The base pass's Laurent coefficients (summed cut contributions).
    pub base: Laurent<Complex<T>>,
    /// Bits the base pass lost (`L`) and its estimate `2^(L - p)`.
    pub lost_bits: f64,
    pub estimate: f64,
    /// The target the lane was judged against.
    pub target: f64,
    /// Bits of every reduction tried, in order: each MPFR rung followed by
    /// its confirming reduction 64 bits wider (the last is the precision of
    /// the served value).
    pub rungs: Vec<u32>,
    /// The last rung's estimate: `2^(L' - p')` times `2^margin_bits`, and at
    /// least the relative disagreement of an MPFR rung with its confirming
    /// reduction.
    pub rung_estimate: Option<f64>,
    /// Whether the shifts were projected onto their exact shells.
    pub projected: bool,
    pub outcome: LadderOutcome,
    /// The served Laurent coefficients rounded to the base carrier
    /// (`Accepted`, `WithinTarget`), else `None`.
    pub value: Option<Laurent<Complex<T>>>,
    /// The best value the ladder has for this lane, served or not: the one
    /// with the smallest estimate among the base pass and every usable rung,
    /// with that estimate and the bits of its carrier. For a refused lane
    /// (`Missed`, `Failed`) the caller can hand it on, wrapped as refused.
    pub best: Option<BestValue<T>>,
}

/// A lane's best available value ([`LaneRescue::best`]).
pub struct BestValue<T> {
    pub value: Laurent<Complex<T>>,
    /// Its estimated relative error `2^(L - p)` (MPFR: tracked precision
    /// included).
    pub estimate: f64,
    /// Bits of the carrier it was computed in.
    pub bits: u32,
}

/// Project one lane's real shifts (`n` four-vectors at `bits`) onto
/// `(V_i - V_j)² = s_ij` for every listed pair (minimal-norm Gauss-Newton,
/// at most six steps). False when the constraints' Jacobian is singular or
/// the residual does not reach `16 2^-bits` of the pairs' scale.
pub fn project_shells(
    shifts: &mut [[Float; 4]],
    pairs: &[(usize, usize, Float)],
    bits: u32,
) -> bool {
    let m = pairs.len();
    if m == 0 {
        return true;
    }
    let zero = Float::with_val(bits, 0);
    let metric = [1i64, -1, -1, -1];
    for step in 0..=6 {
        let d: Vec<[Float; 4]> = pairs
            .iter()
            .map(|(i, j, _)| std::array::from_fn(|mu| exact(&(shifts[*i][mu].clone() - &shifts[*j][mu]), bits)))
            .collect();
        let g: Vec<Float> = d
            .iter()
            .zip(pairs)
            .map(|(x, (_, _, target))| {
                let mut s = exact(&zero, bits);
                for mu in 0..4 {
                    let t = x[mu].clone() * &x[mu];
                    s = if metric[mu] > 0 { s + &t } else { s - &t };
                }
                exact(&(s - target), bits)
            })
            .collect();
        // Converged at the carrier's rounding of the pairs' scale.
        let scale = d
            .iter()
            .map(|x| x.iter().map(|c| c.to_f64() * c.to_f64()).sum::<f64>())
            .fold(0., f64::max);
        let residual = g.iter().map(|x| x.to_f64().abs()).fold(0., f64::max);
        // In the logarithm: `2^-bits` underflows binary64 from 1024 bits on.
        if residual == 0. || residual.log2() <= scale.log2() - f64::from(bits) + 4. {
            return true;
        }
        if step == 6 {
            // Not converged: the caller must not trust this geometry.
            return false;
        }
        // J rows: dg/dV_i = 2 metric d, dg/dV_j = -2 metric d. J Jᵀ (Euclidean
        // in the shifts' components): 4 w Σ_mu d_r d_c, w from shared shifts.
        let mut a = vec![exact(&zero, bits); m * m];
        for r in 0..m {
            for c in 0..m {
                let (ri, rj, _) = &pairs[r];
                let (ci, cj, _) = &pairs[c];
                let delta = |x: &usize, y: &usize| i64::from(x == y);
                let w = delta(ri, ci) - delta(ri, cj) - delta(rj, ci) + delta(rj, cj);
                if w != 0 {
                    let mut dot = exact(&zero, bits);
                    for mu in 0..4 {
                        dot = dot + &(d[r][mu].clone() * &d[c][mu]);
                    }
                    a[r * m + c] = exact(&(dot * &Float::with_val(bits, 4 * w)), bits);
                }
            }
        }
        let mut y = g;
        for col in 0..m {
            let piv = (col..m)
                .max_by(|&p, &q| {
                    a[p * m + col]
                        .to_f64()
                        .abs()
                        .partial_cmp(&a[q * m + col].to_f64().abs())
                        .unwrap_or(Ordering::Equal)
                })
                .unwrap();
            if a[piv * m + col].to_f64() == 0. || !a[piv * m + col].is_finite() {
                return false;
            }
            if piv != col {
                for k in 0..m {
                    a.swap(col * m + k, piv * m + k);
                }
                y.swap(col, piv);
            }
            for r in col + 1..m {
                let f = exact(&(a[r * m + col].clone() / &a[col * m + col]), bits);
                if f.to_f64() == 0. {
                    continue;
                }
                for k in col..m {
                    let t = f.clone() * &a[col * m + k];
                    a[r * m + k] = exact(&(a[r * m + k].clone() - &t), bits);
                }
                let t = f * &y[col];
                y[r] = exact(&(y[r].clone() - &t), bits);
            }
        }
        for col in (0..m).rev() {
            let mut s = y[col].clone();
            for k in col + 1..m {
                s = s - &(a[col * m + k].clone() * &y[k]);
            }
            y[col] = exact(&(s / &a[col * m + col]), bits);
        }
        // ΔV = -Jᵀ y: V_i -= 2 metric d y, V_j += 2 metric d y.
        for (r, (i, j, _)) in pairs.iter().enumerate() {
            for mu in 0..4 {
                let step = exact(
                    &(d[r][mu].clone() * &y[r] * &Float::with_val(bits, 2 * metric[mu])),
                    bits,
                );
                shifts[*i][mu] = exact(&(shifts[*i][mu].clone() - &step), bits);
                shifts[*j][mu] = exact(&(shifts[*j][mu].clone() + &step), bits);
            }
        }
    }
    true
}

/// One lane's inputs in MPFR at `bits`, consistent (module docs).
struct LaneInputs {
    shifts: Vec<FourMomentum<Complex<Float>>>,
    masses: Vec<Complex<Float>>,
    scale: Complex<Float>,
    s_matrix: Vec<Complex<Float>>,
    parameters: Vec<Complex<Float>>,
    projected: bool,
    /// Whether the rung's geometry is consistent to its rounding: projected
    /// and converged (or the caller turned the projection off). False without
    /// an invariant matrix, with complex shifts, or when the projection
    /// failed: such a lane is not served from a rung.
    consistent: bool,
}

#[allow(clippy::too_many_arguments)]
fn lane_inputs<T: LadderCarrier>(
    n: usize,
    lane: usize,
    input: &AmplitudeInput<'_, Complex<T>>,
    parameters: &[Complex<T>],
    width: usize,
    refinement: Option<&dyn LadderRefinement<T>>,
    bits: u32,
    project: bool,
) -> LaneInputs
where
    Complex<T>: Real + EvaluationDomain,
{
    let lift = |v: &Complex<T>| complex_to_float(v, bits);
    let zero_t = input.scales_squared[lane].re.zero();
    let zc = Complex::new(zero_t.clone(), zero_t.clone());
    let mut lo_shifts = vec![std::array::from_fn::<_, 4, _>(|_| zc.clone()); n];
    let mut lo_s = vec![zc.clone(); n * n];
    let mut lo_m = vec![zc.clone(); n];
    let refined = refinement.is_some_and(|r| r.lane(lane, &mut lo_shifts, &mut lo_s, &mut lo_m));
    // hi + lo, summed exactly at `bits` (the pair need not be normalised).
    let sum = |hi: &Complex<T>, lo: &Complex<T>| {
        let (a, b) = (lift(hi), lift(lo));
        Complex::new(exact(&(a.re + &b.re), bits), exact(&(a.im + &b.im), bits))
    };
    let shifts_c: Vec<[Complex<Float>; 4]> = (0..n)
        .map(|i| {
            let hi = input.shifts[lane * n + i].components();
            std::array::from_fn(|mu| {
                if refined { sum(hi[mu], &lo_shifts[i][mu]) } else { lift(hi[mu]) }
            })
        })
        .collect();
    let masses: Vec<Complex<Float>> = (0..n)
        .map(|i| {
            let hi = &input.masses_squared[lane * n + i];
            if refined { sum(hi, &lo_m[i]) } else { lift(hi) }
        })
        .collect();
    let given: Vec<Complex<Float>> = match input.s_matrix {
        Some(s) => (0..n * n)
            .map(|k| {
                let hi = &s[lane * n * n + k];
                if refined { sum(hi, &lo_s[k]) } else { lift(hi) }
            })
            .collect(),
        None => Vec::new(),
    };
    let real_shifts = shifts_c.iter().all(|v| v.iter().all(|c| c.im.is_zero()));
    let mut projected = false;
    let mut shifts_r: Vec<[Float; 4]> = shifts_c.iter().map(|v| v.clone().map(|c| c.re)).collect();
    let mut keep = vec![false; n * n];
    if project && real_shifts && !given.is_empty() {
        // Shell pairs: an exactly zero invariant, or one exactly equal to the
        // real part of a squared mass of the lane (an on-shell leg of a
        // complex-mass particle carries `Re m²`, audit m5).
        let mut pairs = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                let s = &given[i * n + j];
                if !s.im.is_zero() {
                    continue;
                }
                let shell = s.re.is_zero()
                    || masses.iter().any(|m| !m.re.is_zero() && m.re == s.re);
                if shell {
                    keep[i * n + j] = true;
                    keep[j * n + i] = true;
                    pairs.push((i, j, s.re.clone()));
                }
            }
        }
        projected = project_shells(&mut shifts_r, &pairs, bits);
    }
    let zero = Float::with_val(bits, 0);
    let s_matrix: Vec<Complex<Float>> = if given.is_empty() {
        Vec::new()
    } else {
        (0..n * n)
            .map(|k| {
                let (i, j) = (k / n, k % n);
                if i == j {
                    Complex::new(zero.clone(), zero.clone())
                } else if keep[k] || !real_shifts || !projected {
                    given[k].clone()
                } else {
                    let mut s = zero.clone();
                    for mu in 0..4 {
                        let d = exact(&(shifts_r[i][mu].clone() - &shifts_r[j][mu]), bits);
                        let t = d.clone() * &d;
                        s = if mu == 0 { s + &t } else { s - &t };
                    }
                    Complex::new(exact(&s, bits), zero.clone())
                }
            })
            .collect()
    };
    let shifts = if real_shifts {
        shifts_r
            .into_iter()
            .map(|v| FourMomentum::from(v.map(|c| Complex::new(exact(&c, bits), zero.clone()))))
            .collect()
    } else {
        shifts_c.into_iter().map(FourMomentum::from).collect()
    };
    LaneInputs {
        shifts,
        masses,
        scale: lift(&input.scales_squared[lane]),
        s_matrix,
        parameters: parameters[lane * width..(lane + 1) * width].iter().map(lift).collect(),
        projected,
        consistent: !project || projected,
    }
}

/// The generic engine at one rung's carrier `U`.
struct Stage<U: NativeFloat>
where
    Complex<U>: Real + EvaluationDomain,
{
    numerator: PolynomialNumerator<U>,
    amplitude: Amplitude<Complex<U>, SymbolicaAmplitudeKernels<U>>,
    cuts: SymbolicaCuts<U>,
    high: Option<SymbolicaHighCuts<U>>,
    masters: SymbolicaMasters<U>,
    sampling: Option<SymbolicaSamplingScales<U>>,
    prototype: U,
}

/// Per lane of a rung: (status, Laurent sum, lost bits L' (at least the
/// bits the tracked precision of an MPFR result says are gone), term moduli M').
type RungLane<U> = (EvaluationStatus, Laurent<Complex<U>>, f64, f64);

impl<U: NativeFloat> Stage<U>
where
    Complex<U>: Real + EvaluationDomain,
{
    fn new(
        n: usize,
        expression: &Atom,
        variables: &[Atom; 5],
        parameters: &[Atom],
        prototype: U,
        selection: oneloop::PoleSelection,
    ) -> Result<Self> {
        let numerator = PolynomialNumerator::new(
            expression.clone(),
            variables.clone(),
            parameters.to_vec(),
            prototype.clone(),
        )?;
        let amplitude = SymbolicaAmplitudeKernels::new(n, numerator.rank(), prototype.clone())?
            .with_stability_test()
            .into_amplitude(0)?;
        Ok(Self {
            numerator,
            amplitude,
            cuts: SymbolicaCuts::new(prototype.clone())?,
            high: if n >= 4 {
                Some(SymbolicaHighCuts::new(prototype.clone())?)
            } else {
                None
            },
            masters: SymbolicaMasters::new(prototype.clone())
                .with_onshell_threshold(None)
                .with_pole_selection(selection),
            sampling: if n >= 4 {
                Some(SymbolicaSamplingScales::new(n, prototype.clone())?)
            } else {
                None
            },
            prototype,
        })
    }

    fn evaluate(
        &mut self,
        n: usize,
        lanes: &[LaneInputs],
        convert: impl Fn(&Float, &U) -> U,
    ) -> Result<Vec<RungLane<U>>> {
        let count = lanes.len();
        let c = |v: &Complex<Float>| Complex::new(convert(&v.re, &self.prototype), convert(&v.im, &self.prototype));
        let shifts: Vec<FourMomentum<Complex<U>>> = lanes
            .iter()
            .flat_map(|l| l.shifts.iter().map(|v| FourMomentum::from(v.components().map(c))))
            .collect();
        let masses: Vec<Complex<U>> = lanes.iter().flat_map(|l| l.masses.iter().map(c)).collect();
        let scales: Vec<Complex<U>> = lanes.iter().map(|l| c(&l.scale)).collect();
        let s_matrix: Vec<Complex<U>> = lanes.iter().flat_map(|l| l.s_matrix.iter().map(c)).collect();
        let parameters: Vec<Complex<U>> =
            lanes.iter().flat_map(|l| l.parameters.iter().map(c)).collect();
        let zero = Complex::new(self.prototype.zero(), self.prototype.zero());
        let mut sampling = vec![zero.clone(); count];
        if let Some(provider) = &mut self.sampling {
            let mut statuses =
                vec![SamplingScaleStatus::Selected(SamplingScaleSource::UnitFallback); count];
            provider.evaluate_batch(&shifts, &masses, count, &mut sampling, &mut statuses)?;
        }
        self.numerator.set_parameter_batch(&parameters, count)?;
        self.amplitude.reset_batch(count);
        let input = AmplitudeInput {
            shifts: &shifts,
            masses_squared: &masses,
            scales_squared: &scales,
            sampling_scales: &sampling,
            s_matrix: (!s_matrix.is_empty()).then_some(&s_matrix[..]),
        };
        let _ = n;
        if let Some(high) = &mut self.high {
            self.amplitude.evaluate_batch_with_inputs(
                input,
                &mut self.numerator,
                &mut self.cuts,
                high,
                &mut self.masters,
            )?;
        } else {
            self.amplitude.evaluate_batch_with_low_cuts(
                input,
                &mut self.numerator,
                &mut self.cuts,
                &mut self.masters,
            )?;
        }
        Ok((0..count)
            .map(|lane| {
                let status = self.amplitude.statuses()[lane];
                match lane_loss(&self.amplitude, lane, self.prototype.get_precision()) {
                    Some((lost, moduli, sum))
                        if matches!(status, EvaluationStatus::Success | EvaluationStatus::Unstable) =>
                    {
                        (status, sum, lost, moduli)
                    }
                    _ => (status, std::array::from_fn(|_| zero.clone()), f64::INFINITY, 0.),
                }
            })
            .collect())
    }
}

enum AnyStage {
    DoubleDouble(Box<Stage<DoubleFloat>>),
    Arbitrary(Box<Stage<Float>>),
}

/// The ladder of one polynomial numerator template (module docs): the rule
/// and the lazily built engines of its rungs.
pub struct PolynomialLadder {
    n: usize,
    expression: Atom,
    variables: [Atom; 5],
    parameters: Vec<Atom>,
    rule: LadderRule,
    selection: oneloop::PoleSelection,
    stages: Vec<(Rung, AnyStage)>,
    counts: LadderCounts,
}

/// What the ladder did since the last [`PolynomialLadder::take_counts`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LadderCounts {
    /// Lanes served `Accepted` / `BelowResolution`, and refused.
    pub accepted: u64,
    pub below_resolution: u64,
    pub refused: u64,
    /// Lanes the base pass refused `Unstable` whose estimate meets the
    /// target (served at their base value).
    pub within_target: u64,
    /// Lanes re-reduced per rung precision (bits): one per rung tried.
    pub reductions_by_bits: std::collections::BTreeMap<u32, u64>,
}

impl PolynomialLadder {
    /// The ladder of the numerator `numerator` of an `n`-propagator
    /// amplitude (its template and parameters; the carrier is the rungs').
    pub fn new<T: NativeFloat>(n: usize, numerator: &PolynomialNumerator<T>) -> Self
    where
        Complex<T>: Real + EvaluationDomain,
    {
        Self {
            n,
            expression: numerator.expression().clone(),
            variables: numerator.variables().clone(),
            parameters: numerator.parameter_names().to_vec(),
            rule: LadderRule::default(),
            selection: oneloop::PoleSelection::BOTH,
            stages: Vec::new(),
            counts: LadderCounts::default(),
        }
    }

    pub fn with_rule(mut self, rule: LadderRule) -> Self {
        self.rule = rule;
        self
    }

    pub fn rule(&self) -> &LadderRule {
        &self.rule
    }

    pub fn set_rule(&mut self, rule: LadderRule) {
        self.rule = rule;
    }

    /// The masters' pole selection of every rung (the base provider's).
    pub fn set_pole_selection(&mut self, selection: oneloop::PoleSelection) {
        if selection != self.selection {
            self.selection = selection;
            self.stages.clear();
        }
    }

    /// What the ladder did since the last call.
    pub fn take_counts(&mut self) -> LadderCounts {
        std::mem::take(&mut self.counts)
    }

    /// Rescue the lanes of the last evaluation of `amplitude` (reset before
    /// it, built with [`SymbolicaAmplitudeKernels::with_stability_test`]) on
    /// `input` and `parameters` (its numerator's parameter rows, lane-major)
    /// that lost too many bits ([`LadderRule::flags`]), and serves the lanes
    /// the base pass refused `Unstable` whose estimate meets the target
    /// ([`LadderOutcome::WithinTarget`]: the target, not half the carrier's
    /// bits, decides). Lanes that are neither `Success` nor `Unstable`, and
    /// successful lanes not flagged, are not returned. Nothing is done when the amplitude measured nothing.
    pub fn rescue<T: LadderCarrier>(
        &mut self,
        amplitude: &Amplitude<Complex<T>, SymbolicaAmplitudeKernels<T>>,
        input: &AmplitudeInput<'_, Complex<T>>,
        parameters: &[Complex<T>],
        refinement: Option<&dyn LadderRefinement<T>>,
    ) -> Result<Vec<LaneRescue<T>>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        let n = self.n;
        let lanes = amplitude.lanes();
        if amplitude.term_moduli().is_empty() || lanes == 0 {
            return Ok(Vec::new());
        }
        // The working precision: the inputs' largest (MPFR values carry
        // their tracked precision, which only shrinks below the working one).
        let prototype = input.scales_squared[0].re.zero();
        let base = input
            .scales_squared
            .iter()
            .map(|v| v.re.get_precision())
            .chain(input.shifts.iter().flat_map(|v| v.components().map(|c| c.re.get_precision())))
            .max()
            .unwrap_or(prototype.get_precision());
        let width = self.parameters.len();
        let target = self.rule.target(base);
        let margin = 2f64.powf(self.rule.margin_bits);
        // Four units of the base carrier: the floor of any served value.
        let floor = below_bits(4., base);
        let mut pending: Vec<LaneRescue<T>> = Vec::new();
        let mut within: Vec<LaneRescue<T>> = Vec::new();
        for lane in 0..lanes {
            let status = amplitude.statuses()[lane];
            if !matches!(status, EvaluationStatus::Success | EvaluationStatus::Unstable) {
                continue;
            }
            let Some((lost, _, sum)) = lane_loss(amplitude, lane, base) else {
                continue;
            };
            // An MPFR base's estimate is numerica's tracked precision, which
            // can be optimistic (audit B1/M1): it carries the margin, and a
            // lane the engine refused is always re-reduced (its value can
            // only be served once a rung has confirmed it).
            let tracks = T::tracks_precision();
            let base_estimate = estimate(base, lost) * if tracks { margin } else { 1. };
            // A lane the engine refused is served unconfirmed only with the
            // margin (audit m10: double-double estimates are honest within a
            // factor 5-7, not exact); at an MPFR base never.
            let refused = status == EvaluationStatus::Unstable;
            let flagged = base_estimate > target || (refused && (tracks || base_estimate * margin > target));
            if !flagged {
                if status == EvaluationStatus::Unstable {
                    within.push(LaneRescue {
                        lane,
                        value: Some(sum.clone()),
                        best: None,
                        base: sum,
                        lost_bits: lost,
                        estimate: base_estimate,
                        target,
                        rungs: Vec::new(),
                        rung_estimate: None,
                        projected: false,
                        outcome: LadderOutcome::WithinTarget,
                    });
                }
            } else {
                pending.push(LaneRescue {
                    lane,
                    best: Some(BestValue {
                        value: sum.clone(),
                        estimate: base_estimate,
                        bits: base,
                    }),
                    base: sum,
                    lost_bits: lost,
                    estimate: base_estimate,
                    target,
                    rungs: Vec::new(),
                    rung_estimate: None,
                    projected: false,
                    outcome: LadderOutcome::Failed,
                    value: None,
                });
            }
        }
        // An exactly vanishing sum of non-zero terms lost everything the base
        // carrier had: size its rung as such (BelowResolution then serves it).
        let sizing = |lost: f64| if lost.is_finite() { lost } else { f64::from(base) };
        let mut next: Vec<Option<Rung>> = pending
            .iter()
            .map(|r| self.rule.rung(base, sizing(r.lost_bits)))
            .collect();
        let mut open: Vec<usize> = (0..pending.len()).collect();
        let cap = self.rule.max_bits(base);
        for _ in 0..self.rule.max_rungs {
            open.retain(|&k| next[k].is_some());
            if open.is_empty() {
                break;
            }
            let mut rungs: Vec<Rung> = open.iter().map(|&k| next[k].unwrap()).collect();
            rungs.sort();
            rungs.dedup();
            let mut still = Vec::new();
            for rung in rungs {
                let members: Vec<usize> = open
                    .iter()
                    .copied()
                    .filter(|&k| next[k] == Some(rung))
                    .collect();
                for &k in &members {
                    next[k] = None;
                }
                // An MPFR rung is accepted only when an independent reduction
                // 64 bits wider agrees with it within the target: numerica's
                // tracked precision is an estimate, not a bound (optimistic by
                // up to 2^25 at near-collinear points). A double-double rung
                // has fixed precision and is accepted with the margin.
                let confirm = match rung {
                    Rung::DoubleDouble => None,
                    Rung::Arbitrary(bits) => Some(Rung::Arbitrary(bits + MPFR_LIMB_BITS)),
                };
                let mut outcomes = Vec::new();
                let mut failed = false;
                for r in std::iter::once(rung).chain(confirm) {
                    // Inputs formed in MPFR at the rung's bits (at least two
                    // limbs; a double-double rung rounds them).
                    let bits = r.bits().max(2 * MPFR_LIMB_BITS);
                    let inputs: Vec<LaneInputs> = members
                        .iter()
                        .map(|&k| lane_inputs(n, pending[k].lane, input, parameters, width, refinement, bits, self.rule.project))
                        .collect();
                    // An error inside a rung fails its lanes, not the batch.
                    match self.evaluate_rung(r, &inputs) {
                        Ok(lanes) => outcomes.push((inputs, lanes)),
                        Err(_) => {
                            failed = true;
                            break;
                        }
                    }
                    *self.counts.reductions_by_bits.entry(r.bits()).or_default() +=
                        members.len() as u64;
                }
                for (i, &k) in members.iter().enumerate() {
                    let record = &mut pending[k];
                    record.rungs.push(rung.bits());
                    if let (Some(c), false) = (confirm, failed) {
                        record.rungs.push(c.bits());
                    }
                    if failed {
                        record.outcome = LadderOutcome::Failed;
                        continue;
                    }
                    let usable = |(status, sum, _, moduli): &RungLane<Float>| {
                        matches!(status, EvaluationStatus::Success | EvaluationStatus::Unstable)
                            && sum.iter().all(|v| v.re.is_finite() && v.im.is_finite())
                            && moduli.is_finite()
                    };
                    let low = &outcomes[0].1[i];
                    let high = outcomes.get(1).map(|o| &o.1[i]);
                    record.projected = outcomes[0].0[i].projected;
                    if !outcomes.iter().all(|o| o.0[i].consistent) {
                        // Consistency could not be established: no rung value
                        // is trusted (the best value stays the base's).
                        record.outcome = LadderOutcome::Failed;
                        continue;
                    }
                    if !usable(low) || high.is_some_and(|h| !usable(h)) {
                        // A cut, master (e.g. its own precision test) or
                        // stability refusal at this width: climb.
                        record.outcome = LadderOutcome::Missed;
                        let wider = (rung.bits() + 2 * MPFR_LIMB_BITS).max(2 * MPFR_LIMB_BITS);
                        if wider <= cap {
                            next[k] = Some(Rung::Arbitrary(wider.div_ceil(MPFR_LIMB_BITS) * MPFR_LIMB_BITS));
                            still.push(k);
                        }
                        continue;
                    }
                    // The served value: the wider reduction when there is one.
                    let (served_rung, (_, sum, lost, moduli)) = match high {
                        Some(h) => (confirm.unwrap(), h),
                        None => (rung, low),
                    };
                    let own = estimate(served_rung.bits(), *lost);
                    // The agreement of the two reductions, for EVERY Laurent
                    // coefficient (poles included), relative to the served
                    // coefficient (an exactly zero pair agrees).
                    let relative_to = |d: &Complex<Float>, v: &Complex<Float>| {
                        let (d, v) = (taxicab_f(d), taxicab_f(v));
                        if d == 0. { 0. } else if v > 0. { d / v } else { f64::INFINITY }
                    };
                    let disagreement = match high {
                        Some(_) => (0..3)
                            .map(|j| relative_to(&(low.1[j].clone() - &sum[j]), &sum[j]))
                            .fold(0., f64::max),
                        None => 0.,
                    };
                    let honest = (own * margin).max(disagreement);
                    let accurate = (-honest.log2()).floor().clamp(1., f64::from(u32::MAX)) as u32;
                    let rounded: Laurent<Complex<T>> = sum.clone().map(|v| {
                        let mut re = T::from_float(&v.re, &prototype);
                        let mut im = T::from_float(&v.im, &prototype);
                        re.limit_precision(accurate);
                        im.limit_precision(accurate);
                        Complex::new(re, im)
                    });
                    // The estimate of the value actually returned: the rung's,
                    // plus what the narrowing to the base carrier did to every
                    // coefficient (never below a few base units, audit m9).
                    let bits = served_rung.bits().max(2 * MPFR_LIMB_BITS);
                    let narrowing = (0..3)
                        .map(|j| {
                            let back = complex_to_float(&rounded[j], bits);
                            relative_to(&(back - &sum[j]), &sum[j])
                        })
                        .fold(0., f64::max);
                    let reported = honest.max(narrowing).max(floor);
                    record.rung_estimate = Some(reported);
                    if record.best.as_ref().is_none_or(|b| reported < b.estimate) {
                        record.best = Some(BestValue {
                            value: rounded.clone(),
                            estimate: reported,
                            bits: served_rung.bits(),
                        });
                    }
                    // A vanishing lane meets only an absolute criterion: it is
                    // reported `BelowResolution`, typed and refused (its value
                    // carried as the best one), never served as a success.
                    let below = taxicab_f(&sum[0]) <= below_bits(*moduli, base)
                        && f64::from(served_rung.bits()) >= f64::from(base) - target.log2();
                    // Served: the returned value's own estimate meets the
                    // target (narrowing included) and the rung is clearly
                    // better than the base value (audit m5).
                    let served = if honest <= target
                        && honest.max(narrowing) <= target.max(floor)
                        && honest < record.estimate
                    {
                        Some(LadderOutcome::Accepted)
                    } else {
                        None
                    };
                    if below && served.is_none() {
                        record.outcome = LadderOutcome::BelowResolution;
                        continue;
                    }
                    if let Some(outcome) = served {
                        record.outcome = outcome;
                        record.value = Some(rounded);
                    } else {
                        record.outcome = LadderOutcome::Missed;
                        // The next rung: sized from the measured loss, and at
                        // least one limb above the confirming reduction.
                        let floor = served_rung.bits() + MPFR_LIMB_BITS;
                        let sized = self.rule.rung(base, sizing(*lost)).map_or(0, |r| r.bits());
                        let wider = sized.max(floor);
                        if wider <= cap {
                            next[k] = Some(Rung::Arbitrary(wider.div_ceil(MPFR_LIMB_BITS) * MPFR_LIMB_BITS));
                            still.push(k);
                        }
                    }
                }
            }
            open = still;
        }
        for record in &pending {
            match record.outcome {
                LadderOutcome::Accepted => self.counts.accepted += 1,
                LadderOutcome::BelowResolution => self.counts.below_resolution += 1,
                LadderOutcome::Missed | LadderOutcome::Failed => self.counts.refused += 1,
                LadderOutcome::WithinTarget => {}
            }
        }
        self.counts.within_target += within.len() as u64;
        pending.extend(within);
        Ok(pending)
    }

    fn evaluate_rung(&mut self, rung: Rung, inputs: &[LaneInputs]) -> Result<Vec<RungLane<Float>>> {
        let index = match self.stages.iter().position(|(r, _)| *r == rung) {
            Some(index) => index,
            None => {
                let stage = match rung {
                    Rung::DoubleDouble => AnyStage::DoubleDouble(Box::new(Stage::new(
                        self.n,
                        &self.expression,
                        &self.variables,
                        &self.parameters,
                        DoubleFloat::from(0.0),
                        self.selection,
                    )?)),
                    Rung::Arbitrary(bits) => AnyStage::Arbitrary(Box::new(Stage::new(
                        self.n,
                        &self.expression,
                        &self.variables,
                        &self.parameters,
                        Float::with_val(bits, 0),
                        self.selection,
                    )?)),
                };
                self.stages.push((rung, stage));
                self.stages.len() - 1
            }
        };
        let n = self.n;
        Ok(match &mut self.stages[index].1 {
            AnyStage::DoubleDouble(stage) => stage
                .evaluate(n, inputs, |v, _| v.to_double_float())?
                .into_iter()
                .map(|(status, sum, lost, moduli)| (status, sum.map(|v| complex_to_float(&v, 2 * MPFR_LIMB_BITS)), lost, moduli))
                .collect(),
            AnyStage::Arbitrary(stage) => {
                let bits = rung.bits();
                stage.evaluate(n, inputs, |v, _| exact(v, bits))?
            }
        })
    }
}

/// `x 2^-bits` without underflowing `2^-bits` itself (`2f64.powi(-1024)` is
/// zero): through the logarithm, for any width.
fn below_bits(x: f64, bits: u32) -> f64 {
    if x > 0. && x.is_finite() {
        (x.log2() - f64::from(bits)).exp2()
    } else {
        0.
    }
}

fn taxicab_f(value: &Complex<Float>) -> f64 {
    value.re.to_f64().abs() + value.im.to_f64().abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_picks_the_smallest_sufficient_carrier() {
        let rule = LadderRule::default();
        // Default targets: half the carrier's bits.
        assert_eq!(rule.target(53), 2f64.powf(-26.5));
        // binary64: flagged beyond 26.5 bits lost; double-double while
        // L + 26.5 + 8 <= 106.
        assert!(rule.flags(53, 30.));
        assert!(!rule.flags(53, 20.));
        assert_eq!(rule.rung(53, 30.), Some(Rung::DoubleDouble));
        assert_eq!(rule.rung(53, 71.), Some(Rung::DoubleDouble));
        assert_eq!(rule.rung(53, 72.), Some(Rung::Arbitrary(192)));
        // double-double -> MPFR at a whole number of limbs, L more bits for
        // MPFR's tracking.
        assert!(rule.flags(106, 60.));
        assert!(!rule.flags(106, 50.));
        assert_eq!(rule.rung(106, 60.), Some(Rung::Arbitrary(192)));
        assert_eq!(rule.rung(106, 75.), Some(Rung::Arbitrary(256)));
        let untracked = LadderRule { mpfr_tracking: 0., ..rule };
        assert_eq!(untracked.rung(106, 60.), Some(Rung::Arbitrary(128)));
        // MPFR p -> MPFR p'.
        assert_eq!(rule.rung(200, 120.), Some(Rung::Arbitrary(384)));
        // A caller's target: binary64 at 1e-5 flags beyond 36 bits lost.
        let loose = LadderRule { target: Some(1e-5), ..rule };
        assert!(!loose.flags(53, 36.));
        assert!(loose.flags(53, 37.));
        assert_eq!(loose.rung(106, 80.), Some(Rung::Arbitrary(192)));
        // Above the cap: none.
        assert_eq!(rule.rung(106, 500.), None);
        assert_eq!(lost_bits(1., 1024.), 10.);
        assert_eq!(lost_bits(0., 0.), 0.);
        assert!(lost_bits(0., 1.).is_infinite());
        assert_eq!(estimate(53, 23.), 2f64.powi(-30));
    }

    #[test]
    fn projection_makes_pairs_exactly_light_like() {
        let bits = 192;
        let f = |x: f64| Float::with_val(bits, x);
        // Three shifts with V1 - V0 and V2 - V1 light-like only to binary64.
        let p = [500.0, 0.3, 0.4, (500.0f64 * 500.0 - 0.25).sqrt()];
        let k = [300.0, 299.0, 0.0, (300.0f64 * 300.0 - 299.0 * 299.0).sqrt()];
        let mut shifts = vec![
            [f(0.), f(0.), f(0.), f(0.)],
            p.map(f),
            [f(p[0] + k[0]), f(p[1] + k[1]), f(p[2] + k[2]), f(p[3] + k[3])],
        ];
        let pairs = vec![(0, 1, f(0.)), (1, 2, f(0.))];
        assert!(project_shells(&mut shifts, &pairs, bits));
        for (i, j, _) in &pairs {
            let d: Vec<Float> = (0..4).map(|mu| shifts[*i][mu].clone() - &shifts[*j][mu]).collect();
            let s = d[0].clone() * &d[0] - &(d[1].clone() * &d[1]) - &(d[2].clone() * &d[2]) - &(d[3].clone() * &d[3]);
            assert!(s.to_f64().abs() < 1e-45, "{}", s.to_f64());
        }
        // The point moved by the binary64 rounding only.
        assert!((shifts[1][3].to_f64() - p[3]).abs() < 1e-10);
    }

    #[test]
    fn projection_converges_from_1024_bits_on() {
        // Audit m8: the tolerance `2^-bits` underflowed binary64 at 1024 bits.
        for bits in [1024, 1100] {
            let f = |x: f64| Float::with_val(bits, x);
            let p = [500.0, 0.3, 0.4, (500.0f64 * 500.0 - 0.25).sqrt()];
            let mut shifts = vec![[f(0.), f(0.), f(0.), f(0.)], p.map(f)];
            assert!(project_shells(&mut shifts, &[(0, 1, f(0.))], bits), "{bits}");
            let d: Vec<Float> = (0..4).map(|mu| shifts[1][mu].clone() - &shifts[0][mu]).collect();
            let s = d[0].clone() * &d[0] - &(d[1].clone() * &d[1]) - &(d[2].clone() * &d[2]) - &(d[3].clone() * &d[3]);
            assert!(s.to_f64().abs() < 1e-290, "{bits}: {:e}", s.to_f64());
        }
        assert!(below_bits(1e30, 1100) > 0., "no underflow for the moduli of a lane");
    }
}
