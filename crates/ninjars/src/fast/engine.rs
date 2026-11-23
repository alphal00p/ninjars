//! Batched binary64 port of original Ninja's `Amplitude::evaluate` with its
//! default options (no tests, mu-expansion for the box rational term,
//! `ComplexMasses`), including the rank n+1 path of `Xninja.cc`. Rank n+2
//! (beyond original Ninja) is a separate engine, [`super::engine_x2`],
//! selected once in [`FastAmplitude::new`]; nothing below depends on it.
//! With the opt-in [`FastOptions::conditioned_cuts`] the box, bubble and
//! tadpole stages take that engine's conditioned cut geometry
//! (`bases_x2.rs`, `box_x2.rs`) instead of original Ninja's: a separate
//! copy of those stages ([`conditioned`], generated from the ones below),
//! reduced by the [`ConditionedEngine`] that [`FastAmplitude::new`] chooses
//! once, like the rank n+2 engine. The stages below, and with them the
//! default engine's machine code, are as before the option existed.
//!
//! Every stage processes one cut for all active lanes: geometry for each lane,
//! one numerator callback per cut (both orientations of a triangle or of a
//! rank n+1 tadpole share a call), then lane-local divisions by the uncut
//! denominators and coefficient-level subtractions. All master requests of
//! all cuts and lanes are collected and evaluated with one provider call per
//! family before the final contraction in original Ninja's order.

// Index loops mirror original Ninja's component and lane loops.
#![allow(clippy::needless_range_loop)]

use super::bases_x2::TadpoleBasesX2;
use super::complex::{C64, CMom, RMom, mp_cc, mp_cr, mp_rr};
use super::declared_rank::{LowerRank, evaluate_declared_n2};
use super::engine_x2::{DynMasters, DynNumerator, RankN2Engine};
use super::geometry::{
    Basis, box_solutions, bubble_solution, cmom_from_real, combine_real2, gram_det, mp2,
    reference_vectors_ok, scale, sub, tadpole_candidates, triangle_solution,
};
use super::masters::{FastMaster, FastMasters};
use super::numerator::{FastNumerator, t2_len_le3, t3_len};
use super::subtractions as subs;
use super::{
    FastInput, FastRefinement, FastRefinementSource, FastResult, FastStatus, Instability,
    InstabilityTest,
};
use crate::conditioning::{
    BoxMuConditioning, BoxMuPropagation, box_mu_numerator_ratio, box_mu_refused, box_mu_rounding,
};
use crate::{Error, Result};
use std::cell::{Cell, RefCell};

/// The rank <= n+1 stages with the conditioned cut geometry, generated from
/// the default ones below (`tests_conditioned.rs`; not formatted, so that
/// regeneration is exact).
#[rustfmt::skip]
mod conditioned;

/// Options of the fast engine. [`Default`] reproduces original Ninja's
/// defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FastOptions {
    /// Minimal cut size (Ninja's `setCutStop`): stages with fewer cut
    /// propagators are skipped (and contribute nothing). `1` (or `0`)
    /// performs the full reduction.
    pub cut_stop: usize,
    /// Ninja's `INFRARED_EPS` (`1e-9`): massless-momentum threshold of the
    /// cut bases and of the scaleless-bubble skip rule.
    pub infrared_eps: f64,
    /// Ninja's floating-point stability threshold (`REAL_MIN = 1e20 *
    /// DBL_MIN`): a Gram determinant, uncut denominator or expansion leading
    /// term with a smaller taxicab norm marks the lane
    /// [`FastStatus::Unstable`].
    pub stability_threshold: f64,
    /// Precision-aware complement of `stability_threshold` (default
    /// `2^10 ε`, Ninja's `REAL_EPS = 10³ ε`): the box and triangle checks
    /// (Gram determinants, the box normal's factors `p1.e3` and `p1.e4`, uncut
    /// denominators and Laurent leading coefficients) are also unstable when
    /// their taxicab norm is at most this fraction of the magnitudes of the
    /// terms they are computed from. A quantity that vanishes exactly at a
    /// singular point is computed as rounding noise of that size as often as
    /// it is computed as zero, which Ninja's absolute threshold alone accepts
    /// (the generic engine's kernels apply the same test in every precision).
    /// The rank n+2 engine (`engine_x2.rs`) applies it to the same checks.
    /// Bubbles and tadpoles keep Ninja's absolute check: their directions
    /// involve Ninja's fixed irrational reference vectors. `0` restores Ninja's
    /// checks.
    ///
    /// Neither threshold is the box mu-expansion's conditioning bound
    /// ([`crate::conditioning`]), which refuses a lane whose predicted loss of
    /// digits exceeds binary64's precision minus a margin: it stays active
    /// when both are `0`.
    pub relative_tolerance: f64,
    /// Rank <= n+1 only (opt-in): take the cut geometry of the rank n+2
    /// engine instead of original Ninja's, per cut and lane: the bubble
    /// reference and the tadpole basis that best condition the uncut lines
    /// (`bases_x2.rs`) and the box normal, center and solutions of
    /// `box_x2.rs`. This trades bitwise agreement with original Ninja (the
    /// default, `false`) for accuracy on lanes where an uncut line is nearly
    /// orthogonal to Ninja's fixed reference vectors, e.g. propagator pairs
    /// whose shift difference has a zero time component. Rank n+2 always
    /// uses the conditioned geometry. Read once, by [`FastAmplitude::new`],
    /// which then chooses the separately compiled [`ConditionedEngine`]; the
    /// default engine's code does not depend on it.
    pub conditioned_cuts: bool,
    /// Re-reduce in double-double, from the triple cuts down, every lane
    /// whose finite part the binary64 reduction measurably cancels beyond
    /// [`Self::rescue_target`] (default `false`: on for callers that keep a
    /// test of their own behind it, as pyAmpliCol's checked mode does;
    /// `docs/triangle-gram-conditioning.md`): `2^-53` times the moduli of its
    /// integrated terms (each coefficient times its master, and each rational
    /// term) exceeds the target times its finite part. `false` (a caller's
    /// "reduction checks off"): no trigger bookkeeping, no re-reduction, every
    /// value the binary64 reduction's bit for bit.
    /// Lanes of numerators without [`FastNumerator::tensor_coefficients`]
    /// keep their binary64 values. Rank <= n+1 (also lanes of a declared rank
    /// n+2 reduced at a lower rank, and with [`Self::conditioned_cuts`]); not
    /// the rank n+2 engine.
    pub cancellation_rescue: bool,
    /// The target relative accuracy of [`Self::cancellation_rescue`]: a lane
    /// is re-reduced when its measured estimate `2^-53 Σ|terms| / |finite|`
    /// exceeds it, and a re-reduced lane whose own estimate still exceeds it
    /// is refused ([`FastStatus::Unstable`],
    /// [`super::InstabilityTest::CancellationRescue`]). `None`:
    /// [`DEFAULT_RESCUE_TARGET`]. Callers with an accuracy target of their
    /// own pass it here (see `docs/triangle-gram-conditioning.md`, section 3).
    pub rescue_target: Option<f64>,
}

impl Default for FastOptions {
    fn default() -> Self {
        Self {
            cut_stop: 1,
            infrared_eps: 1.0e-9,
            stability_threshold: 1.0e20 * f64::MIN_POSITIVE,
            relative_tolerance: 1024. * f64::EPSILON,
            conditioned_cuts: false,
            cancellation_rescue: false,
            rescue_target: None,
        }
    }
}

/// The [`FastOptions`] that the stages read: all but `conditioned_cuts`,
/// which [`FastAmplitude::new`] consumes when it chooses the engine. Its
/// fields are those of [`FastOptions`] before `conditioned_cuts` existed, so
/// that [`FastAmplitude`] keeps its layout (a larger options field moves the
/// other fields, and with them every stage's machine code).
#[derive(Clone, Copy, Debug)]
pub(super) struct StageOptions {
    pub(super) cut_stop: usize,
    pub(super) infrared_eps: f64,
    pub(super) stability_threshold: f64,
    pub(super) relative_tolerance: f64,
    pub(super) cancellation_rescue: bool,
    /// [`FastOptions::rescue_target`], resolved.
    pub(super) rescue_target: f64,
}

impl StageOptions {
    pub(super) fn new(options: &FastOptions) -> Self {
        Self {
            cut_stop: options.cut_stop,
            infrared_eps: options.infrared_eps,
            stability_threshold: options.stability_threshold,
            relative_tolerance: options.relative_tolerance,
            cancellation_rescue: options.cancellation_rescue,
            rescue_target: options.rescue_target.unwrap_or(DEFAULT_RESCUE_TARGET),
        }
    }

    /// The [`FastOptions`] these come from.
    fn with_conditioned_cuts(self, conditioned_cuts: bool) -> FastOptions {
        FastOptions {
            cut_stop: self.cut_stop,
            infrared_eps: self.infrared_eps,
            stability_threshold: self.stability_threshold,
            relative_tolerance: self.relative_tolerance,
            conditioned_cuts,
            cancellation_rescue: self.cancellation_rescue,
            rescue_target: Some(self.rescue_target),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct TriangleData {
    e3: CMom,
    e4: CMom,
    c: [C64; 15],
}

#[derive(Clone, Copy, Debug, Default)]
struct BubbleData {
    e2: RMom,
    e3: CMom,
    e4: CMom,
    c: [C64; 20],
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct TadpoleData {
    pub(super) c0: C64,
    pub(super) c14: C64,
    pub(super) c15: C64,
    pub(super) e34: f64,
}

/// Scalar products of one uncut propagator `j` with the expansion vectors of
/// the current cut, shared by the Laurent division and the subtractions.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Uncut {
    /// `e3.(V_j - V_0)` and its reciprocal (`2 e3k` is the division's `d0`).
    pub(super) e3k: C64,
    pub(super) ie3k: C64,
    /// `e4.(V_j - V_0)` (unscaled `e4`; `e4eff` products follow by scaling).
    pub(super) e4k: C64,
    /// Bubble `a0 = e1effl` and `a1 = e2eff` products.
    pub(super) e1k: C64,
    pub(super) e2k: f64,
    /// Constant part of the division's `d1`.
    pub(super) d1: C64,
    /// `(V_j - V_0)² + m0² - m_j²`.
    pub(super) f: C64,
}

#[derive(Clone, Copy, Debug, Default)]
struct BoxRow {
    lane: usize,
    vort: CMom,
    q: [CMom; 2],
    mid: CMom,
}

#[derive(Clone, Copy, Debug, Default)]
struct TriangleRow {
    lane: usize,
    e3: CMom,
    e4: CMom,
    amu: CMom,
    param: C64,
    e34: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct BubbleRow {
    lane: usize,
    e: Basis,
    /// Ninja's `e1effl = b0 e2` (before the `-V0` shift) and `e2eff = e1 + b1 e2`.
    b0: C64,
    b1: f64,
    e1eff: CMom,
    e2eff: RMom,
    param: [C64; 3],
}

#[derive(Clone, Copy, Debug, Default)]
struct TadpoleRow {
    lane: usize,
    basis: usize,
}

/// Lane-major staging of numerator callback arguments.
#[derive(Default)]
struct Calls {
    lanes: Vec<usize>,
    v0: Vec<CMom>,
    v1: Vec<CMom>,
    v2: Vec<CMom>,
    v3: Vec<CMom>,
    scalar: Vec<C64>,
    params: Vec<[C64; 3]>,
    out: Vec<C64>,
}

impl Calls {
    fn clear(&mut self) {
        self.lanes.clear();
        self.v0.clear();
        self.v1.clear();
        self.v2.clear();
        self.v3.clear();
        self.scalar.clear();
        self.params.clear();
    }
    fn prepare_out(&mut self, width: usize) {
        self.out.clear();
        self.out.resize(self.lanes.len() * width, C64::ZERO);
    }
}

/// One master family's batched request.
#[derive(Default)]
pub(super) struct MasterBatch {
    pub(super) args: Vec<C64>,
    pub(super) out: Vec<[C64; 3]>,
    pub(super) ok: Vec<bool>,
}

impl MasterBatch {
    pub(super) fn run<M: FastMasters + ?Sized>(
        &mut self,
        family: FastMaster,
        masters: &mut M,
    ) -> Result<()> {
        let rows = self.args.len() / family.arity();
        self.out.clear();
        self.out.resize(rows * family.outputs(), [C64::ZERO; 3]);
        self.ok.clear();
        self.ok.resize(rows, false);
        if rows == 0 {
            return Ok(());
        }
        masters.evaluate(family, &self.args, &mut self.out, &mut self.ok)
    }
}

/// Precomputed cut topology in original Ninja's (lexicographic) cut order.
pub(super) struct Topology {
    pub(super) n: usize,
    pub(super) rank: usize,
    pub(super) boxes: Vec<[usize; 4]>,
    pub(super) triangles: Vec<[usize; 3]>,
    pub(super) bubbles: Vec<[usize; 2]>,
    pub(super) box_complements: Vec<usize>,
    pub(super) triangle_complements: Vec<usize>,
    pub(super) bubble_complements: Vec<usize>,
    pub(super) tadpole_complements: Vec<usize>,
    /// `(triangle, uncut)` parents of each bubble (`isSubPartition2of3`).
    pub(super) bubble_parents: Vec<Vec<(usize, usize)>>,
    /// `(triangle, [uncut0, uncut1])` parents of each tadpole.
    pub(super) tadpole_triangle_parents: Vec<Vec<(usize, [usize; 2])>>,
    /// `(bubble, uncut)` parents of each tadpole.
    pub(super) tadpole_bubble_parents: Vec<Vec<(usize, usize)>>,
}

fn complement(part: &[usize], n: usize, out: &mut Vec<usize>) {
    out.extend((0..n).filter(|i| !part.contains(i)));
}

impl Topology {
    pub(super) fn new(n: usize, rank: usize) -> Self {
        let mut boxes = Vec::new();
        let mut triangles = Vec::new();
        let mut bubbles = Vec::new();
        for a in 0..n {
            for b in a + 1..n {
                bubbles.push([a, b]);
                for c in b + 1..n {
                    triangles.push([a, b, c]);
                    for d in c + 1..n {
                        boxes.push([a, b, c, d]);
                    }
                }
            }
        }
        boxes.sort_unstable();
        triangles.sort_unstable();
        bubbles.sort_unstable();
        let mut box_complements = Vec::new();
        let mut triangle_complements = Vec::new();
        let mut bubble_complements = Vec::new();
        let mut tadpole_complements = Vec::new();
        for p in &boxes {
            complement(p, n, &mut box_complements);
        }
        for p in &triangles {
            complement(p, n, &mut triangle_complements);
        }
        for p in &bubbles {
            complement(p, n, &mut bubble_complements);
        }
        for i in 0..n {
            complement(&[i], n, &mut tadpole_complements);
        }
        let bubble_parents = bubbles
            .iter()
            .map(|b| {
                triangles
                    .iter()
                    .enumerate()
                    .filter_map(|(t, p)| {
                        if b[0] == p[0] && b[1] == p[1] {
                            Some((t, p[2]))
                        } else if b[0] == p[0] && b[1] == p[2] {
                            Some((t, p[1]))
                        } else if b[0] == p[1] && b[1] == p[2] {
                            Some((t, p[0]))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();
        let tadpole_triangle_parents = (0..n)
            .map(|a| {
                triangles
                    .iter()
                    .enumerate()
                    .filter_map(|(t, p)| {
                        if a == p[0] {
                            Some((t, [p[1], p[2]]))
                        } else if a == p[1] {
                            Some((t, [p[0], p[2]]))
                        } else if a == p[2] {
                            Some((t, [p[0], p[1]]))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();
        let tadpole_bubble_parents = (0..n)
            .map(|a| {
                bubbles
                    .iter()
                    .enumerate()
                    .filter_map(|(b, p)| {
                        if a == p[0] {
                            Some((b, p[1]))
                        } else if a == p[1] {
                            Some((b, p[0]))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();
        Self {
            n,
            rank,
            boxes,
            triangles,
            bubbles,
            box_complements,
            triangle_complements,
            bubble_complements,
            tadpole_complements,
            bubble_parents,
            tadpole_triangle_parents,
            tadpole_bubble_parents,
        }
    }

    pub(super) fn rmn(&self) -> isize {
        self.rank as isize - self.n as isize
    }
    pub(super) fn x1(&self) -> bool {
        self.rank == self.n + 1
    }
}

#[inline(always)]
pub(super) fn sub_cr(a: &CMom, b: &RMom) -> CMom {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

#[inline(always)]
pub(super) fn scale_c(a: &CMom, s: f64) -> CMom {
    [a[0] * s, a[1] * s, a[2] * s, a[3] * s]
}

/// `log2 λ²` of the box-Gram bound (`crate::conditioning`,
/// `docs/box-gram-conditioning.md`) from a box's two solutions `l` (centred on
/// its first line): `λ² = (|a|² + |t v|²) / S² = (|l+|² + |l-|²) / (2 S²)` (the
/// parallelogram identity), `S²` the largest squared shift or squared mass of
/// its lines, Euclidean norms of the components.
pub(super) fn box_lambda_bits(l: &[CMom; 2], v: &[RMom], m: &[C64], p: &[usize; 4]) -> f64 {
    let norm = |x: &CMom| x.iter().map(|z| z.norm_sqr()).sum::<f64>();
    let scale = p
        .iter()
        .map(|&i| v[i].iter().map(|x| x * x).sum::<f64>().max(m[i].abs()))
        .fold(0., f64::max);
    ((norm(&l[0]) + norm(&l[1])) / (2. * scale)).log2()
}

/// The predicted relative error of a box cut's coefficients in a carrier of
/// `bits` bits by the box-Gram model: `2^((r + 1)/2 log2 λ² + 11.3 - bits)`
/// (`docs/box-gram-conditioning.md`, section 3: the slope, and the largest
/// measured excess of the loss over it).
pub(super) fn box_gram_error(lambda_bits: f64, rank: usize, bits: u32) -> f64 {
    let (a, b) = crate::conditioning::box_gram_slope(rank);
    2f64.powf(a as f64 / b as f64 * lambda_bits + BOX_GRAM_EXCESS_BITS - f64::from(bits))
}

/// `D = (q+V)² - m²` (Ninja's `Den` at `mu² = 0`).
#[inline(always)]
pub(super) fn den(q: &CMom, v: &RMom, m2: C64) -> C64 {
    let s = [q[0] + v[0], q[1] + v[1], q[2] + v[2], q[3] + v[3]];
    mp_cc(&s, &s) - m2
}

/// Ninja's stability check on a taxicab norm `x`, plus the precision-aware
/// relative test of [`FastOptions::relative_tolerance`] against the scale of
/// the terms `x` was computed from. NaN is unstable.
#[derive(Clone, Copy)]
pub(super) struct Stability {
    threshold: f64,
    tolerance: f64,
}

impl Stability {
    pub(super) fn new(options: &StageOptions) -> Self {
        Self {
            threshold: options.stability_threshold,
            tolerance: options.relative_tolerance,
        }
    }
    /// Ninja's absolute check only.
    #[inline(always)]
    pub(super) fn absolute(self, x: f64) -> bool {
        let stable = x >= self.threshold;
        if !stable {
            failed_check(x, self.threshold);
        }
        stable
    }
    #[inline(always)]
    pub(super) fn relative(self, x: f64, scale: f64) -> bool {
        let stable = x >= self.threshold && x > self.tolerance * scale;
        if !stable {
            failed_check(x, self.threshold.max(self.tolerance * scale));
        }
        stable
    }
    /// [`Self::relative`] against `scale()`, evaluated only when `x` does not
    /// already exceed the tolerance times `bound >= scale()`.
    #[inline(always)]
    pub(super) fn relative_within(self, x: f64, bound: f64, scale: impl FnOnce() -> f64) -> bool {
        if x >= self.threshold && x > self.tolerance * bound {
            return true;
        }
        let scale = scale();
        let stable = x >= self.threshold && x > self.tolerance * scale;
        if !stable {
            failed_check(x, self.threshold.max(self.tolerance * scale));
        }
        stable
    }
}

/// Scale of `2 a.(b - c)` for real `b`, `c`: `2 Σ_μ |a_μ|₁ (|b_μ| + |c_μ|)`.
#[inline(always)]
pub(super) fn difference_scale(a: &CMom, b: &RMom, c: &RMom) -> f64 {
    2. * (0..4)
        .map(|mu| a[mu].taxicab() * (b[mu].abs() + c[mu].abs()))
        .sum::<f64>()
}

/// Scale of `(q + v)² - m²`: `Σ_μ (|q_μ|₁ + |v_μ|)² + |m²|₁`.
#[inline(always)]
pub(super) fn den_scale(q: &CMom, v: &RMom, m2: C64) -> f64 {
    m2.taxicab()
        + (0..4)
            .map(|mu| {
                let t = q[mu].taxicab() + v[mu].abs();
                t * t
            })
            .sum::<f64>()
}

/// Scale of the Gram determinant `k1² k2² - (k1.k2)²` of real momenta.
#[inline(always)]
pub(super) fn gram_scale(k1: &RMom, k2: &RMom) -> f64 {
    let e = |k: &RMom| k.iter().map(|x| x * x).sum::<f64>();
    2. * e(k1) * e(k2)
}

/// `Σ_μ |a_μ|₁`: with `vmax >= |V^mu|`, `4 vmax` times this bounds
/// [`difference_scale`].
#[inline(always)]
pub(super) fn taxicab_sum(a: &CMom) -> f64 {
    a.iter().map(|z| z.taxicab()).sum()
}

/// Three directions spanning a box's hyperplane for the conditioning bound's
/// numerator scale (`crate::conditioning`): the pair sums of the shift
/// differences from the first cut line, which are light-like only by accident.
pub(super) fn box_mu_reference_directions(v1: &RMom, v2: &RMom, v3: &RMom, v4: &RMom) -> [RMom; 3] {
    let (a, b, c) = (sub(v2, v1), sub(v3, v1), sub(v4, v1));
    let add = |x: &RMom, y: &RMom| [x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]];
    [add(&a, &b), add(&b, &c), add(&c, &a)]
}

/// The hot path's upper bound of the Euclidean `R_j` from the taxicab scale
/// `S = 4 max|V| Σ_μ |v_μ|₁` and `|d0|₁` (`crate::conditioning`): the
/// Euclidean scale is at most `2 S` and `|d0| >= |d0|₁ / sqrt(2)`, so
/// `R_j <= 2 sqrt(2) S / |d0|₁ <= 3 S / |d0|₁`.
pub(super) const BOX_MU_PRETEST: f64 = 3.;

/// Euclidean norm `sqrt(Σ_μ |a_μ|²)` of a complex vector's components.
#[inline]
pub(super) fn euclidean_norm(a: &CMom) -> f64 {
    a.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

/// `a - b` for complex vectors.
#[inline]
pub(super) fn sub_cc(a: &CMom, b: &CMom) -> CMom {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

/// The conditioning bound's `R_j = 2 |v| |V_j - V_first| / |2 v.(V_j - V_first)|`
/// (Euclidean norms of the components) over the uncut lines `cp` of a box with
/// normal `vort` and first line `first`.
pub(super) fn euclidean_conditioning(
    vort: &CMom,
    v: &[RMom],
    first: usize,
    cp: &[usize],
) -> BoxMuConditioning {
    let norm = euclidean_norm(vort);
    let mut conditioning = BoxMuConditioning::ONE;
    for &j in cp {
        let kk = sub(&v[j], &v[first]);
        let d0 = mp_cr(vort, &kk) * 2.;
        let length = kk.iter().map(|x| x * x).sum::<f64>().sqrt();
        conditioning.line(2. * norm * length, d0.abs());
    }
    conditioning
}

/// `x` scaled to the taxicab norm `norm`, as a complex vector.
pub(super) fn scaled_reference(x: &RMom, norm: f64) -> CMom {
    let size: f64 = x.iter().map(|c| c.abs()).sum();
    let f = if size > 0. { norm / size } else { 0. };
    x.map(|c| C64::real(c * f))
}

/// Scale of the real-by-complex product `a.k`: `Σ_μ |a_μ|₁ |k_μ|`.
#[inline(always)]
pub(super) fn dot_scale(a: &CMom, k: &RMom) -> f64 {
    (0..4).map(|mu| a[mu].taxicab() * k[mu].abs()).sum()
}

#[inline(always)]
pub(super) fn is_zero(z: C64) -> bool {
    z.re == 0. && z.im == 0.
}

/// Mark a lane unstable by `test`; its later cuts and its results are
/// skipped. Its [`Instability`] takes the last failed check's quantity and
/// bound ([`failed_check`]).
pub(super) fn fail(
    alive: &mut [bool],
    status: &mut [FastStatus],
    lane: usize,
    test: InstabilityTest,
) {
    // Taken in every case: a failed check never outlives its `fail`.
    let (value, bound) = FAILED_CHECK
        .with(Cell::take)
        .unwrap_or((f64::NAN, f64::NAN));
    if alive[lane] {
        alive[lane] = false;
        status[lane] = FastStatus::Unstable;
        INSTABILITIES.with(|list| {
            list.borrow_mut()
                .push((lane, Instability { test, value, bound }))
        });
    }
}

thread_local! {
    /// The quantity and bound of the last failed stability check of this
    /// thread ([`Stability`]), taken by the next [`fail`].
    static FAILED_CHECK: Cell<Option<(f64, f64)>> = const { Cell::new(None) };
    /// The instabilities of this thread's last [`FastAmplitude::evaluate`]
    /// ([`take_instabilities`]).
    static INSTABILITIES: RefCell<Vec<(usize, Instability)>> = const { RefCell::new(Vec::new()) };
}

/// Record a failed stability check (cold: the passing path is untouched).
#[cold]
#[inline(never)]
fn failed_check(value: f64, bound: f64) {
    FAILED_CHECK.with(|check| check.set(Some((value, bound))));
}

/// Record a lane the box mu-expansion's conditioning screen refused
/// (`crate::conditioning::box_mu_refused`): its predicted rounding error and
/// the limit, `2^-BOX_MU_TARGET_BITS` of its finite part.
#[cold]
#[inline(never)]
pub(super) fn screen_refused(lane: usize, flagged: f64, finite: f64) {
    let value = flagged * 2f64.powi(-(f64::MANTISSA_DIGITS as i32));
    let bound = finite * 2f64.powi(-crate::conditioning::BOX_MU_TARGET_BITS);
    INSTABILITIES.with(|list| {
        list.borrow_mut().push((
            lane,
            Instability {
                test: InstabilityTest::BoxMuConditioning,
                value,
                bound,
            },
        ))
    });
}

/// `test`, or [`InstabilityTest::EuclideanPlane`] when the cut basis `e3`
/// of the real momenta `k1`, `k2` is not finite because they span a
/// Euclidean plane (`gram_det(k1, k2) > 0`: `Basis::from_pair` takes the
/// square root of its negative); then the failed check becomes the Gram
/// determinant against 0. Called on the failing path only.
#[cold]
#[inline(never)]
pub(super) fn euclidean_or(
    test: InstabilityTest,
    e3: &CMom,
    k1: &RMom,
    k2: &RMom,
) -> InstabilityTest {
    let gram = gram_det(k1, k2);
    if e3.iter().all(|z| z.is_finite()) || !(gram > 0.) {
        return test;
    }
    FAILED_CHECK.with(|check| check.set(Some((gram, 0.))));
    InstabilityTest::EuclideanPlane
}

/// Whether a triangle cut of `lane` has two real momenta spanning a
/// Euclidean plane (its basis not finite, `gram_det > 0`), which made the
/// lane's result non-finite: a triangle without uncut lines (n = 3) has no
/// check that names it. Records [`InstabilityTest::EuclideanPlane`] with the
/// first such triangle's Gram determinant. Called on the non-finite path
/// only.
#[cold]
#[inline(never)]
pub(super) fn plane_refused(topo: &Topology, input: &FastInput, lane: usize, eps: f64) -> bool {
    let n = topo.n;
    let v = &input.shifts[lane * n..(lane + 1) * n];
    let plane = topo.triangles.iter().find_map(|&[c1, c2, c3]| {
        let (k0, k2) = (sub(&v[c2], &v[c1]), sub(&v[c1], &v[c3]));
        let gram = gram_det(&k2, &k0);
        let basis = Basis::from_pair(&k2, &k0, eps);
        (gram > 0. && !basis.e3.iter().all(|z| z.is_finite())).then_some(gram)
    });
    let Some(gram) = plane else {
        return false;
    };
    FAILED_CHECK.with(Cell::take);
    INSTABILITIES.with(|list| {
        list.borrow_mut().push((
            lane,
            Instability {
                test: InstabilityTest::EuclideanPlane,
                value: gram,
                bound: 0.,
            },
        ))
    });
    true
}

/// Forget this thread's instabilities: [`FastAmplitude::evaluate`] starts
/// with it, so that they stay bounded by one call's lanes.
#[inline]
fn clear_instabilities() {
    FAILED_CHECK.with(Cell::take);
    INSTABILITIES.with(|list| list.borrow_mut().clear());
}

/// The number of instabilities recorded on this thread (the rank n+2 path
/// remaps those of a sub-batch, [`remap_instabilities`]).
pub(super) fn instabilities_len() -> usize {
    INSTABILITIES.with(|list| list.borrow().len())
}

/// Map the lanes of the instabilities recorded since `from` through `map`
/// (a sub-batch's lanes to the batch's).
pub(super) fn remap_instabilities(from: usize, map: impl Fn(usize) -> usize) {
    INSTABILITIES.with(|list| {
        for entry in &mut list.borrow_mut()[from..] {
            entry.0 = map(entry.0);
        }
    });
}

/// The number of rescue records on this thread and their remapping, as
/// [`instabilities_len`] and [`remap_instabilities`] (the lanes of a declared
/// rank n+2 batch's lowered sub-batches).
pub(super) fn rescues_len() -> usize {
    RESCUES.with(|list| list.borrow().len())
}

pub(super) fn remap_rescues(from: usize, map: impl Fn(usize) -> usize) {
    RESCUES.with(|list| {
        for entry in &mut list.borrow_mut()[from..] {
            entry.0 = map(entry.0);
        }
    });
}

/// The instabilities of the calling thread's last
/// [`FastAmplitude::evaluate`] call (each call starts by forgetting the
/// previous one's), `(lane, Instability)` with `lane` the lane of its input,
/// in the order the lanes failed: which test made each
/// [`FastStatus::Unstable`] lane unstable, the tested quantity and its bound
/// ([`Instability`]). Callers that read them take them after each
/// evaluation, on the evaluating thread. Recorded on the failing path only.
pub fn take_instabilities() -> Vec<(usize, Instability)> {
    FAILED_CHECK.with(Cell::take);
    INSTABILITIES.with(|list| std::mem::take(&mut *list.borrow_mut()))
}

/// Polynomial division of a triangle/tadpole Laurent expansion
/// (`divpolyby<3>`) by `d0 t + d1 + (d2[0] + d2[1] mu²)/t`, given `1/d0`.
#[inline(always)]
pub(super) fn divide_t3(num: &mut [C64], nterms: usize, d0inv: C64, d1: C64, d2: [C64; 2]) {
    match nterms {
        1 => divide_t3_n::<1>(num, d0inv, d1, d2),
        2 => divide_t3_n::<2>(num, d0inv, d1, d2),
        3 => divide_t3_n::<3>(num, d0inv, d1, d2),
        4 => divide_t3_n::<4>(num, d0inv, d1, d2),
        _ => divide_t3_n::<5>(num, d0inv, d1, d2),
    }
}

/// [`divide_t3`] for a compile-time number of orders (fully unrolled).
#[inline(always)]
fn divide_t3_n<const NT: usize>(num: &mut [C64], d0inv: C64, d1: C64, d2: [C64; 2]) {
    const OFF: [usize; 6] = [0, 1, 2, 4, 6, 9];
    let num = &mut num[..OFF[NT]];
    for i in 0..NT {
        let size = 1 + i / 2;
        let o = OFF[i];
        for j in 0..size {
            num[o + j] *= d0inv;
        }
        if i + 1 < NT {
            let o1 = OFF[i + 1];
            for j in 0..size {
                let v = num[o + j];
                num[o1 + j] -= d1 * v;
            }
            if i + 2 < NT {
                let o2 = OFF[i + 2];
                for j in 0..size {
                    let v = num[o + j];
                    num[o2 + j] -= d2[0] * v;
                    num[o2 + j + 1] -= d2[1] * v;
                }
            }
        }
    }
}

/// Polynomial division of a bubble Laurent expansion (`divpolyby<2>`) by
/// `d0 t + (d1[0] + d1[1] x) + (d2[0] + d2[1] mu² + d2[2] x + d2[3] x²)/t`,
/// given `1/d0`.
#[inline(always)]
pub(super) fn divide_t2(num: &mut [C64], nterms: usize, d0inv: C64, d1: [C64; 2], d2: [C64; 4]) {
    match nterms {
        1 => divide_t2_n::<1>(num, d0inv, d1, d2),
        2 => divide_t2_n::<2>(num, d0inv, d1, d2),
        3 => divide_t2_n::<3>(num, d0inv, d1, d2),
        _ => divide_t2_n::<4>(num, d0inv, d1, d2),
    }
}

/// [`divide_t2`] for a compile-time number of orders (fully unrolled).
#[inline(always)]
fn divide_t2_n<const NT: usize>(num: &mut [C64], d0inv: C64, d1: [C64; 2], d2: [C64; 4]) {
    const OFF: [usize; 5] = [0, 1, 3, 7, 13];
    let num = &mut num[..OFF[NT]];
    for i in 0..NT {
        let size = 1 + i + i / 2 + i / 3;
        let o = OFF[i];
        for j in 0..size {
            num[o + j] *= d0inv;
        }
        if i + 1 < NT {
            let n1 = OFF[i + 1];
            let v0 = num[o];
            num[n1] -= d1[0] * v0;
            match i + 1 {
                1 => num[n1 + 1] -= d1[1] * v0,
                2 => {
                    let v1 = num[o + 1];
                    num[n1 + 2] -= d1[0] * v1;
                    num[n1 + 2] -= d1[1] * v0;
                    num[n1 + 3] -= d1[1] * v1;
                }
                _ => {
                    let (v1, v2, v3) = (num[o + 1], num[o + 2], num[o + 3]);
                    num[n1 + 1] -= d1[0] * v1;
                    num[n1 + 2] -= d1[0] * v2;
                    num[n1 + 4] -= d1[0] * v3;
                    num[n1 + 2] -= d1[1] * v0;
                    num[n1 + 3] -= d1[1] * v1;
                    num[n1 + 4] -= d1[1] * v2;
                    num[n1 + 5] -= d1[1] * v3;
                }
            }
            if i + 2 < NT {
                let n2 = OFF[i + 2];
                num[n2] -= d2[0] * v0;
                if i == 0 {
                    num[n2 + 1] -= d2[1] * v0;
                    num[n2 + 2] -= d2[2] * v0;
                    num[n2 + 3] -= d2[3] * v0;
                } else {
                    let v1 = num[o + 1];
                    num[n2 + 1] -= d2[1] * v0;
                    num[n2 + 2] -= d2[2] * v0;
                    num[n2 + 4] -= d2[3] * v0;
                    num[n2 + 2] -= d2[0] * v1;
                    num[n2 + 3] -= d2[1] * v1;
                    num[n2 + 4] -= d2[2] * v1;
                    num[n2 + 5] -= d2[3] * v1;
                }
            }
        }
    }
}

type BubbleCorrection = fn(
    &mut [C64],
    &CMom,
    &CMom,
    &[C64],
    &CMom,
    &CMom,
    &[C64; 3],
    C64,
    f64,
    C64,
    C64,
    C64,
    &[C64; 4],
    C64,
);

/// Batched binary64 Ninja reduction for a fixed `(n, rank)` topology.
///
/// Construct once and reuse: all workspaces are retained between calls, so a
/// warmed evaluation with an unchanged number of lanes performs no heap
/// allocation in the engine itself. See [`super`] for conventions.
pub struct FastAmplitude {
    pub(super) topo: Topology,
    options: StageOptions,
    tadpole_bases: [Basis; 3],
    lanes: usize,
    alive: Vec<bool>,
    lane_status: Vec<FastStatus>,
    /// Per lane the first-order rounding estimate of the box mu-expansion
    /// rational terms of the boxes the conditioning screen flagged (units of
    /// the roundoff; 0 when none was flagged), decided against the lane's
    /// finite part in `integrate` (`crate::conditioning`).
    mu_error: Vec<f64>,
    active: Vec<usize>,
    smat: Vec<f64>,
    vsq: Vec<f64>,
    /// Per lane `(max |V_i^mu|, max |m_i²|₁)`: cheap upper bounds of the scales
    /// in the precision-aware stability checks.
    bounds: Vec<(f64, f64)>,
    box_c: Vec<[C64; 2]>,
    /// Per (box, lane) `log2 λ²` of the box-Gram bound (`crate::conditioning`),
    /// for [`FastOptions::cancellation_rescue`]'s trigger.
    box_lambda: Vec<f64>,
    tri: Vec<TriangleData>,
    bub: Vec<BubbleData>,
    tad: Vec<TadpoleData>,
    calls: Calls,
    box_rows: Vec<BoxRow>,
    tri_rows: Vec<TriangleRow>,
    bub_rows: Vec<BubbleRow>,
    tad_rows: Vec<TadpoleRow>,
    second: Vec<usize>,
    uncut: Vec<Uncut>,
    /// Per bubble row and parent triangle: `[a0.et3, a0.et4, a1.et3, a1.et4]`.
    pairs: Vec<[C64; 4]>,
    masters: [MasterBatch; 5],
    staged: Vec<FastResult>,
    /// The engine that replaces this one's stages, present iff
    /// `rank == n + 2` or [`FastOptions::conditioned_cuts`]; it then performs
    /// every evaluation and the fields above stay unused.
    pub(super) replacement: Option<Box<Replacement>>,
}

/// An engine chosen once in [`FastAmplitude::new`] in place of the default
/// rank <= n+1 stages. Both sit behind the one pointer that the rank n+2
/// engine had alone before, so that [`FastAmplitude`]'s layout and the
/// default stages are unchanged by the option and by the lower-rank state of
/// `declared_rank.rs`. (One allocation per engine, so the variants' different
/// sizes do not matter.)
#[allow(clippy::large_enum_variant)]
pub(super) enum Replacement {
    /// Rank n+2 (beyond original Ninja), which always uses the conditioned
    /// geometry; `conditioned_cuts` is the option as given.
    RankN2 {
        engine: RankN2Engine,
        conditioned_cuts: bool,
        /// The rank <= n+1 engines for lanes whose numerator has a lower
        /// rank (`declared_rank.rs`), built on first use.
        lower: Option<Box<LowerRank>>,
    },
    /// Rank <= n+1 with [`FastOptions::conditioned_cuts`].
    Conditioned(ConditionedEngine),
}

/// The rank <= n+1 reduction with the rank n+2 engine's conditioned cut
/// geometry ([`FastOptions::conditioned_cuts`]): the stages of
/// [`conditioned`] on workspaces of their own, with type-erased providers
/// (as for the rank n+2 engine), so that they are compiled once and apart
/// from the default instances (see [`evaluate_conditioned`]).
pub(super) struct ConditionedEngine {
    /// The workspaces; its own `replacement` is `None`.
    amp: FastAmplitude,
    /// Original Ninja's tadpole bases and the extra candidates.
    tadpoles: TadpoleBasesX2,
}

/// The conditioned branch of [`FastAmplitude::evaluate`] (and of the lowered
/// engines of `declared_rank.rs`): a cold, never-inlined entry, so the branch
/// weights (and with them the inlining decisions) of the default rank <= n+1
/// evaluation stay those of an engine without this branch.
///
/// `N` and `M` are [`FastAmplitude::evaluate`]'s, used only to erase the
/// providers' types: they make this entry, and with it the conditioned stages
/// it instantiates (for `dyn` providers, once for all), compiled in the crate
/// that instantiates [`FastAmplitude::evaluate`], next to the default
/// instances. Compiled in this crate instead, the conditioned stages would
/// instantiate here the generic functions they share with the default ones
/// (e.g. `RawVec::grow_one` of the row vectors); thin LTO then keeps one copy
/// of each, in this crate, and the default instances' direct calls to their
/// own copies become calls through the GOT.
#[cold]
#[inline(never)]
pub(super) fn evaluate_conditioned<N: FastNumerator + ?Sized, M: FastMasters + ?Sized>(
    engine: &mut ConditionedEngine,
    input: &FastInput,
    numerator: &mut N,
    masters: &mut M,
    results: &mut [FastResult],
    statuses: &mut [FastStatus],
) -> Result<()> {
    let ConditionedEngine { amp, tadpoles } = engine;
    amp.evaluate_up_to_n1_conditioned::<dyn FastNumerator, dyn FastMasters>(
        input,
        &mut DynNumerator(numerator),
        &mut DynMasters(masters),
        results,
        statuses,
        tadpoles,
    )
}

/// Master batches: D0, C0, rank-2 bubbles, rank-3 bubbles, A0.
const FAMILIES: [FastMaster; 5] = [
    FastMaster::D0,
    FastMaster::C0,
    FastMaster::BubbleRank2,
    FastMaster::BubbleRank3,
    FastMaster::A0,
];

impl FastAmplitude {
    /// A reduction engine for `n >= 1` propagators and `rank <= n + 2`.
    /// Rank n+2 (beyond original Ninja) is reduced by the separate
    /// [`RankN2Engine`], and rank <= n+1 with
    /// [`FastOptions::conditioned_cuts`] by the [`ConditionedEngine`], both
    /// chosen here once.
    pub fn new(n: usize, rank: usize, options: FastOptions) -> Result<Self> {
        if n == 0 {
            return Err(Error::NoPropagators);
        }
        if rank > n + 2 {
            return Err(Error::Rank {
                rank,
                propagators: n,
            });
        }
        let mut amp = Self::workspaces(n, rank, &options);
        amp.replacement = if rank == n + 2 {
            Some(Box::new(Replacement::RankN2 {
                engine: RankN2Engine::new(n, options),
                conditioned_cuts: options.conditioned_cuts,
                lower: None,
            }))
        } else if options.conditioned_cuts {
            Some(Box::new(Replacement::Conditioned(ConditionedEngine {
                amp: Self::workspaces(n, rank, &options),
                tadpoles: TadpoleBasesX2::new(options.infrared_eps),
            })))
        } else {
            None
        };
        Ok(amp)
    }

    /// The default engine's state, without a replacement.
    fn workspaces(n: usize, rank: usize, options: &FastOptions) -> Self {
        Self {
            topo: Topology::new(n, rank),
            tadpole_bases: tadpole_candidates(options.infrared_eps),
            options: StageOptions::new(options),
            lanes: 0,
            alive: Vec::new(),
            lane_status: Vec::new(),
            mu_error: Vec::new(),
            active: Vec::new(),
            smat: Vec::new(),
            vsq: Vec::new(),
            bounds: Vec::new(),
            box_c: Vec::new(),
            box_lambda: Vec::new(),
            tri: Vec::new(),
            bub: Vec::new(),
            tad: Vec::new(),
            calls: Calls::default(),
            box_rows: Vec::new(),
            tri_rows: Vec::new(),
            bub_rows: Vec::new(),
            tad_rows: Vec::new(),
            second: Vec::new(),
            uncut: vec![Uncut::default(); n],
            pairs: Vec::new(),
            masters: Default::default(),
            staged: Vec::new(),
            replacement: None,
        }
    }

    pub fn propagators(&self) -> usize {
        self.topo.n
    }
    pub fn rank(&self) -> usize {
        self.topo.rank
    }
    /// The options the engine was built with.
    pub fn options(&self) -> FastOptions {
        let conditioned_cuts = match self.replacement.as_deref() {
            Some(Replacement::RankN2 {
                conditioned_cuts, ..
            }) => *conditioned_cuts,
            Some(Replacement::Conditioned(_)) => true,
            None => false,
        };
        self.options.with_conditioned_cuts(conditioned_cuts)
    }

    /// Unit tests: the rank n+2 engine, iff `rank == n + 2`.
    #[cfg(test)]
    pub(super) fn x2(&self) -> Option<&RankN2Engine> {
        match self.replacement.as_deref() {
            Some(Replacement::RankN2 { engine, .. }) => Some(engine),
            _ => None,
        }
    }

    fn any_box(&self) -> bool {
        self.topo.n >= 4 && self.options.cut_stop <= 4
    }
    fn any_triangle(&self) -> bool {
        self.topo.n >= 3 && self.options.cut_stop <= 3 && self.topo.rmn() + 3 >= 0
    }
    fn any_bubble(&self) -> bool {
        self.topo.n >= 2 && self.options.cut_stop <= 2 && self.topo.rmn() + 2 >= 0
    }
    fn any_tadpole(&self) -> bool {
        self.options.cut_stop <= 1 && self.topo.rmn() + 1 >= 0
    }

    fn refresh_active(&mut self) {
        let alive = &self.alive;
        self.active.retain(|&lane| alive[lane]);
    }

    /// Reduce and integrate every lane of `input`.
    ///
    /// Successful lanes overwrite `results[lane]`; failed lanes keep their
    /// previous result and receive a failure status. Structural errors (bad
    /// shapes, provider errors) return `Err` and leave results unchanged.
    ///
    /// At rank n+2, a lane whose numerator reports a lower rank `r`
    /// ([`FastNumerator::lane_rank`]) is reduced bitwise as if declared at
    /// rank `r`; [`super::AtDeclaredRank`] prevents it.
    pub fn evaluate<N: FastNumerator + ?Sized, M: FastMasters + ?Sized>(
        &mut self,
        input: &FastInput,
        numerator: &mut N,
        masters: &mut M,
        results: &mut [FastResult],
        statuses: &mut [FastStatus],
    ) -> Result<()> {
        self.evaluate_refined(input, None, numerator, masters, results, statuses)
    }

    /// [`Self::evaluate`] with the optional rescue-precision kinematics of
    /// `input` ([`FastRefinement`]): the re-reduction of flagged lanes
    /// ([`FastOptions::cancellation_rescue`]) builds its geometry, Gram
    /// matrices and invariants from them instead of repairing the binary64
    /// ones. The binary64 pass reads `input` only, bit for bit as without
    /// it. Rank <= n+1 lanes reduced at their declared rank (also with
    /// [`FastOptions::conditioned_cuts`]); the lanes the rank n+2 engine lowers
    /// use the internal repair.
    pub fn evaluate_refined<N: FastNumerator + ?Sized, M: FastMasters + ?Sized>(
        &mut self,
        input: &FastInput,
        refinement: Option<&FastRefinement>,
        numerator: &mut N,
        masters: &mut M,
        results: &mut [FastResult],
        statuses: &mut [FastStatus],
    ) -> Result<()> {
        let n = self.topo.n;
        if let Some(r) = refinement {
            let lanes = input.lanes;
            let s_len = if input.s_matrix.is_some() { lanes * n * n } else { 0 };
            for (name, expected, actual) in [
                ("fast refinement shifts", lanes * n, r.shifts.len()),
                (
                    "fast refinement masses",
                    lanes * n,
                    r.masses_squared.map_or(lanes * n, |m| m.len()),
                ),
                ("fast refinement S-matrix", s_len, r.s_matrix.map_or(s_len, |s| s.len())),
            ] {
                if expected != actual {
                    return Err(Error::Shape {
                        name,
                        expected,
                        actual,
                    });
                }
            }
        }
        self.evaluate_refined_with(
            input,
            refinement.map(|r| r as &dyn FastRefinementSource),
            numerator,
            masters,
            results,
            statuses,
        )
    }

    /// [`Self::evaluate_refined`] with a lazy [`FastRefinementSource`], asked
    /// only for the lanes the re-reduction takes up.
    pub fn evaluate_refined_with<N: FastNumerator + ?Sized, M: FastMasters + ?Sized>(
        &mut self,
        input: &FastInput,
        refinement: Option<&dyn FastRefinementSource>,
        numerator: &mut N,
        masters: &mut M,
        results: &mut [FastResult],
        statuses: &mut [FastStatus],
    ) -> Result<()> {
        clear_instabilities();
        RESCUES.with(|r| r.borrow_mut().clear());
        if masters.requires_scaleless_bubbles() {
            return Err(Error::Unsupported(
                "separate UV/IR poles require scaleless-bubble coefficients; use the generic reduction engine".into(),
            ));
        }

        if let Some(replacement) = self.replacement.as_deref_mut() {
            // Both entries are cold and never inlined.
            return match replacement {
                // The lowered lanes are renumbered: they take the rescue in
                // `evaluate_lowered`, without the refinement.
                Replacement::RankN2 { .. } => {
                    evaluate_declared_n2(self, input, numerator, masters, results, statuses)
                }
                Replacement::Conditioned(engine) => {
                    evaluate_conditioned(engine, input, numerator, masters, results, statuses)?;
                    engine.amp.rescue_after(input, numerator, masters, refinement, results, statuses)
                }
            };
        }
        self.evaluate_up_to_n1(input, numerator, masters, results, statuses)?;
        self.rescue_after(input, numerator, masters, refinement, results, statuses)
    }

    /// [`Self::evaluate`] for the rank <= n+1 engines that `declared_rank.rs`
    /// builds for the lanes of a declared rank n+2 that it reduces at a lower
    /// rank: through the [`ConditionedEngine`] if the engine has one
    /// ([`FastOptions::conditioned_cuts`], inherited from the rank n+2
    /// engine's options), else through the caller's own instance of
    /// [`Self::evaluate_up_to_n1`]. Not through [`Self::evaluate`]: the
    /// dispatch would become recursive, and LLVM then stops inlining it into
    /// its callers.
    pub(super) fn evaluate_lowered<N: FastNumerator + ?Sized, M: FastMasters + ?Sized>(
        &mut self,
        input: &FastInput,
        numerator: &mut N,
        masters: &mut M,
        results: &mut [FastResult],
        statuses: &mut [FastStatus],
    ) -> Result<()> {
        match self.replacement.as_deref_mut() {
            None => {
                self.evaluate_up_to_n1(input, numerator, masters, results, statuses)?;
                self.rescue_after(input, numerator, masters, None, results, statuses)
            }
            Some(Replacement::Conditioned(engine)) => {
                evaluate_conditioned(engine, input, numerator, masters, results, statuses)?;
                engine.amp.rescue_after(input, numerator, masters, None, results, statuses)
            }
            Some(Replacement::RankN2 { .. }) => {
                unreachable!("the lowered engines have rank <= n+1")
            }
        }
    }

    /// [`Self::evaluate`] for rank <= n+1. Kept as its own (never-inlined)
    /// function so that the rank n+2 dispatch above leaves its code as it
    /// was before rank n+2 existed.
    #[inline(never)]
    fn evaluate_up_to_n1<N: FastNumerator + ?Sized, M: FastMasters + ?Sized>(
        &mut self,
        input: &FastInput,
        numerator: &mut N,
        masters: &mut M,
        results: &mut [FastResult],
        statuses: &mut [FastStatus],
    ) -> Result<()> {
        let n = self.topo.n;
        let lanes = input.lanes;
        for (name, expected, actual) in [
            ("fast shifts", lanes * n, input.shifts.len()),
            ("fast masses", lanes * n, input.masses_squared.len()),
            ("fast scales", lanes, input.scales_squared.len()),
            ("fast results", lanes, results.len()),
            ("fast statuses", lanes, statuses.len()),
        ] {
            if expected != actual {
                return Err(Error::Shape {
                    name,
                    expected,
                    actual,
                });
            }
        }
        if let Some(matrix) = input.s_matrix
            && matrix.len() != lanes * n * n
        {
            return Err(Error::Shape {
                name: "fast S-matrix",
                expected: lanes * n * n,
                actual: matrix.len(),
            });
        }
        if numerator.rank() != self.topo.rank {
            return Err(Error::Shape {
                name: "fast numerator rank",
                expected: self.topo.rank,
                actual: numerator.rank(),
            });
        }
        self.prepare(input);
        if self.any_box() {
            self.boxes(input, numerator)?;
        }
        if self.any_triangle() {
            self.triangles(input, numerator)?;
        }
        if self.any_bubble() {
            self.bubbles(input, numerator)?;
        }
        if self.any_tadpole() {
            self.tadpoles(input, numerator)?;
        }
        self.integrate(input, masters)?;
        // Publish only after every provider call succeeded.
        for lane in 0..lanes {
            if self.lane_status[lane] == FastStatus::Success {
                results[lane] = self.staged[lane];
            }
        }
        statuses.copy_from_slice(&self.lane_status[..lanes]);
        Ok(())
    }
}

impl FastAmplitude {
    fn prepare(&mut self, input: &FastInput) {
        let n = self.topo.n;
        let lanes = input.lanes;
        self.lanes = lanes;
        self.alive.clear();
        self.alive.resize(lanes, true);
        self.lane_status.clear();
        self.lane_status.resize(lanes, FastStatus::NotEvaluated);
        self.mu_error.clear();
        self.mu_error.resize(lanes, 0.);
        self.active.clear();
        self.active.extend(0..lanes);
        self.staged.resize(lanes, FastResult::default());
        self.smat.resize(lanes * n * n, 0.);
        self.bounds.resize(lanes, (0., 0.));
        self.vsq.resize(lanes * n, 0.);
        for lane in 0..lanes {
            let v = &input.shifts[lane * n..(lane + 1) * n];
            for i in 0..n {
                self.vsq[lane * n + i] = mp2(&v[i]);
            }
            // Only boxes and triangles with uncut lines use these bounds.
            if n >= 4 {
                self.bounds[lane] = (
                    v.iter().flatten().fold(0_f64, |a, x| a.max(x.abs())),
                    input.masses_squared[lane * n..(lane + 1) * n]
                        .iter()
                        .fold(0_f64, |a, m| a.max(m.taxicab())),
                );
            }
            let s = &mut self.smat[lane * n * n..(lane + 1) * n * n];
            if let Some(matrix) = input.s_matrix {
                s.copy_from_slice(&matrix[lane * n * n..(lane + 1) * n * n]);
            } else {
                // SMatrix::fillFromKinematics
                for i in 0..n {
                    s[i * n + i] = 0.;
                    for j in i + 1..n {
                        let value = mp2(&sub(&v[i], &v[j]));
                        s[i * n + j] = value;
                        s[j * n + i] = value;
                    }
                }
            }
        }
        self.box_c
            .resize(self.topo.boxes.len() * lanes, [C64::ZERO; 2]);
        self.box_lambda.clear();
        if self.options.cancellation_rescue {
            self.box_lambda
                .resize(self.topo.boxes.len() * lanes, f64::NEG_INFINITY);
        }
        self.tri
            .resize(self.topo.triangles.len() * lanes, TriangleData::default());
        self.bub
            .resize(self.topo.bubbles.len() * lanes, BubbleData::default());
        self.tad.resize(n * lanes, TadpoleData::default());
    }

    // ------------------------------------------------------------------ boxes

    fn boxes<N: FastNumerator + ?Sized>(&mut self, input: &FastInput, num: &mut N) -> Result<()> {
        let n = self.topo.n;
        let lanes = self.lanes;
        let eps = self.options.infrared_eps;
        let check = Stability::new(&self.options);
        let x1 = self.topo.x1();
        let with_mu = self.topo.rank >= n;
        for b in 0..self.topo.boxes.len() {
            self.refresh_active();
            if self.active.is_empty() {
                return Ok(());
            }
            let p = self.topo.boxes[b];
            let [c1, c2, c3, c4] = p;
            self.box_rows.clear();
            self.calls.clear();
            for ai in 0..self.active.len() {
                let lane = self.active[ai];
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let k0 = sub(&v[c2], &v[c1]);
                let k1 = sub(&v[c3], &v[c2]);
                let k3 = sub(&v[c1], &v[c4]);
                if !check.relative(gram_det(&k3, &k0).abs(), gram_scale(&k3, &k0)) {
                    fail(
                        &mut self.alive,
                        &mut self.lane_status,
                        lane,
                        InstabilityTest::BoxGram,
                    );
                    continue;
                }
                let e = Basis::from_pair(&k3, &k0, eps);
                let p1e4 = mp_cr(&e.e4, &k1);
                let p1e3 = mp_cr(&e.e3, &k1);
                let mut vort = [C64::ZERO; 4];
                for mu in 0..4 {
                    vort[mu] = p1e4 * e.e3[mu] - p1e3 * e.e4[mu];
                }
                // mp2(Vort) = -2 (p1.e3) (p1.e4) e3.e4 vanishes exactly when the box
                // is degenerate (a light-like or vanishing normal): test the two
                // factors relative to their own scales.
                if !check.absolute(mp_cc(&vort, &vort).taxicab())
                    || !check.relative(p1e3.taxicab(), dot_scale(&e.e3, &k1))
                    || !check.relative(p1e4.taxicab(), dot_scale(&e.e4, &k1))
                {
                    fail(
                        &mut self.alive,
                        &mut self.lane_status,
                        lane,
                        euclidean_or(InstabilityTest::BoxNormal, &e.e3, &k3, &k0),
                    );
                    continue;
                }
                let l = box_solutions(&k3, &k0, &k1, &e, m[c1], m[c2], m[c3], m[c4]);
                // The box's conditioning, for the cancellation rescue's trigger
                // (no work when the rescue is off).
                if self.options.cancellation_rescue {
                    self.box_lambda[b * lanes + lane] = box_lambda_bits(&l, v, m, &p);
                }
                let q = [sub_cr(&l[0], &v[c1]), sub_cr(&l[1], &v[c1])];
                let mut mid = [C64::ZERO; 4];
                if x1 {
                    for mu in 0..4 {
                        mid[mu] = (l[0][mu] + l[1][mu]) * 0.5 - v[c1][mu];
                    }
                }
                self.box_rows.push(BoxRow { lane, vort, q, mid });
                for qq in q {
                    self.calls.lanes.push(lane);
                    self.calls.v0.push(qq);
                    self.calls.scalar.push(C64::ZERO);
                }
            }
            if self.box_rows.is_empty() {
                continue;
            }
            self.calls.prepare_out(1);
            num.evaluate(
                &p,
                &self.calls.lanes,
                &self.calls.v0,
                &self.calls.scalar,
                &mut self.calls.out,
            )?;
            let cp = &self.topo.box_complements[b * (n - 4)..(b + 1) * (n - 4)];
            let mut keep = 0;
            for r in 0..self.box_rows.len() {
                let row = self.box_rows[r];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let (vmax, mmax) = self.bounds[lane];
                let mut values = [C64::ZERO; 2];
                let mut stable = true;
                for s in 0..2 {
                    let mut d = C64::ONE;
                    let mut singular = false;
                    let bound = mmax
                        + row.q[s]
                            .iter()
                            .map(|z| (z.taxicab() + vmax).powi(2))
                            .sum::<f64>();
                    for &j in cp {
                        let factor = den(&row.q[s], &v[j], m[j]);
                        singular |= !check.relative_within(factor.taxicab(), bound, || {
                            den_scale(&row.q[s], &v[j], m[j])
                        });
                        d *= factor;
                    }
                    if singular || !check.absolute(d.taxicab()) {
                        stable = false;
                        break;
                    }
                    values[s] = self.calls.out[2 * r + s] / d;
                }
                if !stable {
                    fail(
                        &mut self.alive,
                        &mut self.lane_status,
                        lane,
                        InstabilityTest::BoxCutDenominator,
                    );
                    continue;
                }
                self.box_c[b * lanes + lane] = [(values[0] + values[1]) * 0.5, C64::ZERO];
                self.box_rows[keep] = row;
                keep += 1;
            }
            self.box_rows.truncate(keep);
            if !with_mu || self.box_rows.is_empty() {
                continue;
            }
            // Rational term from the mu-expansion.
            self.calls.clear();
            for row in &self.box_rows {
                self.calls.lanes.push(row.lane);
                self.calls.v0.push(row.vort);
                if x1 {
                    self.calls.v1.push(row.mid);
                }
            }
            self.calls.prepare_out(if x1 { 2 } else { 1 });
            num.mu_expansion(
                &p,
                &self.calls.lanes,
                &self.calls.v0,
                if x1 { Some(&self.calls.v1) } else { None },
                &mut self.calls.out,
            )?;
            // Rows whose geometric loss (`crate::conditioning`; at rank n
            // its cheap upper bound) exceeds the budget: (row, |n0|, |n1|,
            // the exact geometry above rank n), decided in
            // `box_mu_conditioning`.
            let mut suspects: Vec<(usize, [f64; 2], Option<BoxMuConditioning>)> = Vec::new();
            for r in 0..self.box_rows.len() {
                let row = self.box_rows[r];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let vort2 = mp_cc(&row.vort, &row.vort);
                let bound = 4. * self.bounds[lane].0 * taxicab_sum(&row.vort);
                let mut conditioning = BoxMuConditioning::ONE;
                if !x1 {
                    let mut d = C64::ONE;
                    let mut singular = false;
                    for &j in cp {
                        let factor = mp_cr(&row.vort, &sub(&v[j], &v[c1])) * 2.;
                        singular |= !check.relative_within(factor.taxicab(), bound, || {
                            difference_scale(&row.vort, &v[j], &v[c1])
                        });
                        conditioning.line(BOX_MU_PRETEST * bound, factor.taxicab());
                        d *= factor;
                    }
                    if singular || !check.absolute(d.taxicab()) {
                        fail(
                            &mut self.alive,
                            &mut self.lane_status,
                            lane,
                            InstabilityTest::BoxMuDenominator,
                        );
                        continue;
                    }
                    if conditioning.exceeds(0, 1., f64::MANTISSA_DIGITS) {
                        suspects.push((r, [self.calls.out[r].taxicab(), 0.], None));
                    }
                    self.box_c[b * lanes + lane][1] = self.calls.out[r] / d / (vort2 * vort2);
                } else {
                    let mut nmu = [self.calls.out[2 * r], self.calls.out[2 * r + 1]];
                    let vsq = &self.vsq[lane * n..(lane + 1) * n];
                    let mut stable = true;
                    // Euclidean scale and the solutions' distance from the
                    // center along the normal (`crate::conditioning`).
                    let vnorm = euclidean_norm(&row.vort);
                    let t = euclidean_norm(&sub_cc(&row.q[0], &row.mid)) / vnorm;
                    for &j in cp {
                        let kk = sub(&v[j], &v[c1]);
                        let d0 = mp_cr(&row.vort, &kk) * 2.;
                        if !check.relative_within(d0.taxicab(), bound, || {
                            difference_scale(&row.vort, &v[j], &v[c1])
                        }) {
                            stable = false;
                            break;
                        }
                        let length = kk.iter().map(|x| x * x).sum::<f64>().sqrt();
                        conditioning.line(2. * vnorm * length, d0.abs());
                        let d1 =
                            C64::real(vsq[j] - vsq[c1]) - m[j] + m[c1] + mp_cr(&row.mid, &kk) * 2.;
                        conditioning.expansion(d1.abs(), d0.abs(), t);
                        nmu[0] /= d0;
                        nmu[1] -= d1 * nmu[0];
                        nmu[1] /= d0;
                    }
                    if !stable {
                        fail(
                            &mut self.alive,
                            &mut self.lane_status,
                            lane,
                            InstabilityTest::BoxMuExpansion,
                        );
                        continue;
                    }
                    if conditioning.exceeds(1, 1., f64::MANTISSA_DIGITS) {
                        let numerator = [
                            self.calls.out[2 * r].taxicab(),
                            self.calls.out[2 * r + 1].taxicab(),
                        ];
                        suspects.push((r, numerator, Some(conditioning)));
                    }
                    self.box_c[b * lanes + lane][1] = nmu[1] / (vort2 * vort2);
                }
            }
            if !suspects.is_empty() {
                self.box_mu_conditioning(input, num, &p, b, &suspects, u32::from(x1))?;
            }
        }
        Ok(())
    }

    /// The conditioning screen (`crate::conditioning`) for the box rows over
    /// the budget: at rank n (cheap upper bound) the Euclidean `R_j` of each
    /// uncut line, then, for the rows still over it, `ν` from the numerator's
    /// leading coefficient along three directions of the box's hyperplane.
    /// A row still predicted to lose more than the budget adds its
    /// first-order rounding estimate to its lane's `mu_error`, which
    /// `integrate` decides against the lane's finite part.
    #[cold]
    #[inline(never)]
    fn box_mu_conditioning<N: FastNumerator + ?Sized>(
        &mut self,
        input: &FastInput,
        num: &mut N,
        p: &[usize],
        b: usize,
        suspects: &[(usize, [f64; 2], Option<BoxMuConditioning>)],
        excess: u32,
    ) -> Result<()> {
        let n = self.topo.n;
        let cp = self.topo.box_complements[b * (n - 4)..(b + 1) * (n - 4)].to_vec();
        let mut over: Vec<(usize, [f64; 2], BoxMuConditioning)> = Vec::new();
        for &(r, numerator, exact) in suspects {
            let row = self.box_rows[r];
            let v = &input.shifts[row.lane * n..(row.lane + 1) * n];
            let conditioning =
                exact.unwrap_or_else(|| euclidean_conditioning(&row.vort, v, p[0], &cp));
            if conditioning.exceeds(excess, 1., f64::MANTISSA_DIGITS) {
                over.push((r, numerator, conditioning));
            }
        }
        if over.is_empty() {
            return Ok(());
        }
        self.calls.clear();
        for &(r, _, _) in &over {
            let row = self.box_rows[r];
            let v = &input.shifts[row.lane * n..(row.lane + 1) * n];
            for x in box_mu_reference_directions(&v[p[0]], &v[p[1]], &v[p[2]], &v[p[3]]) {
                self.calls.lanes.push(row.lane);
                self.calls
                    .v0
                    .push(scaled_reference(&x, taxicab_sum(&row.vort)));
            }
        }
        self.calls.prepare_out(1);
        num.mu_expansion(
            p,
            &self.calls.lanes,
            &self.calls.v0,
            None,
            &mut self.calls.out,
        )?;
        for (i, &(r, numerator, conditioning)) in over.iter().enumerate() {
            let reference = self.calls.out[3 * i..3 * i + 3]
                .iter()
                .fold(0_f64, |a, z| a.max(z.taxicab()));
            let nu = box_mu_numerator_ratio(numerator[0], reference);
            if conditioning.exceeds(excess, nu, f64::MANTISSA_DIGITS) {
                let row = self.box_rows[r];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let vsq = &self.vsq[lane * n..(lane + 1) * n];
                let mut propagation = BoxMuPropagation::new([numerator[0], numerator[1], 0.]);
                for &j in &cp {
                    let kk = sub(&v[j], &v[p[0]]);
                    let d0 = (mp_cr(&row.vort, &kk) * 2.).abs();
                    let d1 = if excess == 0 {
                        0.
                    } else {
                        (C64::real(vsq[j] - vsq[p[0]]) - m[j] + m[p[0]] + mp_cr(&row.mid, &kk) * 2.)
                            .abs()
                    };
                    propagation.line(d0, d1);
                }
                let vv = mp_cc(&row.vort, &row.vort).abs();
                self.mu_error[lane] += box_mu_rounding(
                    excess,
                    conditioning.sum,
                    reference,
                    &propagation,
                    vv,
                    0.,
                    0.,
                );
            }
        }
        Ok(())
    }

    // -------------------------------------------------------------- triangles

    fn triangles<N: FastNumerator + ?Sized>(
        &mut self,
        input: &FastInput,
        num: &mut N,
    ) -> Result<()> {
        let n = self.topo.n;
        let lanes = self.lanes;
        let eps = self.options.infrared_eps;
        let check = Stability::new(&self.options);
        let rmn = self.topo.rmn();
        let x1 = self.topo.x1();
        let mindeg = (rmn + 3) as usize;
        let width = t3_len(mindeg);
        let nterms = mindeg + 1;
        let (lin, quad, cub) = (rmn >= -2, rmn >= -1, rmn >= 0);
        for t in 0..self.topo.triangles.len() {
            self.refresh_active();
            if self.active.is_empty() {
                return Ok(());
            }
            let p = self.topo.triangles[t];
            let [c1, c2, c3] = p;
            self.tri_rows.clear();
            for ai in 0..self.active.len() {
                let lane = self.active[ai];
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let k0 = sub(&v[c2], &v[c1]);
                let k2 = sub(&v[c1], &v[c3]);
                if !check.relative(gram_det(&k2, &k0).abs(), gram_scale(&k2, &k0)) {
                    fail(
                        &mut self.alive,
                        &mut self.lane_status,
                        lane,
                        InstabilityTest::TriangleGram,
                    );
                    continue;
                }
                let e = Basis::from_pair(&k2, &k0, eps);
                let (y1, y2, param) = triangle_solution(&k2, &k0, &e, m[c1], m[c2], m[c3]);
                let amu = sub_cr(&combine_real2(y1, &e.e1, y2, &e.e2), &v[c1]);
                self.tri_rows.push(TriangleRow {
                    lane,
                    e3: e.e3,
                    e4: e.e4,
                    amu,
                    param,
                    e34: e.mp34(),
                });
            }
            if self.tri_rows.is_empty() {
                continue;
            }
            // Both orientations share one callback: rows [e3 side..., e4 side...].
            self.calls.clear();
            for orientation in 0..2 {
                for row in &self.tri_rows {
                    let h = 0.5 / row.e34;
                    self.calls.lanes.push(row.lane);
                    self.calls.v0.push(row.amu);
                    if orientation == 0 {
                        self.calls.v1.push(row.e3);
                        self.calls.v2.push(scale_c(&row.e4, h));
                    } else {
                        self.calls.v1.push(row.e4);
                        self.calls.v2.push(scale_c(&row.e3, h));
                    }
                    self.calls.scalar.push(row.param);
                }
            }
            self.calls.prepare_out(width);
            num.t3_expansion(
                &p,
                &self.calls.lanes,
                &self.calls.v0,
                &self.calls.v1,
                &self.calls.v2,
                &self.calls.scalar,
                mindeg,
                &mut self.calls.out,
            )?;
            let rows = self.tri_rows.len();
            let cp = &self.topo.triangle_complements[t * (n - 3)..(t + 1) * (n - 3)];
            let (head, tail) = self.calls.out.split_at_mut(rows * width);
            for r in 0..rows {
                let row = &self.tri_rows[r];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let vsq = &self.vsq[lane * n..(lane + 1) * n];
                let num1 = &mut head[r * width..(r + 1) * width];
                let num2 = &mut tail[r * width..(r + 1) * width];
                let inv34 = 1. / row.e34;
                let vmax4 = 4. * self.bounds[lane].0;
                let (bound3, bound4) = (vmax4 * taxicab_sum(&row.e3), vmax4 * taxicab_sum(&row.e4));
                let mut stable = true;
                for &j in cp {
                    let pdiff = sub(&v[j], &v[c1]);
                    let e3p = mp_cr(&row.e3, &pdiff);
                    let e4p = mp_cr(&row.e4, &pdiff);
                    let (d0a, d0b) = (e3p * 2., e4p * 2.);
                    if !check.relative_within(d0a.taxicab(), bound3, || {
                        difference_scale(&row.e3, &v[j], &v[c1])
                    }) || !check.relative_within(d0b.taxicab(), bound4, || {
                        difference_scale(&row.e4, &v[j], &v[c1])
                    }) {
                        stable = false;
                        break;
                    }
                    // Only the terms the truncated division reads are formed.
                    let d1 = if nterms > 1 {
                        mp_cr(&row.amu, &pdiff) * 2. + (vsq[j] - vsq[c1]) - m[j] + m[c1]
                    } else {
                        C64::ZERO
                    };
                    let (d2a, d2b) = if nterms > 2 {
                        let (a, b) = (e4p * inv34, e3p * inv34);
                        ([row.param * a, a], [row.param * b, b])
                    } else {
                        ([C64::ZERO; 2], [C64::ZERO; 2])
                    };
                    divide_t3(num1, nterms, d0a.inv(), d1, d2a);
                    divide_t3(num2, nterms, d0b.inv(), d1, d2b);
                }
                if !stable {
                    let test = euclidean_or(
                        InstabilityTest::TriangleExpansion,
                        &row.e3,
                        &sub(&v[c1], &v[c3]),
                        &sub(&v[c2], &v[c1]),
                    );
                    fail(&mut self.alive, &mut self.lane_status, lane, test);
                    continue;
                }
                let mut c = [C64::ZERO; 15];
                if !x1 {
                    let jext3 = rmn;
                    let jext2 = rmn + 1;
                    let jext1 = rmn + 2;
                    let jext1mu2 = 3;
                    let jext0 = (rmn + 3) as usize + usize::from(cub);
                    let jext0mu2 = if cub { 5 } else { 3 };
                    c[0] = (num1[jext0] + num2[jext0]) * 0.5;
                    if lin {
                        c[4] = num1[jext1 as usize] * inv34;
                        c[1] = num2[jext1 as usize] * inv34;
                    }
                    if quad {
                        let i2 = inv34 * inv34;
                        c[5] = num1[jext2 as usize] * i2;
                        c[2] = num2[jext2 as usize] * i2;
                        c[7] = (num1[jext0mu2] + num2[jext0mu2]) * 0.5;
                    }
                    if cub {
                        let i3 = inv34 * inv34 * inv34;
                        c[6] = num1[jext3 as usize] * i3;
                        c[3] = num2[jext3 as usize] * i3;
                        c[9] = num1[jext1mu2] * inv34;
                        c[8] = num2[jext1mu2] * inv34;
                    }
                } else {
                    let i2 = inv34 * inv34;
                    let i3 = i2 * inv34;
                    let i4 = i2 * i2;
                    c[0] = (num1[6] + num2[6]) * 0.5;
                    c[4] = num1[4] * inv34;
                    c[1] = num2[4] * inv34;
                    c[5] = num1[2] * i2;
                    c[2] = num2[2] * i2;
                    c[7] = (num1[7] + num2[7]) * 0.5;
                    c[6] = num1[1] * i3;
                    c[3] = num2[1] * i3;
                    c[9] = num1[5] * inv34;
                    c[8] = num2[5] * inv34;
                    c[13] = num1[0] * i4;
                    c[12] = num2[0] * i4;
                    c[11] = num1[3] * i2;
                    c[10] = num2[3] * i2;
                    c[14] = (num1[8] + num2[8]) * 0.5;
                }
                self.tri[t * lanes + lane] = TriangleData {
                    e3: row.e3,
                    e4: row.e4,
                    c,
                };
            }
        }
        Ok(())
    }

    // ---------------------------------------------------------------- bubbles

    fn bubbles<N: FastNumerator + ?Sized>(&mut self, input: &FastInput, num: &mut N) -> Result<()> {
        let n = self.topo.n;
        let lanes = self.lanes;
        let eps = self.options.infrared_eps;
        let check = Stability::new(&self.options);
        let rmn = self.topo.rmn();
        let x1 = self.topo.x1();
        let mindeg = (rmn + 2) as usize;
        let width = t2_len_le3(mindeg);
        let nterms = mindeg + 1;
        let (lin, quad) = (rmn >= -1, rmn >= 0);
        // Triangle parents exist only when the triangle stage ran.
        let (correct_all, correct_spurious): (Option<BubbleCorrection>, Option<BubbleCorrection>) =
            if !self.any_triangle() {
                (None, None)
            } else {
                match rmn {
                    -2 => (Some(subs::bubble_from_triangle_rm2_all), None),
                    -1 => (
                        Some(subs::bubble_from_triangle_rm1_all),
                        Some(subs::bubble_from_triangle_rm1_spurious),
                    ),
                    0 => (
                        Some(subs::bubble_from_triangle_r0_all),
                        Some(subs::bubble_from_triangle_r0_spurious),
                    ),
                    _ => (
                        Some(subs::bubble_from_triangle_x1_all),
                        Some(subs::bubble_from_triangle_x1_spurious),
                    ),
                }
            };
        for b in 0..self.topo.bubbles.len() {
            self.refresh_active();
            if self.active.is_empty() {
                return Ok(());
            }
            let p = self.topo.bubbles[b];
            let [c1, c2] = p;
            self.bub_rows.clear();
            self.calls.clear();
            for ai in 0..self.active.len() {
                let lane = self.active[ai];
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let s = self.smat[lane * n * n + c2 * n + c1];
                // Zero-initialized like a fresh Ninja CutsVector entry.
                self.bub[b * lanes + lane] = BubbleData::default();
                if is_zero(m[c1]) && is_zero(m[c2]) && s.abs() < eps {
                    continue;
                }
                let k = sub(&v[c1], &v[c2]);
                let e = Basis::from_single(&k, eps);
                let (b0, b1, param) = bubble_solution(&k, &e, m[c1], m[c2]);
                let e1effl = [b0 * e.e2[0], b0 * e.e2[1], b0 * e.e2[2], b0 * e.e2[3]];
                let e2eff = [
                    e.e1[0] + b1 * e.e2[0],
                    e.e1[1] + b1 * e.e2[1],
                    e.e1[2] + b1 * e.e2[2],
                    e.e1[3] + b1 * e.e2[3],
                ];
                let e1eff = sub_cr(&e1effl, &v[c1]);
                self.calls.lanes.push(lane);
                self.calls.v0.push(e1eff);
                self.calls.v1.push(cmom_from_real(&e2eff));
                self.calls.v2.push(e.e3);
                self.calls.v3.push(scale_c(&e.e4, 0.5 / e.mp34()));
                self.calls.params.push(param);
                self.bub_rows.push(BubbleRow {
                    lane,
                    e,
                    b0,
                    b1,
                    e1eff,
                    e2eff,
                    param,
                });
            }
            if self.bub_rows.is_empty() {
                continue;
            }
            self.calls.prepare_out(width);
            num.t2_expansion(
                &p,
                &self.calls.lanes,
                &self.calls.v0,
                &self.calls.v1,
                &self.calls.v2,
                &self.calls.v3,
                &self.calls.params,
                mindeg,
                &mut self.calls.out,
            )?;
            let cp = &self.topo.bubble_complements[b * (n - 2)..(b + 1) * (n - 2)];
            let parents = &self.topo.bubble_parents[b];
            self.second.clear();
            for r in 0..self.bub_rows.len() {
                let row = &self.bub_rows[r];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let vsq = &self.vsq[lane * n..(lane + 1) * n];
                let numexp = &mut self.calls.out[r * width..(r + 1) * width];
                let e = &row.e;
                let inv34 = 1. / e.mp34();
                let e4s = 0.5 * inv34;
                let uncut = &mut self.uncut;
                let mut stable = true;
                for &j in cp {
                    let pdiff = sub(&v[j], &v[c1]);
                    let e3k = mp_cr(&e.e3, &pdiff);
                    if !check.absolute(e3k.taxicab() * 2.) {
                        stable = false;
                        break;
                    }
                    // a0 = b0 e2 and a1 = e1 + b1 e2 with real e1, e2.
                    let e2p = mp_rr(&e.e2, &pdiff);
                    let e1k = row.b0 * e2p;
                    uncut[j] = Uncut {
                        e3k,
                        ie3k: e3k.inv(),
                        e4k: mp_cr(&e.e4, &pdiff),
                        e1k,
                        e2k: mp_rr(&e.e1, &pdiff) + row.b1 * e2p,
                        d1: (e1k - mp_rr(&v[c1], &pdiff)) * 2.
                            + (vsq[j] - vsq[c1])
                            + (-m[j] + m[c1]),
                        f: mp2(&pdiff) + (m[c1] - m[j]),
                    };
                }
                if !stable {
                    fail(
                        &mut self.alive,
                        &mut self.lane_status,
                        lane,
                        InstabilityTest::BubbleExpansion,
                    );
                    continue;
                }
                for &j in cp {
                    let u = &uncut[j];
                    let d2 = u.e4k * inv34;
                    divide_t2(
                        numexp,
                        nterms,
                        u.ie3k * 0.5,
                        [u.d1, C64::real(2. * u.e2k)],
                        [row.param[0] * d2, d2, row.param[1] * d2, row.param[2] * d2],
                    );
                }
                if let Some(all) = correct_all {
                    let e4eff = scale_c(&e.e4, e4s);
                    let np = parents.len();
                    if self.pairs.len() < (r + 1) * np {
                        self.pairs.resize((r + 1) * np, [C64::ZERO; 4]);
                    }
                    for (pi, &(t, unc)) in parents.iter().enumerate() {
                        let tri = &self.tri[t * lanes + lane];
                        let u = &uncut[unc];
                        // Shared with the second orientation: a0 = b0 e2,
                        // a1 = e1 + b1 e2 against the triangle's e3, e4.
                        let (e2t3, e2t4) = (mp_cr(&tri.e3, &e.e2), mp_cr(&tri.e4, &e.e2));
                        let pair = [
                            row.b0 * e2t3,
                            row.b0 * e2t4,
                            mp_cr(&tri.e3, &e.e1) + e2t3 * row.b1,
                            mp_cr(&tri.e4, &e.e1) + e2t4 * row.b1,
                        ];
                        self.pairs[r * np + pi] = pair;
                        all(
                            numexp,
                            &tri.e3,
                            &tri.e4,
                            &tri.c,
                            &e.e3,
                            &e4eff,
                            &row.param,
                            u.e1k,
                            u.e2k,
                            u.e3k,
                            u.ie3k,
                            u.e4k * e4s,
                            &pair,
                            u.f,
                        );
                    }
                }
                let inv12 = 1. / e.mp12;
                let data = &mut self.bub[b * lanes + lane];
                data.e2 = e.e2;
                data.e3 = e.e3;
                data.e4 = e.e4;
                let c = &mut data.c;
                if !x1 {
                    let jext2 = rmn;
                    let jext1 = rmn + 1;
                    let jext1x1 = 2;
                    let jext0 = (rmn + 2) as usize + usize::from(quad);
                    let jext0mu2 = 4;
                    let jext0x1 = if quad { 5 } else { 2 };
                    let jext0x2 = 6;
                    c[0] = numexp[jext0];
                    if lin {
                        c[1] = numexp[jext0x1] * inv12;
                        c[5] = numexp[jext1 as usize] * inv34;
                    }
                    if quad {
                        c[9] = numexp[jext0mu2];
                        c[2] = numexp[jext0x2] * (inv12 * inv12);
                        c[6] = numexp[jext2 as usize] * (inv34 * inv34);
                        c[8] = numexp[jext1x1] * (inv12 * inv34);
                    }
                } else {
                    c[0] = numexp[7];
                    c[1] = numexp[9] * inv12;
                    c[5] = numexp[3] * inv34;
                    c[9] = numexp[8];
                    c[2] = numexp[11] * (inv12 * inv12);
                    c[6] = numexp[1] * (inv34 * inv34);
                    c[8] = numexp[5] * (inv12 * inv34);
                    c[10] = numexp[10] * inv12;
                    c[12] = numexp[4] * inv34;
                    c[13] = numexp[12] * (inv12 * inv12 * inv12);
                    c[15] = numexp[0] * (inv34 * inv34 * inv34);
                    c[17] = numexp[6] * (inv34 * inv12 * inv12);
                    c[19] = numexp[2] * (inv34 * inv34 * inv12);
                }
                // Spurious coefficients are needed only as subtraction terms
                // of massive tadpoles (Ninja's spurious_terms_needed).
                if lin && (!is_zero(m[c1]) || !is_zero(m[c2])) {
                    self.second.push(r);
                }
            }
            if self.second.is_empty() {
                continue;
            }
            // Second orientation (e3 <-> e4) with reduced depth.
            let mindeg2 = mindeg - 1;
            let width2 = t2_len_le3(mindeg2);
            self.calls.clear();
            // Rows of lanes that failed in the first orientation never enter
            // `second`, so call row `s` belongs to `bub_rows[second[s]]`.
            for &r in &self.second {
                let row = &self.bub_rows[r];
                self.calls.lanes.push(row.lane);
                self.calls.v0.push(row.e1eff);
                self.calls.v1.push(cmom_from_real(&row.e2eff));
                self.calls.v2.push(row.e.e4);
                self.calls.v3.push(scale_c(&row.e.e3, 0.5 / row.e.mp34()));
                self.calls.params.push(row.param);
            }
            self.calls.prepare_out(width2);
            num.t2_expansion(
                &p,
                &self.calls.lanes,
                &self.calls.v0,
                &self.calls.v1,
                &self.calls.v2,
                &self.calls.v3,
                &self.calls.params,
                mindeg2,
                &mut self.calls.out,
            )?;
            for s in 0..self.second.len() {
                let row = &self.bub_rows[self.second[s]];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let vsq = &self.vsq[lane * n..(lane + 1) * n];
                let numexp = &mut self.calls.out[s * width2..(s + 1) * width2];
                let e = &row.e;
                let inv34 = 1. / e.mp34();
                let e3s = 0.5 * inv34;
                let uncut = &mut self.uncut;
                let mut stable = true;
                for &j in cp {
                    let pdiff = sub(&v[j], &v[c1]);
                    // Orientation e3 <-> e4: `e3k` holds e4.(V_j - V_0) here.
                    let e4k = mp_cr(&e.e4, &pdiff);
                    if !check.absolute(e4k.taxicab() * 2.) {
                        stable = false;
                        break;
                    }
                    let e2p = mp_rr(&e.e2, &pdiff);
                    let e1k = row.b0 * e2p;
                    uncut[j] = Uncut {
                        e3k: e4k,
                        ie3k: e4k.inv(),
                        e4k: mp_cr(&e.e3, &pdiff),
                        e1k,
                        e2k: mp_rr(&e.e1, &pdiff) + row.b1 * e2p,
                        d1: (e1k - mp_rr(&v[c1], &pdiff)) * 2.
                            + (vsq[j] - vsq[c1])
                            + (-m[j] + m[c1]),
                        f: mp2(&pdiff) + (m[c1] - m[j]),
                    };
                }
                if !stable {
                    fail(
                        &mut self.alive,
                        &mut self.lane_status,
                        lane,
                        InstabilityTest::BubbleExpansion,
                    );
                    continue;
                }
                for &j in cp {
                    let u = &uncut[j];
                    let d2 = u.e4k * inv34;
                    divide_t2(
                        numexp,
                        nterms - 1,
                        u.ie3k * 0.5,
                        [u.d1, C64::real(2. * u.e2k)],
                        [row.param[0] * d2, d2, row.param[1] * d2, row.param[2] * d2],
                    );
                }
                if let Some(spurious) = correct_spurious {
                    let e3eff = scale_c(&e.e3, e3s);
                    let np = parents.len();
                    let r = self.second[s];
                    for (pi, &(t, unc)) in parents.iter().enumerate() {
                        let tri = &self.tri[t * lanes + lane];
                        let u = &uncut[unc];
                        spurious(
                            numexp,
                            &tri.e3,
                            &tri.e4,
                            &tri.c,
                            &e.e4,
                            &e3eff,
                            &row.param,
                            u.e1k,
                            u.e2k,
                            u.e3k,
                            u.ie3k,
                            u.e4k * e3s,
                            &self.pairs[r * np + pi],
                            u.f,
                        );
                    }
                }
                let inv12 = 1. / e.mp12;
                let c = &mut self.bub[b * lanes + lane].c;
                if !x1 {
                    c[3] = numexp[(rmn + 1) as usize] * inv34;
                    if quad {
                        c[4] = numexp[rmn as usize] * (inv34 * inv34);
                        c[7] = numexp[2] * (inv12 * inv34);
                    }
                } else {
                    c[3] = numexp[3] * inv34;
                    c[4] = numexp[1] * (inv34 * inv34);
                    c[7] = numexp[5] * (inv12 * inv34);
                    c[11] = numexp[4] * inv34;
                    c[14] = numexp[0] * (inv34 * inv34 * inv34);
                    c[16] = numexp[6] * (inv34 * inv12 * inv12);
                    c[18] = numexp[2] * (inv34 * inv34 * inv12);
                }
            }
        }
        Ok(())
    }

    // --------------------------------------------------------------- tadpoles

    fn tadpoles<N: FastNumerator + ?Sized>(
        &mut self,
        input: &FastInput,
        num: &mut N,
    ) -> Result<()> {
        let n = self.topo.n;
        let lanes = self.lanes;
        let eps = self.options.infrared_eps;
        let check = Stability::new(&self.options);
        let rmn = self.topo.rmn();
        let x1 = self.topo.x1();
        let mindeg = (rmn + 1) as usize;
        let width = t3_len(mindeg);
        let nterms = mindeg + 1;
        let any_triangle = self.any_triangle();
        let any_bubble = self.any_bubble();
        let orientations = if x1 { 2 } else { 1 };
        for a in 0..n {
            self.refresh_active();
            if self.active.is_empty() {
                return Ok(());
            }
            self.tad_rows.clear();
            for ai in 0..self.active.len() {
                let lane = self.active[ai];
                if is_zero(input.masses_squared[lane * n + a]) {
                    continue;
                }
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let basis = (0..2)
                    .find(|&i| {
                        let e = &self.tadpole_bases[i];
                        reference_vectors_ok(v, a, &e.e1, &e.e2, eps)
                    })
                    .unwrap_or(2);
                self.tad_rows.push(TadpoleRow { lane, basis });
            }
            if self.tad_rows.is_empty() {
                continue;
            }
            // Expansion vectors per orientation: rank <= n uses (e3, e4);
            // rank n+1 uses (e1, e2) rows followed by (e3, e4) rows.
            let vectors = |e: &Basis, orientation: usize| -> (CMom, CMom) {
                if x1 && orientation == 0 {
                    (
                        cmom_from_real(&e.e1),
                        cmom_from_real(&scale(&e.e2, 0.5 / e.mp12)),
                    )
                } else {
                    (e.e3, scale_c(&e.e4, 0.5 / e.mp34()))
                }
            };
            self.calls.clear();
            for orientation in 0..orientations {
                for row in &self.tad_rows {
                    let lane = row.lane;
                    let (ev3, ev4) = vectors(&self.tadpole_bases[row.basis], orientation);
                    let v0 = input.shifts[lane * n + a];
                    self.calls.lanes.push(lane);
                    self.calls
                        .v0
                        .push(cmom_from_real(&[-v0[0], -v0[1], -v0[2], -v0[3]]));
                    self.calls.v1.push(ev3);
                    self.calls.v2.push(ev4);
                    self.calls.scalar.push(input.masses_squared[lane * n + a]);
                }
            }
            self.calls.prepare_out(width);
            let cut = [a];
            num.t3_expansion(
                &cut,
                &self.calls.lanes,
                &self.calls.v0,
                &self.calls.v1,
                &self.calls.v2,
                &self.calls.scalar,
                mindeg,
                &mut self.calls.out,
            )?;
            let rows = self.tad_rows.len();
            let cp = &self.topo.tadpole_complements[a * (n - 1)..(a + 1) * (n - 1)];
            let tri_parents = &self.topo.tadpole_triangle_parents[a];
            let bub_parents = &self.topo.tadpole_bubble_parents[a];
            for r in 0..rows {
                let row = self.tad_rows[r];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let vsq = &self.vsq[lane * n..(lane + 1) * n];
                let e = self.tadpole_bases[row.basis];
                let param = m[a];
                let amu = [-v[a][0], -v[a][1], -v[a][2], -v[a][3]];
                let mut stable = true;
                let mut data = TadpoleData {
                    e34: -mp_rr(&e.e1, &e.e2),
                    ..TadpoleData::default()
                };
                for orientation in 0..orientations {
                    let offset = (orientation * rows + r) * width;
                    let numexp = &mut self.calls.out[offset..offset + width];
                    let (ev3, ev4) = vectors(&e, orientation);
                    let uncut = &mut self.uncut;
                    for &j in cp {
                        let pdiff = sub(&v[j], &v[a]);
                        let e3k = mp_cr(&ev3, &pdiff);
                        if !check.absolute(e3k.taxicab() * 2.) {
                            stable = false;
                            break;
                        }
                        uncut[j] = Uncut {
                            e3k,
                            ie3k: e3k.inv(),
                            e4k: mp_cr(&ev4, &pdiff),
                            e1k: C64::ZERO,
                            e2k: 0.,
                            d1: C64::real(2. * mp_rr(&amu, &pdiff) + (vsq[j] - vsq[a])) - m[j]
                                + m[a],
                            f: mp2(&pdiff) + (m[a] - m[j]),
                        };
                    }
                    if !stable {
                        break;
                    }
                    for &j in cp {
                        let u = &uncut[j];
                        let d2 = u.e4k * 2.;
                        divide_t3(numexp, nterms, u.ie3k * 0.5, u.d1, [param * d2, d2]);
                    }
                    if any_triangle {
                        for &(t, [u0, u1]) in tri_parents {
                            let tri = &self.tri[t * lanes + lane];
                            let (k0, k1) = (&uncut[u0], &uncut[u1]);
                            if x1 {
                                subs::tadpole_from_triangle_x1(
                                    numexp, &tri.e3, &tri.e4, &tri.c, &ev3, &ev4, param, k0.e3k,
                                    k0.ie3k, k0.e4k, k0.f, k1.e3k, k1.ie3k, k1.e4k, k1.f,
                                );
                            } else {
                                tadpole_from_triangle(
                                    numexp, &tri.e3, &tri.e4, &tri.c, &ev3, k0, k1, rmn,
                                );
                            }
                        }
                    }
                    if any_bubble {
                        for &(bb, unc) in bub_parents {
                            let bub = &self.bub[bb * lanes + lane];
                            let kbol = sub(&v[self.topo.bubbles[bb][0]], &v[a]);
                            let u = &uncut[unc];
                            if x1 {
                                subs::tadpole_from_bubble_x1(
                                    numexp, &bub.e2, &bub.e3, &bub.e4, &kbol, &bub.c, &ev3, &ev4,
                                    param, u.e3k, u.ie3k, u.e4k, u.f,
                                );
                            } else {
                                tadpole_from_bubble(
                                    numexp, &bub.e2, &bub.e3, &bub.e4, &kbol, &bub.c, &ev3, u, rmn,
                                );
                            }
                        }
                    }
                    if !x1 {
                        data.c0 = numexp[(1 + rmn) as usize];
                    } else if orientation == 0 {
                        data.c0 = numexp[2];
                        data.c14 = numexp[3];
                    } else {
                        data.c15 = (numexp[3] - data.c14) * (2. / e.mp34());
                    }
                }
                if !stable {
                    fail(
                        &mut self.alive,
                        &mut self.lane_status,
                        lane,
                        InstabilityTest::TadpoleExpansion,
                    );
                    continue;
                }
                self.tad[a * lanes + lane] = data;
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------ integration

    fn integrate<M: FastMasters + ?Sized>(
        &mut self,
        input: &FastInput,
        masters: &mut M,
    ) -> Result<()> {
        let n = self.topo.n;
        let lanes = self.lanes;
        let x1 = self.topo.x1();
        let eps = self.options.infrared_eps;
        let (any_box, any_tri, any_bub, any_tad) = (
            self.any_box(),
            self.any_triangle(),
            self.any_bubble(),
            self.any_tadpole(),
        );
        self.refresh_active();
        for batch in &mut self.masters {
            batch.args.clear();
        }
        // Collect requests lane-major in Ninja's contraction order.
        for &lane in &self.active {
            let m = &input.masses_squared[lane * n..(lane + 1) * n];
            let s = &self.smat[lane * n * n..(lane + 1) * n * n];
            let mu2 = C64::real(input.scales_squared[lane]);
            let sm = |i: usize, j: usize| C64::real(s[i * n + j]);
            if any_box {
                let args = &mut self.masters[0].args;
                for &[i1, i2, i3, i4] in &self.topo.boxes {
                    args.extend_from_slice(&[
                        sm(i2, i1),
                        sm(i3, i2),
                        sm(i4, i3),
                        sm(i1, i4),
                        sm(i3, i1),
                        sm(i4, i2),
                        m[i1],
                        m[i2],
                        m[i3],
                        m[i4],
                        mu2,
                    ]);
                }
            }
            if any_tri {
                let args = &mut self.masters[1].args;
                for &[i1, i2, i3] in &self.topo.triangles {
                    args.extend_from_slice(&[
                        sm(i2, i1),
                        sm(i3, i2),
                        sm(i1, i3),
                        m[i1],
                        m[i2],
                        m[i3],
                        mu2,
                    ]);
                }
            }
            if any_bub {
                let v = &input.shifts[lane * n..(lane + 1) * n];
                for (b, &[i1, i2]) in self.topo.bubbles.iter().enumerate() {
                    let k = s[i2 * n + i1];
                    if is_zero(m[i1]) && is_zero(m[i2]) && k.abs() < eps {
                        continue;
                    }
                    let family = if x1 && self.rank3_bubble(b, lane, v) {
                        3
                    } else {
                        2
                    };
                    self.masters[family]
                        .args
                        .extend_from_slice(&[C64::real(k), m[i1], m[i2], mu2]);
                }
            }
            if any_tad {
                for i in 0..n {
                    if !is_zero(m[i]) {
                        self.masters[4].args.extend_from_slice(&[m[i], mu2]);
                    }
                }
            }
        }
        for (batch, family) in self.masters.iter_mut().zip(FAMILIES) {
            batch.run(family, masters)?;
        }
        // Contract in original Ninja's order.
        let mut cursor = [0usize; 5];
        for ai in 0..self.active.len() {
            let lane = self.active[ai];
            let v = &input.shifts[lane * n..(lane + 1) * n];
            let m = &input.masses_squared[lane * n..(lane + 1) * n];
            let s = &self.smat[lane * n * n..(lane + 1) * n * n];
            let mut ok = true;
            let mut res = [C64::ZERO; 3];
            let mut rational = C64::ZERO;
            if any_box {
                let batch = &self.masters[0];
                for b in 0..self.topo.boxes.len() {
                    let row = cursor[0];
                    cursor[0] += 1;
                    ok &= batch.ok[row];
                    let integral = &batch.out[row];
                    let [c0, c4] = self.box_c[b * lanes + lane];
                    for k in 0..3 {
                        res[k] += c0 * integral[k];
                    }
                    rational += -c4 / 6.;
                }
            }
            if any_tri {
                let batch = &self.masters[1];
                for (t, &[i1, i2, i3]) in self.topo.triangles.iter().enumerate() {
                    let row = cursor[1];
                    cursor[1] += 1;
                    ok &= batch.ok[row];
                    let integral = &batch.out[row];
                    let c = &self.tri[t * lanes + lane].c;
                    for k in 0..3 {
                        res[k] += c[0] * integral[k];
                    }
                    if x1 {
                        let (pji, pjk, pki) = (s[i2 * n + i1], s[i3 * n + i2], s[i1 * n + i3]);
                        rational += c[7] * 0.5
                            + c[14] * (C64::real(0.25 * (pjk + pji + pki)) - m[i1] - m[i2] - m[i3])
                                / 6.;
                    } else {
                        rational += c[7] * 0.5;
                    }
                }
            }
            if any_bub {
                for (b, &[i1, i2]) in self.topo.bubbles.iter().enumerate() {
                    let k = s[i2 * n + i1];
                    if is_zero(m[i1]) && is_zero(m[i2]) && k.abs() < eps {
                        continue;
                    }
                    let data = &self.bub[b * lanes + lane];
                    let c = &data.c;
                    let ke2 = mp_rr(&sub(&v[i2], &v[i1]), &data.e2);
                    let rank3 = x1 && self.rank3_bubble(b, lane, v);
                    let (family, width) = if rank3 { (3, 4) } else { (2, 3) };
                    let row = cursor[family];
                    cursor[family] += 1;
                    let batch = &self.masters[family];
                    ok &= batch.ok[row];
                    let out = &batch.out[row * width..(row + 1) * width];
                    let b06 = -(C64::real(k) - (m[i1] + m[i2]) * 3.) / 6.;
                    let c1 = c[1] * ke2;
                    let c2 = c[2] * ke2 * ke2;
                    if rank3 {
                        let c13 = c[13] * ke2 * ke2 * ke2;
                        for j in 0..3 {
                            res[j] += c[0] * out[0][j]
                                + c1 * out[1][j]
                                + c2 * out[2][j]
                                + c13 * out[3][j];
                        }
                    } else {
                        for j in 0..3 {
                            res[j] += c[0] * out[0][j] + c1 * out[1][j] + c2 * out[2][j];
                        }
                    }
                    if x1 {
                        rational += b06 * c[9]
                            + (C64::real(k) - m[i1] * 2. - m[i2] * 4.) * ke2 * c[10] / 12.;
                    } else {
                        rational += b06 * c[9];
                    }
                }
            }
            if any_tad {
                let batch = &self.masters[4];
                for i in 0..n {
                    if is_zero(m[i]) {
                        continue;
                    }
                    let row = cursor[4];
                    cursor[4] += 1;
                    ok &= batch.ok[row];
                    let integral = &batch.out[row];
                    let data = &self.tad[i * lanes + lane];
                    if x1 {
                        let c0 = data.c0 + m[i] * (data.e34 * 0.25) * data.c15;
                        for k in 0..3 {
                            res[k] += c0 * integral[k];
                        }
                        rational += (data.c14 + data.c15 * (data.e34 * 0.25)) * m[i] * m[i] * 0.5;
                    } else {
                        for k in 0..3 {
                            res[k] += data.c0 * integral[k];
                        }
                    }
                }
            }
            if !ok {
                self.lane_status[lane] = FastStatus::MasterFailed;
                continue;
            }
            let result = FastResult {
                laurent: [res[0] + rational, res[1], res[2]],
                cut_constructible: res[0],
                rational,
            };
            if !result.laurent.iter().all(|z| z.is_finite()) || !rational.is_finite() {
                self.lane_status[lane] = if any_tri && plane_refused(&self.topo, input, lane, eps) {
                    FastStatus::Unstable
                } else {
                    FastStatus::NonFinite
                };
                continue;
            }
            // The conditioning screen's flagged boxes (`crate::conditioning`).
            let flagged = self.mu_error[lane];
            if flagged > 0.
                && box_mu_refused(flagged, result.laurent[0].abs(), f64::MANTISSA_DIGITS)
            {
                // Kept for the cancellation rescue, which may re-reduce it;
                // a refused lane is never published.
                self.staged[lane] = result;
                self.lane_status[lane] = FastStatus::Unstable;
                screen_refused(lane, flagged, result.laurent[0].abs());
                continue;
            }
            self.staged[lane] = result;
            self.lane_status[lane] = FastStatus::Success;
        }
        Ok(())
    }

    /// Ninja's rank n+1 choice of the rank-3 bubble integral,
    /// `|c13 ((V2-V1).e2)³| != 0` (tested as an exact complex zero).
    #[inline]
    fn rank3_bubble(&self, b: usize, lane: usize, v: &[RMom]) -> bool {
        let [i1, i2] = self.topo.bubbles[b];
        let data = &self.bub[b * self.lanes + lane];
        let ke2 = mp_rr(&sub(&v[i2], &v[i1]), &data.e2);
        !is_zero(data.c[13] * ke2 * ke2 * ke2)
    }
}

/// `cuts::correcttadcoeffs` (triangle parent) for rank <= n. The cubic
/// triangle coefficients exist only for rank = n. `k0`, `k1` carry the
/// tadpole's precomputed products with the two uncut propagators.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn tadpole_from_triangle(
    divnum: &mut [C64],
    et3: &CMom,
    et4: &CMom,
    c: &[C64],
    e3: &CMom,
    k0: &Uncut,
    k1: &Uncut,
    rmn: isize,
) {
    let (e3k0, e3k1) = (k0.e3k, k1.e3k);
    let e3t3 = mp_cc(e3, et3);
    let e3t4 = mp_cc(e3, et4);
    let e3t3q = e3t3 * e3t3;
    let e3t4q = e3t4 * e3t4;
    let mut numerator = c[2] * 2. * e3k0 * e3k1 * e3t3q + c[5] * 2. * e3k0 * e3k1 * e3t4q;
    if rmn >= 0 {
        let e3t3c = e3t3q * e3t3;
        let e3t4c = e3t4q * e3t4;
        numerator -= (c[3] * e3t3c + c[6] * e3t4c) * (e3k1 * k0.f + e3k0 * k1.f);
    }
    let inverse = k0.ie3k * k0.ie3k * (k1.ie3k * k1.ie3k);
    divnum[(rmn + 1) as usize] -= numerator * 0.125 * inverse;
}

/// `cuts::correcttadcoeffs` (bubble parent) for rank <= n. The quadratic
/// bubble coefficients exist only for rank = n.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn tadpole_from_bubble(
    divnum: &mut [C64],
    eb2: &RMom,
    eb3: &CMom,
    eb4: &CMom,
    kbol: &RMom,
    b: &[C64],
    e3: &CMom,
    k: &Uncut,
    rmn: isize,
) {
    let e3b2 = mp_cr(e3, eb2);
    let e3b3 = mp_cc(e3, eb3);
    let e3b4 = mp_cc(e3, eb4);
    let e3k = k.e3k;
    let value = if rmn >= 0 {
        let kbolb2 = mp_rr(eb2, kbol);
        let abbr9 = b[2] * e3b2 * e3b2
            + b[7] * e3b2 * e3b3
            + b[4] * e3b3 * e3b3
            + b[8] * e3b2 * e3b4
            + b[6] * e3b4 * e3b4;
        -(abbr9 * k.f)
            + e3k
                * 2.
                * (b[1] * e3b2
                    + b[3] * e3b3
                    + b[5] * e3b4
                    + b[2] * 2. * e3b2 * kbolb2
                    + b[7] * e3b3 * kbolb2
                    + b[8] * e3b4 * kbolb2)
    } else {
        e3k * 2. * (b[1] * e3b2 + b[3] * e3b3 + b[5] * e3b4)
    };
    divnum[(rmn + 1) as usize] -= value * 0.25 * (k.ie3k * k.ie3k);
}

/// The default target relative accuracy of [`FastOptions::rescue_target`]:
/// `2^-30`, about four sevenths of binary64's 53 bits (nine of its sixteen
/// digits). The trigger's estimate `2^-53 Σ|terms| / |finite|` measures the
/// rounding of the final contraction; the measured binary64 error of the
/// degenerate lanes is 5 to 50 times `2^-53` times their largest term
/// (`docs/triangle-gram-conditioning.md`, section 1), so a lane served below
/// the target keeps about seven digits at worst, and the calibration of
/// section 4 (`g g > g g g`, `g g > g g g g`: no point below 8 digits) holds.
pub const DEFAULT_RESCUE_TARGET: f64 = 9.313225746154785e-10;

thread_local! {
    /// The double-double workspaces of this thread, per `(n, rank)`, built on
    /// the first rescue (kept out of [`FastAmplitude`], whose layout the
    /// default stages' machine code depends on).
    static DD_STAGES: RefCell<Vec<((usize, usize), Box<super::dd_driver::DdStages>)>> =
        const { RefCell::new(Vec::new()) };
    /// Lanes re-reduced in double-double since the last [`take_rescued_lanes`].
    static RESCUED: Cell<u64> = const { Cell::new(0) };
}

/// The largest measured excess of a box cut's loss over the box-Gram model
/// `(r + 1)/2 log2 λ²`, in bits (`docs/box-gram-conditioning.md`, section 3:
/// synthetic lanes, double-double and MPFR), added to the model in the
/// re-reduced lane's estimate.
const BOX_GRAM_EXCESS_BITS: f64 = 11.3;

/// The weight of a double-double term in a re-reduced lane's estimate,
/// `2^-100`: double-double's `2^-104` (two bits under its `2^-106`) with four
/// bits of margin; each double-double `C0` term is further divided by its
/// triangle's normalised Källén function δ (the provider's `C0` loses
/// `log2(1/δ)` bits, measured against MPFR). Measured: no served lane above
/// the target on two independent sets of exactly consistent synthetic lanes
/// (`docs/triangle-gram-conditioning.md`, section 5).
const DD_TERM_BITS: i32 = 100;

/// The unit at which a re-reduced lane counts the box mu-expansion screen's
/// rounding estimate (`crate::conditioning::box_mu_rounding`, in units of the
/// unit roundoff): double-double's `2^-104`, two bits under its `2^-106`.
const DD_MU_BITS: i32 = 104;

thread_local! {
    /// The re-reductions of this thread's last evaluation ([`take_rescues`]).
    static RESCUES: RefCell<Vec<(usize, super::Rescue)>> = const { RefCell::new(Vec::new()) };
}

/// The lanes this thread's last evaluation re-reduced
/// ([`FastOptions::cancellation_rescue`]), with their precision and estimate
/// ([`super::Rescue`]); cleared when the next evaluation starts.
pub fn take_rescues() -> Vec<(usize, super::Rescue)> {
    RESCUES.with(|r| std::mem::take(&mut *r.borrow_mut()))
}

/// The number of lanes this thread re-reduced in double-double
/// ([`FastOptions::cancellation_rescue`]) since the last call.
pub fn take_rescued_lanes() -> u64 {
    RESCUED.with(|r| r.replace(0))
}

/// Master rows of one lane in `integrate`'s collection order: the cursor of
/// each of the five families at the lane's first row.
#[derive(Clone, Copy, Default)]
struct Cursor([usize; 5]);

impl FastAmplitude {
    /// The cursor of every active lane in `integrate`'s collection order.
    fn master_cursors(&self, input: &FastInput) -> Vec<(usize, Cursor)> {
        let n = self.topo.n;
        let x1 = self.topo.x1();
        let eps = self.options.infrared_eps;
        let (any_box, any_tri, any_bub, any_tad) = (
            self.any_box(),
            self.any_triangle(),
            self.any_bubble(),
            self.any_tadpole(),
        );
        let mut cursor = [0usize; 5];
        let mut out = Vec::with_capacity(self.active.len());
        for &lane in &self.active {
            out.push((lane, Cursor(cursor)));
            let m = &input.masses_squared[lane * n..(lane + 1) * n];
            let s = &self.smat[lane * n * n..(lane + 1) * n * n];
            let v = &input.shifts[lane * n..(lane + 1) * n];
            if any_box {
                cursor[0] += self.topo.boxes.len();
            }
            if any_tri {
                cursor[1] += self.topo.triangles.len();
            }
            if any_bub {
                for (b, &[i1, i2]) in self.topo.bubbles.iter().enumerate() {
                    let k = s[i2 * n + i1];
                    if is_zero(m[i1]) && is_zero(m[i2]) && k.abs() < eps {
                        continue;
                    }
                    let family = if x1 && self.rank3_bubble(b, lane, v) { 3 } else { 2 };
                    cursor[family] += 1;
                }
            }
            if any_tad {
                cursor[4] += (0..n).filter(|&i| !is_zero(m[i])).count();
            }
        }
        out
    }

    /// The moduli of the finite parts of one lane's integrated terms, in
    /// `integrate`'s order: `|c M|` per coefficient and master, the bubble
    /// tensor terms one by one, and every rational term.
    fn term_moduli(&self, input: &FastInput, lane: usize, cursor: Cursor) -> f64 {
        let n = self.topo.n;
        let lanes = self.lanes;
        let x1 = self.topo.x1();
        let eps = self.options.infrared_eps;
        let m = &input.masses_squared[lane * n..(lane + 1) * n];
        let s = &self.smat[lane * n * n..(lane + 1) * n * n];
        let v = &input.shifts[lane * n..(lane + 1) * n];
        let mut cursor = cursor.0;
        let mut sum = 0.;
        if self.any_box() {
            for b in 0..self.topo.boxes.len() {
                let [c0, c4] = self.box_c[b * lanes + lane];
                sum += (c0 * self.masters[0].out[cursor[0]][0]).abs() + c4.abs() / 6.;
                cursor[0] += 1;
            }
        }
        if self.any_triangle() {
            for (t, &[i1, i2, i3]) in self.topo.triangles.iter().enumerate() {
                let c = &self.tri[t * lanes + lane].c;
                sum += (c[0] * self.masters[1].out[cursor[1]][0]).abs() + c[7].abs() * 0.5;
                if x1 {
                    let (pji, pjk, pki) = (s[i2 * n + i1], s[i3 * n + i2], s[i1 * n + i3]);
                    sum += (c[14] * (C64::real(0.25 * (pjk + pji + pki)) - m[i1] - m[i2] - m[i3]))
                        .abs()
                        / 6.;
                }
                cursor[1] += 1;
            }
        }
        if self.any_bubble() {
            for (b, &[i1, i2]) in self.topo.bubbles.iter().enumerate() {
                let k = s[i2 * n + i1];
                if is_zero(m[i1]) && is_zero(m[i2]) && k.abs() < eps {
                    continue;
                }
                let data = &self.bub[b * lanes + lane];
                let c = &data.c;
                let ke2 = mp_rr(&sub(&v[i2], &v[i1]), &data.e2);
                let rank3 = x1 && self.rank3_bubble(b, lane, v);
                let (family, width) = if rank3 { (3, 4) } else { (2, 3) };
                let out = &self.masters[family].out[cursor[family] * width..(cursor[family] + 1) * width];
                cursor[family] += 1;
                let b06 = -(C64::real(k) - (m[i1] + m[i2]) * 3.) / 6.;
                sum += (c[0] * out[0][0]).abs()
                    + (c[1] * ke2 * out[1][0]).abs()
                    + (c[2] * ke2 * ke2 * out[2][0]).abs()
                    + (b06 * c[9]).abs();
                if rank3 {
                    sum += (c[13] * ke2 * ke2 * ke2 * out[3][0]).abs();
                }
                if x1 {
                    sum += ((C64::real(k) - m[i1] * 2. - m[i2] * 4.) * ke2 * c[10]).abs() / 12.;
                }
            }
        }
        if self.any_tadpole() {
            for i in 0..n {
                if is_zero(m[i]) {
                    continue;
                }
                let integral = &self.masters[4].out[cursor[4]];
                cursor[4] += 1;
                let data = &self.tad[i * lanes + lane];
                if x1 {
                    let c0 = data.c0 + m[i] * (data.e34 * 0.25) * data.c15;
                    sum += (c0 * integral[0]).abs()
                        + ((data.c14 + data.c15 * (data.e34 * 0.25)) * m[i] * m[i] * 0.5).abs();
                } else {
                    sum += (data.c0 * integral[0]).abs();
                }
            }
        }
        sum
    }

    /// The rescue after an evaluation published its lanes
    /// ([`FastOptions::cancellation_rescue`]): measure every successful lane
    /// (and every lane the box mu-expansion screen refused), re-reduce in
    /// double-double those whose estimate misses the target, publish the
    /// accepted re-reductions and refuse the others, typed. Nothing the rescue
    /// meets is an error of the evaluation: a failure of its own stages
    /// refuses its lanes.
    #[inline(never)]
    fn rescue_after<N: FastNumerator + ?Sized, M: FastMasters + ?Sized>(
        &mut self,
        input: &FastInput,
        num: &N,
        masters: &mut M,
        refinement: Option<&dyn FastRefinementSource>,
        results: &mut [FastResult],
        statuses: &mut [FastStatus],
    ) -> Result<()> {
        if !self.options.cancellation_rescue {
            return Ok(());
        }
        for (lane, served) in self.rescue_cancelling_lanes(input, num, masters, refinement) {
            if served {
                results[lane] = self.staged[lane];
                statuses[lane] = FastStatus::Success;
                // A box mu-expansion refusal the re-reduction cured.
                INSTABILITIES.with(|list| {
                    list.borrow_mut()
                        .retain(|(l, i)| !(*l == lane && i.test == InstabilityTest::BoxMuConditioning))
                });
            } else {
                // Typed refusal; the slot carries the best available value
                // (`Rescue::held_bits`, `held_estimate`).
                results[lane] = self.staged[lane];
                statuses[lane] = FastStatus::Unstable;
            }
        }
        Ok(())
    }

    /// Record a lane's refusal by the rescue (`estimate`, `absolute`: the
    /// re-reduction's own, NaN when it has none; `held`: the precision in
    /// bits and the relative estimate of the value the lane's slot keeps,
    /// the better of the binary64 and the re-reduced one).
    fn refuse_lane(&mut self, lane: usize, estimate: f64, absolute: f64, held: (u32, f64)) {
        self.lane_status[lane] = FastStatus::Unstable;
        RESCUES.with(|list| {
            list.borrow_mut().push((
                lane,
                super::Rescue {
                    precision_bits: 106,
                    estimate,
                    absolute,
                    served: false,
                    outcome: if estimate.is_nan() {
                        super::RescueOutcome::Failed
                    } else {
                        super::RescueOutcome::Missed
                    },
                    held_bits: held.0,
                    held_estimate: held.1,
                },
            ))
        });
        INSTABILITIES.with(|list| {
            list.borrow_mut().push((
                lane,
                Instability {
                    test: InstabilityTest::CancellationRescue,
                    value: held.1,
                    bound: self.options.rescue_target,
                },
            ))
        });
    }

    /// The trigger and the re-reduction; returns per handled lane whether its
    /// re-reduction was served.
    #[cold]
    #[inline(never)]
    fn rescue_cancelling_lanes<N: FastNumerator + ?Sized, M: FastMasters + ?Sized>(
        &mut self,
        input: &FastInput,
        num: &N,
        masters: &mut M,
        refinement: Option<&dyn FastRefinementSource>,
    ) -> Vec<(usize, bool)> {
        let unit = 2f64.powi(-(f64::MANTISSA_DIGITS as i32));
        let target = self.options.rescue_target;
        let screened: Vec<usize> = INSTABILITIES.with(|list| {
            list.borrow()
                .iter()
                .filter(|(_, i)| i.test == InstabilityTest::BoxMuConditioning)
                .map(|&(l, _)| l)
                .collect()
        });
        let cursors = self.master_cursors(input);
        let mut flagged = Vec::new();
        let mut out = Vec::new();
        for &(lane, cursor) in &cursors {
            let candidate = self.lane_status[lane] == FastStatus::Success
                || (self.lane_status[lane] == FastStatus::Unstable && screened.contains(&lane));
            if !candidate {
                continue;
            }
            // Relative to the lane's value; a lane without terms (exactly
            // zero, e.g. a massless tadpole) has nothing to lose.
            let finite = self.staged[lane].laurent[0].abs();
            let relative = |x: f64| if x == 0. { 0. } else { x / finite };
            let moduli = self.term_moduli(input, lane, cursor);
            let mut estimate = relative(unit * moduli);
            // The same in absolute terms, for the comparison with the
            // re-reduction (the box-Gram loss taken of all terms).
            let mut absolute = (unit * moduli).max(unit * self.mu_error[lane]);
            // The box coefficients' own loss at a nearly degenerate box Gram,
            // which the final contraction does not show.
            for b in 0..self.topo.boxes.len() {
                let gram = box_gram_error(self.box_lambda[b * self.lanes + lane], self.topo.rank, f64::MANTISSA_DIGITS);
                estimate = estimate.max(gram);
                absolute = absolute.max(gram * moduli);
            }
            // The box mu-expansion screen's first-order rounding estimate
            // (`crate::conditioning`, units of the unit roundoff), which the
            // contraction does not show either.
            estimate = estimate.max(relative(unit * self.mu_error[lane]));
            if estimate <= target && self.lane_status[lane] == FastStatus::Success {
                continue;
            }
            if num.tensor_coefficients(lane).is_none() {
                // No re-reduction for this numerator: today's behaviour
                // (`docs/triangle-gram-conditioning.md`, section 5).
                continue;
            }
            flagged.push((lane, cursor, estimate, absolute));
        }
        if flagged.is_empty() {
            return out;
        }
        let (n, rank) = (self.topo.n, self.topo.rank);
        let rescued = DD_STAGES.with(|cell| {
            let mut list = cell.borrow_mut();
            let index = match list.iter().position(|(key, _)| *key == (n, rank)) {
                Some(index) => index,
                None => {
                    list.push((
                        (n, rank),
                        Box::new(super::dd_driver::DdStages::new(n, rank, self.options)),
                    ));
                    list.len() - 1
                }
            };
            let dd = &mut list[index].1;
            dd.set_options(self.options);
            self.rescue_with(dd, input, num, masters, refinement, &flagged)
        });
        out.extend(rescued);
        out
    }

    /// The double-double re-reduction of the flagged lanes (whole lanes: every
    /// stage and every master at one kinematic point, the refined one or the
    /// projected one; only the numerator's coefficients are binary64) and
    /// their acceptance: served when its own estimate meets the target and is
    /// at most a sixteenth of the binary64 estimate, else refused.
    fn rescue_with<N: FastNumerator + ?Sized, M: FastMasters + ?Sized>(
        &mut self,
        dd: &mut super::dd_driver::DdStages,
        input: &FastInput,
        num: &N,
        masters: &mut M,
        refinement: Option<&dyn FastRefinementSource>,
        flagged: &[(usize, Cursor, f64, f64)],
    ) -> Vec<(usize, bool)> {
        use super::dd::{Cd, Dd, mp_rr as mp_rr_d};
        use super::dd_driver::{DdTensor, project_light_like};
        let n = self.topo.n;
        let x1 = self.topo.x1();
        let eps = self.options.infrared_eps;
        let target = self.options.rescue_target;
        let k = flagged.len();
        let mut shifts = Vec::with_capacity(k * n);
        let mut masses = Vec::with_capacity(k * n);
        let mut smat = Vec::with_capacity(k * n * n);
        let mut smat_dd: Vec<Dd> = Vec::with_capacity(k * n * n);
        let mut consistent = vec![true; k];
        // Lanes at the caller's refined point: every master must then be the
        // provider's double-double one at that point (one point per lane).
        let mut at_refined = vec![false; k];
        let mut tensor = DdTensor::new(self.topo.rank);
        let (mut lo_v, mut lo_s, mut lo_m) =
            (vec![[0.; 4]; n], vec![0.; n * n], vec![C64::ZERO; n]);
        for (r, &(lane, _, _, _)) in flagged.iter().enumerate() {
            let s = &self.smat[lane * n * n..(lane + 1) * n * n];
            let hi = &input.shifts[lane * n..(lane + 1) * n];
            lo_v.fill([0.; 4]);
            lo_s.fill(0.);
            lo_m.fill(C64::ZERO);
            let mut refined = refinement.is_some_and(|src| src.lane(lane, &mut lo_v, &mut lo_s, &mut lo_m));
            let first = shifts.len();
            if refined {
                // Exact two-sums: the low parts need not be normalised.
                let v: Vec<[Dd; 4]> = hi
                    .iter()
                    .zip(&lo_v)
                    .map(|(h, l)| std::array::from_fn(|mu| Dd::from(h[mu]) + l[mu]))
                    .collect();
                // A refinement is used only when it is consistent to
                // double-double: every pair the binary64 pass takes as exactly
                // zero, or as a real squared mass of the lane, and (with an
                // invariant matrix) every given invariant, within 2^-100 of the
                // lane's scale of its own shifts; otherwise the projection.
                let scale = v.iter().flatten().fold(Dd::ZERO, |a, x| a.max(x.abs()));
                let tolerance = (scale * scale * 4.).to_f64() * 2f64.powi(-100);
                let mut ok = true;
                for i in 0..n {
                    for j in 0..i {
                        let d: [Dd; 4] = std::array::from_fn(|mu| v[i][mu] - v[j][mu]);
                        let own = mp_rr_d(&d, &d);
                        let given = s[i * n + j];
                        let expected = if given.abs() < eps {
                            Some(Dd::ZERO)
                        } else if let Some(q) = (0..n).find(|&q| {
                            let mq = input.masses_squared[lane * n + q];
                            mq.im == 0. && (given - mq.re).abs() < eps
                        }) {
                            Some(Dd::from(input.masses_squared[lane * n + q].re) + lo_m[q].re)
                        } else if input.s_matrix.is_some() && !lo_s[i * n + j].is_nan() {
                            Some(Dd::from(given) + lo_s[i * n + j])
                        } else {
                            None
                        };
                        if let Some(x) = expected {
                            ok &= (own - x).abs().to_f64() <= tolerance;
                        }
                    }
                }
                if ok {
                    shifts.extend(v);
                } else {
                    refined = false;
                }
            }
            if !refined {
                let mut v: Vec<[Dd; 4]> = hi.iter().map(|x| x.map(Dd::from)).collect();
                consistent[r] = project_light_like(&mut v, s, n, eps);
                shifts.extend(v);
            }
            at_refined[r] = refined;
            // A caller-given invariant strictly between zero and
            // `infrared_eps` is zero by the engine's convention (cut bases,
            // scaleless bubbles, on-shell masters) but not by the caller's
            // kinematics: the lane is not defined well enough for a
            // re-reduction to certify it (typed `Failed`; its slot keeps the
            // binary64 value). Invariants NinjaRS computed from the shifts
            // (no matrix given) are projected instead.
            if input.s_matrix.is_some() && s.iter().any(|&x| x != 0. && x.abs() < eps) {
                consistent[r] = false;
            }
            let vd = &shifts[first..first + n];
            for i in 0..n {
                for j in 0..n {
                    // Refined: the source's invariant (NaN = the refined
                    // shifts'); else the projected geometry's.
                    smat_dd.push(if refined && input.s_matrix.is_some() && !lo_s[i * n + j].is_nan() {
                        Dd::from(s[i * n + j]) + lo_s[i * n + j]
                    } else {
                        let d: [Dd; 4] = std::array::from_fn(|mu| vd[i][mu] - vd[j][mu]);
                        mp_rr_d(&d, &d)
                    });
                }
            }
            masses.extend((0..n).map(|i| {
                let h = Cd::from_c64(input.masses_squared[lane * n + i]);
                if refined { h + Cd::from_c64(lo_m[i]) } else { h }
            }));
            smat.extend_from_slice(s);
            tensor.push_lane(num.tensor_coefficients(lane).expect("checked by the caller"));
        }
        let ddin = super::dd_driver::FastInput {
            lanes: k,
            shifts: &shifts,
            masses_squared: &masses,
        };
        dd.prepare(&ddin, &smat);
        let stages = (|| -> Result<()> {
            if dd.any_box() {
                dd.boxes(&ddin, &mut tensor)?;
            }
            if dd.any_triangle() {
                dd.triangles(&ddin, &mut tensor)?;
            }
            if dd.any_bubble() {
                dd.bubbles(&ddin, &mut tensor)?;
            }
            if dd.any_tadpole() {
                dd.tadpoles(&ddin, &mut tensor)?;
            }
            Ok(())
        })();
        let mut outcome = Vec::with_capacity(k);
        for (r, &(lane, cursor, binary64_estimate, binary64_absolute)) in flagged.iter().enumerate() {
            if stages.is_err() || !dd.alive[r] || !consistent[r] {
                // A degenerate or non-finite quantity, or a projection that
                // could not be solved: no estimate, a typed refusal.
                self.refuse_lane(lane, f64::NAN, f64::NAN, (53, binary64_estimate));
                outcome.push((lane, false));
                continue;
            }
            let mut cursor = cursor.0;
            let m = &input.masses_squared[lane * n..(lane + 1) * n];
            let s = &self.smat[lane * n * n..(lane + 1) * n * n];
            let md = &masses[r * n..(r + 1) * n];
            let vd = &shifts[r * n..(r + 1) * n];
            let sd = &smat_dd[r * n * n..(r + 1) * n * n];
            let zero = |x: f64| x.abs() < eps;
            // The masters' invariants: the lane's point, except that an
            // invariant the binary64 pass takes as zero, or as a real squared
            // mass of the lane, is exactly that (the zero predicate of the
            // bubble skip and of the binary64 masters' on-shell threshold).
            let inv = |i: usize, j: usize| -> Cd {
                let given = s[i * n + j];
                if zero(given) {
                    return Cd::ZERO;
                }
                if let Some(q) = (0..n).find(|&q| m[q].im == 0. && (given - m[q].re).abs() < eps) {
                    return md[q];
                }
                Cd::real(sd[i * n + j])
            };
            let mu2 = input.scales_squared[lane];
            let mu2c = Cd::real(Dd::from(mu2));
            let mut rescue_out = [[Cd::ZERO; 3]; 4];
            let mut rescue_ok = [false];
            let mut missing = false;
            let mut provided = |family: FastMaster, args: &[Cd], width: usize| -> Option<[[Cd; 3]; 4]> {
                rescue_ok[0] = false;
                let given = masters
                    .evaluate_rescue(family, args, &mut rescue_out[..width], &mut rescue_ok)
                    .unwrap_or(false);
                missing |= !(given && rescue_ok[0]);
                (given && rescue_ok[0]).then_some(rescue_out)
            };
            let mut res = [Cd::ZERO; 3];
            let mut rational = Cd::ZERO;
            // The moduli of the finite terms, split by the precision of their
            // least precise factor: the re-reduced lane's own estimate.
            let (mut m64, mut mdd) = (0f64, 0f64);
            let size = |z: Cd| z.abs().to_f64();
            if self.any_box() {
                for (b, &[i1, i2, i3, i4]) in self.topo.boxes.iter().enumerate() {
                    let binary64 = self.masters[0].out[cursor[0]];
                    cursor[0] += 1;
                    let [c0, c4] = dd.box_c[b * k + r];
                    let args = [
                        inv(i2, i1), inv(i3, i2), inv(i4, i3), inv(i1, i4), inv(i3, i1), inv(i4, i2),
                        md[i1], md[i2], md[i3], md[i4], mu2c,
                    ];
                    let given = provided(FastMaster::D0, &args, 1);
                    let integral = given.map_or_else(|| binary64.map(Cd::from_c64), |o| o[0]);
                    for o in 0..3 {
                        res[o] += c0 * integral[o];
                    }
                    rational += c4 * (-1. / 6.);
                    if given.is_some() { mdd += size(c0 * integral[0]) } else { m64 += size(c0 * integral[0]) }
                    mdd += size(c4) / 6.;
                }
            }
            if self.any_triangle() {
                for (t, &[i1, i2, i3]) in self.topo.triangles.iter().enumerate() {
                    let binary64 = self.masters[1].out[cursor[1]];
                    cursor[1] += 1;
                    let given = provided(
                        FastMaster::C0,
                        &[inv(i2, i1), inv(i3, i2), inv(i1, i3), md[i1], md[i2], md[i3], mu2c],
                        1,
                    )
                    .map(|o| o[0]);
                    let master_dd = given.is_some();
                    let integral = given.unwrap_or_else(|| binary64.map(Cd::from_c64));
                    let c = &dd.tri[t * k + r].c;
                    for o in 0..3 {
                        res[o] += c[0] * integral[o];
                    }
                    let before = rational;
                    if x1 {
                        let shape = Cd::real(sd[i2 * n + i1] + sd[i3 * n + i2] + sd[i1 * n + i3]) * 0.25
                            - md[i1]
                            - md[i2]
                            - md[i3];
                        rational += c[7] * 0.5 + c[14] * shape / 6.;
                    } else {
                        rational += c[7] * 0.5;
                    }
                    if master_dd {
                        // The provider's double-double C0 loses log2(1/δ) bits
                        // at a nearly degenerate triangle (δ its normalised
                        // Källén function; measured against 200-bit MPFR: 2^-104
                        // / δ for massless, massive and complex lines).
                        let [a, b, cc] = [inv(i2, i1), inv(i3, i2), inv(i1, i3)].map(|z| z.re.to_f64());
                        let big = a.abs().max(b.abs()).max(cc.abs());
                        let kallen = (a * a + b * b + cc * cc - 2. * (a * b + b * cc + cc * a)).abs().sqrt();
                        let delta = if big > 0. { (kallen / big).clamp(f64::MIN_POSITIVE, 1.) } else { 1. };
                        mdd += size(c[0] * integral[0]) / delta;
                    } else {
                        m64 += size(c[0] * integral[0]);
                    }
                    mdd += size(rational - before);
                }
            }
            // The binary64 tadpole rows of this lane, per line (zero if massless).
            let a0_rows: Vec<[C64; 3]> = if !self.any_tadpole() {
                vec![[C64::ZERO; 3]; n]
            } else {
                let mut row = cursor[4];
                (0..n)
                    .map(|i| {
                        if is_zero(m[i]) {
                            [C64::ZERO; 3]
                        } else {
                            row += 1;
                            self.masters[4].out[row - 1]
                        }
                    })
                    .collect()
            };
            if self.any_bubble() {
                for (b, &[i1, i2]) in self.topo.bubbles.iter().enumerate() {
                    let kk = s[i2 * n + i1];
                    if is_zero(m[i1]) && is_zero(m[i2]) && kk.abs() < eps {
                        continue;
                    }
                    let data = &dd.bub[b * k + r];
                    let c = &data.c;
                    let kv: [Dd; 4] = std::array::from_fn(|mu| vd[i2][mu] - vd[i1][mu]);
                    let ke2 = mp_rr_d(&kv, &data.e2);
                    let rank3 = x1 && self.rank3_bubble(b, lane, &input.shifts[lane * n..(lane + 1) * n]);
                    let (family, width) = if rank3 { (3, 4) } else { (2, 3) };
                    let out = &self.masters[family].out
                        [cursor[family] * width..(cursor[family] + 1) * width];
                    cursor[family] += 1;
                    let kd = sd[i2 * n + i1];
                    let given = provided(
                        if rank3 { FastMaster::BubbleRank3 } else { FastMaster::BubbleRank2 },
                        &[inv(i2, i1), md[i1], md[i2], mu2c],
                        width,
                    );
                    // Without the provider's double-double tensors: its binary64
                    // ones (in the estimate at 2^-53).
                    let master_dd = given.is_some();
                    let tensors = given.unwrap_or_else(|| {
                        std::array::from_fn(|w| if w < width { out[w].map(Cd::from_c64) } else { [Cd::ZERO; 3] })
                    });
                    let c1 = c[1] * ke2;
                    let c2 = c[2] * ke2 * ke2;
                    let c13 = if rank3 { c[13] * ke2 * ke2 * ke2 } else { Cd::ZERO };
                    for o in 0..3 {
                        res[o] += c[0] * tensors[0][o]
                            + c1 * tensors[1][o]
                            + c2 * tensors[2][o]
                            + c13 * tensors[3][o];
                    }
                    let term = size(c[0] * tensors[0][0])
                        + size(c1 * tensors[1][0])
                        + size(c2 * tensors[2][0])
                        + size(c13 * tensors[3][0]);
                    if master_dd { mdd += term } else { m64 += term }
                    let before = rational;
                    let b06 = -(Cd::real(kd) - (md[i1] + md[i2]) * 3.) / 6.;
                    if x1 {
                        rational += b06 * c[9]
                            + (Cd::real(kd) - md[i1] * 2. - md[i2] * 4.) * ke2 * c[10] / 12.;
                    } else {
                        rational += b06 * c[9];
                    }
                    mdd += size(rational - before);
                }
            }
            if self.any_tadpole() {
                for i in 0..n {
                    if is_zero(m[i]) {
                        continue;
                    }
                    let data = &dd.tad[i * k + r];
                    let given = provided(FastMaster::A0, &[md[i], mu2c], 1).map(|o| o[0]);
                    let integral = given.unwrap_or_else(|| a0_rows[i].map(Cd::from_c64));
                    let mi = md[i];
                    let (res_before, rational_before) = (res[0], rational);
                    if x1 {
                        let c0 = data.c0 + mi * (data.e34 * 0.25) * data.c15;
                        for o in 0..3 {
                            res[o] += c0 * integral[o];
                        }
                        rational += (data.c14 + data.c15 * (data.e34 * 0.25)) * mi * mi * 0.5;
                    } else {
                        for o in 0..3 {
                            res[o] += data.c0 * integral[o];
                        }
                    }
                    if given.is_some() { mdd += size(res[0] - res_before) } else { m64 += size(res[0] - res_before) }
                    mdd += size(rational - rational_before);
                }
            }
            let result = FastResult {
                laurent: [(res[0] + rational).to_c64(), res[1].to_c64(), res[2].to_c64()],
                cut_constructible: res[0].to_c64(),
                rational: rational.to_c64(),
            };
            let finite = result.laurent[0].abs();
            let moduli = m64 + mdd;
            // The re-reduced lane's own absolute error estimate: terms with a
            // binary64 factor at 2^-53, double-double terms at
            // 2^-DD_TERM_BITS, and the rounding of the served binary64 value;
            // inside the box stage the double-double coefficients themselves
            // lose `(r + 1)/2 log2 λ²` bits at a nearly degenerate box Gram
            // (a relative error of the box terms, taken of all terms); the box
            // mu-expansion screen's rounding estimate of the same boxes, at
            // the double-double unit.
            let rounding = m64 * 2f64.powi(-53) + mdd * 2f64.powi(-DD_TERM_BITS) + finite * 2f64.powi(-53);
            let gram = (0..self.topo.boxes.len())
                .map(|b| box_gram_error(dd.box_lambda[b * k + r], self.topo.rank, 106))
                .fold(0f64, f64::max);
            let mu = self.mu_error[lane] * 2f64.powi(-DD_MU_BITS);
            let absolute = rounding.max(gram * moduli).max(mu);
            // Its relative estimate (the box-Gram loss as calibrated: relative
            // to the lane's value).
            let estimate = (rounding / finite).max(gram).max(mu / finite);
            // The binary64 lane's absolute estimate, for the comparison: a
            // served value must be clearly better than the binary64 one, at
            // most 1/16 of its estimate (the binary64 error ranges over about
            // 0.03 to 1 times its estimate on the dumped lanes, the
            // re-reduced lane's at most its estimate: four bits keep a served
            // value from being worse).
            let finite_ok = result.laurent.iter().all(|z| z.is_finite()) && result.rational.is_finite();
            // At a refined point a binary64 master (at the input point) would
            // mix two kinematic points in one lane: refused.
            let one_point = !(at_refined[r] && missing);
            let better = absolute <= binary64_absolute / 16.;
            // Accepted: its own estimate meets the target; anything else is a
            // typed refusal. (A lane below binary64's resolution of its own
            // terms is NOT served on that ground: its value is an ordinary
            // part of the point the caller sums, and no absolute scale of the
            // point is known here.)
            let outcome_of = if !(finite_ok && one_point && better) {
                super::RescueOutcome::Missed
            } else if estimate <= target {
                super::RescueOutcome::Accepted
            } else {
                super::RescueOutcome::Missed
            };
            let served = outcome_of != super::RescueOutcome::Missed;
            if served {
                self.staged[lane] = result;
                self.lane_status[lane] = FastStatus::Success;
                RESCUES.with(|list| {
                    list.borrow_mut().push((
                        lane,
                        super::Rescue {
                            precision_bits: 106,
                            estimate,
                            absolute,
                            served: true,
                            outcome: outcome_of,
                            held_bits: 106,
                            held_estimate: estimate,
                        },
                    ))
                });
            } else if !finite_ok {
                self.refuse_lane(lane, f64::NAN, f64::NAN, (53, binary64_estimate));
            } else {
                // Refused, typed; the slot keeps the better value: the
                // re-reduced one when it is at one point and its absolute
                // estimate is below the binary64 lane's.
                let held = if one_point && absolute < binary64_absolute {
                    self.staged[lane] = result;
                    (106, estimate)
                } else {
                    (53, binary64_estimate)
                };
                self.refuse_lane(lane, estimate, absolute, held);
            }
            outcome.push((lane, served));
        }
        RESCUED.with(|c| c.set(c.get() + outcome.iter().filter(|(_, s)| *s).count() as u64));
        outcome
    }
}
