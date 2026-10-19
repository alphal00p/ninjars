//! Batched binary64 port of original Ninja's `Amplitude::evaluate` with its
//! default options (no tests, mu-expansion for the box rational term,
//! `ComplexMasses`), including the rank n+1 path of `Xninja.cc`. Rank n+2
//! (beyond original Ninja) is a separate engine, [`super::engine_x2`],
//! selected once in [`FastAmplitude::new`]; nothing below depends on it.
//!
//! Every stage processes one cut for all active lanes: geometry for each lane,
//! one numerator callback per cut (both orientations of a triangle or of a
//! rank n+1 tadpole share a call), then lane-local divisions by the uncut
//! denominators and coefficient-level subtractions. All master requests of
//! all cuts and lanes are collected and evaluated with one provider call per
//! family before the final contraction in original Ninja's order.

// Index loops mirror original Ninja's component and lane loops.
#![allow(clippy::needless_range_loop)]

use super::complex::{C64, CMom, RMom, mp_cc, mp_cr, mp_rr};
use super::engine_x2::{DynMasters, DynNumerator, RankN2Engine, evaluate_x2};
use super::geometry::{
    Basis, box_solutions, bubble_solution, cmom_from_real, combine_real2, gram_det, mp2,
    reference_vectors_ok, scale, sub, tadpole_candidates, triangle_solution,
};
use super::masters::{FastMaster, FastMasters};
use super::numerator::{FastNumerator, t2_len, t3_len};
use super::subtractions as subs;
use super::{FastInput, FastResult, FastStatus};
use crate::{Error, Result};

/// Options of the fast engine. [`Default`] reproduces original Ninja's
/// defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FastOptions {
    /// Minimal cut size (Ninja's `setCutStop`): stages with fewer cut
    /// propagators are skipped (and contribute nothing). `1` (or `0`)
    /// performs the full reduction.
    pub cut_stop: usize,
    /// Ninja's `INFRARED_EPS` (`1e-9`): massless-momentum threshold of the
    /// cut bases and spinors, and of the scaleless-bubble skip rule.
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
    pub relative_tolerance: f64,
}

impl Default for FastOptions {
    fn default() -> Self {
        Self {
            cut_stop: 1,
            infrared_eps: 1.0e-9,
            stability_threshold: 1.0e20 * f64::MIN_POSITIVE,
            relative_tolerance: 1024. * f64::EPSILON,
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
    fn new(n: usize, rank: usize) -> Self {
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

    fn rmn(&self) -> isize {
        self.rank as isize - self.n as isize
    }
    fn x1(&self) -> bool {
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
    pub(super) fn new(options: &FastOptions) -> Self {
        Self {
            threshold: options.stability_threshold,
            tolerance: options.relative_tolerance,
        }
    }
    /// Ninja's absolute check only.
    #[inline(always)]
    pub(super) fn absolute(self, x: f64) -> bool {
        x >= self.threshold
    }
    #[inline(always)]
    pub(super) fn relative(self, x: f64, scale: f64) -> bool {
        x >= self.threshold && x > self.tolerance * scale
    }
    /// [`Self::relative`] against `scale()`, evaluated only when `x` does not
    /// already exceed the tolerance times `bound >= scale()`.
    #[inline(always)]
    pub(super) fn relative_within(self, x: f64, bound: f64, scale: impl FnOnce() -> f64) -> bool {
        x >= self.threshold && (x > self.tolerance * bound || x > self.tolerance * scale())
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

/// Scale of the real-by-complex product `a.k`: `Σ_μ |a_μ|₁ |k_μ|`.
#[inline(always)]
pub(super) fn dot_scale(a: &CMom, k: &RMom) -> f64 {
    (0..4).map(|mu| a[mu].taxicab() * k[mu].abs()).sum()
}

#[inline(always)]
pub(super) fn is_zero(z: C64) -> bool {
    z.re == 0. && z.im == 0.
}

/// Mark a lane unstable; its later cuts and its results are skipped.
pub(super) fn fail(alive: &mut [bool], status: &mut [FastStatus], lane: usize) {
    if alive[lane] {
        alive[lane] = false;
        status[lane] = FastStatus::Unstable;
    }
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
    topo: Topology,
    options: FastOptions,
    tadpole_bases: [Basis; 3],
    lanes: usize,
    alive: Vec<bool>,
    lane_status: Vec<FastStatus>,
    active: Vec<usize>,
    smat: Vec<f64>,
    vsq: Vec<f64>,
    /// Per lane `(max |V_i^mu|, max |m_i²|₁)`: cheap upper bounds of the scales
    /// in the precision-aware stability checks.
    bounds: Vec<(f64, f64)>,
    box_c: Vec<[C64; 2]>,
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
    /// The rank n+2 engine, present iff `rank == n + 2`; it then performs
    /// every evaluation and the fields above stay unused.
    pub(super) x2: Option<Box<RankN2Engine>>,
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
    /// [`RankN2Engine`], chosen here once.
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
        Ok(Self {
            topo: Topology::new(n, rank),
            tadpole_bases: tadpole_candidates(options.infrared_eps),
            options,
            lanes: 0,
            alive: Vec::new(),
            lane_status: Vec::new(),
            active: Vec::new(),
            smat: Vec::new(),
            vsq: Vec::new(),
            bounds: Vec::new(),
            box_c: Vec::new(),
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
            x2: (rank == n + 2).then(|| Box::new(RankN2Engine::new(n, options))),
        })
    }

    pub fn propagators(&self) -> usize {
        self.topo.n
    }
    pub fn rank(&self) -> usize {
        self.topo.rank
    }
    pub fn options(&self) -> &FastOptions {
        &self.options
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
    pub fn evaluate<N: FastNumerator + ?Sized, M: FastMasters + ?Sized>(
        &mut self,
        input: &FastInput,
        numerator: &mut N,
        masters: &mut M,
        results: &mut [FastResult],
        statuses: &mut [FastStatus],
    ) -> Result<()> {
        if let Some(x2) = self.x2.as_deref_mut() {
            return evaluate_x2(
                x2,
                &self.topo,
                input,
                &mut DynNumerator(numerator),
                &mut DynMasters(masters),
                results,
                statuses,
            );
        }
        self.evaluate_up_to_n1(input, numerator, masters, results, statuses)
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
                    fail(&mut self.alive, &mut self.lane_status, lane);
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
                    fail(&mut self.alive, &mut self.lane_status, lane);
                    continue;
                }
                let l = box_solutions(&k3, &k0, &k1, &e, m[c1], m[c2], m[c3], m[c4]);
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
                    fail(&mut self.alive, &mut self.lane_status, lane);
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
            for r in 0..self.box_rows.len() {
                let row = self.box_rows[r];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let vort2 = mp_cc(&row.vort, &row.vort);
                let bound = 4. * self.bounds[lane].0 * taxicab_sum(&row.vort);
                if !x1 {
                    let mut d = C64::ONE;
                    let mut singular = false;
                    for &j in cp {
                        let factor = mp_cr(&row.vort, &sub(&v[j], &v[c1])) * 2.;
                        singular |= !check.relative_within(factor.taxicab(), bound, || {
                            difference_scale(&row.vort, &v[j], &v[c1])
                        });
                        d *= factor;
                    }
                    if singular || !check.absolute(d.taxicab()) {
                        fail(&mut self.alive, &mut self.lane_status, lane);
                        continue;
                    }
                    self.box_c[b * lanes + lane][1] = self.calls.out[r] / d / (vort2 * vort2);
                } else {
                    let mut nmu = [self.calls.out[2 * r], self.calls.out[2 * r + 1]];
                    let vsq = &self.vsq[lane * n..(lane + 1) * n];
                    let mut stable = true;
                    for &j in cp {
                        let kk = sub(&v[j], &v[c1]);
                        let d0 = mp_cr(&row.vort, &kk) * 2.;
                        if !check.relative_within(d0.taxicab(), bound, || {
                            difference_scale(&row.vort, &v[j], &v[c1])
                        }) {
                            stable = false;
                            break;
                        }
                        let d1 =
                            C64::real(vsq[j] - vsq[c1]) - m[j] + m[c1] + mp_cr(&row.mid, &kk) * 2.;
                        nmu[0] /= d0;
                        nmu[1] -= d1 * nmu[0];
                        nmu[1] /= d0;
                    }
                    if !stable {
                        fail(&mut self.alive, &mut self.lane_status, lane);
                        continue;
                    }
                    self.box_c[b * lanes + lane][1] = nmu[1] / (vort2 * vort2);
                }
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
                    fail(&mut self.alive, &mut self.lane_status, lane);
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
                    fail(&mut self.alive, &mut self.lane_status, lane);
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
        let width = t2_len(mindeg);
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
                    fail(&mut self.alive, &mut self.lane_status, lane);
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
            let width2 = t2_len(mindeg2);
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
                    fail(&mut self.alive, &mut self.lane_status, lane);
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
                    fail(&mut self.alive, &mut self.lane_status, lane);
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
                self.lane_status[lane] = FastStatus::NonFinite;
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
