//! Rank r = n+2 fast engine (beyond original Ninja).
//!
//! [`super::FastAmplitude::new`] builds a [`RankN2Engine`] once when
//! `rank == n + 2` and forwards every evaluation to it, so the rank <= n+1
//! engine in `engine.rs` is untouched. Stage order, skip rules (`cut_stop`,
//! scaleless bubbles, massless tadpoles), stability checks, lane compaction,
//! publish-on-success and statuses are those of `engine.rs`; the recipe per
//! cut is the one of `docs` design section 2 (confirmed exactly against an
//! independent reduction for n = 1..6):
//!
//! - **box**: `c0` from the two four-dimensional solutions as before; the
//!   mu-expansion runs on the exact d-dimensional cut, `q = q̄0 - V0 + t v`,
//!   `mu² = t² v² - beta` with `beta = m0² - q̄0²`
//!   ([`FastNumerator::mu_expansion_x2`]), keeping three orders:
//!   `c6 = n0/(v²)³` and `c4 = n2/(v²)² + 3 beta c6`.
//! - **triangle**: two `t3` expansions at `mindeg` 5 (12 slots,
//!   [`FastNumerator::t3_expansion_x2`]), divided by
//!   the uncut denominators, 21 coefficients; no subtractions.
//! - **bubble**: in `Basis(k)` with the reference vector that best
//!   conditions the uncut lines (`bases_x2.rs`; original Ninja fixes it),
//!   `t2` at `mindeg` 4 (22 slots, levels of 1, 2, 4, 6, 9;
//!   [`FastNumerator::t2_expansion_x2`]) and,
//!   for massive lines, the swapped orientation at `mindeg` 3; the parent
//!   triangles are subtracted by series division
//!   (`subtractions_series_x2.rs`); 35 coefficients in the canonical layout
//!   of `tools/rank_n2`.
//! - **tadpole**: in the best conditioned of original Ninja's three bases and
//!   six more (`bases_x2.rs`), two `t3` orientations at `mindeg` 3 (6 slots);
//!   triangle and bubble parents subtracted on slots 4 and 5; integrated
//!   layout as at rank n+1 (`c0`, `c14`, `c15`).
//! - **integration**: adds the box `mu⁶` rational term, the bubble
//!   `mu² (q.e2)²` and `mu⁴` rational terms and the rank-four bubble family
//!   ([`FastMaster::BubbleRank4`], `B1111`) when `c20 ke2⁴` is not exactly 0.

// Index loops mirror original Ninja's component and lane loops.
#![allow(clippy::needless_range_loop)]

use super::bases_x2::{TadpoleBasesX2, bubble_basis_x2};
use super::complex::{C64, CMom, RMom, mp_cc, mp_cr, mp_rr};
use super::engine::{
    FastOptions, MasterBatch, Stability, TadpoleData, Topology, Uncut, den, den_scale,
    difference_scale, divide_t2, divide_t3, dot_scale, fail, gram_scale, is_zero, scale_c, sub_cr,
    taxicab_sum,
};
use super::geometry::{
    Basis, box_solutions, bubble_solution, cmom_from_real, combine_real2, gram_det, mp2,
    reference_vectors_ok, scale, sub, triangle_solution,
};
use super::masters::{FastMaster, FastMasters};
use super::numerator::{FastNumerator, t2_len, t3_len};
use super::subtractions_series_x2 as subs;
use super::{FastInput, FastResult, FastStatus};
use crate::{Error, Result};

/// Triangle slots at `mindeg` 5, `(t, mu²)` per index (design 1.3).
const T3_WIDTH: usize = t3_len(5);
/// Bubble slots at `mindeg` 4 (orientation A) and 3 (orientation B).
const T2_WIDTH: usize = t2_len(4);
const T2_WIDTH_B: usize = t2_len(3);
/// Tadpole slots at `mindeg` 3: `(3,0) (2,0) (1,0) (1,1) (0,0) (0,1)`.
const TAD_WIDTH: usize = t3_len(3);

#[derive(Clone, Copy, Debug, Default)]
struct TriangleDataX2 {
    e3: CMom,
    e4: CMom,
    /// Canonical rank n+2 triangle layout (`c15..c20`: `u⁵ v⁵ mu²u³ mu²v³
    /// mu⁴u mu⁴v` with `u = q̄.e3`, `v = q̄.e4`).
    c: [C64; 21],
}

#[derive(Clone, Copy, Debug)]
struct BubbleDataX2 {
    e2: RMom,
    e3: CMom,
    e4: CMom,
    /// Canonical rank n+2 bubble layout (`c20 = w⁴`, `c21 = mu² w²`,
    /// `c22 = mu⁴`, `c23..c34` spurious; `w = q̄.e2`).
    c: [C64; 35],
}

impl Default for BubbleDataX2 {
    fn default() -> Self {
        Self {
            e2: [0.; 4],
            e3: [C64::ZERO; 4],
            e4: [C64::ZERO; 4],
            c: [C64::ZERO; 35],
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct BoxRowX2 {
    lane: usize,
    vort: CMom,
    q: [CMom; 2],
    /// `q̄0 - V0` with `q̄0 = (l+ + l-)/2`.
    mid: CMom,
    /// `m0² - q̄0²`: `mu² = t² v² - beta` on the d-dimensional cut.
    beta: C64,
}

#[derive(Clone, Copy, Debug, Default)]
struct TriangleRowX2 {
    lane: usize,
    e3: CMom,
    e4: CMom,
    amu: CMom,
    param: C64,
    e34: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct BubbleRowX2 {
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
struct TadpoleRowX2 {
    lane: usize,
    basis: usize,
}

/// A vector of one-element arrays: the layout of `Vec<T>` (read back with
/// `as_flattened`), but a vector type of its own.
///
/// This crate's rank n+2 code is not generic, so the vector routines it
/// uses are compiled here. Routines such as `RawVec::<T>::grow_one` are then
/// exported to downstream crates, which link to them instead of their own
/// copies, changing the machine code of the generic rank <= n+1 stages they
/// instantiate (`engine.rs` stages push `usize`, `CMom`, `C64`, `[C64; 3]` and
/// row types). Wrapping the element types keeps every routine used here
/// distinct; the row types above are separate types for the same reason.
trait Push1<T> {
    fn push1(&mut self, value: T);
}

impl<T> Push1<T> for Vec<[T; 1]> {
    #[inline(always)]
    fn push1(&mut self, value: T) {
        self.push([value]);
    }
}

/// Lane-major staging of numerator callback arguments (as `engine.rs`
/// `Calls`, with [`Push1`] element types).
#[derive(Default)]
struct CallsX2 {
    lanes: Vec<[usize; 1]>,
    v0: Vec<[CMom; 1]>,
    v1: Vec<[CMom; 1]>,
    v2: Vec<[CMom; 1]>,
    v3: Vec<[CMom; 1]>,
    scalar: Vec<[C64; 1]>,
    params: Vec<[[C64; 3]; 1]>,
    out: Vec<[C64; 1]>,
}

impl CallsX2 {
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
        self.out.resize(self.lanes.len() * width, [C64::ZERO]);
    }
}

/// Master batches: D0, C0, rank-2, rank-3 and rank-4 bubbles, A0.
const FAMILIES_X2: [FastMaster; 6] = [
    FastMaster::D0,
    FastMaster::C0,
    FastMaster::BubbleRank2,
    FastMaster::BubbleRank3,
    FastMaster::BubbleRank4,
    FastMaster::A0,
];
const A0_BATCH: usize = 5;

/// Polynomial division of a rank n+2 triangle Laurent expansion
/// (`mindeg` 5, six orders) by `d0 t + d1 + (d2[0] + d2[1] mu²)/t`, given
/// `1/d0` (`engine.rs` `divide_t3_n` at six orders).
#[inline(always)]
pub(super) fn divide_t3_6(num: &mut [C64], d0inv: C64, d1: C64, d2: [C64; 2]) {
    const NT: usize = 6;
    const OFF: [usize; 7] = [0, 1, 2, 4, 6, 9, 12];
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

/// Level offsets of the t2 layout up to `mindeg` 4 (level sizes 1, 2, 4, 6, 9).
const T2_OFF: [usize; 6] = [0, 1, 3, 7, 13, 22];
/// `d1` maps into level `i`: entry `e` of level `i-1`, `(a, j)`, feeds
/// `(a, j)` through `d1[0]` and `(a+1, j)` through `d1[1] x` (relative
/// indices in level `i`).
const T2_D1: [&[(usize, usize)]; 5] = [
    &[],
    &[(0, 1)],
    &[(0, 2), (2, 3)],
    &[(0, 2), (1, 3), (2, 4), (4, 5)],
    &[(0, 3), (1, 4), (3, 5), (4, 6), (5, 7), (7, 8)],
];
/// `d2` maps into level `i`: entry `e` of level `i-2`, `(a, j)`, feeds
/// `(a, j)`, `(a, j+1)`, `(a+1, j)` and `(a+2, j)` through `d2[0]`,
/// `d2[1] mu²`, `d2[2] x` and `d2[3] x²`.
const T2_D2: [&[(usize, usize, usize, usize)]; 5] = [
    &[],
    &[],
    &[(0, 1, 2, 3)],
    &[(0, 1, 2, 4), (2, 3, 4, 5)],
    &[(0, 1, 3, 5), (1, 2, 4, 6), (3, 4, 5, 7), (5, 6, 7, 8)],
];

/// Polynomial division of a rank n+2 bubble Laurent expansion (`mindeg` 4,
/// five orders, 22 slots) by
/// `d0 t + (d1[0] + d1[1] x) + (d2[0] + d2[1] mu² + d2[2] x + d2[3] x²)/t`,
/// given `1/d0`. Order 4 uses the level-4 index maps of
/// `tools/rank_n2/theory/formulas_rank_n2.json` (`division_T2_index_maps`).
#[inline(always)]
pub(super) fn divide_t2_5(num: &mut [C64], d0inv: C64, d1: [C64; 2], d2: [C64; 4]) {
    const NT: usize = 5;
    let num = &mut num[..T2_OFF[NT]];
    for i in 0..NT {
        let o = T2_OFF[i];
        for j in 0..T2_OFF[i + 1] - o {
            num[o + j] *= d0inv;
        }
        if i + 1 < NT {
            let n1 = T2_OFF[i + 1];
            for (e, &(z, x)) in T2_D1[i + 1].iter().enumerate() {
                let v = num[o + e];
                num[n1 + z] -= d1[0] * v;
                num[n1 + x] -= d1[1] * v;
            }
            if i + 2 < NT {
                let n2 = T2_OFF[i + 2];
                for (e, &(a, m, x, x2)) in T2_D2[i + 2].iter().enumerate() {
                    let v = num[o + e];
                    num[n2 + a] -= d2[0] * v;
                    num[n2 + m] -= d2[1] * v;
                    num[n2 + x] -= d2[2] * v;
                    num[n2 + x2] -= d2[3] * v;
                }
            }
        }
    }
}

/// `z^k` for the reciprocal-power stability check.
#[inline(always)]
fn finite_power(z: C64, k: u32) -> bool {
    let mut p = z;
    for _ in 1..k {
        p *= z;
    }
    p.is_finite()
}

/// Batched binary64 reduction for rank `n + 2` (see the module docs).
pub(super) struct RankN2Engine {
    options: FastOptions,
    /// Original Ninja's three tadpole bases and the extra candidates
    /// (`bases_x2.rs`).
    tadpole_bases: TadpoleBasesX2,
    lanes: usize,
    alive: Vec<bool>,
    lane_status: Vec<FastStatus>,
    active: Vec<usize>,
    smat: Vec<f64>,
    vsq: Vec<f64>,
    /// Per lane `(max |V_i^mu|, max |m_i²|₁)`: cheap upper bounds of the scales
    /// in the precision-aware stability checks (as in `engine.rs`).
    bounds: Vec<(f64, f64)>,
    /// `[c0, c4, c6]` per (box, lane).
    box_c: Vec<[C64; 3]>,
    tri: Vec<TriangleDataX2>,
    bub: Vec<BubbleDataX2>,
    tad: Vec<TadpoleData>,
    /// Master batch (2, 3 or 4) of each (bubble, lane).
    bub_family: Vec<u8>,
    calls: CallsX2,
    box_rows: Vec<BoxRowX2>,
    tri_rows: Vec<TriangleRowX2>,
    bub_rows: Vec<BubbleRowX2>,
    tad_rows: Vec<TadpoleRowX2>,
    second: Vec<usize>,
    uncut: Vec<Uncut>,
    /// Per bubble row and parent triangle: `[a0.et3, a0.et4, a1.et3, a1.et4]`.
    pairs: Vec<[C64; 4]>,
    masters: [MasterBatch; 6],
    staged: Vec<FastResult>,
}

impl RankN2Engine {
    pub(super) fn new(n: usize, options: FastOptions) -> Self {
        Self {
            tadpole_bases: TadpoleBasesX2::new(options.infrared_eps),
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
            bub_family: Vec::new(),
            calls: CallsX2::default(),
            box_rows: Vec::new(),
            tri_rows: Vec::new(),
            bub_rows: Vec::new(),
            tad_rows: Vec::new(),
            second: Vec::new(),
            uncut: vec![Uncut::default(); n],
            pairs: Vec::new(),
            masters: Default::default(),
            staged: Vec::new(),
        }
    }

    fn any_box(&self, topo: &Topology) -> bool {
        topo.n >= 4 && self.options.cut_stop <= 4
    }
    fn any_triangle(&self, topo: &Topology) -> bool {
        topo.n >= 3 && self.options.cut_stop <= 3
    }
    fn any_bubble(&self, topo: &Topology) -> bool {
        topo.n >= 2 && self.options.cut_stop <= 2
    }
    fn any_tadpole(&self) -> bool {
        self.options.cut_stop <= 1
    }

    fn refresh_active(&mut self) {
        let alive = &self.alive;
        self.active.retain(|&lane| alive[lane]);
    }

    /// [`super::FastAmplitude::evaluate`] at rank n+2 (same contract).
    #[inline(never)]
    pub(super) fn evaluate(
        &mut self,
        topo: &Topology,
        input: &FastInput,
        numerator: &mut dyn FastNumerator,
        masters: &mut dyn FastMasters,
        results: &mut [FastResult],
        statuses: &mut [FastStatus],
    ) -> Result<()> {
        let n = topo.n;
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
        if numerator.rank() != topo.rank {
            return Err(Error::Shape {
                name: "fast numerator rank",
                expected: topo.rank,
                actual: numerator.rank(),
            });
        }
        self.prepare(topo, input);
        if self.any_box(topo) {
            self.boxes(topo, input, numerator)?;
        }
        if self.any_triangle(topo) {
            self.triangles(topo, input, numerator)?;
        }
        if self.any_bubble(topo) {
            self.bubbles(topo, input, numerator)?;
        }
        if self.any_tadpole() {
            self.tadpoles(topo, input, numerator)?;
        }
        self.integrate(topo, input, masters)?;
        for lane in 0..lanes {
            if self.lane_status[lane] == FastStatus::Success {
                results[lane] = self.staged[lane];
            }
        }
        statuses.copy_from_slice(&self.lane_status[..lanes]);
        Ok(())
    }

    fn prepare(&mut self, topo: &Topology, input: &FastInput) {
        let n = topo.n;
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
        self.box_c.resize(topo.boxes.len() * lanes, [C64::ZERO; 3]);
        self.tri
            .resize(topo.triangles.len() * lanes, TriangleDataX2::default());
        self.bub
            .resize(topo.bubbles.len() * lanes, BubbleDataX2::default());
        self.bub_family.resize(topo.bubbles.len() * lanes, 2);
        self.tad.resize(n * lanes, TadpoleData::default());
    }

    // ------------------------------------------------------------------ boxes

    fn boxes(
        &mut self,
        topo: &Topology,
        input: &FastInput,
        num: &mut dyn FastNumerator,
    ) -> Result<()> {
        let n = topo.n;
        let lanes = self.lanes;
        let eps = self.options.infrared_eps;
        let check = Stability::new(&self.options);
        for b in 0..topo.boxes.len() {
            self.refresh_active();
            if self.active.is_empty() {
                return Ok(());
            }
            let p = topo.boxes[b];
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
                // factors relative to their own scales (as in `engine.rs`).
                if !check.absolute(mp_cc(&vort, &vort).taxicab())
                    || !check.relative(p1e3.taxicab(), dot_scale(&e.e3, &k1))
                    || !check.relative(p1e4.taxicab(), dot_scale(&e.e4, &k1))
                {
                    fail(&mut self.alive, &mut self.lane_status, lane);
                    continue;
                }
                let l = box_solutions(&k3, &k0, &k1, &e, m[c1], m[c2], m[c3], m[c4]);
                let q = [sub_cr(&l[0], &v[c1]), sub_cr(&l[1], &v[c1])];
                let mut lmid = [C64::ZERO; 4];
                let mut mid = [C64::ZERO; 4];
                for mu in 0..4 {
                    lmid[mu] = (l[0][mu] + l[1][mu]) * 0.5;
                    mid[mu] = lmid[mu] - v[c1][mu];
                }
                let beta = m[c1] - mp_cc(&lmid, &lmid);
                self.box_rows.push(BoxRowX2 {
                    lane,
                    vort,
                    q,
                    mid,
                    beta,
                });
                for qq in q {
                    self.calls.lanes.push1(lane);
                    self.calls.v0.push1(qq);
                    self.calls.scalar.push1(C64::ZERO);
                }
            }
            if self.box_rows.is_empty() {
                continue;
            }
            self.calls.prepare_out(1);
            num.evaluate(
                &p,
                self.calls.lanes.as_flattened(),
                self.calls.v0.as_flattened(),
                self.calls.scalar.as_flattened(),
                self.calls.out.as_flattened_mut(),
            )?;
            let cp = &topo.box_complements[b * (n - 4)..(b + 1) * (n - 4)];
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
                    values[s] = self.calls.out[2 * r + s][0] / d;
                }
                if !stable {
                    fail(&mut self.alive, &mut self.lane_status, lane);
                    continue;
                }
                self.box_c[b * lanes + lane] =
                    [(values[0] + values[1]) * 0.5, C64::ZERO, C64::ZERO];
                self.box_rows[keep] = row;
                keep += 1;
            }
            self.box_rows.truncate(keep);
            if self.box_rows.is_empty() {
                continue;
            }
            // Rational terms from the three-term mu-expansion on the exact
            // d-dimensional cut (mu² = t² v² - beta).
            self.calls.clear();
            for row in &self.box_rows {
                self.calls.lanes.push1(row.lane);
                self.calls.v0.push1(row.vort);
                self.calls.v1.push1(row.mid);
                self.calls.scalar.push1(row.beta);
            }
            self.calls.prepare_out(3);
            num.mu_expansion_x2(
                &p,
                self.calls.lanes.as_flattened(),
                self.calls.v0.as_flattened(),
                self.calls.v1.as_flattened(),
                self.calls.scalar.as_flattened(),
                self.calls.out.as_flattened_mut(),
            )?;
            for r in 0..self.box_rows.len() {
                let row = self.box_rows[r];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let vsq = &self.vsq[lane * n..(lane + 1) * n];
                let vort2 = mp_cc(&row.vort, &row.vort);
                let bound = 4. * self.bounds[lane].0 * taxicab_sum(&row.vort);
                let mut nmu = [
                    self.calls.out[3 * r][0],
                    self.calls.out[3 * r + 1][0],
                    self.calls.out[3 * r + 2][0],
                ];
                let mut stable = true;
                for &j in cp {
                    // D_j = d0 t + d1 exactly on this parametrization.
                    let kk = sub(&v[j], &v[c1]);
                    let d0 = mp_cr(&row.vort, &kk) * 2.;
                    if !check.relative_within(d0.taxicab(), bound, || {
                        difference_scale(&row.vort, &v[j], &v[c1])
                    }) {
                        stable = false;
                        break;
                    }
                    let d1 = C64::real(vsq[j] - vsq[c1]) - m[j] + m[c1] + mp_cr(&row.mid, &kk) * 2.;
                    nmu[0] /= d0;
                    nmu[1] -= d1 * nmu[0];
                    nmu[1] /= d0;
                    nmu[2] -= d1 * nmu[1];
                    nmu[2] /= d0;
                }
                if !stable {
                    fail(&mut self.alive, &mut self.lane_status, lane);
                    continue;
                }
                let vort4 = vort2 * vort2;
                let c6 = nmu[0] / (vort4 * vort2);
                let c4 = nmu[2] / vort4 + row.beta * c6 * 3.;
                let data = &mut self.box_c[b * lanes + lane];
                data[1] = c4;
                data[2] = c6;
            }
        }
        Ok(())
    }

    // -------------------------------------------------------------- triangles

    fn triangles(
        &mut self,
        topo: &Topology,
        input: &FastInput,
        num: &mut dyn FastNumerator,
    ) -> Result<()> {
        let n = topo.n;
        let lanes = self.lanes;
        let eps = self.options.infrared_eps;
        let check = Stability::new(&self.options);
        for t in 0..topo.triangles.len() {
            self.refresh_active();
            if self.active.is_empty() {
                return Ok(());
            }
            let p = topo.triangles[t];
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
                self.tri_rows.push(TriangleRowX2 {
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
                    self.calls.lanes.push1(row.lane);
                    self.calls.v0.push1(row.amu);
                    if orientation == 0 {
                        self.calls.v1.push1(row.e3);
                        self.calls.v2.push1(scale_c(&row.e4, h));
                    } else {
                        self.calls.v1.push1(row.e4);
                        self.calls.v2.push1(scale_c(&row.e3, h));
                    }
                    self.calls.scalar.push1(row.param);
                }
            }
            self.calls.prepare_out(T3_WIDTH);
            num.t3_expansion_x2(
                &p,
                self.calls.lanes.as_flattened(),
                self.calls.v0.as_flattened(),
                self.calls.v1.as_flattened(),
                self.calls.v2.as_flattened(),
                self.calls.scalar.as_flattened(),
                self.calls.out.as_flattened_mut(),
            )?;
            let rows = self.tri_rows.len();
            let cp = &topo.triangle_complements[t * (n - 3)..(t + 1) * (n - 3)];
            let (head, tail) = self
                .calls
                .out
                .as_flattened_mut()
                .split_at_mut(rows * T3_WIDTH);
            for r in 0..rows {
                let row = &self.tri_rows[r];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let vsq = &self.vsq[lane * n..(lane + 1) * n];
                let num1 = &mut head[r * T3_WIDTH..(r + 1) * T3_WIDTH];
                let num2 = &mut tail[r * T3_WIDTH..(r + 1) * T3_WIDTH];
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
                    let d1 = mp_cr(&row.amu, &pdiff) * 2. + (vsq[j] - vsq[c1]) - m[j] + m[c1];
                    let (a, b) = (e4p * inv34, e3p * inv34);
                    divide_t3_6(num1, d0a.inv(), d1, [row.param * a, a]);
                    divide_t3_6(num2, d0b.inv(), d1, [row.param * b, b]);
                }
                if !stable {
                    fail(&mut self.alive, &mut self.lane_status, lane);
                    continue;
                }
                // Slots (t, mu²): (5,0) (4,0) (3,0) (3,1) (2,0) (2,1) (1,0)
                // (1,1) (1,2) (0,0) (0,1) (0,2). num1 (e3 grows) carries the
                // q̄.e4 powers, num2 the q̄.e3 powers.
                let i2 = inv34 * inv34;
                let i3 = i2 * inv34;
                let i4 = i2 * i2;
                let i5 = i4 * inv34;
                let mut c = [C64::ZERO; 21];
                c[0] = (num1[9] + num2[9]) * 0.5;
                c[7] = (num1[10] + num2[10]) * 0.5;
                c[14] = (num1[11] + num2[11]) * 0.5;
                c[4] = num1[6] * inv34;
                c[1] = num2[6] * inv34;
                c[5] = num1[4] * i2;
                c[2] = num2[4] * i2;
                c[6] = num1[2] * i3;
                c[3] = num2[2] * i3;
                c[13] = num1[1] * i4;
                c[12] = num2[1] * i4;
                c[16] = num1[0] * i5;
                c[15] = num2[0] * i5;
                c[9] = num1[7] * inv34;
                c[8] = num2[7] * inv34;
                c[11] = num1[5] * i2;
                c[10] = num2[5] * i2;
                c[18] = num1[3] * i3;
                c[17] = num2[3] * i3;
                c[20] = num1[8] * inv34;
                c[19] = num2[8] * inv34;
                self.tri[t * lanes + lane] = TriangleDataX2 {
                    e3: row.e3,
                    e4: row.e4,
                    c,
                };
            }
        }
        Ok(())
    }

    // ---------------------------------------------------------------- bubbles

    fn bubbles(
        &mut self,
        topo: &Topology,
        input: &FastInput,
        num: &mut dyn FastNumerator,
    ) -> Result<()> {
        let n = topo.n;
        let lanes = self.lanes;
        let eps = self.options.infrared_eps;
        let check = Stability::new(&self.options);
        // Triangle parents exist only when the triangle stage ran.
        let subtract = self.any_triangle(topo);
        for b in 0..topo.bubbles.len() {
            self.refresh_active();
            if self.active.is_empty() {
                return Ok(());
            }
            let p = topo.bubbles[b];
            let [c1, c2] = p;
            let cp = &topo.bubble_complements[b * (n - 2)..(b + 1) * (n - 2)];
            self.bub_rows.clear();
            self.calls.clear();
            for ai in 0..self.active.len() {
                let lane = self.active[ai];
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let s = self.smat[lane * n * n + c2 * n + c1];
                self.bub[b * lanes + lane] = BubbleDataX2::default();
                if is_zero(m[c1]) && is_zero(m[c2]) && s.abs() < eps {
                    continue;
                }
                let k = sub(&v[c1], &v[c2]);
                // Original Ninja's reference unless it is ill conditioned
                // against the uncut lines (bases_x2.rs).
                let e = bubble_basis_x2(&k, v, c1, cp, eps);
                let (b0, b1, param) = bubble_solution(&k, &e, m[c1], m[c2]);
                let e1effl = [b0 * e.e2[0], b0 * e.e2[1], b0 * e.e2[2], b0 * e.e2[3]];
                let e2eff = [
                    e.e1[0] + b1 * e.e2[0],
                    e.e1[1] + b1 * e.e2[1],
                    e.e1[2] + b1 * e.e2[2],
                    e.e1[3] + b1 * e.e2[3],
                ];
                let e1eff = sub_cr(&e1effl, &v[c1]);
                self.calls.lanes.push1(lane);
                self.calls.v0.push1(e1eff);
                self.calls.v1.push1(cmom_from_real(&e2eff));
                self.calls.v2.push1(e.e3);
                self.calls.v3.push1(scale_c(&e.e4, 0.5 / e.mp34()));
                self.calls.params.push1(param);
                self.bub_rows.push(BubbleRowX2 {
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
            self.calls.prepare_out(T2_WIDTH);
            num.t2_expansion_x2(
                &p,
                self.calls.lanes.as_flattened(),
                self.calls.v0.as_flattened(),
                self.calls.v1.as_flattened(),
                self.calls.v2.as_flattened(),
                self.calls.v3.as_flattened(),
                self.calls.params.as_flattened(),
                self.calls.out.as_flattened_mut(),
            )?;
            let parents = &topo.bubble_parents[b];
            self.second.clear();
            for r in 0..self.bub_rows.len() {
                let row = &self.bub_rows[r];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let vsq = &self.vsq[lane * n..(lane + 1) * n];
                let numexp =
                    &mut self.calls.out.as_flattened_mut()[r * T2_WIDTH..(r + 1) * T2_WIDTH];
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
                    let ie3k = e3k.inv();
                    // The new subtraction terms divide by up to (e3.k)^5.
                    if !finite_power(ie3k, 5) {
                        stable = false;
                        break;
                    }
                    let e2p = mp_rr(&e.e2, &pdiff);
                    let e1k = row.b0 * e2p;
                    uncut[j] = Uncut {
                        e3k,
                        ie3k,
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
                    divide_t2_5(
                        numexp,
                        u.ie3k * 0.5,
                        [u.d1, C64::real(2. * u.e2k)],
                        [row.param[0] * d2, d2, row.param[1] * d2, row.param[2] * d2],
                    );
                }
                if subtract {
                    let e4eff = scale_c(&e.e4, e4s);
                    let np = parents.len();
                    if self.pairs.len() < (r + 1) * np {
                        self.pairs.resize((r + 1) * np, [C64::ZERO; 4]);
                    }
                    for (pi, &(t, unc)) in parents.iter().enumerate() {
                        let tri = &self.tri[t * lanes + lane];
                        let u = &uncut[unc];
                        let (e2t3, e2t4) = (mp_cr(&tri.e3, &e.e2), mp_cr(&tri.e4, &e.e2));
                        let pair = [
                            row.b0 * e2t3,
                            row.b0 * e2t4,
                            mp_cr(&tri.e3, &e.e1) + e2t3 * row.b1,
                            mp_cr(&tri.e4, &e.e1) + e2t4 * row.b1,
                        ];
                        self.pairs[r * np + pi] = pair;
                        subs::bubble_from_triangle_all(
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
                // Orientation A slots (t, x, mu²), design 1.3; every
                // coefficient is slot / (mp12^x mp34^t).
                let (k1, k2, k3, k4) = (
                    inv12,
                    inv12 * inv12,
                    inv12 * inv12 * inv12,
                    inv12 * inv12 * (inv12 * inv12),
                );
                let (i1, i2, i3, i4) = (
                    inv34,
                    inv34 * inv34,
                    inv34 * inv34 * inv34,
                    inv34 * inv34 * (inv34 * inv34),
                );
                c[0] = numexp[13];
                c[1] = numexp[16] * k1;
                c[2] = numexp[18] * k2;
                c[13] = numexp[20] * k3;
                c[20] = numexp[21] * k4;
                c[9] = numexp[14];
                c[10] = numexp[17] * k1;
                c[21] = numexp[19] * k2;
                c[22] = numexp[15];
                c[5] = numexp[7] * i1;
                c[6] = numexp[3] * i2;
                c[15] = numexp[1] * i3;
                c[24] = numexp[0] * i4;
                c[8] = numexp[9] * (k1 * i1);
                c[19] = numexp[5] * (k1 * i2);
                c[17] = numexp[11] * (k2 * i1);
                c[26] = numexp[2] * (k1 * i3);
                c[28] = numexp[6] * (k2 * i2);
                c[30] = numexp[12] * (k3 * i1);
                c[12] = numexp[8] * i1;
                c[32] = numexp[4] * i2;
                c[34] = numexp[10] * (k1 * i1);
                // Spurious coefficients are needed only as subtraction terms
                // of massive tadpoles (Ninja's spurious_terms_needed).
                if !is_zero(m[c1]) || !is_zero(m[c2]) {
                    self.second.push(r);
                }
            }
            if self.second.is_empty() {
                continue;
            }
            // Second orientation (e3 <-> e4), mindeg 3: the 13 slots with t >= 1.
            self.calls.clear();
            for &r in &self.second {
                let row = &self.bub_rows[r];
                self.calls.lanes.push1(row.lane);
                self.calls.v0.push1(row.e1eff);
                self.calls.v1.push1(cmom_from_real(&row.e2eff));
                self.calls.v2.push1(row.e.e4);
                self.calls.v3.push1(scale_c(&row.e.e3, 0.5 / row.e.mp34()));
                self.calls.params.push1(row.param);
            }
            self.calls.prepare_out(T2_WIDTH_B);
            num.t2_expansion(
                &p,
                self.calls.lanes.as_flattened(),
                self.calls.v0.as_flattened(),
                self.calls.v1.as_flattened(),
                self.calls.v2.as_flattened(),
                self.calls.v3.as_flattened(),
                self.calls.params.as_flattened(),
                3,
                self.calls.out.as_flattened_mut(),
            )?;
            for s in 0..self.second.len() {
                let row = &self.bub_rows[self.second[s]];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let vsq = &self.vsq[lane * n..(lane + 1) * n];
                let numexp =
                    &mut self.calls.out.as_flattened_mut()[s * T2_WIDTH_B..(s + 1) * T2_WIDTH_B];
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
                    let ie4k = e4k.inv();
                    if !finite_power(ie4k, 5) {
                        stable = false;
                        break;
                    }
                    let e2p = mp_rr(&e.e2, &pdiff);
                    let e1k = row.b0 * e2p;
                    uncut[j] = Uncut {
                        e3k: e4k,
                        ie3k: ie4k,
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
                        4,
                        u.ie3k * 0.5,
                        [u.d1, C64::real(2. * u.e2k)],
                        [row.param[0] * d2, d2, row.param[1] * d2, row.param[2] * d2],
                    );
                }
                if subtract {
                    let e3eff = scale_c(&e.e3, e3s);
                    let np = parents.len();
                    let r = self.second[s];
                    for (pi, &(t, unc)) in parents.iter().enumerate() {
                        let tri = &self.tri[t * lanes + lane];
                        let u = &uncut[unc];
                        subs::bubble_from_triangle_spurious(
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
                let (k1, k2, k3) = (inv12, inv12 * inv12, inv12 * inv12 * inv12);
                let (i1, i2, i3, i4) = (
                    inv34,
                    inv34 * inv34,
                    inv34 * inv34 * inv34,
                    inv34 * inv34 * (inv34 * inv34),
                );
                c[3] = numexp[7] * i1;
                c[4] = numexp[3] * i2;
                c[14] = numexp[1] * i3;
                c[23] = numexp[0] * i4;
                c[7] = numexp[9] * (k1 * i1);
                c[18] = numexp[5] * (k1 * i2);
                c[16] = numexp[11] * (k2 * i1);
                c[25] = numexp[2] * (k1 * i3);
                c[27] = numexp[6] * (k2 * i2);
                c[29] = numexp[12] * (k3 * i1);
                c[11] = numexp[8] * i1;
                c[31] = numexp[4] * i2;
                c[33] = numexp[10] * (k1 * i1);
            }
        }
        Ok(())
    }

    // --------------------------------------------------------------- tadpoles

    fn tadpoles(
        &mut self,
        topo: &Topology,
        input: &FastInput,
        num: &mut dyn FastNumerator,
    ) -> Result<()> {
        let n = topo.n;
        let lanes = self.lanes;
        let eps = self.options.infrared_eps;
        let check = Stability::new(&self.options);
        let any_triangle = self.any_triangle(topo);
        let any_bubble = self.any_bubble(topo);
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
                let ninja = (0..2)
                    .find(|&i| {
                        let e = &self.tadpole_bases.bases[i];
                        reference_vectors_ok(v, a, &e.e1, &e.e2, eps)
                    })
                    .unwrap_or(2);
                // Original Ninja's basis unless it is ill conditioned
                // against the uncut lines (bases_x2.rs).
                let uncut = &topo.tadpole_complements[a * (n - 1)..(a + 1) * (n - 1)];
                let basis = self.tadpole_bases.choose(ninja, v, a, uncut);
                self.tad_rows.push(TadpoleRowX2 { lane, basis });
            }
            if self.tad_rows.is_empty() {
                continue;
            }
            // Orientation 0: (e1, e2/(2 mp12)); orientation 1: (e3, e4/(2 mp34)).
            let vectors = |e: &Basis, orientation: usize| -> (CMom, CMom) {
                if orientation == 0 {
                    (
                        cmom_from_real(&e.e1),
                        cmom_from_real(&scale(&e.e2, 0.5 / e.mp12)),
                    )
                } else {
                    (e.e3, scale_c(&e.e4, 0.5 / e.mp34()))
                }
            };
            self.calls.clear();
            for orientation in 0..2 {
                for row in &self.tad_rows {
                    let lane = row.lane;
                    let (ev3, ev4) = vectors(&self.tadpole_bases.bases[row.basis], orientation);
                    let v0 = input.shifts[lane * n + a];
                    self.calls.lanes.push1(lane);
                    self.calls
                        .v0
                        .push1(cmom_from_real(&[-v0[0], -v0[1], -v0[2], -v0[3]]));
                    self.calls.v1.push1(ev3);
                    self.calls.v2.push1(ev4);
                    self.calls.scalar.push1(input.masses_squared[lane * n + a]);
                }
            }
            self.calls.prepare_out(TAD_WIDTH);
            let cut = [a];
            num.t3_expansion(
                &cut,
                self.calls.lanes.as_flattened(),
                self.calls.v0.as_flattened(),
                self.calls.v1.as_flattened(),
                self.calls.v2.as_flattened(),
                self.calls.scalar.as_flattened(),
                3,
                self.calls.out.as_flattened_mut(),
            )?;
            let rows = self.tad_rows.len();
            let cp = &topo.tadpole_complements[a * (n - 1)..(a + 1) * (n - 1)];
            let tri_parents = &topo.tadpole_triangle_parents[a];
            let bub_parents = &topo.tadpole_bubble_parents[a];
            for r in 0..rows {
                let row = self.tad_rows[r];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let vsq = &self.vsq[lane * n..(lane + 1) * n];
                let e = self.tadpole_bases.bases[row.basis];
                let param = m[a];
                let amu = [-v[a][0], -v[a][1], -v[a][2], -v[a][3]];
                let mut stable = true;
                let mut data = TadpoleData {
                    e34: -mp_rr(&e.e1, &e.e2),
                    ..TadpoleData::default()
                };
                for orientation in 0..2 {
                    let offset = (orientation * rows + r) * TAD_WIDTH;
                    let numexp = &mut self.calls.out.as_flattened_mut()[offset..offset + TAD_WIDTH];
                    let (ev3, ev4) = vectors(&e, orientation);
                    let uncut = &mut self.uncut;
                    for &j in cp {
                        let pdiff = sub(&v[j], &v[a]);
                        let e3k = mp_cr(&ev3, &pdiff);
                        if !check.absolute(e3k.taxicab() * 2.) {
                            stable = false;
                            break;
                        }
                        let ie3k = e3k.inv();
                        // The new subtraction terms divide by up to (e.k)^4.
                        if !finite_power(ie3k, 4) {
                            stable = false;
                            break;
                        }
                        uncut[j] = Uncut {
                            e3k,
                            ie3k,
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
                        divide_t3(numexp, 4, u.ie3k * 0.5, u.d1, [param * d2, d2]);
                    }
                    if any_triangle {
                        for &(t, [u0, u1]) in tri_parents {
                            let tri = &self.tri[t * lanes + lane];
                            let (k0, k1) = (&uncut[u0], &uncut[u1]);
                            subs::tadpole_from_triangle(
                                numexp, &tri.e3, &tri.e4, &tri.c, &ev3, &ev4, param, k0.e3k,
                                k0.ie3k, k0.e4k, k0.f, k1.e3k, k1.ie3k, k1.e4k, k1.f,
                            );
                        }
                    }
                    if any_bubble {
                        for &(bb, unc) in bub_parents {
                            let bub = &self.bub[bb * lanes + lane];
                            let kbol = sub(&v[topo.bubbles[bb][0]], &v[a]);
                            let u = &uncut[unc];
                            subs::tadpole_from_bubble(
                                numexp, &bub.e2, &bub.e3, &bub.e4, &kbol, &bub.c, &ev3, &ev4,
                                param, u.e3k, u.ie3k, u.e4k, u.f,
                            );
                        }
                    }
                    if orientation == 0 {
                        data.c0 = numexp[4];
                        data.c14 = numexp[5];
                    } else {
                        data.c15 = (numexp[5] - data.c14) * (2. / e.mp34());
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

    /// Master batch of a bubble: rank 4 if `c20 ke2⁴` is not exactly zero,
    /// else rank 3 if `c13 ke2³` is not (Ninja's rank n+1 rule), else rank 2.
    #[inline]
    fn bubble_family(&self, topo: &Topology, b: usize, lane: usize, v: &[RMom]) -> u8 {
        let [i1, i2] = topo.bubbles[b];
        let data = &self.bub[b * self.lanes + lane];
        let ke2 = mp_rr(&sub(&v[i2], &v[i1]), &data.e2);
        if !is_zero(data.c[20] * ke2 * ke2 * ke2 * ke2) {
            4
        } else if !is_zero(data.c[13] * ke2 * ke2 * ke2) {
            3
        } else {
            2
        }
    }

    fn integrate(
        &mut self,
        topo: &Topology,
        input: &FastInput,
        masters: &mut dyn FastMasters,
    ) -> Result<()> {
        let n = topo.n;
        let lanes = self.lanes;
        let eps = self.options.infrared_eps;
        let (any_box, any_tri, any_bub, any_tad) = (
            self.any_box(topo),
            self.any_triangle(topo),
            self.any_bubble(topo),
            self.any_tadpole(),
        );
        self.refresh_active();
        for batch in &mut self.masters {
            batch.args.clear();
        }
        // Collect requests lane-major in Ninja's contraction order.
        for ai in 0..self.active.len() {
            let lane = self.active[ai];
            let m = &input.masses_squared[lane * n..(lane + 1) * n];
            let s = &self.smat[lane * n * n..(lane + 1) * n * n];
            let mu2 = C64::real(input.scales_squared[lane]);
            let sm = |i: usize, j: usize| C64::real(s[i * n + j]);
            if any_box {
                let args = &mut self.masters[0].args;
                for &[i1, i2, i3, i4] in &topo.boxes {
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
                for &[i1, i2, i3] in &topo.triangles {
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
                for (b, &[i1, i2]) in topo.bubbles.iter().enumerate() {
                    let k = s[i2 * n + i1];
                    if is_zero(m[i1]) && is_zero(m[i2]) && k.abs() < eps {
                        continue;
                    }
                    let family = self.bubble_family(topo, b, lane, v);
                    self.bub_family[b * lanes + lane] = family;
                    self.masters[family as usize].args.extend_from_slice(&[
                        C64::real(k),
                        m[i1],
                        m[i2],
                        mu2,
                    ]);
                }
            }
            if any_tad {
                for i in 0..n {
                    if !is_zero(m[i]) {
                        self.masters[A0_BATCH].args.extend_from_slice(&[m[i], mu2]);
                    }
                }
            }
        }
        for (batch, family) in self.masters.iter_mut().zip(FAMILIES_X2) {
            batch.run(family, masters)?;
        }
        // Contract in original Ninja's order.
        let mut cursor = [0usize; 6];
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
                for (b, &[i1, i2, i3, i4]) in topo.boxes.iter().enumerate() {
                    let row = cursor[0];
                    cursor[0] += 1;
                    ok &= batch.ok[row];
                    let integral = &batch.out[row];
                    let [c0, c4, c6] = self.box_c[b * lanes + lane];
                    for k in 0..3 {
                        res[k] += c0 * integral[k];
                    }
                    // I4[mu⁴] = -1/6, I4[mu⁶] = sum m²/12 - sum s/60.
                    let ssum = s[i1 * n + i2]
                        + s[i1 * n + i3]
                        + s[i1 * n + i4]
                        + s[i2 * n + i3]
                        + s[i2 * n + i4]
                        + s[i3 * n + i4];
                    let msum = m[i1] + m[i2] + m[i3] + m[i4];
                    rational += -c4 / 6. + c6 * (msum / 12. - C64::real(ssum / 60.));
                }
            }
            if any_tri {
                let batch = &self.masters[1];
                for (t, &[i1, i2, i3]) in topo.triangles.iter().enumerate() {
                    let row = cursor[1];
                    cursor[1] += 1;
                    ok &= batch.ok[row];
                    let integral = &batch.out[row];
                    let c = &self.tri[t * lanes + lane].c;
                    for k in 0..3 {
                        res[k] += c[0] * integral[k];
                    }
                    let (pji, pjk, pki) = (s[i2 * n + i1], s[i3 * n + i2], s[i1 * n + i3]);
                    rational += c[7] * 0.5
                        + c[14] * (C64::real(0.25 * (pjk + pji + pki)) - m[i1] - m[i2] - m[i3])
                            / 6.;
                }
            }
            if any_bub {
                for (b, &[i1, i2]) in topo.bubbles.iter().enumerate() {
                    let k = s[i2 * n + i1];
                    if is_zero(m[i1]) && is_zero(m[i2]) && k.abs() < eps {
                        continue;
                    }
                    let data = &self.bub[b * lanes + lane];
                    let c = &data.c;
                    let ke2 = mp_rr(&sub(&v[i2], &v[i1]), &data.e2);
                    let family = self.bub_family[b * lanes + lane] as usize;
                    let width = family + 1;
                    let row = cursor[family];
                    cursor[family] += 1;
                    let batch = &self.masters[family];
                    ok &= batch.ok[row];
                    let out = &batch.out[row * width..(row + 1) * width];
                    let ke22 = ke2 * ke2;
                    let c1 = c[1] * ke2;
                    let c2 = c[2] * ke2 * ke2;
                    for j in 0..3 {
                        res[j] += c[0] * out[0][j] + c1 * out[1][j] + c2 * out[2][j];
                    }
                    if family >= 3 {
                        let c13 = c[13] * ke2 * ke2 * ke2;
                        for j in 0..3 {
                            res[j] += c13 * out[3][j];
                        }
                    }
                    if family == 4 {
                        let c20 = c[20] * ke2 * ke2 * ke2 * ke2;
                        for j in 0..3 {
                            res[j] += c20 * out[4][j];
                        }
                    }
                    let (m1, m2, sk) = (m[i1], m[i2], C64::real(k));
                    let b06 = -(sk - (m1 + m2) * 3.) / 6.;
                    rational += b06 * c[9] + (sk - m1 * 2. - m2 * 4.) * ke2 * c[10] / 12.;
                    // I2[mu² (q.e2)²] and I2[mu⁴] (design 2.7).
                    rational += c[21] * ke22 * (m1 * 5. + m2 * 15. - sk * 3.) / 60.;
                    rational -= c[22]
                        * (sk * sk - sk * (m1 + m2) * 5. + (m1 * m1 + m1 * m2 + m2 * m2) * 10.)
                        / 60.;
                }
            }
            if any_tad {
                let batch = &self.masters[A0_BATCH];
                for i in 0..n {
                    if is_zero(m[i]) {
                        continue;
                    }
                    let row = cursor[A0_BATCH];
                    cursor[A0_BATCH] += 1;
                    ok &= batch.ok[row];
                    let integral = &batch.out[row];
                    let data = &self.tad[i * lanes + lane];
                    let c0 = data.c0 + m[i] * (data.e34 * 0.25) * data.c15;
                    for k in 0..3 {
                        res[k] += c0 * integral[k];
                    }
                    rational += (data.c14 + data.c15 * (data.e34 * 0.25)) * m[i] * m[i] * 0.5;
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
}

/// The rank n+2 branch of [`super::FastAmplitude::evaluate`]: a cold,
/// never-inlined entry, so the branch weights (and with them the inlining
/// decisions) of the rank <= n+1 evaluation stay those of an engine without
/// this branch.
#[cold]
#[inline(never)]
pub(super) fn evaluate_x2(
    engine: &mut RankN2Engine,
    topo: &Topology,
    input: &FastInput,
    numerator: &mut dyn FastNumerator,
    masters: &mut dyn FastMasters,
    results: &mut [FastResult],
    statuses: &mut [FastStatus],
) -> Result<()> {
    engine.evaluate(topo, input, numerator, masters, results, statuses)
}

/// A sized view of any (possibly unsized) [`FastNumerator`], so the rank n+2
/// engine is compiled once, in this crate, against `dyn FastNumerator`
/// instead of once per numerator type in every caller (which would also
/// reshuffle the callers' codegen units).
pub(super) struct DynNumerator<'a, N: ?Sized>(pub(super) &'a mut N);

impl<N: FastNumerator + ?Sized> FastNumerator for DynNumerator<'_, N> {
    fn rank(&self) -> usize {
        self.0.rank()
    }
    fn evaluate(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        q: &[CMom],
        mu_squared: &[C64],
        out: &mut [C64],
    ) -> Result<()> {
        self.0.evaluate(cut, lanes, q, mu_squared, out)
    }
    fn mu_expansion(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        v0: &[CMom],
        v1: Option<&[CMom]>,
        out: &mut [C64],
    ) -> Result<()> {
        self.0.mu_expansion(cut, lanes, v0, v1, out)
    }
    fn mu_expansion_x2(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        v0: &[CMom],
        v1: &[CMom],
        beta: &[C64],
        out: &mut [C64],
    ) -> Result<()> {
        self.0.mu_expansion_x2(cut, lanes, v0, v1, beta, out)
    }
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
    ) -> Result<()> {
        self.0
            .t3_expansion(cut, lanes, a, e3, e4, param, mindeg, out)
    }
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
    ) -> Result<()> {
        self.0
            .t2_expansion(cut, lanes, a0, a1, e3, e4, param, mindeg, out)
    }
    fn t3_expansion_x2(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        a: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[C64],
        out: &mut [C64],
    ) -> Result<()> {
        self.0.t3_expansion_x2(cut, lanes, a, e3, e4, param, out)
    }
    fn t2_expansion_x2(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        a0: &[CMom],
        a1: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[[C64; 3]],
        out: &mut [C64],
    ) -> Result<()> {
        self.0
            .t2_expansion_x2(cut, lanes, a0, a1, e3, e4, param, out)
    }
}

/// A sized view of any [`FastMasters`] (see [`DynNumerator`]).
pub(super) struct DynMasters<'a, M: ?Sized>(pub(super) &'a mut M);

impl<M: FastMasters + ?Sized> FastMasters for DynMasters<'_, M> {
    fn evaluate(
        &mut self,
        family: FastMaster,
        args: &[C64],
        out: &mut [[C64; 3]],
        ok: &mut [bool],
    ) -> Result<()> {
        self.0.evaluate(family, args, out, ok)
    }
}

/// Per-cut results for the integrand-level unit tests (`tests_x2.rs`).
#[cfg(test)]
impl RankN2Engine {
    /// `[c0, c4, c6]` of box `b`.
    pub(super) fn box_coefficients(&self, b: usize, lane: usize) -> [C64; 3] {
        self.box_c[b * self.lanes + lane]
    }
    /// `(e3, e4, c)` of triangle `t`.
    pub(super) fn triangle_residue(&self, t: usize, lane: usize) -> (CMom, CMom, [C64; 21]) {
        let data = &self.tri[t * self.lanes + lane];
        (data.e3, data.e4, data.c)
    }
    /// `(e2, e3, e4, c)` of bubble `b`.
    pub(super) fn bubble_residue(&self, b: usize, lane: usize) -> (RMom, CMom, CMom, [C64; 35]) {
        let data = &self.bub[b * self.lanes + lane];
        (data.e2, data.e3, data.e4, data.c)
    }
}
