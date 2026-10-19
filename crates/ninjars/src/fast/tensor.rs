//! Ninja's tensor numerator (`tensor_ninja.cc`) for the fast engine.
//!
//! A rank-`R` numerator is `N(q) = sum_k c_k monomial_k(q)` in the
//! [`super::tensor_layout`] ordering (identical to Ninja's `TensorNumerator`
//! and MadLoop). Like the original it ignores `mu²`: only the four-dimensional
//! loop momentum enters.
//!
//! Instead of materializing symmetric tensor powers of the cut vectors (as the
//! original does), every expansion is a single transposed Horner sweep over
//! the coefficient blocks with truncated jets. For `q = t (e3 + u a + u² s e4)`
//! with `u = 1/t` the homogenized numerator
//! `t^-R N(q) = sum_d u^(R-d) N_d(e3 + u a + u² s e4)` is evaluated as a
//! polynomial in `u` truncated at `u^mindeg`, with exact polynomial dependence
//! on `s = param + mu²` (and on `x` for double cuts). Each coefficient of the
//! tensor is touched once per expansion; only orders requested by `mindeg`
//! are formed.
//!
//! Rank n+2 (triangle `mindeg` 5, bubble `mindeg` 4 and the three-term box
//! expansion) uses the separate, never-inlined `t3_expansion_x2`,
//! `t2_expansion_x2` and `mu_expansion_x2` with the `JetX2` jets, so the
//! rank <= n+1 expansions are compiled exactly as before (`t3_expansion` and
//! `t2_expansion` keep Ninja's depth limits).

use super::complex::{C64, CMom};
use super::numerator::{FastNumerator, t2_len, t2_len_le3, t3_len};
use super::tensor_jets::{self as jets, Jet, JetX2};
use super::tensor_layout::{block_offset, ncoeff, ncoeff_homogeneous};
use crate::{Error, Result};

/// Batched Ninja/MadLoop tensor numerator with per-lane coefficients.
#[derive(Clone, Debug)]
pub struct TensorNumerator {
    rank: usize,
    ncoeff: usize,
    lanes: usize,
    coefficients: Vec<C64>,
    work: Vec<C64>,
}

impl TensorNumerator {
    /// A rank-`rank` tensor numerator for `lanes` phase-space lanes with zero
    /// coefficients.
    pub fn new(rank: usize, lanes: usize) -> Self {
        let ncoeff = ncoeff(rank);
        // Two jet levels of at most thirteen components each.
        let level = if rank == 0 {
            1
        } else {
            ncoeff_homogeneous(rank - 1)
        };
        Self {
            rank,
            ncoeff,
            lanes,
            coefficients: vec![C64::ZERO; lanes * ncoeff],
            work: vec![C64::ZERO; 2 * 13 * level],
        }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Number of coefficients per lane, `binomial(rank + 4, 4)`.
    pub fn ncoeff(&self) -> usize {
        self.ncoeff
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }

    /// Change the number of lanes; coefficients of retained lanes are kept.
    pub fn set_lanes(&mut self, lanes: usize) {
        self.lanes = lanes;
        self.coefficients.resize(lanes * self.ncoeff, C64::ZERO);
    }

    /// Lane-major coefficients (`lanes * ncoeff`).
    pub fn coefficients(&self) -> &[C64] {
        &self.coefficients
    }

    pub fn coefficients_mut(&mut self) -> &mut [C64] {
        &mut self.coefficients
    }

    pub fn lane_coefficients(&self, lane: usize) -> &[C64] {
        &self.coefficients[lane * self.ncoeff..(lane + 1) * self.ncoeff]
    }

    pub fn lane_coefficients_mut(&mut self, lane: usize) -> &mut [C64] {
        &mut self.coefficients[lane * self.ncoeff..(lane + 1) * self.ncoeff]
    }

    /// Replace all coefficients with a lane-major `lanes * ncoeff` buffer.
    pub fn set_coefficients(&mut self, lanes: usize, coefficients: &[C64]) -> Result<()> {
        if coefficients.len() != lanes * self.ncoeff {
            return Err(Error::Shape {
                name: "tensor coefficients",
                expected: lanes * self.ncoeff,
                actual: coefficients.len(),
            });
        }
        self.lanes = lanes;
        self.coefficients.clear();
        self.coefficients.extend_from_slice(coefficients);
        Ok(())
    }

    fn lane(&self, lane: usize) -> Result<&[C64]> {
        if lane >= self.lanes {
            return Err(Error::Lane {
                lane,
                lanes: self.lanes,
            });
        }
        Ok(&self.coefficients[lane * self.ncoeff..(lane + 1) * self.ncoeff])
    }

    fn check_rows(rows: usize, out: usize, width: usize) -> Result<()> {
        if out != rows * width {
            return Err(Error::Shape {
                name: "tensor numerator output",
                expected: rows * width,
                actual: out,
            });
        }
        Ok(())
    }

    /// Plain evaluation `N(q)` (scalar Horner over all blocks).
    pub fn evaluate_lane(&mut self, lane: usize, q: &CMom) -> Result<C64> {
        self.lane(lane)?;
        let c = &self.coefficients[lane * self.ncoeff..(lane + 1) * self.ncoeff];
        Ok(horner_scalar(c, self.rank, q, true, &mut self.work))
    }
}

/// Ten-build split sizes of the homogeneous rank-`d` block.
#[inline(always)]
fn splits(d: usize) -> (usize, usize, usize) {
    (d, d * (d + 1) / 2, ncoeff_homogeneous(d - 1))
}

/// Scalar Horner evaluation of `N(q)` (all blocks) or of `N_R(q)` (top block).
#[inline]
pub(super) fn horner_scalar(
    c: &[C64],
    rank: usize,
    q: &CMom,
    all_blocks: bool,
    work: &mut [C64],
) -> C64 {
    if rank == 0 {
        return c[0];
    }
    let level = ncoeff_homogeneous(rank - 1);
    let (buf_a, buf_b) = work.split_at_mut(level);
    let (mut cur, mut nxt) = (buf_a, &mut buf_b[..level]);
    let mut d = rank;
    loop {
        let (r1, r2, r3) = splits(d);
        let next = &mut nxt[..r3];
        let g: &[C64] = if d == rank {
            let off = block_offset(d);
            &c[off..off + ncoeff_homogeneous(d)]
        } else {
            &cur[..ncoeff_homogeneous(d)]
        };
        for i in 0..r3 {
            next[i] = g[1 + r1 + r2 + i] * q[3];
        }
        for i in 0..r2 {
            next[i] += g[1 + r1 + i] * q[2];
        }
        for i in 0..r1 {
            next[i] += g[1 + i] * q[1];
        }
        next[0] += g[0] * q[0];
        if all_blocks {
            let low = &c[block_offset(d - 1)..block_offset(d - 1) + r3];
            for i in 0..r3 {
                next[i] += low[i];
            }
        }
        std::mem::swap(&mut cur, &mut nxt);
        if d == 1 {
            return cur[0];
        }
        d -= 1;
    }
}

/// Component index of `u^i s^j` in a T3 jet (Ninja's t3 layout).
pub(super) const fn t3_len_const(i: usize) -> usize {
    // t3_len(i - 1) with t3_len(-1) = 0
    let mut total = 0;
    let mut k = 0;
    while k < i {
        total += 1 + k / 2;
        k += 1;
    }
    total
}

/// Offset of order `u^i` in a T2 jet.
pub(super) const fn t2_offset(i: usize) -> usize {
    let mut total = 0;
    let mut k = 0;
    while k < i {
        total += 1 + k + k / 2 + k / 3;
        k += 1;
    }
    total
}

/// Truncated-jet transposed Horner sweep; returns the jet of
/// `sum_d u^(R-d) N_d(x(u))` with `x_mu(u) = sum_s slot_s(u) xv[s][mu]`.
///
/// Level `d` maps the homogeneous block of degree `d` (ten_build order:
/// `[x0 M0] ++ [x1 M(<d)] ++ [x2 M(<d(d+1)/2)] ++ [x3 M(all)]`) onto the
/// block of degree `d-1`: output entry `i` collects at most four products,
/// so every output jet is written once per level. The lower block `d-1`
/// enters at the pure power `u^(R-d+1)` (dropped beyond truncation).
#[inline(always)]
pub(super) fn horner_jet<P: Jet<L>, const L: usize>(
    c: &[C64],
    rank: usize,
    xv: &[CMom; 4],
    work: &mut [C64],
) -> [C64; L] {
    let mut result = [C64::ZERO; L];
    if rank == 0 {
        result[0] = c[0];
        return result;
    }
    let slot = |mu: usize| -> [C64; 4] { [xv[0][mu], xv[1][mu], xv[2][mu], xv[3][mu]] };
    let xs = [slot(0), slot(1), slot(2), slot(3)];
    let level = ncoeff_homogeneous(rank - 1);
    let (buf_a, buf_b) = work.split_at_mut(level * L);
    let (mut cur, _) = buf_a.as_chunks_mut::<L>();
    let (mut nxt, _) = buf_b[..level * L].as_chunks_mut::<L>();
    // Level R: scalar coefficients times the cut-vector slots.
    {
        let d = rank;
        let off = block_offset(d);
        let (r1, r2, r3) = splits(d);
        let g = &c[off..off + ncoeff_homogeneous(d)];
        let (g1, g2, g3) = (&g[1..1 + r1], &g[1 + r1..1 + r1 + r2], &g[1 + r1 + r2..]);
        let out = &mut cur[..r3];
        for i in 0..r3 {
            P::first_set(&mut out[i], g3[i], &xs[3]);
            if i < r2 {
                P::first_acc(&mut out[i], g2[i], &xs[2]);
                if i < r1 {
                    P::first_acc(&mut out[i], g1[i], &xs[1]);
                }
            }
        }
        P::first_acc(&mut out[0], g[0], &xs[0]);
        if let Some(pure) = P::PURE[1] {
            let low = &c[block_offset(d - 1)..block_offset(d - 1) + r3];
            for i in 0..r3 {
                out[i][pure] += low[i];
            }
        }
    }
    let mut cur_len = level;
    let mut d = rank - 1;
    while d >= 1 {
        let (r1, r2, r3) = splits(d);
        let g = &cur[..cur_len];
        let (g1, g2, g3) = (&g[1..1 + r1], &g[1 + r1..1 + r1 + r2], &g[1 + r1 + r2..]);
        let out = &mut nxt[..r3];
        for i in 0..r3 {
            P::mul_set(&mut out[i], &g3[i], &xs[3]);
            if i < r2 {
                P::mul_acc(&mut out[i], &g2[i], &xs[2]);
                if i < r1 {
                    P::mul_acc(&mut out[i], &g1[i], &xs[1]);
                }
            }
        }
        P::mul_acc(&mut out[0], &g[0], &xs[0]);
        let power = rank - d + 1;
        if let Some(pure) = P::PURE.get(power).copied().flatten() {
            let low = &c[block_offset(d - 1)..block_offset(d - 1) + r3];
            for i in 0..r3 {
                out[i][pure] += low[i];
            }
        }
        cur_len = r3;
        std::mem::swap(&mut cur, &mut nxt);
        d -= 1;
    }
    result.copy_from_slice(&cur[0]);
    result
}

/// [`horner_jet`] for the rank n+2 jets: identical sweep, with the six-entry
/// pure-power table of [`JetX2`] (a triangle at `mindeg` 5 keeps `u^5`).
/// `work` must hold `2 L ncoeff_homogeneous(rank - 1)` entries.
#[inline(always)]
pub(super) fn horner_jet_x2<P: JetX2<L>, const L: usize>(
    c: &[C64],
    rank: usize,
    xv: &[CMom; 4],
    work: &mut [C64],
) -> [C64; L] {
    let mut result = [C64::ZERO; L];
    if rank == 0 {
        result[0] = c[0];
        return result;
    }
    let slot = |mu: usize| -> [C64; 4] { [xv[0][mu], xv[1][mu], xv[2][mu], xv[3][mu]] };
    let xs = [slot(0), slot(1), slot(2), slot(3)];
    let level = ncoeff_homogeneous(rank - 1);
    let (buf_a, buf_b) = work.split_at_mut(level * L);
    let (mut cur, _) = buf_a.as_chunks_mut::<L>();
    let (mut nxt, _) = buf_b[..level * L].as_chunks_mut::<L>();
    {
        let d = rank;
        let off = block_offset(d);
        let (r1, r2, r3) = splits(d);
        let g = &c[off..off + ncoeff_homogeneous(d)];
        let (g1, g2, g3) = (&g[1..1 + r1], &g[1 + r1..1 + r1 + r2], &g[1 + r1 + r2..]);
        let out = &mut cur[..r3];
        for i in 0..r3 {
            P::first_set(&mut out[i], g3[i], &xs[3]);
            if i < r2 {
                P::first_acc(&mut out[i], g2[i], &xs[2]);
                if i < r1 {
                    P::first_acc(&mut out[i], g1[i], &xs[1]);
                }
            }
        }
        P::first_acc(&mut out[0], g[0], &xs[0]);
        if let Some(pure) = P::PURE_X2[1] {
            let low = &c[block_offset(d - 1)..block_offset(d - 1) + r3];
            for i in 0..r3 {
                out[i][pure] += low[i];
            }
        }
    }
    let mut cur_len = level;
    let mut d = rank - 1;
    while d >= 1 {
        let (r1, r2, r3) = splits(d);
        let g = &cur[..cur_len];
        let (g1, g2, g3) = (&g[1..1 + r1], &g[1 + r1..1 + r1 + r2], &g[1 + r1 + r2..]);
        let out = &mut nxt[..r3];
        for i in 0..r3 {
            P::mul_set(&mut out[i], &g3[i], &xs[3]);
            if i < r2 {
                P::mul_acc(&mut out[i], &g2[i], &xs[2]);
                if i < r1 {
                    P::mul_acc(&mut out[i], &g1[i], &xs[1]);
                }
            }
        }
        P::mul_acc(&mut out[0], &g[0], &xs[0]);
        let power = rank - d + 1;
        if let Some(pure) = P::PURE_X2.get(power).copied().flatten() {
            let low = &c[block_offset(d - 1)..block_offset(d - 1) + r3];
            for i in 0..r3 {
                out[i][pure] += low[i];
            }
        }
        cur_len = r3;
        std::mem::swap(&mut cur, &mut nxt);
        d -= 1;
    }
    result.copy_from_slice(&cur[0]);
    result
}

/// Work entries [`horner_jet_x2`] needs for a rank-`rank` block and jet
/// length `l` (two levels of `l` components).
pub(super) fn work_len_x2(rank: usize, l: usize) -> usize {
    let level = if rank == 0 {
        1
    } else {
        ncoeff_homogeneous(rank - 1)
    };
    2 * l * level
}

/// Convert the internal T3 jet (powers of `s = param + mu²`) to Ninja's
/// layout (powers of `mu²`).
#[inline(always)]
pub(super) fn t3_output<const L: usize>(
    jet: &[C64; L],
    mindeg: usize,
    param: C64,
    out: &mut [C64],
) {
    for i in 0..=mindeg {
        let o = t3_len_const(i);
        match i / 2 {
            0 => out[o] = jet[o],
            1 => {
                out[o] = jet[o] + param * jet[o + 1];
                out[o + 1] = jet[o + 1];
            }
            _ => {
                out[o] = jet[o] + param * jet[o + 1] + param * param * jet[o + 2];
                out[o + 1] = jet[o + 1] + param * jet[o + 2] * 2.;
                out[o + 2] = jet[o + 2];
            }
        }
    }
}

/// Convert the internal T2 jet (`x^k S^j`, `S = p0 + p1 x + p2 x² + mu²`) to
/// Ninja's layout.
#[inline(always)]
pub(super) fn t2_output<const L: usize>(
    jet: &[C64; L],
    mindeg: usize,
    p: &[C64; 3],
    out: &mut [C64],
) {
    for i in 0..=mindeg {
        let o = t2_offset(i);
        match i {
            0 => out[o] = jet[o],
            1 => {
                out[o] = jet[o];
                out[o + 1] = jet[o + 1];
            }
            2 => {
                // internal [1, x, x², S] -> [1, mu², x, x²]
                let (c1, cx, cx2, cs) = (jet[o], jet[o + 1], jet[o + 2], jet[o + 3]);
                out[o] = c1 + p[0] * cs;
                out[o + 1] = cs;
                out[o + 2] = cx + p[1] * cs;
                out[o + 3] = cx2 + p[2] * cs;
            }
            _ => {
                // internal [1, x, x², x³, S, xS] -> [1, mu², x, x mu², x², x³]
                let (c1, cx, cx2, cx3) = (jet[o], jet[o + 1], jet[o + 2], jet[o + 3]);
                let (cs, cxs) = (jet[o + 4], jet[o + 5]);
                out[o] = c1 + p[0] * cs;
                out[o + 1] = cs;
                out[o + 2] = cx + p[1] * cs + p[0] * cxs;
                out[o + 3] = cxs;
                out[o + 4] = cx2 + p[2] * cs + p[1] * cxs;
                out[o + 5] = cx3 + p[2] * cxs;
            }
        }
    }
}

/// [`t2_output`] at `mindeg` 4 (rank n+2 bubbles). Order 4 of the internal
/// jet is `[1, x, x², x³, x⁴, S, xS, x²S, S²]`; with `S = P(x) + mu²`,
/// `P = p0 + p1 x + p2 x²`, the `S²` term spreads over
/// `P² + 2 P mu² + mu⁴` in Ninja's layout `[1, mu², mu⁴, x, x mu², x², x² mu², x³, x⁴]`.
#[inline(always)]
pub(super) fn t2_output_k4(jet: &[C64; 22], p: &[C64; 3], out: &mut [C64]) {
    t2_output(jet, 3, p, out);
    let o = t2_offset(4);
    let (a0, a1, a2, a3, a4) = (jet[o], jet[o + 1], jet[o + 2], jet[o + 3], jet[o + 4]);
    let (b0, b1, b2) = (jet[o + 5], jet[o + 6], jet[o + 7]);
    let c = jet[o + 8];
    let c2 = c * 2.;
    out[o] = a0 + b0 * p[0] + c * (p[0] * p[0]);
    out[o + 1] = b0 + c2 * p[0];
    out[o + 2] = c;
    out[o + 3] = a1 + b0 * p[1] + b1 * p[0] + c2 * (p[0] * p[1]);
    out[o + 4] = b1 + c2 * p[1];
    out[o + 5] = a2 + b0 * p[2] + b1 * p[1] + b2 * p[0] + c * (p[1] * p[1] + p[0] * p[2] * 2.);
    out[o + 6] = b2 + c2 * p[2];
    out[o + 7] = a3 + b1 * p[2] + b2 * p[1] + c2 * (p[1] * p[2]);
    out[o + 8] = a4 + b2 * p[2] + c * (p[2] * p[2]);
}

impl TensorNumerator {
    #[allow(clippy::too_many_arguments)]
    fn t3_rows<P: Jet<L>, const K: usize, const L: usize>(
        &mut self,
        lanes: &[usize],
        a: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[C64],
        out: &mut [C64],
    ) -> Result<()> {
        let rank = self.rank;
        let ncoeff = self.ncoeff;
        for (row, &lane) in lanes.iter().enumerate() {
            if lane >= self.lanes {
                return Err(Error::Lane {
                    lane,
                    lanes: self.lanes,
                });
            }
            let c = &self.coefficients[lane * ncoeff..(lane + 1) * ncoeff];
            let xv = [e3[row], a[row], e4[row], [C64::ZERO; 4]];
            let jet = horner_jet::<P, L>(c, rank, &xv, &mut self.work);
            t3_output(&jet, K, param[row], &mut out[row * L..(row + 1) * L]);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn t2_rows<P: Jet<L>, const K: usize, const L: usize>(
        &mut self,
        lanes: &[usize],
        a0: &[CMom],
        a1: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[[C64; 3]],
        out: &mut [C64],
    ) -> Result<()> {
        let rank = self.rank;
        let ncoeff = self.ncoeff;
        for (row, &lane) in lanes.iter().enumerate() {
            if lane >= self.lanes {
                return Err(Error::Lane {
                    lane,
                    lanes: self.lanes,
                });
            }
            let c = &self.coefficients[lane * ncoeff..(lane + 1) * ncoeff];
            let xv = [e3[row], a0[row], a1[row], e4[row]];
            let jet = horner_jet::<P, L>(c, rank, &xv, &mut self.work);
            t2_output(&jet, K, &param[row], &mut out[row * L..(row + 1) * L]);
        }
        Ok(())
    }
}

impl TensorNumerator {
    /// Grow the jet workspace for the rank n+2 jets (once, on first use).
    fn reserve_x2(&mut self, l: usize) {
        let needed = work_len_x2(self.rank, l);
        if self.work.len() < needed {
            self.work.resize(needed, C64::ZERO);
        }
    }
}

impl FastNumerator for TensorNumerator {
    fn rank(&self) -> usize {
        self.rank
    }

    fn evaluate(
        &mut self,
        _cut: &[usize],
        lanes: &[usize],
        q: &[CMom],
        _mu_squared: &[C64],
        out: &mut [C64],
    ) -> Result<()> {
        Self::check_rows(lanes.len(), out.len(), 1)?;
        for (row, &lane) in lanes.iter().enumerate() {
            self.lane(lane)?;
            let c = &self.coefficients[lane * self.ncoeff..(lane + 1) * self.ncoeff];
            out[row] = horner_scalar(c, self.rank, &q[row], true, &mut self.work);
        }
        Ok(())
    }

    fn mu_expansion(
        &mut self,
        _cut: &[usize],
        lanes: &[usize],
        v0: &[CMom],
        v1: Option<&[CMom]>,
        out: &mut [C64],
    ) -> Result<()> {
        match v1 {
            None => {
                Self::check_rows(lanes.len(), out.len(), 1)?;
                for (row, &lane) in lanes.iter().enumerate() {
                    self.lane(lane)?;
                    let c = &self.coefficients[lane * self.ncoeff..(lane + 1) * self.ncoeff];
                    out[row] = horner_scalar(c, self.rank, &v0[row], false, &mut self.work);
                }
            }
            Some(v1) => {
                Self::check_rows(lanes.len(), out.len(), 2)?;
                let rank = self.rank;
                let ncoeff = self.ncoeff;
                for (row, &lane) in lanes.iter().enumerate() {
                    if lane >= self.lanes {
                        return Err(Error::Lane {
                            lane,
                            lanes: self.lanes,
                        });
                    }
                    let c = &self.coefficients[lane * ncoeff..(lane + 1) * ncoeff];
                    let xv = [v0[row], v1[row], [C64::ZERO; 4], [C64::ZERO; 4]];
                    let jet = horner_jet::<jets::Mu, 2>(c, rank, &xv, &mut self.work);
                    out[2 * row..2 * row + 2].copy_from_slice(&jet);
                }
            }
        }
        Ok(())
    }

    /// The rank n+2 box expansion `q = t v0 + v1`: the leading three
    /// coefficients of `N(q)` (this numerator ignores `mu²`, so `beta` does
    /// not enter).
    #[inline(never)]
    fn mu_expansion_x2(
        &mut self,
        _cut: &[usize],
        lanes: &[usize],
        v0: &[CMom],
        v1: &[CMom],
        _beta: &[C64],
        out: &mut [C64],
    ) -> Result<()> {
        Self::check_rows(lanes.len(), out.len(), 3)?;
        self.reserve_x2(3);
        let (rank, ncoeff) = (self.rank, self.ncoeff);
        for (row, &lane) in lanes.iter().enumerate() {
            self.lane(lane)?;
            let c = &self.coefficients[lane * ncoeff..(lane + 1) * ncoeff];
            let xv = [v0[row], v1[row], [C64::ZERO; 4], [C64::ZERO; 4]];
            let jet = horner_jet_x2::<jets::Mu3, 3>(c, rank, &xv, &mut self.work);
            out[3 * row..3 * row + 3].copy_from_slice(&jet);
        }
        Ok(())
    }

    /// `mindeg` 5 (rank n+2 triangles) with the `T3K5` jet; `t3_expansion`
    /// keeps Ninja's depth limit of 4.
    #[inline(never)]
    fn t3_expansion_x2(
        &mut self,
        _cut: &[usize],
        lanes: &[usize],
        a: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[C64],
        out: &mut [C64],
    ) -> Result<()> {
        Self::check_rows(lanes.len(), out.len(), t3_len(5))?;
        self.reserve_x2(12);
        let (rank, ncoeff) = (self.rank, self.ncoeff);
        for (row, &lane) in lanes.iter().enumerate() {
            self.lane(lane)?;
            let c = &self.coefficients[lane * ncoeff..(lane + 1) * ncoeff];
            let xv = [e3[row], a[row], e4[row], [C64::ZERO; 4]];
            let jet = horner_jet_x2::<jets::T3K5, 12>(c, rank, &xv, &mut self.work);
            t3_output(&jet, 5, param[row], &mut out[row * 12..(row + 1) * 12]);
        }
        Ok(())
    }

    /// `mindeg` 4 (rank n+2 bubbles) with the `T2K4` jet; `t2_expansion`
    /// keeps Ninja's depth limit of 3.
    #[inline(never)]
    fn t2_expansion_x2(
        &mut self,
        _cut: &[usize],
        lanes: &[usize],
        a0: &[CMom],
        a1: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[[C64; 3]],
        out: &mut [C64],
    ) -> Result<()> {
        Self::check_rows(lanes.len(), out.len(), t2_len(4))?;
        self.reserve_x2(22);
        let (rank, ncoeff) = (self.rank, self.ncoeff);
        for (row, &lane) in lanes.iter().enumerate() {
            self.lane(lane)?;
            let c = &self.coefficients[lane * ncoeff..(lane + 1) * ncoeff];
            let xv = [e3[row], a0[row], a1[row], e4[row]];
            let jet = horner_jet_x2::<jets::T2K4, 22>(c, rank, &xv, &mut self.work);
            t2_output_k4(&jet, &param[row], &mut out[row * 22..(row + 1) * 22]);
        }
        Ok(())
    }

    fn t3_expansion(
        &mut self,
        _cut: &[usize],
        lanes: &[usize],
        a: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[C64],
        mindeg: usize,
        out: &mut [C64],
    ) -> Result<()> {
        if mindeg > 4 {
            return Err(Error::Expansion(format!("t3 mindeg {mindeg} > 4")));
        }
        Self::check_rows(lanes.len(), out.len(), t3_len(mindeg))?;
        match mindeg {
            0 => self.t3_rows::<jets::T3K0, 0, 1>(lanes, a, e3, e4, param, out),
            1 => self.t3_rows::<jets::T3K1, 1, 2>(lanes, a, e3, e4, param, out),
            2 => self.t3_rows::<jets::T3K2, 2, 4>(lanes, a, e3, e4, param, out),
            3 => self.t3_rows::<jets::T3K3, 3, 6>(lanes, a, e3, e4, param, out),
            _ => self.t3_rows::<jets::T3K4, 4, 9>(lanes, a, e3, e4, param, out),
        }
    }

    fn t2_expansion(
        &mut self,
        _cut: &[usize],
        lanes: &[usize],
        a0: &[CMom],
        a1: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[[C64; 3]],
        mindeg: usize,
        out: &mut [C64],
    ) -> Result<()> {
        if mindeg > 3 {
            return Err(Error::Expansion(format!("t2 mindeg {mindeg} > 3")));
        }
        Self::check_rows(lanes.len(), out.len(), t2_len_le3(mindeg))?;
        match mindeg {
            0 => self.t2_rows::<jets::T2K0, 0, 1>(lanes, a0, a1, e3, e4, param, out),
            1 => self.t2_rows::<jets::T2K1, 1, 3>(lanes, a0, a1, e3, e4, param, out),
            2 => self.t2_rows::<jets::T2K2, 2, 7>(lanes, a0, a1, e3, e4, param, out),
            _ => self.t2_rows::<jets::T2K3, 3, 13>(lanes, a0, a1, e3, e4, param, out),
        }
    }
}
