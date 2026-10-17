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

use super::complex::{C64, CMom};
use super::numerator::{FastNumerator, t2_len, t3_len};
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
fn horner_scalar(c: &[C64], rank: usize, q: &CMom, all_blocks: bool, work: &mut [C64]) -> C64 {
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

/// A truncated jet algebra: components, products with the cut vectors and
/// the slots of pure powers of `u`.
trait Plan {
    /// Number of cut-vector slots (`x` components carry one value per slot).
    const SLOTS: usize;
    /// `(out, in, slot)`: `jet[in] * x[slot]` contributes to `out`.
    const PRODUCTS: ([(u8, u8, u8); 40], usize);
    /// `(out, slot)` products of a scalar (component 0) input.
    const FIRST: ([(u8, u8); 4], usize);
    /// Component of `u^k` (no other variables), `u8::MAX` beyond truncation.
    const PURE: [u8; 5];
}

const fn t3_index(i: usize, j: usize) -> usize {
    t3_len_const(i) + j
}

const fn t3_len_const(i: usize) -> usize {
    // t3_len(i - 1) with t3_len(-1) = 0
    let mut total = 0;
    let mut k = 0;
    while k < i {
        total += 1 + k / 2;
        k += 1;
    }
    total
}

const fn t2_offset(i: usize) -> usize {
    let mut total = 0;
    let mut k = 0;
    while k < i {
        total += 1 + k + k / 2 + k / 3;
        k += 1;
    }
    total
}

/// Internal T2 component of `u^i x^k S^j` (j <= 1 since mindeg <= 3).
const fn t2_index(i: usize, k: usize, j: usize) -> usize {
    t2_offset(i) + if j == 0 { k } else { i + 1 + k }
}

struct T3Plan<const K: usize>;
struct T2Plan<const K: usize>;
struct MuPlan;

impl<const K: usize> Plan for T3Plan<K> {
    const SLOTS: usize = 3;
    const PRODUCTS: ([(u8, u8, u8); 40], usize) = {
        // slots: 0 = e3 (u^0), 1 = a (u^1), 2 = e4 (u^2 s)
        const STEP: [(usize, usize); 3] = [(0, 0), (1, 0), (2, 1)];
        let mut table = [(0u8, 0u8, 0u8); 40];
        let mut count = 0;
        let mut i = 0;
        while i <= K {
            let mut j = 0;
            while j <= i / 2 {
                let mut s = 0;
                while s < 3 {
                    let (di, dj) = STEP[s];
                    if i + di <= K {
                        table[count] = (
                            t3_index(i + di, j + dj) as u8,
                            t3_index(i, j) as u8,
                            s as u8,
                        );
                        count += 1;
                    }
                    s += 1;
                }
                j += 1;
            }
            i += 1;
        }
        (table, count)
    };
    const FIRST: ([(u8, u8); 4], usize) = {
        let mut table = [(0u8, 0u8); 4];
        let mut count = 0;
        table[count] = (0, 0);
        count += 1;
        if K >= 1 {
            table[count] = (t3_index(1, 0) as u8, 1);
            count += 1;
        }
        if K >= 2 {
            table[count] = (t3_index(2, 1) as u8, 2);
            count += 1;
        }
        (table, count)
    };
    const PURE: [u8; 5] = {
        let mut pure = [u8::MAX; 5];
        let mut k = 0;
        while k <= K {
            pure[k] = t3_index(k, 0) as u8;
            k += 1;
        }
        pure
    };
}

impl<const K: usize> Plan for T2Plan<K> {
    const SLOTS: usize = 4;
    const PRODUCTS: ([(u8, u8, u8); 40], usize) = {
        // slots: 0 = e3, 1 = a0 (u), 2 = a1 (u x), 3 = e4 (u^2 S)
        const STEP: [(usize, usize, usize); 4] = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 0, 1)];
        let mut table = [(0u8, 0u8, 0u8); 40];
        let mut count = 0;
        let mut i = 0;
        while i <= K {
            let mut j = 0;
            while j <= i / 2 {
                let mut k = 0;
                while k + 2 * j <= i {
                    let mut s = 0;
                    while s < 4 {
                        let (di, dk, dj) = STEP[s];
                        if i + di <= K {
                            table[count] = (
                                t2_index(i + di, k + dk, j + dj) as u8,
                                t2_index(i, k, j) as u8,
                                s as u8,
                            );
                            count += 1;
                        }
                        s += 1;
                    }
                    k += 1;
                }
                j += 1;
            }
            i += 1;
        }
        (table, count)
    };
    const FIRST: ([(u8, u8); 4], usize) = {
        let mut table = [(0u8, 0u8); 4];
        let mut count = 0;
        table[count] = (0, 0);
        count += 1;
        if K >= 1 {
            table[count] = (t2_index(1, 0, 0) as u8, 1);
            count += 1;
            table[count] = (t2_index(1, 1, 0) as u8, 2);
            count += 1;
        }
        if K >= 2 {
            table[count] = (t2_index(2, 0, 1) as u8, 3);
            count += 1;
        }
        (table, count)
    };
    const PURE: [u8; 5] = {
        let mut pure = [u8::MAX; 5];
        let mut k = 0;
        while k <= K {
            pure[k] = t2_index(k, 0, 0) as u8;
            k += 1;
        }
        pure
    };
}

impl Plan for MuPlan {
    const SLOTS: usize = 2;
    const PRODUCTS: ([(u8, u8, u8); 40], usize) = {
        let mut table = [(0u8, 0u8, 0u8); 40];
        table[0] = (0, 0, 0);
        table[1] = (1, 0, 1);
        table[2] = (1, 1, 0);
        (table, 3)
    };
    const FIRST: ([(u8, u8); 4], usize) = ([(0, 0), (1, 1), (0, 0), (0, 0)], 2);
    const PURE: [u8; 5] = [0, 1, u8::MAX, u8::MAX, u8::MAX];
}

/// Truncated-jet transposed Horner sweep; returns the jet of
/// `sum_d u^(R-d) N_d(x(u))` with `x_mu(u) = sum_s slot_s(u) xv[s][mu]`.
#[inline(always)]
fn horner_jet<P: Plan, const L: usize>(
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
    let (products, nproducts) = P::PRODUCTS;
    let (first, nfirst) = P::FIRST;
    let slot = |mu: usize| -> [C64; 4] {
        let mut xs = [C64::ZERO; 4];
        for s in 0..P::SLOTS {
            xs[s] = xv[s][mu];
        }
        xs
    };
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
        let scalar = |out: &mut [C64; L], value: C64, x: &[C64; 4], set: bool| {
            if set {
                *out = [C64::ZERO; L];
            }
            for &(o, s) in &first[..nfirst] {
                out[o as usize] += value * x[s as usize];
            }
        };
        for i in 0..r3 {
            scalar(&mut cur[i], g[1 + r1 + r2 + i], &xs[3], true);
        }
        for i in 0..r2 {
            scalar(&mut cur[i], g[1 + r1 + i], &xs[2], false);
        }
        for i in 0..r1 {
            scalar(&mut cur[i], g[1 + i], &xs[1], false);
        }
        scalar(&mut cur[0], g[0], &xs[0], false);
        let pure = P::PURE[1];
        if pure != u8::MAX {
            let low = &c[block_offset(d - 1)..block_offset(d - 1) + r3];
            for i in 0..r3 {
                cur[i][pure as usize] += low[i];
            }
        }
    }
    let mut cur_len = level;
    let mut d = rank - 1;
    while d >= 1 {
        let (r1, r2, r3) = splits(d);
        let g = &cur[..cur_len];
        let next = &mut nxt[..r3];
        let product = |out: &mut [C64; L], value: &[C64; L], x: &[C64; 4], set: bool| {
            if set {
                *out = [C64::ZERO; L];
            }
            for &(o, i, s) in &products[..nproducts] {
                out[o as usize] += value[i as usize] * x[s as usize];
            }
        };
        for i in 0..r3 {
            product(&mut next[i], &g[1 + r1 + r2 + i], &xs[3], true);
        }
        for i in 0..r2 {
            product(&mut next[i], &g[1 + r1 + i], &xs[2], false);
        }
        for i in 0..r1 {
            product(&mut next[i], &g[1 + i], &xs[1], false);
        }
        product(&mut next[0], &g[0], &xs[0], false);
        let power = rank - d + 1;
        if power < 5 && P::PURE[power] != u8::MAX {
            let pure = P::PURE[power] as usize;
            let low = &c[block_offset(d - 1)..block_offset(d - 1) + r3];
            for i in 0..r3 {
                next[i][pure] += low[i];
            }
        }
        cur_len = r3;
        std::mem::swap(&mut cur, &mut nxt);
        d -= 1;
    }
    result.copy_from_slice(&cur[0]);
    result
}

/// Convert the internal T3 jet (powers of `s = param + mu²`) to Ninja's
/// layout (powers of `mu²`).
#[inline(always)]
fn t3_output<const L: usize>(jet: &[C64; L], mindeg: usize, param: C64, out: &mut [C64]) {
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
fn t2_output<const L: usize>(jet: &[C64; L], mindeg: usize, p: &[C64; 3], out: &mut [C64]) {
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

impl TensorNumerator {
    #[allow(clippy::too_many_arguments)]
    fn t3_rows<const K: usize, const L: usize>(
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
            let jet = horner_jet::<T3Plan<K>, L>(c, rank, &xv, &mut self.work);
            t3_output(&jet, K, param[row], &mut out[row * L..(row + 1) * L]);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn t2_rows<const K: usize, const L: usize>(
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
            let jet = horner_jet::<T2Plan<K>, L>(c, rank, &xv, &mut self.work);
            t2_output(&jet, K, &param[row], &mut out[row * L..(row + 1) * L]);
        }
        Ok(())
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
                    let jet = horner_jet::<MuPlan, 2>(c, rank, &xv, &mut self.work);
                    out[2 * row..2 * row + 2].copy_from_slice(&jet);
                }
            }
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
            0 => self.t3_rows::<0, 1>(lanes, a, e3, e4, param, out),
            1 => self.t3_rows::<1, 2>(lanes, a, e3, e4, param, out),
            2 => self.t3_rows::<2, 4>(lanes, a, e3, e4, param, out),
            3 => self.t3_rows::<3, 6>(lanes, a, e3, e4, param, out),
            _ => self.t3_rows::<4, 9>(lanes, a, e3, e4, param, out),
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
        Self::check_rows(lanes.len(), out.len(), t2_len(mindeg))?;
        match mindeg {
            0 => self.t2_rows::<0, 1>(lanes, a0, a1, e3, e4, param, out),
            1 => self.t2_rows::<1, 3>(lanes, a0, a1, e3, e4, param, out),
            2 => self.t2_rows::<2, 7>(lanes, a0, a1, e3, e4, param, out),
            _ => self.t2_rows::<3, 13>(lanes, a0, a1, e3, e4, param, out),
        }
    }
}
