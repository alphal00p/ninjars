//! Tensor numerator with explicit `mu²` dependence for the fast engine.
//!
//! A numerator prepared in a dimensional scheme (for instance FDH, where every
//! self-contraction of the d-dimensional loop momentum becomes `q² - mu²`) is
//! a polynomial in the four loop components and `mu²`:
//! `N(q, mu²) = sum_{k=0}^{K} mu^(2k) T_k(q)`, where `T_k` is a dense
//! rank-`(R - 2k)` tensor in the [`super::tensor_layout`] ordering and `mu²`
//! has rank weight two. Lanes store the blocks `T_0, ..., T_K` contiguously
//! ([`super::tensor_layout::ncoeff_mu`] coefficients per lane).
//!
//! Every expansion reuses the truncated-jet Horner sweep of
//! [`super::TensorNumerator`] block by block: `t^-R mu^(2k) T_k(q)` equals
//! `mu^(2k) u^(2k) J_k(u)` with `u = 1/t` and `J_k` the jet of `T_k` relative
//! to `t^(R-2k)`, so block `k` enters the Laurent layouts `2k` orders later
//! and `k` powers of `mu²` higher. In the box mu-expansion
//! (`q = t v0 + v1`, `mu² = t² v0²`) it contributes `(v0²)^k` times its own
//! leading coefficients. Original Ninja's `TensorNumerator` has no `mu²`
//! slots; [`super::TensorNumerator`] remains the path for `mu²`-free numerators.

use super::complex::{C64, CMom, mp_cc};
use super::numerator::{FastNumerator, t2_len, t3_len};
use super::tensor::{horner_jet, horner_scalar, t2_offset, t2_output, t3_len_const, t3_output};
use super::tensor_jets::{self as jets, Jet};
use super::tensor_layout::{mu_block_offset, ncoeff, ncoeff_homogeneous, ncoeff_mu};
use crate::{Error, Result};

/// Batched tensor numerator `sum_k mu^(2k) T_k(q)` with per-lane coefficients.
#[derive(Clone, Debug)]
pub struct MuTensorNumerator {
    rank: usize,
    mu_powers: usize,
    ncoeff: usize,
    lanes: usize,
    coefficients: Vec<C64>,
    work: Vec<C64>,
}

/// One block's converted jet: at most `t3_len(4) = 9` or `t2_len(3) = 13` entries.
type Scratch = [C64; 13];

impl MuTensorNumerator {
    /// A rank-`rank` numerator with `mu²` powers `0..=mu_powers` for `lanes`
    /// phase-space lanes, with zero coefficients. Requires `2 mu_powers <= rank`.
    pub fn new(rank: usize, mu_powers: usize, lanes: usize) -> Result<Self> {
        if 2 * mu_powers > rank {
            return Err(Error::Unsupported(format!(
                "mu^{} exceeds the rank-{rank} weight (mu² counts two)",
                2 * mu_powers
            )));
        }
        let ncoeff = ncoeff_mu(rank, mu_powers);
        let level = if rank == 0 {
            1
        } else {
            ncoeff_homogeneous(rank - 1)
        };
        Ok(Self {
            rank,
            mu_powers,
            ncoeff,
            lanes,
            coefficients: vec![C64::ZERO; lanes * ncoeff],
            work: vec![C64::ZERO; 2 * 13 * level],
        })
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Highest power `K` of `mu²`.
    pub fn mu_powers(&self) -> usize {
        self.mu_powers
    }

    /// Coefficients per lane, [`super::tensor_layout::ncoeff_mu`].
    pub fn ncoeff(&self) -> usize {
        self.ncoeff
    }

    /// Range of the `mu^(2k)` block inside one lane's coefficients.
    pub fn block_range(&self, k: usize) -> std::ops::Range<usize> {
        let start = mu_block_offset(self.rank, k);
        start..start + ncoeff(self.rank - 2 * k)
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
                name: "mu tensor coefficients",
                expected: lanes * self.ncoeff,
                actual: coefficients.len(),
            });
        }
        self.lanes = lanes;
        self.coefficients.clear();
        self.coefficients.extend_from_slice(coefficients);
        Ok(())
    }

    fn check_lane(&self, lane: usize) -> Result<()> {
        if lane >= self.lanes {
            return Err(Error::Lane {
                lane,
                lanes: self.lanes,
            });
        }
        Ok(())
    }

    fn check_rows(rows: usize, out: usize, width: usize) -> Result<()> {
        if out != rows * width {
            return Err(Error::Shape {
                name: "mu tensor numerator output",
                expected: rows * width,
                actual: out,
            });
        }
        Ok(())
    }

    /// `(coefficients of block k, its rank)` for one lane.
    #[inline(always)]
    fn block(coefficients: &[C64], rank: usize, k: usize) -> (&[C64], usize) {
        let start = mu_block_offset(rank, k);
        let block_rank = rank - 2 * k;
        (&coefficients[start..start + ncoeff(block_rank)], block_rank)
    }
}

/// Jet of one block at `mindeg`, converted to Ninja's t3 layout in `out`.
#[inline(always)]
fn t3_block<P: Jet<L>, const K: usize, const L: usize>(
    c: &[C64],
    rank: usize,
    xv: &[CMom; 4],
    param: C64,
    work: &mut [C64],
    out: &mut [C64],
) {
    let jet = horner_jet::<P, L>(c, rank, xv, work);
    t3_output(&jet, K, param, out);
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn t3_dispatch(
    mindeg: usize,
    c: &[C64],
    rank: usize,
    xv: &[CMom; 4],
    param: C64,
    work: &mut [C64],
    out: &mut [C64],
) {
    match mindeg {
        0 => t3_block::<jets::T3K0, 0, 1>(c, rank, xv, param, work, out),
        1 => t3_block::<jets::T3K1, 1, 2>(c, rank, xv, param, work, out),
        2 => t3_block::<jets::T3K2, 2, 4>(c, rank, xv, param, work, out),
        3 => t3_block::<jets::T3K3, 3, 6>(c, rank, xv, param, work, out),
        _ => t3_block::<jets::T3K4, 4, 9>(c, rank, xv, param, work, out),
    }
}

/// Jet of one block at `mindeg`, converted to Ninja's t2 layout in `out`.
#[inline(always)]
fn t2_block<P: Jet<L>, const K: usize, const L: usize>(
    c: &[C64],
    rank: usize,
    xv: &[CMom; 4],
    param: &[C64; 3],
    work: &mut [C64],
    out: &mut [C64],
) {
    let jet = horner_jet::<P, L>(c, rank, xv, work);
    t2_output(&jet, K, param, out);
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn t2_dispatch(
    mindeg: usize,
    c: &[C64],
    rank: usize,
    xv: &[CMom; 4],
    param: &[C64; 3],
    work: &mut [C64],
    out: &mut [C64],
) {
    match mindeg {
        0 => t2_block::<jets::T2K0, 0, 1>(c, rank, xv, param, work, out),
        1 => t2_block::<jets::T2K1, 1, 3>(c, rank, xv, param, work, out),
        2 => t2_block::<jets::T2K2, 2, 7>(c, rank, xv, param, work, out),
        _ => t2_block::<jets::T2K3, 3, 13>(c, rank, xv, param, work, out),
    }
}

impl FastNumerator for MuTensorNumerator {
    fn rank(&self) -> usize {
        self.rank
    }

    fn evaluate(
        &mut self,
        _cut: &[usize],
        lanes: &[usize],
        q: &[CMom],
        mu_squared: &[C64],
        out: &mut [C64],
    ) -> Result<()> {
        Self::check_rows(lanes.len(), out.len(), 1)?;
        let (rank, ncoeff) = (self.rank, self.ncoeff);
        for (row, &lane) in lanes.iter().enumerate() {
            self.check_lane(lane)?;
            let c = &self.coefficients[lane * ncoeff..(lane + 1) * ncoeff];
            // Horner in mu²: T_0 + mu² (T_1 + mu² (...)).
            let mut value = C64::ZERO;
            for k in (0..=self.mu_powers).rev() {
                let (block, block_rank) = Self::block(c, rank, k);
                value = value * mu_squared[row]
                    + horner_scalar(block, block_rank, &q[row], true, &mut self.work);
            }
            out[row] = value;
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
        let (rank, ncoeff) = (self.rank, self.ncoeff);
        match v1 {
            None => {
                Self::check_rows(lanes.len(), out.len(), 1)?;
                for (row, &lane) in lanes.iter().enumerate() {
                    self.check_lane(lane)?;
                    let c = &self.coefficients[lane * ncoeff..(lane + 1) * ncoeff];
                    // mu² = t² v0²: block k contributes (v0²)^k times its t^(R-2k) term.
                    let v02 = mp_cc(&v0[row], &v0[row]);
                    let mut value = C64::ZERO;
                    for k in (0..=self.mu_powers).rev() {
                        let (block, block_rank) = Self::block(c, rank, k);
                        value = value * v02
                            + horner_scalar(block, block_rank, &v0[row], false, &mut self.work);
                    }
                    out[row] = value;
                }
            }
            Some(v1) => {
                Self::check_rows(lanes.len(), out.len(), 2)?;
                for (row, &lane) in lanes.iter().enumerate() {
                    self.check_lane(lane)?;
                    let c = &self.coefficients[lane * ncoeff..(lane + 1) * ncoeff];
                    let v02 = mp_cc(&v0[row], &v0[row]);
                    let xv = [v0[row], v1[row], [C64::ZERO; 4], [C64::ZERO; 4]];
                    let mut value = [C64::ZERO; 2];
                    for k in (0..=self.mu_powers).rev() {
                        let (block, block_rank) = Self::block(c, rank, k);
                        let jet = horner_jet::<jets::Mu, 2>(block, block_rank, &xv, &mut self.work);
                        value = [value[0] * v02 + jet[0], value[1] * v02 + jet[1]];
                    }
                    out[2 * row..2 * row + 2].copy_from_slice(&value);
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
        let width = t3_len(mindeg);
        Self::check_rows(lanes.len(), out.len(), width)?;
        let (rank, ncoeff) = (self.rank, self.ncoeff);
        let top = self.mu_powers.min(mindeg / 2);
        let mut scratch: Scratch = [C64::ZERO; 13];
        for (row, &lane) in lanes.iter().enumerate() {
            self.check_lane(lane)?;
            let c = &self.coefficients[lane * ncoeff..(lane + 1) * ncoeff];
            let xv = [e3[row], a[row], e4[row], [C64::ZERO; 4]];
            let target = &mut out[row * width..(row + 1) * width];
            let (block, block_rank) = Self::block(c, rank, 0);
            t3_dispatch(
                mindeg,
                block,
                block_rank,
                &xv,
                param[row],
                &mut self.work,
                target,
            );
            for k in 1..=top {
                let reduced = mindeg - 2 * k;
                let (block, block_rank) = Self::block(c, rank, k);
                let converted = &mut scratch[..t3_len(reduced)];
                t3_dispatch(
                    reduced,
                    block,
                    block_rank,
                    &xv,
                    param[row],
                    &mut self.work,
                    converted,
                );
                // Order i, mu^(2j) of block k is order i + 2k, mu^(2(j + k)).
                for i in 0..=reduced {
                    let (from, to) = (t3_len_const(i), t3_len_const(i + 2 * k) + k);
                    for j in 0..=i / 2 {
                        target[to + j] += converted[from + j];
                    }
                }
            }
        }
        Ok(())
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
        let width = t2_len(mindeg);
        Self::check_rows(lanes.len(), out.len(), width)?;
        let (rank, ncoeff) = (self.rank, self.ncoeff);
        // Only mu² itself fits below order four: block 1 enters orders 2 and 3.
        let shifted = self.mu_powers >= 1 && mindeg >= 2;
        let mut scratch: Scratch = [C64::ZERO; 13];
        for (row, &lane) in lanes.iter().enumerate() {
            self.check_lane(lane)?;
            let c = &self.coefficients[lane * ncoeff..(lane + 1) * ncoeff];
            let xv = [e3[row], a0[row], a1[row], e4[row]];
            let target = &mut out[row * width..(row + 1) * width];
            let (block, block_rank) = Self::block(c, rank, 0);
            t2_dispatch(
                mindeg,
                block,
                block_rank,
                &xv,
                &param[row],
                &mut self.work,
                target,
            );
            if shifted {
                let reduced = mindeg - 2;
                let (block, block_rank) = Self::block(c, rank, 1);
                let converted = &mut scratch[..t2_len(reduced)];
                t2_dispatch(
                    reduced,
                    block,
                    block_rank,
                    &xv,
                    &param[row],
                    &mut self.work,
                    converted,
                );
                // Order 0 `[1]` -> order 2 `mu²`; order 1 `[1, x]` -> order 3 `[mu², x mu²]`.
                target[t2_offset(2) + 1] += converted[0];
                if reduced >= 1 {
                    target[t2_offset(3) + 1] += converted[t2_offset(1)];
                    target[t2_offset(3) + 3] += converted[t2_offset(1) + 1];
                }
            }
        }
        Ok(())
    }
}
