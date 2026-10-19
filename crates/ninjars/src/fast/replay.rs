//! Recording and replaying providers for timing the pure reduction.
//!
//! A `Recording*` provider forwards every call to a wrapped provider and keeps
//! its outputs in call order. The matching `Replay*` provider returns those
//! outputs, in the same order, with one copy per call. Replays wrap around
//! after the last recorded call, so repeated evaluations of the recorded
//! batch need no explicit rewind. Call shapes are checked; inputs are not.

use super::complex::{C64, CMom};
use super::masters::{FastMaster, FastMasters};
use super::numerator::FastNumerator;
use crate::{Error, Result};

/// Records every numerator output of a wrapped [`FastNumerator`], and its
/// latest answer to [`FastNumerator::lane_rank`] per lane.
pub struct RecordingNumerator<N> {
    inner: N,
    values: Vec<C64>,
    calls: Vec<usize>,
    lane_ranks: Vec<Option<usize>>,
}

impl<N: FastNumerator> RecordingNumerator<N> {
    pub fn new(inner: N) -> Self {
        Self {
            inner,
            values: Vec::new(),
            calls: Vec::new(),
            lane_ranks: Vec::new(),
        }
    }
    pub fn inner(&self) -> &N {
        &self.inner
    }
    pub fn inner_mut(&mut self) -> &mut N {
        &mut self.inner
    }
    pub fn into_inner(self) -> N {
        self.inner
    }
    /// Discard the recording.
    pub fn clear(&mut self) {
        self.values.clear();
        self.calls.clear();
        self.lane_ranks.clear();
    }
    /// Number of recorded callbacks.
    pub fn calls(&self) -> usize {
        self.calls.len()
    }
    /// Number of recorded output values.
    pub fn values(&self) -> usize {
        self.values.len()
    }
    /// A replaying numerator returning the recorded outputs.
    pub fn replay(&self) -> ReplayNumerator {
        ReplayNumerator {
            rank: self.inner.rank(),
            values: self.values.clone(),
            calls: self.calls.clone(),
            call: 0,
            cursor: 0,
            lane_ranks: self.lane_ranks.clone(),
            declared: None,
        }
    }
    fn record(&mut self, out: &[C64]) {
        self.values.extend_from_slice(out);
        self.calls.push(out.len());
    }
}

impl<N: FastNumerator> FastNumerator for RecordingNumerator<N> {
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn evaluate(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        q: &[CMom],
        mu_squared: &[C64],
        out: &mut [C64],
    ) -> Result<()> {
        self.inner.evaluate(cut, lanes, q, mu_squared, out)?;
        self.record(out);
        Ok(())
    }
    fn mu_expansion(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        v0: &[CMom],
        v1: Option<&[CMom]>,
        out: &mut [C64],
    ) -> Result<()> {
        self.inner.mu_expansion(cut, lanes, v0, v1, out)?;
        self.record(out);
        Ok(())
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
        self.inner.mu_expansion_x2(cut, lanes, v0, v1, beta, out)?;
        self.record(out);
        Ok(())
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
        self.inner
            .t3_expansion(cut, lanes, a, e3, e4, param, mindeg, out)?;
        self.record(out);
        Ok(())
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
        self.inner
            .t2_expansion(cut, lanes, a0, a1, e3, e4, param, mindeg, out)?;
        self.record(out);
        Ok(())
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
        self.inner
            .t3_expansion_x2(cut, lanes, a, e3, e4, param, out)?;
        self.record(out);
        Ok(())
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
        self.inner
            .t2_expansion_x2(cut, lanes, a0, a1, e3, e4, param, out)?;
        self.record(out);
        Ok(())
    }
    fn lane_rank(&mut self, lane: usize) -> Option<usize> {
        let answer = self.inner.lane_rank(lane);
        if self.lane_ranks.len() <= lane {
            self.lane_ranks.resize(lane + 1, None);
        }
        self.lane_ranks[lane] = answer;
        answer
    }
    fn lower_rank(&mut self, lanes: &[usize], rank: usize) -> Result<()> {
        self.inner.lower_rank(lanes, rank)
    }
    fn restore_rank(&mut self) {
        self.inner.restore_rank();
    }
}

/// Returns recorded numerator outputs in call order at O(1) cost per call,
/// and the recorded [`FastNumerator::lane_rank`] answers, so that the engine
/// takes the recorded path.
#[derive(Clone, Debug)]
pub struct ReplayNumerator {
    rank: usize,
    values: Vec<C64>,
    calls: Vec<usize>,
    call: usize,
    cursor: usize,
    lane_ranks: Vec<Option<usize>>,
    /// The recorded rank while lowered.
    declared: Option<usize>,
}

impl ReplayNumerator {
    /// Restart from the first recorded call.
    pub fn rewind(&mut self) {
        self.call = 0;
        self.cursor = 0;
    }

    #[inline]
    fn next(&mut self, out: &mut [C64]) -> Result<()> {
        if self.call == self.calls.len() {
            self.rewind();
        }
        let len = self.calls.get(self.call).copied().unwrap_or(usize::MAX);
        if len != out.len() {
            return Err(Error::Shape {
                name: "replayed numerator call",
                expected: len,
                actual: out.len(),
            });
        }
        out.copy_from_slice(&self.values[self.cursor..self.cursor + len]);
        self.cursor += len;
        self.call += 1;
        Ok(())
    }
}

impl FastNumerator for ReplayNumerator {
    fn rank(&self) -> usize {
        self.rank
    }
    fn evaluate(
        &mut self,
        _: &[usize],
        _: &[usize],
        _: &[CMom],
        _: &[C64],
        out: &mut [C64],
    ) -> Result<()> {
        self.next(out)
    }
    fn mu_expansion(
        &mut self,
        _: &[usize],
        _: &[usize],
        _: &[CMom],
        _: Option<&[CMom]>,
        out: &mut [C64],
    ) -> Result<()> {
        self.next(out)
    }
    fn mu_expansion_x2(
        &mut self,
        _: &[usize],
        _: &[usize],
        _: &[CMom],
        _: &[CMom],
        _: &[C64],
        out: &mut [C64],
    ) -> Result<()> {
        self.next(out)
    }
    fn t3_expansion(
        &mut self,
        _: &[usize],
        _: &[usize],
        _: &[CMom],
        _: &[CMom],
        _: &[CMom],
        _: &[C64],
        _: usize,
        out: &mut [C64],
    ) -> Result<()> {
        self.next(out)
    }
    fn t2_expansion(
        &mut self,
        _: &[usize],
        _: &[usize],
        _: &[CMom],
        _: &[CMom],
        _: &[CMom],
        _: &[CMom],
        _: &[[C64; 3]],
        _: usize,
        out: &mut [C64],
    ) -> Result<()> {
        self.next(out)
    }
    fn lane_rank(&mut self, lane: usize) -> Option<usize> {
        self.lane_ranks.get(lane).copied().flatten()
    }
    /// Reports rank `rank` until [`FastNumerator::restore_rank`]; the
    /// recorded outputs already are those of the lowered numerator.
    fn lower_rank(&mut self, _: &[usize], rank: usize) -> Result<()> {
        if self.declared.is_some() || rank >= self.rank {
            return Err(Error::Unsupported(format!(
                "cannot lower the replayed rank-{} numerator to rank {rank}",
                self.rank
            )));
        }
        self.declared = Some(self.rank);
        self.rank = rank;
        Ok(())
    }
    fn restore_rank(&mut self) {
        if let Some(rank) = self.declared.take() {
            self.rank = rank;
        }
    }
}

/// Records every master family call of a wrapped [`FastMasters`].
pub struct RecordingMasters<M> {
    inner: M,
    values: Vec<[C64; 3]>,
    ok: Vec<bool>,
    calls: Vec<(FastMaster, usize)>,
}

impl<M: FastMasters> RecordingMasters<M> {
    pub fn new(inner: M) -> Self {
        Self {
            inner,
            values: Vec::new(),
            ok: Vec::new(),
            calls: Vec::new(),
        }
    }
    pub fn inner(&self) -> &M {
        &self.inner
    }
    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.inner
    }
    pub fn into_inner(self) -> M {
        self.inner
    }
    pub fn clear(&mut self) {
        self.values.clear();
        self.ok.clear();
        self.calls.clear();
    }
    /// Recorded `(family, rows)` calls.
    pub fn calls(&self) -> &[(FastMaster, usize)] {
        &self.calls
    }
    pub fn replay(&self) -> ReplayMasters {
        ReplayMasters {
            values: self.values.clone(),
            ok: self.ok.clone(),
            calls: self.calls.clone(),
            call: 0,
            cursor: 0,
            row: 0,
        }
    }
}

impl<M: FastMasters> FastMasters for RecordingMasters<M> {
    fn evaluate(
        &mut self,
        family: FastMaster,
        args: &[C64],
        out: &mut [[C64; 3]],
        ok: &mut [bool],
    ) -> Result<()> {
        self.inner.evaluate(family, args, out, ok)?;
        self.values.extend_from_slice(out);
        self.ok.extend_from_slice(ok);
        self.calls.push((family, ok.len()));
        Ok(())
    }
}

/// Returns recorded master values in call order at O(1) cost per call.
#[derive(Clone, Debug)]
pub struct ReplayMasters {
    values: Vec<[C64; 3]>,
    ok: Vec<bool>,
    calls: Vec<(FastMaster, usize)>,
    call: usize,
    cursor: usize,
    row: usize,
}

impl ReplayMasters {
    pub fn rewind(&mut self) {
        self.call = 0;
        self.cursor = 0;
        self.row = 0;
    }
}

impl FastMasters for ReplayMasters {
    fn evaluate(
        &mut self,
        family: FastMaster,
        _args: &[C64],
        out: &mut [[C64; 3]],
        ok: &mut [bool],
    ) -> Result<()> {
        if self.call == self.calls.len() {
            self.rewind();
        }
        match self.calls.get(self.call) {
            Some(&(recorded, rows)) if recorded == family && rows == ok.len() => {
                let len = out.len();
                out.copy_from_slice(&self.values[self.cursor..self.cursor + len]);
                ok.copy_from_slice(&self.ok[self.row..self.row + rows]);
                self.cursor += len;
                self.row += rows;
                self.call += 1;
                Ok(())
            }
            _ => Err(Error::Unsupported(format!(
                "replayed master call {} does not match {family:?} with {} rows",
                self.call,
                ok.len()
            ))),
        }
    }
}
