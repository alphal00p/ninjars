//! Lanes declared at rank n+2 whose numerator has a lower rank.
//!
//! A numerator of rank `r < n + 2` declared at rank n+2 has exactly zero
//! coefficients above degree `r`. The rank n+2 engine (`engine_x2.rs`)
//! reduces it correctly, but differently from the rank-`r` declaration: it
//! chooses its own bubble references and tadpole bases (`bases_x2.rs`),
//! expands one order deeper and subtracts by series division, and it costs
//! several times as much. The binary64 results therefore depended on the
//! declared rank. On the final review's 166 rank n+1 twins (final review,
//! finding 6) the rank n+2 engine's error exceeds the rank n+1 declaration's
//! by more than ten times on 6 lanes (981 times on `fr-light-equal-n5-r7`
//! lane 1) and is more than ten times smaller on 28. Forcing Ninja's bubble
//! references and tadpole bases in the rank n+2 engine leaves one lane more
//! than ten times worse and two more than ten times better: the differences
//! come mostly from the basis choices.
//!
//! At declared rank n+2, [`super::FastAmplitude::evaluate`] asks the
//! numerator for the rank of every lane ([`FastNumerator::lane_rank`];
//! `TensorNumerator` and `MuTensorNumerator` scan their coefficients, other
//! numerators answer `None` and keep the rank n+2 engine). A lane of rank
//! `r < n + 2` is reduced by the rank <= n+1 engine built at rank `r`, with
//! the numerator presented at that rank ([`FastNumerator::lower_rank`]), so
//! its result is bitwise that of declaring rank `r`, at the cost of rank
//! `r`. That is the undeclared result (the CLI derives the case's rank) when
//! the case's rank is the lane's rank. Only declared rank n+2 is lowered:
//! declaring rank n+1 for a lane of rank <= n still changes its result, as in
//! original Ninja (the rank n+1 engine stays bitwise Ninja). On the suite
//! controls declared at n+2, lowering to rank n+1 instead of the lane's rank
//! failed 4 lanes (`rn2-int-mixed-n2-r2` lane 1: 1.5e-13 against a tolerance
//! of 3.8e-14). [`AtDeclaredRank`] keeps the rank n+2 engine instead, for
//! instance for its conditioned bases.
//!
//! A batch whose lanes have different ranks is split: each rank's lanes are
//! reduced as their own sub-batch, and the results are scattered back. Both
//! engines reduce every lane independently of the others, so every lane's
//! result depends on the lane alone, and a batch stays bitwise identical to
//! its lanes run singly (as at every other rank). Keeping the rank n+2 engine
//! for a mixed batch would make a lane's result depend on its neighbours.
//!
//! The rank <= n+1 reductions run through the caller's own instance of the
//! rank <= n+1 body (`evaluate_up_to_n1`), so no second copy of that engine
//! is compiled. It is called directly rather than through
//! [`super::FastAmplitude::evaluate`]: a call back into the dispatcher would
//! make it recursive, and LLVM then stops inlining it into its callers. Only
//! the dispatch below is generic; the rest is compiled once, in this crate,
//! with its own vector element types (as in `engine_x2.rs`, so that this crate
//! exports no new instances of the vector routines the generic rank <= n+1
//! stages use).

use super::complex::{C64, CMom, RMom};
use super::engine::{FastAmplitude, FastOptions};
use super::engine_x2::{DynMasters, DynNumerator, evaluate_x2};
use super::masters::FastMasters;
use super::numerator::FastNumerator;
use super::{FastInput, FastResult, FastStatus};
use crate::{Error, Result};

/// A numerator reduced at its declared rank.
///
/// It forwards every method of the wrapped numerator except
/// [`FastNumerator::lane_rank`], which it answers with `None`, so
/// [`super::FastAmplitude`] keeps the rank n+2 engine also for lanes of a
/// lower rank. Validation uses it to run the rank n+2 engine on numerators of
/// lower rank with known results (design V5). It also selects the rank n+2
/// engine's conditioned bases for such numerators, which are often more
/// accurate than the rank <= n+1 engine's (module docs), at about 2 to 9
/// times the instructions, 4.5 times in geometric mean on padded benchmark
/// cases.
pub struct AtDeclaredRank<'a, N: ?Sized>(pub &'a mut N);

impl<N: FastNumerator + ?Sized> FastNumerator for AtDeclaredRank<'_, N> {
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

/// Lane-major inputs of one part of a split batch.
#[derive(Default)]
struct Part {
    /// The batch lane of every part lane.
    lanes: Vec<[usize; 1]>,
    shifts: Vec<[RMom; 1]>,
    masses: Vec<[C64; 1]>,
    scales: Vec<[f64; 1]>,
    s_matrix: Vec<[f64; 1]>,
    /// The caller's previous results (kept by failed lanes), then the part's.
    results: Vec<[FastResult; 1]>,
    statuses: Vec<[FastStatus; 1]>,
}

impl Part {
    /// Copy the inputs and the previous results of `self.lanes`.
    #[cold]
    fn gather(&mut self, n: usize, input: &FastInput, results: &[FastResult]) {
        let lanes = self.lanes.len();
        self.shifts.clear();
        self.shifts.resize(lanes * n, [[0.; 4]]);
        self.masses.clear();
        self.masses.resize(lanes * n, [C64::ZERO]);
        self.scales.clear();
        self.scales.resize(lanes, [0.]);
        self.s_matrix.clear();
        if input.s_matrix.is_some() {
            self.s_matrix.resize(lanes * n * n, [0.]);
        }
        self.results.clear();
        self.results.resize(lanes, [FastResult::default()]);
        self.statuses.clear();
        self.statuses.resize(lanes, [FastStatus::NotEvaluated]);
        let shifts = self.shifts.as_flattened_mut();
        let masses = self.masses.as_flattened_mut();
        for (row, &[lane]) in self.lanes.iter().enumerate() {
            shifts[row * n..(row + 1) * n].copy_from_slice(&input.shifts[lane * n..(lane + 1) * n]);
            masses[row * n..(row + 1) * n]
                .copy_from_slice(&input.masses_squared[lane * n..(lane + 1) * n]);
            self.scales[row] = [input.scales_squared[lane]];
            self.results[row] = [results[lane]];
            if let Some(matrix) = input.s_matrix {
                self.s_matrix.as_flattened_mut()[row * n * n..(row + 1) * n * n]
                    .copy_from_slice(&matrix[lane * n * n..(lane + 1) * n * n]);
            }
        }
    }

    /// `(batch lanes, input, results, statuses)` of the part.
    fn split(
        &mut self,
        with_s_matrix: bool,
    ) -> (
        &[usize],
        FastInput<'_>,
        &mut [FastResult],
        &mut [FastStatus],
    ) {
        let input = FastInput {
            lanes: self.lanes.len(),
            shifts: self.shifts.as_flattened(),
            masses_squared: self.masses.as_flattened(),
            scales_squared: self.scales.as_flattened(),
            s_matrix: with_s_matrix.then(|| self.s_matrix.as_flattened()),
        };
        (
            self.lanes.as_flattened(),
            input,
            self.results.as_flattened_mut(),
            self.statuses.as_flattened_mut(),
        )
    }

    /// Write the part's results and statuses to their batch lanes.
    #[cold]
    fn scatter(&self, results: &mut [FastResult], statuses: &mut [FastStatus]) {
        for (row, &[lane]) in self.lanes.iter().enumerate() {
            results[lane] = self.results[row][0];
            statuses[lane] = self.statuses[row][0];
        }
    }
}

/// The state of [`FastAmplitude`] at declared rank n+2 for lanes of a lower
/// rank, built on the first such evaluation.
pub(super) struct LowerRank {
    /// The rank-`r` engine at index `r <= n + 1`, built on first use.
    engines: Vec<Option<FastAmplitude>>,
    /// The lanes of rank `r` at index `r <= n + 2` (`n + 2`: the rank n+2
    /// engine).
    parts: Vec<Part>,
    /// Batch lanes of a rank n+2 numerator call.
    mapped: Vec<[usize; 1]>,
}

impl LowerRank {
    #[cold]
    fn new(n: usize) -> Self {
        Self {
            engines: (0..=n + 1).map(|_| None).collect(),
            parts: (0..=n + 2).map(|_| Part::default()).collect(),
            mapped: Vec::new(),
        }
    }
}

/// The rank-`rank` engine of `engines`, built on first use.
#[cold]
fn engine(
    engines: &mut [Option<FastAmplitude>],
    n: usize,
    rank: usize,
    options: FastOptions,
) -> Result<&mut FastAmplitude> {
    if engines[rank].is_none() {
        engines[rank] = Some(FastAmplitude::new(n, rank, options)?);
    }
    Ok(engines[rank].as_mut().unwrap())
}

/// Restores the numerator lowered by [`FastNumerator::lower_rank`] when
/// dropped, also when the reduction fails or panics.
struct Lowered<'a, N: FastNumerator + ?Sized>(&'a mut N);

impl<N: FastNumerator + ?Sized> Drop for Lowered<'_, N> {
    fn drop(&mut self) {
        self.0.restore_rank();
    }
}

/// The shape checks of both engines, made before a batch is split.
#[cold]
fn check_shapes(n: usize, input: &FastInput, results: usize, statuses: usize) -> Result<()> {
    let lanes = input.lanes;
    for (name, expected, actual) in [
        ("fast shifts", lanes * n, input.shifts.len()),
        ("fast masses", lanes * n, input.masses_squared.len()),
        ("fast scales", lanes, input.scales_squared.len()),
        ("fast results", lanes, results),
        ("fast statuses", lanes, statuses),
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
    Ok(())
}

/// [`super::FastAmplitude::evaluate`] at declared rank n+2 (same contract;
/// see the module docs). `amplitude.lower` holds the lower-rank state once a
/// lane of a lower rank has occurred.
#[cold]
#[inline(never)]
pub(super) fn evaluate_declared_n2<N: FastNumerator + ?Sized, M: FastMasters + ?Sized>(
    amplitude: &mut FastAmplitude,
    input: &FastInput,
    numerator: &mut N,
    masters: &mut M,
    results: &mut [FastResult],
    statuses: &mut [FastStatus],
) -> Result<()> {
    let FastAmplitude {
        topo,
        options,
        x2,
        lower,
        ..
    } = amplitude;
    let x2 = x2.as_deref_mut().expect("the rank n+2 engine");
    let (n, declared, lanes) = (topo.n, topo.rank, input.lanes);
    // The lane's rank if it is below the declared one.
    let lower_rank =
        |numerator: &mut N, lane: usize| numerator.lane_rank(lane).filter(|&rank| rank < declared);
    if numerator.rank() != declared || !(0..lanes).any(|lane| lower_rank(numerator, lane).is_some())
    {
        return evaluate_x2(
            x2,
            topo,
            input,
            &mut DynNumerator(numerator),
            &mut DynMasters(masters),
            results,
            statuses,
        );
    }
    check_shapes(n, input, results.len(), statuses.len())?;
    let LowerRank {
        engines,
        parts,
        mapped,
    } = &mut **lower.get_or_insert_with(|| Box::new(LowerRank::new(n)));
    for part in parts.iter_mut() {
        part.lanes.clear();
    }
    for lane in 0..lanes {
        let rank = lower_rank(numerator, lane).unwrap_or(declared);
        parts[rank].lanes.push([lane]);
    }
    if let Some(rank) = (0..declared).find(|&rank| parts[rank].lanes.len() == lanes) {
        // Every lane of one rank: the batch as it is.
        let engine = engine(engines, n, rank, *options)?;
        numerator.lower_rank(parts[rank].lanes.as_flattened(), rank)?;
        let lowered = Lowered(numerator);
        return engine.evaluate_up_to_n1(input, &mut *lowered.0, masters, results, statuses);
    }
    let with_s_matrix = input.s_matrix.is_some();
    // Every part is reduced before anything is published, so an error leaves
    // `results` and `statuses` unchanged, as in the engines.
    for (rank, part) in parts[..declared].iter_mut().enumerate() {
        if part.lanes.is_empty() {
            continue;
        }
        part.gather(n, input, results);
        let engine = engine(engines, n, rank, *options)?;
        let (lanes, part, part_results, part_statuses) = part.split(with_s_matrix);
        numerator.lower_rank(lanes, rank)?;
        let lowered = Lowered(&mut *numerator);
        engine.evaluate_up_to_n1(&part, &mut *lowered.0, masters, part_results, part_statuses)?;
    }
    let high = &mut parts[declared];
    if !high.lanes.is_empty() {
        high.gather(n, input, results);
        let (map, part, part_results, part_statuses) = high.split(with_s_matrix);
        evaluate_x2(
            x2,
            topo,
            &part,
            &mut LaneMap {
                inner: &mut DynNumerator(numerator),
                map,
                mapped,
            },
            &mut DynMasters(masters),
            part_results,
            part_statuses,
        )?;
    }
    for part in parts.iter() {
        part.scatter(results, statuses);
    }
    Ok(())
}

/// The numerator of a rank n+2 sub-batch: part lane `i` is batch lane
/// `map[i]`.
struct LaneMap<'a> {
    inner: &'a mut dyn FastNumerator,
    map: &'a [usize],
    mapped: &'a mut Vec<[usize; 1]>,
}

impl LaneMap<'_> {
    /// `(numerator, batch lanes of the part lanes)`.
    fn split<'s>(&'s mut self, lanes: &[usize]) -> (&'s mut dyn FastNumerator, &'s [usize]) {
        self.mapped.clear();
        self.mapped
            .extend(lanes.iter().map(|&lane| [self.map[lane]]));
        (&mut *self.inner, self.mapped.as_flattened())
    }
}

impl FastNumerator for LaneMap<'_> {
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
        let (inner, lanes) = self.split(lanes);
        inner.evaluate(cut, lanes, q, mu_squared, out)
    }
    fn mu_expansion(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        v0: &[CMom],
        v1: Option<&[CMom]>,
        out: &mut [C64],
    ) -> Result<()> {
        let (inner, lanes) = self.split(lanes);
        inner.mu_expansion(cut, lanes, v0, v1, out)
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
        let (inner, lanes) = self.split(lanes);
        inner.mu_expansion_x2(cut, lanes, v0, v1, beta, out)
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
        let (inner, lanes) = self.split(lanes);
        inner.t3_expansion(cut, lanes, a, e3, e4, param, mindeg, out)
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
        let (inner, lanes) = self.split(lanes);
        inner.t2_expansion(cut, lanes, a0, a1, e3, e4, param, mindeg, out)
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
        let (inner, lanes) = self.split(lanes);
        inner.t3_expansion_x2(cut, lanes, a, e3, e4, param, out)
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
        let (inner, lanes) = self.split(lanes);
        inner.t2_expansion_x2(cut, lanes, a0, a1, e3, e4, param, out)
    }
}
