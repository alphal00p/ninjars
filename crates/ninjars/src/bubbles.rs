//! Batched double-cut reduction with complete triangle-parent subtraction.

use crate::residue_storage::VecPool;
use crate::{
    BatchShape, BubbleCut, BubbleCutRequest, CutFailure, CutOutcome, CutProvider, Error,
    ExpansionKind, ExpansionRequest, FloatLike, FourMomentum, Numerator, Result, SampleId,
    Topology, TriangleResidue,
};

pub struct BubbleDivisionRequest<'a, T> {
    pub partition: &'a [usize],
    pub complement: &'a [usize],
    pub shifts: &'a [FourMomentum<T>],
    pub masses_squared: &'a [T],
    /// [a0,a1,growing,shrinking_effective].
    pub vectors: &'a [FourMomentum<T>; 4],
    pub parameters: &'a [T; 3],
}

pub struct BubbleTriangleCorrectionRequest<'a, T> {
    pub id: SampleId,
    pub partition: &'a [usize],
    pub source_partition: &'a [usize],
    pub source: &'a TriangleResidue<T>,
    pub shifts: &'a [FourMomentum<T>],
    pub masses_squared: &'a [T],
    pub vectors: &'a [FourMomentum<T>; 4],
    pub parameters: &'a [T; 3],
}

// These descriptors borrow all numerical data; copying never clones a carrier.
impl<T> Copy for BubbleTriangleCorrectionRequest<'_, T> {}
impl<T> Clone for BubbleTriangleCorrectionRequest<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Arithmetic boundary. Implementations perform numerical arithmetic across
/// complete batches; request packing and dependency handling remain in core.
pub trait BubbleReductionKernels<T> {
    fn propagators(&self) -> usize;
    fn rank(&self) -> usize;
    fn numerator_expansion(&self) -> &ExpansionKind;
    fn quotient_columns(&self) -> usize;
    fn coefficient_count(&self) -> usize;
    fn swapped_effective_batch(
        &mut self,
        cuts: &[&BubbleCut<T>],
        output: &mut [FourMomentum<T>],
    ) -> Result<()>;
    fn divide_batch(
        &mut self,
        requests: &[BubbleDivisionRequest<'_, T>],
        numerator: &[T],
        output: &mut [T],
        stable: &mut [bool],
    ) -> Result<()>;
    fn triangle_corrections_batch(
        &mut self,
        requests: &[BubbleTriangleCorrectionRequest<'_, T>],
        output: &mut [T],
        stable: &mut [bool],
    ) -> Result<()>;
    /// Divide and correct one immutable batch. Parent requests are ordered by
    /// division row, then parent. The two orientation division rows are adjacent,
    /// each followed by its own parent block. Only
    /// targets whose two divisions succeed reach parent validation/arithmetic.
    /// The default preserves standalone implementations and makes one compact
    /// correction call. Implementations may reuse denominators within this call.
    /// Buffer-shape errors are rejected before writes. As with separate calls,
    /// a later correction error may leave successful quotients/statuses written;
    /// these are stage scratch outputs, not transactionally published residues.
    #[allow(clippy::too_many_arguments)]
    fn divide_with_corrections_batch(
        &mut self,
        divisions: &[BubbleDivisionRequest<'_, T>],
        numerator: &[T],
        quotients: &mut [T],
        quotient_stable: &mut [bool],
        requests: &[BubbleTriangleCorrectionRequest<'_, T>],
        corrections: &mut [T],
        correction_stable: &mut [bool],
    ) -> Result<()>
    where
        T: Clone,
    {
        let parents = self.propagators() - 2;
        let width = self.quotient_columns();
        if !divisions.len().is_multiple_of(2) {
            return Err(Error::Expansion(
                "bubble division orientations are incomplete".into(),
            ));
        }
        BatchShape {
            rows: divisions.len(),
            columns: parents,
        }
        .validate("bubble parent request rows", requests.len())?;
        BatchShape {
            rows: requests.len(),
            columns: width,
        }
        .validate("bubble parent correction output", corrections.len())?;
        BatchShape {
            rows: requests.len(),
            columns: 1,
        }
        .validate("bubble parent correction status", correction_stable.len())?;
        self.divide_batch(divisions, numerator, quotients, quotient_stable)?;
        correction_stable.fill(false);
        let selected: Vec<_> = (0..requests.len())
            .filter(|&row| {
                let target = row / parents / 2;
                quotient_stable[2 * target..2 * target + 2]
                    .iter()
                    .all(|&v| v)
            })
            .collect();
        let selected_requests: Vec<_> = selected.iter().map(|&row| requests[row]).collect();
        let mut values: Vec<_> = selected
            .iter()
            .flat_map(|&row| corrections[row * width..(row + 1) * width].iter().cloned())
            .collect();
        let mut stable = vec![false; selected.len()];
        self.triangle_corrections_batch(&selected_requests, &mut values, &mut stable)?;
        for (index, &row) in selected.iter().enumerate() {
            correction_stable[row] = stable[index];
            if stable[index] {
                corrections[row * width..(row + 1) * width]
                    .clone_from_slice(&values[index * width..(index + 1) * width]);
            }
        }
        Ok(())
    }
    /// Combined arithmetic whose scratch outputs are consumed only by residue
    /// subtraction and projection. The first orientation is complete; an
    /// implementation may omit swapped heads unused by its complete residue
    /// projection. Such slots are unspecified scratch, not raw coefficients.
    ///
    /// Ordering, validation and publication rules match the full combined call.
    /// The default preserves its complete outputs and existing implementations.
    /// Call [`Self::divide_with_corrections_batch`] for complete raw quotients
    /// and corrections, independently of any projection-specific optimization.
    #[allow(clippy::too_many_arguments)]
    fn divide_with_corrections_for_projection_batch(
        &mut self,
        divisions: &[BubbleDivisionRequest<'_, T>],
        numerator: &[T],
        quotients: &mut [T],
        quotient_stable: &mut [bool],
        requests: &[BubbleTriangleCorrectionRequest<'_, T>],
        corrections: &mut [T],
        correction_stable: &mut [bool],
    ) -> Result<()>
    where
        T: Clone,
    {
        self.divide_with_corrections_batch(
            divisions,
            numerator,
            quotients,
            quotient_stable,
            requests,
            corrections,
            correction_stable,
        )
    }
    /// Correction rows are [orientation row, triangle parent, quotient column].
    fn subtract_batch(
        &mut self,
        quotients: &[T],
        corrections: &[T],
        output: &mut [T],
        rows: usize,
    ) -> Result<()>;
    /// Each row has the two corrected orientations followed separately by e12.
    fn project_batch(&mut self, quotients: &[T], e12: &[T], output: &mut [T]) -> Result<()>;
    fn values_are_finite(&self, values: &[T]) -> bool;
}

#[derive(Clone, Debug)]
pub struct BubbleResidue<T> {
    pub cut_index: usize,
    pub id: SampleId,
    pub geometry: BubbleCut<T>,
    /// Complete original Ninja coefficient order: 10 (rank <= n), 20 (n+1) or
    /// 35 (n+2, `tools/rank_n2/theory/formulas_rank_n2.json` order).
    pub coefficients: Vec<T>,
}

/// Double-cut stage. Parent outcomes must follow the same topology's triangle
/// cut-major, lane-minor order returned by `TriangleReducer`. Required unstable
/// parents invalidate only the dependent cut/lane. API errors leave output
/// unchanged. This stage alone does not constitute an integrated amplitude.
pub struct BubbleReducer<T, K> {
    topology: Topology,
    cuts: Vec<usize>,
    triangles: Vec<usize>,
    parents: Vec<Vec<usize>>,
    kernels: K,
    zero: T,
    numerator: Vec<T>,
    quotients: Vec<T>,
    quotient_stable: Vec<bool>,
    corrections: Vec<T>,
    correction_stable: Vec<bool>,
    geometry_output: Vec<CutOutcome<BubbleCut<T>>>,
    active: Vec<usize>,
    staged: Vec<CutOutcome<BubbleResidue<T>>>,
    coefficient_pool: VecPool<T>,
    requested_rows: Vec<usize>,
    complement_shifts: Vec<FourMomentum<T>>,
    swapped: Vec<FourMomentum<T>>,
    vectors: Vec<[FourMomentum<T>; 4]>,
    correction_starts: Vec<Option<usize>>,
    projected: Vec<usize>,
    e12: Vec<T>,
    raw: Vec<T>,
    correction_rows: Vec<T>,
    corrected: Vec<T>,
    coefficients: Vec<T>,
}

impl<T: FloatLike, K: BubbleReductionKernels<T>> BubbleReducer<T, K> {
    pub fn new(kernels: K, prototype: T) -> Result<Self> {
        if kernels.propagators() < 2 {
            return Err(Error::Unsupported(
                "bubble reduction requires at least two propagators".into(),
            ));
        }
        let topology = Topology::new(kernels.propagators(), kernels.rank())?;
        let indices = |size| {
            topology
                .cuts()
                .iter()
                .enumerate()
                .filter_map(|(i, cut)| {
                    (cut.partition.len() == size
                        && topology.rank().saturating_add(size) >= topology.propagators())
                    .then_some(i)
                })
                .collect::<Vec<_>>()
        };
        let cuts = indices(2);
        let triangles = indices(3);
        let parents = cuts
            .iter()
            .map(|&cut| {
                triangles
                    .iter()
                    .enumerate()
                    .filter_map(|(row, &parent)| {
                        topology.cuts()[cut]
                            .parents
                            .contains(&parent)
                            .then_some(row)
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            topology,
            cuts,
            triangles,
            parents,
            kernels,
            zero: prototype.zero(),
            geometry_output: Vec::new(),
            active: Vec::new(),
            staged: Vec::new(),
            coefficient_pool: VecPool::default(),
            requested_rows: Vec::new(),
            complement_shifts: Vec::new(),
            swapped: Vec::new(),
            vectors: Vec::new(),
            correction_starts: Vec::new(),
            projected: Vec::new(),
            e12: Vec::new(),
            numerator: Vec::new(),
            quotients: Vec::new(),
            quotient_stable: Vec::new(),
            corrections: Vec::new(),
            correction_stable: Vec::new(),
            raw: Vec::new(),
            correction_rows: Vec::new(),
            corrected: Vec::new(),
            coefficients: Vec::new(),
        })
    }

    pub fn topology(&self) -> &Topology {
        &self.topology
    }
    pub fn cut_count(&self) -> usize {
        self.cuts.len()
    }

    /// Recycle an explicitly discarded suffix; live caller outputs are never borrowed.
    pub(crate) fn discard_outcomes(
        &mut self,
        outcomes: &mut Vec<CutOutcome<BubbleResidue<T>>>,
        keep: usize,
    ) {
        for outcome in outcomes.drain(keep..) {
            if let CutOutcome::Stable(residue) = outcome {
                self.coefficient_pool.recycle(residue.coefficients);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_batch<N: Numerator<T>, C: CutProvider<T>>(
        &mut self,
        shifts: &[FourMomentum<T>],
        masses_squared: &[T],
        lanes: usize,
        triangle_topology: &Topology,
        triangles: &[CutOutcome<TriangleResidue<T>>],
        numerator: &mut N,
        geometry: &mut C,
        output: &mut [CutOutcome<BubbleResidue<T>>],
    ) -> Result<()> {
        let n = self.topology.propagators();
        if triangle_topology.propagators() != n || triangle_topology.rank() != self.topology.rank()
        {
            return Err(Error::Expansion(
                "parent triangle topology or rank differs from the bubble plan".into(),
            ));
        }
        BatchShape {
            rows: lanes,
            columns: n,
        }
        .validate("propagator shifts", shifts.len())?;
        BatchShape {
            rows: lanes,
            columns: n,
        }
        .validate("propagator squared masses", masses_squared.len())?;
        BatchShape {
            rows: lanes,
            columns: self.cuts.len(),
        }
        .validate("bubble residues", output.len())?;
        BatchShape {
            rows: lanes,
            columns: self.triangles.len(),
        }
        .validate("parent triangle residues", triangles.len())?;
        if numerator.rank() > self.topology.rank() {
            return Err(Error::Unsupported(
                "numerator rank exceeds configured bubble rank".into(),
            ));
        }
        let parent_coefficients = crate::residue_coefficient_count(3, self.topology.rank_class());
        for (cut_row, &cut_index) in self.triangles.iter().enumerate() {
            for lane in 0..lanes {
                if let CutOutcome::Stable(parent) = &triangles[cut_row * lanes + lane]
                    && (parent.cut_index != cut_index
                        || parent.id
                            != (SampleId {
                                lane,
                                sample: cut_index,
                            })
                        || parent.coefficients.len() != parent_coefficients)
                {
                    return Err(Error::Expansion(
                        "parent triangle identity or rank does not match the bubble topology"
                            .into(),
                    ));
                }
            }
        }
        if output.is_empty() {
            return Ok(());
        }
        // These are old outputs from a completed prior swap, never the live output.
        for outcome in self.staged.drain(..) {
            if let CutOutcome::Stable(residue) = outcome {
                self.coefficient_pool.recycle(residue.coefficients);
            }
        }
        let staged = &mut self.staged;
        staged.resize(
            output.len(),
            CutOutcome::Unstable(CutFailure::NonFiniteResult),
        );
        staged.fill(CutOutcome::Unstable(CutFailure::NonFiniteResult));
        let requested_rows = &mut self.requested_rows;
        requested_rows.clear();
        for cut_row in 0..self.cuts.len() {
            for lane in 0..lanes {
                let row = cut_row * lanes + lane;
                if let Some(failure) = self.parents[cut_row].iter().find_map(|&parent| {
                    match &triangles[parent * lanes + lane] {
                        CutOutcome::Unstable(failure) => Some(*failure),
                        _ => None,
                    }
                }) {
                    staged[row] = CutOutcome::Unstable(failure);
                    continue;
                }
                requested_rows.push(row);
            }
        }
        // Every uncut line is divided on the cut, so the provider sees all of
        // them when it chooses the transverse directions.
        let uncut = n - 2;
        let complement_shifts = &mut self.complement_shifts;
        complement_shifts.clear();
        complement_shifts.extend(requested_rows.iter().flat_map(|&row| {
            self.topology.cuts()[self.cuts[row / lanes]]
                .complement
                .iter()
                .map(move |&line| shifts[row % lanes * n + line].clone())
        }));
        let requests: Vec<_> = requested_rows
            .iter()
            .enumerate()
            .map(|(i, &row)| {
                let cut_index = self.cuts[row / lanes];
                let lane = row % lanes;
                let partition = &self.topology.cuts()[cut_index].partition;
                BubbleCutRequest {
                    id: SampleId {
                        lane,
                        sample: cut_index,
                    },
                    shifts: std::array::from_fn(|i| &shifts[lane * n + partition[i]]),
                    masses_squared: std::array::from_fn(|i| {
                        &masses_squared[lane * n + partition[i]]
                    }),
                    complementary_shifts: &complement_shifts[i * uncut..(i + 1) * uncut],
                }
            })
            .collect();
        let cuts = &mut self.geometry_output;
        cuts.resize(
            requests.len(),
            CutOutcome::Unstable(CutFailure::NonFiniteResult),
        );
        cuts.fill(CutOutcome::Unstable(CutFailure::NonFiniteResult));
        geometry.bubble_batch(&requests, cuts)?;
        let active = &mut self.active;
        active.clear();
        active.extend(cuts.iter().enumerate().filter_map(|(row, cut)| match cut {
            CutOutcome::Stable(_) => Some(row),
            CutOutcome::Unstable(failure) => {
                staged[requested_rows[row]] = CutOutcome::Unstable(*failure);
                None
            }
        }));
        let active_cuts: Vec<_> = active
            .iter()
            .map(|&row| {
                let CutOutcome::Stable(cut) = &cuts[row] else {
                    unreachable!()
                };
                cut
            })
            .collect();
        let zero_momentum = FourMomentum::from_args(
            self.zero.clone(),
            self.zero.clone(),
            self.zero.clone(),
            self.zero.clone(),
        );
        let swapped = &mut self.swapped;
        swapped.resize(active.len(), zero_momentum);
        self.kernels
            .swapped_effective_batch(&active_cuts, swapped)?;
        let vectors = &mut self.vectors;
        vectors.clear();
        for (cut, swap) in active_cuts.iter().zip(swapped.drain(..)) {
            vectors.push([
                cut.a0.clone(),
                cut.a1.clone(),
                cut.basis.e3.clone(),
                cut.e4_effective.clone(),
            ]);
            vectors.push([cut.a0.clone(), cut.a1.clone(), cut.basis.e4.clone(), swap]);
        }
        let mut expansions = Vec::with_capacity(vectors.len());
        let mut divisions = Vec::with_capacity(vectors.len());
        for (index, &request_row) in active.iter().enumerate() {
            let id = requests[request_row].id;
            let cut = &self.topology.cuts()[id.sample];
            for orientation in 0..2 {
                let vectors = &vectors[2 * index + orientation];
                expansions.push(ExpansionRequest {
                    id: SampleId {
                        lane: id.lane,
                        sample: 2 * id.sample + orientation,
                    },
                    partition: &cut.partition,
                    vectors,
                    parameters: &active_cuts[index].parameters,
                });
                divisions.push(BubbleDivisionRequest {
                    partition: &cut.partition,
                    complement: &cut.complement,
                    shifts: &shifts[id.lane * n..(id.lane + 1) * n],
                    masses_squared: &masses_squared[id.lane * n..(id.lane + 1) * n],
                    vectors,
                    parameters: &active_cuts[index].parameters,
                });
            }
        }
        let kind = self.kernels.numerator_expansion();
        self.numerator
            .resize(expansions.len() * kind.len(), self.zero.clone());
        numerator.expansion_batch(kind, &expansions, &mut self.numerator)?;
        let width = self.kernels.quotient_columns();
        self.quotients
            .resize(expansions.len() * width, self.zero.clone());
        self.quotient_stable.resize(expansions.len(), false);
        let parent_count = n - 2;
        let mut correction_requests = Vec::with_capacity(active.len() * 2 * parent_count);
        // Every eligible bubble has exactly n-2 triangle parents, in cached order.
        let correction_starts = &mut self.correction_starts;
        correction_starts.resize(active.len(), None);
        correction_starts.fill(None);
        for (index, &request_row) in active.iter().enumerate() {
            correction_starts[index] = Some(correction_requests.len());
            let id = requests[request_row].id;
            let target_row = requested_rows[request_row] / lanes;
            let partition = &self.topology.cuts()[id.sample].partition;
            for orientation in 0..2 {
                for &parent_row in &self.parents[target_row] {
                    let CutOutcome::Stable(source) = &triangles[parent_row * lanes + id.lane]
                    else {
                        unreachable!()
                    };
                    correction_requests.push(BubbleTriangleCorrectionRequest {
                        id: SampleId {
                            lane: id.lane,
                            sample: 2 * id.sample + orientation,
                        },
                        partition,
                        source_partition: &self.topology.cuts()[source.cut_index].partition,
                        source,
                        shifts: &shifts[id.lane * n..(id.lane + 1) * n],
                        masses_squared: &masses_squared[id.lane * n..(id.lane + 1) * n],
                        vectors: &vectors[2 * index + orientation],
                        parameters: &active_cuts[index].parameters,
                    });
                }
            }
        }
        self.corrections
            .resize(correction_requests.len() * width, self.zero.clone());
        self.correction_stable
            .resize(correction_requests.len(), false);
        self.kernels.divide_with_corrections_for_projection_batch(
            &divisions,
            &self.numerator,
            &mut self.quotients,
            &mut self.quotient_stable,
            &correction_requests,
            &mut self.corrections,
            &mut self.correction_stable,
        )?;
        self.raw.clear();
        self.correction_rows.clear();
        let projected = &mut self.projected;
        projected.clear();
        let e12 = &mut self.e12;
        e12.clear();
        for (index, start) in correction_starts.iter().copied().enumerate() {
            if !self.quotient_stable[2 * index..2 * index + 2]
                .iter()
                .all(|&v| v)
            {
                continue;
            }
            let Some(start) = start else {
                continue;
            };
            let end = start + 2 * parent_count;
            if !self.correction_stable[start..end]
                .iter()
                .all(|&stable| stable)
            {
                continue;
            }
            projected.push(index);
            e12.push(active_cuts[index].basis.e12.clone());
            self.raw
                .extend_from_slice(&self.quotients[2 * index * width..2 * (index + 1) * width]);
            self.correction_rows
                .extend_from_slice(&self.corrections[start * width..end * width]);
        }
        self.corrected.resize(self.raw.len(), self.zero.clone());
        self.kernels.subtract_batch(
            &self.raw,
            &self.correction_rows,
            &mut self.corrected,
            projected.len() * 2,
        )?;
        let count = self.kernels.coefficient_count();
        self.coefficients
            .resize(projected.len() * count, self.zero.clone());
        self.kernels
            .project_batch(&self.corrected, e12, &mut self.coefficients)?;
        for (projected_row, &index) in projected.iter().enumerate() {
            let values = &self.coefficients[projected_row * count..(projected_row + 1) * count];
            if self.kernels.values_are_finite(values) {
                let request_row = active[index];
                let id = requests[request_row].id;
                staged[requested_rows[request_row]] = CutOutcome::Stable(BubbleResidue {
                    cut_index: id.sample,
                    id,
                    geometry: (*active_cuts[index]).clone(),
                    coefficients: self.coefficient_pool.take_from_slice(values),
                });
            }
        }
        output.swap_with_slice(staged);
        Ok(())
    }
}
