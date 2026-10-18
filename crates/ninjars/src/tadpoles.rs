//! Tadpole coefficients needed for integration, with higher-cut subtraction.

use crate::residue_storage::VecPool;
use crate::{
    BatchShape, BubbleResidue, CutFailure, CutOutcome, CutProvider, Error, ExpansionKind,
    ExpansionRequest, FloatLike, FourMomentum, Numerator, Result, SampleId, TadpoleCut,
    TadpoleCutRequest, Topology, TriangleResidue,
};

pub struct TadpoleDivisionRequest<'a, T> {
    pub partition: &'a [usize],
    pub complement: &'a [usize],
    pub shifts: &'a [FourMomentum<T>],
    pub masses_squared: &'a [T],
    pub vectors: &'a [FourMomentum<T>; 3],
    pub beta: &'a T,
}

pub enum TadpoleSource<'a, T> {
    Triangle(&'a TriangleResidue<T>),
    Bubble(&'a BubbleResidue<T>),
}

pub struct TadpoleCorrectionRequest<'a, T> {
    /// Original phase-space lane and target single-cut index.
    pub id: SampleId,
    pub source: TadpoleSource<'a, T>,
    pub shifts: &'a [FourMomentum<T>],
    pub masses_squared: &'a [T],
    pub vectors: &'a [FourMomentum<T>; 3],
    pub beta: &'a T,
}

impl<T> Copy for TadpoleSource<'_, T> {}
impl<T> Clone for TadpoleSource<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for TadpoleCorrectionRequest<'_, T> {}
impl<T> Clone for TadpoleCorrectionRequest<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

/// All numerical arithmetic is supplied by the Symbolica adapter.
pub trait TadpoleReductionKernels<T> {
    fn propagators(&self) -> usize;
    fn rank(&self) -> usize;
    fn numerator_expansion(&self) -> &ExpansionKind;
    fn quotient_columns(&self) -> usize;
    fn coefficient_count(&self) -> usize;
    fn computed_coefficients(&self) -> &[bool];
    fn value_status(&self, values: &[T]) -> Option<CutFailure>;
    fn divide_batch(
        &mut self,
        requests: &[TadpoleDivisionRequest<'_, T>],
        numerator: &[T],
        output: &mut [T],
        stable: &mut [bool],
    ) -> Result<()>;
    fn correction_batch(
        &mut self,
        requests: &[TadpoleCorrectionRequest<'_, T>],
        output: &mut [T],
        stable: &mut [bool],
    ) -> Result<()>;
    /// Same-call division and parent correction. Requests are ordered by
    /// division row, then parent; higher-rank orientation pairs stay adjacent.
    /// Failed target groups never reach parent validation or arithmetic.
    /// Buffer-shape errors are rejected before writes. As with separate calls,
    /// a later correction error may leave successful quotients/statuses written;
    /// these are stage scratch outputs, not transactionally published residues.
    #[allow(clippy::too_many_arguments)]
    fn divide_with_corrections_batch(
        &mut self,
        divisions: &[TadpoleDivisionRequest<'_, T>],
        numerator: &[T],
        quotients: &mut [T],
        quotient_stable: &mut [bool],
        requests: &[TadpoleCorrectionRequest<'_, T>],
        corrections: &mut [T],
        correction_stable: &mut [bool],
    ) -> Result<()>
    where
        T: Clone,
    {
        let n = self.propagators();
        let parents = n
            .checked_mul(n - 1)
            .ok_or_else(|| Error::Expansion("tadpole parent count overflows".into()))?
            / 2;
        let orientations = if self.rank() > n { 2 } else { 1 };
        let width = self.quotient_columns();
        if !divisions.len().is_multiple_of(orientations) {
            return Err(Error::Expansion(
                "tadpole division orientations are incomplete".into(),
            ));
        }
        BatchShape {
            rows: divisions.len(),
            columns: parents,
        }
        .validate("tadpole parent request rows", requests.len())?;
        BatchShape {
            rows: requests.len(),
            columns: width,
        }
        .validate("tadpole parent correction output", corrections.len())?;
        BatchShape {
            rows: requests.len(),
            columns: 1,
        }
        .validate("tadpole parent correction status", correction_stable.len())?;
        self.divide_batch(divisions, numerator, quotients, quotient_stable)?;
        correction_stable.fill(false);
        let selected: Vec<_> = (0..requests.len())
            .filter(|&row| {
                let target = row / parents / orientations;
                quotient_stable[orientations * target..orientations * (target + 1)]
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
        self.correction_batch(&selected_requests, &mut values, &mut stable)?;
        for (index, &row) in selected.iter().enumerate() {
            correction_stable[row] = stable[index];
            if stable[index] {
                corrections[row * width..(row + 1) * width]
                    .clone_from_slice(&values[index * width..(index + 1) * width]);
            }
        }
        Ok(())
    }
    /// Corrections are orientation-row-major, then cached parent order.
    fn combine_batch(
        &mut self,
        raw: &[T],
        corrections: &[T],
        rows: usize,
        output: &mut [T],
    ) -> Result<()>;
    /// Ordinary rows are transverse. Higher-rank rows are longitudinal, then transverse.
    fn project_batch(&mut self, corrected: &[T], e12: &[T], output: &mut [T]) -> Result<()>;
}

/// Minimal integrated tadpole data, not a complete reconstructed residue.
#[derive(Clone, Debug)]
pub struct IntegratedTadpoleResidue<T> {
    pub cut_index: usize,
    pub id: SampleId,
    pub geometry: TadpoleCut<T>,
    /// Original 5/16-entry Ninja indexing; uncomputed entries are placeholders.
    pub coefficients: Vec<T>,
    /// Only c0 (ordinary), or c0/c14/c15 (rank n+1), are computed.
    pub computed: Vec<bool>,
}

#[derive(Clone, Copy)]
enum Parent {
    Triangle(usize),
    Bubble(usize),
}

/// Single-cut Laurent reduction with all triangle and corrected-bubble parents.
///
/// Parent outcomes and outputs use cut-major, lane-minor order. Exact topology
/// and rank metadata are required even when a parent stage is structurally empty.
/// API errors leave output unchanged. Numerical failures propagate only to
/// dependent cuts in the same lane. Massless points are retained here; master
/// integration decides whether an integral or its derivatives are scaleless.
pub struct TadpoleReducer<T, K> {
    topology: Topology,
    cuts: Vec<usize>,
    triangle_cuts: Vec<usize>,
    bubble_cuts: Vec<usize>,
    parents: Vec<Vec<Parent>>,
    kernels: K,
    zero: T,
    numerator: Vec<T>,
    quotients: Vec<T>,
    corrections: Vec<T>,
    raw: Vec<T>,
    selected_corrections: Vec<T>,
    combined: Vec<T>,
    projection: Vec<T>,
    coefficients: Vec<T>,
    division_stable: Vec<bool>,
    correction_stable: Vec<bool>,
    geometry_output: Vec<CutOutcome<TadpoleCut<T>>>,
    active: Vec<usize>,
    staged: Vec<CutOutcome<IntegratedTadpoleResidue<T>>>,
    coefficient_pool: VecPool<T>,
    mask_pool: VecPool<bool>,
    eligible: Vec<usize>,
    complement_shifts: Vec<FourMomentum<T>>,
    vectors: Vec<[FourMomentum<T>; 3]>,
    divided: Vec<usize>,
    corrected: Vec<usize>,
    projected: Vec<usize>,
    e12: Vec<T>,
}

impl<T: FloatLike, K: TadpoleReductionKernels<T>> TadpoleReducer<T, K> {
    pub fn new(kernels: K, prototype: T) -> Result<Self> {
        let topology = Topology::new(kernels.propagators(), kernels.rank())?;
        let stage = |size| {
            topology
                .cuts()
                .iter()
                .enumerate()
                .filter_map(|(index, cut)| {
                    (cut.partition.len() == size
                        && topology.rank().saturating_add(size) >= topology.propagators())
                    .then_some(index)
                })
                .collect::<Vec<_>>()
        };
        let cuts = stage(1);
        let triangle_cuts = stage(3);
        let bubble_cuts = stage(2);
        // Topology parents are ascending, so triangles (larger cuts, earlier)
        // precede bubbles, each in row order.
        let mut rows = vec![None; topology.cuts().len()];
        for (row, &triangle) in triangle_cuts.iter().enumerate() {
            rows[triangle] = Some(Parent::Triangle(row));
        }
        for (row, &bubble) in bubble_cuts.iter().enumerate() {
            rows[bubble] = Some(Parent::Bubble(row));
        }
        let parents = cuts
            .iter()
            .map(|&index| {
                topology.cuts()[index]
                    .parents
                    .iter()
                    .filter_map(|&parent| rows[parent])
                    .collect()
            })
            .collect();
        if kernels.computed_coefficients().len() != kernels.coefficient_count() {
            return Err(Error::Unsupported(
                "tadpole coefficient mask has incorrect dimensions".into(),
            ));
        }
        Ok(Self {
            topology,
            cuts,
            triangle_cuts,
            bubble_cuts,
            parents,
            kernels,
            zero: prototype.zero(),
            geometry_output: Vec::new(),
            active: Vec::new(),
            staged: Vec::new(),
            coefficient_pool: VecPool::default(),
            mask_pool: VecPool::default(),
            eligible: Vec::new(),
            complement_shifts: Vec::new(),
            vectors: Vec::new(),
            divided: Vec::new(),
            corrected: Vec::new(),
            projected: Vec::new(),
            e12: Vec::new(),
            numerator: Vec::new(),
            quotients: Vec::new(),
            corrections: Vec::new(),
            raw: Vec::new(),
            selected_corrections: Vec::new(),
            combined: Vec::new(),
            projection: Vec::new(),
            coefficients: Vec::new(),
            division_stable: Vec::new(),
            correction_stable: Vec::new(),
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
        outcomes: &mut Vec<CutOutcome<IntegratedTadpoleResidue<T>>>,
        keep: usize,
    ) {
        for outcome in outcomes.drain(keep..) {
            if let CutOutcome::Stable(residue) = outcome {
                self.coefficient_pool.recycle(residue.coefficients);
                self.mask_pool.recycle(residue.computed);
            }
        }
    }

    fn source_status(kernels: &K, source: TadpoleSource<'_, T>) -> Option<CutFailure> {
        let mut check = |value: &T| kernels.value_status(std::slice::from_ref(value));
        match source {
            TadpoleSource::Triangle(r) => {
                let b = &r.geometry.basis;
                r.coefficients
                    .iter()
                    .chain([&b.e12, &r.geometry.beta])
                    .chain(
                        [
                            &b.e1,
                            &b.e2,
                            &b.e3,
                            &b.e4,
                            &r.geometry.a,
                            &r.geometry.e4_effective,
                        ]
                        .into_iter()
                        .flat_map(FourMomentum::components),
                    )
                    .find_map(&mut check)
            }
            TadpoleSource::Bubble(r) => {
                let b = &r.geometry.basis;
                r.coefficients
                    .iter()
                    .chain([&b.e12])
                    .chain(&r.geometry.parameters)
                    .chain(
                        [
                            &b.e1,
                            &b.e2,
                            &b.e3,
                            &b.e4,
                            &r.geometry.a0,
                            &r.geometry.a1,
                            &r.geometry.e4_effective,
                        ]
                        .into_iter()
                        .flat_map(FourMomentum::components),
                    )
                    .find_map(&mut check)
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
        bubble_topology: &Topology,
        bubbles: &[CutOutcome<BubbleResidue<T>>],
        numerator: &mut N,
        geometry: &mut C,
        output: &mut [CutOutcome<IntegratedTadpoleResidue<T>>],
    ) -> Result<()> {
        let n = self.topology.propagators();
        let rank = self.topology.rank();
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
        .validate("integrated tadpole residues", output.len())?;
        for parent in [triangle_topology, bubble_topology] {
            if parent.propagators() != n || parent.rank() != rank {
                return Err(Error::Unsupported(
                    "parent residue topology/rank differs from tadpole topology".into(),
                ));
            }
        }
        BatchShape {
            rows: lanes,
            columns: self.triangle_cuts.len(),
        }
        .validate("parent triangle residues", triangles.len())?;
        BatchShape {
            rows: lanes,
            columns: self.bubble_cuts.len(),
        }
        .validate("parent bubble residues", bubbles.len())?;
        let higher = rank > n;
        for (row, parent) in triangles.iter().enumerate() {
            if let CutOutcome::Stable(parent) = parent {
                let expected = self.triangle_cuts[row / lanes];
                if parent.cut_index != expected
                    || parent.id
                        != (SampleId {
                            lane: row % lanes,
                            sample: expected,
                        })
                    || parent.coefficients.len() != if higher { 15 } else { 10 }
                {
                    return Err(Error::Unsupported(
                        "triangle parent provenance/coefficient layout is inconsistent".into(),
                    ));
                }
            }
        }
        for (row, parent) in bubbles.iter().enumerate() {
            if let CutOutcome::Stable(parent) = parent {
                let expected = self.bubble_cuts[row / lanes];
                if parent.cut_index != expected
                    || parent.id
                        != (SampleId {
                            lane: row % lanes,
                            sample: expected,
                        })
                    || parent.coefficients.len() != if higher { 20 } else { 10 }
                {
                    return Err(Error::Unsupported(
                        "bubble parent provenance/coefficient layout is inconsistent".into(),
                    ));
                }
            }
        }
        if numerator.rank() > rank {
            return Err(Error::Rank {
                rank: numerator.rank(),
                propagators: n,
            });
        }
        // These are old outputs from a completed prior swap, never the live output.
        for outcome in self.staged.drain(..) {
            if let CutOutcome::Stable(residue) = outcome {
                self.coefficient_pool.recycle(residue.coefficients);
                self.mask_pool.recycle(residue.computed);
            }
        }
        let staged = &mut self.staged;
        staged.resize(
            output.len(),
            CutOutcome::Unstable(CutFailure::NonFiniteResult),
        );
        staged.fill(CutOutcome::Unstable(CutFailure::NonFiniteResult));
        let eligible = &mut self.eligible;
        eligible.clear();
        for cut in 0..self.cuts.len() {
            for lane in 0..lanes {
                let failure = self.parents[cut].iter().find_map(|parent| match *parent {
                    Parent::Triangle(index) => match &triangles[index * lanes + lane] {
                        CutOutcome::Unstable(f) => Some(*f),
                        CutOutcome::Stable(r) => {
                            Self::source_status(&self.kernels, TadpoleSource::Triangle(r))
                        }
                    },
                    Parent::Bubble(index) => match &bubbles[index * lanes + lane] {
                        CutOutcome::Unstable(f) => Some(*f),
                        CutOutcome::Stable(r) => {
                            Self::source_status(&self.kernels, TadpoleSource::Bubble(r))
                        }
                    },
                });
                let row = cut * lanes + lane;
                if let Some(failure) = failure {
                    staged[row] = CutOutcome::Unstable(failure);
                } else {
                    eligible.push(row);
                }
            }
        }
        let complement_shifts = &mut self.complement_shifts;
        complement_shifts.clear();
        complement_shifts.extend(eligible.iter().flat_map(|&row| {
            self.topology.cuts()[self.cuts[row / lanes]]
                .complement
                .iter()
                .map(move |&line| shifts[row % lanes * n + line].clone())
        }));
        let requests: Vec<_> = eligible
            .iter()
            .enumerate()
            .map(|(i, &row)| {
                let cut_index = self.cuts[row / lanes];
                let lane = row % lanes;
                let line = self.topology.cuts()[cut_index].partition[0];
                TadpoleCutRequest {
                    id: SampleId {
                        lane,
                        sample: cut_index,
                    },
                    shift: &shifts[lane * n + line],
                    mass_squared: &masses_squared[lane * n + line],
                    complementary_shifts: &complement_shifts[i * (n - 1)..(i + 1) * (n - 1)],
                }
            })
            .collect();
        let cuts = &mut self.geometry_output;
        cuts.resize(
            requests.len(),
            CutOutcome::Unstable(CutFailure::NonFiniteResult),
        );
        cuts.fill(CutOutcome::Unstable(CutFailure::NonFiniteResult));
        geometry.tadpole_batch(&requests, cuts)?;
        for (&row, cut) in eligible.iter().zip(cuts.iter()) {
            if let CutOutcome::Unstable(f) = cut {
                staged[row] = CutOutcome::Unstable(*f);
            }
        }
        let active = &mut self.active;
        active.clear();
        active.extend(
            cuts.iter()
                .enumerate()
                .filter_map(|(i, cut)| matches!(cut, CutOutcome::Stable(_)).then_some(i)),
        );
        let active_cut = |index: usize| {
            let CutOutcome::Stable(cut) = &cuts[active[index]] else {
                unreachable!()
            };
            cut
        };
        let orientations = if higher { 2 } else { 1 };
        let vectors = &mut self.vectors;
        vectors.clear();
        for index in 0..active.len() {
            let cut = active_cut(index);
            if higher {
                vectors.push([
                    cut.a.clone(),
                    cut.basis.e1.clone(),
                    cut.e2_effective.clone(),
                ]);
            }
            vectors.push([
                cut.a.clone(),
                cut.basis.e3.clone(),
                cut.e4_effective.clone(),
            ]);
        }
        let mut expansions = Vec::with_capacity(vectors.len());
        let mut divisions = Vec::with_capacity(vectors.len());
        for (index, &active_row) in active.iter().enumerate() {
            let id = requests[active_row].id;
            let cut = &self.topology.cuts()[id.sample];
            for orientation in 0..orientations {
                let vectors = &vectors[index * orientations + orientation];
                expansions.push(ExpansionRequest {
                    id: SampleId {
                        lane: id.lane,
                        sample: orientations * id.sample + orientation,
                    },
                    partition: &cut.partition,
                    vectors,
                    parameters: std::slice::from_ref(&active_cut(index).beta),
                });
                divisions.push(TadpoleDivisionRequest {
                    partition: &cut.partition,
                    complement: &cut.complement,
                    shifts: &shifts[id.lane * n..(id.lane + 1) * n],
                    masses_squared: &masses_squared[id.lane * n..(id.lane + 1) * n],
                    vectors,
                    beta: &active_cut(index).beta,
                });
            }
        }
        let kind = self.kernels.numerator_expansion();
        let width = self.kernels.quotient_columns();
        self.numerator
            .resize(expansions.len() * kind.len(), self.zero.clone());
        numerator.expansion_batch(kind, &expansions, &mut self.numerator)?;
        self.quotients
            .resize(expansions.len() * width, self.zero.clone());
        self.division_stable.resize(expansions.len(), false);
        let parent_count = self.parents.first().map_or(0, Vec::len);
        let mut corrections = Vec::with_capacity(active.len() * orientations * parent_count);
        for index in 0..active.len() {
            let active_row = active[index];
            let id = requests[active_row].id;
            let target = eligible[active_row] / lanes;
            for orientation in 0..orientations {
                for &parent in &self.parents[target] {
                    let source = match parent {
                        Parent::Triangle(i) => {
                            let CutOutcome::Stable(r) = &triangles[i * lanes + id.lane] else {
                                unreachable!()
                            };
                            TadpoleSource::Triangle(r)
                        }
                        Parent::Bubble(i) => {
                            let CutOutcome::Stable(r) = &bubbles[i * lanes + id.lane] else {
                                unreachable!()
                            };
                            TadpoleSource::Bubble(r)
                        }
                    };
                    corrections.push(TadpoleCorrectionRequest {
                        id,
                        source,
                        shifts: &shifts[id.lane * n..(id.lane + 1) * n],
                        masses_squared: &masses_squared[id.lane * n..(id.lane + 1) * n],
                        vectors: &vectors[index * orientations + orientation],
                        beta: &active_cut(index).beta,
                    });
                }
            }
        }
        self.corrections
            .resize(corrections.len() * width, self.zero.clone());
        self.correction_stable.resize(corrections.len(), false);
        self.kernels.divide_with_corrections_batch(
            &divisions,
            &self.numerator,
            &mut self.quotients,
            &mut self.division_stable,
            &corrections,
            &mut self.corrections,
            &mut self.correction_stable,
        )?;
        let divided = &mut self.divided;
        divided.clear();
        divided.extend((0..active.len()).filter(|&i| {
            self.division_stable[i * orientations..(i + 1) * orientations]
                .iter()
                .all(|&s| s)
        }));
        self.raw.clear();
        self.selected_corrections.clear();
        let corrected = &mut self.corrected;
        corrected.clear();
        for &index in divided.iter() {
            let start = index * orientations * parent_count;
            let end = start + orientations * parent_count;
            if !self.correction_stable[start..end].iter().all(|&s| s) {
                continue;
            }
            corrected.push(index);
            self.raw.extend_from_slice(
                &self.quotients[index * orientations * width..(index + 1) * orientations * width],
            );
            self.selected_corrections
                .extend_from_slice(&self.corrections[start * width..end * width]);
        }
        self.combined.resize(self.raw.len(), self.zero.clone());
        self.kernels.combine_batch(
            &self.raw,
            &self.selected_corrections,
            corrected.len() * orientations,
            &mut self.combined,
        )?;
        self.projection.clear();
        let projected = &mut self.projected;
        projected.clear();
        let e12 = &mut self.e12;
        e12.clear();
        for (i, &index) in corrected.iter().enumerate() {
            let values = &self.combined[i * orientations * width..(i + 1) * orientations * width];
            if self.kernels.value_status(values).is_some() {
                continue;
            }
            projected.push(index);
            self.projection.extend_from_slice(values);
            e12.push(active_cut(index).basis.e12.clone());
        }
        let coefficient_count = self.kernels.coefficient_count();
        self.coefficients
            .resize(projected.len() * coefficient_count, self.zero.clone());
        self.kernels
            .project_batch(&self.projection, e12, &mut self.coefficients)?;
        for (i, &index) in projected.iter().enumerate() {
            let coefficients =
                &self.coefficients[i * coefficient_count..(i + 1) * coefficient_count];
            if self.kernels.value_status(coefficients).is_some() {
                continue;
            }
            let id = requests[active[index]].id;
            staged[eligible[active[index]]] = CutOutcome::Stable(IntegratedTadpoleResidue {
                cut_index: id.sample,
                id,
                geometry: active_cut(index).clone(),
                coefficients: self.coefficient_pool.take_from_slice(coefficients),
                computed: self
                    .mask_pool
                    .take_from_slice(self.kernels.computed_coefficients()),
            });
        }
        output.swap_with_slice(staged);
        Ok(())
    }
}
