//! Batched triple-cut reduction, independent of numerator preparation.

use crate::{
    BatchShape, CutFailure, CutOutcome, CutProvider, Error, ExpansionKind, ExpansionRequest,
    FloatLike, FourMomentum, Numerator, Result, SampleId, Topology, TriangleCut,
    TriangleCutRequest,
};

/// One orientation of a triple cut and its uncut denominators.
pub struct TriangleDivisionRequest<'a, T> {
    pub partition: &'a [usize],
    pub complement: &'a [usize],
    pub shifts: &'a [FourMomentum<T>],
    pub masses_squared: &'a [T],
    pub vectors: &'a [FourMomentum<T>; 3],
    pub beta: &'a T,
}

/// Numerical kernels supplied by the Symbolica adapter. The coordinator only
/// packs requests and restores their cut/lane order; all arithmetic is batched.
pub trait TriangleReductionKernels<T> {
    fn propagators(&self) -> usize;
    fn rank(&self) -> usize;
    fn numerator_expansion(&self) -> &ExpansionKind;
    fn quotient_columns(&self) -> usize;
    fn coefficient_count(&self) -> usize;
    fn swapped_effective_batch(
        &mut self,
        cuts: &[&TriangleCut<T>],
        output: &mut [FourMomentum<T>],
    ) -> Result<()>;
    fn divide_batch(
        &mut self,
        requests: &[TriangleDivisionRequest<'_, T>],
        numerator: &[T],
        output: &mut [T],
        stable: &mut [bool],
    ) -> Result<()>;
    /// Quotient rows contain both orientations consecutively.
    fn project_batch(&mut self, quotients: &[T], e12: &[T], output: &mut [T]) -> Result<()>;
    fn values_are_finite(&self, values: &[T]) -> bool;
}

#[derive(Clone, Debug)]
pub struct TriangleResidue<T> {
    /// Index in the cached topology's complete cut list.
    pub cut_index: usize,
    pub id: SampleId,
    pub geometry: TriangleCut<T>,
    /// Original Ninja coefficient order (10 ordinary or 15 higher-rank entries).
    pub coefficients: Vec<T>,
}

/// Triple-cut stage of Laurent reduction, for any topology with at least 3 lines.
///
/// This produces triangle residues, not an integrated amplitude. Bubble and
/// tadpole subtractions and master contraction are subsequent stages. Input
/// shifts and masses are lane-major with `propagators()` entries per lane;
/// output order is triangle-cut-major, then phase-space lane. Numerical failures
/// are isolated per cut/lane. An API error leaves the caller's output unchanged.
pub struct TriangleReducer<T, K> {
    topology: Topology,
    cuts: Vec<usize>,
    kernels: K,
    zero: T,
    numerator_values: Vec<T>,
    quotients: Vec<T>,
    coefficients: Vec<T>,
    stable: Vec<bool>,
    geometry_output: Vec<CutOutcome<TriangleCut<T>>>,
    active: Vec<usize>,
    staged: Vec<CutOutcome<TriangleResidue<T>>>,
    swapped: Vec<FourMomentum<T>>,
    vectors: Vec<[FourMomentum<T>; 3]>,
    e12: Vec<T>,
}

impl<T: FloatLike, K: TriangleReductionKernels<T>> TriangleReducer<T, K> {
    pub fn new(kernels: K, prototype: T) -> Result<Self> {
        if kernels.propagators() < 3 {
            return Err(Error::Unsupported(
                "triangle reduction needs at least three propagators".into(),
            ));
        }
        let topology = Topology::new(kernels.propagators(), kernels.rank())?;
        let cuts = topology
            .cuts()
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                (c.partition.len() == 3
                    && topology.rank().saturating_add(3) >= topology.propagators())
                .then_some(i)
            })
            .collect();
        Ok(Self {
            topology,
            cuts,
            kernels,
            zero: prototype.zero(),
            geometry_output: Vec::new(),
            active: Vec::new(),
            staged: Vec::new(),
            swapped: Vec::new(),
            vectors: Vec::new(),
            e12: Vec::new(),
            numerator_values: Vec::new(),
            quotients: Vec::new(),
            coefficients: Vec::new(),
            stable: Vec::new(),
        })
    }
    pub fn topology(&self) -> &Topology {
        &self.topology
    }
    pub fn cut_count(&self) -> usize {
        self.cuts.len()
    }

    pub fn evaluate_batch<N: Numerator<T>, C: CutProvider<T>>(
        &mut self,
        shifts: &[FourMomentum<T>],
        masses_squared: &[T],
        lanes: usize,
        numerator: &mut N,
        geometry: &mut C,
        output: &mut [CutOutcome<TriangleResidue<T>>],
    ) -> Result<()> {
        let n = self.topology.propagators();
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
        .validate("triangle residues", output.len())?;
        if numerator.rank() > self.topology.rank() {
            return Err(Error::Unsupported(format!(
                "numerator rank {} exceeds configured rank {}",
                numerator.rank(),
                self.topology.rank()
            )));
        }
        if output.is_empty() {
            return Ok(());
        }
        let mut requests = Vec::with_capacity(output.len());
        for &cut_index in &self.cuts {
            let partition = &self.topology.cuts()[cut_index].partition;
            for lane in 0..lanes {
                requests.push(TriangleCutRequest {
                    id: SampleId {
                        lane,
                        sample: cut_index,
                    },
                    shifts: std::array::from_fn(|i| &shifts[lane * n + partition[i]]),
                    masses_squared: std::array::from_fn(|i| {
                        &masses_squared[lane * n + partition[i]]
                    }),
                });
            }
        }
        let cuts = &mut self.geometry_output;
        cuts.resize(
            output.len(),
            CutOutcome::Unstable(CutFailure::NonFiniteResult),
        );
        cuts.fill(CutOutcome::Unstable(CutFailure::NonFiniteResult));
        geometry.triangle_batch(&requests, cuts)?;
        let active = &mut self.active;
        active.clear();
        active.extend(
            cuts.iter()
                .enumerate()
                .filter_map(|(i, c)| matches!(c, CutOutcome::Stable(_)).then_some(i)),
        );
        let active_cuts: Vec<_> = active
            .iter()
            .map(|&i| {
                let CutOutcome::Stable(c) = &cuts[i] else {
                    unreachable!()
                };
                c
            })
            .collect();
        let swapped = &mut self.swapped;
        swapped.resize_with(active.len(), || {
            FourMomentum::from_args(
                self.zero.clone(),
                self.zero.clone(),
                self.zero.clone(),
                self.zero.clone(),
            )
        });
        self.kernels
            .swapped_effective_batch(&active_cuts, swapped)?;
        let vectors = &mut self.vectors;
        vectors.clear();
        for (cut, effective) in active_cuts.iter().zip(swapped.drain(..)) {
            vectors.push([
                cut.a.clone(),
                cut.basis.e3.clone(),
                cut.e4_effective.clone(),
            ]);
            vectors.push([cut.a.clone(), cut.basis.e4.clone(), effective]);
        }
        let mut expansions = Vec::with_capacity(vectors.len());
        let mut divisions = Vec::with_capacity(vectors.len());
        for (index, &row) in active.iter().enumerate() {
            let id = requests[row].id;
            let cut = &self.topology.cuts()[id.sample];
            for orientation in 0..2 {
                expansions.push(ExpansionRequest {
                    id: SampleId {
                        lane: id.lane,
                        sample: 2 * id.sample + orientation,
                    },
                    partition: &cut.partition,
                    vectors: &vectors[2 * index + orientation],
                    parameters: std::slice::from_ref(&active_cuts[index].beta),
                });
                divisions.push(TriangleDivisionRequest {
                    partition: &cut.partition,
                    complement: &cut.complement,
                    shifts: &shifts[id.lane * n..(id.lane + 1) * n],
                    masses_squared: &masses_squared[id.lane * n..(id.lane + 1) * n],
                    vectors: &vectors[2 * index + orientation],
                    beta: &active_cuts[index].beta,
                });
            }
        }
        let kind = self.kernels.numerator_expansion();
        self.numerator_values
            .resize(expansions.len() * kind.len(), self.zero.clone());
        numerator.expansion_batch(kind, &expansions, &mut self.numerator_values)?;
        let width = self.kernels.quotient_columns();
        self.quotients
            .resize(expansions.len() * width, self.zero.clone());
        self.stable.resize(expansions.len(), false);
        self.kernels.divide_batch(
            &divisions,
            &self.numerator_values,
            &mut self.quotients,
            &mut self.stable,
        )?;
        let e12 = &mut self.e12;
        e12.clear();
        e12.extend(active_cuts.iter().map(|c| c.basis.e12.clone()));
        let coefficients = self.kernels.coefficient_count();
        self.coefficients
            .resize(active.len() * coefficients, self.zero.clone());
        self.kernels
            .project_batch(&self.quotients, e12, &mut self.coefficients)?;
        let staged = &mut self.staged;
        staged.clear();
        staged.extend(cuts.iter().map(|c| match c {
            CutOutcome::Unstable(f) => CutOutcome::Unstable(*f),
            CutOutcome::Stable(_) => CutOutcome::Unstable(CutFailure::NonFiniteResult),
        }));
        for (index, &row) in active.iter().enumerate() {
            let values = &self.coefficients[index * coefficients..(index + 1) * coefficients];
            if self.stable[2 * index]
                && self.stable[2 * index + 1]
                && self.kernels.values_are_finite(values)
            {
                staged[row] = CutOutcome::Stable(TriangleResidue {
                    cut_index: requests[row].id.sample,
                    id: requests[row].id,
                    geometry: (*active_cuts[index]).clone(),
                    coefficients: values.to_vec(),
                });
            }
        }
        output.swap_with_slice(staged);
        Ok(())
    }
}
