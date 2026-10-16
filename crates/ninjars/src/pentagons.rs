//! Batched five-cut residues. Pentagons have no direct scalar-master contribution.

use crate::{
    BatchShape, CutFailure, CutOutcome, Error, FloatLike, FourMomentum, HighCutProvider, Numerator,
    NumeratorRequest, PentagonCut, PentagonCutRequest, Result, SampleId, Topology,
};

pub struct PentagonDivisionRequest<'a, T> {
    pub partition: &'a [usize],
    pub complement: &'a [usize],
    pub shifts: &'a [FourMomentum<T>],
    pub masses_squared: &'a [T],
    pub geometry: &'a PentagonCut<T>,
}

/// Algebra backend for c0 = N(q,mu²)/(mu² product(D_uncut)).
pub trait PentagonReductionKernels<T> {
    fn propagators(&self) -> usize;
    fn rank(&self) -> usize;
    fn divide_batch(
        &mut self,
        requests: &[PentagonDivisionRequest<'_, T>],
        numerator: &[T],
        output: &mut [T],
        failures: &mut [Option<CutFailure>],
    ) -> Result<()>;
}

#[derive(Clone, Debug)]
pub struct PentagonResidue<T> {
    pub cut_index: usize,
    pub id: SampleId,
    pub geometry: PentagonCut<T>,
    /// The five-cut residue is coefficient * mu².
    pub coefficient: T,
}

/// Produces all five-cut residues in cut-major, lane-minor order.
/// Numerical failures are isolated; structural errors preserve caller output.
pub struct PentagonReducer<T, K> {
    topology: Topology,
    cuts: Vec<usize>,
    kernels: K,
    zero: T,
    numerator: Vec<T>,
    coefficients: Vec<T>,
    failures: Vec<Option<CutFailure>>,
}

impl<T: FloatLike, K: PentagonReductionKernels<T>> PentagonReducer<T, K> {
    pub fn new(kernels: K, prototype: T) -> Result<Self> {
        if kernels.propagators() < 5 {
            return Err(Error::Unsupported(
                "pentagon reduction needs at least five propagators".into(),
            ));
        }
        let topology = Topology::new(kernels.propagators(), kernels.rank())?;
        let cuts = topology
            .cuts()
            .iter()
            .enumerate()
            .filter_map(|(index, cut)| (cut.partition.len() == 5).then_some(index))
            .collect();
        Ok(Self {
            topology,
            cuts,
            kernels,
            zero: prototype.zero(),
            numerator: Vec::new(),
            coefficients: Vec::new(),
            failures: Vec::new(),
        })
    }
    pub fn topology(&self) -> &Topology {
        &self.topology
    }
    pub fn cut_count(&self) -> usize {
        self.cuts.len()
    }

    pub fn evaluate_batch<N: Numerator<T>, C: HighCutProvider<T>>(
        &mut self,
        shifts: &[FourMomentum<T>],
        masses_squared: &[T],
        lanes: usize,
        numerator: &mut N,
        geometry: &mut C,
        output: &mut [CutOutcome<PentagonResidue<T>>],
    ) -> Result<()> {
        let n = self.topology.propagators();
        BatchShape {
            rows: lanes,
            columns: n,
        }
        .validate("pentagon shifts", shifts.len())?;
        BatchShape {
            rows: lanes,
            columns: n,
        }
        .validate("pentagon masses", masses_squared.len())?;
        BatchShape {
            rows: lanes,
            columns: self.cuts.len(),
        }
        .validate("pentagon residues", output.len())?;
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
                requests.push(PentagonCutRequest {
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
        let mut cuts = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); output.len()];
        geometry.pentagon_batch(&requests, &mut cuts)?;
        let active: Vec<_> = cuts
            .iter()
            .enumerate()
            .filter_map(|(row, cut)| matches!(cut, CutOutcome::Stable(_)).then_some(row))
            .collect();
        let mut callbacks = Vec::with_capacity(active.len());
        let mut divisions = Vec::with_capacity(active.len());
        for &row in &active {
            let CutOutcome::Stable(cut) = &cuts[row] else {
                unreachable!()
            };
            let id = requests[row].id;
            let topology_cut = &self.topology.cuts()[id.sample];
            callbacks.push(NumeratorRequest {
                id,
                q: &cut.q,
                mu_squared: &cut.mu_squared,
                partition: &topology_cut.partition,
            });
            divisions.push(PentagonDivisionRequest {
                partition: &topology_cut.partition,
                complement: &topology_cut.complement,
                shifts: &shifts[id.lane * n..(id.lane + 1) * n],
                masses_squared: &masses_squared[id.lane * n..(id.lane + 1) * n],
                geometry: cut,
            });
        }
        self.numerator.resize(active.len(), self.zero.clone());
        self.coefficients.resize(active.len(), self.zero.clone());
        self.failures.resize(active.len(), None);
        numerator.evaluate_batch(&callbacks, &mut self.numerator)?;
        self.kernels.divide_batch(
            &divisions,
            &self.numerator,
            &mut self.coefficients,
            &mut self.failures,
        )?;
        let mut staged: Vec<_> = cuts
            .iter()
            .map(|cut| match cut {
                CutOutcome::Unstable(failure) => CutOutcome::Unstable(*failure),
                CutOutcome::Stable(_) => CutOutcome::Unstable(CutFailure::NonFiniteResult),
            })
            .collect();
        for (index, &row) in active.iter().enumerate() {
            if let Some(failure) = self.failures[index] {
                staged[row] = CutOutcome::Unstable(failure);
            } else {
                let CutOutcome::Stable(cut) = &cuts[row] else {
                    unreachable!()
                };
                let id = requests[row].id;
                staged[row] = CutOutcome::Stable(PentagonResidue {
                    cut_index: id.sample,
                    id,
                    geometry: cut.clone(),
                    coefficient: self.coefficients[index].clone(),
                });
            }
        }
        output.clone_from_slice(&staged);
        Ok(())
    }
}
