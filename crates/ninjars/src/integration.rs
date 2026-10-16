//! Batched contraction of reduced residues with scalar and tensor masters.

use crate::{
    BatchShape, CutBasis, Error, FloatLike, FourMomentum, Laurent, MasterFailure, MasterIntegral,
    MasterProvider, MasterRequest, MasterStatus, Result, SampleId,
};

/// A reduced residue and the physical propagators used to integrate it.
/// All numerical slices refer to one phase-space lane. The partition retains
/// original denominator order; the basis is the one used to extract coefficients.
pub struct ResidueIntegrationRequest<'a, T> {
    pub id: SampleId,
    pub partition: &'a [usize],
    pub shifts: &'a [FourMomentum<T>],
    pub masses_squared: &'a [T],
    pub coefficients: &'a [T],
    /// Required for cuts of size 1–3; box contraction is independent of a basis.
    pub basis: Option<&'a CutBasis<T>>,
    pub scale_squared: &'a T,
    /// Optional row-major n×n invariants (V_i−V_j)² for this lane.
    /// Values can retain exact known zeros lost in rounded momentum components.
    pub s_matrix: Option<&'a [T]>,
}

impl<T> ResidueIntegrationRequest<'_, T> {
    pub fn validate(&self, cut_size: usize, coefficients: usize) -> Result<()> {
        validate_request(self, cut_size, coefficients)
    }
}

/// Finite, single pole, double pole, cut-constructible finite, explicit rational.
pub type IntegratedContribution<T> = [T; 5];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegrationStatus {
    Success,
    InvalidInput,
    MasterFailed {
        integral: MasterIntegral,
        failure: MasterFailure,
    },
    NonFiniteResult,
}

/// Arithmetic supplied by an algebra backend. Requests in a batch all have the
/// same cut size and rank layout. Coefficients retain original Ninja indexing.
pub trait IntegrationKernels<T> {
    fn cut_size(&self) -> usize;
    fn coefficient_count(&self) -> usize;
    fn masters(&self) -> &[MasterIntegral];
    fn invariant_count(&self) -> usize;
    /// Return row-major invariants and `[master weights..., rational]` rows.
    /// Invalid numerical rows leave numerical outputs unchanged and set false.
    fn prepare_batch(
        &mut self,
        requests: &[ResidueIntegrationRequest<'_, T>],
        invariants: &mut [T],
        weights: &mut [T],
        valid: &mut [bool],
    ) -> Result<()>;
    fn contract_batch(
        &mut self,
        weights: &[T],
        masters: &[Laurent<T>],
        output: &mut [IntegratedContribution<T>],
    ) -> Result<()>;
    fn value_is_zero(&self, value: &T) -> bool;
    fn values_are_finite(&self, values: &[T]) -> bool;
}

/// Integrates one cut family across all cuts and phase-space lanes in a batch.
/// Each master family receives one compact batch, including derivative-only
/// weights. A failed row leaves its output unchanged; structural errors leave
/// every caller output and status unchanged. This stage does not sum cuts.
pub struct ResidueIntegrator<T, K> {
    kernels: K,
    zero: T,
    invariants: Vec<T>,
    weights: Vec<T>,
    master_values: Vec<Laurent<T>>,
    valid: Vec<bool>,
    indices: Vec<usize>,
    arguments: Vec<T>,
    masses: Vec<T>,
    scales: Vec<T>,
    results: Vec<Laurent<T>>,
    master_status: Vec<MasterStatus>,
    staged_status: Vec<IntegrationStatus>,
    active_weights: Vec<T>,
    active_masters: Vec<Laurent<T>>,
    contributions: Vec<IntegratedContribution<T>>,
}

impl<T: FloatLike, K: IntegrationKernels<T>> ResidueIntegrator<T, K> {
    pub fn new(kernels: K, prototype: T) -> Self {
        Self {
            kernels,
            zero: prototype.zero(),
            invariants: Vec::new(),
            weights: Vec::new(),
            master_values: Vec::new(),
            valid: Vec::new(),
            indices: Vec::new(),
            arguments: Vec::new(),
            masses: Vec::new(),
            scales: Vec::new(),
            results: Vec::new(),
            master_status: Vec::new(),
            staged_status: Vec::new(),
            active_weights: Vec::new(),
            active_masters: Vec::new(),
            contributions: Vec::new(),
        }
    }

    pub fn cut_size(&self) -> usize {
        self.kernels.cut_size()
    }

    pub fn evaluate_batch<M: MasterProvider<T>>(
        &mut self,
        requests: &[ResidueIntegrationRequest<'_, T>],
        provider: &mut M,
        output: &mut [IntegratedContribution<T>],
        statuses: &mut [IntegrationStatus],
    ) -> Result<()> {
        let rows = requests.len();
        BatchShape { rows, columns: 1 }.validate("integrated residues", output.len())?;
        BatchShape { rows, columns: 1 }.validate("integration statuses", statuses.len())?;
        for request in requests {
            validate_request(request, self.cut_size(), self.kernels.coefficient_count())?;
        }
        let invariants = self.kernels.invariant_count();
        let families = self.kernels.masters().to_vec();
        let masters = families.len();
        let columns = masters + 1;
        self.invariants.resize(rows * invariants, self.zero.clone());
        self.weights.resize(rows * columns, self.zero.clone());
        self.valid.resize(rows, false);
        self.kernels.prepare_batch(
            requests,
            &mut self.invariants,
            &mut self.weights,
            &mut self.valid,
        )?;
        self.staged_status.clear();
        self.staged_status.extend(self.valid.iter().map(|&valid| {
            if valid {
                IntegrationStatus::Success
            } else {
                IntegrationStatus::InvalidInput
            }
        }));
        let zero_laurent = std::array::from_fn(|_| self.zero.clone());
        self.master_values
            .resize(rows * masters, zero_laurent.clone());
        self.master_values.fill(zero_laurent.clone());
        for (master_index, &integral) in families.iter().enumerate() {
            self.indices.clear();
            self.arguments.clear();
            self.masses.clear();
            self.scales.clear();
            for (row, request) in requests.iter().enumerate() {
                if !self.valid[row]
                    || self
                        .kernels
                        .value_is_zero(&self.weights[row * columns + master_index])
                {
                    continue;
                }
                self.indices.push(row);
                self.arguments
                    .extend_from_slice(&self.invariants[row * invariants..(row + 1) * invariants]);
                self.masses.extend(
                    request
                        .partition
                        .iter()
                        .map(|&i| request.masses_squared[i].clone()),
                );
                self.scales.push(request.scale_squared.clone());
            }
            if self.indices.is_empty() {
                continue;
            }
            self.results
                .resize(self.indices.len(), zero_laurent.clone());
            self.master_status
                .resize(self.indices.len(), MasterStatus::Success);
            provider.evaluate_batch(
                MasterRequest {
                    integral,
                    invariants: &self.arguments,
                    masses_squared: &self.masses,
                    scales_squared: &self.scales,
                },
                &mut self.results,
                &mut self.master_status,
            )?;
            for (index, &row) in self.indices.iter().enumerate() {
                match &self.master_status[index] {
                    MasterStatus::Success
                        if self.kernels.values_are_finite(&self.results[index]) =>
                    {
                        self.master_values[row * masters + master_index]
                            .clone_from(&self.results[index]);
                    }
                    status => {
                        self.valid[row] = false;
                        let failure = match status {
                            MasterStatus::Failed(failure) => *failure,
                            MasterStatus::Success => MasterFailure::NonFiniteResult,
                        };
                        self.staged_status[row] =
                            IntegrationStatus::MasterFailed { integral, failure };
                    }
                }
            }
        }
        self.indices.clear();
        self.active_weights.clear();
        self.active_masters.clear();
        for row in 0..rows {
            if self.valid[row] {
                self.indices.push(row);
                self.active_weights
                    .extend_from_slice(&self.weights[row * columns..(row + 1) * columns]);
                self.active_masters
                    .extend_from_slice(&self.master_values[row * masters..(row + 1) * masters]);
            }
        }
        self.contributions.resize_with(self.indices.len(), || {
            std::array::from_fn(|_| self.zero.clone())
        });
        self.kernels.contract_batch(
            &self.active_weights,
            &self.active_masters,
            &mut self.contributions,
        )?;
        for (index, &row) in self.indices.iter().enumerate() {
            if self.kernels.values_are_finite(&self.contributions[index]) {
                output[row].clone_from(&self.contributions[index]);
            } else {
                self.staged_status[row] = IntegrationStatus::NonFiniteResult;
            }
        }
        statuses.clone_from_slice(&self.staged_status);
        Ok(())
    }
}

pub(crate) fn validate_request<T>(
    request: &ResidueIntegrationRequest<'_, T>,
    cut_size: usize,
    coefficients: usize,
) -> Result<()> {
    if let Some(matrix) = request.s_matrix {
        BatchShape {
            rows: request.shifts.len(),
            columns: request.shifts.len(),
        }
        .validate("integration S-matrix", matrix.len())?;
    }
    if cut_size <= 3 && request.basis.is_none() {
        return Err(Error::Expansion(
            "cuts of size 1–3 require their extraction basis".into(),
        ));
    }
    if request.partition.len() != cut_size
        || request.shifts.len() != request.masses_squared.len()
        || request.partition.iter().any(|&i| i >= request.shifts.len())
        || request.partition.windows(2).any(|p| p[0] >= p[1])
    {
        return Err(Error::Expansion(
            "integration partition does not match its ordered propagators".into(),
        ));
    }
    BatchShape {
        rows: 1,
        columns: coefficients,
    }
    .validate("integration coefficients", request.coefficients.len())
}
