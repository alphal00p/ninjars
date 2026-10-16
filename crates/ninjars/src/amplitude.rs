//! Integrated one-, two- and three-propagator amplitudes with lane-local accumulation.

use crate::{
    BatchShape, BubbleReducer, BubbleReductionKernels, BubbleResidue, CutBasis, CutFailure,
    CutOutcome, CutProvider, Error, FloatLike, FourMomentum, IntegratedContribution,
    IntegratedTadpoleResidue, IntegrationKernels, IntegrationStatus, MasterProvider, Numerator,
    ResidueIntegrationRequest, ResidueIntegrator, Result, SampleId, TadpoleReducer,
    TadpoleReductionKernels, Topology, TriangleReducer, TriangleReductionKernels, TriangleResidue,
};

/// Factory and generated arithmetic used by the integrated coordinator.
pub trait AmplitudeKernels<T> {
    type Triangles: TriangleReductionKernels<T>;
    type Bubbles: BubbleReductionKernels<T>;
    type Tadpoles: TadpoleReductionKernels<T>;
    type Integration: IntegrationKernels<T>;
    fn propagators(&self) -> usize;
    fn rank(&self) -> usize;
    fn triangle_kernels(&self) -> Result<Self::Triangles>;
    fn bubble_kernels(&self) -> Result<Self::Bubbles>;
    fn tadpole_kernels(&self) -> Result<Self::Tadpoles>;
    fn integration_kernels(&self, cut_size: usize) -> Result<Self::Integration>;
    /// Input rows: previous five components, then five components per topology
    /// cut in cached order. Output rows have the five accumulated components.
    fn accumulate_batch(&mut self, input: &[T], output: &mut [T], rows: usize) -> Result<()>;
    fn values_are_finite(&self, values: &[T]) -> bool;
    /// Compare complete carriers, including every derivative component.
    fn values_equal(&self, left: &T, right: &T) -> bool;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluationStatus {
    NotEvaluated,
    Success,
    CutFailed {
        cut_index: usize,
        failure: CutFailure,
    },
    IntegrationFailed {
        cut_index: usize,
        status: IntegrationStatus,
    },
    NonFiniteResult,
}

/// Ninja-style accumulation over a fixed phase-space batch.
///
/// This implementation currently accepts up to three propagators and rank n+1.
/// Calls add successful lanes to existing results. Every field of a failed lane
/// remains unchanged; its last-attempt status is recorded separately. Structural
/// errors leave results and last-attempt statuses unchanged. Shifts and masses
/// are lane-major; one positive real squared scale is supplied per lane.
/// Tadpole diagnostics contain only the coefficients needed for integration.
pub struct Amplitude<T, K: AmplitudeKernels<T>> {
    topology: Topology,
    lanes: usize,
    kernels: K,
    zero: T,
    triangle: Option<TriangleReducer<T, K::Triangles>>,
    bubble: Option<BubbleReducer<T, K::Bubbles>>,
    tadpole: Option<TadpoleReducer<T, K::Tadpoles>>,
    integration: [Option<ResidueIntegrator<T, K::Integration>>; 3],
    triangles: Vec<CutOutcome<TriangleResidue<T>>>,
    bubbles: Vec<CutOutcome<BubbleResidue<T>>>,
    tadpoles: Vec<CutOutcome<IntegratedTadpoleResidue<T>>>,
    results: Vec<IntegratedContribution<T>>,
    statuses: Vec<EvaluationStatus>,
    contributions: Vec<IntegratedContribution<T>>,
    input: Vec<T>,
    output: Vec<T>,
}

impl<T: FloatLike, K: AmplitudeKernels<T>> Amplitude<T, K> {
    pub fn new(kernels: K, prototype: T, lanes: usize) -> Result<Self> {
        let topology = Topology::new(kernels.propagators(), kernels.rank())?;
        if topology.propagators() > 3 {
            return Err(Error::Unsupported(
                "integrated reduction currently supports at most three propagators".into(),
            ));
        }
        let active = |size| {
            size <= topology.propagators()
                && topology.rank().saturating_add(size) >= topology.propagators()
        };
        let triangle = if active(3) {
            Some(TriangleReducer::new(
                kernels.triangle_kernels()?,
                prototype.clone(),
            )?)
        } else {
            None
        };
        let bubble = if active(2) {
            Some(BubbleReducer::new(
                kernels.bubble_kernels()?,
                prototype.clone(),
            )?)
        } else {
            None
        };
        let tadpole = if active(1) {
            Some(TadpoleReducer::new(
                kernels.tadpole_kernels()?,
                prototype.clone(),
            )?)
        } else {
            None
        };
        let mut integration = [None, None, None];
        for size in 1..=3 {
            if active(size) {
                integration[size - 1] = Some(ResidueIntegrator::new(
                    kernels.integration_kernels(size)?,
                    prototype.clone(),
                ));
            }
        }
        let zero = prototype.zero();
        Ok(Self {
            topology,
            lanes,
            kernels,
            triangle,
            bubble,
            tadpole,
            integration,
            triangles: Vec::new(),
            bubbles: Vec::new(),
            tadpoles: Vec::new(),
            results: vec![std::array::from_fn(|_| zero.clone()); lanes],
            statuses: vec![EvaluationStatus::NotEvaluated; lanes],
            contributions: Vec::new(),
            input: Vec::new(),
            output: Vec::new(),
            zero,
        })
    }

    pub fn topology(&self) -> &Topology {
        &self.topology
    }
    pub fn lanes(&self) -> usize {
        self.lanes
    }
    /// `[finite, single pole, double pole, cut-constructible finite, rational]`.
    pub fn results(&self) -> &[IntegratedContribution<T>] {
        &self.results
    }
    /// Laurent coefficients in original Ninja order: finite, single, double pole.
    pub fn laurent(&self, lane: usize) -> Option<&crate::Laurent<T>> {
        self.results.get(lane)?.first_chunk::<3>()
    }
    pub fn rational_part(&self, lane: usize) -> Option<&T> {
        self.results.get(lane).map(|value| &value[4])
    }
    pub fn cut_constructible_part(&self, lane: usize) -> Option<&T> {
        self.results.get(lane).map(|value| &value[3])
    }
    pub fn statuses(&self) -> &[EvaluationStatus] {
        &self.statuses
    }
    pub fn triangles(&self) -> &[CutOutcome<TriangleResidue<T>>] {
        &self.triangles
    }
    pub fn bubbles(&self) -> &[CutOutcome<BubbleResidue<T>>] {
        &self.bubbles
    }
    pub fn tadpoles(&self) -> &[CutOutcome<IntegratedTadpoleResidue<T>>] {
        &self.tadpoles
    }

    pub fn reset(&mut self) {
        self.reset_batch(self.lanes);
    }
    /// Explicitly discard accumulated values when changing the lane count.
    pub fn reset_batch(&mut self, lanes: usize) {
        self.lanes = lanes;
        self.results
            .resize_with(lanes, || std::array::from_fn(|_| self.zero.clone()));
        self.results
            .fill(std::array::from_fn(|_| self.zero.clone()));
        self.statuses.resize(lanes, EvaluationStatus::NotEvaluated);
        self.statuses.fill(EvaluationStatus::NotEvaluated);
        self.triangles.clear();
        self.bubbles.clear();
        self.tadpoles.clear();
    }

    pub fn evaluate_batch<N: Numerator<T>, C: CutProvider<T>, M: MasterProvider<T>>(
        &mut self,
        shifts: &[FourMomentum<T>],
        masses_squared: &[T],
        scales_squared: &[T],
        numerator: &mut N,
        geometry: &mut C,
        masters: &mut M,
    ) -> Result<()> {
        let n = self.topology.propagators();
        let lanes = self.lanes;
        BatchShape {
            rows: lanes,
            columns: n,
        }
        .validate("amplitude shifts", shifts.len())?;
        BatchShape {
            rows: lanes,
            columns: n,
        }
        .validate("amplitude masses", masses_squared.len())?;
        BatchShape {
            rows: lanes,
            columns: 1,
        }
        .validate("amplitude scales", scales_squared.len())?;
        if numerator.rank() > self.topology.rank() {
            return Err(Error::Unsupported(format!(
                "numerator rank {} exceeds configured rank {}",
                numerator.rank(),
                self.topology.rank()
            )));
        }
        for lane in 0..lanes {
            for first in 0..n {
                for second in first + 1..n {
                    let a = lane * n + first;
                    let b = lane * n + second;
                    if self
                        .kernels
                        .values_equal(&masses_squared[a], &masses_squared[b])
                        && shifts[a]
                            .components()
                            .into_iter()
                            .zip(shifts[b].components())
                            .all(|(x, y)| self.kernels.values_equal(x, y))
                    {
                        return Err(Error::Unsupported(
                            "raised or repeated loop propagators are not supported".into(),
                        ));
                    }
                }
            }
        }
        if let Some(stage) = &mut self.triangle {
            self.triangles.resize(
                stage.cut_count() * lanes,
                CutOutcome::Unstable(CutFailure::NonFiniteResult),
            );
            stage.evaluate_batch(
                shifts,
                masses_squared,
                lanes,
                numerator,
                geometry,
                &mut self.triangles,
            )?;
        }
        if let Some(stage) = &mut self.bubble {
            self.bubbles.resize(
                stage.cut_count() * lanes,
                CutOutcome::Unstable(CutFailure::NonFiniteResult),
            );
            stage.evaluate_batch(
                shifts,
                masses_squared,
                lanes,
                &self.topology,
                &self.triangles,
                numerator,
                geometry,
                &mut self.bubbles,
            )?;
        }
        if let Some(stage) = &mut self.tadpole {
            self.tadpoles.resize(
                stage.cut_count() * lanes,
                CutOutcome::Unstable(CutFailure::NonFiniteResult),
            );
            stage.evaluate_batch(
                shifts,
                masses_squared,
                lanes,
                &self.topology,
                &self.triangles,
                &self.topology,
                &self.bubbles,
                numerator,
                geometry,
                &mut self.tadpoles,
            )?;
        }
        let mut statuses = vec![EvaluationStatus::Success; lanes];
        let cuts = self.topology.cuts().len();
        self.contributions
            .resize_with(cuts * lanes, || std::array::from_fn(|_| self.zero.clone()));
        self.contributions
            .fill(std::array::from_fn(|_| self.zero.clone()));
        for size in (1..=3).rev() {
            let Some(integration) = &mut self.integration[size - 1] else {
                continue;
            };
            match size {
                3 => integrate_stage(
                    &self.topology,
                    lanes,
                    shifts,
                    masses_squared,
                    scales_squared,
                    &self.triangles,
                    integration,
                    masters,
                    &self.zero,
                    &mut self.contributions,
                    &mut statuses,
                )?,
                2 => integrate_stage(
                    &self.topology,
                    lanes,
                    shifts,
                    masses_squared,
                    scales_squared,
                    &self.bubbles,
                    integration,
                    masters,
                    &self.zero,
                    &mut self.contributions,
                    &mut statuses,
                )?,
                _ => integrate_stage(
                    &self.topology,
                    lanes,
                    shifts,
                    masses_squared,
                    scales_squared,
                    &self.tadpoles,
                    integration,
                    masters,
                    &self.zero,
                    &mut self.contributions,
                    &mut statuses,
                )?,
            }
        }
        let active: Vec<_> = statuses
            .iter()
            .enumerate()
            .filter_map(|(i, s)| (*s == EvaluationStatus::Success).then_some(i))
            .collect();
        self.input.clear();
        for &lane in &active {
            self.input.extend_from_slice(&self.results[lane]);
            for cut in 0..cuts {
                self.input
                    .extend_from_slice(&self.contributions[cut * lanes + lane]);
            }
        }
        self.output.resize(active.len() * 5, self.zero.clone());
        self.kernels
            .accumulate_batch(&self.input, &mut self.output, active.len())?;
        for (index, &lane) in active.iter().enumerate() {
            let row = &self.output[index * 5..(index + 1) * 5];
            if self.kernels.values_are_finite(row) {
                self.results[lane].clone_from_slice(row);
            } else {
                statuses[lane] = EvaluationStatus::NonFiniteResult;
            }
        }
        self.statuses = statuses;
        Ok(())
    }
}

trait ResidueView<T> {
    fn id(&self) -> SampleId;
    fn index(&self) -> usize;
    fn coefficients(&self) -> &[T];
    fn basis(&self) -> &CutBasis<T>;
}
macro_rules! residue_view {
    ($name:ident) => {
        impl<T> ResidueView<T> for $name<T> {
            fn id(&self) -> SampleId {
                self.id
            }
            fn index(&self) -> usize {
                self.cut_index
            }
            fn coefficients(&self) -> &[T] {
                &self.coefficients
            }
            fn basis(&self) -> &CutBasis<T> {
                &self.geometry.basis
            }
        }
    };
}
residue_view!(TriangleResidue);
residue_view!(BubbleResidue);
residue_view!(IntegratedTadpoleResidue);

#[allow(clippy::too_many_arguments)]
fn integrate_stage<
    T: FloatLike,
    K: IntegrationKernels<T>,
    M: MasterProvider<T>,
    R: ResidueView<T>,
>(
    topology: &Topology,
    lanes: usize,
    shifts: &[FourMomentum<T>],
    masses: &[T],
    scales: &[T],
    residues: &[CutOutcome<R>],
    integration: &mut ResidueIntegrator<T, K>,
    provider: &mut M,
    zero: &T,
    contributions: &mut [IntegratedContribution<T>],
    statuses: &mut [EvaluationStatus],
) -> Result<()> {
    let n = topology.propagators();
    let cuts: Vec<_> = topology
        .cuts()
        .iter()
        .enumerate()
        .filter_map(|(i, c)| (c.partition.len() == integration.cut_size()).then_some(i))
        .collect();
    let mut requests = Vec::new();
    for (row, outcome) in residues.iter().enumerate() {
        let lane = row % lanes;
        match outcome {
            CutOutcome::Unstable(failure) => {
                if statuses[lane] == EvaluationStatus::Success {
                    statuses[lane] = EvaluationStatus::CutFailed {
                        cut_index: cuts[row / lanes],
                        failure: *failure,
                    };
                }
            }
            CutOutcome::Stable(residue) if statuses[lane] == EvaluationStatus::Success => requests
                .push(ResidueIntegrationRequest {
                    id: residue.id(),
                    partition: &topology.cuts()[residue.index()].partition,
                    shifts: &shifts[lane * n..(lane + 1) * n],
                    masses_squared: &masses[lane * n..(lane + 1) * n],
                    coefficients: residue.coefficients(),
                    basis: residue.basis(),
                    scale_squared: &scales[lane],
                }),
            _ => (),
        }
    }
    let mut values = vec![std::array::from_fn(|_| zero.clone()); requests.len()];
    let mut integration_status = vec![IntegrationStatus::InvalidInput; requests.len()];
    integration.evaluate_batch(&requests, provider, &mut values, &mut integration_status)?;
    for ((request, value), status) in requests.iter().zip(values).zip(integration_status) {
        if status == IntegrationStatus::Success {
            contributions[request.id.sample * lanes + request.id.lane] = value;
        } else if statuses[request.id.lane] == EvaluationStatus::Success {
            statuses[request.id.lane] = EvaluationStatus::IntegrationFailed {
                cut_index: request.id.sample,
                status,
            };
        }
    }
    Ok(())
}
