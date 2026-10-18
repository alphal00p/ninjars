//! Batched quadruple-cut reduction with both rational extraction modes.

use crate::residue_storage::VecPool;
use crate::{
    BatchShape, BoxCut, BoxCutRequest, BoxSampleRequest, CutFailure, CutOutcome, Error,
    ExpansionKind, ExpansionRequest, FloatLike, FourMomentum, HighCutProvider, Numerator,
    NumeratorRequest, PentagonResidue, Result, SampleId, Topology,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoxCoefficientMode {
    Integrated,
    Full,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoxMuMode {
    Expansion,
    Sampling,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoxReductionOptions {
    pub coefficients: BoxCoefficientMode,
    pub mu_extraction: BoxMuMode,
}
impl BoxReductionOptions {
    /// mu² levels sampled per box: 0, +s, -s and, at rank n+2 in sampling
    /// mode, 2s (the mu⁶ coefficient couples to mu² in the even part).
    pub fn sample_count(self, n: usize, rank: usize) -> usize {
        let sampled = if rank > n.saturating_add(1) { 4 } else { 3 };
        match (self.coefficients, self.mu_extraction) {
            (BoxCoefficientMode::Full, BoxMuMode::Sampling) => sampled,
            (BoxCoefficientMode::Full, BoxMuMode::Expansion) => 2,
            (BoxCoefficientMode::Integrated, BoxMuMode::Sampling) if rank >= n => sampled,
            _ => 1,
        }
    }
    pub fn needs_pentagons(self, n: usize, rank: usize) -> bool {
        n > 4 && self.sample_count(n, rank) > 1
    }
}

pub struct BoxDivisionRequest<'a, T> {
    pub partition: &'a [usize],
    pub complement: &'a [usize],
    pub shifts: &'a [FourMomentum<T>],
    pub masses_squared: &'a [T],
    pub momentum: &'a FourMomentum<T>,
    pub mu_squared: &'a T,
    /// One coefficient per complementary line: the containing pentagon is
    /// the box partition together with that line. None is valid at mu²=0.
    pub pentagon_coefficients: Option<&'a [T]>,
}
pub struct BoxMuDivisionRequest<'a, T> {
    pub partition: &'a [usize],
    pub complement: &'a [usize],
    pub shifts: &'a [FourMomentum<T>],
    pub masses_squared: &'a [T],
    pub geometry: &'a BoxCut<T>,
}
pub struct BoxProjectionRequest<'a, T> {
    pub geometry: &'a BoxCut<T>,
    pub shift: &'a FourMomentum<T>,
    /// Pairs ordered mu²=0,+s,-s,+2s, truncated to the selected mode's count.
    pub momenta: &'a [[FourMomentum<T>; 2]],
    pub quotients: &'a [[T; 2]],
    pub sampling_scale: &'a T,
    /// c4,c5,c6 from the mu expansion, or zero when sampling is selected.
    /// c6 (mu⁶) is nonzero only at rank n+2.
    pub mu_coefficients: &'a [T; 3],
}

pub trait BoxReductionKernels<T> {
    fn propagators(&self) -> usize;
    fn rank(&self) -> usize;
    fn options(&self) -> BoxReductionOptions;
    fn numerator_mu_expansion(&self) -> Option<&ExpansionKind>;
    fn coefficient_count(&self) -> usize;
    fn sampling_mu_batch(&mut self, scales: &[T], output: &mut [T]) -> Result<()>;
    fn divide_samples_batch(
        &mut self,
        requests: &[BoxDivisionRequest<'_, T>],
        numerator: &[T],
        output: &mut [T],
        stable: &mut [bool],
    ) -> Result<()>;
    fn divide_mu_batch(
        &mut self,
        requests: &[BoxMuDivisionRequest<'_, T>],
        numerator: &[T],
        output: &mut [[T; 3]],
        stable: &mut [bool],
    ) -> Result<()>;
    fn project_batch(
        &mut self,
        requests: &[BoxProjectionRequest<'_, T>],
        output: &mut [T],
        stable: &mut [bool],
    ) -> Result<()>;
    fn values_are_finite(&self, values: &[T]) -> bool;
}

#[derive(Clone, Debug)]
pub struct BoxResidue<T> {
    pub cut_index: usize,
    pub id: SampleId,
    pub geometry: BoxCut<T>,
    /// Original Ninja order: c0+c2*mu²+c4*mu⁴+(c1+c3*mu²+c5*mu⁴)*u, plus
    /// c6*mu⁶ at rank n+2.
    pub coefficients: Vec<T>,
    /// Integrated mode computes c0/c4 (and c6 at rank n+2); other slots are
    /// explicitly uncomputed.
    pub computed: Vec<bool>,
}

/// Every box is retained, even when rank+4<n: scalar pentagons and hexagons
/// can have nonzero box coefficients. Output is cut-major, lane-minor.
pub struct BoxReducer<T, K> {
    topology: Topology,
    cuts: Vec<usize>,
    pentagons: Vec<usize>,
    parents: Vec<Vec<usize>>,
    kernels: K,
    zero: T,
    numerator: Vec<T>,
    quotients: Vec<T>,
    coefficients: Vec<T>,
    staged: Vec<CutOutcome<BoxResidue<T>>>,
    coefficient_pool: VecPool<T>,
    mask_pool: VecPool<bool>,
    rows: Vec<usize>,
    geometry_output: Vec<CutOutcome<BoxCut<T>>>,
    active: Vec<usize>,
    cut_geometries: Vec<BoxCut<T>>,
    scales: Vec<T>,
    mus: Vec<T>,
    samples: Vec<CutOutcome<[FourMomentum<T>; 2]>>,
    viable: Vec<usize>,
    parents_flat: Vec<T>,
    division_stable: Vec<bool>,
    final_rows: Vec<(usize, usize)>,
    mu_coefficients: Vec<[T; 3]>,
    mu_stable: Vec<bool>,
    vectors: Vec<[FourMomentum<T>; 2]>,
    mu_values: Vec<T>,
    projection_pairs: Vec<[FourMomentum<T>; 2]>,
    projection_rows: Vec<usize>,
    projection_stable: Vec<bool>,
    computed: Vec<bool>,
}
impl<T: FloatLike, K: BoxReductionKernels<T>> BoxReducer<T, K> {
    pub fn new(kernels: K, prototype: T) -> Result<Self> {
        if kernels.propagators() < 4 {
            return Err(Error::Unsupported(
                "box reduction requires at least four propagators".into(),
            ));
        }
        let topology = Topology::new(kernels.propagators(), kernels.rank())?;
        let indices = |size| {
            topology
                .cuts()
                .iter()
                .enumerate()
                .filter_map(|(i, c)| (c.partition.len() == size).then_some(i))
                .collect::<Vec<_>>()
        };
        let cuts = indices(4);
        let pentagons = indices(5);
        let parents = cuts
            .iter()
            .map(|&cut| {
                topology.cuts()[cut]
                    .complement
                    .iter()
                    .map(|extra| {
                        pentagons
                            .iter()
                            .position(|&parent| {
                                let p = &topology.cuts()[parent].partition;
                                p.contains(extra)
                                    && topology.cuts()[cut].partition.iter().all(|i| p.contains(i))
                            })
                            .expect("every box complement line defines a pentagon")
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            topology,
            cuts,
            pentagons,
            parents,
            kernels,
            zero: prototype.zero(),
            numerator: Vec::new(),
            quotients: Vec::new(),
            coefficients: Vec::new(),
            staged: Vec::new(),
            coefficient_pool: VecPool::default(),
            mask_pool: VecPool::default(),
            rows: Vec::new(),
            geometry_output: Vec::new(),
            active: Vec::new(),
            cut_geometries: Vec::new(),
            scales: Vec::new(),
            mus: Vec::new(),
            samples: Vec::new(),
            viable: Vec::new(),
            parents_flat: Vec::new(),
            division_stable: Vec::new(),
            final_rows: Vec::new(),
            mu_coefficients: Vec::new(),
            mu_stable: Vec::new(),
            vectors: Vec::new(),
            mu_values: Vec::new(),
            projection_pairs: Vec::new(),
            projection_rows: Vec::new(),
            projection_stable: Vec::new(),
            computed: Vec::new(),
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
        outcomes: &mut Vec<CutOutcome<BoxResidue<T>>>,
        keep: usize,
    ) {
        for outcome in outcomes.drain(keep..) {
            if let CutOutcome::Stable(residue) = outcome {
                self.coefficient_pool.recycle(residue.coefficients);
                self.mask_pool.recycle(residue.computed);
            }
        }
    }

    pub fn options(&self) -> BoxReductionOptions {
        self.kernels.options()
    }

    /// Sampling scales are nonzero lane-local mu² values, independent of the
    /// renormalization scale. Structural errors leave the caller output intact.
    /// Pentagon outcomes may be empty when the selected mode does not need them.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_batch<N: Numerator<T>, C: HighCutProvider<T>>(
        &mut self,
        shifts: &[FourMomentum<T>],
        masses_squared: &[T],
        sampling_scales: &[T],
        lanes: usize,
        pentagon_topology: &Topology,
        pentagons: &[CutOutcome<PentagonResidue<T>>],
        numerator: &mut N,
        geometry: &mut C,
        output: &mut [CutOutcome<BoxResidue<T>>],
    ) -> Result<()> {
        let n = self.topology.propagators();
        let options = self.kernels.options();
        let levels = options.sample_count(n, self.topology.rank());
        let need_parents = options.needs_pentagons(n, self.topology.rank());
        for (name, len, columns) in [
            ("box shifts", shifts.len(), n),
            ("box masses", masses_squared.len(), n),
            ("box sampling scales", sampling_scales.len(), 1),
            ("box outcomes", output.len(), self.cuts.len()),
        ] {
            BatchShape {
                rows: lanes,
                columns,
            }
            .validate(name, len)?;
        }
        if numerator.rank() > self.topology.rank() {
            return Err(Error::Unsupported(
                "numerator rank exceeds configured box rank".into(),
            ));
        }
        if need_parents || !pentagons.is_empty() || options.coefficients == BoxCoefficientMode::Full
        {
            if pentagon_topology.propagators() != n
                || pentagon_topology.rank() != self.topology.rank()
            {
                return Err(Error::Expansion(
                    "pentagon topology or rank differs from box plan".into(),
                ));
            }
            BatchShape {
                rows: lanes,
                columns: self.pentagons.len(),
            }
            .validate("box pentagon parents", pentagons.len())?;
            for (parent_row, &index) in self.pentagons.iter().enumerate() {
                for lane in 0..lanes {
                    if let CutOutcome::Stable(parent) = &pentagons[parent_row * lanes + lane]
                        && (parent.cut_index != index
                            || parent.id
                                != (SampleId {
                                    lane,
                                    sample: index,
                                }))
                    {
                        return Err(Error::Expansion(
                            "pentagon identity differs from box topology".into(),
                        ));
                    }
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
                self.mask_pool.recycle(residue.computed);
            }
        }
        let staged = &mut self.staged;
        staged.resize(
            output.len(),
            CutOutcome::Unstable(CutFailure::NonFiniteResult),
        );
        staged.fill(CutOutcome::Unstable(CutFailure::NonFiniteResult));
        let rows = &mut self.rows;
        rows.clear();
        let mut requests = Vec::with_capacity(output.len());
        for (cut_row, &index) in self.cuts.iter().enumerate() {
            for (lane, scale) in sampling_scales.iter().enumerate() {
                let row = cut_row * lanes + lane;
                let values = &masses_squared[lane * n..(lane + 1) * n];
                let valid = self.kernels.values_are_finite(values)
                    && self.kernels.values_are_finite(std::slice::from_ref(scale))
                    && shifts[lane * n..(lane + 1) * n].iter().all(|v| {
                        v.components()
                            .iter()
                            .all(|c| self.kernels.values_are_finite(std::slice::from_ref(*c)))
                    });
                if !valid {
                    staged[row] = CutOutcome::Unstable(CutFailure::NonFiniteInput);
                    continue;
                }
                if need_parents
                    && let Some(failure) = self.parents[cut_row].iter().find_map(|&p| {
                        match &pentagons[p * lanes + lane] {
                            CutOutcome::Unstable(f) => Some(*f),
                            CutOutcome::Stable(parent)
                                if !self.kernels.values_are_finite(std::slice::from_ref(
                                    &parent.coefficient,
                                )) =>
                            {
                                Some(CutFailure::NonFiniteInput)
                            }
                            _ => None,
                        }
                    })
                {
                    staged[row] = CutOutcome::Unstable(failure);
                    continue;
                }
                let partition = &self.topology.cuts()[index].partition;
                rows.push(row);
                requests.push(BoxCutRequest {
                    id: SampleId {
                        lane,
                        sample: index,
                    },
                    shifts: std::array::from_fn(|i| &shifts[lane * n + partition[i]]),
                    masses_squared: std::array::from_fn(|i| {
                        &masses_squared[lane * n + partition[i]]
                    }),
                });
            }
        }
        let cut_output = &mut self.geometry_output;
        cut_output.resize(
            rows.len(),
            CutOutcome::Unstable(CutFailure::NonFiniteResult),
        );
        cut_output.fill(CutOutcome::Unstable(CutFailure::NonFiniteResult));
        geometry.box_batch(&requests, cut_output)?;
        let active = &mut self.active;
        active.clear();
        let cuts = &mut self.cut_geometries;
        cuts.clear();
        for (i, cut) in cut_output.drain(..).enumerate() {
            match cut {
                CutOutcome::Stable(cut) => {
                    active.push(rows[i]);
                    cuts.push(cut);
                }
                CutOutcome::Unstable(f) => staged[rows[i]] = CutOutcome::Unstable(f),
            }
        }
        let scales = &mut self.scales;
        scales.clear();
        scales.extend(
            active
                .iter()
                .map(|&row| sampling_scales[row % lanes].clone()),
        );
        let mus = &mut self.mus;
        mus.resize(active.len() * levels, self.zero.clone());
        self.kernels.sampling_mu_batch(scales, mus)?;
        let mut sample_requests = Vec::with_capacity(active.len() * levels);
        for (i, &row) in active.iter().enumerate() {
            for level in 0..levels {
                sample_requests.push(BoxSampleRequest {
                    id: SampleId {
                        lane: row % lanes,
                        sample: self.cuts[row / lanes] * levels + level,
                    },
                    cut: &cuts[i],
                    mu_squared: &mus[i * levels + level],
                });
            }
        }
        let samples = &mut self.samples;
        samples.resize(
            sample_requests.len(),
            CutOutcome::Unstable(CutFailure::NonFiniteResult),
        );
        samples.fill(CutOutcome::Unstable(CutFailure::NonFiniteResult));
        geometry.sample_box_batch(&sample_requests, samples)?;
        let viable = &mut self.viable;
        viable.clear();
        for (i, &row) in active.iter().enumerate() {
            if let Some(f) = samples[i * levels..(i + 1) * levels]
                .iter()
                .find_map(|s| match s {
                    CutOutcome::Unstable(f) => Some(*f),
                    _ => None,
                })
            {
                staged[row] = CutOutcome::Unstable(f);
            } else {
                viable.push(i);
            }
        }
        let parents = &mut self.parents_flat;
        parents.clear();
        let parent_count = if need_parents { n - 4 } else { 0 };
        if need_parents {
            for &i in viable.iter() {
                let row = active[i];
                parents.extend(self.parents[row / lanes].iter().map(|&p| {
                    let CutOutcome::Stable(parent) = &pentagons[p * lanes + row % lanes] else {
                        unreachable!()
                    };
                    parent.coefficient.clone()
                }));
            }
        }
        let mut eval_requests = Vec::with_capacity(viable.len() * levels * 2);
        let mut division_requests = Vec::with_capacity(viable.len() * levels * 2);
        for (v, &i) in viable.iter().enumerate() {
            let row = active[i];
            let lane = row % lanes;
            let cut_index = self.cuts[row / lanes];
            let cut = &self.topology.cuts()[cut_index];
            for level in 0..levels {
                let CutOutcome::Stable(pair) = &samples[i * levels + level] else {
                    unreachable!()
                };
                for (root, q) in pair.iter().enumerate() {
                    let mu = &mus[i * levels + level];
                    eval_requests.push(NumeratorRequest {
                        id: SampleId {
                            lane,
                            sample: (cut_index * levels + level) * 2 + root,
                        },
                        q,
                        mu_squared: mu,
                        partition: &cut.partition,
                    });
                    division_requests.push(BoxDivisionRequest {
                        partition: &cut.partition,
                        complement: &cut.complement,
                        shifts: &shifts[lane * n..(lane + 1) * n],
                        masses_squared: &masses_squared[lane * n..(lane + 1) * n],
                        momentum: q,
                        mu_squared: mu,
                        pentagon_coefficients: (level > 0 && need_parents)
                            .then_some(&parents[v * parent_count..(v + 1) * parent_count]),
                    });
                }
            }
        }
        self.numerator
            .resize(eval_requests.len(), self.zero.clone());
        numerator.evaluate_batch(&eval_requests, &mut self.numerator)?;
        self.quotients
            .resize(eval_requests.len(), self.zero.clone());
        let stable = &mut self.division_stable;
        stable.resize(eval_requests.len(), false);
        stable.fill(false);
        self.kernels.divide_samples_batch(
            &division_requests,
            &self.numerator,
            &mut self.quotients,
            stable,
        )?;
        let final_rows = &mut self.final_rows;
        final_rows.clear();
        for (v, &i) in viable.iter().enumerate() {
            if stable[v * levels * 2..(v + 1) * levels * 2]
                .iter()
                .all(|s| *s)
            {
                final_rows.push((v, i));
            } else {
                staged[active[i]] = CutOutcome::Unstable(CutFailure::SingularParameter);
            }
        }
        let mu_coefficients = &mut self.mu_coefficients;
        let zeros = [self.zero.clone(), self.zero.clone(), self.zero.clone()];
        mu_coefficients.resize(final_rows.len(), zeros.clone());
        mu_coefficients.fill(zeros);
        let mu_stable = &mut self.mu_stable;
        mu_stable.resize(final_rows.len(), true);
        mu_stable.fill(true);
        if let Some(kind) = self.kernels.numerator_mu_expansion() {
            // Rank n+2 expands on the exact d-dimensional cut,
            // mu² = t² v_orth² - beta (`ExpansionKind::MuShifted`).
            let shifted = matches!(kind, ExpansionKind::MuShifted { .. });
            let vectors = &mut self.vectors;
            vectors.clear();
            vectors.extend(
                final_rows
                    .iter()
                    .map(|&(_, i)| [cuts[i].v_orth.clone(), cuts[i].a.clone()]),
            );
            let mut expansions = Vec::with_capacity(final_rows.len());
            let mut divisions = Vec::with_capacity(final_rows.len());
            for (j, &(_, i)) in final_rows.iter().enumerate() {
                let row = active[i];
                let lane = row % lanes;
                let cut_index = self.cuts[row / lanes];
                let cut = &self.topology.cuts()[cut_index];
                expansions.push(ExpansionRequest {
                    id: SampleId {
                        lane,
                        sample: cut_index,
                    },
                    partition: &cut.partition,
                    vectors: &vectors[j],
                    parameters: if shifted {
                        std::slice::from_ref(&cuts[i].beta)
                    } else {
                        &[]
                    },
                });
                divisions.push(BoxMuDivisionRequest {
                    partition: &cut.partition,
                    complement: &cut.complement,
                    shifts: &shifts[lane * n..(lane + 1) * n],
                    masses_squared: &masses_squared[lane * n..(lane + 1) * n],
                    geometry: &cuts[i],
                });
            }
            let values = &mut self.mu_values;
            values.resize(kind.len() * final_rows.len(), self.zero.clone());
            numerator.expansion_batch(kind, &expansions, values)?;
            self.kernels
                .divide_mu_batch(&divisions, values, mu_coefficients, mu_stable)?;
        }
        let pairs = &mut self.projection_pairs;
        pairs.clear();
        for &(_, i) in final_rows.iter() {
            pairs.extend((0..levels).map(|level| {
                let CutOutcome::Stable(pair) = &samples[i * levels + level] else {
                    unreachable!()
                };
                pair.clone()
            }));
        }
        let projection_rows = &mut self.projection_rows;
        projection_rows.clear();
        let mut projection_requests = Vec::with_capacity(final_rows.len());
        for (j, &(v, i)) in final_rows.iter().enumerate() {
            let row = active[i];
            if !mu_stable[j] {
                staged[row] = CutOutcome::Unstable(CutFailure::SingularParameter);
                continue;
            }
            let lane = row % lanes;
            let cut = &self.topology.cuts()[self.cuts[row / lanes]];
            projection_rows.push(i);
            projection_requests.push(BoxProjectionRequest {
                geometry: &cuts[i],
                shift: &shifts[lane * n + cut.partition[0]],
                momenta: &pairs[j * levels..(j + 1) * levels],
                quotients: self.quotients[v * levels * 2..(v + 1) * levels * 2]
                    .as_chunks::<2>()
                    .0,
                sampling_scale: &sampling_scales[lane],
                mu_coefficients: &mu_coefficients[j],
            });
        }
        let width = self.kernels.coefficient_count();
        self.coefficients
            .resize(projection_requests.len() * width, self.zero.clone());
        let stable = &mut self.projection_stable;
        stable.resize(projection_requests.len(), false);
        stable.fill(false);
        self.kernels
            .project_batch(&projection_requests, &mut self.coefficients, stable)?;
        let computed = &mut self.computed;
        computed.clear();
        // c6 (mu⁶) exists only in the rank n+2 width of seven.
        computed.extend((0..width).map(|i| {
            options.coefficients == BoxCoefficientMode::Full || i == 0 || i == 4 || i == 6
        }));
        for (j, &i) in projection_rows.iter().enumerate() {
            let row = active[i];
            let coefficients = &self.coefficients[j * width..(j + 1) * width];
            staged[row] = if stable[j] && self.kernels.values_are_finite(coefficients) {
                CutOutcome::Stable(BoxResidue {
                    cut_index: self.cuts[row / lanes],
                    id: SampleId {
                        lane: row % lanes,
                        sample: self.cuts[row / lanes],
                    },
                    geometry: cuts[i].clone(),
                    coefficients: self.coefficient_pool.take_from_slice(coefficients),
                    computed: self.mask_pool.take_from_slice(computed),
                })
            } else {
                CutOutcome::Unstable(CutFailure::NonFiniteResult)
            };
        }
        output.swap_with_slice(staged);
        Ok(())
    }
}
