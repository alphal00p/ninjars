//! Batched quadruple-cut reduction with both rational extraction modes.

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
    pub fn sample_count(self, n: usize, rank: usize) -> usize {
        match (self.coefficients, self.mu_extraction) {
            (BoxCoefficientMode::Full, BoxMuMode::Sampling) => 3,
            (BoxCoefficientMode::Full, BoxMuMode::Expansion) => 2,
            (BoxCoefficientMode::Integrated, BoxMuMode::Sampling) if rank >= n => 3,
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
    /// Pairs ordered mu²=0,+s,-s, truncated to the selected mode's count.
    pub momenta: &'a [[FourMomentum<T>; 2]],
    pub quotients: &'a [[T; 2]],
    pub sampling_scale: &'a T,
    /// c4,c5 from the mu expansion, or zero when sampling is selected.
    pub mu_coefficients: &'a [T; 2],
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
        output: &mut [[T; 2]],
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
    /// Original Ninja order: c0+c2*mu²+c4*mu⁴+(c1+c3*mu²+c5*mu⁴)*u.
    pub coefficients: Vec<T>,
    /// Integrated mode computes c0/c4; other slots are explicitly uncomputed.
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
        })
    }
    pub fn topology(&self) -> &Topology {
        &self.topology
    }
    pub fn cut_count(&self) -> usize {
        self.cuts.len()
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
        let mut staged = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); output.len()];
        let mut rows = Vec::new();
        let mut requests = Vec::new();
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
        let mut cut_output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); rows.len()];
        geometry.box_batch(&requests, &mut cut_output)?;
        let mut active = Vec::new();
        let mut cuts = Vec::new();
        for (i, cut) in cut_output.into_iter().enumerate() {
            match cut {
                CutOutcome::Stable(cut) => {
                    active.push(rows[i]);
                    cuts.push(cut);
                }
                CutOutcome::Unstable(f) => staged[rows[i]] = CutOutcome::Unstable(f),
            }
        }
        let scales: Vec<_> = active
            .iter()
            .map(|&row| sampling_scales[row % lanes].clone())
            .collect();
        let mut mus = vec![self.zero.clone(); active.len() * levels];
        self.kernels.sampling_mu_batch(&scales, &mut mus)?;
        let mut sample_requests = Vec::new();
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
        let mut samples =
            vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); sample_requests.len()];
        geometry.sample_box_batch(&sample_requests, &mut samples)?;
        let mut viable = Vec::new();
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
        let parents: Vec<Vec<T>> = viable
            .iter()
            .map(|&i| {
                let row = active[i];
                if need_parents {
                    self.parents[row / lanes]
                        .iter()
                        .map(|&p| match &pentagons[p * lanes + row % lanes] {
                            CutOutcome::Stable(p) => p.coefficient.clone(),
                            _ => unreachable!(),
                        })
                        .collect()
                } else {
                    Vec::new()
                }
            })
            .collect();
        let mut eval_requests = Vec::new();
        let mut division_requests = Vec::new();
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
                            .then_some(parents[v].as_slice()),
                    });
                }
            }
        }
        self.numerator
            .resize(eval_requests.len(), self.zero.clone());
        numerator.evaluate_batch(&eval_requests, &mut self.numerator)?;
        self.quotients
            .resize(eval_requests.len(), self.zero.clone());
        let mut stable = vec![false; eval_requests.len()];
        self.kernels.divide_samples_batch(
            &division_requests,
            &self.numerator,
            &mut self.quotients,
            &mut stable,
        )?;
        let mut final_rows = Vec::new();
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
        let mut mu_coefficients = vec![[self.zero.clone(), self.zero.clone()]; final_rows.len()];
        let mut mu_stable = vec![true; final_rows.len()];
        if let Some(kind) = self.kernels.numerator_mu_expansion().cloned() {
            let vectors: Vec<_> = final_rows
                .iter()
                .map(|&(_, i)| [cuts[i].v_orth.clone(), cuts[i].a.clone()])
                .collect();
            let mut expansions = Vec::new();
            let mut divisions = Vec::new();
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
                    parameters: &[],
                });
                divisions.push(BoxMuDivisionRequest {
                    partition: &cut.partition,
                    complement: &cut.complement,
                    shifts: &shifts[lane * n..(lane + 1) * n],
                    masses_squared: &masses_squared[lane * n..(lane + 1) * n],
                    geometry: &cuts[i],
                });
            }
            let mut values = vec![self.zero.clone(); kind.len() * final_rows.len()];
            numerator.expansion_batch(&kind, &expansions, &mut values)?;
            self.kernels.divide_mu_batch(
                &divisions,
                &values,
                &mut mu_coefficients,
                &mut mu_stable,
            )?;
        }
        let pairs: Vec<Vec<_>> = final_rows
            .iter()
            .map(|&(_, i)| {
                (0..levels)
                    .map(|level| match &samples[i * levels + level] {
                        CutOutcome::Stable(pair) => pair.clone(),
                        _ => unreachable!(),
                    })
                    .collect()
            })
            .collect();
        let quotients: Vec<Vec<_>> = final_rows
            .iter()
            .map(|&(v, _)| {
                self.quotients[v * levels * 2..(v + 1) * levels * 2]
                    .as_chunks::<2>()
                    .0
                    .to_vec()
            })
            .collect();
        let mut projection_rows = Vec::new();
        let mut projection_requests = Vec::new();
        for (j, &(_, i)) in final_rows.iter().enumerate() {
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
                momenta: &pairs[j],
                quotients: &quotients[j],
                sampling_scale: &sampling_scales[lane],
                mu_coefficients: &mu_coefficients[j],
            });
        }
        let width = self.kernels.coefficient_count();
        self.coefficients
            .resize(projection_requests.len() * width, self.zero.clone());
        let mut stable = vec![false; projection_requests.len()];
        self.kernels
            .project_batch(&projection_requests, &mut self.coefficients, &mut stable)?;
        let computed: Vec<_> = (0..width)
            .map(|i| options.coefficients == BoxCoefficientMode::Full || i == 0 || i == 4)
            .collect();
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
                    coefficients: coefficients.to_vec(),
                    computed: computed.clone(),
                })
            } else {
                CutOutcome::Unstable(CutFailure::NonFiniteResult)
            };
        }
        output.swap_with_slice(&mut staged);
        Ok(())
    }
}
