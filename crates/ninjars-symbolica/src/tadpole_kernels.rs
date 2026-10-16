//! Symbolica arithmetic for the minimal integrated tadpole stage.

use crate::{
    DenominatorExpansion, LaurentDivision, TadpoleSourceProjection, TadpoleSubtraction,
    kernel::{CompiledDualKernel, CompiledKernel, ExactKernel, KernelExecutor},
    rank_specialization::{self, GuardedCorrection},
    value_checks::{DualChecks, IntegrationChecks, OrdinaryChecks},
};
use ninjars::{
    BatchShape, Complex, CutFailure, Error, ExpansionKind, FloatLike, FourMomentum, Real, Result,
    SingleFloat, TadpoleCorrectionRequest, TadpoleDivisionRequest, TadpoleReductionKernels,
    TadpoleSource, Topology,
};
use symbolica::{atom::Atom, domains::dual::DualNumber, evaluate::EvaluationDomain, symbol};

#[cfg(test)]
thread_local! {
    pub(crate) static DENOMINATOR_ROWS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

struct Engine<V, K, Checks> {
    topology: Topology,
    layout: LaurentDivision,
    mask: Vec<bool>,
    parents: usize,
    denominator: K,
    division: K,
    projection: K,
    sum: K,
    source_projection: [Option<K>; 2],
    subtraction: [Option<K>; 2],
    specialized_subtraction: [Option<GuardedCorrection<K>>; 2],
    correction_columns: usize,
    correction_map: Vec<usize>,
    extras: Vec<Vec<usize>>,
    zero: V,
    checks: std::marker::PhantomData<fn() -> Checks>,
    input: Vec<V>,
    denominators: Vec<V>,
    division_offsets: Vec<Option<usize>>,
    projections: Vec<V>,
    values: Vec<V>,
    eligible: Vec<usize>,
    active: Vec<usize>,
}

fn vars(n: usize, scope: &str) -> Vec<Atom> {
    (0..n)
        .map(|i| Atom::var(symbol!(format!("ninjars_tadpole_{scope}::p{i}"))))
        .collect()
}
impl<V: FloatLike, K: KernelExecutor<V>, Checks: IntegrationChecks<V>> Engine<V, K, Checks> {
    fn new(
        n: usize,
        rank: usize,
        prototype: &V,
        mut compile: impl FnMut(&ExactKernel) -> Result<K>,
    ) -> Result<Self> {
        let topology = Topology::new(n, rank)?;
        let layout = LaurentDivision::new(n, rank, 1)?;
        let higher = rank > n;
        let orientations = if higher { 2 } else { 1 };
        let width = layout.quotient_expansion().len();
        let denominator = compile(&DenominatorExpansion::new(1)?.exact)?;
        let division = compile(&layout.exact)?;
        let parents = n
            .checked_mul(n - 1)
            .map(|v| v / 2)
            .ok_or_else(|| Error::Unsupported("tadpole parent count overflows usize".into()))?;
        let p = vars((1 + parents) * width, "sum");
        let expressions = (0..width)
            .map(|column| {
                (0..parents).fold(p[column].clone(), |sum, parent| {
                    sum - &p[(1 + parent) * width + column]
                })
            })
            .collect();
        let sum = compile(&ExactKernel {
            expressions,
            parameters: p,
        })?;
        let p = vars(orientations * width + 1, "projection");
        let ExpansionKind::T3 { powers } = layout.quotient_expansion() else {
            unreachable!()
        };
        let coefficient = |orientation: usize, mu: u16| {
            powers
                .iter()
                .position(|&power| power == (0, mu))
                .map_or(Atom::Zero, |i| p[orientation * width + i].clone())
        };
        let mut expressions = vec![Atom::Zero; if higher { 16 } else { 5 }];
        let mut mask = vec![false; expressions.len()];
        mask[0] = true;
        expressions[0] = coefficient(0, 0);
        if higher {
            expressions[14] = coefficient(0, 1);
            expressions[15] =
                -Atom::num(2) * (coefficient(1, 1) - coefficient(0, 1)) / &p[2 * width];
            mask[14] = true;
            mask[15] = true;
        }
        let projection = compile(&ExactKernel {
            expressions,
            parameters: p,
        })?;
        let mut source_projection = [None, None];
        let mut subtraction = [None, None];
        let mut correction_columns = 0;
        let mut correction_map = Vec::new();
        for (kind, source_size) in [3, 2].into_iter().enumerate() {
            if source_size > n {
                continue;
            }
            let plan = TadpoleSubtraction::new(source_size, higher)?;
            if correction_columns == 0 {
                correction_columns = plan.output_expansion().len();
                let ExpansionKind::T3 { powers: full } = plan.output_expansion() else {
                    unreachable!()
                };
                correction_map = powers
                    .iter()
                    .map(|power| {
                        full.iter().position(|p| p == power).ok_or_else(|| {
                            Error::Expansion(
                                "tadpole subtraction lacks a required quotient power".into(),
                            )
                        })
                    })
                    .collect::<Result<_>>()?;
            }
            subtraction[kind] = Some(compile(&plan.exact)?);
            source_projection[kind] =
                Some(compile(&TadpoleSourceProjection::new(source_size)?.exact)?);
        }
        let extras = topology
            .cuts()
            .iter()
            .flat_map(|source| {
                (0..n).map(move |target| {
                    if matches!(source.partition.len(), 2 | 3) && source.partition.contains(&target)
                    {
                        source
                            .partition
                            .iter()
                            .copied()
                            .filter(|&line| line != target)
                            .collect()
                    } else {
                        Vec::new()
                    }
                })
            })
            .collect();
        Ok(Self {
            topology,
            layout,
            mask,
            parents,
            denominator,
            division,
            projection,
            sum,
            source_projection,
            subtraction,
            specialized_subtraction: [None, None],
            correction_columns,
            correction_map,
            extras,
            zero: prototype.zero(),
            checks: std::marker::PhantomData,
            input: Vec::new(),
            denominators: Vec::new(),
            division_offsets: Vec::new(),
            projections: Vec::new(),
            values: Vec::new(),
            eligible: Vec::new(),
            active: Vec::new(),
        })
    }

    fn specialize_corrections(
        &mut self,
        mut compile: impl FnMut(&ExactKernel) -> Result<K>,
    ) -> Result<()> {
        let n = self.topology.propagators();
        let rank = self.topology.rank();
        rank_specialization::require_supported(n, rank)?;
        if self.layout.quotient_expansion()
            != &(ExpansionKind::T3 {
                powers: vec![(0, 0)],
            })
        {
            return Err(Error::Unsupported(
                "guarded tadpole specialization requires the constant-only T3 quotient".into(),
            ));
        }
        let mut specialized = [None, None];
        for (kind, source_size) in [3, 2].into_iter().enumerate() {
            let zero_slots = rank_specialization::zero_slots(n, rank, source_size)?;
            let plan = TadpoleSubtraction::new(source_size, false)?;
            let ExpansionKind::T3 { powers } = plan.output_expansion() else {
                unreachable!()
            };
            let constant = powers.iter().position(|&p| p == (0, 0)).ok_or_else(|| {
                Error::Expansion("tadpole correction has no constant head".into())
            })?;
            let exact = rank_specialization::specialize(
                &plan.exact,
                &[constant],
                plan.input_layout().residue.clone(),
                &zero_slots,
            );
            specialized[kind] = Some(GuardedCorrection {
                kernel: compile(&exact)?,
                zero_slots,
            });
        }
        self.specialized_subtraction = specialized;
        Ok(())
    }

    fn status(&self, values: &[V]) -> Option<CutFailure> {
        if values.iter().any(|v| !Checks::shape_matches(v, &self.zero)) {
            Some(CutFailure::DerivativeShape)
        } else if values.iter().any(|v| !Checks::finite(v)) {
            Some(CutFailure::NonFiniteInput)
        } else {
            None
        }
    }

    fn divide(
        &mut self,
        requests: &[TadpoleDivisionRequest<'_, V>],
        numerator: &[V],
        output: &mut [V],
        stable: &mut [bool],
    ) -> Result<()> {
        let n = self.topology.propagators();
        let rows = requests.len();
        let columns = self.layout.numerator_expansion().len();
        let width = self.layout.quotient_expansion().len();
        let uncut = n - 1;
        BatchShape { rows, columns }.validate("tadpole numerator coefficients", numerator.len())?;
        BatchShape {
            rows,
            columns: width,
        }
        .validate("tadpole quotient coefficients", output.len())?;
        BatchShape { rows, columns: 1 }.validate("tadpole division status", stable.len())?;
        for r in requests {
            if r.partition.len() != 1
                || r.complement.len() != uncut
                || r.shifts.len() != n
                || r.masses_squared.len() != n
                || r.partition.iter().chain(r.complement).any(|&i| i >= n)
                || (0..n).any(|line| {
                    r.partition
                        .iter()
                        .chain(r.complement)
                        .filter(|&&i| i == line)
                        .count()
                        != 1
                })
            {
                return Err(Error::Expansion(
                    "tadpole denominator request does not match its topology".into(),
                ));
            }
        }
        self.input.clear();
        self.eligible.clear();
        self.division_offsets.clear();
        self.division_offsets.resize(rows, None);
        for (row, r) in requests.iter().enumerate() {
            stable[row] = r
                .shifts
                .iter()
                .chain(r.vectors)
                .flat_map(FourMomentum::components)
                .chain(r.masses_squared)
                .chain([r.beta])
                .chain(&numerator[row * columns..(row + 1) * columns])
                .all(|v| Checks::shape_matches(v, &self.zero) && Checks::finite(v));
            if !stable[row] {
                continue;
            }
            self.division_offsets[row] = Some(self.eligible.len() * uncut * 4);
            self.eligible.push(row);
            for &line in r.complement {
                for v in r.vectors {
                    self.input.extend(v.components().into_iter().cloned());
                }
                self.input.push(r.beta.clone());
                self.input
                    .extend(r.shifts[r.partition[0]].components().into_iter().cloned());
                self.input.push(r.masses_squared[r.partition[0]].clone());
                self.input
                    .extend(r.shifts[line].components().into_iter().cloned());
                self.input.push(r.masses_squared[line].clone());
            }
        }
        self.denominators
            .resize(self.eligible.len() * uncut * 4, self.zero.clone());
        #[cfg(test)]
        DENOMINATOR_ROWS.with(|count| count.set(count.get() + self.eligible.len() * uncut));
        self.denominator.evaluate(
            &self.input,
            &mut self.denominators,
            self.eligible.len() * uncut,
        )?;
        self.active.clear();
        self.input.clear();
        for (index, &row) in self.eligible.iter().enumerate() {
            let denominator = &self.denominators[index * uncut * 4..(index + 1) * uncut * 4];
            stable[row] = denominator.iter().all(Checks::finite)
                && denominator
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|d| Checks::nonzero(&d[0]));
            if stable[row] {
                self.active.push(row);
                self.input
                    .extend_from_slice(&numerator[row * columns..(row + 1) * columns]);
                self.input.extend_from_slice(denominator);
            }
        }
        self.values
            .resize(self.active.len() * width, self.zero.clone());
        self.division
            .evaluate(&self.input, &mut self.values, self.active.len())?;
        for (i, &row) in self.active.iter().enumerate() {
            let values = &self.values[i * width..(i + 1) * width];
            stable[row] = values.iter().all(Checks::finite);
            if stable[row] {
                output[row * width..(row + 1) * width].clone_from_slice(values);
            }
        }
        Ok(())
    }

    fn correction(
        &mut self,
        requests: &[TadpoleCorrectionRequest<'_, V>],
        output: &mut [V],
        stable: &mut [bool],
        prepared: Option<(&[TadpoleDivisionRequest<'_, V>], &[bool])>,
    ) -> Result<()> {
        let n = self.topology.propagators();
        let higher = self.topology.rank() > n;
        let width = self.layout.quotient_expansion().len();
        let rows = requests.len();
        BatchShape {
            rows,
            columns: width,
        }
        .validate("tadpole parent corrections", output.len())?;
        BatchShape { rows, columns: 1 }
            .validate("tadpole parent correction status", stable.len())?;
        let parents = self.parents;
        let orientations = if higher { 2 } else { 1 };
        let selected = |row: usize| {
            prepared.is_none_or(|(_, stable)| {
                let target = row / parents / orientations;
                stable[target * orientations..(target + 1) * orientations]
                    .iter()
                    .all(|&v| v)
            })
        };
        for (row, request) in requests.iter().enumerate() {
            if !selected(row) {
                continue;
            }
            let (index, id, size, count) = match &request.source {
                TadpoleSource::Triangle(r) => (r.cut_index, r.id, 3, r.coefficients.len()),
                TadpoleSource::Bubble(r) => (r.cut_index, r.id, 2, r.coefficients.len()),
            };
            let expected_count = match (size, higher) {
                (3, true) => 15,
                (2, true) => 20,
                _ => 10,
            };
            if request.id.sample >= self.topology.cuts().len()
                || index >= self.topology.cuts().len()
                || self.topology.cuts()[request.id.sample].partition.len() != 1
                || self.topology.cuts()[index].partition.len() != size
                || !self.topology.cuts()[request.id.sample]
                    .parents
                    .contains(&index)
                || id.lane != request.id.lane
                || id.sample != index
                || count != expected_count
                || request.shifts.len() != n
                || request.masses_squared.len() != n
            {
                return Err(Error::Expansion(
                    "tadpole source provenance/containment/layout is inconsistent".into(),
                ));
            }
        }
        let reuse = prepared.is_some_and(|(divisions, _)| {
            requests
                .iter()
                .enumerate()
                .filter(|(row, _)| selected(*row))
                .all(|(row, r)| {
                    let d = &divisions[row / parents];
                    self.topology.cuts()[r.id.sample].partition == d.partition
                        && std::ptr::eq(r.vectors, d.vectors)
                        && std::ptr::eq(r.beta, d.beta)
                        && std::ptr::eq(r.shifts, d.shifts)
                        && std::ptr::eq(r.masses_squared, d.masses_squared)
                })
        });
        stable.fill(false);
        for kind in 0..2 {
            let source_size = if kind == 0 { 3 } else { 2 };
            if source_size > n {
                continue;
            }
            let axes = if kind == 0 { 2 } else { 3 };
            let uncut = source_size - 1;
            self.eligible.clear();
            self.input.clear();
            for (row, request) in requests.iter().enumerate() {
                if !selected(row) {
                    continue;
                }
                let (index, coefficients, basis) = match &request.source {
                    TadpoleSource::Triangle(r) if kind == 0 => {
                        (r.cut_index, &r.coefficients, &r.geometry.basis)
                    }
                    TadpoleSource::Bubble(r) if kind == 1 => {
                        (r.cut_index, &r.coefficients, &r.geometry.basis)
                    }
                    _ => continue,
                };
                let valid = request
                    .shifts
                    .iter()
                    .chain(request.vectors)
                    .flat_map(FourMomentum::components)
                    .chain(request.masses_squared)
                    .chain([request.beta, &basis.e12])
                    .chain(coefficients)
                    .chain(
                        [&basis.e1, &basis.e2, &basis.e3, &basis.e4]
                            .into_iter()
                            .flat_map(FourMomentum::components),
                    )
                    .all(|v| Checks::shape_matches(v, &self.zero) && Checks::finite(v));
                if !valid {
                    continue;
                }
                self.eligible.push(row);
                for v in request.vectors {
                    self.input.extend(v.components().into_iter().cloned());
                }
                self.input.extend(
                    request.shifts[self.topology.cuts()[index].partition[0]]
                        .components()
                        .into_iter()
                        .cloned(),
                );
                if kind == 1 {
                    self.input
                        .extend(basis.e2.components().into_iter().cloned());
                }
                self.input
                    .extend(basis.e3.components().into_iter().cloned());
                self.input
                    .extend(basis.e4.components().into_iter().cloned());
            }
            self.projections
                .resize(self.eligible.len() * axes * 3, self.zero.clone());
            self.source_projection[kind].as_mut().unwrap().evaluate(
                &self.input,
                &mut self.projections,
                self.eligible.len(),
            )?;
            self.input.clear();
            if !reuse {
                for &row in &self.eligible {
                    let r = &requests[row];
                    let source = match &r.source {
                        TadpoleSource::Triangle(r) => r.cut_index,
                        TadpoleSource::Bubble(r) => r.cut_index,
                    };
                    let target = self.topology.cuts()[r.id.sample].partition[0];
                    for &line in &self.extras[source * n + target] {
                        for v in r.vectors {
                            self.input.extend(v.components().into_iter().cloned());
                        }
                        self.input.push(r.beta.clone());
                        self.input
                            .extend(r.shifts[target].components().into_iter().cloned());
                        self.input.push(r.masses_squared[target].clone());
                        self.input
                            .extend(r.shifts[line].components().into_iter().cloned());
                        self.input.push(r.masses_squared[line].clone());
                    }
                }
                self.denominators
                    .resize(self.eligible.len() * uncut * 4, self.zero.clone());
                #[cfg(test)]
                DENOMINATOR_ROWS.with(|count| count.set(count.get() + self.eligible.len() * uncut));
                self.denominator.evaluate(
                    &self.input,
                    &mut self.denominators,
                    self.eligible.len() * uncut,
                )?;
            }
            self.input.clear();
            self.active.clear();
            for (index, &row) in self.eligible.iter().enumerate() {
                let mut denominators = [&[][..]; 2];
                if reuse {
                    let division = row / parents;
                    let d = &prepared.unwrap().0[division];
                    let r = &requests[row];
                    let source = match &r.source {
                        TadpoleSource::Triangle(r) => r.cut_index,
                        TadpoleSource::Bubble(r) => r.cut_index,
                    };
                    let target = self.topology.cuts()[r.id.sample].partition[0];
                    for (out, line) in denominators
                        .iter_mut()
                        .zip(&self.extras[source * n + target])
                    {
                        let offset = self.division_offsets[division].unwrap()
                            + 4 * d.complement.iter().position(|i| i == line).unwrap();
                        *out = &self.denominators[offset..offset + 4];
                    }
                } else {
                    for (column, out) in denominators[..uncut].iter_mut().enumerate() {
                        let offset = (index * uncut + column) * 4;
                        *out = &self.denominators[offset..offset + 4];
                    }
                }
                let denominators = &denominators[..uncut];
                let projections = &self.projections[index * axes * 3..(index + 1) * axes * 3];
                if !denominators
                    .iter()
                    .flat_map(|d| d.iter())
                    .chain(projections)
                    .all(Checks::finite)
                    || !denominators.iter().all(|d| Checks::nonzero(&d[0]))
                {
                    continue;
                }
                self.active.push(row);
                let r = &requests[row];
                let coefficients = match &r.source {
                    TadpoleSource::Triangle(r) => &r.coefficients,
                    TadpoleSource::Bubble(r) => &r.coefficients,
                };
                self.input.extend_from_slice(coefficients);
                self.input.extend_from_slice(projections);
                self.input.push(r.beta.clone());
                for den in denominators {
                    self.input.extend_from_slice(den);
                }
            }
            // Keep all rows of a source kind together. Shape/finite checks
            // above precede the all-component structural-zero guard.
            let specialized = self.specialized_subtraction[kind]
                .as_ref()
                .is_some_and(|plan| {
                    self.active.iter().all(|&row| {
                        let coefficients = match &requests[row].source {
                            TadpoleSource::Triangle(r) => &r.coefficients,
                            TadpoleSource::Bubble(r) => &r.coefficients,
                        };
                        plan.zero_slots
                            .iter()
                            .all(|&slot| Checks::is_zero(&coefficients[slot]))
                    })
                });
            let columns = if specialized {
                1
            } else {
                self.correction_columns
            };
            self.values
                .resize(self.active.len() * columns, self.zero.clone());
            let kernel = if specialized {
                &mut self.specialized_subtraction[kind].as_mut().unwrap().kernel
            } else {
                self.subtraction[kind].as_mut().unwrap()
            };
            #[cfg(test)]
            rank_specialization::record(1 + kind, specialized, self.active.len());
            kernel.evaluate(&self.input, &mut self.values, self.active.len())?;
            for (index, &row) in self.active.iter().enumerate() {
                let values = &self.values[index * columns..(index + 1) * columns];
                // The generic path still evaluates and checks BOTH heads,
                // even though only its constant is copied to the quotient.
                stable[row] = values.iter().all(Checks::finite);
                if stable[row] {
                    if specialized {
                        output[row] = values[0].clone();
                    } else {
                        for (column, &source) in self.correction_map.iter().enumerate() {
                            output[row * width + column] = values[source].clone();
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn divide_with_corrections(
        &mut self,
        divisions: &[TadpoleDivisionRequest<'_, V>],
        numerator: &[V],
        quotients: &mut [V],
        quotient_stable: &mut [bool],
        requests: &[TadpoleCorrectionRequest<'_, V>],
        corrections: &mut [V],
        correction_stable: &mut [bool],
    ) -> Result<()> {
        let n = self.topology.propagators();
        let parents = n
            .checked_mul(n - 1)
            .ok_or_else(|| Error::Expansion("tadpole parent count overflows".into()))?
            / 2;
        let orientations = if self.topology.rank() > n { 2 } else { 1 };
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
            columns: self.layout.quotient_expansion().len(),
        }
        .validate("tadpole parent correction output", corrections.len())?;
        BatchShape {
            rows: requests.len(),
            columns: 1,
        }
        .validate("tadpole parent correction status", correction_stable.len())?;
        self.divide(divisions, numerator, quotients, quotient_stable)?;
        self.correction(
            requests,
            corrections,
            correction_stable,
            Some((divisions, quotient_stable)),
        )
    }

    fn combine(
        &mut self,
        raw: &[V],
        corrections: &[V],
        rows: usize,
        output: &mut [V],
    ) -> Result<()> {
        let width = self.layout.quotient_expansion().len();
        BatchShape {
            rows,
            columns: width,
        }
        .validate("raw tadpole quotients", raw.len())?;
        BatchShape {
            rows,
            columns: width * self.parents,
        }
        .validate("summed tadpole corrections", corrections.len())?;
        BatchShape {
            rows,
            columns: width,
        }
        .validate("corrected tadpole quotients", output.len())?;
        self.input.clear();
        for row in 0..rows {
            self.input
                .extend_from_slice(&raw[row * width..(row + 1) * width]);
            self.input.extend_from_slice(
                &corrections[row * width * self.parents..(row + 1) * width * self.parents],
            );
        }
        self.sum.evaluate(&self.input, output, rows)
    }

    fn project(&mut self, corrected: &[V], e12: &[V], output: &mut [V]) -> Result<()> {
        let rows = e12.len();
        let width = self.layout.quotient_expansion().len()
            * if self.topology.rank() > self.topology.propagators() {
                2
            } else {
                1
            };
        BatchShape {
            rows,
            columns: width,
        }
        .validate("tadpole projection quotients", corrected.len())?;
        BatchShape {
            rows,
            columns: self.mask.len(),
        }
        .validate("integrated tadpole coefficients", output.len())?;
        self.input.clear();
        for (row, g) in e12.iter().enumerate() {
            self.input
                .extend_from_slice(&corrected[row * width..(row + 1) * width]);
            self.input.push(g.clone());
        }
        self.projection.evaluate(&self.input, output, rows)
    }
}

/// Minimal tadpole integration coefficients; no full residue reconstruction.
pub struct SymbolicaTadpoleKernels<T> {
    engine: Engine<Complex<T>, CompiledKernel<T>, OrdinaryChecks>,
}
impl<T: Real + SingleFloat> SymbolicaTadpoleKernels<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn new(propagators: usize, rank: usize, prototype: T) -> Result<Self> {
        let zero = Complex::new(prototype.zero(), prototype.zero());
        Ok(Self {
            engine: Engine::new(propagators, rank, &zero, |exact| exact.compile(&prototype))?,
        })
    }

    /// Opt into interpreted rank-specialized constant corrections for n=5/r=4.
    /// A whole source-kind batch uses the alternate only when every active
    /// parent's structural-zero coefficients are exactly zero. The generic
    /// fallback retains its full output and finite checks.
    pub fn with_rank_specialized_corrections(mut self) -> Result<Self> {
        let prototype = self.engine.zero.re.clone();
        self.engine
            .specialize_corrections(|exact| exact.compile(&prototype))?;
        Ok(self)
    }

    pub fn rank_specialized_corrections_enabled(&self) -> bool {
        self.engine
            .specialized_subtraction
            .iter()
            .all(Option::is_some)
    }
}

/// The same tadpole coefficients through shared static or runtime Dual bodies.
pub struct SymbolicaDualTadpoleKernels<T, D> {
    engine: Engine<D, CompiledDualKernel<T, D>, DualChecks<T>>,
}
impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> SymbolicaDualTadpoleKernels<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn new(propagators: usize, rank: usize, prototype: T, dual: &D) -> Result<Self> {
        Ok(Self {
            engine: Engine::new(propagators, rank, dual, |exact| {
                exact.compile_dual(&prototype, dual)
            })?,
        })
    }

    /// See [`SymbolicaTadpoleKernels::with_rank_specialized_corrections`].
    /// Structural zeros are checked across every higher/mixed Dual component.
    pub fn with_rank_specialized_corrections(mut self) -> Result<Self> {
        let dual = self.engine.zero.clone();
        let prototype = dual.values()[0].re.clone();
        self.engine
            .specialize_corrections(|exact| exact.compile_dual(&prototype, &dual))?;
        Ok(self)
    }

    pub fn rank_specialized_corrections_enabled(&self) -> bool {
        self.engine
            .specialized_subtraction
            .iter()
            .all(Option::is_some)
    }
}

macro_rules! implement {
    ($name:ident<$($g:ident),+>,$value:ty,$($bounds:tt)*)=> {
        impl<$($g),+> TadpoleReductionKernels<$value> for $name<$($g),+> where $($bounds)* {
            fn propagators(&self)->usize {self.engine.topology.propagators()}
            fn rank(&self)->usize {self.engine.topology.rank()}
            fn numerator_expansion(&self)->&ExpansionKind {self.engine.layout.numerator_expansion()}
            fn quotient_columns(&self)->usize {self.engine.layout.quotient_expansion().len()}
            fn coefficient_count(&self)->usize {self.engine.mask.len()}
            fn computed_coefficients(&self)->&[bool] {&self.engine.mask}
            fn value_status(&self,values:&[$value])->Option<CutFailure> {self.engine.status(values)}
            fn divide_batch(&mut self,requests:&[TadpoleDivisionRequest<'_,$value>],numerator:&[$value],output:&mut[$value],stable:&mut[bool])->Result<()> {self.engine.divide(requests,numerator,output,stable)}
            fn correction_batch(&mut self,requests:&[TadpoleCorrectionRequest<'_,$value>],output:&mut[$value],stable:&mut[bool])->Result<()> {self.engine.correction(requests,output,stable,None)}
            fn divide_with_corrections_batch(&mut self,divisions:&[TadpoleDivisionRequest<'_,$value>],numerator:&[$value],quotients:&mut[$value],quotient_stable:&mut[bool],requests:&[TadpoleCorrectionRequest<'_,$value>],corrections:&mut[$value],correction_stable:&mut[bool])->Result<()> {self.engine.divide_with_corrections(divisions,numerator,quotients,quotient_stable,requests,corrections,correction_stable)}
            fn combine_batch(&mut self,raw:&[$value],corrections:&[$value],rows:usize,output:&mut[$value])->Result<()> {self.engine.combine(raw,corrections,rows,output)}
            fn project_batch(&mut self,corrected:&[$value],e12:&[$value],output:&mut[$value])->Result<()> {self.engine.project(corrected,e12,output)}
        }
    };
}
implement!(SymbolicaTadpoleKernels<T>,Complex<T>,T:Real+SingleFloat,Complex<T>:Real+EvaluationDomain);
implement!(SymbolicaDualTadpoleKernels<T,D>,D,T:Real+SingleFloat,D:Real+DualNumber<Complex<T>>,Complex<T>:Real+EvaluationDomain);

#[cfg(feature = "native-kernels")]
impl<T: oneloop::rust_codegen::runtime::NativeFloat> SymbolicaTadpoleKernels<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    /// Opt into generated denominator arithmetic; other kernels stay interpreted.
    pub fn new_with_native_denominators(
        propagators: usize,
        rank: usize,
        prototype: T,
    ) -> Result<Self> {
        let mut kernels = Self::new(propagators, rank, prototype.clone())?;
        crate::native_kernels::install_denominator(
            &mut kernels.engine.denominator,
            crate::native_kernels::DenominatorPlan::T3,
            &prototype,
        )?;
        Ok(kernels)
    }

    /// Select denominators and both available parent-axis projection plans.
    /// Tadpole subtraction, residue projection and geometry stay interpreted.
    pub fn new_with_native_kernels(propagators: usize, rank: usize, prototype: T) -> Result<Self> {
        use crate::native_kernels::tadpoles::{TadpoleSourcePlan, install_projection};
        let mut kernels = Self::new_with_native_denominators(propagators, rank, prototype.clone())?;
        for (projection, source_size) in kernels.engine.source_projection.iter_mut().zip([3, 2]) {
            if let Some(projection) = projection {
                install_projection(
                    projection,
                    TadpoleSourcePlan::for_source(source_size)?,
                    &prototype,
                )?;
            }
        }
        Ok(kernels)
    }
}

#[cfg(feature = "native-kernels")]
impl<T: oneloop::rust_codegen::runtime::NativeFloat, D: Real + DualNumber<Complex<T>>>
    SymbolicaDualTadpoleKernels<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    /// Opt in for catalogued ordered Dual shapes; other shapes stay interpreted.
    pub fn new_with_native_denominators(
        propagators: usize,
        rank: usize,
        prototype: T,
        dual: &D,
    ) -> Result<Self> {
        let mut kernels = Self::new(propagators, rank, prototype.clone(), dual)?;
        crate::native_kernels::install_dual_denominator(
            &mut kernels.engine.denominator,
            crate::native_kernels::DenominatorPlan::T3,
            &prototype,
        )?;
        Ok(kernels)
    }

    /// Select available denominator and source-axis bodies for the supported
    /// ordered Dual shape. Other shapes retain the complete interpreter path.
    pub fn new_with_native_kernels(
        propagators: usize,
        rank: usize,
        prototype: T,
        dual: &D,
    ) -> Result<Self> {
        use crate::native_kernels::tadpoles::{TadpoleSourcePlan, install_dual_projection};
        let mut kernels =
            Self::new_with_native_denominators(propagators, rank, prototype.clone(), dual)?;
        for (projection, source_size) in kernels.engine.source_projection.iter_mut().zip([3, 2]) {
            if let Some(projection) = projection {
                install_dual_projection(
                    projection,
                    TadpoleSourcePlan::for_source(source_size)?,
                    &prototype,
                )?;
            }
        }
        Ok(kernels)
    }
}
