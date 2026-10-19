//! Generated numerical kernels for the core double-cut coordinator.

use crate::{
    BubbleSourceProjection, DenominatorExpansion, LaurentDivision, ResidueProjection,
    TriangleBubbleSubtraction,
    kernel::{CompiledDualKernel, CompiledKernel, ExactKernel, KernelExecutor},
    rank_specialization::{self, GuardedCorrection},
    value_checks::{
        DualChecks, IntegrationChecks, Negligible, OrdinaryChecks, ValueChecks,
        difference_dot_scale,
    },
};
use ninjars::{
    BatchShape, BubbleCut, BubbleDivisionRequest, BubbleReductionKernels,
    BubbleTriangleCorrectionRequest, Complex, Error, ExpansionKind, FloatLike, FourMomentum, Real,
    Result, SingleFloat,
};
use symbolica::{atom::Atom, domains::dual::DualNumber, evaluate::EvaluationDomain, symbol};

#[cfg(test)]
thread_local! {
    pub(crate) static DENOMINATOR_ROWS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

struct Engine<V, K, Checks, R> {
    n: usize,
    rank: usize,
    layout: LaurentDivision,
    coefficient_count: usize,
    source_count: usize,
    denominator: K,
    division: K,
    projection: K,
    swap: K,
    source_projection: K,
    correction: K,
    specialized_correction: Option<GuardedCorrection<K>>,
    pruned_swapped: Option<PrunedHeads<V, K>>,
    subtraction: K,
    zero: V,
    negligible: Negligible<R>,
    checks: std::marker::PhantomData<fn() -> Checks>,
    input: Vec<V>,
    denominators: Vec<V>,
    division_offsets: Vec<Option<usize>>,
    projections: Vec<V>,
    values: Vec<V>,
    eligible: Vec<usize>,
    active: Vec<usize>,
}

/// Configured projection-only bubble arithmetic, not evidence of dispatch.
///
/// Counts are quotient/correction heads per orientation, before Dual flattening.
/// Full callbacks, parameter arrays and raw APIs remain unchanged. All selected
/// graphs use the Symbolica interpreter independently of generic native coverage
/// and native row scheduling. Inactive bubble stages report zero heads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrunedSwappedArithmetic {
    pub full_columns: usize,
    pub swapped_columns: usize,
    pub guarded_rank_correction: bool,
}

impl PrunedSwappedArithmetic {
    pub const fn backend(&self) -> &'static str {
        "symbolica-interpreter"
    }

    pub(crate) fn for_topology(n: usize, rank: usize, guarded: bool) -> Self {
        // These are the existing T2 layout cardinalities, not numerical algebra.
        let degree = (n >= 2)
            .then(|| rank.checked_add(2)?.checked_sub(n))
            .flatten();
        let full_columns = degree.map_or(0, |d| [1, 3, 7, 13, 22][d]);
        let swapped_columns = degree.map_or(0, |d| [0, 1, 3, 7, 13][d]);
        Self {
            full_columns,
            swapped_columns,
            guarded_rank_correction: guarded,
        }
    }
}

fn positive_heads(layout: &LaurentDivision) -> Vec<usize> {
    let ExpansionKind::T2 { powers } = layout.quotient_expansion() else {
        unreachable!("bubble quotient is T2")
    };
    powers
        .iter()
        .enumerate()
        .filter_map(|(i, &(t, _, _))| (t > 0).then_some(i))
        .collect()
}

fn selected_heads(exact: &ExactKernel, heads: &[usize]) -> ExactKernel {
    ExactKernel {
        parameters: exact.parameters.clone(),
        expressions: heads
            .iter()
            .map(|&i| exact.expressions[i].clone())
            .collect(),
    }
}

#[derive(Clone, Copy)]
enum PrunedGraph {
    Division,
    Correction { specialized: bool },
}

struct PrunedHeads<V, K> {
    heads: Vec<usize>,
    // No empty graph is compiled or invoked when d <= 0.
    division: Option<K>,
    correction: Option<K>,
    specialized_correction: Option<K>,
    input: Vec<V>,
    values: Vec<V>,
    selected: Vec<usize>,
}

impl<V: FloatLike, K: KernelExecutor<V>> PrunedHeads<V, K> {
    #[allow(clippy::too_many_arguments)]
    fn evaluate(
        &mut self,
        full: &mut K,
        graph: PrunedGraph,
        input: &[V],
        input_columns: usize,
        rows: &[usize],
        orientation: impl Fn(usize) -> usize,
        output: &mut [V],
        width: usize,
        zero: &V,
    ) -> Result<()> {
        for swapped in [false, true] {
            self.selected.clear();
            self.selected.extend(
                rows.iter().enumerate().filter_map(|(index, &row)| {
                    ((orientation(row) == 1) == swapped).then_some(index)
                }),
            );
            if self.selected.is_empty() {
                continue;
            }
            let columns = if swapped { self.heads.len() } else { width };
            if columns != 0 {
                self.input.clear();
                for &index in &self.selected {
                    self.input.extend_from_slice(
                        &input[index * input_columns..(index + 1) * input_columns],
                    );
                }
                self.values
                    .resize(self.selected.len() * columns, zero.clone());
                let kernel = if swapped {
                    match graph {
                        PrunedGraph::Division => self.division.as_mut(),
                        PrunedGraph::Correction { specialized: true } => {
                            self.specialized_correction.as_mut()
                        }
                        PrunedGraph::Correction { specialized: false } => self.correction.as_mut(),
                    }
                    .expect("nonempty selected graph was compiled")
                } else {
                    &mut *full
                };
                kernel.evaluate(&self.input, &mut self.values, self.selected.len())?;
            }
            for (group_row, &index) in self.selected.iter().enumerate() {
                let out = &mut output[index * width..(index + 1) * width];
                if swapped {
                    out.fill(zero.clone());
                    for (column, &head) in self.heads.iter().enumerate() {
                        out[head] = self.values[group_row * columns + column].clone();
                    }
                } else {
                    out.clone_from_slice(
                        &self.values[group_row * columns..(group_row + 1) * columns],
                    );
                }
            }
        }
        Ok(())
    }
}

fn parameters(namespace: &str, count: usize) -> Vec<Atom> {
    (0..count)
        .map(|i| Atom::var(symbol!(format!("{namespace}::p{i}"))))
        .collect()
}

fn append_denominator<V: Clone>(
    output: &mut Vec<V>,
    vectors: &[FourMomentum<V>; 4],
    parameters: &[V; 3],
    shifts: &[FourMomentum<V>],
    masses: &[V],
    anchor: usize,
    line: usize,
) {
    for vector in vectors {
        output.extend(vector.components().into_iter().cloned());
    }
    output.extend_from_slice(parameters);
    output.extend(shifts[anchor].components().into_iter().cloned());
    output.push(masses[anchor].clone());
    output.extend(shifts[line].components().into_iter().cloned());
    output.push(masses[line].clone());
}

impl<
    V: FloatLike,
    K: KernelExecutor<V>,
    R: Real + SingleFloat,
    Checks: IntegrationChecks<V> + ValueChecks<V, Real = R>,
> Engine<V, K, Checks, R>
{
    #[cfg(feature = "native-kernels")]
    fn set_native_batch_execution(&mut self, policy: crate::BatchExecutionPolicy) {
        for kernel in [
            &mut self.denominator,
            &mut self.division,
            &mut self.projection,
            &mut self.swap,
            &mut self.source_projection,
            &mut self.correction,
            &mut self.subtraction,
        ] {
            kernel.set_native_batch_execution(policy);
        }
        if let Some(plan) = &mut self.specialized_correction {
            plan.kernel.set_native_batch_execution(policy);
        }
        if let Some(pruned) = &mut self.pruned_swapped {
            for kernel in [
                &mut pruned.division,
                &mut pruned.correction,
                &mut pruned.specialized_correction,
            ]
            .into_iter()
            .flatten()
            {
                kernel.set_native_batch_execution(policy);
            }
        }
    }

    fn values_are_finite(&self, values: &[V]) -> bool {
        values.iter().all(Checks::finite)
    }

    fn new(
        n: usize,
        rank: usize,
        prototype: &V,
        mut compile: impl FnMut(&ExactKernel) -> Result<K>,
    ) -> Result<Self> {
        let layout = LaurentDivision::new(n, rank, 2)?;
        let denominator = compile(&DenominatorExpansion::new(2)?.exact)?;
        let division = compile(&layout.exact)?;
        let projection_plan = ResidueProjection::new(&layout)?;
        let coefficient_count = projection_plan.coefficient_count();
        let projection = compile(&projection_plan.exact)?;
        let p = parameters("ninjars_bubble_swap", 5);
        let swap = compile(&ExactKernel {
            expressions: p[..4].iter().map(|v| -v / (&p[4] * 2)).collect(),
            parameters: p,
        })?;
        let source_projection = compile(&BubbleSourceProjection::new().exact)?;
        let source_count = ninjars::residue_coefficient_count(3, layout.rank_class());
        let correction = compile(&TriangleBubbleSubtraction::for_quotient(&layout)?.exact)?;
        let width = layout.quotient_expansion().len();
        let p = parameters("ninjars_bubble_parent_sum", width * (n - 1));
        let expressions = (0..width)
            .map(|column| {
                (0..n - 2).fold(p[column].clone(), |value, parent| {
                    value - &p[width * (parent + 1) + column]
                })
            })
            .collect();
        let subtraction = compile(&ExactKernel {
            expressions,
            parameters: p,
        })?;
        Ok(Self {
            n,
            rank,
            layout,
            coefficient_count,
            source_count,
            denominator,
            division,
            projection,
            swap,
            source_projection,
            correction,
            specialized_correction: None,
            pruned_swapped: None,
            subtraction,
            zero: prototype.zero(),
            negligible: Negligible::new(&Checks::magnitude(prototype)),
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

    fn valid(&self, value: &V) -> bool {
        Checks::shape_matches(value, &self.zero) && Checks::finite(value)
    }

    /// Whether the leading Laurent coefficient `d0 = 2 v_t.(V_line - V_anchor)`
    /// of an uncut denominator (Ninja's `denc.d0`) is negligible.
    fn negligible_leading(
        &self,
        vectors: &[FourMomentum<V>; 4],
        shifts: &[FourMomentum<V>],
        anchor: usize,
        line: usize,
        d0: &V,
    ) -> bool {
        let scale = difference_dot_scale::<V, Checks>(&vectors[2], &shifts[line], &shifts[anchor]);
        self.negligible.value::<V, Checks>(d0, &scale)
    }

    fn specialize_correction(
        &mut self,
        mut compile: impl FnMut(&ExactKernel) -> Result<K>,
    ) -> Result<()> {
        rank_specialization::require_supported(self.n, self.rank)?;
        let plan =
            rank_specialization::RankSpecializedCorrectionPlan::TriangleToBubbleN5R4.prepare()?;
        if self.layout.quotient_expansion() != &plan.output {
            return Err(Error::Unsupported(
                "guarded bubble correction quotient differs from its semantic plan".into(),
            ));
        }
        if let Some(pruned) = &mut self.pruned_swapped {
            pruned.specialized_correction = if pruned.heads.is_empty() {
                None
            } else {
                Some(compile(&selected_heads(&plan.exact, &pruned.heads))?)
            };
        }
        self.specialized_correction = Some(GuardedCorrection {
            kernel: compile(&plan.exact)?,
            zero_slots: plan.zero_slots,
        });
        Ok(())
    }

    fn prune_swapped(&mut self, mut compile: impl FnMut(&ExactKernel) -> Result<K>) -> Result<()> {
        let heads = positive_heads(&self.layout);
        let (division, correction, specialized_correction) = if heads.is_empty() {
            (None, None, None)
        } else {
            let division = compile(&selected_heads(&self.layout.exact, &heads))?;
            let correction = compile(&selected_heads(
                &TriangleBubbleSubtraction::for_quotient(&self.layout)?.exact,
                &heads,
            ))?;
            let specialized = if self.specialized_correction.is_some() {
                let plan = rank_specialization::RankSpecializedCorrectionPlan::TriangleToBubbleN5R4
                    .prepare()?;
                Some(compile(&selected_heads(&plan.exact, &heads))?)
            } else {
                None
            };
            (Some(division), Some(correction), specialized)
        };
        self.pruned_swapped = Some(PrunedHeads {
            heads,
            division,
            correction,
            specialized_correction,
            input: Vec::new(),
            values: Vec::new(),
            selected: Vec::new(),
        });
        Ok(())
    }

    fn pruned_swapped_arithmetic(&self) -> Option<PrunedSwappedArithmetic> {
        self.pruned_swapped
            .as_ref()
            .map(|plan| PrunedSwappedArithmetic {
                full_columns: self.layout.quotient_expansion().len(),
                swapped_columns: plan.heads.len(),
                guarded_rank_correction: self.specialized_correction.is_some(),
            })
    }

    fn swapped(&mut self, cuts: &[&BubbleCut<V>], output: &mut [FourMomentum<V>]) -> Result<()> {
        BatchShape {
            rows: cuts.len(),
            columns: 1,
        }
        .validate("swapped bubble directions", output.len())?;
        self.input.clear();
        for cut in cuts {
            self.input
                .extend(cut.basis.e3.components().into_iter().cloned());
            self.input.push(cut.basis.e12.clone());
        }
        self.values.resize(cuts.len() * 4, self.zero.clone());
        self.swap
            .evaluate(&self.input, &mut self.values, cuts.len())?;
        for (out, p) in output.iter_mut().zip(self.values.as_chunks::<4>().0) {
            *out = FourMomentum::from_args(p[0].clone(), p[1].clone(), p[2].clone(), p[3].clone());
        }
        Ok(())
    }

    fn divide(
        &mut self,
        requests: &[BubbleDivisionRequest<'_, V>],
        numerator: &[V],
        output: &mut [V],
        stable: &mut [bool],
        prune_swapped: bool,
    ) -> Result<()> {
        let rows = requests.len();
        let columns = self.layout.numerator_expansion().len();
        let width = self.layout.quotient_expansion().len();
        let uncut = self.layout.uncut_count();
        BatchShape { rows, columns }.validate("bubble numerator coefficients", numerator.len())?;
        BatchShape {
            rows,
            columns: width,
        }
        .validate("bubble quotient coefficients", output.len())?;
        BatchShape { rows, columns: 1 }.validate("bubble division status", stable.len())?;
        for r in requests {
            if r.partition.len() != 2
                || r.complement.len() != uncut
                || r.shifts.len() != self.n
                || r.masses_squared.len() != self.n
                || r.partition.iter().chain(r.complement).any(|&i| i >= self.n)
                || (0..self.n).any(|line| {
                    r.partition
                        .iter()
                        .chain(r.complement)
                        .filter(|&&i| i == line)
                        .count()
                        != 1
                })
            {
                return Err(Error::Expansion(
                    "bubble denominator request does not match its topology".into(),
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
                .chain(r.parameters)
                .chain(&numerator[row * columns..(row + 1) * columns])
                .all(|v| self.valid(v));
            if !stable[row] {
                continue;
            }
            self.division_offsets[row] = Some(self.eligible.len() * uncut * 7);
            self.eligible.push(row);
            for &line in r.complement {
                append_denominator(
                    &mut self.input,
                    r.vectors,
                    r.parameters,
                    r.shifts,
                    r.masses_squared,
                    r.partition[0],
                    line,
                );
            }
        }
        self.denominators
            .resize(self.eligible.len() * uncut * 7, self.zero.clone());
        #[cfg(test)]
        DENOMINATOR_ROWS.with(|count| count.set(count.get() + self.eligible.len() * uncut));
        self.denominator.evaluate(
            &self.input,
            &mut self.denominators,
            self.eligible.len() * uncut,
        )?;
        self.input.clear();
        self.active.clear();
        for (index, &row) in self.eligible.iter().enumerate() {
            let r = &requests[row];
            let den = &self.denominators[index * uncut * 7..(index + 1) * uncut * 7];
            // Leading coefficient 2 v_t.(V_j - V_0) (Ninja's `denc.d0` check).
            stable[row] = den.iter().all(Checks::finite)
                && den
                    .as_chunks::<7>()
                    .0
                    .iter()
                    .zip(r.complement)
                    .all(|(d, &line)| {
                        !self.negligible_leading(r.vectors, r.shifts, r.partition[0], line, &d[0])
                    });
            if stable[row] {
                self.active.push(row);
                self.input
                    .extend_from_slice(&numerator[row * columns..(row + 1) * columns]);
                self.input.extend_from_slice(den);
            }
        }
        self.values
            .resize(self.active.len() * width, self.zero.clone());
        if prune_swapped {
            self.pruned_swapped
                .as_mut()
                .expect("projection-only pruning enabled")
                .evaluate(
                    &mut self.division,
                    PrunedGraph::Division,
                    &self.input,
                    self.layout.input_columns(),
                    &self.active,
                    |row| row % 2,
                    &mut self.values,
                    width,
                    &self.zero,
                )?;
        } else {
            self.division
                .evaluate(&self.input, &mut self.values, self.active.len())?;
        }
        for (index, &row) in self.active.iter().enumerate() {
            let values = &self.values[index * width..(index + 1) * width];
            stable[row] = values.iter().all(Checks::finite);
            if stable[row] {
                output[row * width..(row + 1) * width].clone_from_slice(values);
            }
        }
        Ok(())
    }

    fn corrections(
        &mut self,
        requests: &[BubbleTriangleCorrectionRequest<'_, V>],
        output: &mut [V],
        stable: &mut [bool],
        prepared: Option<(&[BubbleDivisionRequest<'_, V>], &[bool])>,
        prune_swapped: bool,
    ) -> Result<()> {
        let rows = requests.len();
        let width = self.layout.quotient_expansion().len();
        BatchShape {
            rows,
            columns: width,
        }
        .validate("triangle bubble corrections", output.len())?;
        BatchShape { rows, columns: 1 }
            .validate("triangle bubble correction status", stable.len())?;
        let parents = self.n - 2;
        let selected = |row: usize| {
            prepared.is_none_or(|(_, stable)| {
                let target = row / parents / 2;
                stable[2 * target..2 * target + 2].iter().all(|&v| v)
            })
        };
        for (row, r) in requests.iter().enumerate() {
            if !selected(row) {
                continue;
            }
            if r.partition.len() != 2
                || r.source_partition.len() != 3
                || r.shifts.len() != self.n
                || r.masses_squared.len() != self.n
                || r.source.coefficients.len() != self.source_count
                || r.source.id.lane != r.id.lane
                || r.partition
                    .iter()
                    .chain(r.source_partition)
                    .any(|&i| i >= self.n)
                || r.partition[0] == r.partition[1]
                || r.source_partition
                    .iter()
                    .any(|i| r.source_partition.iter().filter(|j| j == &i).count() != 1)
                || !r.partition.iter().all(|i| r.source_partition.contains(i))
            {
                return Err(Error::Expansion("triangle parent does not contain the requested bubble or has incompatible data".into()));
            }
        }
        let reuse = prepared.is_some_and(|(divisions, _)| {
            requests
                .iter()
                .enumerate()
                .filter(|(row, _)| selected(*row))
                .all(|(row, r)| {
                    let d = &divisions[row / parents];
                    r.partition == d.partition
                        && std::ptr::eq(r.vectors, d.vectors)
                        && std::ptr::eq(r.parameters, d.parameters)
                        && std::ptr::eq(r.shifts, d.shifts)
                        && std::ptr::eq(r.masses_squared, d.masses_squared)
                })
        });
        self.input.clear();
        self.eligible.clear();
        stable.fill(false);
        for (row, r) in requests.iter().enumerate() {
            if !selected(row) {
                continue;
            }
            stable[row] = r
                .shifts
                .iter()
                .chain(r.vectors)
                .chain([&r.source.geometry.basis.e3, &r.source.geometry.basis.e4])
                .flat_map(FourMomentum::components)
                .chain(r.masses_squared)
                .chain(r.parameters)
                .chain(&r.source.coefficients)
                .all(|v| self.valid(v));
            if !stable[row] {
                continue;
            }
            self.eligible.push(row);
            let line = *r
                .source_partition
                .iter()
                .find(|i| !r.partition.contains(i))
                .unwrap();
            if !reuse {
                append_denominator(
                    &mut self.input,
                    r.vectors,
                    r.parameters,
                    r.shifts,
                    r.masses_squared,
                    r.partition[0],
                    line,
                );
            }
        }
        if !reuse {
            self.denominators
                .resize(self.eligible.len() * 7, self.zero.clone());
            #[cfg(test)]
            DENOMINATOR_ROWS.with(|count| count.set(count.get() + self.eligible.len()));
            self.denominator
                .evaluate(&self.input, &mut self.denominators, self.eligible.len())?;
        }
        self.input.clear();
        for &row in &self.eligible {
            let r = &requests[row];
            for v in r.vectors {
                self.input.extend(v.components().into_iter().cloned());
            }
            self.input.extend(
                r.shifts[r.source_partition[0]]
                    .components()
                    .into_iter()
                    .cloned(),
            );
            self.input
                .extend(r.source.geometry.basis.e3.components().into_iter().cloned());
            self.input
                .extend(r.source.geometry.basis.e4.components().into_iter().cloned());
        }
        self.projections
            .resize(self.eligible.len() * 8, self.zero.clone());
        self.source_projection
            .evaluate(&self.input, &mut self.projections, self.eligible.len())?;
        self.input.clear();
        self.active.clear();
        for (index, &row) in self.eligible.iter().enumerate() {
            let offset = if reuse {
                let division = row / parents;
                let d = &prepared.unwrap().0[division];
                let r = &requests[row];
                let line = r
                    .source_partition
                    .iter()
                    .find(|i| !r.partition.contains(i))
                    .unwrap();
                self.division_offsets[division].unwrap()
                    + 7 * d.complement.iter().position(|i| i == line).unwrap()
            } else {
                index * 7
            };
            let den = &self.denominators[offset..offset + 7];
            let proj = &self.projections[index * 8..(index + 1) * 8];
            let r = &requests[row];
            let line = *r
                .source_partition
                .iter()
                .find(|i| !r.partition.contains(i))
                .unwrap();
            stable[row] = den.iter().chain(proj).all(Checks::finite)
                && !self.negligible_leading(r.vectors, r.shifts, r.partition[0], line, &den[0]);
            if stable[row] {
                self.active.push(row);
                self.input
                    .extend_from_slice(&requests[row].source.coefficients);
                self.input.extend_from_slice(proj);
                self.input.extend_from_slice(requests[row].parameters);
                self.input.extend_from_slice(den);
            }
        }
        self.values
            .resize(self.active.len() * width, self.zero.clone());
        // An arbitrary public stage caller may supply coefficients outside the
        // rank subspace. Test every derivative component of every active row;
        // a single violation keeps the entire batch on its generic backend.
        let specialized = self.specialized_correction.as_ref().is_some_and(|plan| {
            self.active.iter().all(|&row| {
                plan.zero_slots
                    .iter()
                    .all(|&slot| Checks::is_zero(&requests[row].source.coefficients[slot]))
            })
        });
        let correction = if specialized {
            &mut self.specialized_correction.as_mut().unwrap().kernel
        } else {
            &mut self.correction
        };
        #[cfg(test)]
        rank_specialization::record(0, specialized, self.active.len());
        if prune_swapped {
            self.pruned_swapped
                .as_mut()
                .expect("projection-only pruning enabled")
                .evaluate(
                    correction,
                    PrunedGraph::Correction { specialized },
                    &self.input,
                    self.source_count + 18,
                    &self.active,
                    |row| (row / parents) % 2,
                    &mut self.values,
                    width,
                    &self.zero,
                )?;
        } else {
            correction.evaluate(&self.input, &mut self.values, self.active.len())?;
        }
        for (index, &row) in self.active.iter().enumerate() {
            let values = &self.values[index * width..(index + 1) * width];
            stable[row] = values.iter().all(Checks::finite);
            if stable[row] {
                output[row * width..(row + 1) * width].clone_from_slice(values);
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn divide_with_corrections(
        &mut self,
        divisions: &[BubbleDivisionRequest<'_, V>],
        numerator: &[V],
        quotients: &mut [V],
        quotient_stable: &mut [bool],
        requests: &[BubbleTriangleCorrectionRequest<'_, V>],
        corrections: &mut [V],
        correction_stable: &mut [bool],
        prune_swapped: bool,
    ) -> Result<()> {
        if !divisions.len().is_multiple_of(2) {
            return Err(Error::Expansion(
                "bubble division orientations are incomplete".into(),
            ));
        }
        BatchShape {
            rows: divisions.len(),
            columns: self.n - 2,
        }
        .validate("bubble parent request rows", requests.len())?;
        BatchShape {
            rows: requests.len(),
            columns: self.layout.quotient_expansion().len(),
        }
        .validate("bubble parent correction output", corrections.len())?;
        BatchShape {
            rows: requests.len(),
            columns: 1,
        }
        .validate("bubble parent correction status", correction_stable.len())?;
        self.divide(
            divisions,
            numerator,
            quotients,
            quotient_stable,
            prune_swapped,
        )?;
        self.corrections(
            requests,
            corrections,
            correction_stable,
            Some((divisions, quotient_stable)),
            prune_swapped,
        )
    }

    fn subtract(
        &mut self,
        quotients: &[V],
        corrections: &[V],
        output: &mut [V],
        rows: usize,
    ) -> Result<()> {
        let width = self.layout.quotient_expansion().len();
        let count = self.n - 2;
        BatchShape {
            rows,
            columns: width,
        }
        .validate("uncorrected bubble quotients", quotients.len())?;
        BatchShape {
            rows,
            columns: width * count,
        }
        .validate("bubble parent corrections", corrections.len())?;
        BatchShape {
            rows,
            columns: width,
        }
        .validate("corrected bubble quotients", output.len())?;
        self.input.clear();
        for row in 0..rows {
            self.input
                .extend_from_slice(&quotients[row * width..(row + 1) * width]);
            self.input
                .extend_from_slice(&corrections[row * width * count..(row + 1) * width * count]);
        }
        self.subtraction.evaluate(&self.input, output, rows)
    }

    fn project(&mut self, quotients: &[V], e12: &[V], output: &mut [V]) -> Result<()> {
        let rows = e12.len();
        let columns = 2 * self.layout.quotient_expansion().len();
        BatchShape { rows, columns }.validate("bubble projection quotients", quotients.len())?;
        BatchShape {
            rows,
            columns: self.coefficient_count,
        }
        .validate("bubble residue coefficients", output.len())?;
        self.input.clear();
        for (row, g) in e12.iter().enumerate() {
            self.input
                .extend_from_slice(&quotients[row * columns..(row + 1) * columns]);
            self.input.push(g.clone());
        }
        self.projection.evaluate(&self.input, output, rows)
    }
}

/// Bubble reduction arithmetic, with precision-aware singular/nonfinite
/// isolation. Precision rescue remains separate acceptance work.
pub struct SymbolicaBubbleKernels<T> {
    engine: Engine<Complex<T>, CompiledKernel<T>, OrdinaryChecks, T>,
}
impl<T: Real + SingleFloat> SymbolicaBubbleKernels<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    #[cfg(feature = "native-kernels")]
    pub(crate) fn with_native_batch_execution(
        mut self,
        policy: crate::BatchExecutionPolicy,
    ) -> Self {
        self.engine.set_native_batch_execution(policy);
        self
    }

    pub fn new(propagators: usize, rank: usize, prototype: T) -> Result<Self> {
        let zero = Complex::new(prototype.zero(), prototype.zero());
        Ok(Self {
            engine: Engine::new(propagators, rank, &zero, |exact| exact.compile(&prototype))?,
        })
    }

    /// Restrict only the swapped arithmetic consumed by complete residue
    /// projection. Callbacks and all existing raw APIs retain their full layouts.
    /// Selected graphs remain interpreted even with generic native kernels.
    ///
    /// Full inputs, denominators and source projections are still validated.
    /// Overflow confined to discarded arithmetic no longer rejects a lane;
    /// complete live residue coefficients and publication checks remain intact.
    pub fn with_pruned_swapped_arithmetic(mut self) -> Result<Self> {
        let prototype = self.engine.zero.re.clone();
        self.engine
            .prune_swapped(|exact| exact.compile(&prototype))?;
        Ok(self)
    }

    pub fn pruned_swapped_arithmetic(&self) -> Option<PrunedSwappedArithmetic> {
        self.engine.pruned_swapped_arithmetic()
    }

    /// Opt into interpreted rank-specialized triangle corrections for n=5/r=4.
    /// Every active parent's structural zeros are checked exactly, including
    /// all Dual components in the Dual variant. Any violation selects the
    /// existing generic batch backend. Raw APIs keep both orientations complete;
    /// projection-only pruning is configured independently.
    pub fn with_rank_specialized_corrections(mut self) -> Result<Self> {
        let prototype = self.engine.zero.re.clone();
        self.engine
            .specialize_correction(|exact| exact.compile(&prototype))?;
        Ok(self)
    }

    pub fn rank_specialized_corrections_enabled(&self) -> bool {
        self.engine.specialized_correction.is_some()
    }

    /// Backend of the guarded alternate; failed guards use the generic backend.
    #[cfg(feature = "native-kernels")]
    pub fn rank_specialized_correction_backend(&self) -> Option<crate::KernelBackend> {
        self.engine
            .specialized_correction
            .as_ref()
            .map(|plan| plan.kernel.backend())
    }
}

/// Static/dynamic Dual execution of the same exact bubble graphs.
pub struct SymbolicaDualBubbleKernels<T, D> {
    engine: Engine<D, CompiledDualKernel<T, D>, DualChecks<T>, T>,
}
impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> SymbolicaDualBubbleKernels<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    #[cfg(all(test, feature = "native-kernels"))]
    pub(crate) fn native_workspace_rows(&self) -> Vec<(&'static str, usize)> {
        let mut result = Vec::new();
        for (name, kernel) in [
            ("denominator", &self.engine.denominator),
            ("source_projection", &self.engine.source_projection),
            ("correction", &self.engine.correction),
        ] {
            if let Some((rows, _)) = kernel.probe_workspace() {
                result.push((name, rows));
            }
        }
        if let Some(plan) = &self.engine.specialized_correction
            && let Some((rows, _)) = plan.kernel.probe_workspace()
        {
            result.push(("rank_alternate", rows));
        }
        result
    }

    #[cfg(feature = "native-kernels")]
    pub(crate) fn with_native_batch_execution(
        mut self,
        policy: crate::BatchExecutionPolicy,
    ) -> Self {
        self.engine.set_native_batch_execution(policy);
        self
    }

    pub fn new(propagators: usize, rank: usize, prototype: T, dual: &D) -> Result<Self> {
        Ok(Self {
            engine: Engine::new(propagators, rank, dual, |exact| {
                exact.compile_dual(&prototype, dual)
            })?,
        })
    }

    /// See [`SymbolicaBubbleKernels::with_pruned_swapped_arithmetic`]. Every
    /// derivative component and the complete ordered Dual shape are retained.
    pub fn with_pruned_swapped_arithmetic(mut self) -> Result<Self> {
        let dual = self.engine.zero.clone();
        let prototype = self.engine.division.scalar_prototype().clone();
        self.engine
            .prune_swapped(|exact| exact.compile_dual(&prototype, &dual))?;
        Ok(self)
    }

    pub fn pruned_swapped_arithmetic(&self) -> Option<PrunedSwappedArithmetic> {
        self.engine.pruned_swapped_arithmetic()
    }

    /// See [`SymbolicaBubbleKernels::with_rank_specialized_corrections`].
    pub fn with_rank_specialized_corrections(mut self) -> Result<Self> {
        let dual = self.engine.zero.clone();
        let prototype = self.engine.division.scalar_prototype().clone();
        self.engine
            .specialize_correction(|exact| exact.compile_dual(&prototype, &dual))?;
        Ok(self)
    }

    pub fn rank_specialized_corrections_enabled(&self) -> bool {
        self.engine.specialized_correction.is_some()
    }

    /// Backend of the guarded alternate; failed guards use the generic backend.
    #[cfg(feature = "native-kernels")]
    pub fn rank_specialized_correction_backend(&self) -> Option<crate::KernelBackend> {
        self.engine
            .specialized_correction
            .as_ref()
            .map(|plan| plan.kernel.backend())
    }
}

macro_rules! implementation {
    ($name:ident<$($g:ident),+>,$value:ty,$($bounds:tt)*) => {
        impl<$($g),+> BubbleReductionKernels<$value> for $name<$($g),+> where $($bounds)* {
            fn propagators(&self)->usize {self.engine.n}
            fn rank(&self)->usize {self.engine.rank}
            fn numerator_expansion(&self)->&ExpansionKind {self.engine.layout.numerator_expansion()}
            fn quotient_columns(&self)->usize {self.engine.layout.quotient_expansion().len()}
            fn coefficient_count(&self)->usize {self.engine.coefficient_count}
            fn swapped_effective_batch(&mut self,cuts:&[&BubbleCut<$value>],output:&mut[FourMomentum<$value>])->Result<()> {self.engine.swapped(cuts,output)}
            fn divide_batch(&mut self,requests:&[BubbleDivisionRequest<'_,$value>],numerator:&[$value],output:&mut[$value],stable:&mut[bool])->Result<()> {self.engine.divide(requests,numerator,output,stable,false)}
            fn triangle_corrections_batch(&mut self,requests:&[BubbleTriangleCorrectionRequest<'_,$value>],output:&mut[$value],stable:&mut[bool])->Result<()> {self.engine.corrections(requests,output,stable,None,false)}
            fn divide_with_corrections_batch(&mut self,divisions:&[BubbleDivisionRequest<'_,$value>],numerator:&[$value],quotients:&mut[$value],quotient_stable:&mut[bool],requests:&[BubbleTriangleCorrectionRequest<'_,$value>],corrections:&mut[$value],correction_stable:&mut[bool])->Result<()> {self.engine.divide_with_corrections(divisions,numerator,quotients,quotient_stable,requests,corrections,correction_stable,false)}
            fn divide_with_corrections_for_projection_batch(&mut self,divisions:&[BubbleDivisionRequest<'_,$value>],numerator:&[$value],quotients:&mut[$value],quotient_stable:&mut[bool],requests:&[BubbleTriangleCorrectionRequest<'_,$value>],corrections:&mut[$value],correction_stable:&mut[bool])->Result<()> {
                let prune = self.engine.pruned_swapped.is_some();
                self.engine.divide_with_corrections(divisions,numerator,quotients,quotient_stable,requests,corrections,correction_stable,prune)
            }
            fn subtract_batch(&mut self,quotients:&[$value],corrections:&[$value],output:&mut[$value],rows:usize)->Result<()> {self.engine.subtract(quotients,corrections,output,rows)}
            fn project_batch(&mut self,quotients:&[$value],e12:&[$value],output:&mut[$value])->Result<()> {self.engine.project(quotients,e12,output)}
            fn values_are_finite(&self,values:&[$value])->bool {self.engine.values_are_finite(values)}
        }
    };
}
implementation!(SymbolicaBubbleKernels<T>,Complex<T>,T:Real+SingleFloat,Complex<T>:Real+EvaluationDomain);
implementation!(SymbolicaDualBubbleKernels<T,D>,D,T:Real+SingleFloat,D:Real+DualNumber<Complex<T>>,Complex<T>:Real+EvaluationDomain);

#[cfg(feature = "native-kernels")]
impl<T: oneloop::rust_codegen::runtime::NativeFloat> SymbolicaBubbleKernels<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    /// Opt into generated n=5/r=4 correction bodies with the same exact-zero
    /// guards. Generic native selections are retained for whole-batch fallback.
    pub fn with_native_rank_specialized_corrections(self) -> Result<Self> {
        let mut kernels = self.with_rank_specialized_corrections()?;
        let prototype = kernels.engine.zero.re.clone();
        crate::native_kernels::rank_specialization::install_ordinary(
            &mut kernels
                .engine
                .specialized_correction
                .as_mut()
                .expect("specialization installed")
                .kernel,
            crate::rank_specialization::RankSpecializedCorrectionPlan::TriangleToBubbleN5R4,
            &prototype,
        )?;
        Ok(kernels)
    }

    /// Opt into generated denominator arithmetic; other kernels stay interpreted.
    pub fn new_with_native_denominators(
        propagators: usize,
        rank: usize,
        prototype: T,
    ) -> Result<Self> {
        let mut kernels = Self::new(propagators, rank, prototype.clone())?;
        crate::native_kernels::install_denominator(
            &mut kernels.engine.denominator,
            crate::native_kernels::DenominatorPlan::T2,
            &prototype,
        )?;
        Ok(kernels)
    }

    /// Select denominators, shared source-axis projections, and the exact
    /// triangle-parent correction for this topology's quotient degree. Rank
    /// n+2 has no catalogued correction body and keeps the interpreter.
    pub fn new_with_native_kernels(propagators: usize, rank: usize, prototype: T) -> Result<Self> {
        use crate::native_kernels::bubbles::{BubblePlan, install};
        let mut kernels = Self::new_with_native_denominators(propagators, rank, prototype.clone())?;
        install(
            &mut kernels.engine.source_projection,
            BubblePlan::SourceProjection,
            &prototype,
        )?;
        if let Some(plan) = BubblePlan::correction(
            kernels.engine.layout.rank_class(),
            kernels.engine.layout.quotient_expansion(),
        )? {
            install(&mut kernels.engine.correction, plan, &prototype)?;
        }
        Ok(kernels)
    }
}

#[cfg(feature = "native-kernels")]
impl<T: oneloop::rust_codegen::runtime::NativeFloat, D: Real + DualNumber<Complex<T>>>
    SymbolicaDualBubbleKernels<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    /// Opt into generated n=5/r=4 correction bodies with the same exact-zero
    /// guards. Generic native selections are retained for whole-batch fallback.
    /// Only the catalogued ordered five-component shape is generated; other
    /// shapes use the interpreted alternate and report that backend.
    pub fn with_native_rank_specialized_corrections(self) -> Result<Self> {
        let mut kernels = self.with_rank_specialized_corrections()?;
        let prototype = kernels.engine.division.scalar_prototype().clone();
        crate::native_kernels::rank_specialization::install_dual(
            &mut kernels
                .engine
                .specialized_correction
                .as_mut()
                .expect("specialization installed")
                .kernel,
            crate::rank_specialization::RankSpecializedCorrectionPlan::TriangleToBubbleN5R4,
            &prototype,
        )?;
        Ok(kernels)
    }

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
            crate::native_kernels::DenominatorPlan::T2,
            &prototype,
        )?;
        Ok(kernels)
    }
}

#[cfg(test)]
#[path = "denominator_reuse_tests.rs"]
mod denominator_reuse_tests;

#[cfg(test)]
#[path = "swapped_arithmetic_tests.rs"]
mod swapped_arithmetic_tests;
