//! Symbolica-generated invariant, master-weight and contraction arithmetic.

use crate::{
    MasterContraction,
    kernel::{CompiledDualKernel, CompiledKernel, ExactKernel, KernelExecutor},
    value_checks::{DualChecks, IntegrationChecks, OrdinaryChecks},
};
use ninjars::{
    BatchShape, Complex, FloatLike, FourMomentum, IntegratedContribution, IntegrationKernels,
    Laurent, MasterIntegral, RankClass, Real, ResidueIntegrationRequest, Result, SingleFloat,
};
use std::marker::PhantomData;
use symbolica::{atom::Atom, domains::dual::DualNumber, evaluate::EvaluationDomain, symbol};

struct Engine<V, K, Checks> {
    size: usize,
    plan: MasterContraction,
    momentum: K,
    weights: K,
    contraction: K,
    /// The terms of the rational weight, for the opt-in cancellation test.
    rational_terms: Option<(K, usize)>,
    zero: V,
    checks: PhantomData<fn() -> Checks>,
    active: Vec<usize>,
    arguments: Vec<V>,
    kinematics: Vec<V>,
    values: Vec<V>,
    term_values: Vec<V>,
}

impl<V: FloatLike, K: KernelExecutor<V>, Checks: IntegrationChecks<V>> Engine<V, K, Checks> {
    fn new(
        size: usize,
        class: RankClass,
        prototype: &V,
        rational_terms: bool,
        mut compile: impl FnMut(&ExactKernel) -> Result<K>,
    ) -> Result<Self> {
        let plan = MasterContraction::new_class(size, class)?;
        let parameters: Vec<_> = (0..4 * size + if size == 2 { 4 } else { 0 })
            .map(|i| Atom::var(symbol!(format!("ninjars_integration_momentum::p{i}"))))
            .collect();
        let vectors: Vec<_> = parameters
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| {
                FourMomentum::from_args(p[0].clone(), p[1].clone(), p[2].clone(), p[3].clone())
            })
            .collect();
        let mut expressions = Vec::new();
        for (first, second) in match size {
            2 => vec![(0, 1)],
            3 => vec![(0, 1), (1, 2), (2, 0)],
            4 => vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)],
            _ => Vec::new(),
        } {
            let difference = vectors[second].clone() - vectors[first].clone();
            expressions.push(difference.dot(&difference));
        }
        if size == 2 {
            expressions.push((vectors[1].clone() - vectors[0].clone()).dot(&vectors[size]));
        }
        Ok(Self {
            size,
            momentum: compile(&ExactKernel {
                parameters,
                expressions,
            })?,
            weights: compile(&plan.weights)?,
            contraction: compile(&plan.contraction)?,
            rational_terms: if rational_terms {
                Some((
                    compile(&plan.rational_terms)?,
                    plan.rational_terms.expressions.len(),
                ))
            } else {
                None
            },
            plan,
            zero: prototype.zero(),
            checks: PhantomData,
            active: Vec::new(),
            arguments: Vec::new(),
            kinematics: Vec::new(),
            values: Vec::new(),
            term_values: Vec::new(),
        })
    }

    fn rational_term_count(&self) -> usize {
        self.rational_terms.as_ref().map_or(0, |(_, count)| *count)
    }

    fn prepare(
        &mut self,
        requests: &[ResidueIntegrationRequest<'_, V>],
        invariants: &mut [V],
        weights: &mut [V],
        valid: &mut [bool],
    ) -> Result<()> {
        let rows = requests.len();
        let inv = self.plan.input_layout().invariants.len();
        let base = self.plan.weight_columns();
        let terms = self.rational_term_count();
        let columns = base + terms;
        BatchShape { rows, columns: inv }.validate("integration invariants", invariants.len())?;
        BatchShape { rows, columns }.validate("integration weights", weights.len())?;
        BatchShape { rows, columns: 1 }.validate("integration eligibility", valid.len())?;
        for request in requests {
            request.validate(self.size, self.plan.input_layout().coefficients.len())?;
        }
        self.active.clear();
        self.arguments.clear();
        for (row, request) in requests.iter().enumerate() {
            valid[row] = request
                .shifts
                .iter()
                .flat_map(FourMomentum::components)
                .chain(request.masses_squared)
                .chain(request.coefficients)
                .chain(request.basis.into_iter().flat_map(|basis| {
                    [&basis.e1, &basis.e2, &basis.e3, &basis.e4]
                        .into_iter()
                        .flat_map(FourMomentum::components)
                        .chain(std::iter::once(&basis.e12))
                }))
                .chain([request.scale_squared])
                .chain(request.s_matrix.into_iter().flatten())
                .all(|v| Checks::shape_matches(v, &self.zero) && Checks::finite(v))
                && Checks::valid_scale(request.scale_squared);
            if let Some(matrix) = request.s_matrix {
                let n = request.shifts.len();
                valid[row] &= (0..n).all(|i| {
                    Checks::is_zero(&matrix[i * n + i])
                        && (0..i).all(|j| Checks::equal(&matrix[i * n + j], &matrix[j * n + i]))
                });
            }
            if !valid[row] {
                continue;
            }
            self.active.push(row);
            for &line in request.partition {
                self.arguments
                    .extend(request.shifts[line].components().into_iter().cloned());
            }
            if self.size == 2 {
                self.arguments.extend(
                    request
                        .basis
                        .expect("validated basis")
                        .e2
                        .components()
                        .into_iter()
                        .cloned(),
                );
            }
        }
        let kinematic_columns = inv + usize::from(self.size == 2);
        self.kinematics
            .resize(self.active.len() * kinematic_columns, self.zero.clone());
        self.momentum
            .evaluate(&self.arguments, &mut self.kinematics, self.active.len())?;
        self.arguments.clear();
        for (index, &row) in self.active.iter().enumerate() {
            let request = &requests[row];
            if let Some(matrix) = request.s_matrix {
                let pairs: &[(usize, usize)] = match self.size {
                    2 => &[(0, 1)],
                    3 => &[(0, 1), (1, 2), (2, 0)],
                    4 => &[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)],
                    _ => &[],
                };
                for (column, &(first, second)) in pairs.iter().enumerate() {
                    self.kinematics[index * kinematic_columns + column] =
                        matrix[request.partition[first] * request.shifts.len()
                            + request.partition[second]]
                            .clone();
                }
            }
            self.arguments.extend_from_slice(request.coefficients);
            let kin = &self.kinematics[index * kinematic_columns..(index + 1) * kinematic_columns];
            self.arguments.extend_from_slice(&kin[..inv]);
            self.arguments.extend(
                request
                    .partition
                    .iter()
                    .map(|&line| request.masses_squared[line].clone()),
            );
            match self.size {
                1 => self
                    .arguments
                    .push(request.basis.expect("validated basis").e12.clone()),
                2 => self.arguments.push(kin[inv].clone()),
                _ => (),
            }
        }
        self.values
            .resize(self.active.len() * base, self.zero.clone());
        self.weights
            .evaluate(&self.arguments, &mut self.values, self.active.len())?;
        if let Some((kernel, _)) = &mut self.rational_terms {
            self.term_values
                .resize(self.active.len() * terms, self.zero.clone());
            kernel.evaluate(&self.arguments, &mut self.term_values, self.active.len())?;
        }
        for (index, &row) in self.active.iter().enumerate() {
            let kin = &self.kinematics[index * kinematic_columns..(index + 1) * kinematic_columns];
            let value = &self.values[index * base..(index + 1) * base];
            valid[row] = kin.iter().chain(value).all(Checks::finite);
            if valid[row] {
                invariants[row * inv..(row + 1) * inv].clone_from_slice(&kin[..inv]);
                weights[row * columns..row * columns + base].clone_from_slice(value);
                weights[row * columns + base..(row + 1) * columns]
                    .clone_from_slice(&self.term_values[index * terms..(index + 1) * terms]);
            }
        }
        Ok(())
    }

    fn value_is_zero(&self, value: &V) -> bool {
        Checks::is_zero(value)
    }

    fn values_are_finite(&self, values: &[V]) -> bool {
        values.iter().all(Checks::finite)
    }

    fn contract(
        &mut self,
        weights: &[V],
        masters: &[Laurent<V>],
        output: &mut [IntegratedContribution<V>],
    ) -> Result<()> {
        let rows = output.len();
        let count = self.plan.masters().len();
        BatchShape {
            rows,
            columns: count + 1,
        }
        .validate("integration contraction weights", weights.len())?;
        BatchShape {
            rows,
            columns: count,
        }
        .validate("integration contraction masters", masters.len())?;
        self.arguments.clear();
        for row in 0..rows {
            self.arguments
                .extend_from_slice(&weights[row * (count + 1)..(row + 1) * (count + 1)]);
            for master in &masters[row * count..(row + 1) * count] {
                self.arguments.extend_from_slice(master);
            }
        }
        self.values.resize(rows * 5, self.zero.clone());
        self.contraction
            .evaluate(&self.arguments, &mut self.values, rows)?;
        for (out, value) in output.iter_mut().zip(self.values.as_chunks::<5>().0) {
            out.clone_from(value);
        }
        Ok(())
    }
}

/// The rank class of the `bool` constructors: rank n+1 when `higher_rank`.
fn higher_class(higher_rank: bool) -> RankClass {
    if higher_rank {
        RankClass::NPlus1
    } else {
        RankClass::UpToN
    }
}

/// Batched integration kernels for complex scalar numerical carriers.
pub struct SymbolicaIntegrationKernels<T> {
    engine: Engine<Complex<T>, CompiledKernel<T>, OrdinaryChecks>,
}

impl<T: Real + SingleFloat> SymbolicaIntegrationKernels<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn new(cut_size: usize, higher_rank: bool, prototype: T) -> Result<Self> {
        Self::new_class(cut_size, higher_class(higher_rank), prototype)
    }

    /// Kernels for residues of the given rank class.
    pub fn new_class(cut_size: usize, class: RankClass, prototype: T) -> Result<Self> {
        Self::build(cut_size, class, prototype, false)
    }

    /// The same kernels, which also evaluate the additive terms of each
    /// row's rational part ([`IntegrationKernels::rational_term_count`]) for
    /// the opt-in cancellation test. Contracted values are unchanged.
    pub fn new_class_with_rational_terms(
        cut_size: usize,
        class: RankClass,
        prototype: T,
    ) -> Result<Self> {
        Self::build(cut_size, class, prototype, true)
    }

    fn build(cut_size: usize, class: RankClass, prototype: T, terms: bool) -> Result<Self> {
        let zero = Complex::new(prototype.zero(), prototype.zero());
        Ok(Self {
            engine: Engine::new(cut_size, class, &zero, terms, |k| k.compile(&prototype))?,
        })
    }
}

/// The same exact arithmetic transformed by Symbolica for every Dual component.
pub struct SymbolicaDualIntegrationKernels<T, D> {
    engine: Engine<D, CompiledDualKernel<T, D>, DualChecks<T>>,
}

impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> SymbolicaDualIntegrationKernels<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn new(cut_size: usize, higher_rank: bool, prototype: T, dual: &D) -> Result<Self> {
        Self::new_class(cut_size, higher_class(higher_rank), prototype, dual)
    }

    /// Kernels for residues of the given rank class.
    pub fn new_class(cut_size: usize, class: RankClass, prototype: T, dual: &D) -> Result<Self> {
        Self::build(cut_size, class, prototype, dual, false)
    }

    /// See [`SymbolicaIntegrationKernels::new_class_with_rational_terms`].
    pub fn new_class_with_rational_terms(
        cut_size: usize,
        class: RankClass,
        prototype: T,
        dual: &D,
    ) -> Result<Self> {
        Self::build(cut_size, class, prototype, dual, true)
    }

    fn build(
        cut_size: usize,
        class: RankClass,
        prototype: T,
        dual: &D,
        terms: bool,
    ) -> Result<Self> {
        Ok(Self {
            engine: Engine::new(cut_size, class, dual, terms, |k| {
                k.compile_dual(&prototype, dual)
            })?,
        })
    }
}

macro_rules! implement {
    ($name:ident $(, $dual:ident)?) => {
        impl<T: Real + SingleFloat $(, $dual: Real + DualNumber<Complex<T>>)?>
            IntegrationKernels<implement!(@value T $(,$dual)?)> for $name<T $(,$dual)?>
        where Complex<T>: Real + EvaluationDomain {
            fn cut_size(&self) -> usize { self.engine.size }
            fn coefficient_count(&self) -> usize { self.engine.plan.input_layout().coefficients.len() }
            fn masters(&self) -> &[MasterIntegral] { self.engine.plan.masters() }
            fn invariant_count(&self) -> usize { self.engine.plan.input_layout().invariants.len() }
            fn rational_term_count(&self) -> usize { self.engine.rational_term_count() }
            fn prepare_batch(&mut self, requests: &[ResidueIntegrationRequest<'_, implement!(@value T $(,$dual)?)>],
                invariants: &mut [implement!(@value T $(,$dual)?)], weights: &mut [implement!(@value T $(,$dual)?)], valid: &mut [bool]) -> Result<()> {
                self.engine.prepare(requests, invariants, weights, valid)
            }
            fn contract_batch(&mut self, weights: &[implement!(@value T $(,$dual)?)], masters: &[Laurent<implement!(@value T $(,$dual)?)>],
                output: &mut [IntegratedContribution<implement!(@value T $(,$dual)?)>]) -> Result<()> { self.engine.contract(weights, masters, output) }
            fn value_is_zero(&self, value: &implement!(@value T $(,$dual)?)) -> bool { self.engine.value_is_zero(value) }
            fn values_are_finite(&self, values: &[implement!(@value T $(,$dual)?)]) -> bool { self.engine.values_are_finite(values) }
        }
    };
    (@value $t:ident) => { Complex<$t> };
    (@value $t:ident, $d:ident) => { $d };
}
implement!(SymbolicaIntegrationKernels);
implement!(SymbolicaDualIntegrationKernels, D);
