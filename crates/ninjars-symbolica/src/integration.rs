//! Symbolica-generated invariant, master-weight and contraction arithmetic.

use crate::{
    MasterContraction,
    kernel::{CompiledDualKernel, CompiledKernel, ExactKernel, KernelExecutor},
};
use ninjars::{
    BatchShape, Complex, FloatLike, FourMomentum, IntegratedContribution, IntegrationKernels,
    Laurent, MasterIntegral, Real, ResidueIntegrationRequest, Result, SingleFloat,
};
use symbolica::{atom::Atom, domains::dual::DualNumber, evaluate::EvaluationDomain, symbol};

struct Engine<V, K> {
    size: usize,
    plan: MasterContraction,
    momentum: K,
    weights: K,
    contraction: K,
    zero: V,
    finite: fn(&V) -> bool,
    compatible: fn(&V, &V) -> bool,
    is_zero: fn(&V) -> bool,
    valid_scale: fn(&V) -> bool,
    active: Vec<usize>,
    arguments: Vec<V>,
    kinematics: Vec<V>,
    values: Vec<V>,
}

impl<V: FloatLike, K: KernelExecutor<V>> Engine<V, K> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        size: usize,
        higher: bool,
        prototype: &V,
        mut compile: impl FnMut(&ExactKernel) -> Result<K>,
        finite: fn(&V) -> bool,
        compatible: fn(&V, &V) -> bool,
        is_zero: fn(&V) -> bool,
        valid_scale: fn(&V) -> bool,
    ) -> Result<Self> {
        let plan = MasterContraction::new(size, higher)?;
        let parameters: Vec<_> = (0..4 * size + 4)
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
            plan,
            zero: prototype.zero(),
            finite,
            compatible,
            is_zero,
            valid_scale,
            active: Vec::new(),
            arguments: Vec::new(),
            kinematics: Vec::new(),
            values: Vec::new(),
        })
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
        let columns = self.plan.weight_columns();
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
                .chain(
                    [
                        &request.basis.e1,
                        &request.basis.e2,
                        &request.basis.e3,
                        &request.basis.e4,
                    ]
                    .into_iter()
                    .flat_map(FourMomentum::components),
                )
                .chain([&request.basis.e12, request.scale_squared])
                .all(|v| (self.compatible)(v, &self.zero) && (self.finite)(v))
                && (self.valid_scale)(request.scale_squared);
            if !valid[row] {
                continue;
            }
            self.active.push(row);
            for &line in request.partition {
                self.arguments
                    .extend(request.shifts[line].components().into_iter().cloned());
            }
            self.arguments
                .extend(request.basis.e2.components().into_iter().cloned());
        }
        let kinematic_columns = inv + usize::from(self.size == 2);
        self.kinematics
            .resize(self.active.len() * kinematic_columns, self.zero.clone());
        self.momentum
            .evaluate(&self.arguments, &mut self.kinematics, self.active.len())?;
        self.arguments.clear();
        for (index, &row) in self.active.iter().enumerate() {
            let request = &requests[row];
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
                1 => self.arguments.push(request.basis.e12.clone()),
                2 => self.arguments.push(kin[inv].clone()),
                _ => (),
            }
        }
        self.values
            .resize(self.active.len() * columns, self.zero.clone());
        self.weights
            .evaluate(&self.arguments, &mut self.values, self.active.len())?;
        for (index, &row) in self.active.iter().enumerate() {
            let kin = &self.kinematics[index * kinematic_columns..(index + 1) * kinematic_columns];
            let value = &self.values[index * columns..(index + 1) * columns];
            valid[row] = kin.iter().chain(value).all(self.finite);
            if valid[row] {
                invariants[row * inv..(row + 1) * inv].clone_from_slice(&kin[..inv]);
                weights[row * columns..(row + 1) * columns].clone_from_slice(value);
            }
        }
        Ok(())
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

/// Batched integration kernels for complex scalar numerical carriers.
pub struct SymbolicaIntegrationKernels<T> {
    engine: Engine<Complex<T>, CompiledKernel<T>>,
}

impl<T: Real + SingleFloat> SymbolicaIntegrationKernels<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn new(cut_size: usize, higher_rank: bool, prototype: T) -> Result<Self> {
        let zero = Complex::new(prototype.zero(), prototype.zero());
        Ok(Self {
            engine: Engine::new(
                cut_size,
                higher_rank,
                &zero,
                |k| k.compile(&prototype),
                |v| v.re.is_finite() && v.im.is_finite(),
                // Float tracks significant bits per component. Cancellation
                // changes that count; it does not change the scalar carrier.
                |_, _| true,
                |v| v.re.is_zero() && v.im.is_zero(),
                |v| {
                    v.im.is_zero()
                        && v.re.real_cmp(&v.re.zero()) == Some(std::cmp::Ordering::Greater)
                },
            )?,
        })
    }
}

/// The same exact arithmetic transformed by Symbolica for every Dual component.
pub struct SymbolicaDualIntegrationKernels<T, D> {
    engine: Engine<D, CompiledDualKernel<T, D>>,
}

impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> SymbolicaDualIntegrationKernels<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn new(cut_size: usize, higher_rank: bool, prototype: T, dual: &D) -> Result<Self> {
        Ok(Self {
            engine: Engine::new(
                cut_size,
                higher_rank,
                dual,
                |k| k.compile_dual(&prototype, dual),
                |v| {
                    v.values()
                        .iter()
                        .all(|v| v.re.is_finite() && v.im.is_finite())
                },
                |v, p| v.values().len() == p.values().len() && v.get_shape() == p.get_shape(),
                |v| v.values().iter().all(|v| v.re.is_zero() && v.im.is_zero()),
                |v| {
                    v.values()[0].im.is_zero()
                        && v.values()[0].re.real_cmp(&v.values()[0].re.zero())
                            == Some(std::cmp::Ordering::Greater)
                },
            )?,
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
            fn prepare_batch(&mut self, requests: &[ResidueIntegrationRequest<'_, implement!(@value T $(,$dual)?)>],
                invariants: &mut [implement!(@value T $(,$dual)?)], weights: &mut [implement!(@value T $(,$dual)?)], valid: &mut [bool]) -> Result<()> {
                self.engine.prepare(requests, invariants, weights, valid)
            }
            fn contract_batch(&mut self, weights: &[implement!(@value T $(,$dual)?)], masters: &[Laurent<implement!(@value T $(,$dual)?)>],
                output: &mut [IntegratedContribution<implement!(@value T $(,$dual)?)>]) -> Result<()> { self.engine.contract(weights, masters, output) }
            fn value_is_zero(&self, value: &implement!(@value T $(,$dual)?)) -> bool { (self.engine.is_zero)(value) }
            fn values_are_finite(&self, values: &[implement!(@value T $(,$dual)?)]) -> bool { values.iter().all(self.engine.finite) }
        }
    };
    (@value $t:ident) => { Complex<$t> };
    (@value $t:ident, $d:ident) => { $d };
}
implement!(SymbolicaIntegrationKernels);
implement!(SymbolicaDualIntegrationKernels, D);
