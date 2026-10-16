//! Generated five-cut division using shared Symbolica/FeynKit algebra.

use crate::kernel::{CompiledDualKernel, CompiledKernel, ExactKernel, KernelExecutor};
use ninjars::{
    BatchShape, Complex, CutFailure, Error, FloatLike, FourMomentum, PentagonDivisionRequest,
    PentagonReductionKernels, Real, Result, SingleFloat, Topology,
};
use symbolica::{atom::Atom, domains::dual::DualNumber, evaluate::EvaluationDomain, symbol};

struct Engine<V, K> {
    n: usize,
    rank: usize,
    denominator: K,
    division: K,
    zero: V,
    finite: fn(&V) -> bool,
    nonzero: fn(&V) -> bool,
    compatible: fn(&V, &V) -> bool,
    eligible: Vec<usize>,
    active: Vec<usize>,
    input: Vec<V>,
    denominators: Vec<V>,
    values: Vec<V>,
}

impl<V: FloatLike, K: KernelExecutor<V>> Engine<V, K> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        n: usize,
        rank: usize,
        prototype: &V,
        mut compile: impl FnMut(&ExactKernel) -> Result<K>,
        finite: fn(&V) -> bool,
        nonzero: fn(&V) -> bool,
        compatible: fn(&V, &V) -> bool,
    ) -> Result<Self> {
        Topology::new(n, rank)?;
        if n < 5 {
            return Err(Error::Unsupported(
                "pentagon division needs at least five propagators".into(),
            ));
        }
        let parameters: Vec<_> = (0..5 + 5 * (n - 5))
            .map(|i| Atom::var(symbol!(format!("ninjars_pentagon_division::p{i}"))))
            .collect();
        let q = FourMomentum::from_args(
            parameters[0].clone(),
            parameters[1].clone(),
            parameters[2].clone(),
            parameters[3].clone(),
        );
        let mu = &parameters[4];
        let mut product = mu.clone();
        for uncut in 0..n - 5 {
            let p = &parameters[5 + 5 * uncut..10 + 5 * uncut];
            let shifted = q.clone()
                + FourMomentum::from_args(p[0].clone(), p[1].clone(), p[2].clone(), p[3].clone());
            product *= shifted.dot(&shifted) - &p[4] - mu;
        }
        let denominator = compile(&ExactKernel {
            parameters,
            expressions: vec![product],
        })?;
        let parameters = vec![
            Atom::var(symbol!("ninjars_pentagon_division::numerator")),
            Atom::var(symbol!("ninjars_pentagon_division::denominator")),
        ];
        let division = compile(&ExactKernel {
            expressions: vec![&parameters[0] / &parameters[1]],
            parameters,
        })?;
        Ok(Self {
            n,
            rank,
            denominator,
            division,
            zero: prototype.zero(),
            finite,
            nonzero,
            compatible,
            eligible: Vec::new(),
            active: Vec::new(),
            input: Vec::new(),
            denominators: Vec::new(),
            values: Vec::new(),
        })
    }

    fn divide(
        &mut self,
        requests: &[PentagonDivisionRequest<'_, V>],
        numerator: &[V],
        output: &mut [V],
        failures: &mut [Option<CutFailure>],
    ) -> Result<()> {
        let rows = requests.len();
        for (name, len) in [
            ("pentagon numerator values", numerator.len()),
            ("pentagon coefficients", output.len()),
            ("pentagon division failures", failures.len()),
        ] {
            BatchShape { rows, columns: 1 }.validate(name, len)?;
        }
        for r in requests {
            if r.partition.len() != 5
                || r.complement.len() != self.n - 5
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
                    "pentagon division request does not match its topology".into(),
                ));
            }
        }
        self.eligible.clear();
        self.input.clear();
        for (row, r) in requests.iter().enumerate() {
            failures[row] = None;
            for value in r
                .shifts
                .iter()
                .flat_map(FourMomentum::components)
                .chain(r.masses_squared)
                .chain(r.geometry.q.components())
                .chain([&r.geometry.mu_squared, &numerator[row]])
            {
                if !(self.compatible)(value, &self.zero) {
                    failures[row] = Some(CutFailure::DerivativeShape);
                    break;
                }
                if !(self.finite)(value) {
                    failures[row] = Some(CutFailure::NonFiniteInput);
                    break;
                }
            }
            if failures[row].is_some() {
                continue;
            }
            if !(self.nonzero)(&r.geometry.mu_squared) {
                failures[row] = Some(CutFailure::SingularParameter);
                continue;
            }
            self.eligible.push(row);
            self.input
                .extend(r.geometry.q.components().into_iter().cloned());
            self.input.push(r.geometry.mu_squared.clone());
            for &line in r.complement {
                self.input
                    .extend(r.shifts[line].components().into_iter().cloned());
                self.input.push(r.masses_squared[line].clone());
            }
        }
        self.denominators
            .resize(self.eligible.len(), self.zero.clone());
        self.denominator
            .evaluate(&self.input, &mut self.denominators, self.eligible.len())?;
        self.active.clear();
        self.input.clear();
        for (index, &row) in self.eligible.iter().enumerate() {
            let den = &self.denominators[index];
            if !(self.finite)(den) {
                failures[row] = Some(CutFailure::NonFiniteResult);
            } else if !(self.nonzero)(den) {
                failures[row] = Some(CutFailure::SingularParameter);
            } else {
                self.active.push(row);
                self.input.extend([numerator[row].clone(), den.clone()]);
            }
        }
        self.values.resize(self.active.len(), self.zero.clone());
        self.division
            .evaluate(&self.input, &mut self.values, self.active.len())?;
        for (index, &row) in self.active.iter().enumerate() {
            if (self.finite)(&self.values[index]) {
                output[row] = self.values[index].clone();
            } else {
                failures[row] = Some(CutFailure::NonFiniteResult);
            }
        }
        Ok(())
    }
}

pub struct SymbolicaPentagonKernels<T> {
    engine: Engine<Complex<T>, CompiledKernel<T>>,
}
impl<T: Real + SingleFloat> SymbolicaPentagonKernels<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn new(propagators: usize, rank: usize, prototype: T) -> Result<Self> {
        let zero = Complex::new(prototype.zero(), prototype.zero());
        Ok(Self {
            engine: Engine::new(
                propagators,
                rank,
                &zero,
                |k| k.compile(&prototype),
                |v| v.is_finite(),
                |v| !v.is_fully_zero(),
                |_, _| true,
            )?,
        })
    }
}
pub struct SymbolicaDualPentagonKernels<T, D> {
    engine: Engine<D, CompiledDualKernel<T, D>>,
}
impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> SymbolicaDualPentagonKernels<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn new(propagators: usize, rank: usize, prototype: T, dual: &D) -> Result<Self> {
        Ok(Self {
            engine: Engine::new(
                propagators,
                rank,
                dual,
                |k| k.compile_dual(&prototype, dual),
                |v| v.values().iter().all(|x| x.is_finite()),
                |v| !v.values()[0].is_fully_zero(),
                |v, p| v.values().len() == p.values().len() && v.get_shape() == p.get_shape(),
            )?,
        })
    }
}
macro_rules! implementation {
    ($name:ident<$($g:ident),+>,$value:ty,$($bounds:tt)*)=>{
        impl<$($g),+> PentagonReductionKernels<$value> for $name<$($g),+> where $($bounds)* {
            fn propagators(&self)->usize {self.engine.n}
            fn rank(&self)->usize {self.engine.rank}
            fn divide_batch(&mut self,requests:&[PentagonDivisionRequest<'_,$value>],numerator:&[$value],output:&mut[$value],failures:&mut[Option<CutFailure>])->Result<()> {self.engine.divide(requests,numerator,output,failures)}
        }
    }
}
implementation!(SymbolicaPentagonKernels<T>,Complex<T>,T:Real+SingleFloat,Complex<T>:Real+EvaluationDomain);
implementation!(SymbolicaDualPentagonKernels<T,D>,D,T:Real+SingleFloat,D:Real+DualNumber<Complex<T>>,Complex<T>:Real+EvaluationDomain);
