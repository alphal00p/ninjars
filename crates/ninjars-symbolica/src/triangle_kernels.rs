//! Symbolica arithmetic for the core's batched triple-cut coordinator.

use crate::{
    DenominatorExpansion, LaurentDivision, ResidueProjection,
    kernel::{CompiledDualKernel, CompiledKernel, ExactKernel, KernelExecutor},
    value_checks::{DualChecks, OrdinaryChecks, ValueChecks},
};
use ninjars::{
    BatchShape, Complex, Error, ExpansionKind, FloatLike, FourMomentum, Real, Result, SingleFloat,
    TriangleCut, TriangleDivisionRequest, TriangleReductionKernels,
};
use symbolica::{atom::Atom, domains::dual::DualNumber, evaluate::EvaluationDomain, symbol};

struct Engine<V, K, Checks> {
    n: usize,
    rank: usize,
    layout: LaurentDivision,
    coefficient_count: usize,
    denominator: K,
    division: K,
    projection: K,
    swap: K,
    zero: V,
    checks: std::marker::PhantomData<fn() -> Checks>,
    input: Vec<V>,
    denominators: Vec<V>,
    values: Vec<V>,
    active: Vec<usize>,
    eligible: Vec<usize>,
}

impl<V: FloatLike, K: KernelExecutor<V>, Checks: ValueChecks<V>> Engine<V, K, Checks> {
    fn values_are_finite(&self, values: &[V]) -> bool {
        values.iter().all(Checks::finite)
    }

    fn new(
        n: usize,
        rank: usize,
        prototype: &V,
        mut compile: impl FnMut(&ExactKernel) -> Result<K>,
    ) -> Result<Self> {
        let layout = LaurentDivision::new(n, rank, 3)?;
        let denominator = compile(&DenominatorExpansion::new(3)?.exact)?;
        let division = compile(&layout.exact)?;
        let p = ResidueProjection::new(&layout)?;
        let coefficient_count = p.coefficient_count();
        let projection = compile(&p.exact)?;
        let parameters: Vec<_> = (0..5)
            .map(|i| Atom::var(symbol!(format!("ninjars_triangle_swap::p{i}"))))
            .collect();
        let expressions = parameters[..4]
            .iter()
            .map(|v| -v / (&parameters[4] * 2))
            .collect();
        let swap = compile(&ExactKernel {
            expressions,
            parameters,
        })?;
        Ok(Self {
            n,
            rank,
            layout,
            coefficient_count,
            denominator,
            division,
            projection,
            swap,
            zero: prototype.zero(),
            checks: std::marker::PhantomData,
            input: Vec::new(),
            denominators: Vec::new(),
            values: Vec::new(),
            active: Vec::new(),
            eligible: Vec::new(),
        })
    }

    fn swapped(&mut self, cuts: &[&TriangleCut<V>], output: &mut [FourMomentum<V>]) -> Result<()> {
        BatchShape {
            rows: cuts.len(),
            columns: 1,
        }
        .validate("swapped triangle directions", output.len())?;
        self.input.clear();
        for cut in cuts {
            self.input
                .extend(cut.basis.e3.components().into_iter().cloned());
            self.input.push(cut.basis.e12.clone());
        }
        self.values.resize(cuts.len() * 4, self.zero.clone());
        self.swap
            .evaluate(&self.input, &mut self.values, cuts.len())?;
        for (out, v) in output.iter_mut().zip(self.values.as_chunks::<4>().0) {
            *out = FourMomentum::from_args(v[0].clone(), v[1].clone(), v[2].clone(), v[3].clone());
        }
        Ok(())
    }

    fn divide(
        &mut self,
        requests: &[TriangleDivisionRequest<'_, V>],
        numerator: &[V],
        output: &mut [V],
        stable: &mut [bool],
    ) -> Result<()> {
        let rows = requests.len();
        let columns = self.layout.numerator_expansion().len();
        let width = self.layout.quotient_expansion().len();
        let uncut = self.layout.uncut_count();
        BatchShape { rows, columns }
            .validate("triangle numerator coefficients", numerator.len())?;
        BatchShape {
            rows,
            columns: width,
        }
        .validate("triangle quotient coefficients", output.len())?;
        BatchShape { rows, columns: 1 }.validate("triangle division status", stable.len())?;
        for r in requests {
            if r.partition.len() != 3
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
                    "triangle denominator request does not match its topology".into(),
                ));
            }
        }
        self.input.clear();
        self.eligible.clear();
        for (row, r) in requests.iter().enumerate() {
            stable[row] = r
                .shifts
                .iter()
                .chain(r.vectors)
                .flat_map(FourMomentum::components)
                .chain(r.masses_squared)
                .chain(std::iter::once(r.beta))
                .chain(&numerator[row * columns..(row + 1) * columns])
                .all(|value| Checks::shape_matches(value, &self.zero) && Checks::finite(value));
            if !stable[row] {
                continue;
            }
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
        self.denominator.evaluate(
            &self.input,
            &mut self.denominators,
            self.eligible.len() * uncut,
        )?;
        self.active.clear();
        self.input.clear();
        for (index, &row) in self.eligible.iter().enumerate() {
            let status = &mut stable[row];
            let num = &numerator[row * columns..(row + 1) * columns];
            let den = &self.denominators[index * uncut * 4..(index + 1) * uncut * 4];
            *status = num.iter().all(Checks::finite)
                && den.iter().all(Checks::finite)
                && den
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|d| Checks::nonzero(&d[0]));
            if *status {
                self.active.push(row);
                self.input.extend_from_slice(num);
                self.input.extend_from_slice(den);
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

    fn project(&mut self, quotients: &[V], e12: &[V], output: &mut [V]) -> Result<()> {
        let rows = e12.len();
        let columns = 2 * self.layout.quotient_expansion().len();
        BatchShape { rows, columns }.validate("triangle projection quotients", quotients.len())?;
        BatchShape {
            rows,
            columns: self.coefficient_count,
        }
        .validate("triangle residue coefficients", output.len())?;
        self.input.clear();
        for (row, g) in e12.iter().enumerate() {
            self.input
                .extend_from_slice(&quotients[row * columns..(row + 1) * columns]);
            self.input.push(g.clone());
        }
        self.projection.evaluate(&self.input, output, rows)
    }
}

/// Arithmetic kernels for `TriangleReducer<Complex<T>, _>`.
/// Zero leading uncut denominators and nonfinite rows are isolated. Precision
/// rescue and a scaled small-denominator criterion remain separate diagnostics.
pub struct SymbolicaTriangleKernels<T> {
    engine: Engine<Complex<T>, CompiledKernel<T>, OrdinaryChecks>,
}
impl<T: Real + SingleFloat> SymbolicaTriangleKernels<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn new(propagators: usize, rank: usize, prototype: T) -> Result<Self> {
        let zero = Complex::new(prototype.zero(), prototype.zero());
        Ok(Self {
            engine: Engine::new(propagators, rank, &zero, |exact| exact.compile(&prototype))?,
        })
    }
}

/// The same core triangle stage with a Symbolica static or dynamic Dual shape.
pub struct SymbolicaDualTriangleKernels<T, D> {
    engine: Engine<D, CompiledDualKernel<T, D>, DualChecks<T>>,
}
impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> SymbolicaDualTriangleKernels<T, D>
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
}

macro_rules! implementation {
    ($name:ident<$($g:ident),+>, $value:ty, $($bounds:tt)*) => {
        impl<$($g),+> TriangleReductionKernels<$value> for $name<$($g),+> where $($bounds)* {
            fn propagators(&self)->usize {self.engine.n}
            fn rank(&self)->usize {self.engine.rank}
            fn numerator_expansion(&self)->&ExpansionKind {self.engine.layout.numerator_expansion()}
            fn quotient_columns(&self)->usize {self.engine.layout.quotient_expansion().len()}
            fn coefficient_count(&self)->usize {self.engine.coefficient_count}
            fn swapped_effective_batch(&mut self,cuts:&[&TriangleCut<$value>],output:&mut[FourMomentum<$value>])->Result<()> {self.engine.swapped(cuts,output)}
            fn divide_batch(&mut self,requests:&[TriangleDivisionRequest<'_,$value>],numerator:&[$value],output:&mut[$value],stable:&mut[bool])->Result<()> {self.engine.divide(requests,numerator,output,stable)}
            fn project_batch(&mut self,quotients:&[$value],e12:&[$value],output:&mut[$value])->Result<()> {self.engine.project(quotients,e12,output)}
            fn values_are_finite(&self,values:&[$value])->bool {self.engine.values_are_finite(values)}
        }
    };
}
implementation!(SymbolicaTriangleKernels<T>,Complex<T>,T:Real+SingleFloat,Complex<T>:Real+EvaluationDomain);
implementation!(SymbolicaDualTriangleKernels<T,D>,D,T:Real+SingleFloat,D:Real+DualNumber<Complex<T>>,Complex<T>:Real+EvaluationDomain);

#[cfg(feature = "native-kernels")]
impl<T: oneloop::rust_codegen::runtime::NativeFloat> SymbolicaTriangleKernels<T>
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
}

#[cfg(feature = "native-kernels")]
impl<T: oneloop::rust_codegen::runtime::NativeFloat, D: Real + DualNumber<Complex<T>>>
    SymbolicaDualTriangleKernels<T, D>
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
}
