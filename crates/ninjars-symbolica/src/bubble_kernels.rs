//! Generated numerical kernels for the core double-cut coordinator.

use crate::{
    BubbleSourceProjection, DenominatorExpansion, LaurentDivision, ResidueProjection,
    TriangleBubbleSubtraction,
    kernel::{CompiledDualKernel, CompiledKernel, ExactKernel, KernelExecutor},
};
use ninjars::{
    BatchShape, BubbleCut, BubbleDivisionRequest, BubbleReductionKernels,
    BubbleTriangleCorrectionRequest, Complex, Error, ExpansionKind, FloatLike, FourMomentum, Real,
    Result, SingleFloat,
};
use symbolica::{atom::Atom, domains::dual::DualNumber, evaluate::EvaluationDomain, symbol};

struct Engine<V, K> {
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
    subtraction: K,
    zero: V,
    finite: fn(&V) -> bool,
    nonzero: fn(&V) -> bool,
    shape_matches: fn(&V, &V) -> bool,
    input: Vec<V>,
    denominators: Vec<V>,
    projections: Vec<V>,
    values: Vec<V>,
    eligible: Vec<usize>,
    active: Vec<usize>,
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

impl<V: FloatLike, K: KernelExecutor<V>> Engine<V, K> {
    fn new(
        n: usize,
        rank: usize,
        prototype: &V,
        mut compile: impl FnMut(&ExactKernel) -> Result<K>,
        finite: fn(&V) -> bool,
        nonzero: fn(&V) -> bool,
        shape_matches: fn(&V, &V) -> bool,
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
        let source_count = if rank > n { 15 } else { 10 };
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
            subtraction,
            zero: prototype.zero(),
            finite,
            nonzero,
            shape_matches,
            input: Vec::new(),
            denominators: Vec::new(),
            projections: Vec::new(),
            values: Vec::new(),
            eligible: Vec::new(),
            active: Vec::new(),
        })
    }

    fn valid(&self, value: &V) -> bool {
        (self.shape_matches)(value, &self.zero) && (self.finite)(value)
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
        self.denominator.evaluate(
            &self.input,
            &mut self.denominators,
            self.eligible.len() * uncut,
        )?;
        self.input.clear();
        self.active.clear();
        for (index, &row) in self.eligible.iter().enumerate() {
            let den = &self.denominators[index * uncut * 7..(index + 1) * uncut * 7];
            stable[row] = den.iter().all(self.finite)
                && den.as_chunks::<7>().0.iter().all(|d| (self.nonzero)(&d[0]));
            if stable[row] {
                self.active.push(row);
                self.input
                    .extend_from_slice(&numerator[row * columns..(row + 1) * columns]);
                self.input.extend_from_slice(den);
            }
        }
        self.values
            .resize(self.active.len() * width, self.zero.clone());
        self.division
            .evaluate(&self.input, &mut self.values, self.active.len())?;
        for (index, &row) in self.active.iter().enumerate() {
            let values = &self.values[index * width..(index + 1) * width];
            stable[row] = values.iter().all(self.finite);
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
        for r in requests {
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
        self.input.clear();
        self.eligible.clear();
        for (row, r) in requests.iter().enumerate() {
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
        self.denominators
            .resize(self.eligible.len() * 7, self.zero.clone());
        self.denominator
            .evaluate(&self.input, &mut self.denominators, self.eligible.len())?;
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
            let den = &self.denominators[index * 7..(index + 1) * 7];
            let proj = &self.projections[index * 8..(index + 1) * 8];
            stable[row] = den.iter().chain(proj).all(self.finite) && (self.nonzero)(&den[0]);
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
        self.correction
            .evaluate(&self.input, &mut self.values, self.active.len())?;
        for (index, &row) in self.active.iter().enumerate() {
            let values = &self.values[index * width..(index + 1) * width];
            stable[row] = values.iter().all(self.finite);
            if stable[row] {
                output[row * width..(row + 1) * width].clone_from_slice(values);
            }
        }
        Ok(())
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

/// Bubble reduction arithmetic, with exact-zero/nonfinite isolation. Scaled
/// conditioning checks and precision rescue remain separate acceptance work.
pub struct SymbolicaBubbleKernels<T> {
    engine: Engine<Complex<T>, CompiledKernel<T>>,
}
impl<T: Real + SingleFloat> SymbolicaBubbleKernels<T>
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
                |exact| exact.compile(&prototype),
                |v| v.is_finite(),
                |v| !v.is_fully_zero(),
                |_, _| true,
            )?,
        })
    }
}

/// Static/dynamic Dual execution of the same exact bubble graphs.
pub struct SymbolicaDualBubbleKernels<T, D> {
    engine: Engine<D, CompiledDualKernel<T, D>>,
}
impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> SymbolicaDualBubbleKernels<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn new(propagators: usize, rank: usize, prototype: T, dual: &D) -> Result<Self> {
        Ok(Self {
            engine: Engine::new(
                propagators,
                rank,
                dual,
                |exact| exact.compile_dual(&prototype, dual),
                |v| v.values().iter().all(|x| x.is_finite()),
                |v| !v.values()[0].is_fully_zero(),
                |v, p| v.values().len() == p.values().len() && v.get_shape() == p.get_shape(),
            )?,
        })
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
            fn divide_batch(&mut self,requests:&[BubbleDivisionRequest<'_,$value>],numerator:&[$value],output:&mut[$value],stable:&mut[bool])->Result<()> {self.engine.divide(requests,numerator,output,stable)}
            fn triangle_corrections_batch(&mut self,requests:&[BubbleTriangleCorrectionRequest<'_,$value>],output:&mut[$value],stable:&mut[bool])->Result<()> {self.engine.corrections(requests,output,stable)}
            fn subtract_batch(&mut self,quotients:&[$value],corrections:&[$value],output:&mut[$value],rows:usize)->Result<()> {self.engine.subtract(quotients,corrections,output,rows)}
            fn project_batch(&mut self,quotients:&[$value],e12:&[$value],output:&mut[$value])->Result<()> {self.engine.project(quotients,e12,output)}
            fn values_are_finite(&self,values:&[$value])->bool {values.iter().all(self.engine.finite)}
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
}

#[cfg(feature = "native-kernels")]
impl<T: oneloop::rust_codegen::runtime::NativeFloat, D: Real + DualNumber<Complex<T>>>
    SymbolicaDualBubbleKernels<T, D>
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
            crate::native_kernels::DenominatorPlan::T2,
            &prototype,
        )?;
        Ok(kernels)
    }
}
