//! Symbolica arithmetic for the minimal integrated tadpole stage.

use crate::{
    DenominatorExpansion, LaurentDivision, TadpoleSubtraction,
    kernel::{CompiledDualKernel, CompiledKernel, ExactKernel, KernelExecutor},
};
use ninjars::{
    BatchShape, Complex, CutFailure, Error, ExpansionKind, FloatLike, FourMomentum, Real, Result,
    SingleFloat, TadpoleCorrectionRequest, TadpoleDivisionRequest, TadpoleReductionKernels,
    TadpoleSource, Topology,
};
use symbolica::{atom::Atom, domains::dual::DualNumber, evaluate::EvaluationDomain, symbol};

struct Engine<V, K> {
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
    correction_columns: usize,
    correction_map: Vec<usize>,
    extras: Vec<Vec<usize>>,
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

fn vars(n: usize, scope: &str) -> Vec<Atom> {
    (0..n)
        .map(|i| Atom::var(symbol!(format!("ninjars_tadpole_{scope}::p{i}"))))
        .collect()
}
fn momentum(p: &[Atom], start: usize) -> FourMomentum<Atom> {
    FourMomentum::from_args(
        p[start].clone(),
        p[start + 1].clone(),
        p[start + 2].clone(),
        p[start + 3].clone(),
    )
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
        shape_matches: fn(&V, &V) -> bool,
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
            let axes = if source_size == 3 { 2 } else { 3 };
            let p = vars(
                16 + 4 * axes,
                if kind == 0 {
                    "triangle_axes"
                } else {
                    "bubble_axes"
                },
            );
            let offset = momentum(&p, 0) + momentum(&p, 12);
            let mut expressions = Vec::new();
            for axis in 0..axes {
                let e = momentum(&p, 16 + axis * 4);
                expressions.extend([
                    offset.dot(&e),
                    momentum(&p, 4).dot(&e),
                    momentum(&p, 8).dot(&e),
                ]);
            }
            source_projection[kind] = Some(compile(&ExactKernel {
                expressions,
                parameters: p,
            })?);
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
            correction_columns,
            correction_map,
            extras,
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

    fn status(&self, values: &[V]) -> Option<CutFailure> {
        if values.iter().any(|v| !(self.shape_matches)(v, &self.zero)) {
            Some(CutFailure::DerivativeShape)
        } else if values.iter().any(|v| !(self.finite)(v)) {
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
        for (row, r) in requests.iter().enumerate() {
            stable[row] = r
                .shifts
                .iter()
                .chain(r.vectors)
                .flat_map(FourMomentum::components)
                .chain(r.masses_squared)
                .chain([r.beta])
                .chain(&numerator[row * columns..(row + 1) * columns])
                .all(|v| (self.shape_matches)(v, &self.zero) && (self.finite)(v));
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
            let denominator = &self.denominators[index * uncut * 4..(index + 1) * uncut * 4];
            stable[row] = denominator.iter().all(self.finite)
                && denominator
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|d| (self.nonzero)(&d[0]));
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
            stable[row] = values.iter().all(self.finite);
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
        for request in requests {
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
                    .all(|v| (self.shape_matches)(v, &self.zero) && (self.finite)(v));
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
            self.denominator.evaluate(
                &self.input,
                &mut self.denominators,
                self.eligible.len() * uncut,
            )?;
            self.input.clear();
            self.active.clear();
            for (index, &row) in self.eligible.iter().enumerate() {
                let den = &self.denominators[index * uncut * 4..(index + 1) * uncut * 4];
                let projections = &self.projections[index * axes * 3..(index + 1) * axes * 3];
                if !den.iter().chain(projections).all(self.finite)
                    || !den.as_chunks::<4>().0.iter().all(|d| (self.nonzero)(&d[0]))
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
                self.input.extend_from_slice(den);
            }
            self.values.resize(
                self.active.len() * self.correction_columns,
                self.zero.clone(),
            );
            self.subtraction[kind].as_mut().unwrap().evaluate(
                &self.input,
                &mut self.values,
                self.active.len(),
            )?;
            for (index, &row) in self.active.iter().enumerate() {
                let values = &self.values
                    [index * self.correction_columns..(index + 1) * self.correction_columns];
                stable[row] = values.iter().all(self.finite);
                if stable[row] {
                    for (column, &source) in self.correction_map.iter().enumerate() {
                        output[row * width + column] = values[source].clone();
                    }
                }
            }
        }
        Ok(())
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
    engine: Engine<Complex<T>, CompiledKernel<T>>,
}
impl<T: Real + SingleFloat> SymbolicaTadpoleKernels<T>
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

/// The same tadpole coefficients through shared static or runtime Dual bodies.
pub struct SymbolicaDualTadpoleKernels<T, D> {
    engine: Engine<D, CompiledDualKernel<T, D>>,
}
impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> SymbolicaDualTadpoleKernels<T, D>
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
                |v| v.values().iter().all(|c| c.is_finite()),
                |v| !v.values()[0].is_fully_zero(),
                |v, p| v.values().len() == p.values().len() && v.get_shape() == p.get_shape(),
            )?,
        })
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
            fn correction_batch(&mut self,requests:&[TadpoleCorrectionRequest<'_,$value>],output:&mut[$value],stable:&mut[bool])->Result<()> {self.engine.correction(requests,output,stable)}
            fn combine_batch(&mut self,raw:&[$value],corrections:&[$value],rows:usize,output:&mut[$value])->Result<()> {self.engine.combine(raw,corrections,rows,output)}
            fn project_batch(&mut self,corrected:&[$value],e12:&[$value],output:&mut[$value])->Result<()> {self.engine.project(corrected,e12,output)}
        }
    };
}
implement!(SymbolicaTadpoleKernels<T>,Complex<T>,T:Real+SingleFloat,Complex<T>:Real+EvaluationDomain);
implement!(SymbolicaDualTadpoleKernels<T,D>,D,T:Real+SingleFloat,D:Real+DualNumber<Complex<T>>,Complex<T>:Real+EvaluationDomain);
