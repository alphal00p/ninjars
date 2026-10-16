//! Symbolica-generated physical magnitudes for automatic mu sampling scales.

use crate::kernel::{CompiledKernel, ExactKernel};
use ninjars::{
    BatchShape, Complex, CutFailure, Error, FloatLike, FourMomentum, Real, Result,
    SamplingScaleProvider, SamplingScaleSource, SamplingScaleStatus, SingleFloat,
};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::dual::DualNumber,
    evaluate::EvaluationDomain,
    symbol,
};

struct Engine<T, V> {
    propagators: usize,
    kernel: CompiledKernel<T>,
    sources: Vec<SamplingScaleSource>,
    prototype: V,
    one: Complex<T>,
    components: fn(&V) -> &[Complex<T>],
    compatible: fn(&V, &V) -> bool,
    constant: fn(&V, &Complex<T>) -> V,
    active: Vec<usize>,
    failures: Vec<Option<CutFailure>>,
    arguments: Vec<Complex<T>>,
    values: Vec<Complex<T>>,
}

impl<T: Real + SingleFloat, V: FloatLike> Engine<T, V>
where
    Complex<T>: Real + EvaluationDomain,
{
    fn new(
        propagators: usize,
        prototype: T,
        carrier: V,
        components: fn(&V) -> &[Complex<T>],
        compatible: fn(&V, &V) -> bool,
        constant: fn(&V, &Complex<T>) -> V,
    ) -> Result<Self> {
        if propagators == 0 {
            return Err(Error::NoPropagators);
        }
        let count = propagators
            .checked_mul(5)
            .ok_or_else(|| Error::Unsupported("sampling scale dimensions overflow usize".into()))?;
        let parameters: Vec<_> = (0..count)
            .map(|i| Atom::var(symbol!(format!("ninjars_sampling_scale::p{i}"))))
            .collect();
        let momenta: Vec<_> = parameters[..4 * propagators]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| {
                FourMomentum::from_args(p[0].clone(), p[1].clone(), p[2].clone(), p[3].clone())
            })
            .collect();
        let mut expressions = Vec::new();
        let mut sources = Vec::new();
        for index in 0..propagators {
            expressions.push(parameters[4 * propagators + index].abs());
            sources.push(SamplingScaleSource::Mass { propagator: index });
        }
        for first in 0..propagators {
            for second in first + 1..propagators {
                let difference = momenta[second].clone() - momenta[first].clone();
                expressions.push(difference.dot(&difference).abs());
                sources.push(SamplingScaleSource::Invariant { first, second });
            }
        }
        Ok(Self {
            propagators,
            kernel: ExactKernel {
                parameters,
                expressions,
            }
            .compile(&prototype)?,
            sources,
            prototype: carrier.zero(),
            one: Complex::new(prototype.one(), prototype.zero()),
            components,
            compatible,
            constant,
            active: Vec::new(),
            failures: Vec::new(),
            arguments: Vec::new(),
            values: Vec::new(),
        })
    }

    fn evaluate(
        &mut self,
        shifts: &[FourMomentum<V>],
        masses: &[V],
        lanes: usize,
        scales: &mut [V],
        statuses: &mut [SamplingScaleStatus],
    ) -> Result<()> {
        BatchShape {
            rows: lanes,
            columns: self.propagators,
        }
        .validate("sampling scale shifts", shifts.len())?;
        BatchShape {
            rows: lanes,
            columns: self.propagators,
        }
        .validate("sampling scale masses", masses.len())?;
        BatchShape {
            rows: lanes,
            columns: 1,
        }
        .validate("sampling scales", scales.len())?;
        BatchShape {
            rows: lanes,
            columns: 1,
        }
        .validate("sampling scale statuses", statuses.len())?;
        self.active.clear();
        self.arguments.clear();
        self.failures.resize(lanes, None);
        self.failures.fill(None);
        for (lane, failure) in self.failures.iter_mut().enumerate() {
            let start = lane * self.propagators;
            let p = &shifts[start..start + self.propagators];
            let m = &masses[start..start + self.propagators];
            for value in p.iter().flat_map(FourMomentum::components).chain(m) {
                if !(self.compatible)(value, &self.prototype) {
                    *failure = Some(CutFailure::DerivativeShape);
                    break;
                }
                if (self.components)(value)
                    .iter()
                    .any(|c| !c.re.is_finite() || !c.im.is_finite())
                {
                    *failure = Some(CutFailure::NonFiniteInput);
                    break;
                }
            }
            if failure.is_none() {
                self.active.push(lane);
                self.arguments.extend(
                    p.iter()
                        .flat_map(FourMomentum::components)
                        .chain(m)
                        .map(|v| (self.components)(v)[0].clone()),
                );
            }
        }
        let columns = self.sources.len();
        self.values
            .resize(self.active.len() * columns, self.one.zero());
        self.kernel
            .evaluate_batch(&self.arguments, &mut self.values, self.active.len())?;
        for (&lane, row) in self.active.iter().zip(self.values.chunks_exact(columns)) {
            if row.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
                self.failures[lane] = Some(CutFailure::NonFiniteResult);
                continue;
            }
            let mut selected = 0;
            for index in 1..columns {
                if row[index].re.real_cmp(&row[selected].re) == Some(std::cmp::Ordering::Greater) {
                    selected = index;
                }
            }
            let (value, source) = if row[selected].re.is_fully_zero() {
                (&self.one, SamplingScaleSource::UnitFallback)
            } else {
                (&row[selected], self.sources[selected])
            };
            scales[lane] = (self.constant)(&self.prototype, value);
            statuses[lane] = SamplingScaleStatus::Selected(source);
        }
        for (status, &failure) in statuses.iter_mut().zip(&self.failures) {
            if let Some(failure) = failure {
                *status = SamplingScaleStatus::InvalidInput(failure);
            }
        }
        Ok(())
    }
}

pub struct SymbolicaSamplingScales<T> {
    engine: Engine<T, Complex<T>>,
}
impl<T: Real + SingleFloat> SymbolicaSamplingScales<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn new(propagators: usize, prototype: T) -> Result<Self> {
        let carrier = Complex::new(prototype.zero(), prototype.zero());
        Ok(Self {
            engine: Engine::new(
                propagators,
                prototype,
                carrier,
                std::slice::from_ref,
                |_, _| true,
                |_, v| v.clone(),
            )?,
        })
    }
}
pub struct SymbolicaDualSamplingScales<T, D> {
    engine: Engine<T, D>,
}
impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> SymbolicaDualSamplingScales<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn new(propagators: usize, prototype: T, dual: D) -> Result<Self> {
        if dual.values().is_empty() {
            return Err(Error::Unsupported(
                "sampling scales require a Dual primal component".into(),
            ));
        }
        Ok(Self {
            engine: Engine::new(
                propagators,
                prototype,
                dual,
                DualNumber::values,
                |a, b| a.values().len() == b.values().len() && a.get_shape() == b.get_shape(),
                |p, v| {
                    let mut out = p.zero();
                    out.values_mut()[0] = v.clone();
                    out
                },
            )?,
        })
    }
}
macro_rules! implement {
    ($name:ident<$($g:ident),+>,$value:ty,$($bounds:tt)*)=>{
        impl<$($g),+> SamplingScaleProvider<$value> for $name<$($g),+> where $($bounds)* {
            fn propagators(&self)->usize {self.engine.propagators}
            fn evaluate_batch(&mut self,shifts:&[FourMomentum<$value>],masses_squared:&[$value],lanes:usize,scales:&mut[$value],statuses:&mut[SamplingScaleStatus])->Result<()> {self.engine.evaluate(shifts,masses_squared,lanes,scales,statuses)}
        }
    };
}
implement!(SymbolicaSamplingScales<T>,Complex<T>,T:Real+SingleFloat,Complex<T>:Real+EvaluationDomain);
implement!(SymbolicaDualSamplingScales<T,D>,D,T:Real+SingleFloat,D:Real+DualNumber<Complex<T>>,Complex<T>:Real+EvaluationDomain);
