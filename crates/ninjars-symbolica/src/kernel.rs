//! Shared execution of exact Symbolica instruction graphs.

use ninjars::{BatchShape, Complex, Error, Real, Result, SingleFloat};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        dual::{DualNumber, HyperDual},
        rational::Rational,
    },
    evaluate::{BatchExpressionEvaluator, Dualizer, EvaluationDomain, ExpressionEvaluator},
};

use crate::PolynomialNumerator;

pub(crate) trait KernelExecutor<V> {
    fn evaluate(&mut self, input: &[V], output: &mut [V], rows: usize) -> Result<()>;
}

impl<T: Real + SingleFloat> KernelExecutor<Complex<T>> for CompiledKernel<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    fn evaluate(
        &mut self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        rows: usize,
    ) -> Result<()> {
        self.evaluate_batch(input, output, rows)
    }
}
impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> KernelExecutor<D>
    for CompiledDualKernel<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    fn evaluate(&mut self, input: &[D], output: &mut [D], rows: usize) -> Result<()> {
        self.evaluate_batch(input, output, rows)
    }
}

pub(crate) struct ExactKernel {
    pub expressions: Vec<Atom>,
    pub parameters: Vec<Atom>,
}

impl ExactKernel {
    fn evaluator(&self) -> Result<ExpressionEvaluator<Complex<Rational>>> {
        Atom::evaluator_multiple(&self.expressions, &self.parameters)
            .cores(1)
            .build()
            .map_err(|e| Error::Symbolic(e.to_string()))
    }

    pub fn compile<T: Real + SingleFloat>(&self, prototype: &T) -> Result<CompiledKernel<T>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        Ok(CompiledKernel {
            inputs: self.parameters.len(),
            outputs: self.expressions.len(),
            evaluator: if self.expressions.is_empty() {
                None
            } else {
                Some(PolynomialNumerator::map_evaluator(
                    self.evaluator()?,
                    prototype,
                ))
            },
        })
    }

    pub fn compile_dual<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>>(
        &self,
        prototype: &T,
        dual: &D,
    ) -> Result<CompiledDualKernel<T, D>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        let shape: Vec<_> = dual
            .get_shape()
            .into_iter()
            .map(<[usize]>::to_vec)
            .collect();
        let evaluator = if self.expressions.is_empty() {
            None
        } else {
            let structure = HyperDual::<Complex<Rational>>::new(shape.clone());
            let exact = self
                .evaluator()?
                .vectorize(&Dualizer::new(structure, Vec::new()))
                .map_err(|e| Error::Symbolic(e.to_string()))?;
            Some(PolynomialNumerator::map_evaluator(exact, prototype))
        };
        Ok(CompiledDualKernel {
            kernel: CompiledKernel {
                inputs: self.parameters.len() * shape.len(),
                outputs: self.expressions.len() * shape.len(),
                evaluator,
            },
            inputs: self.parameters.len(),
            outputs: self.expressions.len(),
            shape,
            prototype: dual.zero(),
            base_zero: Complex::new(prototype.zero(), prototype.zero()),
            arguments: Vec::new(),
            results: Vec::new(),
        })
    }
}

pub(crate) struct CompiledKernel<T> {
    inputs: usize,
    outputs: usize,
    evaluator: Option<BatchExpressionEvaluator<Complex<T>>>,
}

impl<T: Real + SingleFloat> CompiledKernel<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn evaluate_batch(
        &mut self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        rows: usize,
    ) -> Result<()> {
        BatchShape {
            rows,
            columns: self.inputs,
        }
        .validate("kernel inputs", input.len())?;
        BatchShape {
            rows,
            columns: self.outputs,
        }
        .validate("kernel outputs", output.len())?;
        if let Some(evaluator) = &mut self.evaluator {
            evaluator
                .evaluate_batch(input, output, rows)
                .map_err(|e| Error::Symbolic(e.to_string()))?;
        }
        Ok(())
    }
}

pub(crate) struct CompiledDualKernel<T, D> {
    kernel: CompiledKernel<T>,
    inputs: usize,
    outputs: usize,
    shape: Vec<Vec<usize>>,
    prototype: D,
    base_zero: Complex<T>,
    arguments: Vec<Complex<T>>,
    results: Vec<Complex<T>>,
}

impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> CompiledDualKernel<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    fn valid_shape(&self, value: &D) -> bool {
        value.values().len() == self.shape.len()
            && value
                .get_shape()
                .into_iter()
                .eq(self.shape.iter().map(Vec::as_slice))
    }

    pub fn evaluate_batch(&mut self, input: &[D], output: &mut [D], rows: usize) -> Result<()> {
        BatchShape {
            rows,
            columns: self.inputs,
        }
        .validate("Dual kernel inputs", input.len())?;
        BatchShape {
            rows,
            columns: self.outputs,
        }
        .validate("Dual kernel outputs", output.len())?;
        if input.iter().any(|v| !self.valid_shape(v)) {
            return Err(Error::Unsupported("kernel Dual shapes differ".into()));
        }
        let result_len = output
            .len()
            .checked_mul(self.shape.len())
            .ok_or_else(|| Error::Unsupported("Dual kernel dimensions overflow usize".into()))?;
        self.arguments.clear();
        for value in input {
            self.arguments.extend_from_slice(value.values());
        }
        self.results.resize(result_len, self.base_zero.clone());
        self.kernel
            .evaluate_batch(&self.arguments, &mut self.results, rows)?;
        for (index, out) in output.iter_mut().enumerate() {
            if !self.valid_shape(out) {
                *out = self.prototype.zero();
            }
            let start = index * self.shape.len();
            out.values_mut()
                .clone_from_slice(&self.results[start..start + self.shape.len()]);
        }
        Ok(())
    }
}
