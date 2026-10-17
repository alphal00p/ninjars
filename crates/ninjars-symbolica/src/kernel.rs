//! Shared execution of exact Symbolica instruction graphs.

use ninjars::{BatchShape, Complex, Error, Real, Result, SingleFloat};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        dual::{DualNumber, DualNumberStructure, HyperDual},
        rational::Rational,
    },
    evaluate::{BatchExpressionEvaluator, Dualizer, EvaluationDomain, ExpressionEvaluator},
};

use crate::PolynomialNumerator;

#[cfg(test)]
#[path = "kernel_jit_probe.rs"]
mod jit_probe;

#[cfg(all(test, feature = "native-kernels"))]
#[path = "native_blocking_probe.rs"]
mod blocking_probe;

#[cfg(test)]
thread_local! {
    /// Requested scalar precision and actual mapped evaluator constants.
    pub(crate) static DUAL_COMPILATION_PRECISIONS: std::cell::RefCell<Vec<(u32, Vec<u32>)>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

pub(crate) trait KernelExecutor<V> {
    fn evaluate(&mut self, input: &[V], output: &mut [V], rows: usize) -> Result<()>;

    #[cfg(feature = "native-kernels")]
    fn set_native_batch_execution(&mut self, _policy: crate::BatchExecutionPolicy) {}

    #[cfg(all(test, feature = "native-kernels"))]
    fn probe_workspace(&self) -> Option<(usize, usize)> {
        None
    }
}

impl<T: Real + SingleFloat> KernelExecutor<Complex<T>> for CompiledKernel<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    #[cfg(all(test, feature = "native-kernels"))]
    fn probe_workspace(&self) -> Option<(usize, usize)> {
        self.native
            .as_ref()
            .and_then(|native| native.probe_workspace())
    }

    #[cfg(feature = "native-kernels")]
    fn set_native_batch_execution(&mut self, policy: crate::BatchExecutionPolicy) {
        if let Some(native) = &mut self.native {
            native.set_native_batch_execution(policy);
        }
    }

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
    #[cfg(all(test, feature = "native-kernels"))]
    fn probe_workspace(&self) -> Option<(usize, usize)> {
        self.kernel.probe_workspace()
    }

    #[cfg(feature = "native-kernels")]
    fn set_native_batch_execution(&mut self, policy: crate::BatchExecutionPolicy) {
        self.kernel.set_native_batch_execution(policy);
    }

    fn evaluate(&mut self, input: &[D], output: &mut [D], rows: usize) -> Result<()> {
        self.evaluate_batch(input, output, rows)
    }
}

pub(crate) struct ExactKernel {
    pub expressions: Vec<Atom>,
    pub parameters: Vec<Atom>,
}

impl ExactKernel {
    pub(crate) fn evaluator(&self) -> Result<ExpressionEvaluator<Complex<Rational>>> {
        Atom::evaluator_multiple(&self.expressions, &self.parameters)
            .cores(1)
            .build()
            .map_err(|e| Error::Symbolic(e.to_string()))
    }

    pub(crate) fn dual_evaluator<D: DualNumberStructure + Clone>(
        &self,
        structure: &D,
    ) -> Result<ExpressionEvaluator<Complex<Rational>>> {
        self.evaluator()?
            .vectorize(&Dualizer::new(structure.clone(), Vec::new()))
            .map_err(|e| Error::Symbolic(e.to_string()))
    }

    pub fn compile<T: Real + SingleFloat>(&self, prototype: &T) -> Result<CompiledKernel<T>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        Ok(CompiledKernel {
            #[cfg(feature = "native-kernels")]
            native: None,
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
        if shape.is_empty() || dual.values().len() != shape.len() {
            return Err(Error::Unsupported(
                "kernel Dual prototype shape and values differ".into(),
            ));
        }
        let evaluator = if self.expressions.is_empty() {
            None
        } else {
            let structure = HyperDual::<Complex<Rational>>::new(shape.clone());
            let exact = self.dual_evaluator(&structure)?;
            Some(PolynomialNumerator::map_evaluator(exact, prototype))
        };
        #[cfg(test)]
        DUAL_COMPILATION_PRECISIONS.with(|records| {
            let constants = evaluator.as_ref().map_or_else(Vec::new, |evaluator| {
                evaluator
                    .clone()
                    .into_expression_evaluator()
                    .export_instructions()
                    .constants
                    .iter()
                    .flat_map(|value| [value.re.get_precision(), value.im.get_precision()])
                    .collect()
            });
            records
                .borrow_mut()
                .push((prototype.get_precision(), constants));
        });
        Ok(CompiledDualKernel {
            kernel: CompiledKernel {
                #[cfg(feature = "native-kernels")]
                native: None,
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
    #[cfg(feature = "native-kernels")]
    native: Option<Box<dyn KernelExecutor<Complex<T>> + Send + Sync>>,
    inputs: usize,
    outputs: usize,
    evaluator: Option<BatchExpressionEvaluator<Complex<T>>>,
}

impl<T: Real + SingleFloat> CompiledKernel<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    #[cfg(feature = "native-kernels")]
    pub(crate) fn install_native(
        &mut self,
        executor: Box<dyn KernelExecutor<Complex<T>> + Send + Sync>,
        inputs: usize,
        outputs: usize,
    ) -> Result<()> {
        if (self.inputs, self.outputs) != (inputs, outputs) {
            return Err(Error::Unsupported(
                "native kernel plan dimensions differ".into(),
            ));
        }
        #[cfg(test)]
        let executor = blocking_probe::wrap(executor, inputs, outputs);
        self.native = Some(executor);
        self.evaluator = None;
        Ok(())
    }

    #[cfg(feature = "native-kernels")]
    pub(crate) fn backend(&self) -> crate::KernelBackend {
        if self.native.is_some() {
            crate::KernelBackend::NativeRust
        } else {
            crate::KernelBackend::SymbolicaInterpreter
        }
    }

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
        #[cfg(feature = "native-kernels")]
        if let Some(native) = &mut self.native {
            return native.evaluate(input, output, rows);
        }
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
    /// Scalar zero at the explicit constructor precision. The Dual carrier's
    /// own prototype may use a different precision and cannot substitute for it.
    pub(crate) fn scalar_prototype(&self) -> &T {
        &self.base_zero.re
    }

    #[cfg(feature = "native-kernels")]
    pub(crate) fn native_parts(&mut self) -> (&[Vec<usize>], &mut CompiledKernel<T>) {
        (&self.shape, &mut self.kernel)
    }

    #[cfg(feature = "native-kernels")]
    pub(crate) fn backend(&self) -> crate::KernelBackend {
        self.kernel.backend()
    }

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
