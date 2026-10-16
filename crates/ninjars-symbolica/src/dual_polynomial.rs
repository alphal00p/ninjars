use std::collections::HashMap;

use ninjars::{
    BatchShape, Error, ExpansionKind, ExpansionRequest, Numerator, NumeratorRequest, Result,
};
use symbolica::{
    domains::{
        dual::{DualNumber, HyperDual},
        float::{Complex, Real, SingleFloat},
        rational::Rational,
    },
    evaluate::{BatchExpressionEvaluator, Dualizer, EvaluationDomain, ExpressionEvaluator},
};

use crate::PolynomialNumerator;

/// Numerator callbacks using any existing Symbolica static or dynamic Dual.
///
/// All components of cut vectors, cut parameters, mu² and external coefficients
/// are propagated through Symbolica's Dualizer and batch interpreter. The
/// supplied Dual prototype determines derivative shape; the polynomial adapter
/// determines working precision. No finite differences are used.
pub struct DualPolynomialNumerator<T, D> {
    source: PolynomialNumerator<T>,
    prototype: D,
    shape: Vec<Vec<usize>>,
    lanes: usize,
    parameters: Vec<Complex<T>>,
    evaluator: BatchExpressionEvaluator<Complex<T>>,
    expansions: HashMap<ExpansionKind, BatchExpressionEvaluator<Complex<T>>>,
    inputs: Vec<Complex<T>>,
    results: Vec<Complex<T>>,
}

impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> DualPolynomialNumerator<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    /// Construct callbacks with a fixed derivative shape. Bind phase-space
    /// parameters separately with `set_parameter_batch`, including for no parameters.
    pub fn new(source: PolynomialNumerator<T>, prototype: D) -> Result<Self> {
        let shape = prototype
            .get_shape()
            .into_iter()
            .map(<[usize]>::to_vec)
            .collect::<Vec<_>>();
        let evaluator = Self::convert(source.exact_evaluator()?, &shape, source.representative())?;
        Ok(Self {
            source,
            prototype: prototype.zero(),
            shape,
            lanes: 0,
            parameters: Vec::new(),
            evaluator,
            expansions: HashMap::new(),
            inputs: Vec::new(),
            results: Vec::new(),
        })
    }

    fn convert(
        exact: ExpressionEvaluator<Complex<Rational>>,
        shape: &[Vec<usize>],
        representative: &T,
    ) -> Result<BatchExpressionEvaluator<Complex<T>>> {
        let structure = HyperDual::<Complex<Rational>>::new(shape.to_vec());
        let exact = exact
            .vectorize(&Dualizer::new(structure, Vec::new()))
            .map_err(|error| Error::Symbolic(error.to_string()))?;
        Ok(PolynomialNumerator::map_evaluator(exact, representative))
    }

    fn validate_dual(&self, value: &D) -> Result<()> {
        if value
            .get_shape()
            .iter()
            .copied()
            .ne(self.shape.iter().map(Vec::as_slice))
            || value.values().len() != self.shape.len()
        {
            return Err(Error::Unsupported("numerator Dual shapes differ".into()));
        }
        Ok(())
    }

    fn validate_lane(&self, lane: usize) -> Result<()> {
        if lane < self.lanes {
            Ok(())
        } else {
            Err(Error::Lane {
                lane,
                lanes: self.lanes,
            })
        }
    }

    pub fn set_parameter_batch(&mut self, values: &[D], lanes: usize) -> Result<()> {
        BatchShape {
            rows: lanes,
            columns: self.source.parameter_names().len(),
        }
        .validate("numerator parameters", values.len())?;
        for value in values {
            self.validate_dual(value)?;
        }
        self.parameters.clear();
        for value in values {
            self.parameters.extend_from_slice(value.values());
        }
        self.lanes = lanes;
        Ok(())
    }

    fn resize_results(&mut self, outputs: usize) -> Result<()> {
        let length = outputs
            .checked_mul(self.shape.len())
            .ok_or_else(|| Error::Unsupported("Dual output dimensions overflow usize".into()))?;
        let representative = self.source.representative();
        self.results.resize_with(length, || {
            Complex::new(representative.zero(), representative.zero())
        });
        Ok(())
    }

    fn restore(&self, outputs: &mut [D]) {
        for (output, values) in outputs
            .iter_mut()
            .zip(self.results.chunks_exact(self.shape.len()))
        {
            if self.validate_dual(output).is_err() {
                *output = self.prototype.zero();
            }
            output.values_mut().clone_from_slice(values);
        }
    }
}

impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> Numerator<D>
    for DualPolynomialNumerator<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    fn rank(&self) -> usize {
        self.source.rank()
    }

    fn evaluate_batch(
        &mut self,
        requests: &[NumeratorRequest<'_, D>],
        output: &mut [D],
    ) -> Result<()> {
        BatchShape {
            rows: requests.len(),
            columns: 1,
        }
        .validate("numerator output", output.len())?;
        for request in requests {
            self.validate_lane(request.id.lane)?;
            for value in request
                .q
                .components()
                .into_iter()
                .chain([request.mu_squared])
            {
                self.validate_dual(value)?;
            }
        }
        self.inputs.clear();
        let columns = self.source.parameter_names().len() * self.shape.len();
        for request in requests {
            for value in request
                .q
                .components()
                .into_iter()
                .chain([request.mu_squared])
            {
                self.inputs.extend_from_slice(value.values());
            }
            self.inputs.extend_from_slice(
                &self.parameters[request.id.lane * columns..(request.id.lane + 1) * columns],
            );
        }
        self.resize_results(output.len())?;
        self.evaluator
            .evaluate_batch(&self.inputs, &mut self.results, requests.len())
            .map_err(|error| Error::Symbolic(error.to_string()))?;
        self.restore(output);
        Ok(())
    }

    fn expansion_batch(
        &mut self,
        kind: &ExpansionKind,
        requests: &[ExpansionRequest<'_, D>],
        output: &mut [D],
    ) -> Result<()> {
        BatchShape {
            rows: requests.len(),
            columns: kind.len(),
        }
        .validate("expansion output", output.len())?;
        let (vectors, scalars) = match kind {
            ExpansionKind::Mu { .. } => (2, 0),
            ExpansionKind::T3 { .. } => (3, 1),
            ExpansionKind::T2 { .. } => (4, 3),
        };
        for request in requests {
            self.validate_lane(request.id.lane)?;
            if request.vectors.len() != vectors
                && !(matches!(kind, ExpansionKind::Mu { .. }) && request.vectors.len() == 1)
            {
                return Err(Error::Shape {
                    name: "cut vectors",
                    expected: vectors,
                    actual: request.vectors.len(),
                });
            }
            BatchShape {
                rows: 1,
                columns: scalars,
            }
            .validate("cut parameters", request.parameters.len())?;
            for value in request
                .vectors
                .iter()
                .flat_map(|v| v.components())
                .chain(request.parameters)
            {
                self.validate_dual(value)?;
            }
        }
        if requests.is_empty() || kind.is_empty() {
            return Ok(());
        }
        if !self.expansions.contains_key(kind) {
            let evaluator = Self::convert(
                self.source.exact_expansion(kind)?,
                &self.shape,
                self.source.representative(),
            )?;
            self.expansions.insert(kind.clone(), evaluator);
        }
        self.inputs.clear();
        let columns = self.source.parameter_names().len() * self.shape.len();
        for request in requests {
            for value in request.vectors.iter().flat_map(|v| v.components()) {
                self.inputs.extend_from_slice(value.values());
            }
            if request.vectors.len() < vectors {
                for _ in 0..4 {
                    self.inputs.extend_from_slice(self.prototype.values());
                }
            }
            for value in request.parameters {
                self.inputs.extend_from_slice(value.values());
            }
            self.inputs.extend_from_slice(
                &self.parameters[request.id.lane * columns..(request.id.lane + 1) * columns],
            );
        }
        self.resize_results(output.len())?;
        self.expansions
            .get_mut(kind)
            .unwrap()
            .evaluate_batch(&self.inputs, &mut self.results, requests.len())
            .map_err(|error| Error::Symbolic(error.to_string()))?;
        self.restore(output);
        Ok(())
    }
}
