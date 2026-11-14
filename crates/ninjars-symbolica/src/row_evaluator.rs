//! Row-major batches over Symbolica's scalar [`ExpressionEvaluator`].

use symbolica::{
    atom::Atom,
    domains::float::Real,
    evaluate::{EvaluationError, ExpressionEvaluator},
};

/// Evaluates `rows` row-major samples with one scalar evaluation per row.
///
/// Every row runs the interpreter program of a single evaluation, so the
/// results are those of [`ExpressionEvaluator::evaluate`] on each row. Shape
/// errors are reported before any row is evaluated and leave `out` unchanged.
#[derive(Clone)]
pub(crate) struct RowEvaluator<T> {
    evaluator: ExpressionEvaluator<T>,
    inputs: usize,
    outputs: usize,
}

impl<T: Real> From<ExpressionEvaluator<T>> for RowEvaluator<T> {
    fn from(mut evaluator: ExpressionEvaluator<T>) -> Self {
        // `get_input_len`/`get_output_len` require `T: Default`, which
        // `Complex` does not implement: read both counts from the shape errors
        // of `try_evaluate`, which are reported before anything is evaluated.
        let shape_error = |error| unreachable!("unexpected evaluator probe error: {error}");
        let (inputs, outputs) = match evaluator.try_evaluate(&[], &mut []) {
            Ok(()) => (0, 0),
            Err(EvaluationError::InvalidOutputCount { expected, .. }) => (0, expected),
            Err(EvaluationError::InvalidParameterCount { expected, .. }) => {
                let zeros = vec![T::new_zero(); expected];
                match evaluator.try_evaluate(&zeros, &mut []) {
                    Ok(()) => (expected, 0),
                    Err(EvaluationError::InvalidOutputCount {
                        expected: outputs, ..
                    }) => (expected, outputs),
                    Err(error) => shape_error(error),
                }
            }
            Err(error) => shape_error(error),
        };
        Self {
            evaluator,
            inputs,
            outputs,
        }
    }
}

impl<T> RowEvaluator<T> {
    #[cfg(test)]
    pub(crate) fn into_expression_evaluator(self) -> ExpressionEvaluator<T> {
        self.evaluator
    }
}

impl<T: Real> RowEvaluator<T> {
    pub(crate) fn evaluate_batch(
        &mut self,
        parameters: &[T],
        out: &mut [T],
        rows: usize,
    ) -> Result<(), EvaluationError> {
        let (inputs, outputs) = (self.inputs, self.outputs);
        let overflow = || EvaluationError::EvaluationFailed {
            expression: Atom::num(0),
            reason: "Batch dimensions exceed the addressable buffer size".into(),
        };
        let expected = rows.checked_mul(inputs).ok_or_else(overflow)?;
        if parameters.len() != expected {
            return Err(EvaluationError::InvalidParameterCount {
                expected,
                actual: parameters.len(),
            });
        }
        let expected = rows.checked_mul(outputs).ok_or_else(overflow)?;
        if out.len() != expected {
            return Err(EvaluationError::InvalidOutputCount {
                expected,
                actual: out.len(),
            });
        }
        for row in 0..rows {
            self.evaluator.try_evaluate(
                &parameters[row * inputs..(row + 1) * inputs],
                &mut out[row * outputs..(row + 1) * outputs],
            )?;
        }
        Ok(())
    }
}
