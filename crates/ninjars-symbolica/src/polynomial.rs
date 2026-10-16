//! Numerator callbacks prepared by Symbolica substitution and Laurent polynomials.

use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
};

use ninjars::{
    BatchShape, Error, ExpansionKind, ExpansionRequest, FourMomentum, Numerator, NumeratorRequest,
    Result,
};
use symbolica::{
    atom::{Atom, AtomCore, AtomView},
    domains::{
        atom::AtomField,
        float::{Complex, Real, SingleFloat},
        rational::Rational,
    },
    evaluate::{BatchExpressionEvaluator, EvaluationDomain, ExpressionEvaluator},
    id::{Pattern, Replacement},
    poly::polynomial::MultivariatePolynomial,
    symbol,
};

static NEXT_NAMESPACE: AtomicU64 = AtomicU64::new(0);

/// Batched Ninja callbacks over a Symbolica polynomial.
///
/// `T` is the real coefficient carrier (`f64`, `DoubleFloat`, or `Float`).
/// Cut momenta and evaluator values use `Complex<T>`. External coefficients are
/// selected by each request's phase-space lane, independent of request ordering.
/// Symbolica performs all substitutions, polynomial arithmetic and coefficient
/// extraction; numerical execution uses its instruction-major batch evaluator.
pub struct PolynomialNumerator<T> {
    polynomial: MultivariatePolynomial<AtomField, i32>,
    expression: Atom,
    variables: [Atom; 5],
    parameter_names: Vec<Atom>,
    parameter_values: Vec<Complex<T>>,
    lanes: usize,
    rank: usize,
    prototype: T,
    evaluator: BatchExpressionEvaluator<Complex<T>>,
    expansions: HashMap<ExpansionKind, BatchExpressionEvaluator<Complex<T>>>,
    inputs: Vec<Complex<T>>,
}

fn symbolic_error(error: impl std::fmt::Display) -> Error {
    Error::Symbolic(error.to_string())
}

fn exact_field() -> AtomField {
    AtomField {
        statistical_zero_test: false,
        ..AtomField::new()
    }
}

impl<T: Real + SingleFloat> PolynomialNumerator<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    /// Prepare a numerator in `(q0,q1,q2,q3,mu_squared)` and external parameters.
    /// `prototype` fixes the precision of every rational coefficient. The input
    /// must be polynomial in the five loop variables; coefficients may contain
    /// Symbolica-supported functions of external parameters.
    pub fn new(
        expression: Atom,
        variables: [Atom; 5],
        parameter_names: Vec<Atom>,
        prototype: T,
    ) -> Result<Self> {
        let all_variables: Vec<_> = variables.iter().chain(&parameter_names).collect();
        for (index, variable) in all_variables.iter().enumerate() {
            if !matches!(variable.as_view(), AtomView::Var(_)) {
                return Err(Error::Symbolic(
                    "numerator variables and parameters must be symbols".into(),
                ));
            }
            if all_variables[..index].contains(variable) {
                return Err(Error::Symbolic(
                    "numerator variables and parameters must be distinct".into(),
                ));
            }
        }
        let expression = expression.expand();
        let polynomial =
            expression.to_polynomial_in_vars_with_field::<i32>(&variables, &exact_field());
        let mut rank = 0;
        for term in &polynomial {
            if term.exponents.iter().any(|power| *power < 0)
                || variables
                    .iter()
                    .any(|variable| term.coefficient.contains(variable))
            {
                return Err(Error::Symbolic(
                    "numerator is not polynomial in the loop variables".into(),
                ));
            }
            let weight = term
                .exponents
                .iter()
                .enumerate()
                .map(|(index, power)| *power as usize * if index == 4 { 2 } else { 1 })
                .sum();
            rank = rank.max(weight);
        }
        let parameters: Vec<_> = variables.iter().chain(&parameter_names).cloned().collect();
        let evaluator = Self::compile(std::slice::from_ref(&expression), &parameters, &prototype)?;
        Ok(Self {
            polynomial,
            expression,
            variables,
            parameter_names,
            parameter_values: Vec::new(),
            lanes: 0,
            rank,
            prototype,
            evaluator,
            expansions: HashMap::new(),
            inputs: Vec::new(),
        })
    }

    /// Access the shared Symbolica polynomial used for rank and coefficient checks.
    pub fn polynomial(&self) -> &MultivariatePolynomial<AtomField, i32> {
        &self.polynomial
    }

    pub fn expression(&self) -> &Atom {
        &self.expression
    }
    pub fn variables(&self) -> &[Atom; 5] {
        &self.variables
    }
    pub fn parameter_names(&self) -> &[Atom] {
        &self.parameter_names
    }
    pub(crate) fn representative(&self) -> &T {
        &self.prototype
    }

    pub(crate) fn exact_evaluator(&self) -> Result<ExpressionEvaluator<Complex<Rational>>> {
        let parameters: Vec<_> = self
            .variables
            .iter()
            .chain(&self.parameter_names)
            .cloned()
            .collect();
        Atom::evaluator_multiple(std::slice::from_ref(&self.expression), &parameters)
            .cores(1)
            .build()
            .map_err(symbolic_error)
    }

    /// Bind row-major external parameters for a phase-space batch. Even when the
    /// numerator has no parameters, `lanes` specifies its valid phase-space IDs.
    pub fn set_parameter_batch(&mut self, values: &[Complex<T>], lanes: usize) -> Result<()> {
        BatchShape {
            rows: lanes,
            columns: self.parameter_names.len(),
        }
        .validate("numerator parameters", values.len())?;
        self.parameter_values.clear();
        self.parameter_values.extend_from_slice(values);
        self.lanes = lanes;
        Ok(())
    }

    fn compile(
        expressions: &[Atom],
        parameters: &[Atom],
        prototype: &T,
    ) -> Result<BatchExpressionEvaluator<Complex<T>>> {
        let evaluator: ExpressionEvaluator<Complex<Rational>> =
            Atom::evaluator_multiple(expressions, parameters)
                .cores(1)
                .build()
                .map_err(symbolic_error)?;
        Ok(Self::map_evaluator(evaluator, prototype))
    }

    pub(crate) fn map_evaluator(
        evaluator: ExpressionEvaluator<Complex<Rational>>,
        prototype: &T,
    ) -> BatchExpressionEvaluator<Complex<T>> {
        evaluator
            .map_coeff_with_prec(
                &|value| {
                    Complex::new(
                        prototype.from_rational(&value.re),
                        prototype.from_rational(&value.im),
                    )
                },
                prototype.get_precision(),
            )
            .into_batch_evaluator()
    }

    fn check_lane(&self, lane: usize) -> Result<()> {
        if lane >= self.lanes {
            Err(Error::Lane {
                lane,
                lanes: self.lanes,
            })
        } else {
            Ok(())
        }
    }

    fn prepare_expansion(
        &self,
        kind: &ExpansionKind,
    ) -> Result<BatchExpressionEvaluator<Complex<T>>> {
        Ok(Self::map_evaluator(
            self.exact_expansion(kind)?,
            &self.prototype,
        ))
    }

    pub(crate) fn exact_expansion(
        &self,
        kind: &ExpansionKind,
    ) -> Result<ExpressionEvaluator<Complex<Rational>>> {
        // Fresh symbols prevent collisions with caller-owned parameter names.
        let symbols = loop {
            let id = NEXT_NAMESPACE.fetch_add(1, Ordering::Relaxed);
            let symbols: Vec<_> = (0..22 + self.polynomial.coefficients.len())
                .map(|i| Atom::var(symbol!(format!("ninjars_cut::c{id}v{i}"))))
                .collect();
            if symbols.iter().all(|value| {
                !self.expression.contains(value)
                    && !self.parameter_names.contains(value)
                    && !self.variables.contains(value)
            }) {
                break symbols;
            }
        };
        let (t, x, mu2) = (&symbols[0], &symbols[1], &symbols[2]);
        let vector = |index: usize| {
            FourMomentum::from_args(
                symbols[3 + 4 * index].clone(),
                symbols[4 + 4 * index].clone(),
                symbols[5 + 4 * index].clone(),
                symbols[6 + 4 * index].clone(),
            )
        };
        let (q, mu, vectors, scalars) = match kind {
            ExpansionKind::Mu { .. } => {
                let a = vector(0);
                let mu = t.pow(2) * a.dot(&a);
                (a * t.clone() + vector(1), mu, 2, 0)
            }
            ExpansionKind::T3 { .. } => (
                vector(0) + vector(1) * t.clone() + vector(2) * ((&symbols[19] + mu2) / t),
                mu2.clone(),
                3,
                1,
            ),
            ExpansionKind::T2 { .. } => (
                vector(0)
                    + vector(1) * x.clone()
                    + vector(2) * t.clone()
                    + vector(3)
                        * ((&symbols[19] + &symbols[20] * x + &symbols[21] * x.pow(2) + mu2) / t),
                mu2.clone(),
                4,
                3,
            ),
        };
        let replacements: Vec<_> = self
            .variables
            .iter()
            .zip(q.components().into_iter().chain([&mu]))
            .map(|(from, to)| {
                Replacement::new(Pattern::Literal(from.clone()), Pattern::Literal(to.clone()))
            })
            .collect();
        // Protect the already collected external coefficients through Symbolica
        // evaluator aliases. Polynomial conversion then handles only the small
        // cut-variable expressions. Alias definitions stay exact and are shared
        // by every phase-space lane, including when the evaluator is Dualized.
        let protected = self.polynomial.map_coeff(
            |coefficient| {
                let index = self
                    .polynomial
                    .coefficients
                    .iter()
                    .position(|candidate| candidate == coefficient)
                    .unwrap();
                symbols[22 + index].clone()
            },
            exact_field(),
        );
        let collected = protected.flatten(false);
        let substituted = collected.replace_multiple(&replacements);
        let substituted = substituted.expand_in(t).expand_in(x).expand_in(mu2);
        // Signed exponents in Symbolica already support finite Laurent polynomials.
        let variables = [t.clone(), x.clone(), mu2.clone()];
        let polynomial =
            substituted.to_polynomial_in_vars_with_field::<i32>(&variables, &exact_field());
        if (&polynomial)
            .into_iter()
            .any(|term| variables.iter().any(|v| term.coefficient.contains(v)))
        {
            return Err(Error::Expansion(
                "Symbolica could not extract a finite Laurent polynomial".into(),
            ));
        }
        let powers: Vec<[i32; 3]> = match kind {
            ExpansionKind::Mu { powers } => powers.iter().map(|&a| [a, 0, 0]).collect(),
            ExpansionKind::T3 { powers } => {
                powers.iter().map(|&(a, c)| [a, 0, i32::from(c)]).collect()
            }
            ExpansionKind::T2 { powers } => powers
                .iter()
                .map(|&(a, b, c)| [a, i32::from(b), i32::from(c)])
                .collect(),
        };
        let expressions: Vec<_> = powers
            .iter()
            .map(|power| polynomial.coefficient(power).unwrap_or(Atom::Zero))
            .collect();
        let parameters: Vec<_> = symbols[3..3 + 4 * vectors]
            .iter()
            .chain(&symbols[19..19 + scalars])
            .chain(&self.parameter_names)
            .cloned()
            .collect();
        Atom::evaluator_multiple(&expressions, &parameters)
            .add_aliases(
                self.polynomial
                    .coefficients
                    .iter()
                    .enumerate()
                    .map(|(index, coefficient)| (symbols[22 + index].clone(), coefficient.clone())),
            )
            .map_err(symbolic_error)?
            .cores(1)
            .build()
            .map_err(symbolic_error)
    }
}

impl<T: Real + SingleFloat> Numerator<Complex<T>> for PolynomialNumerator<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    fn rank(&self) -> usize {
        self.rank
    }

    fn evaluate_batch(
        &mut self,
        requests: &[NumeratorRequest<'_, Complex<T>>],
        output: &mut [Complex<T>],
    ) -> Result<()> {
        BatchShape {
            rows: requests.len(),
            columns: 1,
        }
        .validate("numerator output", output.len())?;
        for request in requests {
            self.check_lane(request.id.lane)?;
        }
        self.inputs.clear();
        let columns = self.parameter_names.len();
        for request in requests {
            self.inputs
                .extend(request.q.components().into_iter().cloned());
            self.inputs.push(request.mu_squared.clone());
            self.inputs.extend_from_slice(
                &self.parameter_values[request.id.lane * columns..(request.id.lane + 1) * columns],
            );
        }
        self.evaluator
            .evaluate_batch(&self.inputs, output, requests.len())
            .map_err(symbolic_error)
    }

    fn expansion_batch(
        &mut self,
        kind: &ExpansionKind,
        requests: &[ExpansionRequest<'_, Complex<T>>],
        output: &mut [Complex<T>],
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
            self.check_lane(request.id.lane)?;
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
        }
        if requests.is_empty() || kind.is_empty() {
            return Ok(());
        }
        if !self.expansions.contains_key(kind) {
            self.expansions
                .insert(kind.clone(), self.prepare_expansion(kind)?);
        }
        self.inputs.clear();
        let columns = self.parameter_names.len();
        for request in requests {
            for vector in request.vectors {
                self.inputs.extend(vector.components().into_iter().cloned());
            }
            if request.vectors.len() < vectors {
                self.inputs.extend(
                    (0..4).map(|_| Complex::new(self.prototype.zero(), self.prototype.zero())),
                );
            }
            self.inputs.extend_from_slice(request.parameters);
            self.inputs.extend_from_slice(
                &self.parameter_values[request.id.lane * columns..(request.id.lane + 1) * columns],
            );
        }
        self.expansions
            .get_mut(kind)
            .unwrap()
            .evaluate_batch(&self.inputs, output, requests.len())
            .map_err(symbolic_error)
    }
}

#[cfg(test)]
#[path = "polynomial_tests.rs"]
mod coefficient_preservation_tests;
