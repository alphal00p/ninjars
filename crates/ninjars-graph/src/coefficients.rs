//! Polynomial coefficient preparation, independent of reduction callbacks.
//!
//! Symbolica owns the polynomial, coefficient mapping, selective expansion and
//! evaluator graph. Its public map_coeff/flatten APIs, implementations in
//! poly.rs/poly/polynomial.rs, and FeynKit/Spenso coefficient-preservation usages
//! were checked before composing this plan. There is no second polynomial IR.

use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
};

use ninjars_positional::positional_evaluator;
use symbolica::{
    atom::{Atom, AtomCore, AtomView},
    domains::{
        atom::AtomField,
        float::{Complex, Real, SingleFloat},
        rational::Rational,
    },
    evaluate::{BatchExpressionEvaluator, EvaluationDomain, ExpressionEvaluator},
    poly::polynomial::MultivariatePolynomial,
    symbol,
};
use thiserror::Error;

static NEXT_COEFFICIENT_NAMESPACE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Error)]
pub enum CoefficientError {
    #[error("loop variables and external parameters must be distinct symbols")]
    Variables,
    #[error("numerator is not polynomial in the five loop variables")]
    NotPolynomial,
    #[error("coefficient dimensions or weighted numerator rank overflow usize")]
    Overflow,
    #[error("{buffer} has {actual} entries; expected {expected}")]
    Shape {
        buffer: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("nonfinite external parameter at row {row}, column {column}")]
    NonFiniteInput { row: usize, column: usize },
    #[error("nonfinite polynomial coefficient at row {row}, column {column}")]
    NonFiniteCoefficient { row: usize, column: usize },
    #[error("Symbolica coefficient preparation: {0}")]
    Symbolic(String),
    #[error("coefficient JIT compilation: {0}")]
    Jit(String),
    #[error(
        "numerator has an explicit mu^2 term (power {power}); Ninja's dense tensor numerator has no mu^2 slots"
    )]
    ExplicitMuSquared { power: i32 },
    #[error("tensor rank {requested} is below the numerator degree {required}")]
    TensorRank { requested: usize, required: usize },
}

type Result<T> = std::result::Result<T, CoefficientError>;

fn field() -> AtomField {
    AtomField {
        statistical_zero_test: false,
        ..AtomField::new()
    }
}

/// A shared Symbolica polynomial with one evaluator output per distinct exact
/// coefficient. The template replaces those coefficients by fresh parameters.
///
/// Pass an already normalized physical numerator, including its overall factor
/// exactly once. This plan adds no graph factors or normalization conventions.
/// Loop variables are `(q0,q1,q2,q3,mu_squared)`; mu_squared has rank weight two.
pub struct PolynomialCoefficientPlan {
    polynomial: MultivariatePolynomial<AtomField, i32>,
    variables: [Atom; 5],
    parameters: Vec<Atom>,
    heads: Vec<Atom>,
    symbols: Vec<Atom>,
    template: Atom,
    rank: usize,
}

impl PolynomialCoefficientPlan {
    pub fn new(expression: Atom, variables: [Atom; 5], parameters: Vec<Atom>) -> Result<Self> {
        let all: Vec<_> = variables.iter().chain(&parameters).collect();
        if all.iter().enumerate().any(|(i, value)| {
            !matches!(value.as_view(), AtomView::Var(_)) || all[..i].contains(value)
        }) {
            return Err(CoefficientError::Variables);
        }
        let expanded = variables
            .iter()
            .fold(expression.clone(), |value, var| value.expand_in(var));
        let polynomial = expanded.to_polynomial_in_vars_with_field::<i32>(&variables, &field());
        let mut rank = 0;
        for term in &polynomial {
            if term.exponents.iter().any(|power| *power < 0)
                || variables.iter().any(|var| term.coefficient.contains(var))
            {
                return Err(CoefficientError::NotPolynomial);
            }
            let weight =
                term.exponents
                    .iter()
                    .enumerate()
                    .try_fold(0_usize, |sum, (i, power)| {
                        (*power as usize)
                            .checked_mul(if i == 4 { 2 } else { 1 })
                            .and_then(|v| sum.checked_add(v))
                            .ok_or(CoefficientError::Overflow)
                    })?;
            rank = rank.max(weight);
        }
        let mut indices = HashMap::new();
        let mut heads = Vec::new();
        for coefficient in &polynomial.coefficients {
            if !indices.contains_key(coefficient) {
                indices.insert(coefficient.clone(), heads.len());
                heads.push(coefficient.clone());
            }
        }
        let symbols = loop {
            let namespace = NEXT_COEFFICIENT_NAMESPACE.fetch_add(1, Ordering::Relaxed);
            let candidates: Vec<_> = (0..heads.len())
                .map(|i| {
                    Atom::var(symbol!(format!(
                        "ninjars_prepared_coefficients::c{namespace}_{i}"
                    )))
                })
                .collect();
            if candidates
                .iter()
                .all(|symbol| !expression.contains(symbol) && !all.contains(&symbol))
            {
                break candidates;
            }
        };
        let template = polynomial
            .map_coeff(|coefficient| symbols[indices[coefficient]].clone(), field())
            .flatten(false);
        Ok(Self {
            polynomial,
            variables,
            parameters,
            heads,
            symbols,
            template,
            rank,
        })
    }

    pub fn polynomial(&self) -> &MultivariatePolynomial<AtomField, i32> {
        &self.polynomial
    }

    /// Factor external coefficients exactly with Symbolica before compiling.
    /// This is an optional setup operation: it can change floating-point
    /// evaluation order and does not guarantee improved conditioning. Rebuild
    /// the template because previously distinct heads may become identical.
    pub fn factor_coefficients(self) -> Result<Self> {
        // Algebra audit: AtomCore::factor and its documented exact identity;
        // poly/factor.rs and examples/factorization.rs; FeynKit's installed
        // FeynCalc WW/Higgs/Z controls. Mapping/flattening also uses Symbolica.
        let expression = self
            .polynomial
            .map_coeff(|coefficient| coefficient.factor(), field())
            .flatten(false);
        Self::new(expression, self.variables, self.parameters)
    }
    pub fn variables(&self) -> &[Atom; 5] {
        &self.variables
    }
    pub fn parameters(&self) -> &[Atom] {
        &self.parameters
    }
    pub fn coefficient_expressions(&self) -> &[Atom] {
        &self.heads
    }
    pub fn coefficient_symbols(&self) -> &[Atom] {
        &self.symbols
    }
    pub fn template(&self) -> &Atom {
        &self.template
    }
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Build every coefficient head in one Symbolica graph, sharing common
    /// arithmetic between outputs. Parameters become positional slots, so
    /// the evaluator does not depend on what the process registered before.
    /// Zero polynomials need no evaluator.
    pub fn exact_evaluator(&self) -> Result<Option<ExpressionEvaluator<Complex<Rational>>>> {
        if self.heads.is_empty() {
            return Ok(None);
        }
        positional_evaluator(&self.heads, &self.parameters, &[], true)
            .map(Some)
            .map_err(|e| CoefficientError::Symbolic(e.to_string()))
    }

    /// Generic interpreter control at the requested working precision. No input
    /// value is promoted, rounded through binary64, or silently specialized.
    pub fn compile<T: Real + SingleFloat>(&self, prototype: T) -> Result<CoefficientEvaluator<T>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        let evaluator = self.exact_evaluator()?.map(|e| {
            e.map_coeff_with_prec(
                &|v| {
                    Complex::new(
                        prototype.from_rational(&v.re),
                        prototype.from_rational(&v.im),
                    )
                },
                prototype.get_precision(),
            )
            .into_batch_evaluator()
        });
        Ok(CoefficientEvaluator {
            inputs: self.parameters.len(),
            outputs: self.heads.len(),
            evaluator,
            scratch: Vec::new(),
            zero: Complex::new(prototype.zero(), prototype.zero()),
        })
    }
}

fn validate_dimensions(
    inputs: usize,
    outputs: usize,
    input_len: usize,
    output_len: usize,
    rows: usize,
) -> Result<()> {
    for (buffer, columns, actual) in [
        ("coefficient inputs", inputs, input_len),
        ("coefficient outputs", outputs, output_len),
    ] {
        let expected = rows
            .checked_mul(columns)
            .ok_or(CoefficientError::Overflow)?;
        if expected != actual {
            return Err(CoefficientError::Shape {
                buffer,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

fn finite<T: Real + SingleFloat>(values: &[Complex<T>], columns: usize, input: bool) -> Result<()> {
    if let Some(index) = values
        .iter()
        .position(|v| !v.re.is_finite() || !v.im.is_finite())
    {
        let (row, column) = (index / columns, index % columns);
        return Err(if input {
            CoefficientError::NonFiniteInput { row, column }
        } else {
            CoefficientError::NonFiniteCoefficient { row, column }
        });
    }
    Ok(())
}

/// Retained workspaces for evaluating a complete coefficient batch before it is
/// bound to reduction callbacks. Errors leave the caller's output unchanged.
pub struct CoefficientEvaluator<T> {
    inputs: usize,
    outputs: usize,
    evaluator: Option<BatchExpressionEvaluator<Complex<T>>>,
    scratch: Vec<Complex<T>>,
    zero: Complex<T>,
}

impl<T: Real + SingleFloat> CoefficientEvaluator<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn evaluate_batch(
        &mut self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        rows: usize,
    ) -> Result<()> {
        validate_dimensions(self.inputs, self.outputs, input.len(), output.len(), rows)?;
        finite(input, self.inputs, true)?;
        if rows == 0 || self.outputs == 0 {
            return Ok(());
        }
        self.scratch.resize(output.len(), self.zero.clone());
        self.evaluator
            .as_mut()
            .expect("nonempty heads have an evaluator")
            .evaluate_batch(input, &mut self.scratch, rows)
            .map_err(|e| CoefficientError::Symbolic(e.to_string()))?;
        finite(&self.scratch, self.outputs, false)?;
        output.clone_from_slice(&self.scratch);
        Ok(())
    }
}

#[cfg(feature = "numerator-jit")]
mod jit {
    use super::*;
    use symbolica::evaluate::{Instruction, JITCompilationSettings, JITCompiledEvaluator};

    /// Binary64 coefficient execution only. The reduction callbacks and their
    /// generic carriers are separate from this preparation backend.
    pub struct SymJitCoefficientEvaluator {
        inputs: usize,
        outputs: usize,
        evaluator: Option<JITCompiledEvaluator<Complex<f64>>>,
        scratch: Vec<Complex<f64>>,
        timings: SymJitCompilationTimings,
    }

    /// Wall-clock split of [`PolynomialCoefficientPlan::compile_symjit_o2`].
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct SymJitCompilationTimings {
        /// Symbolica's shared multi-output evaluator (graph optimization) and
        /// its exact-to-binary64 coefficient mapping.
        pub evaluator: std::time::Duration,
        /// SymJIT O2 translation and machine-code compilation.
        pub jit: std::time::Duration,
    }

    impl PolynomialCoefficientPlan {
        pub fn compile_symjit_o2(&self) -> Result<SymJitCoefficientEvaluator> {
            if std::env::var_os("SYMJIT_TOML").is_some()
                || std::fs::exists("symjit.toml")
                    .map_err(|e| CoefficientError::Jit(e.to_string()))?
            {
                return Err(CoefficientError::Jit("external SymJIT configuration overrides must be unset for the explicit O2 policy".into()));
            }
            let mut timings = SymJitCompilationTimings::default();
            let start = std::time::Instant::now();
            let evaluator = self
                .exact_evaluator()?
                .map(|exact| {
                    let mapped = exact
                        .map_coeff_with_prec(&|v| Complex::new(v.re.to_f64(), v.im.to_f64()), 53);
                    timings.evaluator = start.elapsed();
                    let start = std::time::Instant::now();
                    // Complex helper calls consume two real argument slots. Inspect
                    // the shared exported graph; top-level arity is not this bound.
                    let graph = mapped.export_instructions();
                    let mut pending = vec![&graph];
                    let mut visited = std::collections::HashSet::new();
                    let mut maximum = 0;
                    while let Some(graph) = pending.pop() {
                        for instruction in &graph.instructions {
                            if let Instruction::Fun(_, function, _) = instruction {
                                maximum = maximum.max(function.2.len());
                            }
                        }
                        for body in &graph.sub_evaluators {
                            if visited.insert(std::sync::Arc::as_ptr(&body.instructions)) {
                                pending.push(body.instructions.as_ref());
                            }
                        }
                    }
                    let slots = maximum
                        .checked_mul(2)
                        .ok_or(CoefficientError::Overflow)?
                        .max(64);
                    // The pinned SymJIT 2.27.3 configuration caps num_args at 2048.
                    if slots > 2048 {
                        return Err(CoefficientError::Jit(
                            "helper arity exceeds the pinned SymJIT argument-slot limit".into(),
                        ));
                    }
                    let compiled = mapped
                        .jit_compile(
                            JITCompilationSettings::new()
                                .optimization_level(2)
                                .direct_translation(false)
                                .with_option("fastmath", "false")
                                .with_option("fast_complex", "true")
                                .with_option("use_threads", "false")
                                .with_option("use_simd", "false")
                                .with_option("simd_branch", "false")
                                .with_option("enable_simd512", "false")
                                .with_option("num_args", slots.to_string()),
                        )
                        .map_err(CoefficientError::Jit);
                    timings.jit = start.elapsed();
                    compiled
                })
                .transpose()?;
            Ok(SymJitCoefficientEvaluator {
                inputs: self.parameters.len(),
                outputs: self.heads.len(),
                evaluator,
                scratch: Vec::new(),
                timings,
            })
        }
    }

    impl SymJitCoefficientEvaluator {
        /// Setup cost of this evaluator, split into evaluator construction and
        /// SymJIT compilation. Both are zero for a zero polynomial.
        pub fn compilation_timings(&self) -> SymJitCompilationTimings {
            self.timings
        }

        pub fn evaluate_batch(
            &mut self,
            input: &[Complex<f64>],
            output: &mut [Complex<f64>],
            rows: usize,
        ) -> Result<()> {
            validate_dimensions(self.inputs, self.outputs, input.len(), output.len(), rows)?;
            finite(input, self.inputs, true)?;
            if rows == 0 || self.outputs == 0 {
                return Ok(());
            }
            self.scratch.resize(output.len(), Complex::new(0., 0.));
            let evaluator = self
                .evaluator
                .as_mut()
                .expect("nonempty heads have an evaluator");
            if rows == 1 {
                evaluator.evaluate(input, &mut self.scratch);
            } else {
                evaluator.batch_evaluate(input, &mut self.scratch, rows);
            }
            finite(&self.scratch, self.outputs, false)?;
            output.clone_from_slice(&self.scratch);
            Ok(())
        }
    }
}

#[cfg(feature = "numerator-jit")]
pub use jit::{SymJitCoefficientEvaluator, SymJitCompilationTimings};
