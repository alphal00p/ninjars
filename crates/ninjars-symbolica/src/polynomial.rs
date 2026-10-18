//! Numerator callbacks prepared by Symbolica substitution and Laurent polynomials.

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
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
        integer::{Integer, IntegerRing, Z},
        rational::Rational,
    },
    evaluate::{BatchExpressionEvaluator, EvaluationDomain, ExpressionEvaluator},
    id::{Pattern, Replacement},
    poly::{PolyVariable, polynomial::MultivariatePolynomial},
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

type IntegerPolynomial = MultivariatePolynomial<IntegerRing, u16>;

/// Keep the terms whose power of `t` (variable 0) is at least `lowest`.
fn truncate(polynomial: &IntegerPolynomial, lowest: i64) -> IntegerPolynomial {
    if lowest <= 0 {
        return polynomial.clone();
    }
    let mut kept = polynomial.zero_with_capacity(polynomial.nterms());
    for monomial in polynomial {
        if i64::from(monomial.exponents[0]) >= lowest {
            kept.append_monomial(monomial.coefficient.clone(), monomial.exponents);
        }
    }
    kept
}

/// One cut parameterization of the loop variables as integer polynomials with
/// non-negative exponents. Laurent kinds store `t q` instead of `q` (`shift`
/// is one: a component power `q^k` has its orders lowered by `k`).
struct CutSubstitution {
    zero: IntegerPolynomial,
    variables: Vec<PolyVariable>,
    q: [IntegerPolynomial; 4],
    mu: IntegerPolynomial,
    /// Highest power of `t` in each stored component.
    q_top: i64,
    shift: i64,
    /// Power of `t` in the substituted `mu²` (it has no lower orders).
    mu_top: i64,
    x: Option<usize>,
    mu2: Option<usize>,
    /// The mu kind keeps `a.a` as a factor, as the full expansion in `t` does:
    /// a placeholder variable during multiplication, restored at the end.
    opaque: Option<(Atom, Atom)>,
}

impl CutSubstitution {
    /// Uses only `symbols` (the fresh symbols of [`PolynomialNumerator::exact_expansion`]),
    /// so preparation registers no further symbols, which would perturb the
    /// operation order of every evaluator built afterwards.
    fn new(kind: &ExpansionKind, symbols: &[Atom]) -> Self {
        let symbol = |atom: &Atom| PolyVariable::from(atom.get_symbol().unwrap());
        let vector = |index: usize| 3 + 4 * index..7 + 4 * index;
        // [t, x, mu², vectors, scalars], or [t, a.a, a, v] for the mu kind.
        let (indices, x, mu2): (Vec<usize>, _, _) = match kind {
            ExpansionKind::Mu { .. } => ([0, 11].into_iter().chain(3..11).collect(), None, None),
            ExpansionKind::T3 { .. } => (
                [0, 2].into_iter().chain(3..15).chain([19]).collect(),
                None,
                Some(1),
            ),
            ExpansionKind::T2 { .. } => (
                (0..3).chain(3..19).chain(19..22).collect(),
                Some(1),
                Some(2),
            ),
        };
        let variables: Vec<_> = indices.iter().map(|&i| symbol(&symbols[i])).collect();
        let zero = IntegerPolynomial::new(&Z, None, Arc::new(variables.clone()));
        let var = |symbol_index: usize| {
            let position = indices.iter().position(|&i| i == symbol_index).unwrap();
            let mut exponents = vec![0; indices.len()];
            exponents[position] = 1;
            zero.monomial(Integer::one(), exponents)
        };
        let t = var(0);
        let component = |index: usize, mu: usize| var(vector(index).start + mu);
        match kind {
            ExpansionKind::Mu { .. } => {
                let q = std::array::from_fn(|mu| &(&t * &component(0, mu)) + &component(1, mu));
                Self {
                    mu: &(&t * &t) * &var(11),
                    zero,
                    variables,
                    q,
                    q_top: 1,
                    shift: 0,
                    mu_top: 2,
                    x,
                    mu2,
                    opaque: Some((symbols[11].clone(), {
                        let a = FourMomentum::from_args(
                            symbols[3].clone(),
                            symbols[4].clone(),
                            symbols[5].clone(),
                            symbols[6].clone(),
                        );
                        a.dot(&a)
                    })),
                }
            }
            ExpansionKind::T3 { .. } => {
                // t q = t v0 + t² v1 + (s + mu²) v2.
                let mu = var(2);
                let parameter = &var(19) + &mu;
                let q = std::array::from_fn(|i| {
                    &(&(&t * &component(0, i)) + &(&(&t * &t) * &component(1, i)))
                        + &(&parameter * &component(2, i))
                });
                Self {
                    zero,
                    variables,
                    q,
                    mu,
                    q_top: 2,
                    shift: 1,
                    mu_top: 0,
                    x,
                    mu2,
                    opaque: None,
                }
            }
            ExpansionKind::T2 { .. } => {
                // t q = t v0 + t x v1 + t² v2 + (s0 + s1 x + s2 x² + mu²) v3.
                let (xv, mu) = (var(1), var(2));
                let parameter =
                    &(&(&var(19) + &(&var(20) * &xv)) + &(&(&var(21) * &xv) * &xv)) + &mu;
                let q = std::array::from_fn(|i| {
                    &(&(&(&t * &component(0, i)) + &(&(&t * &xv) * &component(1, i)))
                        + &(&(&t * &t) * &component(2, i)))
                        + &(&parameter * &component(3, i))
                });
                Self {
                    zero,
                    variables,
                    q,
                    mu,
                    q_top: 2,
                    shift: 1,
                    mu_top: 0,
                    x,
                    mu2,
                    opaque: None,
                }
            }
        }
    }

    /// Highest power of `t` a numerator monomial reaches after substitution.
    fn top_order(&self, exponents: &[i32]) -> i64 {
        let degree: i64 = exponents[..4].iter().map(|&e| i64::from(e)).sum();
        (self.q_top - self.shift) * degree + self.mu_top * i64::from(exponents[4])
    }

    /// `alias` times the integer polynomial `monomials`, as a normalized atom.
    fn expression(&self, alias: &Atom, monomials: HashMap<Vec<u16>, Integer>) -> Atom {
        let mut variables = self.variables.clone();
        variables.push(PolyVariable::from(alias.get_symbol().unwrap()));
        let mut polynomial = IntegerPolynomial::new(&Z, Some(monomials.len()), Arc::new(variables));
        let mut monomials: Vec<_> = monomials.into_iter().collect();
        monomials.sort_unstable();
        for (mut exponents, coefficient) in monomials {
            exponents.push(1);
            polynomial.append_monomial(coefficient, &exponents);
        }
        polynomial.to_expression()
    }

    fn restore(&self, expression: Atom) -> Atom {
        match &self.opaque {
            Some((placeholder, value)) => expression.replace_multiple(&[Replacement::new(
                Pattern::Literal(placeholder.clone()),
                Pattern::Literal(value.clone()),
            )]),
            None => expression,
        }
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

    /// Coefficients of the requested powers of `t`, `x` and `mu²` after the
    /// cut substitution, with external coefficient `k` replaced by the alias
    /// `symbols[22 + k']` of its first equal coefficient `k'`.
    ///
    /// Substituting and expanding the whole numerator before extracting a few
    /// coefficients grows combinatorially with the rank (44 s for the leading
    /// power of a rank-10 double cut, about 900 s to prepare a sparse rank-12
    /// numerator), while the reduction keeps only the leading powers of `t`
    /// (all of them only when few lines are uncut). Here each monomial is multiplied
    /// out in Symbolica integer polynomials, discarding at every step the terms
    /// that can no longer reach the lowest requested power of `t`. The kept
    /// terms are exactly those of the full expansion, so the normalized atoms,
    /// and hence the evaluators and their results, are unchanged.
    pub(crate) fn laurent_coefficients(&self, kind: &ExpansionKind, symbols: &[Atom]) -> Vec<Atom> {
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
        let Some(lowest) = powers.iter().map(|power| power[0]).min() else {
            return Vec::new();
        };
        let substitution = CutSubstitution::new(kind, symbols);
        let one = substitution.zero.one();
        // Powers of each substituted component, truncated to the orders any
        // monomial can still use (`deficit` below the top power of `t`).
        let budget = (&self.polynomial)
            .into_iter()
            .map(|term| substitution.top_order(term.exponents) - i64::from(lowest))
            .max()
            .unwrap_or(-1);
        let mut component_powers: [Vec<IntegerPolynomial>; 4] =
            std::array::from_fn(|_| vec![one.clone()]);
        let mut wanted: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
        for (index, power) in powers.iter().enumerate() {
            wanted.entry(power.map(i64::from)).or_default().push(index);
        }
        // Integer monomials (without t, x and mu²) per requested power and alias.
        let mut collected: Vec<HashMap<usize, HashMap<Vec<u16>, Integer>>> =
            vec![HashMap::new(); powers.len()];
        for term in &self.polynomial {
            let exponents = term.exponents;
            let top = substitution.top_order(exponents);
            if top < i64::from(lowest) {
                continue;
            }
            let alias = self
                .polynomial
                .coefficients
                .iter()
                .position(|candidate| candidate == term.coefficient)
                .unwrap();
            let allowed = top - i64::from(lowest);
            let mut product = one.clone();
            let mut product_top = 0;
            for component in 0..4 {
                let degree = exponents[component] as usize;
                if degree == 0 {
                    continue;
                }
                let cache = &mut component_powers[component];
                while cache.len() <= degree {
                    let next = cache.len();
                    let raised = cache.last().unwrap() * &substitution.q[component];
                    cache.push(truncate(&raised, substitution.q_top * next as i64 - budget));
                }
                product = &product * &cache[degree];
                product_top += substitution.q_top * degree as i64;
                product = truncate(&product, product_top - allowed);
            }
            let mu_degree = exponents[4] as usize;
            if mu_degree > 0 {
                product = &product * &substitution.mu.pow(mu_degree);
            }
            let shift = substitution.shift * (exponents[..4].iter().sum::<i32>() as i64);
            for monomial in &product {
                let e = monomial.exponents;
                let key = [
                    i64::from(e[0]) - shift,
                    substitution.x.map_or(0, |x| i64::from(e[x])),
                    substitution.mu2.map_or(0, |m| i64::from(e[m])),
                ];
                let Some(indices) = wanted.get(&key) else {
                    continue;
                };
                let mut rest = e.to_vec();
                rest[0] = 0;
                for index in substitution.x.into_iter().chain(substitution.mu2) {
                    rest[index] = 0;
                }
                for &index in indices {
                    let coefficient = collected[index]
                        .entry(alias)
                        .or_default()
                        .entry(rest.clone())
                        .or_insert_with(Integer::zero);
                    *coefficient += monomial.coefficient;
                }
            }
        }
        collected
            .into_iter()
            .map(|by_alias| {
                let mut aliases: Vec<_> = by_alias.into_iter().collect();
                aliases.sort_unstable_by_key(|(alias, _)| *alias);
                let terms: Vec<Atom> = aliases
                    .into_iter()
                    .map(|(alias, monomials)| {
                        substitution.expression(&symbols[22 + alias], monomials)
                    })
                    .collect();
                if terms.is_empty() {
                    Atom::Zero
                } else {
                    substitution.restore(Atom::add_many(&terms))
                }
            })
            .collect()
    }

    /// Fresh symbols prevent collisions with caller-owned parameter names:
    /// `t`, `x`, `mu²`, four vectors, three scalars, then one alias per
    /// external coefficient.
    pub(crate) fn fresh_symbols(&self) -> Vec<Atom> {
        loop {
            let id = NEXT_NAMESPACE.fetch_add(1, Ordering::Relaxed);
            let symbols: Vec<_> = (0..22 + self.polynomial.coefficients.len())
                .map(|i| Atom::var(symbol!(format!("ninjars_cut::c{id}v{i}"))))
                .collect();
            if symbols.iter().all(|value| {
                !self.expression.contains(value)
                    && !self.parameter_names.contains(value)
                    && !self.variables.contains(value)
            }) {
                return symbols;
            }
        }
    }

    pub(crate) fn exact_expansion(
        &self,
        kind: &ExpansionKind,
    ) -> Result<ExpressionEvaluator<Complex<Rational>>> {
        let symbols = self.fresh_symbols();
        let (vectors, scalars) = match kind {
            ExpansionKind::Mu { .. } => (2, 0),
            ExpansionKind::T3 { .. } => (3, 1),
            ExpansionKind::T2 { .. } => (4, 3),
        };
        // Protect the already collected external coefficients through Symbolica
        // evaluator aliases. Expansion then handles only the small cut-variable
        // expressions. Alias definitions stay exact and are shared by every
        // phase-space lane, including when the evaluator is Dualized.
        let expressions = self.laurent_coefficients(kind, &symbols);
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
