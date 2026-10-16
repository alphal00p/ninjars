//! OneLOop's exact formulas executed by Symbolica's batched interpreter.

use std::collections::BTreeMap;

use ninjars::{
    Complex, Error, FloatLike, Laurent, MasterIntegral, MasterProvider, MasterRequest, Result,
};
use oneloop::{NativeFloat, OneLoopExpressions};
use symbolica::{
    domains::{
        dual::{DualNumber, DualNumberStructure, HyperDual},
        rational::Rational,
    },
    evaluate::{BatchExpressionEvaluator, Dualizer, EvaluationDomain, ExpressionEvaluator},
};

type ExactEvaluator = ExpressionEvaluator<Complex<Rational>>;

/// Reusable master provider for `f64`, `DoubleFloat`, and arbitrary-precision `Float`.
///
/// Constants are mapped directly from rationals at the representative's working
/// precision. No input is converted through binary64. Each instruction and
/// shared helper processes all active lanes; Symbolica retains lazy branches.
/// Expression preparation uses the calling thread, which must provide the stack
/// required by oneloopmaster (128 MiB is its documented regression-test setting).
pub struct SymbolicaMasters<T> {
    representative: T,
    evaluators: BTreeMap<MasterIntegral, BatchExpressionEvaluator<Complex<T>>>,
    arguments: Vec<Complex<T>>,
    results: Vec<Complex<T>>,
}

impl<T: NativeFloat> SymbolicaMasters<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn new(representative: T) -> Self {
        Self {
            representative,
            evaluators: BTreeMap::new(),
            arguments: Vec::new(),
            results: Vec::new(),
        }
    }

    pub fn precision(&self) -> u32 {
        self.representative.get_precision()
    }

    /// Compile a family and reserve its reusable workspace before timing calls.
    pub fn prepare(&mut self, integral: MasterIntegral, rows: usize) -> Result<()> {
        if !self.evaluators.contains_key(&integral) {
            let exact = Self::exact_evaluator(integral)?;
            let evaluator = Self::numeric_evaluator(exact, &self.representative);
            self.evaluators.insert(integral, evaluator);
        }
        self.evaluators
            .get_mut(&integral)
            .unwrap()
            .reserve(rows)
            .map_err(|error| Error::Symbolic(error.to_string()))
    }

    fn numeric_evaluator(
        exact: ExactEvaluator,
        representative: &T,
    ) -> BatchExpressionEvaluator<Complex<T>> {
        exact
            .map_coeff_with_prec(
                &|value| {
                    Complex::new(
                        representative.from_rational(&value.re),
                        representative.from_rational(&value.im),
                    )
                },
                representative.get_precision(),
            )
            .into_batch_evaluator()
    }

    fn exact_evaluator(integral: MasterIntegral) -> Result<ExactEvaluator> {
        let args = (0..integral.argument_count())
            .map(|index| symbolica::symbol!(format!("ninjars::master_argument_{index}")).to_atom())
            .collect::<Vec<_>>();
        let expressions = OneLoopExpressions::new();
        let series = match integral {
            MasterIntegral::A0 => oneloop::a0(&args[0], &args[1]),
            MasterIntegral::B0 => oneloop::b0(&args[0], &args[1], &args[2], &args[3]),
            MasterIntegral::DB0 => oneloop::db0(&args[0], &args[1], &args[2], &args[3]),
            MasterIntegral::B1 => oneloop::b1(&args[0], &args[1], &args[2], &args[3]),
            MasterIntegral::B11 => oneloop::b11(&args[0], &args[1], &args[2], &args[3]),
            MasterIntegral::B111 => oneloop::b111(&args[0], &args[1], &args[2], &args[3]),
            MasterIntegral::An { rank } => oneloop::an(rank, &args[0], &args[1])
                .map_err(|_| Error::Unsupported("tadpole tensor rank exceeds four".into()))?
                .pop()
                .expect("oneloopmaster includes A0 at every supported rank"),
            MasterIntegral::C0 => expressions.c0(
                [&args[0], &args[1], &args[2]],
                [&args[3], &args[4], &args[5]],
                &args[6],
            ),
            MasterIntegral::D0 => expressions.d0(
                [&args[0], &args[1], &args[2], &args[3], &args[4], &args[5]],
                [&args[6], &args[7], &args[8], &args[9]],
                &args[10],
            ),
        };
        expressions
            .evaluator(series.coefficients(), &args)
            .map_err(|error| Error::Symbolic(error.to_string()))
    }

    fn validate_values(&self, request: &MasterRequest<'_, Complex<T>>) -> Result<()> {
        let zero = self.representative.zero();
        for value in request
            .invariants
            .iter()
            .chain(request.masses_squared)
            .chain(request.scales_squared)
        {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(Error::Unsupported("master arguments must be finite".into()));
            }
            if value.re.get_precision() != self.precision()
                || value.im.get_precision() != self.precision()
            {
                return Err(Error::Unsupported(
                    "master input precision differs from its provider".into(),
                ));
            }
        }
        if request.invariants.iter().any(|value| !value.im.is_zero()) {
            return Err(Error::Unsupported("master invariants must be real".into()));
        }
        if request.masses_squared.iter().any(|value| value.im > zero) {
            return Err(Error::Unsupported(
                "squared masses require nonpositive imaginary parts".into(),
            ));
        }
        if request
            .scales_squared
            .iter()
            .any(|value| !value.im.is_zero() || value.re <= zero)
        {
            return Err(Error::Unsupported(
                "squared scales must be positive and real".into(),
            ));
        }
        Ok(())
    }
}

/// Batched higher/mixed derivatives using Symbolica's static or dynamic Dual carriers.
///
/// Coefficients follow the supplied ancestor-closed Symbolica shape, including
/// its Taylor convention: component `alpha` equals the derivative divided by
/// `alpha!`. Derivatives
/// are currently supported and tested for positive real masses and scales in tadpoles and
/// strictly spacelike bubbles, with real parameter seeds. Other domains return
/// an explicit error, including removable zero-momentum limits whose specialized
/// value formulas do not determine their neighborhood derivatives.
pub struct SymbolicaDualMasters<T> {
    representative: T,
    shape: Vec<Vec<usize>>,
    evaluators: BTreeMap<MasterIntegral, BatchExpressionEvaluator<Complex<T>>>,
    arguments: Vec<Complex<T>>,
    results: Vec<Complex<T>>,
}

impl<T: NativeFloat> SymbolicaDualMasters<T>
where
    Complex<T>: EvaluationDomain,
{
    /// Use the shape of any Symbolica static or dynamic Dual instance.
    pub fn new(representative: T, structure: &impl DualNumberStructure) -> Self {
        Self {
            representative,
            shape: structure
                .get_shape()
                .into_iter()
                .map(<[usize]>::to_vec)
                .collect(),
            evaluators: BTreeMap::new(),
            arguments: Vec::new(),
            results: Vec::new(),
        }
    }

    fn validate<D: DualNumber<Complex<T>>>(&self, request: &MasterRequest<'_, D>) -> Result<()> {
        if matches!(request.integral, MasterIntegral::C0 | MasterIntegral::D0) {
            return Err(Error::Unsupported("triangle/box master derivatives require additional continuation and removable-limit validation".into()));
        }
        let zero = self.representative.zero();
        for dual in request
            .invariants
            .iter()
            .chain(request.masses_squared)
            .chain(request.scales_squared)
        {
            if dual.get_shape() != self.shape.iter().map(Vec::as_slice).collect::<Vec<_>>() {
                return Err(Error::Unsupported("master Dual shapes differ".into()));
            }
            for component in dual.values() {
                if !component.re.is_finite() || !component.im.is_finite() || !component.im.is_zero()
                {
                    return Err(Error::Unsupported("supported master derivatives require finite real values and real parameter seeds".into()));
                }
                if component.re.get_precision() != self.representative.get_precision()
                    || component.im.get_precision() != self.representative.get_precision()
                {
                    return Err(Error::Unsupported(
                        "master Dual component precision differs from its provider".into(),
                    ));
                }
            }
        }
        if request
            .invariants
            .iter()
            .any(|value| value.values()[0].re >= zero)
        {
            return Err(Error::Unsupported("bubble derivatives currently require strictly spacelike nonzero momentum; zero-momentum neighborhood expansions are not yet implemented".into()));
        }
        if request
            .masses_squared
            .iter()
            .chain(request.scales_squared)
            .any(|value| value.values()[0].re <= zero)
        {
            return Err(Error::Unsupported("master derivatives currently require strictly positive masses and scales; massless and nonanalytic limits are not yet implemented".into()));
        }
        Ok(())
    }
}

impl<T: NativeFloat, D: FloatLike + DualNumber<Complex<T>>> MasterProvider<D>
    for SymbolicaDualMasters<T>
where
    Complex<T>: EvaluationDomain,
{
    fn evaluate(&mut self, request: MasterRequest<'_, D>, output: &mut [Laurent<D>]) -> Result<()> {
        request.validate(output.len())?;
        let rows = request.rows();
        let components = self.shape.len();
        if rows == 0 {
            return Ok(());
        }
        self.validate(&request)?;
        if !self.evaluators.contains_key(&request.integral) {
            let exact = SymbolicaMasters::<T>::exact_evaluator(request.integral)?;
            let structure = HyperDual::<Complex<Rational>>::new(self.shape.clone());
            let dual = exact
                .vectorize(&Dualizer::new(structure, Vec::new()))
                .map_err(|error| Error::Symbolic(error.to_string()))?;
            self.evaluators.insert(
                request.integral,
                SymbolicaMasters::numeric_evaluator(dual, &self.representative),
            );
        }
        let invariants = request.integral.invariant_count();
        let masses = request.integral.mass_count();
        self.arguments.clear();
        for row in 0..rows {
            for dual in request.invariants[row * invariants..(row + 1) * invariants]
                .iter()
                .chain(&request.masses_squared[row * masses..(row + 1) * masses])
                .chain(std::iter::once(&request.scales_squared[row]))
            {
                self.arguments.extend_from_slice(dual.values());
            }
        }
        let columns = components
            .checked_mul(3)
            .ok_or_else(|| Error::Unsupported("master Dual dimensions overflow usize".into()))?;
        let size = rows
            .checked_mul(columns)
            .ok_or_else(|| Error::Unsupported("master Dual dimensions overflow usize".into()))?;
        self.results.resize_with(size, || {
            Complex::new(self.representative.zero(), self.representative.zero())
        });
        self.evaluators
            .get_mut(&request.integral)
            .unwrap()
            .evaluate_batch(&self.arguments, &mut self.results, rows)
            .map_err(|error| Error::Symbolic(error.to_string()))?;
        if self
            .results
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(Error::Unsupported(
                "master derivative evaluation is nonfinite".into(),
            ));
        }
        for (row, values) in output.iter_mut().zip(self.results.chunks_exact(columns)) {
            for (coefficient, values) in row.iter_mut().zip(values.chunks_exact(components)) {
                if coefficient.get_shape()
                    == self.shape.iter().map(Vec::as_slice).collect::<Vec<_>>()
                {
                    coefficient.values_mut().clone_from_slice(values);
                } else {
                    *coefficient = request.scales_squared[0].zero();
                    coefficient.values_mut().clone_from_slice(values);
                }
            }
        }
        oneloop::record_usage();
        Ok(())
    }
}

impl<T: NativeFloat> MasterProvider<Complex<T>> for SymbolicaMasters<T>
where
    Complex<T>: EvaluationDomain,
{
    fn evaluate(
        &mut self,
        request: MasterRequest<'_, Complex<T>>,
        output: &mut [Laurent<Complex<T>>],
    ) -> Result<()> {
        request.validate(output.len())?;
        self.validate_values(&request)?;
        let rows = request.rows();
        if rows == 0 {
            return Ok(());
        }
        self.prepare(request.integral, rows)?;
        let invariants = request.integral.invariant_count();
        let masses = request.integral.mass_count();
        self.arguments.clear();
        for row in 0..rows {
            self.arguments
                .extend_from_slice(&request.invariants[row * invariants..(row + 1) * invariants]);
            self.arguments
                .extend_from_slice(&request.masses_squared[row * masses..(row + 1) * masses]);
            self.arguments.push(request.scales_squared[row].clone());
        }
        let size = rows
            .checked_mul(3)
            .ok_or_else(|| Error::Unsupported("master output dimensions overflow usize".into()))?;
        self.results.resize_with(size, || {
            Complex::new(self.representative.zero(), self.representative.zero())
        });
        self.evaluators
            .get_mut(&request.integral)
            .unwrap()
            .evaluate_batch(&self.arguments, &mut self.results, rows)
            .map_err(|error| Error::Symbolic(error.to_string()))?;
        if self
            .results
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(Error::Unsupported(
                "master evaluation is nonfinite at the supplied kinematics".into(),
            ));
        }
        for (row, values) in output.iter_mut().zip(self.results.as_chunks::<3>().0) {
            row.clone_from_slice(values);
        }
        oneloop::record_usage();
        Ok(())
    }
}
