//! One-loop master integrals from oneloopmaster.
//!
//! For the fixed-precision carriers (binary64 and DoubleFloat), scalar
//! families (A0, B0, dB0, C0, D0) use oneloopmaster's native direct-Rust
//! implementation generated from its exact expressions. Arbitrary-precision
//! `Float` keeps Symbolica's interpreter, which tracks the significant
//! precision of every input component instead of padding inputs to the
//! working precision. Tensor families without a native implementation and
//! all Dual (derivative) evaluations also use the same exact formulas through
//! the interpreter.

use std::collections::BTreeMap;

use ninjars::{
    BatchShape, Complex, Error, FloatLike, Laurent, MasterFailure, MasterIntegral, MasterProvider,
    MasterRequest, MasterStatus, Result,
};
use oneloop::{NativeEvaluator, NativeFloat, OneLoopExpressions, ScalarIntegral};
use symbolica::{
    domains::{
        dual::{DualNumber, DualNumberStructure, HyperDual},
        rational::Rational,
    },
    evaluate::{BatchExpressionEvaluator, Dualizer, EvaluationDomain, ExpressionEvaluator},
};

type ExactEvaluator = ExpressionEvaluator<Complex<Rational>>;

// These views are formed only after the complete request has passed structural
// validation. Row identities remain those of the caller after compaction.
fn row_request<'a, T>(request: &MasterRequest<'a, T>, row: usize) -> MasterRequest<'a, T> {
    let invariants = request.integral.invariant_count();
    let masses = request.integral.mass_count();
    MasterRequest {
        integral: request.integral,
        invariants: &request.invariants[row * invariants..(row + 1) * invariants],
        masses_squared: &request.masses_squared[row * masses..(row + 1) * masses],
        scales_squared: &request.scales_squared[row..row + 1],
    }
}

fn strict_status(statuses: &[MasterStatus]) -> Result<()> {
    if let Some((row, MasterStatus::Failed(failure))) = statuses
        .iter()
        .enumerate()
        .find(|(_, status)| matches!(status, MasterStatus::Failed(_)))
    {
        return Err(Error::Unsupported(format!(
            "master row {row} failed: {failure:?}"
        )));
    }
    Ok(())
}

/// Reusable master provider for `f64`, `DoubleFloat`, and arbitrary-precision `Float`.
///
/// Constants are mapped directly from rationals at the representative's working
/// precision. No input is converted through binary64. Each instruction and
/// shared helper processes all active lanes; Symbolica retains lazy branches.
/// Working precision specifies constant preparation, not a required precision
/// for every input or output component. Symbolica `Float` tracks significant
/// bits through arithmetic, so cancellation can lower a component's precision
/// even when its absolute error is small. Inputs and computed outputs retain
/// that information; this provider never pads their precision. A successful
/// status reports a finite result in the supported domain, not attainment of
/// the working precision. Accuracy requires reconstruction or convergence checks.
/// `MasterProvider::evaluate_batch` compacts eligible rows, evaluates the family
/// once, and scatters only finite results. Failed rows retain their old output;
/// `evaluate` retains its strict all-or-nothing contract.
/// Expression preparation uses the calling thread, which must provide the stack
/// required by oneloopmaster (128 MiB is its documented regression-test setting).
pub struct SymbolicaMasters<T: NativeFloat> {
    representative: T,
    evaluators: BTreeMap<MasterIntegral, BatchExpressionEvaluator<Complex<T>>>,
    native: BTreeMap<MasterIntegral, NativeEvaluator<T>>,
    arguments: Vec<Complex<T>>,
    results: Vec<Complex<T>>,
    active: Vec<usize>,
    statuses: Vec<MasterStatus>,
}

impl<T: NativeFloat> SymbolicaMasters<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn new(representative: T) -> Self {
        Self {
            representative,
            evaluators: BTreeMap::new(),
            native: BTreeMap::new(),
            arguments: Vec::new(),
            results: Vec::new(),
            active: Vec::new(),
            statuses: Vec::new(),
        }
    }

    /// The oneloopmaster family with a native implementation, if any.
    fn native_family(integral: MasterIntegral) -> Option<ScalarIntegral> {
        match integral {
            MasterIntegral::A0 => Some(ScalarIntegral::A0),
            MasterIntegral::B0 => Some(ScalarIntegral::B0),
            MasterIntegral::DB0 => Some(ScalarIntegral::DB0),
            MasterIntegral::C0 => Some(ScalarIntegral::C0),
            MasterIntegral::D0 => Some(ScalarIntegral::D0),
            _ => None,
        }
    }

    fn native_evaluator(
        &mut self,
        integral: MasterIntegral,
    ) -> Result<Option<&mut NativeEvaluator<T>>> {
        // The native backend converts every input to the working precision;
        // only use it where that precision is fixed by the carrier type.
        let (Some(family), Some(_)) = (Self::native_family(integral), T::FIXED_BITS) else {
            return Ok(None);
        };
        if !self.native.contains_key(&integral) {
            let evaluator = NativeEvaluator::new(family).map_err(Error::Symbolic)?;
            self.native.insert(integral, evaluator);
        }
        Ok(self.native.get_mut(&integral))
    }

    /// Working bits used to prepare constants; computed component precision may differ.
    pub fn precision(&self) -> u32 {
        self.representative.get_precision()
    }

    /// Compile a family and reserve its reusable workspace before timing calls.
    pub fn prepare(&mut self, integral: MasterIntegral, rows: usize) -> Result<()> {
        if self.native_evaluator(integral)?.is_some() {
            return Ok(());
        }
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

    fn validate_values(
        &self,
        request: &MasterRequest<'_, Complex<T>>,
    ) -> std::result::Result<(), MasterFailure> {
        let zero = self.representative.zero();
        for value in request
            .invariants
            .iter()
            .chain(request.masses_squared)
            .chain(request.scales_squared)
        {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(MasterFailure::NonFiniteInput);
            }
        }
        if request
            .invariants
            .iter()
            .any(|value| !value.im.is_fully_zero())
        {
            return Err(MasterFailure::NonRealInvariant);
        }
        if request.masses_squared.iter().any(|value| value.im > zero) {
            return Err(MasterFailure::InvalidMass);
        }
        if request
            .scales_squared
            .iter()
            .any(|value| !value.im.is_fully_zero() || value.re <= zero)
        {
            return Err(MasterFailure::InvalidScale);
        }
        Ok(())
    }

    fn stage(&mut self, request: &MasterRequest<'_, Complex<T>>, strict: bool) -> Result<()> {
        self.active.clear();
        self.arguments.clear();
        self.statuses.clear();
        for row in 0..request.rows() {
            let point = row_request(request, row);
            match self.validate_values(&point) {
                Err(failure) => self.statuses.push(MasterStatus::Failed(failure)),
                Ok(()) => {
                    self.statuses.push(MasterStatus::Success);
                    self.active.push(row);
                    self.arguments.extend_from_slice(point.invariants);
                    self.arguments.extend_from_slice(point.masses_squared);
                    self.arguments.extend_from_slice(point.scales_squared);
                }
            }
        }
        if strict {
            strict_status(&self.statuses)?;
        }
        if self.active.is_empty() {
            return Ok(());
        }
        let rows = self.active.len();
        self.prepare(request.integral, rows)?;
        let size = rows
            .checked_mul(3)
            .ok_or_else(|| Error::Unsupported("master output dimensions overflow usize".into()))?;
        self.results.resize_with(size, || {
            Complex::new(self.representative.zero(), self.representative.zero())
        });
        let results = &mut self.results[..size];
        if let Some(native) = self.native.get_mut(&request.integral) {
            native
                .evaluate_batch(&self.arguments, results, rows)
                .map_err(Error::Symbolic)?;
        } else {
            self.evaluators
                .get_mut(&request.integral)
                .unwrap()
                .evaluate_batch(&self.arguments, results, rows)
                .map_err(|error| Error::Symbolic(error.to_string()))?;
        }
        for (&row, values) in self.active.iter().zip(self.results.as_chunks::<3>().0) {
            if values
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
            {
                self.statuses[row] = MasterStatus::Failed(MasterFailure::NonFiniteResult);
            }
        }
        Ok(())
    }

    fn commit(&self, output: &mut [Laurent<Complex<T>>]) {
        for (&row, values) in self.active.iter().zip(self.results.as_chunks::<3>().0) {
            if self.statuses[row] == MasterStatus::Success {
                output[row].clone_from_slice(values);
            }
        }
        if !self.active.is_empty() {
            oneloop::record_usage();
        }
    }
}

/// Batched higher/mixed derivatives using Symbolica's static or dynamic Dual carriers.
///
/// Coefficients follow the supplied ancestor-closed Symbolica shape, including
/// its Taylor convention: component `alpha` equals the derivative divided by
/// `alpha!`. Derivatives
/// are currently supported and tested for positive real masses and scales in tadpoles and
/// strictly spacelike bubbles, with real parameter seeds. Other domains return
/// an explicit failure status (or an error through strict `evaluate`), including
/// removable zero-momentum limits whose specialized value formulas do not
/// determine their neighborhood derivatives. Shape, domain and finiteness checks
/// cover every component, before compaction or arithmetic. As for
/// [`SymbolicaMasters`], constants use the representative's working precision;
/// each input and computed component keeps its own tracked significant precision.
/// Success does not certify that all derivatives achieve the working precision.
pub struct SymbolicaDualMasters<T: NativeFloat> {
    representative: T,
    shape: Vec<Vec<usize>>,
    evaluators: BTreeMap<MasterIntegral, BatchExpressionEvaluator<Complex<T>>>,
    arguments: Vec<Complex<T>>,
    results: Vec<Complex<T>>,
    active: Vec<usize>,
    statuses: Vec<MasterStatus>,
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
            active: Vec::new(),
            statuses: Vec::new(),
        }
    }

    fn valid_shape<D: DualNumber<Complex<T>>>(&self, value: &D) -> bool {
        !self.shape.is_empty()
            && value.values().len() == self.shape.len()
            && value
                .get_shape()
                .into_iter()
                .eq(self.shape.iter().map(Vec::as_slice))
    }

    fn validate<D: DualNumber<Complex<T>>>(
        &self,
        request: &MasterRequest<'_, D>,
    ) -> std::result::Result<(), MasterFailure> {
        if matches!(request.integral, MasterIntegral::C0 | MasterIntegral::D0) {
            return Err(MasterFailure::UnsupportedDerivativeDomain);
        }
        let zero = self.representative.zero();
        for dual in request
            .invariants
            .iter()
            .chain(request.masses_squared)
            .chain(request.scales_squared)
        {
            if !self.valid_shape(dual) {
                return Err(MasterFailure::DerivativeShape);
            }
            for component in dual.values() {
                if !component.re.is_finite() || !component.im.is_finite() {
                    return Err(MasterFailure::NonFiniteInput);
                }
                if !component.im.is_fully_zero() {
                    return Err(MasterFailure::UnsupportedDerivativeDomain);
                }
            }
        }
        if request
            .invariants
            .iter()
            .any(|value| value.values()[0].re >= zero)
        {
            return Err(MasterFailure::UnsupportedDerivativeDomain);
        }
        if request
            .masses_squared
            .iter()
            .any(|value| value.values()[0].re <= zero)
        {
            return Err(MasterFailure::UnsupportedDerivativeDomain);
        }
        if request
            .scales_squared
            .iter()
            .any(|value| value.values()[0].re <= zero)
        {
            return Err(MasterFailure::InvalidScale);
        }
        Ok(())
    }

    fn stage<D: DualNumber<Complex<T>>>(
        &mut self,
        request: &MasterRequest<'_, D>,
        strict: bool,
    ) -> Result<()> {
        self.active.clear();
        self.arguments.clear();
        self.statuses.clear();
        for row in 0..request.rows() {
            let point = row_request(request, row);
            match self.validate(&point) {
                Err(failure) => self.statuses.push(MasterStatus::Failed(failure)),
                Ok(()) => {
                    self.statuses.push(MasterStatus::Success);
                    self.active.push(row);
                    for dual in point
                        .invariants
                        .iter()
                        .chain(point.masses_squared)
                        .chain(point.scales_squared)
                    {
                        self.arguments.extend_from_slice(dual.values());
                    }
                }
            }
        }
        if strict {
            strict_status(&self.statuses)?;
        }
        if self.active.is_empty() {
            return Ok(());
        }
        let rows = self.active.len();
        let components = self.shape.len();
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
        for (&row, values) in self.active.iter().zip(self.results.chunks_exact(columns)) {
            if values
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
            {
                self.statuses[row] = MasterStatus::Failed(MasterFailure::NonFiniteResult);
            }
        }
        Ok(())
    }

    fn commit<D: FloatLike + DualNumber<Complex<T>>>(
        &self,
        request: &MasterRequest<'_, D>,
        output: &mut [Laurent<D>],
    ) {
        if self.active.is_empty() {
            return;
        }
        let components = self.shape.len();
        for (&row, values) in self
            .active
            .iter()
            .zip(self.results.chunks_exact(3 * components))
        {
            if self.statuses[row] != MasterStatus::Success {
                continue;
            }
            for (coefficient, values) in output[row].iter_mut().zip(values.chunks_exact(components))
            {
                if !self.valid_shape(coefficient) {
                    *coefficient = request.scales_squared[row].zero();
                }
                coefficient.values_mut().clone_from_slice(values);
            }
        }
        oneloop::record_usage();
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
        self.stage(&request, true)?;
        strict_status(&self.statuses)?;
        self.commit(output);
        Ok(())
    }

    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, Complex<T>>,
        output: &mut [Laurent<Complex<T>>],
        statuses: &mut [MasterStatus],
    ) -> Result<()> {
        request.validate(output.len())?;
        BatchShape {
            rows: request.rows(),
            columns: 1,
        }
        .validate("master statuses", statuses.len())?;
        self.stage(&request, false)?;
        self.commit(output);
        statuses.copy_from_slice(&self.statuses);
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
        self.stage(&request, true)?;
        strict_status(&self.statuses)?;
        self.commit(&request, output);
        Ok(())
    }

    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, D>,
        output: &mut [Laurent<D>],
        statuses: &mut [MasterStatus],
    ) -> Result<()> {
        request.validate(output.len())?;
        BatchShape {
            rows: request.rows(),
            columns: 1,
        }
        .validate("master statuses", statuses.len())?;
        self.stage(&request, false)?;
        self.commit(&request, output);
        statuses.copy_from_slice(&self.statuses);
        Ok(())
    }
}
