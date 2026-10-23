//! Binary64 masters from oneloopmaster's direct Rust implementations
//! (`NativeEvaluator<f64>` for the generated scalar integrals and
//! `NativeTensorEvaluator<f64>` for OneLOop's bubble and tadpole tensor
//! routines): no Symbolica expression is interpreted or JIT-compiled at
//! evaluation time. The rank-four bubble family (`B1111`, for numerator rank
//! n+2) is oneloopmaster's `TensorIntegral::B1111`: OneLOop's `bub1111` with
//! the corrected mass-swap sign (OneLOop itself has the wrong sign there).
//!
//! Inputs first pass OneLOop's on-shell threshold exactly as original Ninja
//! configures it (`olo_onshell(INFRARED_EPS = 1e-9)`): invariants and squared
//! masses below the threshold become zero and real invariants within it of a
//! real squared mass become exactly on shell.

use ninjars::fast::{C64, FastMaster, FastMasters};
use ninjars::{
    BatchShape, Complex, Error, Laurent, MasterFailure, MasterIntegral, MasterProvider,
    MasterRequest, MasterStatus, Result,
};
use oneloop::{NativeEvaluator, NativeTensorEvaluator, ScalarIntegral, TensorIntegral};

type L3 = [C64; 3];

fn is_zero(z: C64) -> bool {
    z.re == 0. && z.im == 0.
}

/// Original Ninja's `INFRARED_EPS`, passed to OneLOop as its on-shell threshold.
pub const ONSHELL_THRESHOLD: f64 = 1e-9;

/// Apply OneLOop's on-shell threshold to `[invariants..., masses..., mu²]`.
fn snap_onshell(args: &mut [C64], invariants: usize) {
    let masses = args.len() - invariants - 1;
    let small = |z: &C64| z.re.abs() + z.im.abs() < ONSHELL_THRESHOLD;
    for mass in &mut args[invariants..invariants + masses] {
        if small(mass) {
            *mass = C64::ZERO;
        }
    }
    let (momenta, rest) = args.split_at_mut(invariants);
    for momentum in momenta {
        if small(momentum) {
            *momentum = C64::ZERO;
        } else if let Some(mass) = rest[..masses]
            .iter()
            .find(|m| m.im == 0. && (momentum.re - m.re).abs() < ONSHELL_THRESHOLD)
        {
            *momentum = *mass;
        }
    }
}

/// One-loop masters in binary64 through oneloopmaster's generated native Rust.
///
/// Implements both [`FastMasters`] and the generic
/// [`MasterProvider<Complex<f64>>`]; identical arguments produce bitwise
/// identical values through either interface. Rows with invalid inputs
/// (non-finite values, complex invariants, masses with positive imaginary
/// parts, non-positive scales) or non-finite results fail individually.
pub struct OneLoopNativeMasters {
    pole_selection: oneloop::PoleSelection,
    a0: NativeEvaluator<f64>,
    b0: NativeEvaluator<f64>,
    db0: NativeEvaluator<f64>,
    c0: NativeEvaluator<f64>,
    d0: NativeEvaluator<f64>,
    an: NativeTensorEvaluator<f64>,
    b11: NativeTensorEvaluator<f64>,
    b111: NativeTensorEvaluator<f64>,
    b1111: NativeTensorEvaluator<f64>,
    args: [C64; 11],
    input: Vec<Complex<f64>>,
    output: [Complex<f64>; 15],
    batch_input: Vec<Complex<f64>>,
    batch_output: Vec<Complex<f64>>,
}

fn native_error(error: String) -> Error {
    Error::Unsupported(format!("oneloopmaster native backend: {error}"))
}

impl OneLoopNativeMasters {
    pub fn new() -> Result<Self> {
        Ok(Self {
            pole_selection: oneloop::PoleSelection::BOTH,
            a0: NativeEvaluator::new(ScalarIntegral::A0).map_err(native_error)?,
            b0: NativeEvaluator::new(ScalarIntegral::B0).map_err(native_error)?,
            db0: NativeEvaluator::new(ScalarIntegral::DB0).map_err(native_error)?,
            c0: NativeEvaluator::new(ScalarIntegral::C0).map_err(native_error)?,
            d0: NativeEvaluator::new(ScalarIntegral::D0).map_err(native_error)?,
            an: NativeTensorEvaluator::new(TensorIntegral::An).map_err(native_error)?,
            b11: NativeTensorEvaluator::new(TensorIntegral::B11).map_err(native_error)?,
            b111: NativeTensorEvaluator::new(TensorIntegral::B111).map_err(native_error)?,
            b1111: NativeTensorEvaluator::new(TensorIntegral::B1111).map_err(native_error)?,
            args: [C64::ZERO; 11],
            input: Vec::with_capacity(11),
            output: [Complex::new(0., 0.); 15],
            batch_input: Vec::new(),
            batch_output: Vec::new(),
        })
    }

    /// Select the bare loop's poles, preserving finite coefficients. Selected
    /// UV/IR providers can be used with the generic engine; FastAmplitude
    /// rejects them before reduction because it skips scaleless bubbles.
    pub fn with_pole_selection(mut self, selection: oneloop::PoleSelection) -> Self {
        self.set_pole_selection(selection);
        self
    }

    /// Changing this selection also updates all retained scalar/tensor
    /// evaluators. Each evaluator projects only successful ordinary results.
    pub fn set_pole_selection(&mut self, selection: oneloop::PoleSelection) {
        self.pole_selection = selection;
        self.a0.set_pole_selection(selection);
        self.b0.set_pole_selection(selection);
        self.db0.set_pole_selection(selection);
        self.c0.set_pole_selection(selection);
        self.d0.set_pole_selection(selection);
        self.an.set_pole_selection(selection);
        self.b11.set_pole_selection(selection);
        self.b111.set_pole_selection(selection);
        self.b1111.set_pole_selection(selection);
    }

    pub fn pole_selection(&self) -> oneloop::PoleSelection {
        self.pole_selection
    }

    /// oneloopmaster's native domain contract for one thresholded row.
    fn row_is_valid(row: &[C64], invariants: usize) -> bool {
        let mu2 = row[row.len() - 1];
        row.iter().all(|z| z.is_finite())
            && row[..invariants].iter().all(|z| z.im == 0.)
            && row[invariants..row.len() - 1].iter().all(|z| z.im <= 0.)
            && mu2.im == 0.
            && mu2.re > 0.
    }

    /// Stage every row of a family call (thresholded, in Symbolica's
    /// complex layout). Returns whether all rows satisfy the domain contract.
    fn stage_batch(&mut self, args: &[C64], arity: usize, invariants: usize) -> bool {
        self.batch_input.clear();
        let mut valid = true;
        for point in args.chunks_exact(arity) {
            let row = &mut self.args[..arity];
            row.copy_from_slice(point);
            snap_onshell(row, invariants);
            valid &= Self::row_is_valid(row, invariants);
            self.batch_input
                .extend(row.iter().map(|z| z.to_symbolica()));
        }
        valid
    }

    /// Copy and threshold the arguments of one row.
    fn stage(&mut self, args: &[C64], invariants: usize) {
        self.args[..args.len()].copy_from_slice(args);
        snap_onshell(&mut self.args[..args.len()], invariants);
        self.input.clear();
        self.input
            .extend(self.args[..args.len()].iter().map(|z| z.to_symbolica()));
    }

    fn scalar(&mut self, family: ScalarIntegral, args: &[C64]) -> Option<L3> {
        let invariants = match family {
            ScalarIntegral::A0 => 0,
            ScalarIntegral::B0 | ScalarIntegral::DB0 => 1,
            ScalarIntegral::C0 => 3,
            ScalarIntegral::D0 => 6,
        };
        self.stage(args, invariants);
        // B0 is symmetric in its masses. The generated B0 returns NaN for
        // some (m0² != 0, m1² = 0) points (an inexact unit root in its
        // general branch); the (0, m²) ordering takes the closed form.
        if family == ScalarIntegral::B0 && is_zero(self.args[2]) && !is_zero(self.args[1]) {
            self.input.swap(1, 2);
        }
        let evaluator = match family {
            ScalarIntegral::A0 => &mut self.a0,
            ScalarIntegral::B0 => &mut self.b0,
            ScalarIntegral::DB0 => &mut self.db0,
            ScalarIntegral::C0 => &mut self.c0,
            ScalarIntegral::D0 => &mut self.d0,
        };
        evaluator
            .evaluate(&self.input, &mut self.output[..3])
            .ok()?;
        let out = [
            C64::from_symbolica(self.output[0]),
            C64::from_symbolica(self.output[1]),
            C64::from_symbolica(self.output[2]),
        ];
        out.iter().all(|z| z.is_finite()).then_some(out)
    }

    /// Tensor family `family` at one row; writes `family.coefficients()` triples.
    fn tensor(&mut self, family: TensorIntegral, args: &[C64], out: &mut [L3]) -> bool {
        self.stage(args, if family == TensorIntegral::An { 0 } else { 1 });
        let evaluator = match family {
            TensorIntegral::An => &mut self.an,
            TensorIntegral::B11 => &mut self.b11,
            TensorIntegral::B111 => &mut self.b111,
            TensorIntegral::B1111 => &mut self.b1111,
        };
        let outputs = family.outputs();
        if evaluator
            .evaluate(&self.input, &mut self.output[..outputs])
            .is_err()
        {
            return false;
        }
        for (triple, values) in out
            .iter_mut()
            .zip(self.output[..outputs].as_chunks::<3>().0)
        {
            *triple = [
                C64::from_symbolica(values[0]),
                C64::from_symbolica(values[1]),
                C64::from_symbolica(values[2]),
            ];
        }
        out.iter().flatten().all(|z| z.is_finite())
    }

    /// `[B0, B1, B11, B111]` at `(p², m0², m1², mu²)`.
    pub fn bubble(&mut self, p: C64, m0: C64, m1: C64, mu2: C64) -> Option<[L3; 4]> {
        let mut out = [[C64::ZERO; 3]; 4];
        self.tensor(TensorIntegral::B111, &[p, m0, m1, mu2], &mut out)
            .then_some(out)
    }

    /// `[B0, B1, B11, B111, B1111]` at `(p², m0², m1², mu²)` (rank n+2
    /// bubbles). The first four coefficients are bitwise those of
    /// [`Self::bubble`].
    pub fn bubble_rank4(&mut self, p: C64, m0: C64, m1: C64, mu2: C64) -> Option<[L3; 5]> {
        let mut out = [[C64::ZERO; 3]; 5];
        self.tensor(TensorIntegral::B1111, &[p, m0, m1, mu2], &mut out)
            .then_some(out)
    }

    /// Tadpole tensor coefficient returned last by OneLOop `an(rank)`:
    /// A0 for rank 0/1, A00 for rank 2/3 and A0000 for rank 4.
    fn an(&mut self, rank: usize, m: C64, mu2: C64) -> Option<L3> {
        let mut out = [[C64::ZERO; 3]; 3];
        self.tensor(TensorIntegral::An, &[m, mu2], &mut out)
            .then_some(out[(rank / 2).min(2)])
    }

    /// Evaluate one generic NinjaRS master row.
    fn generic_row(
        &mut self,
        integral: MasterIntegral,
        invariants: &[C64],
        masses: &[C64],
        mu2: C64,
    ) -> std::result::Result<L3, MasterFailure> {
        let values = invariants.iter().chain(masses).chain([&mu2]);
        if values.clone().any(|z| !z.is_finite()) {
            return Err(MasterFailure::NonFiniteInput);
        }
        if invariants.iter().any(|z| z.im != 0.) {
            return Err(MasterFailure::NonRealInvariant);
        }
        if masses.iter().any(|z| z.im > 0.) {
            return Err(MasterFailure::InvalidMass);
        }
        if mu2.im != 0. || mu2.re <= 0. {
            return Err(MasterFailure::InvalidScale);
        }
        let mut args = [C64::ZERO; 11];
        let count = invariants.len() + masses.len() + 1;
        args[..invariants.len()].copy_from_slice(invariants);
        args[invariants.len()..count - 1].copy_from_slice(masses);
        args[count - 1] = mu2;
        let args = &args[..count];
        let value = match integral {
            MasterIntegral::A0 => self.scalar(ScalarIntegral::A0, args),
            // Bubbles come from the same tensor families as the batched
            // `FastMasters` interface, so both interfaces agree bit for bit.
            MasterIntegral::B0 | MasterIntegral::B1 | MasterIntegral::B11 => {
                let slot = match integral {
                    MasterIntegral::B0 => 0,
                    MasterIntegral::B1 => 1,
                    _ => 2,
                };
                let mut out = [[C64::ZERO; 3]; 3];
                self.tensor(TensorIntegral::B11, args, &mut out)
                    .then_some(out[slot])
            }
            MasterIntegral::DB0 => self.scalar(ScalarIntegral::DB0, args),
            MasterIntegral::C0 => self.scalar(ScalarIntegral::C0, args),
            MasterIntegral::D0 => self.scalar(ScalarIntegral::D0, args),
            MasterIntegral::B111 => self
                .bubble(args[0], args[1], args[2], args[3])
                .map(|b| b[3]),
            MasterIntegral::An { rank } => self.an(rank, args[0], args[1]),
            MasterIntegral::B1111 => self
                .bubble_rank4(args[0], args[1], args[2], args[3])
                .map(|b| b[4]),
        };
        value.ok_or(MasterFailure::NonFiniteResult)
    }
}

impl FastMasters for OneLoopNativeMasters {
    fn requires_scaleless_bubbles(&self) -> bool {
        self.pole_selection.uv != self.pole_selection.ir
    }

    fn evaluate(
        &mut self,
        family: FastMaster,
        args: &[C64],
        out: &mut [[C64; 3]],
        ok: &mut [bool],
    ) -> Result<()> {
        let rows = ok.len();
        let arity = family.arity();
        let width = family.outputs();
        BatchShape {
            rows,
            columns: arity,
        }
        .validate("native master arguments", args.len())?;
        BatchShape {
            rows,
            columns: width,
        }
        .validate("native master outputs", out.len())?;
        if rows == 0 {
            return Ok(());
        }
        let invariants = match family {
            FastMaster::A0 => 0,
            FastMaster::BubbleRank2 | FastMaster::BubbleRank3 | FastMaster::BubbleRank4 => 1,
            FastMaster::C0 => 3,
            FastMaster::D0 => 6,
        };
        // Common case: one batched native call for the whole family. Rows
        // violating the domain contract (the native batch call would reject
        // the whole batch) fall back to row-wise evaluation below.
        if self.stage_batch(args, arity, invariants) {
            self.batch_output
                .resize(rows * width * 3, Complex::new(0., 0.));
            let input = &self.batch_input;
            let output = &mut self.batch_output;
            let result = match family {
                FastMaster::A0 => self.a0.evaluate_batch(input, output, rows),
                FastMaster::C0 => self.c0.evaluate_batch(input, output, rows),
                FastMaster::D0 => self.d0.evaluate_batch(input, output, rows),
                FastMaster::BubbleRank2 => self.b11.evaluate_batch(input, output, rows),
                FastMaster::BubbleRank3 => self.b111.evaluate_batch(input, output, rows),
                FastMaster::BubbleRank4 => self.b1111.evaluate_batch(input, output, rows),
            };
            if result.is_ok() {
                for ((values, result), ok) in self
                    .batch_output
                    .chunks_exact(width * 3)
                    .zip(out.chunks_exact_mut(width))
                    .zip(ok.iter_mut())
                {
                    for (triple, value) in result.iter_mut().zip(values.as_chunks::<3>().0) {
                        *triple = [
                            C64::from_symbolica(value[0]),
                            C64::from_symbolica(value[1]),
                            C64::from_symbolica(value[2]),
                        ];
                    }
                    *ok = result.iter().flatten().all(|z| z.is_finite());
                }
                return Ok(());
            }
        }
        for (row, (point, result)) in args
            .chunks_exact(arity)
            .zip(out.chunks_exact_mut(width))
            .enumerate()
        {
            ok[row] = match family {
                FastMaster::A0 | FastMaster::C0 | FastMaster::D0 => {
                    let scalar = match family {
                        FastMaster::A0 => ScalarIntegral::A0,
                        FastMaster::C0 => ScalarIntegral::C0,
                        _ => ScalarIntegral::D0,
                    };
                    match self.scalar(scalar, point) {
                        Some(value) => {
                            result[0] = value;
                            true
                        }
                        None => false,
                    }
                }
                FastMaster::BubbleRank2 => self.tensor(TensorIntegral::B11, point, result),
                FastMaster::BubbleRank3 => self.tensor(TensorIntegral::B111, point, result),
                FastMaster::BubbleRank4 => self.tensor(TensorIntegral::B1111, point, result),
            };
        }
        Ok(())
    }
}

fn request_rows(request: &MasterRequest<'_, Complex<f64>>) -> usize {
    request.rows()
}

impl MasterProvider<Complex<f64>> for OneLoopNativeMasters {
    fn requires_scaleless_bubbles(&self) -> bool {
        self.pole_selection.uv != self.pole_selection.ir
    }

    fn evaluate(
        &mut self,
        request: MasterRequest<'_, Complex<f64>>,
        output: &mut [Laurent<Complex<f64>>],
    ) -> Result<()> {
        let mut statuses = vec![MasterStatus::Success; request_rows(&request)];
        let mut staged = output.to_vec();
        self.evaluate_batch(request, &mut staged, &mut statuses)?;
        if let Some((row, MasterStatus::Failed(failure))) = statuses
            .iter()
            .enumerate()
            .find(|(_, s)| matches!(s, MasterStatus::Failed(_)))
        {
            return Err(Error::Unsupported(format!(
                "master row {row} failed: {failure:?}"
            )));
        }
        output.copy_from_slice(&staged);
        Ok(())
    }

    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, Complex<f64>>,
        output: &mut [Laurent<Complex<f64>>],
        statuses: &mut [MasterStatus],
    ) -> Result<()> {
        request.validate(output.len())?;
        let rows = request.rows();
        if statuses.len() != rows {
            return Err(Error::Shape {
                name: "master statuses",
                expected: rows,
                actual: statuses.len(),
            });
        }
        let ni = request.integral.invariant_count();
        let nm = request.integral.mass_count();
        for row in 0..rows {
            let invariants: Vec<C64> = request.invariants[row * ni..(row + 1) * ni]
                .iter()
                .map(|z| C64::from_symbolica(*z))
                .collect();
            let masses: Vec<C64> = request.masses_squared[row * nm..(row + 1) * nm]
                .iter()
                .map(|z| C64::from_symbolica(*z))
                .collect();
            let mu2 = C64::from_symbolica(request.scales_squared[row]);
            match self.generic_row(request.integral, &invariants, &masses, mu2) {
                Ok(value) => {
                    output[row] = value.map(C64::to_symbolica);
                    statuses[row] = MasterStatus::Success;
                }
                Err(failure) => statuses[row] = MasterStatus::Failed(failure),
            }
        }
        Ok(())
    }
}
