//! Binary64 masters from oneloopmaster's ahead-of-time generated Rust
//! (`NativeEvaluator<f64>`, feature `generated-evaluators`): no Symbolica
//! expression is interpreted or JIT-compiled at evaluation time.
//!
//! A0, B0, C0, D0 (and dB0) are the generated native functions. The bubble
//! tensor coefficients B1, B11, B111 (with B00, B001) are built numerically
//! from native A0/B0 with exactly the Passarino-Veltman relations and
//! zero-momentum limits that `oneloop::{b1, b11, b111}` encode symbolically
//! (exact-zero dispatch on `p²` and the masses, lower-lip logarithms).

use ninjars::fast::{C64, FastMaster, FastMasters};
use ninjars::{
    BatchShape, Complex, Error, Laurent, MasterFailure, MasterIntegral, MasterProvider,
    MasterRequest, MasterStatus, Result,
};
use oneloop::{NativeEvaluator, ScalarIntegral};

type L3 = [C64; 3];

fn zero3() -> L3 {
    [C64::ZERO; 3]
}

fn is_zero(z: C64) -> bool {
    z.re == 0. && z.im == 0.
}

/// Principal logarithm continued to the lower lip of the negative real axis
/// (oneloopmaster's `physical_log`).
fn physical_log(z: C64) -> C64 {
    if z.im != 0. {
        C64::new(z.abs().ln(), z.im.atan2(z.re))
    } else if z.re < 0. {
        C64::new((-z.re).ln(), -std::f64::consts::PI)
    } else {
        C64::real(z.re.ln())
    }
}

/// `x log(x/mu²)`, zero at `x = 0`.
fn x_log_ratio(x: C64, mu2: f64) -> C64 {
    if is_zero(x) {
        C64::ZERO
    } else {
        x * physical_log(x / mu2)
    }
}

/// oneloopmaster's `bnlog(rank, root)`.
fn bnlog(rank: usize, root: C64) -> C64 {
    if is_zero(root) {
        return C64::ZERO;
    }
    if root == C64::ONE {
        return C64::real(-(1..=rank + 1).map(|k| 1. / k as f64).sum::<f64>());
    }
    let mut power = C64::ONE;
    let mut polynomial = C64::ZERO;
    for p in 0..=rank {
        polynomial += power / (rank + 1 - p) as f64;
        power *= root;
    }
    (C64::ONE - power) * physical_log(C64::ONE - C64::ONE / root) - polynomial
}

fn b0_at_zero_momentum(m0: C64, m1: C64, mu2: f64) -> C64 {
    let difference = m0 - m1;
    if !is_zero(difference) {
        C64::ONE - (x_log_ratio(m0, mu2) - x_log_ratio(m1, mu2)) / difference
    } else if is_zero(m0) {
        C64::ZERO
    } else {
        -physical_log(m0 / mu2)
    }
}

/// Shared zero-momentum anchors `(log(m1/mu²) ± bnlog(rank, m1/(m1-m0)))/(rank+1)`.
fn anchor(rank: usize, sign: f64, m0: C64, m1: C64, mu2: f64) -> C64 {
    let logarithm = physical_log(m1 / mu2);
    let difference = m1 - m0;
    let scale = sign / (rank + 1) as f64;
    if is_zero(difference) {
        logarithm * scale
    } else {
        (logarithm + bnlog(rank, m1 / difference)) * scale
    }
}

fn anchor1(m0: C64, m1: C64, mu2: f64) -> C64 {
    anchor(1, 1., m0, m1, mu2)
}
fn anchor11(m0: C64, m1: C64, mu2: f64) -> C64 {
    anchor(2, -1., m0, m1, mu2)
}
fn anchor111(m0: C64, m1: C64, mu2: f64) -> C64 {
    anchor(3, 1., m0, m1, mu2)
}

fn b1_zero(m0: C64, m1: C64, mu2: f64) -> C64 {
    if !is_zero(m1) {
        -b0_at_zero_momentum(m0, m1, mu2) - anchor1(m0, m1, mu2)
    } else if !is_zero(m0) {
        anchor1(m1, m0, mu2)
    } else {
        C64::ZERO
    }
}

fn b11_zero(m0: C64, m1: C64, mu2: f64) -> C64 {
    if !is_zero(m1) {
        anchor11(m0, m1, mu2) + anchor1(m0, m1, mu2) * 2. + b0_at_zero_momentum(m0, m1, mu2)
    } else if !is_zero(m0) {
        anchor11(m1, m0, mu2)
    } else {
        C64::ZERO
    }
}

fn b111_zero(m0: C64, m1: C64, mu2: f64) -> C64 {
    if !is_zero(m1) {
        -anchor111(m0, m1, mu2)
            - anchor11(m0, m1, mu2) * 3.
            - anchor1(m0, m1, mu2) * 3.
            - b0_at_zero_momentum(m0, m1, mu2)
    } else if !is_zero(m0) {
        anchor111(m1, m0, mu2)
    } else {
        C64::ZERO
    }
}

/// Bubble coefficients `[B0, B1, B11, B111]` from native `B0` and `A0`.
fn bubble_tensors(p: f64, m0: C64, m1: C64, mu2: f64, b0: L3, a0_m0: L3, a0_m1: L3) -> [L3; 4] {
    if p == 0. && is_zero(m0) && is_zero(m1) {
        return [zero3(); 4];
    }
    let massive = !is_zero(m0) || !is_zero(m1);
    let pole_or = |value: f64| if p != 0. || massive { value } else { 0. };
    let kinematic = m1 - m0 - p;
    let b1 = if p != 0. {
        [
            (a0_m0[0] - a0_m1[0] + (m1 - m0 - p) * b0[0]) / (2. * p),
            C64::real(-0.5),
            C64::ZERO,
        ]
    } else {
        [b1_zero(m0, m1, mu2), C64::real(pole_or(-0.5)), C64::ZERO]
    };
    // B00 = [nf/6 + np/9, np/6]
    let b00_num = [
        a0_m1[0] + m0 * b0[0] * 2. - kinematic * b1[0],
        a0_m1[1] + m0 * b0[1] * 2. - kinematic * b1[1],
    ];
    let b00 = [b00_num[0] / 6. + b00_num[1] / 9., b00_num[1] / 6.];
    let b11 = if p != 0. {
        let d = 2. * p;
        [
            (a0_m1[0] + kinematic * b1[0] - b00[0] * 2.) / d,
            (a0_m1[1] + kinematic * b1[1] - b00[1] * 2.) / d,
            C64::ZERO,
        ]
    } else {
        [
            b11_zero(m0, m1, mu2),
            C64::real(pole_or(1. / 3.)),
            C64::ZERO,
        ]
    };
    // B001 = [nf/8 + np/16, np/8]
    let b001_num = [
        m0 * b1[0] * 2. - a0_m1[0] - kinematic * b11[0],
        m0 * b1[1] * 2. - a0_m1[1] - kinematic * b11[1],
    ];
    let b001 = [b001_num[0] / 8. + b001_num[1] / 16., b001_num[1] / 8.];
    let b111 = if p != 0. {
        let d = 2. * p;
        [
            (-a0_m1[0] + kinematic * b11[0] - b001[0] * 4.) / d,
            (-a0_m1[1] + kinematic * b11[1] - b001[1] * 4.) / d,
            C64::ZERO,
        ]
    } else {
        [b111_zero(m0, m1, mu2), C64::real(pole_or(-0.25)), C64::ZERO]
    };
    [b0, b1, b11, b111]
}

/// One-loop masters in binary64 through oneloopmaster's generated native Rust.
///
/// Implements both [`FastMasters`] and the generic
/// [`MasterProvider<Complex<f64>>`]; identical arguments produce bitwise
/// identical values through either interface. Rows with invalid inputs
/// (non-finite values, complex invariants, masses with positive imaginary
/// parts, non-positive scales) or non-finite results fail individually.
pub struct OneLoopNativeMasters {
    a0: NativeEvaluator<f64>,
    b0: NativeEvaluator<f64>,
    db0: NativeEvaluator<f64>,
    c0: NativeEvaluator<f64>,
    d0: NativeEvaluator<f64>,
    input: Vec<Complex<f64>>,
    output: [Complex<f64>; 3],
}

fn native_error(error: String) -> Error {
    Error::Unsupported(format!("oneloopmaster native backend: {error}"))
}

impl OneLoopNativeMasters {
    pub fn new() -> Result<Self> {
        Ok(Self {
            a0: NativeEvaluator::new(ScalarIntegral::A0).map_err(native_error)?,
            b0: NativeEvaluator::new(ScalarIntegral::B0).map_err(native_error)?,
            db0: NativeEvaluator::new(ScalarIntegral::DB0).map_err(native_error)?,
            c0: NativeEvaluator::new(ScalarIntegral::C0).map_err(native_error)?,
            d0: NativeEvaluator::new(ScalarIntegral::D0).map_err(native_error)?,
            input: Vec::with_capacity(11),
            output: [Complex::new(0., 0.); 3],
        })
    }

    fn scalar(&mut self, family: ScalarIntegral, args: &[C64]) -> Option<L3> {
        self.input.clear();
        self.input.extend(args.iter().map(|z| z.to_symbolica()));
        // B0 is symmetric in its masses. The generated B0 at commit c9dd82f
        // returns NaN for p² < 0 with (m0², m1²) = (m², 0); the (0, m²)
        // ordering is regular, so evaluate that one.
        if family == ScalarIntegral::B0 && is_zero(args[2]) && !is_zero(args[1]) {
            self.input.swap(1, 2);
        }
        let evaluator = match family {
            ScalarIntegral::A0 => &mut self.a0,
            ScalarIntegral::B0 => &mut self.b0,
            ScalarIntegral::DB0 => &mut self.db0,
            ScalarIntegral::C0 => &mut self.c0,
            ScalarIntegral::D0 => &mut self.d0,
        };
        evaluator.evaluate(&self.input, &mut self.output).ok()?;
        let out = self.output.map(C64::from_symbolica);
        out.iter().all(|z| z.is_finite()).then_some(out)
    }

    /// `[B0, B1, B11, B111]` at `(p², m0², m1², mu²)`.
    pub fn bubble(&mut self, p: C64, m0: C64, m1: C64, mu2: C64) -> Option<[L3; 4]> {
        let b0 = self.scalar(ScalarIntegral::B0, &[p, m0, m1, mu2])?;
        let a0_m0 = self.a0_or_zero(m0, mu2)?;
        let a0_m1 = self.a0_or_zero(m1, mu2)?;
        let out = bubble_tensors(p.re, m0, m1, mu2.re, b0, a0_m0, a0_m1);
        out.iter().flatten().all(|z| z.is_finite()).then_some(out)
    }

    fn a0_or_zero(&mut self, m: C64, mu2: C64) -> Option<L3> {
        self.scalar(ScalarIntegral::A0, &[m, mu2])
    }

    /// Tadpole tensor coefficient returned last by OneLOop `an(rank)`.
    fn an(&mut self, rank: usize, m: C64, mu2: C64) -> Option<L3> {
        let a0 = self.scalar(ScalarIntegral::A0, &[m, mu2])?;
        if rank < 2 || is_zero(m) {
            return Some(if rank < 2 { a0 } else { zero3() });
        }
        let logarithm = physical_log(m / mu2.re);
        if rank < 4 {
            let m2 = m * m;
            Some([m2 * (C64::real(1.5) - logarithm) / 4., m2 / 4., C64::ZERO])
        } else {
            let m3 = m * m * m;
            Some([
                m3 * (C64::real(11. / 6.) - logarithm) / 24.,
                m3 / 24.,
                C64::ZERO,
            ])
        }
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
            MasterIntegral::B0 => self.scalar(ScalarIntegral::B0, args),
            MasterIntegral::DB0 => self.scalar(ScalarIntegral::DB0, args),
            MasterIntegral::C0 => self.scalar(ScalarIntegral::C0, args),
            MasterIntegral::D0 => self.scalar(ScalarIntegral::D0, args),
            MasterIntegral::B1 | MasterIntegral::B11 | MasterIntegral::B111 => {
                let slot = match integral {
                    MasterIntegral::B1 => 1,
                    MasterIntegral::B11 => 2,
                    _ => 3,
                };
                self.bubble(args[0], args[1], args[2], args[3])
                    .map(|b| b[slot])
            }
            MasterIntegral::An { rank } => self.an(rank, args[0], args[1]),
        };
        value.ok_or(MasterFailure::NonFiniteResult)
    }
}

impl FastMasters for OneLoopNativeMasters {
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
                FastMaster::BubbleRank2 | FastMaster::BubbleRank3 => {
                    match self.bubble(point[0], point[1], point[2], point[3]) {
                        Some(value) => {
                            result.copy_from_slice(&value[..width]);
                            true
                        }
                        None => false,
                    }
                }
            };
        }
        Ok(())
    }
}

fn request_rows(request: &MasterRequest<'_, Complex<f64>>) -> usize {
    request.rows()
}

impl MasterProvider<Complex<f64>> for OneLoopNativeMasters {
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
