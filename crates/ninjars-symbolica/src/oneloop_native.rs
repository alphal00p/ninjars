//! Binary64 masters for the fast path: oneloopmaster's ahead-of-time
//! generated Rust (`NativeEvaluator<f64>`, feature `generated-evaluators`) for
//! A0, C0, D0 (and dB0), and a port of OneLOop's bubble routines for B0, B1,
//! B11, B111. No Symbolica expression is interpreted or JIT-compiled at
//! evaluation time.
//!
//! The bubbles follow OneLOop's `bub0`/`bub1`/`bub11`/`bub111` (avh_olo 3.6):
//! every coefficient is computed directly from the roots of the Feynman-
//! parameter quadratic through `bnlog`, never through Passarino-Veltman
//! relations, whose divisions by `p²` lose about `(m²/p²)^rank` relative
//! accuracy (B11 off by 3e3, B111 by 9e9 at `p² = 1e-2`, `m² = 173²`).
//! OneLOop's on-shell threshold is the one original Ninja sets (`1e-9`).

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
/// (oneloopmaster's `physical_log`, OneLOop's `logc(qonv(z, -1))`).
fn physical_log(z: C64) -> C64 {
    if z.im != 0. {
        C64::new(z.abs().ln(), z.im.atan2(z.re))
    } else if z.re < 0. {
        C64::new((-z.re).ln(), -std::f64::consts::PI)
    } else {
        C64::real(z.re.ln())
    }
}

/// OneLOop's `EPSN` (binary64 machine epsilon) and infinitesimal `IEPS = EPSN²`
/// (the imaginary part carrying the `i0` prescription of real masses).
const EPSN: f64 = f64::EPSILON;
const IEPS: f64 = EPSN * EPSN;
/// OneLOop's on-shell threshold as configured by original Ninja (`INFRARED_EPS`):
/// smaller `|p²|` and squared-mass taxicab norms count as zero.
const ONSHELL: f64 = 1e-9;
/// Beyond this modulus `bnlog` is summed as a series in `1/x`.
const BNLOG_SERIES: f64 = 3.;

/// OneLOop's `bnlog(r, x) = (1 - x^(r+1)) log(1 - 1/x) - sum_{p=0}^{r}
/// x^p/(r+1-p)` for `r = 0..=3` (`bnlog_c`/`bnlog_r`). A real `x`
/// (`|Im x| <= EPSN |Re x|`) carries the sign of its infinitesimal imaginary
/// part onto the logarithm's branch. For `|x| >= 3` the cancellation between
/// the logarithm and the polynomial is resolved analytically:
/// `bnlog(r, x) = -(r+1) sum_{j>=1} x^(-j) / (j (j+r+1))`.
fn bnlogs(x: C64) -> [C64; 4] {
    let real = x.im.abs() <= EPSN * x.re.abs();
    let xr = if real { C64::real(x.re) } else { x };
    if real && x.re == 0. {
        return [C64::ZERO; 4];
    }
    if (xr - C64::ONE).abs() <= EPSN * 10. {
        // -(1 + 1/2 + ... + 1/(r+1))
        return [-1., -1.5, -11. / 6., -25. / 12.].map(C64::real);
    }
    let mut out = [C64::ZERO; 4];
    if xr.abs() >= BNLOG_SERIES {
        let inverse = xr.inv();
        let mut power = inverse;
        for j in 1..=64 {
            let jf = j as f64;
            let mut largest = 0f64;
            for (r, sum) in out.iter_mut().enumerate() {
                let term = power * (1. / (jf * (jf + (r + 1) as f64)));
                *sum += term;
                largest = largest.max(term.abs() / sum.abs());
            }
            if largest <= 1e-18 {
                break;
            }
            power *= inverse;
        }
        for (r, sum) in out.iter_mut().enumerate() {
            *sum *= -((r + 1) as f64);
        }
        return out;
    }
    let log = if real {
        let y = 1. - 1. / x.re;
        if y < 0. {
            let sign = if x.im < 0. { -1. } else { 1. };
            C64::new((-y).ln(), sign * std::f64::consts::PI)
        } else {
            C64::real(y.ln())
        }
    } else {
        let y = C64::ONE - xr.inv();
        C64::new(y.abs().ln(), y.im.atan2(y.re))
    };
    // 1 + x + ... + x^r and sum_p x^p/(r+1-p), by Horner in r.
    let mut geometric = C64::ONE;
    let mut polynomial = C64::ONE;
    for (r, value) in out.iter_mut().enumerate() {
        if r > 0 {
            geometric = C64::ONE + xr * geometric;
            polynomial = polynomial * xr + C64::real(1. / (r + 1) as f64);
        }
        *value = (C64::ONE - xr) * geometric * log - polynomial;
    }
    out
}

/// Roots of `a x² + b x + c = 0` (OneLOop's `solabc`, `imode = 0`), `a, c != 0`.
fn solabc(a: C64, b: C64, c: C64) -> (C64, C64) {
    let d = (b * b - a * c * 4.).sqrt();
    let q = -b + d;
    let h = -b - d;
    if q.abs() >= h.abs() {
        (q / (a * 2.), c * 2. / q)
    } else {
        (c * 2. / h, h / (a * 2.))
    }
}

/// `[B0, B1, B11, B111]` (finite part, 1/eps, 1/eps²) at `(p², m0², m1², mu²)`:
/// OneLOop's `bub111` (OneLOop/Ninja tensor-bubble conventions).
fn bubble_tensors(p: f64, m0i: C64, m1i: C64, mu2: f64) -> [L3; 4] {
    let taxicab = |m: C64| {
        let a = m.re.abs() + m.im.abs();
        if a < ONSHELL { 0. } else { a }
    };
    let app = if p.abs() < ONSHELL { 0. } else { p.abs() };
    let (am0i, am1i) = (taxicab(m0i), taxicab(m1i));
    let maxm = am0i.max(am1i);
    if maxm == 0. && app == 0. {
        return [zero3(); 4];
    }
    // Order the masses so that |m1| is the larger one.
    let (m0, am0, m1, am1, switch) = if am1i >= maxm {
        (m0i, am0i, m1i, am1i, false)
    } else {
        (m1i, am1i, m0i, am0i, true)
    };
    let pp = C64::real(p);
    // b1 -> -b0 - b1 etc.: the coefficients with the masses exchanged.
    let reflect = |b: [C64; 4]| {
        [
            b[0],
            -b[0] - b[1],
            b[2] + b[1] * 2. + b[0],
            -b[3] - b[2] * 3. - b[1] * 3. - b[0],
        ]
    };
    let ranks = |f: &dyn Fn(usize) -> C64| [f(0), -f(1), f(2), -f(3)];
    let b = if app == 0. {
        let lna = -physical_log(m1 / mu2);
        let b = if (m0 - m1).abs() <= am1 * EPSN * 10. {
            ranks(&|r| lna / (r + 1) as f64)
        } else {
            let l1 = bnlogs((m1 - C64::new(0., am1 * IEPS)) / (m1 - m0));
            ranks(&|r| (lna - l1[r]) / (r + 1) as f64)
        };
        if switch { b } else { reflect(b) }
    } else if am0 == 0. {
        let b = if (pp - m1).abs() <= am1 * EPSN * 10. {
            let lna = -physical_log(m1 / mu2);
            ranks(&|r| (lna * (r + 1) as f64 + 2.) / ((r + 1) * (r + 1)) as f64)
        } else {
            let lna = -physical_log((m1 - pp) / mu2);
            let l1 = bnlogs((pp - m1 + C64::new(0., am1 * IEPS)) / pp);
            ranks(&|r| ((lna - l1[r]) * (r + 1) as f64 + 1.) / ((r + 1) * (r + 1)) as f64)
        };
        if switch { reflect(b) } else { b }
    } else {
        let lna = -physical_log(m0 / mu2);
        let (x1, x2) = solabc(pp, m1 - m0 - pp, m0 - C64::new(0., am0 * IEPS));
        let (l1, l2) = (bnlogs(x1), bnlogs(x2));
        let b = ranks(&|r| (lna - l1[r] - l2[r]) / (r + 1) as f64);
        if switch { reflect(b) } else { b }
    };
    [
        [b[0], C64::ONE, C64::ZERO],
        [b[1], C64::real(-0.5), C64::ZERO],
        [b[2], C64::real(1. / 3.), C64::ZERO],
        [b[3], C64::real(-0.25), C64::ZERO],
    ]
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

    /// `[B0, B1, B11, B111]` at `(p², m0², m1², mu²)` (OneLOop's `bub111`).
    /// Fails for non-finite inputs, a complex `p²`, masses with positive
    /// imaginary parts or a non-positive scale.
    pub fn bubble(&mut self, p: C64, m0: C64, m1: C64, mu2: C64) -> Option<[L3; 4]> {
        let valid = [p, m0, m1, mu2].iter().all(|z| z.is_finite())
            && p.im == 0.
            && m0.im <= 0.
            && m1.im <= 0.
            && mu2.im == 0.
            && mu2.re > 0.;
        if !valid {
            return None;
        }
        let out = bubble_tensors(p.re, m0, m1, mu2.re);
        out.iter().flatten().all(|z| z.is_finite()).then_some(out)
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
            MasterIntegral::DB0 => self.scalar(ScalarIntegral::DB0, args),
            MasterIntegral::C0 => self.scalar(ScalarIntegral::C0, args),
            MasterIntegral::D0 => self.scalar(ScalarIntegral::D0, args),
            MasterIntegral::B0
            | MasterIntegral::B1
            | MasterIntegral::B11
            | MasterIntegral::B111 => {
                let slot = match integral {
                    MasterIntegral::B0 => 0,
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
