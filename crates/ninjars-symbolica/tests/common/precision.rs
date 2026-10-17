//! Arbitrary-precision reference for the binary64 conditioning of the fast
//! engine and of original Ninja.
//!
//! The generic NinjaRS engine reduces a case's tensor numerator in
//! `Float(bits)` arithmetic and contracts it with exactly the OneLOop master
//! values original Ninja used (recorded by `validation/benchmark/ninja_bench.cc`,
//! widened to `Float`). Its result is the exact reduction of the binary64
//! inputs with those masters, so the binary64 errors of the fast engine (same
//! masters, see [`super::ninja_reference`]) and of original Ninja can be
//! measured against it. Shared by `tests/fast_conditioning.rs` and
//! `examples/fast_precision.rs`.

#![allow(dead_code)]

use super::ninja_reference::{
    Batch, Case, Error, MasterCall, NinjaLane, NinjaMasters, Point, values,
};
use ninjars::fast::{C64, FastAmplitude, FastMaster, FastOptions, FastResult, FastStatus};
use ninjars::{
    AmplitudeInput, Complex, EvaluationStatus, Float, FourMomentum, Laurent, MasterIntegral,
    MasterProvider, MasterRequest, MasterStatus,
};
use ninjars_symbolica::{
    PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaHighCuts,
};
use symbolica::{atom::Atom, parser::ParseSettings};

type CF = Complex<Float>;

fn parse(text: &str) -> Atom {
    Atom::parse(text, "ninjars_fast_precision", ParseSettings::default()).unwrap()
}

/// Recorded OneLOop values for the generic engine's master requests. Box and
/// triangle requests are matched up to the integrals' relabelling symmetries;
/// scaleless integrals (never requested by original Ninja) vanish.
pub struct RecordedMasters {
    calls: Vec<MasterCall>,
    bits: u32,
}

fn close(a: C64, b: C64) -> bool {
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1e-300)
}

impl RecordedMasters {
    pub fn new(calls: Vec<MasterCall>, bits: u32) -> Self {
        Self { calls, bits }
    }

    fn find(&self, family: FastMaster, args: &[C64]) -> Option<&MasterCall> {
        self.calls
            .iter()
            .find(|c| c.family == family && c.args.iter().zip(args).all(|(&x, &y)| close(x, y)))
    }

    fn lookup(&self, integral: MasterIntegral, args: &[C64]) -> Option<[C64; 3]> {
        let mu2 = *args.last()?;
        let scaleless = match integral {
            MasterIntegral::A0 | MasterIntegral::An { .. } => args[0].is_zero(),
            MasterIntegral::B0
            | MasterIntegral::B1
            | MasterIntegral::B11
            | MasterIntegral::B111 => args[..3].iter().all(|z| z.is_zero()),
            _ => false,
        };
        if scaleless {
            return Some([C64::ZERO; 3]);
        }
        let with = |mut a: Vec<C64>, m: Vec<C64>| {
            a.extend(m);
            a.push(mu2);
            a
        };
        match integral {
            MasterIntegral::D0 => {
                // Cyclic shifts and the reflection of the box.
                let (mut p, mut m) = (args[..6].to_vec(), args[6..10].to_vec());
                let mut candidates = Vec::new();
                for _ in 0..4 {
                    candidates.push(with(p.clone(), m.clone()));
                    candidates.push(with(
                        vec![p[3], p[2], p[1], p[0], p[4], p[5]],
                        vec![m[0], m[3], m[2], m[1]],
                    ));
                    p = vec![p[1], p[2], p[3], p[0], p[5], p[4]];
                    m = vec![m[1], m[2], m[3], m[0]];
                }
                candidates
                    .iter()
                    .find_map(|a| self.find(FastMaster::D0, a).map(|c| c.values[0]))
            }
            MasterIntegral::C0 => {
                let (mut p, mut m) = (args[..3].to_vec(), args[3..6].to_vec());
                let mut candidates = Vec::new();
                for _ in 0..3 {
                    candidates.push(with(p.clone(), m.clone()));
                    candidates.push(with(vec![p[2], p[1], p[0]], vec![m[0], m[2], m[1]]));
                    p = vec![p[1], p[2], p[0]];
                    m = vec![m[1], m[2], m[0]];
                }
                candidates
                    .iter()
                    .find_map(|a| self.find(FastMaster::C0, a).map(|c| c.values[0]))
            }
            MasterIntegral::B0
            | MasterIntegral::B1
            | MasterIntegral::B11
            | MasterIntegral::B111 => {
                let index = match integral {
                    MasterIntegral::B0 => 0,
                    MasterIntegral::B1 => 1,
                    MasterIntegral::B11 => 2,
                    _ => 3,
                };
                let direct = [args[0], args[1], args[2], mu2];
                let swapped = [args[0], args[2], args[1], mu2];
                for family in [FastMaster::BubbleRank3, FastMaster::BubbleRank2] {
                    if index >= family.outputs() {
                        continue;
                    }
                    if let Some(c) = self.find(family, &direct) {
                        return Some(c.values[index]);
                    }
                    // B0 is symmetric in the masses.
                    if index == 0
                        && let Some(c) = self.find(family, &swapped)
                    {
                        return Some(c.values[0]);
                    }
                }
                None
            }
            MasterIntegral::A0 | MasterIntegral::An { rank: 0 | 1 } => self
                .find(FastMaster::A0, &[args[0], mu2])
                .map(|c| c.values[0]),
            _ => None,
        }
    }
}

fn to_c64(z: &CF) -> C64 {
    let f = z.to_f64();
    C64::new(f.re, f.im)
}

impl MasterProvider<CF> for RecordedMasters {
    fn evaluate(
        &mut self,
        request: MasterRequest<'_, CF>,
        output: &mut [Laurent<CF>],
    ) -> ninjars::Result<()> {
        let mut statuses = vec![MasterStatus::Success; request.rows()];
        self.evaluate_batch(request, output, &mut statuses)
    }

    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, CF>,
        output: &mut [Laurent<CF>],
        statuses: &mut [MasterStatus],
    ) -> ninjars::Result<()> {
        let ni = request.integral.invariant_count();
        let nm = request.integral.mass_count();
        for row in 0..request.rows() {
            let mut args: Vec<C64> = request.invariants[row * ni..(row + 1) * ni]
                .iter()
                .chain(&request.masses_squared[row * nm..(row + 1) * nm])
                .map(to_c64)
                .collect();
            args.push(to_c64(&request.scales_squared[row]));
            let value = self.lookup(request.integral, &args).ok_or_else(|| {
                ninjars::Error::Unsupported(format!(
                    "no recorded master for {:?} {args:?}",
                    request.integral
                ))
            })?;
            let bits = self.bits;
            output[row] = value
                .map(|z| Complex::new(Float::with_val(bits, z.re), Float::with_val(bits, z.im)));
            statuses[row] = MasterStatus::Success;
        }
        Ok(())
    }
}

/// Errors of one lane against the exact reduction, relative to the largest
/// modulus among its exact Laurent coefficients and split.
#[derive(Clone, Debug)]
pub struct LanePrecision {
    pub id: String,
    /// Exact `[eps0, eps_m1, eps_m2, cut_constructible, rational]`, rounded.
    pub exact: [C64; 5],
    pub scale: f64,
    pub fast_status: FastStatus,
    pub fast_laurent: f64,
    pub ninja_laurent: f64,
    /// Split errors: the generic engine attributes some rank >= n bubble
    /// terms to the other part (identical totals), so these compare the
    /// fast/Ninja split only with each other's.
    pub fast_split: f64,
    pub ninja_split: f64,
}

/// Exact reductions (generic engine, `Float(bits)`) of the selected lanes of
/// `case` with original Ninja's master values, and the binary64 errors of the
/// fast engine and of original Ninja.
pub fn precision(
    case: &Case,
    ninja: &[NinjaLane],
    bits: u32,
    only: Option<&[usize]>,
) -> Result<Vec<LanePrecision>, Error> {
    let (n, rank) = (case.n, case.rank);
    let proto = Float::new(bits);
    let cf = |z: C64| Complex::new(Float::with_val(bits, z.re), Float::with_val(bits, z.im));
    // The tensor numerator as a polynomial with one parameter per monomial.
    let monomials = ninjars::fast::tensor_layout::monomials(rank);
    let names = ["q0", "q1", "q2", "q3"];
    let mut text = String::new();
    for (k, m) in monomials.iter().enumerate() {
        if k > 0 {
            text.push('+');
        }
        text.push_str(&format!("p{k}"));
        for (mu, &e) in m.iter().enumerate() {
            if e > 0 {
                text.push_str(&format!("*{}^{e}", names[mu]));
            }
        }
    }
    let parameters: Vec<Atom> = (0..monomials.len())
        .map(|k| parse(&format!("p{k}")))
        .collect();
    let variables = ["q0", "q1", "q2", "q3", "mu2"].map(parse);
    let mut numerator =
        PolynomialNumerator::new(parse(&text), variables, parameters, proto.clone())?;
    let mut low = SymbolicaCuts::new(proto.clone())?;
    let mut high = SymbolicaHighCuts::new(proto.clone())?;
    let mut amplitude =
        SymbolicaAmplitudeKernels::new(n, rank, proto.clone())?.into_amplitude(1)?;
    let mut fast = FastAmplitude::new(n, rank, FastOptions::default())?;
    let mut out = Vec::new();
    for (index, point) in case.points.iter().enumerate() {
        if only.is_some_and(|o| !o.contains(&index)) {
            continue;
        }
        let mut masters = RecordedMasters::new(ninja[index].calls.clone(), bits);
        let coefficients: Vec<CF> = point.tensor.iter().map(|&z| cf(z)).collect();
        numerator.set_parameter_batch(&coefficients, 1)?;
        let shifts: Vec<FourMomentum<CF>> = point
            .shifts
            .iter()
            .map(|v| FourMomentum::from(v.map(|x| cf(C64::real(x)))))
            .collect();
        let masses: Vec<CF> = point.masses.iter().map(|&m| cf(m)).collect();
        let matrix: Vec<f64> = match &point.smat {
            Some(s) => s.clone(),
            None => (0..n * n)
                .map(|ij| {
                    let (a, b) = (point.shifts[ij / n], point.shifts[ij % n]);
                    let d: Vec<f64> = (0..4).map(|k| a[k] - b[k]).collect();
                    d[0] * d[0] - d[1] * d[1] - d[2] * d[2] - d[3] * d[3]
                })
                .collect(),
        };
        let generic_matrix: Vec<CF> = matrix.iter().map(|&x| cf(C64::real(x))).collect();
        let sampling = matrix
            .iter()
            .map(|x| x.abs())
            .chain(point.masses.iter().map(|z| z.re.abs()))
            .fold(1., f64::max);
        amplitude.reset_batch(1);
        amplitude.evaluate_batch_with_inputs(
            AmplitudeInput {
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: &[cf(C64::real(point.mu2))],
                sampling_scales: &[cf(C64::real(sampling))],
                s_matrix: Some(&generic_matrix),
            },
            &mut numerator,
            &mut low,
            &mut high,
            &mut masters,
        )?;
        if amplitude.statuses()[0] != EvaluationStatus::Success {
            return Err(format!("generic engine failed: {:?}", amplitude.statuses()[0]).into());
        }
        let mut exact = [C64::ZERO; 5];
        for (e, z) in exact.iter_mut().zip(amplitude.results()[0].iter()) {
            *e = to_c64(z);
        }
        // The fast engine with the same masters at width 1.
        let single = Case {
            name: case.name.clone(),
            n,
            rank,
            points: vec![Point {
                id: point.id.clone(),
                shifts: point.shifts.clone(),
                masses: point.masses.clone(),
                mu2: point.mu2,
                smat: point.smat.clone(),
                tensor: point.tensor.clone(),
            }],
        };
        let mut batch = Batch::new(&single, &[0]);
        let mut tensor = std::mem::replace(
            &mut batch.tensor,
            ninjars::fast::TensorNumerator::new(rank, 0),
        );
        let mut results = [FastResult::default()];
        let mut statuses = [FastStatus::NotEvaluated];
        let mut fast_masters = NinjaMasters::new(&ninja[index..=index]);
        fast.evaluate(
            &batch.input(),
            &mut tensor,
            &mut fast_masters,
            &mut results,
            &mut statuses,
        )?;
        let fast_values = values(&results[0]);
        let ninja_values = ninja[index].values;
        let scale = exact.iter().map(|z| z.abs()).fold(1e-300, f64::max);
        let error = |v: &[C64; 5], range: std::ops::Range<usize>| {
            range
                .map(|k| (v[k] - exact[k]).abs() / scale)
                .fold(0., f64::max)
        };
        out.push(LanePrecision {
            id: point.id.clone(),
            exact,
            scale,
            fast_status: statuses[0],
            fast_laurent: error(&fast_values, 0..3),
            ninja_laurent: error(&ninja_values, 0..3),
            fast_split: error(&fast_values, 3..5),
            ninja_split: error(&ninja_values, 3..5),
        });
    }
    Ok(out)
}
