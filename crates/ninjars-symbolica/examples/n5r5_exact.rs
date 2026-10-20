//! Exact reference for one bench case: per lane, the generic NinjaRS engine
//! in `Float(bits)` with exact (oneloopmaster, same precision) masters, and the
//! errors of original Ninja, the binary64 fast engine with OneLOop / native /
//! correctly rounded masters, and the generic engine in DoubleFloat and f64.
//!
//! ```text
//! cargo run --profile perf -p ninjars-symbolica --example n5r5_exact -- \
//!     [--bits 256] CASE.json NINJA.json [MASTERS.json]
//! ```

#[path = "../tests/common/ninja_reference.rs"]
mod ninja_reference;
#[path = "../tests/common/precision.rs"]
mod precision;

use ninja_reference::{Batch, Case, Error, NinjaMasters, parse_case, parse_ninja, values};
use ninjars::fast::GenericMasters;
use ninjars::fast::{C64, FastAmplitude, FastMasters, FastOptions, FastResult, FastStatus};
use ninjars::{
    AmplitudeInput, Complex, DoubleFloat, EvaluationStatus, Float, FourMomentum, Laurent,
    MasterProvider, MasterRequest, MasterStatus,
};
use ninjars_symbolica::{
    OneLoopNativeMasters, PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts,
    SymbolicaHighCuts, SymbolicaMasters,
};
use oneloop::NativeFloat;
use precision::RecordedMasters;
use serde_json::json;
use symbolica::{
    atom::Atom,
    domains::float::{FloatLike, RealLike},
    parser::ParseSettings,
};

trait Carrier: NativeFloat {
    fn make(proto: &Self, x: f64) -> Self;
}
impl Carrier for Float {
    fn make(proto: &Self, x: f64) -> Self {
        Float::with_val(proto.get_precision(), x)
    }
}
impl Carrier for DoubleFloat {
    fn make(_: &Self, x: f64) -> Self {
        DoubleFloat::from(x)
    }
}
impl Carrier for f64 {
    fn make(_: &Self, x: f64) -> Self {
        x
    }
}

fn parse(text: &str) -> Atom {
    Atom::parse(text, "ninjars_n5r5", ParseSettings::default()).unwrap()
}

fn c<T: Carrier>(p: &T, z: C64) -> Complex<T> {
    Complex::new(T::make(p, z.re), T::make(p, z.im))
}
fn back<T: Carrier>(z: &Complex<T>) -> C64 {
    C64::new(z.re.to_f64(), z.im.to_f64())
}

/// Generic NinjaRS reduction of one lane in carrier `T` with `masters`.
fn generic<T: Carrier>(
    proto: T,
    case: &Case,
    index: usize,
    masters: &mut impl MasterProvider<Complex<T>>,
) -> Result<Vec<Complex<T>>, Error>
where
    Complex<T>: symbolica::evaluate::EvaluationDomain,
{
    let (n, rank) = (case.n, case.rank);
    let point = &case.points[index];
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
    let coefficients: Vec<Complex<T>> = point.tensor.iter().map(|&z| c(&proto, z)).collect();
    numerator.set_parameter_batch(&coefficients, 1)?;
    let shifts: Vec<FourMomentum<Complex<T>>> = point
        .shifts
        .iter()
        .map(|v| FourMomentum::from(v.map(|x| c(&proto, C64::real(x)))))
        .collect();
    let masses: Vec<Complex<T>> = point.masses.iter().map(|&m| c(&proto, m)).collect();
    let matrix: Vec<f64> = point.smat.clone().expect("s matrix");
    let generic_matrix: Vec<Complex<T>> = matrix.iter().map(|&x| c(&proto, C64::real(x))).collect();
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
            scales_squared: &[c(&proto, C64::real(point.mu2))],
            sampling_scales: &[c(&proto, C64::real(sampling))],
            s_matrix: Some(&generic_matrix),
        },
        &mut numerator,
        &mut low,
        &mut high,
        masters,
    )?;
    if amplitude.statuses()[0] != EvaluationStatus::Success {
        return Err(format!("generic engine failed: {:?}", amplitude.statuses()[0]).into());
    }
    Ok(amplitude.results()[0].to_vec())
}

/// Binary64 `MasterProvider` answering with exact `Float(bits)` masters
/// rounded to binary64.
struct RoundedExact {
    inner: SymbolicaMasters<Float>,
    bits: u32,
}

impl MasterProvider<Complex<f64>> for RoundedExact {
    fn evaluate(
        &mut self,
        request: MasterRequest<'_, Complex<f64>>,
        output: &mut [Laurent<Complex<f64>>],
    ) -> ninjars::Result<()> {
        let mut statuses = vec![MasterStatus::Success; output.len()];
        self.evaluate_batch(request, output, &mut statuses)
    }
    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, Complex<f64>>,
        output: &mut [Laurent<Complex<f64>>],
        statuses: &mut [MasterStatus],
    ) -> ninjars::Result<()> {
        let b = self.bits;
        let w = |z: &Complex<f64>| Complex::new(Float::with_val(b, z.re), Float::with_val(b, z.im));
        let inv: Vec<_> = request.invariants.iter().map(w).collect();
        let mas: Vec<_> = request.masses_squared.iter().map(w).collect();
        let sca: Vec<_> = request.scales_squared.iter().map(w).collect();
        let zero = Complex::new(Float::with_val(b, 0), Float::with_val(b, 0));
        let mut out = vec![[zero.clone(), zero.clone(), zero]; output.len()];
        self.inner.evaluate_batch(
            MasterRequest {
                integral: request.integral,
                invariants: &inv,
                masses_squared: &mas,
                scales_squared: &sca,
            },
            &mut out,
            statuses,
        )?;
        for (o, v) in output.iter_mut().zip(&out) {
            *o = v
                .clone()
                .map(|z| Complex::new(z.re.to_f64(), z.im.to_f64()));
        }
        Ok(())
    }
}

/// `SymbolicaMasters<Float>` that logs failing rows.
struct Logged(SymbolicaMasters<Float>);
impl MasterProvider<Complex<Float>> for Logged {
    fn evaluate(
        &mut self,
        request: MasterRequest<'_, Complex<Float>>,
        output: &mut [Laurent<Complex<Float>>],
    ) -> ninjars::Result<()> {
        let mut statuses = vec![MasterStatus::Success; output.len()];
        self.evaluate_batch(request, output, &mut statuses)
    }
    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, Complex<Float>>,
        output: &mut [Laurent<Complex<Float>>],
        statuses: &mut [MasterStatus],
    ) -> ninjars::Result<()> {
        let integral = request.integral;
        let ni = integral.invariant_count();
        let nm = integral.mass_count();
        let inv: Vec<_> = request.invariants.to_vec();
        let mas: Vec<_> = request.masses_squared.to_vec();
        let r = self.0.evaluate_batch(request, output, statuses);
        for (row, st) in statuses.iter().enumerate() {
            let finite = output[row].iter().all(|z| back(z).is_finite());
            if *st != MasterStatus::Success || !finite {
                eprintln!(
                    "FAILED {integral:?} {st:?} out {:?} inv {:?} m2 {:?}",
                    output[row].iter().map(back).collect::<Vec<_>>(),
                    inv[row * ni..(row + 1) * ni]
                        .iter()
                        .map(back)
                        .collect::<Vec<_>>(),
                    mas[row * nm..(row + 1) * nm]
                        .iter()
                        .map(back)
                        .collect::<Vec<_>>()
                );
            }
        }
        r
    }
}

fn fast_lane(
    case: &Case,
    index: usize,
    masters: &mut impl FastMasters,
) -> Result<([C64; 5], FastStatus), Error> {
    let mut fast = FastAmplitude::new(case.n, case.rank, FastOptions::default())?;
    let mut batch = Batch::new(case, &[index]);
    let mut tensor = std::mem::replace(
        &mut batch.tensor,
        ninjars::fast::TensorNumerator::new(case.rank, 0),
    );
    let mut results = [FastResult::default()];
    let mut statuses = [FastStatus::NotEvaluated];
    fast.evaluate(
        &batch.input(),
        &mut tensor,
        masters,
        &mut results,
        &mut statuses,
    )?;
    Ok((values(&results[0]), statuses[0]))
}

fn rel(a: &[C64; 5], b: &[C64; 5], scale: f64) -> Vec<f64> {
    (0..5).map(|k| (a[k] - b[k]).abs() / scale).collect()
}

fn main() -> Result<(), Error> {
    let mut bits = 256u32;
    let mut files = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bits" => bits = args.next().ok_or("--bits")?.parse()?,
            _ => files.push(arg),
        }
    }
    let case = parse_case(&std::fs::read_to_string(&files[0])?)?;
    let masters_file = files.get(2).map(std::fs::read_to_string).transpose()?;
    let ninja = parse_ninja(
        &std::fs::read_to_string(&files[1])?,
        masters_file.as_deref(),
    )?;
    let to5 = |v: &[C64]| -> [C64; 5] { [v[0], v[1], v[2], v[3], v[4]] };

    // Master-level comparison on Ninja's recorded calls.
    let mut native = OneLoopNativeMasters::new()?;
    let mut exact_fast = GenericMasters::new(RoundedExact {
        inner: SymbolicaMasters::new(Float::new(bits)),
        bits,
    });
    for lane in &ninja {
        let mut worst_olo = 0f64;
        let mut worst_native = 0f64;
        let mut worst_family = String::new();
        for call in &lane.calls {
            let width = call.family.outputs();
            let mut e = vec![[C64::ZERO; 3]; width];
            let mut nv = vec![[C64::ZERO; 3]; width];
            let mut ok = [false];
            exact_fast.evaluate(call.family, &call.args, &mut e, &mut ok)?;
            assert!(ok[0]);
            FastMasters::evaluate(&mut native, call.family, &call.args, &mut nv, &mut ok)?;
            assert!(ok[0]);
            for k in 0..width {
                for j in 0..3 {
                    let s = e[k][j].abs().max(1e-300);
                    if e[k][j].abs() == 0. && call.values[k][j].abs() == 0. && nv[k][j].abs() == 0.
                    {
                        continue;
                    }
                    let do_ = (call.values[k][j] - e[k][j]).abs() / s;
                    let dn = (nv[k][j] - e[k][j]).abs() / s;
                    if dn > 1e-12 {
                        eprintln!(
                            "MASTER {} {:?}[{k}][{j}] args {:?} exact {:?} oneloop {:?} native {:?} rel_native {dn:e} rel_oneloop {do_:e}",
                            lane.id, call.family, call.args, e[k][j], call.values[k][j], nv[k][j]
                        );
                    }
                    if do_ > worst_olo {
                        worst_olo = do_;
                    }
                    if dn > worst_native {
                        worst_native = dn;
                        worst_family = format!("{:?}[{k}][{j}]", call.family);
                    }
                }
            }
        }
        println!(
            "{}",
            json!({"lane": lane.id, "calls": lane.calls.len(),
                "max_rel_master_err_oneloop": worst_olo,
                "max_rel_master_err_native": worst_native, "native_worst": worst_family})
        );
    }

    if let Ok(b) = std::env::var("N5R5_PLAIN_BITS") {
        // Reproduce the plain SymbolicaMasters<Float> failure on every lane.
        let b: u32 = b.parse()?;
        for index in 0..case.points.len() {
            let mut m = SymbolicaMasters::new(Float::new(b));
            match generic(Float::new(b), &case, index, &mut m) {
                Ok(r) => eprintln!("PLAIN {b} lane {index}: ok {:?}", back(&r[0])),
                Err(e) => eprintln!("PLAIN {b} lane {index}: {e}"),
            }
            let mut m = Logged(SymbolicaMasters::new(Float::new(b)));
            match generic(Float::new(b), &case, index, &mut m) {
                Ok(r) => eprintln!("LOGGED {b} lane {index}: ok {:?}", back(&r[0])),
                Err(e) => eprintln!("LOGGED {b} lane {index}: {e}"),
            }
        }
        return Ok(());
    }
    for index in 0..case.points.len() {
        let id = case.points[index].id.clone();
        // Exact: Float(bits) and Float(2 bits) with exact masters.
        let exact = {
            let mut m = Logged(SymbolicaMasters::new(Float::new(bits)));
            let r = generic(Float::new(bits), &case, index, &mut m)?;
            to5(&r.iter().map(back).collect::<Vec<_>>())
        };
        let (exact2, exact2_full) = {
            eprintln!("lane {index}: exact {bits} done");
            let mut m = Logged(SymbolicaMasters::new(Float::new(2 * bits)));
            let r = generic(Float::new(2 * bits), &case, index, &mut m)?;
            (to5(&r.iter().map(back).collect::<Vec<_>>()), r)
        };
        // Exact reduction with OneLOop's binary64 masters (as Ninja used).
        let exact_olo = {
            let mut m = RecordedMasters::new(ninja[index].calls.clone(), bits);
            let r = generic(Float::new(bits), &case, index, &mut m)?;
            to5(&r.iter().map(back).collect::<Vec<_>>())
        };
        // Exact reduction with correctly rounded binary64 masters.
        let dd = {
            let mut m = SymbolicaMasters::new(DoubleFloat::from(0.));
            let r = generic(DoubleFloat::from(0.), &case, index, &mut m)?;
            to5(&r.iter().map(back).collect::<Vec<_>>())
        };
        let g64 = {
            let mut m = SymbolicaMasters::new(0f64);
            match generic(0f64, &case, index, &mut m) {
                Ok(r) => Some(to5(&r.iter().map(back).collect::<Vec<_>>())),
                Err(e) => {
                    eprintln!("generic f64 failed: {e}");
                    None
                }
            }
        };
        let (f_olo, s1) = fast_lane(&case, index, &mut NinjaMasters::new(&ninja[index..=index]))?;
        let (f_native, s2) = fast_lane(&case, index, &mut OneLoopNativeMasters::new()?)?;
        let (f_exm, s3) = fast_lane(&case, index, &mut exact_fast)?;
        let nv = ninja[index].values;
        let scale = exact.iter().map(|z| z.abs()).fold(0., f64::max);
        let eps0 = exact[0].abs();
        let report = |name: &str, v: &[C64; 5]| {
            json!({"name": name,
                "value": v.iter().map(|z| [z.re, z.im]).collect::<Vec<_>>(),
                "err_over_scale": rel(v, &exact, scale),
                "err_over_eps0": rel(v, &exact, eps0)})
        };
        let mut rows = vec![
            report("exact_2bits", &exact2),
            report("exact_reduction_oneloop_masters", &exact_olo),
            report("ninja", &nv),
            report("fast_oneloop_masters", &f_olo),
            report("fast_native_masters", &f_native),
            report("fast_rounded_exact_masters", &f_exm),
            report("generic_doublefloat", &dd),
        ];
        if let Some(g) = g64 {
            rows.push(report("generic_f64", &g));
        }
        let ns = nv[..3].iter().map(|z| z.abs()).fold(0., f64::max);
        let native_vs_ninja: Vec<f64> = (0..3).map(|k| (f_native[k] - nv[k]).abs() / ns).collect();
        println!(
            "{}",
            json!({"lane": id, "bits": bits,
                "exact": exact.iter().map(|z| [z.re, z.im]).collect::<Vec<_>>(),
                "exact_2bits_eps0": format!("{} {}", exact2_full[0].re, exact2_full[0].im),
                "scale": scale, "eps0_abs": eps0,
                "fast_statuses": [format!("{s1:?}"), format!("{s2:?}"), format!("{s3:?}")],
                "fast_native_vs_ninja_scaled_by_ninja_laurent": native_vs_ninja,
                "rows": rows})
        );
    }
    Ok(())
}
