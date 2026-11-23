//! Per-lane look at the cancellation ladder (`cancellation_ladder`) on
//! dumped pyAmpliCol lanes (one fixture per JSON line, the
//! `ninjars-box-mu-conditioning-v1` shape with a single `tensor`).
//!
//! ```text
//! cargo run --release -p ninjars-symbolica --features native-masters \
//!     --example ladder_diag -- [--base dd|f64|BITS] [--ref BITS] LANES.jsonl
//! ```
//!
//! Each lane is first made consistent at `--ref` bits (shells projected,
//! other invariants from the projected shifts), its reference computed there,
//! and its inputs rounded to the base carrier (as a caller's cure in that
//! carrier gives them). Then the base carrier's generic reduction (with the
//! stability test) and the ladder's rescue. One JSON line per lane: base
//! status, measured loss `L`, base and rescued relative errors, rungs, and
//! the cut sizes carrying the largest term moduli.

#[path = "../tests/common/conditioning.rs"]
mod conditioning;

use conditioning::{Error, parse_fixture};
use ninjars::fast::tensor_layout::{monomials, ncoeff};
use ninjars::{
    AmplitudeInput, Complex, DoubleFloat, Float, FourMomentum, SamplingScaleProvider,
    SamplingScaleSource, SamplingScaleStatus,
};
use ninjars_symbolica::cancellation_ladder::{
    LadderCarrier, LadderRule, PolynomialLadder, lane_loss, project_shells,
};
use ninjars_symbolica::{
    PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaHighCuts,
    SymbolicaMasters, SymbolicaSamplingScales,
};
use serde_json::json;
use symbolica::{
    atom::Atom,
    domains::float::{Real, RealLike, Scalar},
    evaluate::EvaluationDomain,
    parser::ParseSettings,
};

fn atom(text: &str) -> Atom {
    Atom::parse(text, "ninjars_ladder_diag", ParseSettings::default()).unwrap()
}

fn template(rank: usize) -> (Atom, Vec<Atom>) {
    let names = ["q0", "q1", "q2", "q3"];
    let terms: Vec<String> = monomials(rank)
        .iter()
        .enumerate()
        .map(|(k, alpha)| {
            let mut factors = vec![format!("c{k}")];
            for (variable, &power) in names.iter().zip(alpha) {
                if power > 0 {
                    factors.push(format!("{variable}^{power}"));
                }
            }
            factors.join("*")
        })
        .collect();
    let parameters = (0..ncoeff(rank)).map(|k| atom(&format!("c{k}"))).collect();
    (atom(&terms.join("+")), parameters)
}

struct Lane {
    n: usize,
    rank: usize,
    shifts: Vec<[Float; 4]>,
    masses: Vec<Complex<Float>>,
    s: Vec<Complex<Float>>,
    scale: Complex<Float>,
    coefficients: Vec<Complex<Float>>,
}

fn consistent(f: &conditioning::Fixture, bits: u32) -> Lane {
    let n = f.n;
    let p = |t: &str| Float::parse(t, Some(bits)).unwrap();
    let zero = Float::with_val(bits, 0);
    let c = |z: &[String; 2]| Complex::new(p(&z[0]), p(&z[1]));
    let mut shifts: Vec<[Float; 4]> = f.shifts.iter().map(|v| v.clone().map(|x| p(&x))).collect();
    let masses: Vec<Complex<Float>> = f.masses_squared.iter().map(c).collect();
    let given: Vec<Float> = f.s_matrix.iter().map(|x| p(x)).collect();
    let mut keep = vec![false; n * n];
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            let s = &given[i * n + j];
            if s.is_zero() || masses.iter().any(|m| m.im.is_zero() && !m.re.is_zero() && &m.re == s) {
                keep[i * n + j] = true;
                keep[j * n + i] = true;
                pairs.push((i, j, s.clone()));
            }
        }
    }
    assert!(project_shells(&mut shifts, &pairs, bits));
    let s = (0..n * n)
        .map(|k| {
            let (i, j) = (k / n, k % n);
            let v = if i == j {
                zero.clone()
            } else if keep[k] {
                given[k].clone()
            } else {
                let mut s = zero.clone();
                for mu in 0..4 {
                    let d = shifts[i][mu].clone() - &shifts[j][mu];
                    let t = d.clone() * &d;
                    s = if mu == 0 { s + &t } else { s - &t };
                }
                s
            };
            Complex::new(v, zero.clone())
        })
        .collect();
    Lane {
        n,
        rank: f.rank,
        shifts,
        masses,
        s,
        scale: Complex::new(p(&f.mu_squared), zero.clone()),
        coefficients: f.tensors[0].iter().map(c).collect(),
    }
}

/// Exact zeros stay zero; everything else rounded to the carrier.
fn round<T: LadderCarrier>(x: &Float, prototype: &T) -> T
where
    Complex<T>: Real + EvaluationDomain,
{
    T::from_float(x, prototype)
}

#[allow(clippy::type_complexity)]
fn run<T: LadderCarrier>(
    lane: &Lane,
    prototype: T,
    ladder: bool,
    target: Option<f64>,
) -> Result<(String, Complex<f64>, f64, Vec<(usize, f64)>, Option<(Complex<f64>, Vec<u32>, Option<f64>, bool, String)>), Error>
where
    Complex<T>: Real + EvaluationDomain,
{
    let n = lane.n;
    let zero = prototype.zero();
    let cr = |x: &Float| Complex::new(round(x, &prototype), zero.clone());
    let cc = |z: &Complex<Float>| Complex::new(round(&z.re, &prototype), round(&z.im, &prototype));
    let shifts: Vec<FourMomentum<Complex<T>>> =
        lane.shifts.iter().map(|v| FourMomentum::from(v.clone().map(|x| cr(&x)))).collect();
    let masses: Vec<Complex<T>> = lane.masses.iter().map(cc).collect();
    let s_matrix: Vec<Complex<T>> = lane.s.iter().map(cc).collect();
    let scales = vec![cc(&lane.scale)];
    let coefficients: Vec<Complex<T>> = lane.coefficients.iter().map(cc).collect();
    let (expression, parameters) = template(lane.rank);
    let variables = ["q0", "q1", "q2", "q3", "mu2"].map(atom);
    let mut numerator = PolynomialNumerator::new(expression, variables, parameters, prototype.clone())?;
    numerator.set_parameter_batch(&coefficients, 1)?;
    let mut sampling = vec![Complex::new(zero.clone(), zero.clone()); 1];
    if n >= 4 {
        let mut status = vec![SamplingScaleStatus::Selected(SamplingScaleSource::UnitFallback); 1];
        SymbolicaSamplingScales::new(n, prototype.clone())?.evaluate_batch(&shifts, &masses, 1, &mut sampling, &mut status)?;
    }
    let mut low = SymbolicaCuts::new(prototype.clone())?;
    let mut masters = SymbolicaMasters::new(prototype.clone()).with_onshell_threshold(None);
    let mut amplitude = SymbolicaAmplitudeKernels::new(n, lane.rank, prototype.clone())?
        .with_stability_test()
        .into_amplitude(1)?;
    amplitude.reset_batch(1);
    let input = AmplitudeInput {
        shifts: &shifts,
        masses_squared: &masses,
        scales_squared: &scales,
        sampling_scales: &sampling,
        s_matrix: Some(&s_matrix),
    };
    if n >= 4 {
        let mut high = SymbolicaHighCuts::new(prototype.clone())?;
        amplitude.evaluate_batch_with_inputs(
            AmplitudeInput { ..input },
            &mut numerator,
            &mut low,
            &mut high,
            &mut masters,
        )?;
    } else {
        amplitude.evaluate_batch_with_low_cuts(AmplitudeInput { ..input }, &mut numerator, &mut low, &mut masters)?;
    }
    let status = format!("{:?}", amplitude.statuses()[0]);
    let (lost, _, sum) = lane_loss(&amplitude, 0).ok_or("nothing measured")?;
    let value = Complex::new(sum[0].re.to_f64(), sum[0].im.to_f64());
    // Largest term moduli per cut size.
    let cuts = amplitude.topology().cuts();
    let mut by_size = vec![0f64; 6];
    for (k, cut) in cuts.iter().enumerate() {
        let m = amplitude.term_moduli()[k][0].re.to_f64();
        let size = cut.partition.len().min(5);
        by_size[size] = by_size[size].max(m);
    }
    let by_size = by_size.into_iter().enumerate().filter(|(_, m)| *m > 0.).collect();
    let rescued = if ladder {
        let mut ladder = PolynomialLadder::new(n, &numerator).with_rule(LadderRule { target, ..LadderRule::default() });
        let input = AmplitudeInput {
            shifts: &shifts,
            masses_squared: &masses,
            scales_squared: &scales,
            sampling_scales: &sampling,
            s_matrix: Some(&s_matrix),
        };
        let out = ladder.rescue(&amplitude, &input, &coefficients, None)?;
        out.into_iter().next().map(|r| {
            let v = r
                .value
                .as_ref()
                .map(|v| Complex::new(v[0].re.to_f64(), v[0].im.to_f64()))
                .unwrap_or(Complex::new(f64::NAN, f64::NAN));
            (v, r.rungs, r.rung_estimate, r.projected, format!("{:?}", r.outcome))
        })
    } else {
        None
    };
    Ok((status, value, lost, by_size, rescued))
}

fn rel(a: Complex<f64>, b: Complex<f64>) -> f64 {
    let d = Complex::new(a.re - b.re, a.im - b.im);
    (d.re.hypot(d.im)) / b.re.hypot(b.im)
}

fn main() -> Result<(), Error> {
    let mut base = "dd".to_string();
    let mut reference = 512u32;
    let mut target: Option<f64> = None;
    let mut files = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--base" => base = args.next().ok_or("--base")?,
            "--ref" => reference = args.next().ok_or("--ref")?.parse()?,
            "--target" => target = Some(args.next().ok_or("--target")?.parse()?),
            _ => files.push(arg),
        }
    }
    for file in files {
        for line in std::fs::read_to_string(&file)?.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let fixture = parse_fixture(line)?;
            let lane = consistent(&fixture, reference);
            let started = std::time::Instant::now();
            let (_, truth, lref, _, _) = run(&lane, Float::with_val(reference, 0), false, None)?;
            let tref = started.elapsed().as_secs_f64();
            let started = std::time::Instant::now();
            let (status, value, lost, sizes, rescued) = match base.as_str() {
                "dd" => run(&lane, DoubleFloat::from(0.0), true, target)?,
                "f64" => run(&lane, 0f64, true, target)?,
                bits => run(&lane, Float::with_val(bits.parse()?, 0), true, target)?,
            };
            let t = started.elapsed().as_secs_f64();
            let out = json!({
                "name": fixture.name, "file": file, "n": lane.n, "rank": lane.rank,
                "truth": [truth.re, truth.im], "ref_lost_bits": lref, "ref_seconds": tref,
                "status": status, "lost_bits": lost, "base_rel_err": rel(value, truth),
                "base_and_ladder_seconds": t,
                "max_term_moduli_by_cut_size": sizes,
                "ladder": rescued.map(|(v, rungs, e2, proj, outcome)| json!({"rel_err": rel(v, truth), "rungs": rungs, "rung_estimate": e2, "projected": proj, "outcome": outcome})),
            });
            println!("{out}");
        }
    }
    Ok(())
}
