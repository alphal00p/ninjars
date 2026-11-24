//! Shared by the cancellation ladder's test and its `ladder_diag` example:
//! a dumped lane made consistent in MPFR, and the generic engine with the
//! ladder on it in any carrier.

#![allow(dead_code)]

use super::conditioning::{self, Error};
use ninjars::fast::tensor_layout::{monomials, ncoeff};
use ninjars::{
    AmplitudeInput, Complex, Float, FourMomentum, SamplingScaleProvider, SamplingScaleSource,
    SamplingScaleStatus,
};
use ninjars_symbolica::cancellation_ladder::{
    LadderCarrier, LadderOutcome, LadderRule, PolynomialLadder, lane_loss, project_shells,
};
use ninjars_symbolica::{
    PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaHighCuts,
    SymbolicaMasters, SymbolicaSamplingScales,
};
use symbolica::{
    atom::Atom,
    domains::float::{Real, Scalar},
    evaluate::EvaluationDomain,
    parser::ParseSettings,
};

pub fn atom(text: &str) -> Atom {
    Atom::parse(text, "ninjars_ladder_diag", ParseSettings::default()).unwrap()
}

pub fn template(rank: usize) -> (Atom, Vec<Atom>) {
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

pub struct Lane {
    pub n: usize,
    pub rank: usize,
    pub shifts: Vec<[Float; 4]>,
    pub masses: Vec<Complex<Float>>,
    pub s: Vec<Complex<Float>>,
    pub scale: Complex<Float>,
    pub coefficients: Vec<Complex<Float>>,
}

pub fn consistent(f: &conditioning::Fixture, bits: u32) -> Lane {
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
pub fn round<T: LadderCarrier>(x: &Float, prototype: &T) -> T
where
    Complex<T>: Real + EvaluationDomain,
{
    T::from_float(x, prototype)
}

/// The ladder's result on one lane.
pub struct Rescued {
    pub value: Complex<f64>,
    /// Every served Laurent coefficient (finite, single, double pole).
    pub laurent: Option<[Complex<f64>; 3]>,
    pub rungs: Vec<u32>,
    pub estimate: Option<f64>,
    pub projected: bool,
    pub outcome: LadderOutcome,
    /// The best value's finite part, its estimate and carrier bits.
    pub best: Option<(Complex<f64>, f64, u32)>,
    /// The last rung's absolute bound per coefficient and term moduli.
    pub absolute_bound: Option<[f64; 3]>,
    pub term_moduli: Option<[f64; 3]>,
}

/// One lane through the generic engine (stability test on) in a carrier.
pub struct Run {
    pub status: String,
    /// The base pass's finite part (summed cut contributions).
    pub value: Complex<f64>,
    /// The base pass's Laurent coefficients.
    pub laurent: [Complex<f64>; 3],
    pub lost_bits: f64,
    /// Largest finite-part term moduli per cut size.
    pub by_size: Vec<(usize, f64)>,
    /// `None`: the ladder was not run or did not flag the lane.
    pub ladder: Option<Rescued>,
}

pub fn run<T: LadderCarrier>(
    lane: &Lane,
    prototype: T,
    ladder: bool,
    target: Option<f64>,
) -> Result<Run, Error>
where
    Complex<T>: Real + EvaluationDomain,
{
    run_rule(lane, prototype, ladder, LadderRule { target, ..LadderRule::default() })
}

/// [`run`] with a whole rule.
pub fn run_rule<T: LadderCarrier>(
    lane: &Lane,
    prototype: T,
    ladder: bool,
    rule: LadderRule,
) -> Result<Run, Error>
where
    Complex<T>: Real + EvaluationDomain,
{
    run_inputs(lane, prototype, ladder, rule, None)
}

/// [`run_rule`] with the shifts' (tracked) precision lowered to
/// `shift_bits`: a point known to fewer bits than the carrier.
pub fn run_inputs<T: LadderCarrier>(
    lane: &Lane,
    prototype: T,
    ladder: bool,
    rule: LadderRule,
    shift_bits: Option<u32>,
) -> Result<Run, Error>
where
    Complex<T>: Real + EvaluationDomain,
{
    let n = lane.n;
    let zero = prototype.zero();
    let cr = |x: &Float| Complex::new(round(x, &prototype), zero.clone());
    let cc = |z: &Complex<Float>| Complex::new(round(&z.re, &prototype), round(&z.im, &prototype));
    let coarse = |x: &Float| {
        let mut v = cr(x);
        if let Some(bits) = shift_bits {
            v.re.set_precision(bits);
        }
        v
    };
    let shifts: Vec<FourMomentum<Complex<T>>> =
        lane.shifts.iter().map(|v| FourMomentum::from(v.clone().map(|x| coarse(&x)))).collect();
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
    let (lost, _, sum) = lane_loss(&amplitude, 0, prototype.get_precision()).ok_or("nothing measured")?;
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
        let mut ladder = PolynomialLadder::new(n, &numerator).with_rule(rule);
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
            Rescued {
                value: v,
                laurent: r.value.as_ref().map(|l| std::array::from_fn(|k| Complex::new(l[k].re.to_f64(), l[k].im.to_f64()))),
                rungs: r.rungs,
                estimate: r.rung_estimate,
                projected: r.projected,
                outcome: r.outcome,
                absolute_bound: r.absolute_bound,
                term_moduli: r.term_moduli,
                best: r.best.as_ref().map(|b| {
                    (Complex::new(b.value[0].re.to_f64(), b.value[0].im.to_f64()), b.estimate, b.bits)
                }),
            }
        })
    } else {
        None
    };
    Ok(Run {
        status,
        laurent: std::array::from_fn(|k| Complex::new(sum[k].re.to_f64(), sum[k].im.to_f64())),
        value,
        lost_bits: lost,
        by_size,
        ladder: rescued,
    })
}

