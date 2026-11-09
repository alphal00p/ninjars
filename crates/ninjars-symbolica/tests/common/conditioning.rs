//! Box mu-expansion conditioning fixtures (`validation/conditioning/box-mu`):
//! single-lane inputs at near-planar 2->3 points, as exact decimal strings,
//! reduced by every engine and carrier of NinjaRS.
//!
//! A fixture holds one lane: the shifts, squared masses, scale and invariant
//! matrix of a family as an amplitude executor forms them (exact sums of the
//! binary64 external momenta, on-shell external invariants), a dense or sparse
//! tensor numerator in [`tensor_layout`](ninjars::fast::tensor_layout) order,
//! and a reference: the generic engine's finite part in two working precisions
//! of at least 100 digits that agree. Every carrier parses the same decimal
//! strings (rounded once to its own precision), so the carriers reduce the
//! same exact lane. Shared by `tests/box_mu_conditioning.rs` and
//! `examples/box_mu_conditioning.rs`.

#![allow(dead_code)]

use ninjars::fast::{
    C64, FastAmplitude, FastInput, FastOptions, FastResult, FastStatus, RMom, TensorNumerator,
    tensor_layout::{monomials, ncoeff},
};
use ninjars::{
    AmplitudeInput, Complex, EvaluationStatus, Float, FourMomentum, SamplingScaleProvider,
    SamplingScaleSource, SamplingScaleStatus,
};
use ninjars_symbolica::{
    OneLoopNativeMasters, PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts,
    SymbolicaHighCuts, SymbolicaMasters, SymbolicaSamplingScales,
};
use serde_json::Value;
use symbolica::{
    atom::Atom,
    domains::float::{DoubleFloat, Real, RealLike},
    parser::ParseSettings,
};

pub type Error = Box<dyn std::error::Error>;

/// Format tag of a fixture file.
pub const FORMAT: &str = "ninjars-box-mu-conditioning-v1";

#[derive(Clone, Debug)]
pub struct Fixture {
    pub name: String,
    pub n: usize,
    pub rank: usize,
    pub shifts: Vec<[String; 4]>,
    pub masses_squared: Vec<[String; 2]>,
    pub mu_squared: String,
    pub s_matrix: Vec<String>,
    /// Per lane `ncoeff(rank)` complex coefficients (absent monomials `"0"`):
    /// `tensor`, or every entry of `tensors` (lanes sharing the geometry).
    pub tensors: Vec<Vec<[String; 2]>>,
    /// Per lane the reference finite part `[re, im]`, if recorded.
    pub references: Vec<Option<[String; 2]>>,
}

fn text(value: &Value) -> Result<String, Error> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::Number(n) => Ok(n.to_string()),
        _ => Err("expected a decimal string".into()),
    }
}

fn pair(value: &Value) -> Result<[String; 2], Error> {
    Ok([text(&value[0])?, text(&value[1])?])
}

pub fn parse_fixture(source: &str) -> Result<Fixture, Error> {
    let value: Value = serde_json::from_str(source)?;
    if value["format"] != FORMAT {
        return Err(format!("not a {FORMAT} file").into());
    }
    let n = value["n"].as_u64().ok_or("n")? as usize;
    let rank = value["rank"].as_u64().ok_or("rank")? as usize;
    let shifts = value["shifts"]
        .as_array()
        .ok_or("shifts")?
        .iter()
        .map(|v| Ok([text(&v[0])?, text(&v[1])?, text(&v[2])?, text(&v[3])?]))
        .collect::<Result<Vec<_>, Error>>()?;
    let masses_squared = value["masses_squared"]
        .as_array()
        .ok_or("masses_squared")?
        .iter()
        .map(pair)
        .collect::<Result<Vec<_>, _>>()?;
    let s_matrix = value["s_matrix"]
        .as_array()
        .ok_or("s_matrix")?
        .iter()
        .flat_map(|row| row.as_array().cloned().unwrap_or_default())
        .map(|x| text(&x))
        .collect::<Result<Vec<_>, _>>()?;
    let tensor = |value: &Value| -> Result<Vec<[String; 2]>, Error> {
        value
            .as_array()
            .ok_or("tensor")?
            .iter()
            .map(pair)
            .collect::<Result<Vec<_>, _>>()
    };
    let reference = |value: &Value| -> Result<Option<[String; 2]>, Error> {
        Ok(match value {
            Value::Null => None,
            r => Some(pair(&r["finite"])?),
        })
    };
    let (tensors, references) = match &value["tensors"] {
        Value::Array(lanes) => (
            lanes.iter().map(tensor).collect::<Result<Vec<_>, _>>()?,
            match &value["references"] {
                Value::Array(r) => r.iter().map(reference).collect::<Result<Vec<_>, _>>()?,
                _ => vec![None; lanes.len()],
            },
        ),
        _ => (
            vec![tensor(&value["tensor"])?],
            vec![reference(&value["reference"])?],
        ),
    };
    if shifts.len() != n
        || masses_squared.len() != n
        || s_matrix.len() != n * n
        || tensors.iter().any(|t| t.len() != ncoeff(rank))
        || references.len() != tensors.len()
    {
        return Err("inconsistent fixture shapes".into());
    }
    Ok(Fixture {
        name: value["name"].as_str().unwrap_or("").to_string(),
        n,
        rank,
        shifts,
        masses_squared,
        mu_squared: text(&value["mu_squared"])?,
        s_matrix,
        tensors,
        references,
    })
}

/// A carrier's reduction of one fixture: the status and, on success,
/// `[finite, 1/eps, 1/eps^2]` rounded to binary64 for comparison.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub status: String,
    pub success: bool,
    pub laurent: Option<[C64; 3]>,
    /// `[cut-constructible, rational]` finite parts, rounded to binary64.
    pub parts: Option<[C64; 2]>,
    /// The finite part as a decimal string in the carrier's precision.
    pub finite_text: Option<[String; 2]>,
}

fn f64_of(text: &str) -> f64 {
    text.parse::<f64>().expect("decimal")
}

/// One lane of a fixture: the fixture and the index of its tensor.
pub type LaneRef<'a> = (&'a Fixture, usize);

/// Every lane of `fixture`.
pub fn lanes(fixture: &Fixture) -> Vec<LaneRef<'_>> {
    (0..fixture.tensors.len())
        .map(|lane| (fixture, lane))
        .collect()
}

/// The fast binary64 engine with `TensorNumerator` and `OneLoopNativeMasters`
/// (the native masters a binary64 amplitude executor uses), on lanes of
/// equal `(n, rank)` (each with its own geometry) in one batch.
pub fn fast_lanes(lanes: &[LaneRef<'_>], options: FastOptions) -> Result<Vec<Outcome>, Error> {
    let Some(&(first, _)) = lanes.first() else {
        return Ok(Vec::new());
    };
    let (n, rank, count) = (first.n, first.rank, lanes.len());
    if lanes.iter().any(|(f, _)| f.n != n || f.rank != rank) {
        return Err("a fast batch needs equal (n, rank)".into());
    }
    let mut shifts: Vec<RMom> = Vec::with_capacity(n * count);
    let mut masses: Vec<C64> = Vec::with_capacity(n * count);
    let mut s_matrix: Vec<f64> = Vec::with_capacity(n * n * count);
    let mut scales = Vec::with_capacity(count);
    let mut coefficients: Vec<C64> = Vec::new();
    for &(f, lane) in lanes {
        shifts.extend(
            f.shifts
                .iter()
                .map(|v| [f64_of(&v[0]), f64_of(&v[1]), f64_of(&v[2]), f64_of(&v[3])]),
        );
        masses.extend(
            f.masses_squared
                .iter()
                .map(|m| C64::new(f64_of(&m[0]), f64_of(&m[1]))),
        );
        s_matrix.extend(f.s_matrix.iter().map(|x| f64_of(x)));
        scales.push(f64_of(&f.mu_squared));
        coefficients.extend(
            f.tensors[lane]
                .iter()
                .map(|z| C64::new(f64_of(&z[0]), f64_of(&z[1]))),
        );
    }
    let mut amplitude = FastAmplitude::new(n, rank, options)?;
    let mut numerator = TensorNumerator::new(rank, 0);
    numerator.set_coefficients(count, &coefficients)?;
    let mut masters = OneLoopNativeMasters::new()?;
    let mut results = vec![FastResult::default(); count];
    let mut statuses = vec![FastStatus::NotEvaluated; count];
    amplitude.evaluate(
        &FastInput {
            lanes: count,
            shifts: &shifts,
            masses_squared: &masses,
            scales_squared: &scales,
            s_matrix: Some(&s_matrix),
        },
        &mut numerator,
        &mut masters,
        &mut results,
        &mut statuses,
    )?;
    Ok((0..count)
        .map(|lane| {
            let success = statuses[lane] == FastStatus::Success;
            let r = &results[lane];
            Outcome {
                status: format!("{:?}", statuses[lane]),
                success,
                laurent: success.then_some(r.laurent),
                parts: success.then_some([r.cut_constructible, r.rational]),
                finite_text: success.then(|| {
                    [
                        format!("{:e}", r.laurent[0].re),
                        format!("{:e}", r.laurent[0].im),
                    ]
                }),
            }
        })
        .collect())
}

/// [`fast_lanes`] on every lane of one fixture.
pub fn fast(fixture: &Fixture, options: FastOptions) -> Result<Vec<Outcome>, Error> {
    fast_lanes(&lanes(fixture), options)
}

/// A carrier of the generic engine, parsed from the fixture's decimals.
pub trait GenericCarrier: oneloop::NativeFloat + RealLike + std::fmt::LowerExp
where
    Complex<Self>: Real + symbolica::evaluate::EvaluationDomain,
{
    fn parse(text: &str, bits: u32) -> Self;
}

impl GenericCarrier for f64 {
    fn parse(text: &str, _: u32) -> Self {
        f64_of(text)
    }
}

impl GenericCarrier for DoubleFloat {
    fn parse(text: &str, _: u32) -> Self {
        Float::parse(text, Some(106)).unwrap().to_double_float()
    }
}

impl GenericCarrier for Float {
    fn parse(text: &str, bits: u32) -> Self {
        Float::parse(text, Some(bits)).unwrap()
    }
}

fn atom(text: &str) -> Atom {
    Atom::parse(
        text,
        "ninjars_box_mu_conditioning",
        ParseSettings::default(),
    )
    .unwrap()
}

/// `Σ_k c_k q^α_k` over the tensor layout, one parameter per coefficient
/// (an amplitude executor's generic numerator template).
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

/// The precision-generic engine in carrier `T` (`bits` for `Float`): the
/// polynomial template numerator, `SymbolicaMasters` without an on-shell
/// threshold and the auxiliary sampling scales of `SymbolicaSamplingScales`,
/// on lanes of equal `(n, rank)` (each with its own geometry) in one batch.
pub fn generic_lanes<T: GenericCarrier>(
    lanes: &[LaneRef<'_>],
    prototype: T,
    bits: u32,
) -> Result<Vec<Outcome>, Error>
where
    Complex<T>: Real + symbolica::evaluate::EvaluationDomain,
{
    let Some(&(first, _)) = lanes.first() else {
        return Ok(Vec::new());
    };
    let (n, rank, count) = (first.n, first.rank, lanes.len());
    if lanes.iter().any(|(f, _)| f.n != n || f.rank != rank) {
        return Err("a generic batch needs equal (n, rank)".into());
    }
    let zero = prototype.zero();
    let real = |text: &str| Complex::new(T::parse(text, bits), zero.clone());
    let complex = |z: &[String; 2]| Complex::new(T::parse(&z[0], bits), T::parse(&z[1], bits));
    let mut shifts: Vec<FourMomentum<Complex<T>>> = Vec::with_capacity(n * count);
    let mut masses: Vec<Complex<T>> = Vec::with_capacity(n * count);
    let mut s_matrix: Vec<Complex<T>> = Vec::with_capacity(n * n * count);
    let mut scales = Vec::with_capacity(count);
    let mut coefficients: Vec<Complex<T>> = Vec::new();
    for &(f, lane) in lanes {
        shifts.extend(
            f.shifts
                .iter()
                .map(|v| FourMomentum::from([real(&v[0]), real(&v[1]), real(&v[2]), real(&v[3])])),
        );
        masses.extend(f.masses_squared.iter().map(complex));
        s_matrix.extend(f.s_matrix.iter().map(|x| real(x)));
        scales.push(real(&f.mu_squared));
        coefficients.extend(f.tensors[lane].iter().map(complex));
    }
    let (expression, parameters) = template(rank);
    let variables = ["q0", "q1", "q2", "q3", "mu2"].map(atom);
    let mut numerator =
        PolynomialNumerator::new(expression, variables, parameters, prototype.clone())?;
    numerator.set_parameter_batch(&coefficients, count)?;
    let mut sampling = vec![Complex::new(zero.clone(), zero.clone()); count];
    let mut sampling_status =
        vec![SamplingScaleStatus::Selected(SamplingScaleSource::UnitFallback); count];
    SymbolicaSamplingScales::new(n, prototype.clone())?.evaluate_batch(
        &shifts,
        &masses,
        count,
        &mut sampling,
        &mut sampling_status,
    )?;
    let mut low = SymbolicaCuts::new(prototype.clone())?;
    let mut high = SymbolicaHighCuts::new(prototype.clone())?;
    let mut masters = SymbolicaMasters::new(prototype.clone()).with_onshell_threshold(None);
    let mut amplitude =
        SymbolicaAmplitudeKernels::new(n, rank, prototype.clone())?.into_amplitude(count)?;
    amplitude.reset_batch(count);
    amplitude.evaluate_batch_with_inputs(
        AmplitudeInput {
            shifts: &shifts,
            masses_squared: &masses,
            scales_squared: &scales,
            sampling_scales: &sampling,
            s_matrix: Some(&s_matrix),
        },
        &mut numerator,
        &mut low,
        &mut high,
        &mut masters,
    )?;
    let digits = (f64::from(bits.max(53)) * std::f64::consts::LOG10_2).ceil() as usize + 2;
    let c = |z: &Complex<T>| C64::new(z.re.to_f64(), z.im.to_f64());
    Ok((0..count)
        .map(|lane| {
            let status = amplitude.statuses()[lane];
            let success = status == EvaluationStatus::Success;
            let r = &amplitude.results()[lane];
            Outcome {
                status: format!("{status:?}"),
                success,
                laurent: success.then(|| [c(&r[0]), c(&r[1]), c(&r[2])]),
                parts: success.then(|| [c(&r[3]), c(&r[4])]),
                finite_text: success.then(|| {
                    [
                        format!("{:.digits$e}", r[0].re),
                        format!("{:.digits$e}", r[0].im),
                    ]
                }),
            }
        })
        .collect())
}

/// [`generic_lanes`] on every lane of one fixture.
pub fn generic<T: GenericCarrier>(
    fixture: &Fixture,
    prototype: T,
    bits: u32,
) -> Result<Vec<Outcome>, Error>
where
    Complex<T>: Real + symbolica::evaluate::EvaluationDomain,
{
    generic_lanes(&lanes(fixture), prototype, bits)
}

/// Relative distance of a binary64 finite part from a decimal reference.
pub fn relative_error(value: C64, reference: &[String; 2]) -> f64 {
    let r = C64::new(f64_of(&reference[0]), f64_of(&reference[1]));
    (value - r).abs() / r.abs()
}
