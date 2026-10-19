//! V0: bitwise regression fixtures for numerator rank <= n+1.
//!
//! Rank n+2 support (`tools/rank_n2/README.md`) must leave every existing
//! rank bitwise unchanged. This test recomputes, for every fixture case of
//! `validation/fast/ninja_reference/{cases,conditioning/cases}` and
//! `validation/benchmark/{cases,synthetic}`:
//! - the fast engine with `TensorNumerator` and `OneLoopNativeMasters` at
//!   width 1 (one evaluation per point) and width 32 (the points repeated
//!   cyclically in one batch): the bits of the Laurent coefficients, the
//!   cut-constructible and rational parts, and the lane status;
//! - the generic engine in binary64 (`PolynomialNumerator` holding the same
//!   tensor, `OneLoopNativeMasters` through `MasterProvider`): bits. Above
//!   rank 5 the polynomial keeps a fixed sixteenth of the tensor's monomials
//!   (always including `q0^rank`; the others are dropped), because Symbolica's
//!   preparation of dense rank-6/7 expansions takes minutes to hours;
//! - numerators with explicit `mu²` terms on a subset of the kinematics: the
//!   fast engine through `GenericNumerator` and `MuTensorNumerator`, and the
//!   generic engine in binary64: bits;
//! - the generic engine in `Float(256)` with `SymbolicaMasters` (exact
//!   oneloopmaster expressions) on six cases: decimal strings.
//!
//! The expected values in `validation/regression/r-le-n1-d53dd08.json` are those
//! of NinjaRS main d53dd08 with oneloopmaster 09b2c74, without any rank n+2
//! change (first generated at e4c16a7), except the generic engine's sections,
//! re-blessed in place for the native bubble tensor coefficients and for the
//! generic cut provider's bubble-reference and tadpole-basis selection; see
//! `validation/regression/README.md` (History) for every reviewed re-bless.
//! Any difference fails.
//! `NINJARS_BLESS=1` rewrites the file (only ever on a reviewed baseline).
//! Everything runs sequentially in one test so Symbolica symbols are always
//! registered in the same order.

mod common;

use common::ninja_reference::{Batch, Case, parse_case};
use ninjars::fast::{
    C64, FastAmplitude, FastOptions, FastResult, FastStatus, GenericNumerator, MuTensorNumerator,
    TensorNumerator,
    tensor_layout::{monomials, mu_block_offset, ncoeff},
};
use ninjars::{AmplitudeInput, Complex, Float, FourMomentum};
use ninjars_symbolica::{
    OneLoopNativeMasters, PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts,
    SymbolicaHighCuts, SymbolicaMasters,
};
use serde_json::{Map, Value, json};
use std::path::{Path, PathBuf};
use symbolica::{atom::Atom, domains::rational::Rational, parser::ParseSettings};

const FIXTURE: &str = "validation/regression/r-le-n1-d53dd08.json";
const NAMESPACE: &str = "ninjars_regression";
const WIDE: usize = 32;
const FLOAT_BITS: u32 = 256;

/// Case files (relative to the repository root) in a fixed order.
const CASE_DIRECTORIES: [&str; 4] = [
    "validation/fast/ninja_reference/cases",
    "validation/fast/ninja_reference/conditioning/cases",
    "validation/benchmark/cases",
    "validation/benchmark/synthetic",
];

/// Kinematics reused with explicit `mu²` numerators of the case's rank.
const MU_CASES: [&str; 9] = [
    "validation/fast/ninja_reference/cases/edge-tadpole-n1-r2.json",
    "validation/benchmark/synthetic/syn-n2-r3.json",
    "validation/benchmark/synthetic/syn-n3-r3.json",
    "validation/benchmark/synthetic/syn-n3-r4.json",
    "validation/benchmark/synthetic/syn-n4-r4.json",
    "validation/benchmark/synthetic/syn-n4-r5.json",
    "validation/benchmark/synthetic/syn-n5-r4.json",
    "validation/benchmark/synthetic/syn-n5-r6.json",
    "validation/benchmark/synthetic/syn-n6-r7.json",
];

/// Cases reduced by the generic engine in `Float(256)`.
const FLOAT_CASES: [&str; 6] = [
    "validation/fast/ninja_reference/cases/edge-tadpole-n1-r2.json",
    "validation/benchmark/synthetic/syn-n2-r3.json",
    "validation/benchmark/synthetic/syn-n3-r4.json",
    "validation/benchmark/synthetic/syn-n4-r5.json",
    "validation/benchmark/cases/gg-hhh-top-pentagon-26.json",
    "validation/benchmark/synthetic/syn-n6-r7.json",
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn parse(text: &str) -> Atom {
    Atom::parse(text, NAMESPACE, ParseSettings::default()).unwrap()
}

fn load(relative: &str) -> Case {
    let text = std::fs::read_to_string(root().join(relative)).unwrap();
    parse_case(&text).unwrap_or_else(|e| panic!("{relative}: {e}"))
}

fn case_files() -> Vec<String> {
    let mut files = Vec::new();
    for directory in CASE_DIRECTORIES {
        let mut names: Vec<String> = std::fs::read_dir(root().join(directory))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".json"))
            .collect();
        names.sort();
        files.extend(names.into_iter().map(|name| format!("{directory}/{name}")));
    }
    files
}

fn hex(z: C64) -> String {
    format!("{:016x}{:016x}", z.re.to_bits(), z.im.to_bits())
}

fn fast_entry(result: &FastResult, status: FastStatus) -> String {
    let values = [
        result.laurent[0],
        result.laurent[1],
        result.laurent[2],
        result.cut_constructible,
        result.rational,
    ];
    let bits: String = values.into_iter().map(hex).collect();
    format!("{status:?}:{bits}")
}

fn generic_entry_f64(values: &[Complex<f64>; 5], status: &impl std::fmt::Debug) -> String {
    let bits: String = values
        .iter()
        .map(|z| hex(C64::from_symbolica(*z)))
        .collect();
    format!("{status:?}:{bits}")
}

fn generic_entry_float(values: &[Complex<Float>; 5], status: &impl std::fmt::Debug) -> Value {
    json!({
        "status": format!("{status:?}"),
        "values": values.iter().map(|z| [z.re.to_string(), z.im.to_string()]).collect::<Vec<_>>(),
    })
}

/// The case's S-matrices (lane-major), when any point carries one; missing
/// matrices are computed from the momenta exactly as `Batch` does.
fn s_matrix(case: &Case) -> Option<Vec<f64>> {
    let points: Vec<usize> = (0..case.points.len()).collect();
    Batch::new(case, &points).smat
}

/// Ninja's dimensional rule for the auxiliary box mu² sampling scale.
fn sampling_scales(case: &Case, matrix: &Option<Vec<f64>>) -> Vec<f64> {
    let n = case.n;
    case.points
        .iter()
        .enumerate()
        .map(|(lane, point)| {
            let invariants = match matrix {
                Some(s) => s[lane * n * n..(lane + 1) * n * n].to_vec(),
                None => Vec::new(),
            };
            invariants
                .iter()
                .map(|x| x.abs())
                .chain(point.masses.iter().map(|m| m.re.abs()))
                .fold(1., f64::max)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Fast engine with the case tensors.

fn fast_tensor(case: &Case) -> Value {
    let mut amplitude = FastAmplitude::new(case.n, case.rank, FastOptions::default()).unwrap();
    let mut masters = OneLoopNativeMasters::new().unwrap();
    let mut evaluate = |points: &[usize]| {
        let mut batch = Batch::new(case, points);
        let mut tensor = std::mem::replace(&mut batch.tensor, TensorNumerator::new(case.rank, 0));
        let mut results = vec![FastResult::default(); points.len()];
        let mut statuses = vec![FastStatus::NotEvaluated; points.len()];
        match amplitude.evaluate(
            &batch.input(),
            &mut tensor,
            &mut masters,
            &mut results,
            &mut statuses,
        ) {
            Ok(()) => results
                .iter()
                .zip(&statuses)
                .map(|(r, &s)| fast_entry(r, s))
                .collect::<Vec<_>>(),
            Err(error) => vec![format!("error:{error}")],
        }
    };
    let count = case.points.len();
    let single: Vec<String> = (0..count).flat_map(|i| evaluate(&[i])).collect();
    let cyclic: Vec<usize> = (0..WIDE).map(|i| i % count).collect();
    let wide: Vec<String> = evaluate(&cyclic)
        .into_iter()
        .enumerate()
        .map(|(lane, entry)| {
            // Repeated points are stored by reference to the width-1 entry.
            if single.get(cyclic[lane]) == Some(&entry) {
                format!("={}", cyclic[lane])
            } else {
                entry
            }
        })
        .collect();
    json!({"n": case.n, "rank": case.rank, "width1": single, "width32": wide})
}

// ---------------------------------------------------------------------------
// Polynomial numerators.

/// `(q exponents, mu² power)` of every term of a rank-`rank` numerator with
/// `mu²` powers `0..=mu_powers`, in `MuTensorNumerator` block order.
fn terms(rank: usize, mu_powers: usize) -> Vec<([u32; 4], u32)> {
    let mut terms = Vec::new();
    for k in 0..=mu_powers {
        terms.extend(monomials(rank - 2 * k).into_iter().map(|m| (m, k as u32)));
    }
    terms
}

/// Dense through rank 5; above, a fixed pseudo-random sixteenth of the terms
/// (always including the leading `q0^rank`) keeps Symbolica's preparation of
/// the generic engine's numerator expansions affordable (a dense rank-6
/// numerator over seven propagators takes about two minutes, rank 7 far
/// longer).
fn active(rank: usize, k: usize) -> bool {
    rank <= 5 || k == ncoeff(rank - 1) || (k.wrapping_mul(2_654_435_761) >> 7).is_multiple_of(16)
}

/// `sum_k p_k * term_k` over the selected terms, with one parameter per term.
fn polynomial(terms: &[([u32; 4], u32)], keep: impl Fn(usize) -> bool) -> (Atom, Vec<Atom>) {
    let names = ["q0", "q1", "q2", "q3"];
    let mut text = String::new();
    for (k, (m, j)) in terms.iter().enumerate() {
        if !keep(k) {
            continue;
        }
        if !text.is_empty() {
            text.push('+');
        }
        text.push_str(&format!("p{k}"));
        for (mu, &e) in m.iter().enumerate() {
            if e > 0 {
                text.push_str(&format!("*{}^{e}", names[mu]));
            }
        }
        if *j > 0 {
            text.push_str(&format!("*mu2^{j}"));
        }
    }
    if text.is_empty() {
        text.push('0');
    }
    let parameters = (0..terms.len()).map(|k| parse(&format!("p{k}"))).collect();
    (parse(&text), parameters)
}

fn variables() -> [Atom; 5] {
    ["q0", "q1", "q2", "q3", "mu2"].map(parse)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.
    }
}

fn seed(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

// ---------------------------------------------------------------------------
// Generic engine.

fn exact<T: ninjars::SingleFloat>(prototype: &T, x: f64) -> T {
    prototype.from_rational(&Rational::try_from(x).unwrap())
}

fn complex<T: ninjars::SingleFloat>(prototype: &T, z: C64) -> Complex<T> {
    Complex::new(exact(prototype, z.re), exact(prototype, z.im))
}

/// Per-lane generic results and statuses.
type GenericLanes<T> = (Vec<[Complex<T>; 5]>, Vec<ninjars::EvaluationStatus>);

/// Generic reduction of all points of `case` with per-lane coefficients
/// `coefficients` (row-major, one row per point) of `numerator`.
fn generic<T, M>(
    case: &Case,
    numerator: &mut PolynomialNumerator<T>,
    coefficients: &[C64],
    prototype: T,
    masters: &mut M,
) -> Result<GenericLanes<T>, String>
where
    T: oneloop::NativeFloat,
    Complex<T>: ninjars::Real + symbolica::evaluate::EvaluationDomain,
    M: ninjars::MasterProvider<Complex<T>>,
{
    let p = &prototype;
    let n = case.n;
    let count = case.points.len();
    let values: Vec<_> = coefficients.iter().map(|&z| complex(p, z)).collect();
    numerator
        .set_parameter_batch(&values, count)
        .map_err(|e| e.to_string())?;
    let real = |x: f64| complex(p, C64::real(x));
    let shifts: Vec<FourMomentum<Complex<T>>> = case
        .points
        .iter()
        .flat_map(|point| point.shifts.iter().map(|v| FourMomentum::from(v.map(real))))
        .collect();
    let masses: Vec<_> = case
        .points
        .iter()
        .flat_map(|point| point.masses.iter().map(|&m| complex(p, m)))
        .collect();
    let matrix = s_matrix(case);
    let generic_matrix: Option<Vec<_>> = matrix
        .as_ref()
        .map(|s| s.iter().map(|&x| real(x)).collect());
    let scales: Vec<_> = case.points.iter().map(|point| real(point.mu2)).collect();
    let sampling: Vec<_> = sampling_scales(case, &matrix)
        .into_iter()
        .map(real)
        .collect();
    let mut low = SymbolicaCuts::new(p.clone()).map_err(|e| e.to_string())?;
    let mut high = SymbolicaHighCuts::new(p.clone()).map_err(|e| e.to_string())?;
    let mut amplitude = SymbolicaAmplitudeKernels::new(n, numerator_rank(case), p.clone())
        .and_then(|kernels| kernels.into_amplitude(count))
        .map_err(|e| e.to_string())?;
    amplitude
        .evaluate_batch_with_inputs(
            AmplitudeInput {
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: &scales,
                sampling_scales: &sampling,
                s_matrix: generic_matrix.as_deref(),
            },
            numerator,
            &mut low,
            &mut high,
            masters,
        )
        .map_err(|e| e.to_string())?;
    Ok((amplitude.results().to_vec(), amplitude.statuses().to_vec()))
}

fn numerator_rank(case: &Case) -> usize {
    case.rank
}

fn tensor_polynomial<T>(case: &Case, prototype: T) -> PolynomialNumerator<T>
where
    T: oneloop::NativeFloat,
    Complex<T>: ninjars::Real + symbolica::evaluate::EvaluationDomain,
{
    let (expression, parameters) = polynomial(&terms(case.rank, 0), |k| active(case.rank, k));
    PolynomialNumerator::new(expression, variables(), parameters, prototype).unwrap()
}

/// The case tensors, with the coefficients of dropped monomials zeroed (they
/// are absent from the polynomial; see [`active`]).
fn tensor_coefficients(case: &Case) -> Vec<C64> {
    case.points
        .iter()
        .flat_map(|p| {
            p.tensor
                .iter()
                .enumerate()
                .map(|(k, &z)| if active(case.rank, k) { z } else { C64::ZERO })
        })
        .collect()
}

fn generic_tensor_f64(case: &Case) -> Value {
    let mut numerator = tensor_polynomial(case, 0f64);
    let mut masters = OneLoopNativeMasters::new().unwrap();
    match generic(
        case,
        &mut numerator,
        &tensor_coefficients(case),
        0f64,
        &mut masters,
    ) {
        Ok((results, statuses)) => json!(
            results
                .iter()
                .zip(&statuses)
                .map(|(r, s)| generic_entry_f64(r, s))
                .collect::<Vec<_>>()
        ),
        Err(error) => json!([format!("error:{error}")]),
    }
}

fn generic_tensor_float(case: &Case) -> Value {
    let prototype = Float::new(FLOAT_BITS);
    let mut numerator = tensor_polynomial(case, prototype.clone());
    let mut masters = SymbolicaMasters::new(prototype.clone());
    match generic(
        case,
        &mut numerator,
        &tensor_coefficients(case),
        prototype,
        &mut masters,
    ) {
        Ok((results, statuses)) => json!(
            results
                .iter()
                .zip(&statuses)
                .map(|(r, s)| generic_entry_float(r, s))
                .collect::<Vec<_>>()
        ),
        Err(error) => json!([format!("error:{error}")]),
    }
}

// ---------------------------------------------------------------------------
// Explicit mu² numerators.

/// One fast evaluation of all lanes with some numerator.
type FastRun<'a> = dyn FnMut(
        &mut FastAmplitude,
        &mut OneLoopNativeMasters,
        &mut [FastResult],
        &mut [FastStatus],
    ) -> ninjars::Result<()>
    + 'a;

fn mu_numerator(case: &Case, name: &str) -> Value {
    let (n, rank) = (case.n, case.rank);
    let mu_powers = rank / 2;
    let terms = terms(rank, mu_powers);
    let count = case.points.len();
    let mut rng = Rng(seed(name));
    let mut coefficients = vec![C64::ZERO; count * terms.len()];
    for lane in 0..count {
        for k in 0..terms.len() {
            if active(rank, k) {
                coefficients[lane * terms.len() + k] = C64::new(rng.uniform(), rng.uniform());
            }
        }
    }
    let (expression, parameters) = polynomial(&terms, |k| active(rank, k));
    let mut numerator =
        PolynomialNumerator::new(expression, variables(), parameters, 0f64).unwrap();

    // Generic engine first (it binds the parameters of `numerator`).
    let mut masters = OneLoopNativeMasters::new().unwrap();
    let generic_f64 = match generic(case, &mut numerator, &coefficients, 0f64, &mut masters) {
        Ok((results, statuses)) => results
            .iter()
            .zip(&statuses)
            .map(|(r, s)| generic_entry_f64(r, s))
            .collect::<Vec<_>>(),
        Err(error) => vec![format!("error:{error}")],
    };

    let points: Vec<usize> = (0..count).collect();
    let batch = Batch::new(case, &points);
    let input = batch.input();
    let mut amplitude = FastAmplitude::new(n, rank, FastOptions::default()).unwrap();
    let mut fast_masters = OneLoopNativeMasters::new().unwrap();
    let mut run = |numerator: &mut FastRun<'_>| {
        let mut results = vec![FastResult::default(); count];
        let mut statuses = vec![FastStatus::NotEvaluated; count];
        match numerator(
            &mut amplitude,
            &mut fast_masters,
            &mut results,
            &mut statuses,
        ) {
            Ok(()) => results
                .iter()
                .zip(&statuses)
                .map(|(r, &s)| fast_entry(r, s))
                .collect::<Vec<_>>(),
            Err(error) => vec![format!("error:{error}")],
        }
    };

    let mut adapted = GenericNumerator::with_rank(numerator, rank);
    let fast_generic = run(&mut |amplitude, masters, results, statuses| {
        amplitude.evaluate(&input, &mut adapted, masters, results, statuses)
    });

    let mut tensor = MuTensorNumerator::new(rank, mu_powers, count).unwrap();
    for lane in 0..count {
        tensor
            .lane_coefficients_mut(lane)
            .copy_from_slice(&coefficients[lane * terms.len()..(lane + 1) * terms.len()]);
    }
    assert_eq!(mu_block_offset(rank, mu_powers + 1), terms.len());
    let fast_mu_tensor = run(&mut |amplitude, masters, results, statuses| {
        amplitude.evaluate(&input, &mut tensor, masters, results, statuses)
    });

    json!({
        "n": n,
        "rank": rank,
        "mu_powers": mu_powers,
        "terms": terms.len(),
        "generic_f64": generic_f64,
        "fast_generic_numerator": fast_generic,
        "fast_mu_tensor": fast_mu_tensor,
    })
}

// ---------------------------------------------------------------------------

fn compute() -> Value {
    let started = std::time::Instant::now();
    let mut fast = Map::new();
    let mut generic_f64 = Map::new();
    for file in case_files() {
        let case = load(&file);
        fast.insert(file.clone(), fast_tensor(&case));
        let start = std::time::Instant::now();
        generic_f64.insert(file.clone(), generic_tensor_f64(&case));
        eprintln!(
            "V0 {file}: n={} rank={} generic f64 {:.1}s",
            case.n,
            case.rank,
            start.elapsed().as_secs_f64()
        );
    }
    let mut mu = Map::new();
    for file in MU_CASES {
        let start = std::time::Instant::now();
        mu.insert(file.into(), mu_numerator(&load(file), file));
        eprintln!("V0 mu {file}: {:.1}s", start.elapsed().as_secs_f64());
    }
    let mut float = Map::new();
    for file in FLOAT_CASES {
        let start = std::time::Instant::now();
        float.insert(file.into(), generic_tensor_float(&load(file)));
        eprintln!("V0 Float {file}: {:.1}s", start.elapsed().as_secs_f64());
    }
    eprintln!("V0 total {:.1}s", started.elapsed().as_secs_f64());
    json!({
        "format": "ninjars-regression-bitwise-v1",
        "baseline": "d53dd08 (NinjaRS main without rank n+2; first generated at e4c16a7), oneloopmaster 09b2c74",
        "description": "Bitwise outputs for numerator rank <= n+1: fast engine (TensorNumerator + OneLoopNativeMasters, widths 1 and 32; entries 'status:hex' of [laurent0..2, cut_constructible, rational] as (re,im) f64 bits, '=k' repeats width-1 entry k), generic f64 (PolynomialNumerator + OneLoopNativeMasters), explicit mu^2 numerators (fast via GenericNumerator and MuTensorNumerator, generic f64), generic Float(256) with SymbolicaMasters (decimal strings). Test: crates/ninjars-symbolica/tests/regression_bitwise.rs.",
        "fast_tensor": fast,
        "generic_f64": generic_f64,
        "mu_numerators": mu,
        "generic_float256": float,
    })
}

/// First differing JSON paths, for a readable failure.
fn differences(path: &str, expected: &Value, actual: &Value, out: &mut Vec<String>) {
    if out.len() >= 20 {
        return;
    }
    match (expected, actual) {
        (Value::Object(a), Value::Object(b)) => {
            for key in a.keys().chain(b.keys().filter(|k| !a.contains_key(*k))) {
                differences(
                    &format!("{path}/{key}"),
                    a.get(key).unwrap_or(&Value::Null),
                    b.get(key).unwrap_or(&Value::Null),
                    out,
                );
            }
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                differences(&format!("{path}[{i}]"), x, y, out);
            }
        }
        _ if expected != actual => out.push(format!("{path}: expected {expected}, got {actual}")),
        _ => {}
    }
}

#[test]
fn rank_up_to_n_plus_one_is_bitwise_unchanged() {
    let actual = compute();
    let path: &Path = &root().join(FIXTURE);
    if std::env::var("NINJARS_BLESS").is_ok_and(|v| v == "1") {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut text = serde_json::to_string_pretty(&actual).unwrap();
        text.push('\n');
        std::fs::write(path, text).unwrap();
        eprintln!("V0 blessed {}", path.display());
        return;
    }
    let expected: Value = serde_json::from_str(
        &std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("{}: {e} (bless with NINJARS_BLESS=1)", path.display())),
    )
    .unwrap();
    let mut diffs = Vec::new();
    for key in [
        "fast_tensor",
        "generic_f64",
        "mu_numerators",
        "generic_float256",
    ] {
        differences(key, &expected[key], &actual[key], &mut diffs);
    }
    assert!(
        diffs.is_empty(),
        "rank <= n+1 results changed:\n{}",
        diffs.join("\n")
    );
}
