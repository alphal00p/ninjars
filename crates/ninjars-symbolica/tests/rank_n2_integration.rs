//! Rank n+2 integration in the generic engine.
//!
//! - `MasterContraction::new_class(_, NPlus2)`: coefficient counts 16/35/21/7,
//!   the bubble's B1111 weight `c20 ke2⁴` and the new rational terms of the
//!   bubble (`c21` mu²(q·e2)², `c22` mu⁴) and the box (`c6` mu⁶), checked
//!   against the closed forms of `tools/rank_n2/README.md`; the rank n+1 part
//!   is the unchanged rank n+1 plan.
//! - V4 smoke: the reference suite's complex-mass lanes n1-r3-mu2, n2-r4,
//!   n3-r5 and n4-r6 with and without explicit mu² tensors, in Float(256)
//!   with exact masters (`SymbolicaMasters`, including oneloopmaster's exact
//!   B1111), against (a) the review's independent 60-digit reduction
//!   (`tools/rank_n2/review/red.py` with exact masters,
//!   `tests/rank_n2_data/red-*.json`) to <= 1e-25, and (b) the high-precision
//!   COLLIER references (60-digit preprocessing, `tools/rank_n2/review/hpref`
//!   and the same procedure for the mu²-free cases), which the engine must
//!   match as well as (a) does. See `tests/rank_n2_data/README.md`.
//! - Pinch identity: `I_n[D_j M] = I_(n-1)[M]` for rank(M) = n, so that the
//!   left side runs the rank n+2 path and the right side the unchanged rank
//!   n+1 path (n = 2..4, every j, exact masters in Float(256)).

mod rank_n2_support;

use std::path::{Path, PathBuf};

use ninjars::fast::{C64, RMom, tensor_layout::monomials};
use ninjars::{
    AmplitudeInput, Complex, EvaluationStatus, Float, FourMomentum, MasterIntegral, Numerator,
    RankClass,
};
use ninjars_symbolica::{
    MasterContraction, PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts,
    SymbolicaHighCuts, SymbolicaMasters,
};
use rank_n2_support::{
    Problem, Rng, complex, distance, laurent_error, modulus, parse, to_c64, variables,
};
use serde_json::Value;

const BITS: u32 = 256;

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/rank_n2_data")
        .join(name)
}

fn read(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn float(x: f64) -> Complex<Float> {
    complex(&Float::new(BITS), C64::real(x))
}

fn fc(z: C64) -> Complex<Float> {
    complex(&Float::new(BITS), z)
}

// ---------------------------------------------------------------------------
// Master contraction.

#[test]
fn rank_n2_master_contraction() {
    for (cut, count) in [(1, 16), (2, 35), (3, 21), (4, 7)] {
        let plan = MasterContraction::new_class(cut, RankClass::NPlus2).unwrap();
        assert_eq!(plan.input_layout().coefficients, 0..count);
        let higher = MasterContraction::new_class(cut, RankClass::NPlus1).unwrap();
        let expected: Vec<_> = if cut == 2 {
            let mut masters = higher.masters().to_vec();
            masters.push(MasterIntegral::B1111);
            masters
        } else {
            higher.masters().to_vec()
        };
        assert_eq!(plan.masters(), expected.as_slice(), "cut {cut}");
    }
    assert!(MasterContraction::new_class(5, RankClass::NPlus2).is_err());

    let mut rng = Rng(0xc0_2017);
    let p = Float::new(BITS);
    // Weights of a rank n+2 plan minus those of the rank n+1 plan fed with the
    // shared coefficients: only the new terms remain.
    for cut in [1usize, 2, 3, 4] {
        let plan = MasterContraction::new_class(cut, RankClass::NPlus2).unwrap();
        let higher = MasterContraction::new_class(cut, RankClass::NPlus1).unwrap();
        let layout = plan.input_layout().clone();
        let input: Vec<_> = (0..plan.input_columns())
            .map(|_| fc(rng.complex()))
            .collect();
        let shared = higher.input_layout().coefficients.len();
        let mut lower_input = input[..shared].to_vec();
        lower_input.extend_from_slice(&input[layout.coefficients.end..]);
        let mut weights = vec![fc(C64::ZERO); plan.weight_columns()];
        plan.compile(&p)
            .unwrap()
            .weights_batch(&input, &mut weights, 1)
            .unwrap();
        let mut lower = vec![fc(C64::ZERO); higher.weight_columns()];
        higher
            .compile(&p)
            .unwrap()
            .weights_batch(&lower_input, &mut lower, 1)
            .unwrap();
        let c = &input[layout.coefficients.clone()];
        let s = &input[layout.invariants.clone()];
        let m = &input[layout.masses_squared.clone()];
        let rational_difference = weights.last().unwrap().clone() - lower.last().unwrap().clone();
        let (expected_rational, expected_b1111) = match cut {
            2 => {
                let ke2 = &input[layout.ke2.unwrap()];
                let ke2_2 = ke2.clone() * ke2;
                let rational = c[21].clone()
                    * &ke2_2
                    * &(m[0].clone() * &float(5.) + m[1].clone() * &float(15.)
                        - s[0].clone() * &float(3.))
                    / &float(60.)
                    - c[22].clone()
                        * &(s[0].clone() * &s[0]
                            - s[0].clone() * &(m[0].clone() + &m[1]) * &float(5.)
                            + (m[0].clone() * &m[0] + m[0].clone() * &m[1] + m[1].clone() * &m[1])
                                * &float(10.))
                        / &float(60.);
                (rational, Some(c[20].clone() * &ke2_2 * &ke2_2))
            }
            4 => {
                let masses = m.iter().fold(fc(C64::ZERO), |a, b| a + b);
                let invariants = s.iter().fold(fc(C64::ZERO), |a, b| a + b);
                (
                    c[6].clone() * &(masses / &float(12.) - invariants / &float(60.)),
                    None,
                )
            }
            _ => (fc(C64::ZERO), None),
        };
        let scale = modulus(&expected_rational).max(1.);
        assert!(
            distance(&rational_difference, &expected_rational) <= 1e-70 * scale,
            "cut {cut}: {:?} vs {:?}",
            to_c64(&rational_difference),
            to_c64(&expected_rational)
        );
        for (k, w) in lower.iter().take(higher.masters().len()).enumerate() {
            assert!(distance(&weights[k], w) <= 1e-70 * modulus(w).max(1.));
        }
        if let Some(b1111) = expected_b1111 {
            assert!(distance(&weights[4], &b1111) <= 1e-70 * modulus(&b1111).max(1.));
        }
    }
}

// ---------------------------------------------------------------------------
// V4 smoke against high-precision COLLIER references.

struct CaseLane {
    shifts: Vec<RMom>,
    masses: Vec<C64>,
    scale: f64,
    s_matrix: Vec<f64>,
    /// `(q exponents, mu² power)` and coefficient.
    terms: Vec<(([u32; 4], u32), C64)>,
}

struct Case {
    name: String,
    n: usize,
    rank: usize,
    lanes: Vec<CaseLane>,
}

fn number(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

fn rank_of(length: usize) -> usize {
    (0..)
        .find(|&r| ninjars::fast::tensor_layout::ncoeff(r) == length)
        .unwrap()
}

fn load_case(path: &Path) -> Case {
    let value = read(path);
    assert_eq!(value["format"], "ninjars-bench-case-v1");
    let n = value["n"].as_u64().unwrap() as usize;
    let rank = value["rank"].as_u64().unwrap() as usize;
    let lanes = value["lanes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|lane| {
            let pair = |v: &Value| C64::new(number(&v[0]), number(&v[1]));
            let shifts = lane["shifts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| [0, 1, 2, 3].map(|i| number(&p[i])))
                .collect();
            let masses = lane["masses_squared"]
                .as_array()
                .unwrap()
                .iter()
                .map(pair)
                .collect();
            let s_matrix = lane["s_matrix"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|row| row.as_array().unwrap().iter().map(number))
                .collect();
            let mut terms = Vec::new();
            let mut add = |tensor: &Value, power: u32| {
                let tensor = tensor.as_array().unwrap();
                for (m, c) in monomials(rank_of(tensor.len())).into_iter().zip(tensor) {
                    terms.push(((m, power), pair(c)));
                }
            };
            add(&lane["tensor"], 0);
            if let Some(extra) = lane["mu2_tensors"].as_array() {
                for block in extra {
                    add(&block["tensor"], block["power"].as_u64().unwrap() as u32);
                }
            }
            CaseLane {
                shifts,
                masses,
                scale: number(&lane["mu_squared"]),
                s_matrix,
                terms,
            }
        })
        .collect();
    Case {
        name: value["name"].as_str().unwrap().to_string(),
        n,
        rank,
        lanes,
    }
}

/// `[eps0, eps-1, eps-2]` per lane from a reference file's decimal strings.
fn load_reference(path: &Path) -> Vec<[Complex<Float>; 3]> {
    read(path)["lanes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|lane| {
            ["eps0", "eps_m1", "eps_m2"].map(|k| {
                let z = &lane["laurent_decimal"][k];
                let part = |v: &Value| Float::parse(v.as_str().unwrap(), Some(BITS + 64)).unwrap();
                Complex::new(part(&z[0]), part(&z[1]))
            })
        })
        .collect()
}

/// The generic engine on every lane of a case in Float(256), exact masters.
fn evaluate_case(case: &Case) -> Vec<[Complex<Float>; 5]> {
    let p = Float::new(BITS);
    let count = case.lanes.len();
    // One parameter per distinct term, zero where a lane lacks it.
    let mut keys: Vec<([u32; 4], u32)> = Vec::new();
    for lane in &case.lanes {
        for (key, _) in &lane.terms {
            if !keys.contains(key) {
                keys.push(*key);
            }
        }
    }
    let (expression, parameters) = rank_n2_support::polynomial(&keys);
    let mut numerator =
        PolynomialNumerator::new(expression, variables(), parameters, p.clone()).unwrap();
    assert_eq!(numerator_rank(&numerator), case.rank);
    let mut values = vec![fc(C64::ZERO); count * keys.len()];
    for (l, lane) in case.lanes.iter().enumerate() {
        for (key, c) in &lane.terms {
            let k = keys.iter().position(|x| x == key).unwrap();
            values[l * keys.len() + k] = values[l * keys.len() + k].clone() + &fc(*c);
        }
    }
    numerator.set_parameter_batch(&values, count).unwrap();
    let shifts: Vec<FourMomentum<Complex<Float>>> = case
        .lanes
        .iter()
        .flat_map(|lane| lane.shifts.iter().map(|v| FourMomentum::from(v.map(float))))
        .collect();
    let masses: Vec<_> = case
        .lanes
        .iter()
        .flat_map(|lane| lane.masses.iter().map(|&m| fc(m)))
        .collect();
    let matrix: Vec<_> = case
        .lanes
        .iter()
        .flat_map(|lane| lane.s_matrix.iter().map(|&s| float(s)))
        .collect();
    let scales: Vec<_> = case.lanes.iter().map(|lane| float(lane.scale)).collect();
    // Integrated box coefficients need no mu² sampling scale.
    let sampling: Vec<_> = case.lanes.iter().map(|_| float(1.)).collect();
    let mut amplitude = SymbolicaAmplitudeKernels::new(case.n, case.rank, p.clone())
        .unwrap()
        .into_amplitude(count)
        .unwrap();
    let mut low = SymbolicaCuts::new(p.clone()).unwrap();
    let mut high = SymbolicaHighCuts::new(p.clone()).unwrap();
    let mut masters = SymbolicaMasters::new(p.clone());
    amplitude
        .evaluate_batch_with_inputs(
            AmplitudeInput {
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: &scales,
                sampling_scales: &sampling,
                s_matrix: Some(&matrix),
            },
            &mut numerator,
            &mut low,
            &mut high,
            &mut masters,
        )
        .unwrap();
    for (lane, status) in amplitude.statuses().iter().enumerate() {
        assert_eq!(
            *status,
            EvaluationStatus::Success,
            "{} lane {lane}",
            case.name
        );
    }
    amplitude.results().to_vec()
}

fn numerator_rank(numerator: &PolynomialNumerator<Float>) -> usize {
    numerator.rank()
}

const SMOKE_CASES: [&str; 7] = [
    "rn2-int-complex-n1-r3-mu2",
    "rn2-int-complex-n2-r4",
    "rn2-int-complex-n2-r4-mu2",
    "rn2-int-complex-n3-r5",
    "rn2-int-complex-n3-r5-mu2",
    "rn2-int-complex-n4-r6",
    "rn2-int-complex-n4-r6-mu2",
];

/// Every smoke lane must match the independent 60-digit reduction to
/// <= 1e-25 (scaled). The COLLIER references are limited by COLLIER's own
/// quad-precision tensor integrals (about 1e-17..1e-14 on some lanes, its
/// tadpole coefficients in particular); the engine must agree with them as
/// well as the independent reduction does, to within 1e-25.
#[test]
fn rank_n2_reference_smoke() {
    let hpref = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/rank_n2/review/hpref");
    let mut report = Vec::new();
    let (mut worst_red, mut worst_excess) = (0_f64, 0_f64);
    for name in SMOKE_CASES {
        let collier = if name.ends_with("-mu2") {
            hpref.join(format!("{name}.json"))
        } else {
            data(&format!("hpref-{name}.json"))
        };
        let started = std::time::Instant::now();
        let case = load_case(&data(&format!("{name}.json")));
        let results = evaluate_case(&case);
        let collier = load_reference(&collier);
        let red = load_reference(&data(&format!("red-{name}.json")));
        assert_eq!(collier.len(), results.len());
        assert_eq!(red.len(), results.len());
        for (lane, ours) in results.iter().enumerate() {
            let against_red = laurent_error(ours, &red[lane]);
            let against_collier = laurent_error(ours, &collier[lane]);
            let red_against_collier = laurent_error(&red[lane], &collier[lane]);
            worst_red = worst_red.max(against_red);
            worst_excess = worst_excess.max(against_collier - red_against_collier);
            report.push(format!(
                "{name} lane {lane}: independent reduction {against_red:.2e}; COLLIER hp \
                 {against_collier:.2e} (independent reduction vs COLLIER {red_against_collier:.2e})"
            ));
        }
        report.push(format!("{name}: {:.1}s", started.elapsed().as_secs_f64()));
    }
    eprintln!("RANK_N2_SMOKE\n{}", report.join("\n"));
    assert!(
        worst_red <= 1e-25,
        "worst difference from the independent reduction {worst_red:e}"
    );
    assert!(
        worst_excess <= 1e-25,
        "COLLIER difference exceeds the independent reduction's by {worst_excess:e}"
    );
}

// ---------------------------------------------------------------------------
// Pinch identity.

#[test]
fn rank_n2_pinch_identity() {
    let p = Float::new(BITS);
    let mut report = Vec::new();
    let mut worst = 0_f64;
    for n in 2..=4usize {
        // M: dense rank n with mu² terms over the n - 1 remaining lines.
        let problem = Problem::new(n, n, 0x9_1c4 + n as u64, true, 1);
        let lanes = &problem.lanes[..2];
        for j in 0..n {
            let (m_expression, m_parameters) =
                (problem.expression.clone(), problem.parameters.clone());
            let v: Vec<_> = (0..4).map(|i| parse(&format!("v{i}"))).collect();
            let q = variables();
            let shifted: Vec<_> = (0..4).map(|i| &q[i] + &v[i]).collect();
            let d = &shifted[0] * &shifted[0]
                - &shifted[1] * &shifted[1]
                - &shifted[2] * &shifted[2]
                - &shifted[3] * &shifted[3]
                - parse("mj")
                - &q[4];
            let mut parameters = m_parameters.clone();
            parameters.extend(v.iter().cloned());
            parameters.push(parse("mj"));
            let mut full =
                PolynomialNumerator::new(d * &m_expression, variables(), parameters, p.clone())
                    .unwrap();
            let mut pinched =
                PolynomialNumerator::new(m_expression, variables(), m_parameters, p.clone())
                    .unwrap();
            assert_eq!(full.rank(), n + 2);
            assert_eq!(pinched.rank(), n);
            let width = problem.parameters.len();
            let mut full_values = Vec::new();
            let mut pinched_values = Vec::new();
            for (l, lane) in lanes.iter().enumerate() {
                let coefficients: Vec<_> = problem.coefficients[l * width..(l + 1) * width]
                    .iter()
                    .map(|&z| fc(z))
                    .collect();
                full_values.extend(coefficients.iter().cloned());
                full_values.extend(lane.shifts[j].iter().map(|&x| float(x)));
                full_values.push(fc(lane.masses[j]));
                pinched_values.extend(coefficients);
            }
            full.set_parameter_batch(&full_values, lanes.len()).unwrap();
            pinched
                .set_parameter_batch(&pinched_values, lanes.len())
                .unwrap();
            let run = |numerator: &mut PolynomialNumerator<Float>, keep: &dyn Fn(usize) -> bool| {
                let lines: Vec<usize> = (0..n).filter(|&i| keep(i)).collect();
                let count = lanes.len();
                let shifts: Vec<FourMomentum<Complex<Float>>> = lanes
                    .iter()
                    .flat_map(|lane| {
                        lines
                            .iter()
                            .map(|&i| FourMomentum::from(lane.shifts[i].map(float)))
                    })
                    .collect();
                let masses: Vec<_> = lanes
                    .iter()
                    .flat_map(|lane| lines.iter().map(|&i| fc(lane.masses[i])))
                    .collect();
                let scales: Vec<_> = (0..count).map(|i| float([1.0, 2.5][i])).collect();
                let sampling: Vec<_> = (0..count).map(|_| float(1.)).collect();
                let mut amplitude =
                    SymbolicaAmplitudeKernels::new(lines.len(), numerator.rank(), p.clone())
                        .unwrap()
                        .into_amplitude(count)
                        .unwrap();
                let mut low = SymbolicaCuts::new(p.clone()).unwrap();
                let mut high = SymbolicaHighCuts::new(p.clone()).unwrap();
                let mut masters = SymbolicaMasters::new(p.clone());
                amplitude
                    .evaluate_batch_with_inputs(
                        AmplitudeInput {
                            shifts: &shifts,
                            masses_squared: &masses,
                            scales_squared: &scales,
                            sampling_scales: &sampling,
                            s_matrix: None,
                        },
                        numerator,
                        &mut low,
                        &mut high,
                        &mut masters,
                    )
                    .unwrap();
                for status in amplitude.statuses() {
                    assert_eq!(*status, EvaluationStatus::Success, "n={n} j={j}");
                }
                amplitude.results().to_vec()
            };
            let left = run(&mut full, &|_| true);
            let right = run(&mut pinched, &|i| i != j);
            for (lane, (a, b)) in left.iter().zip(&right).enumerate() {
                let error = laurent_error(a, b);
                worst = worst.max(error);
                report.push(format!("n={n} j={j} lane={lane}: {error:.3e}"));
            }
        }
    }
    eprintln!("RANK_N2_PINCH\n{}", report.join("\n"));
    assert!(worst <= 1e-50, "worst pinch difference {worst:e}");
}

// ---------------------------------------------------------------------------
// Scalar C0/D0 values for the independent reference reduction.

/// Exact Float(256) C0 and D0 (oneloopmaster through `SymbolicaMasters`) of
/// every three- and four-line subset of every smoke lane, written to
/// `$NINJARS_RANK_N2_MASTERS_OUT` as decimal strings. They feed the review's
/// independent reduction (`tools/rank_n2/review/red.py`, 60 digits), which
/// produced `tests/rank_n2_data/red-*.json`; see `tests/rank_n2_data/README.md`.
#[test]
#[ignore = "generates reference inputs"]
fn rank_n2_dump_scalar_masters() {
    use ninjars::{MasterProvider, MasterRequest, MasterStatus};
    let out = std::env::var("NINJARS_RANK_N2_MASTERS_OUT").unwrap();
    let p = Float::new(BITS);
    let mut masters = SymbolicaMasters::new(p.clone());
    let mut document = serde_json::Map::new();
    for name in SMOKE_CASES {
        let case = load_case(&data(&format!("{name}.json")));
        let mut lanes = Vec::new();
        for lane in &case.lanes {
            let n = case.n;
            let s = |i: usize, j: usize| float(lane.s_matrix[i * n + j]);
            let mut entries = serde_json::Map::new();
            for size in [3usize, 4] {
                if size > n {
                    continue;
                }
                let mut subset: Vec<usize> = (0..size).collect();
                loop {
                    let (integral, invariants) = if size == 3 {
                        let [a, b, c] = [subset[0], subset[1], subset[2]];
                        (MasterIntegral::C0, vec![s(b, a), s(c, b), s(a, c)])
                    } else {
                        let [a, b, c, d] = [subset[0], subset[1], subset[2], subset[3]];
                        (
                            MasterIntegral::D0,
                            vec![s(b, a), s(c, b), s(d, c), s(a, d), s(c, a), s(d, b)],
                        )
                    };
                    let masses: Vec<_> = subset.iter().map(|&i| fc(lane.masses[i])).collect();
                    let mut output = vec![[fc(C64::ZERO), fc(C64::ZERO), fc(C64::ZERO)]];
                    let mut status = [MasterStatus::Success];
                    masters
                        .evaluate_batch(
                            MasterRequest {
                                integral,
                                invariants: &invariants,
                                masses_squared: &masses,
                                scales_squared: &[float(lane.scale)],
                            },
                            &mut output,
                            &mut status,
                        )
                        .unwrap();
                    assert_eq!(status, [MasterStatus::Success]);
                    let key = subset
                        .iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    let value: Vec<_> = output[0]
                        .iter()
                        .map(|z| serde_json::json!([z.re.to_string(), z.im.to_string()]))
                        .collect();
                    entries.insert(key, serde_json::json!(value));
                    let Some(i) = (0..size).rev().find(|&i| subset[i] < n - size + i) else {
                        break;
                    };
                    subset[i] += 1;
                    for k in i + 1..size {
                        subset[k] = subset[k - 1] + 1;
                    }
                }
            }
            lanes.push(Value::Object(entries));
        }
        document.insert(name.to_string(), Value::Array(lanes));
    }
    std::fs::write(
        out,
        serde_json::to_string_pretty(&Value::Object(document)).unwrap(),
    )
    .unwrap();
}

// ---------------------------------------------------------------------------
// Native kernel selection.

/// `with_native_kernels` builds at rank n+2: the catalogued T2/T3 denominators
/// and source projections stay native, the uncatalogued degree-4
/// triangle-to-bubble correction keeps its interpreter body (and is reported
/// so), and the results agree with the fully interpreted engine.
#[cfg(feature = "native-kernels")]
#[test]
fn rank_n2_native_kernels_keep_the_interpreted_correction() {
    use ninjars::{BoxCoefficientMode, BoxMuMode};
    use ninjars_symbolica::{KernelBackend, KernelFamily};
    use rank_n2_support::{PolynomialMasters, options, reduce_with};
    for n in 2..=4usize {
        let selection = SymbolicaAmplitudeKernels::new(n, n + 2, 0f64)
            .unwrap()
            .with_native_kernels()
            .kernel_backends();
        for entry in selection {
            let expected = match entry.family {
                KernelFamily::T2Denominators
                | KernelFamily::T3Denominators
                | KernelFamily::BubbleSourceProjection
                | KernelFamily::TadpoleSourceProjection => KernelBackend::NativeRust,
                _ => KernelBackend::SymbolicaInterpreter,
            };
            assert_eq!(entry.backend, expected, "n={n} {:?}", entry.family);
        }
        let problem = Problem::new(n, n + 2, 0x4a7 + n as u64, true, 1);
        let mode = options(BoxCoefficientMode::Integrated, BoxMuMode::Expansion);
        let mut numerator = problem.numerator(&0f64);
        let interpreted = reduce_with(
            &problem,
            &mut numerator,
            &0f64,
            mode,
            &mut PolynomialMasters,
            |k| k,
        )
        .results()
        .to_vec();
        let native = reduce_with(
            &problem,
            &mut numerator,
            &0f64,
            mode,
            &mut PolynomialMasters,
            |k| k.with_native_kernels(),
        )
        .results()
        .to_vec();
        for (a, b) in native.iter().zip(&interpreted) {
            let error = laurent_error(a, b);
            assert!(error <= 1e-10, "n={n}: {error:e}");
        }
    }
}
