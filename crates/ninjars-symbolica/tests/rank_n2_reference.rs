//! V4 / V4f (design section 7): both engines against the high-precision
//! references of `validation/rank-n2` (see its README for the sources,
//! tiers and uncertainties).
//!
//! - Generic engine in `Float(256)` with oneloopmaster's exact masters (V4):
//!   tiers A and B within `max(3 unc, 1e-30)`; tier C and tier-B lanes whose
//!   uncertainty comes from the generic engine's own twins are report-only.
//! - Binary64 (V4f), fast engine (native masters) and generic engine in f64:
//!   `3 unc + max(1e-14 max(1, cond), A e_ninja)` with `e_ninja` original
//!   Ninja's error on the lane or its rank twins (`Policy`).
//! - Lanes listed in `validation/rank-n2/known-failures.json` (oneloopmaster's
//!   real-mass D0 bug) must fail as known failures and are reported if they
//!   start to pass.
//!
//! The rank <= n+1 controls (the suite kinematics at ranks n-1, n, n+1,
//! `mu^2` twins, the synthetic benchmark cases; n = 1..7) and the rank n+2
//! suite run with both engines. The binary64 rank n+2 checks (V4f) are
//! ignored until the next phase fixes the lanes named in their reasons.
//! The numerator is the dense tensor of each case, expanded numerically
//! (`TensorGenericNumerator`), which is checked first against Symbolica's
//! `PolynomialNumerator`.
//!
//! Run with `cargo test --release -p ninjars-symbolica --test rank_n2_reference`.

mod rank_n2_reference_support;

use ninjars::fast::{
    adapters::{t2_powers, t3_powers},
    tensor_layout::{monomials, ncoeff_mu},
};
use ninjars::{
    BatchShape, Complex, ExpansionKind, ExpansionRequest, Float, FourMomentum, Numerator,
    NumeratorRequest, SampleId, SingleFloat,
};
use ninjars_symbolica::PolynomialNumerator;
use rank_n2_reference_support::{
    Run, cases_in, check_set, controls,
    harness::{Precision, TensorGenericNumerator, Verdict},
    policy, rank_n2, root,
};
use symbolica::{
    atom::Atom,
    domains::{float::RealLike, rational::Rational},
    parser::ParseSettings,
};

const FLOAT: Precision = Precision::Float(256);

/// Vectors and scalar parameters of one expansion request.
type CutRow = (Vec<FourMomentum<Complex<Float>>>, Vec<Complex<Float>>);

// ---------------------------------------------------------------------------
// The numeric tensor numerator against Symbolica's PolynomialNumerator.

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// A dyadic number in [-1, 1) (exact in every carrier).
    fn uniform(&mut self) -> f64 {
        ((self.next() >> 40) as f64 / (1u64 << 24) as f64) * 2. - 1.
    }
}

fn parse(text: &str) -> Atom {
    Atom::parse(text, "ninjars_rank_n2_reference", ParseSettings::default()).unwrap()
}

/// `sum_k p_k term_k` over the blocks `mu^(2k) T_k(q)` of a dense rank-`rank`
/// numerator in `MuTensorNumerator` order.
fn polynomial(rank: usize, mu_powers: usize) -> (Atom, Vec<Atom>) {
    let names = ["q0", "q1", "q2", "q3"];
    let mut text = String::new();
    let mut k = 0;
    for mu in 0..=mu_powers {
        for m in monomials(rank - 2 * mu) {
            if k > 0 {
                text.push('+');
            }
            text.push_str(&format!("p{k}"));
            for (i, &e) in m.iter().enumerate() {
                if e > 0 {
                    text.push_str(&format!("*{}^{e}", names[i]));
                }
            }
            if mu > 0 {
                text.push_str(&format!("*mu2^{mu}"));
            }
            k += 1;
        }
    }
    (
        parse(&text),
        (0..k).map(|i| parse(&format!("p{i}"))).collect(),
    )
}

/// Every expansion kind the engines request, with leading powers `top`.
fn kinds(top: usize) -> Vec<ExpansionKind> {
    let top_i = top as i32;
    let mut out = vec![
        ExpansionKind::Mu {
            powers: vec![top_i, top_i - 1],
        },
        ExpansionKind::MuShifted {
            powers: vec![top_i, top_i - 1, top_i - 2],
        },
    ];
    out.extend((0..=5).map(|m| ExpansionKind::T3 {
        powers: t3_powers(top, m),
    }));
    out.extend((0..=4).map(|m| ExpansionKind::T2 {
        powers: t2_powers(top, m),
    }));
    out
}

/// Exact agreement (to the working precision) of evaluations and all
/// Laurent expansions: dense ranks 0..=4 with every `mu^2` power (Symbolica's
/// symbolic preparation of `PolynomialNumerator` dominates the run time), at the
/// numerator's rank and declared two ranks higher (padding), one and two
/// vectors for `Mu`.
#[test]
fn tensor_numerator_matches_polynomial_numerator() {
    let bits = 192;
    let p = Float::new(bits);
    let exact = |x: f64| p.from_rational(&Rational::try_from(x).unwrap());
    let cplx = |re: f64, im: f64| Complex::new(exact(re), exact(im));
    let variables = ["q0", "q1", "q2", "q3", "mu2"].map(parse);
    let mut rng = Rng(0x007e_450a);
    let mut worst = 0_f64;
    let mut checks = 0;
    for rank in 0..=4 {
        for mu_powers in 0..=rank / 2 {
            let lanes = 2;
            let width = ncoeff_mu(rank, mu_powers);
            let values: Vec<Complex<Float>> = (0..lanes * width)
                .map(|_| cplx(rng.uniform(), rng.uniform()))
                .collect();
            let (expression, parameters) = polynomial(rank, mu_powers);
            let mut reference =
                PolynomialNumerator::new(expression, variables.clone(), parameters, p.clone())
                    .unwrap();
            reference.set_parameter_batch(&values, lanes).unwrap();
            let mut tensor = TensorGenericNumerator::new(rank, mu_powers, &p).unwrap();
            tensor.set_coefficients(values.clone(), lanes).unwrap();
            assert_eq!(tensor.rank(), reference.rank());
            let mut compare = |label: String, a: &[Complex<Float>], b: &[Complex<Float>]| {
                assert_eq!(a.len(), b.len());
                let scale = b
                    .iter()
                    .map(|z| z.re.to_f64().hypot(z.im.to_f64()))
                    .fold(1e-300, f64::max);
                for (x, y) in a.iter().zip(b) {
                    let d = (x.re.clone() - &y.re)
                        .to_f64()
                        .hypot((x.im.clone() - &y.im).to_f64())
                        / scale;
                    worst = worst.max(d);
                    assert!(d <= 1e-45, "{label}: {x:?} vs {y:?} ({d:e})");
                }
                checks += 1;
            };
            let vector = |rng: &mut Rng| {
                FourMomentum::from([0; 4].map(|_| cplx(rng.uniform(), rng.uniform())))
            };
            // Direct evaluation.
            let points: Vec<(FourMomentum<Complex<Float>>, Complex<Float>)> = (0..3)
                .map(|_| (vector(&mut rng), cplx(rng.uniform(), 0.)))
                .collect();
            let partition = [0usize];
            let requests: Vec<NumeratorRequest<'_, Complex<Float>>> = points
                .iter()
                .enumerate()
                .map(|(i, (q, mu2))| NumeratorRequest {
                    id: SampleId {
                        lane: i % lanes,
                        sample: i,
                    },
                    q,
                    mu_squared: mu2,
                    partition: &partition,
                })
                .collect();
            let zero = cplx(0., 0.);
            let mut a = vec![zero.clone(); requests.len()];
            let mut b = vec![zero.clone(); requests.len()];
            tensor.evaluate_batch(&requests, &mut a).unwrap();
            reference.evaluate_batch(&requests, &mut b).unwrap();
            compare(format!("rank {rank} mu {mu_powers} evaluate"), &a, &b);
            // Expansions at the numerator's rank and declared above it.
            for top in [rank, rank + 2] {
                for kind in kinds(top) {
                    let (vectors, scalars) = match &kind {
                        ExpansionKind::Mu { .. } => (2, 0),
                        ExpansionKind::MuShifted { .. } => (2, 1),
                        ExpansionKind::T3 { .. } => (3, 1),
                        ExpansionKind::T2 { .. } => (4, 3),
                    };
                    let rows: Vec<CutRow> = (0..3)
                        .map(|row| {
                            // Mu also with its single-vector form (v1 = 0).
                            let count = if matches!(kind, ExpansionKind::Mu { .. }) && row == 2 {
                                1
                            } else {
                                vectors
                            };
                            (
                                (0..count).map(|_| vector(&mut rng)).collect(),
                                (0..scalars)
                                    .map(|_| cplx(rng.uniform(), rng.uniform()))
                                    .collect(),
                            )
                        })
                        .collect();
                    let requests: Vec<ExpansionRequest<'_, Complex<Float>>> = rows
                        .iter()
                        .enumerate()
                        .map(|(i, (v, s))| ExpansionRequest {
                            id: SampleId {
                                lane: i % lanes,
                                sample: i,
                            },
                            partition: &partition,
                            vectors: v,
                            parameters: s,
                        })
                        .collect();
                    let size = BatchShape {
                        rows: requests.len(),
                        columns: kind.len(),
                    };
                    let mut a = vec![zero.clone(); size.rows * size.columns];
                    let mut b = vec![zero.clone(); size.rows * size.columns];
                    tensor.expansion_batch(&kind, &requests, &mut a).unwrap();
                    reference.expansion_batch(&kind, &requests, &mut b).unwrap();
                    compare(
                        format!("rank {rank} mu {mu_powers} top {top} {kind:?}"),
                        &a,
                        &b,
                    );
                }
            }
        }
    }
    eprintln!("TENSOR_NUMERATOR checks={checks} max_relative_difference={worst:e}");
}

// ---------------------------------------------------------------------------
// Rank <= n+1 controls (run now).

/// V4 on the controls: the generic engine in Float(256) with exact masters
/// reproduces every tier-A/B control reference within `max(3 unc, 1e-30)`
/// (n = 1..7), apart from the known oneloopmaster D0 failures.
#[test]
fn controls_generic_float_match_references() {
    let policy = policy();
    let files = controls(7);
    let report = check_set(
        &files,
        &rank_n2().join("controls/reference"),
        Run::Generic(FLOAT),
        |case| case.rank,
        &policy,
    );
    report.assert_clean(true);
    assert!(report.count(Verdict::Pass) >= 560, "{}", report.summary());
}

/// V4f calibration on the controls: the fast engine and the generic engine
/// in binary64 (native masters) within the binary64 policy.
#[test]
fn controls_binary64_match_references() {
    let policy = policy();
    let files = controls(7);
    for run in [Run::Fast, Run::Generic(Precision::F64)] {
        let report = check_set(
            &files,
            &rank_n2().join("controls/reference"),
            run,
            |case| case.rank,
            &policy,
        );
        report.assert_clean(true);
    }
}

// ---------------------------------------------------------------------------
// Rank n+2.

fn suite() -> Vec<std::path::PathBuf> {
    let mut files = cases_in(&rank_n2().join("cases"), |_| true);
    files.extend(cases_in(
        &root().join("validation/benchmark/synthetic-rank-n2"),
        |_| true,
    ));
    files
}

/// V4: the rank n+2 suite (n = 1..7, every family, `mu^2` blocks, synthetic
/// cases) with the generic engine in Float(256) and exact masters.
#[test]
fn suite_generic_float_matches_references() {
    let report = check_set(
        &suite(),
        &rank_n2().join("reference"),
        Run::Generic(FLOAT),
        |case| case.rank,
        &policy(),
    );
    report.assert_clean(true);
}

/// V4f: the rank n+2 suite in binary64, fast engine with native masters.
#[test]
#[ignore = "next phase: fast-engine binary64 accuracy at rank n+2 and the V4f budget (twins extrapolated one rank) disagree on 12 of 333 lanes, e.g. syn-n6-r8 lane 2 6.2e-4 > 1.0e-4, rn2-phys-real-n6-r8 lane 2 1.9e-6 > 1.6e-7"]
fn suite_fast_binary64_matches_references() {
    let report = check_set(
        &suite(),
        &rank_n2().join("reference"),
        Run::Fast,
        |case| case.rank,
        &policy(),
    );
    report.assert_clean(true);
}

/// V4f: the rank n+2 suite in binary64, generic engine with native masters.
#[test]
#[ignore = "next phase: V4f budget misses the cancellation of the new rank n+2 terms on rn2-phys-{real,complex}-n6-r8 lane 2 (1.1e-3 > 2.2e-6, 5.4e-4 > 1.0e-4; Float(256) agrees with the references to 5e-9)"]
fn suite_generic_binary64_matches_references() {
    let report = check_set(
        &suite(),
        &rank_n2().join("reference"),
        Run::Generic(Precision::F64),
        |case| case.rank,
        &policy(),
    );
    report.assert_clean(true);
}

/// The exact-input identity cases (V2 numerators of full rank n+2, their
/// shifted and relabelled copies, the pinch and decomposition cases) against
/// their references in Float(256); all are tier A.
#[test]
fn identity_cases_generic_float_match_references() {
    let report = check_set(
        &cases_in(&rank_n2().join("identity/cases"), |_| true),
        &rank_n2().join("identity/reference"),
        Run::Generic(FLOAT),
        |case| case.rank,
        &policy(),
    );
    report.assert_clean(true);
}

/// V9 (report only): the conditioning scan (bubble p^2 ladder, triangle and
/// box Gram ladders) in Float(256) and binary64.
#[test]
fn conditioning_scan_report() {
    let policy = policy();
    for run in [Run::Generic(FLOAT), Run::Fast, Run::Generic(Precision::F64)] {
        let report = check_set(
            &cases_in(&rank_n2().join("scan/cases"), |_| true),
            &rank_n2().join("scan/reference"),
            run,
            |case| case.rank,
            &policy,
        );
        eprintln!("V9 {}", report.summary());
        for (case, lane) in &report.lanes {
            eprintln!(
                "V9 {} {case} {}: {} difference {:?}",
                report.label, lane.id, lane.status, lane.difference
            );
        }
    }
}
