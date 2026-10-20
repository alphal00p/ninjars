//! `ninjars case-check` on bench cases with high-precision and original-Ninja
//! references (`validation/rank-n2`, `validation/benchmark/ninja`).
//!
//! - On rank <= n+1 cases its fast-engine reference check is exactly
//!   `case-bench`'s (same batch, same lanes, same numbers).
//! - The controls pass with both engines (tier-C lanes report-only), also
//!   declared at a higher rank.
//! - Failing lanes, stale known failures and the engine and precision scope
//!   of known-failure entries are reported.
//! - `--bits B --out-dir DIR` is the `GENERIC_RUNNER` interface of
//!   `validation/rank-n2/harness/run_suite.sh`.
//! - A sample of the rank n+2 suite passes with both engines.

use serde_json::Value;
use std::{
    path::PathBuf,
    process::{Command, Output},
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run(command: &str, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ninjars"))
        .current_dir(root())
        .arg(command)
        .args(args)
        .output()
        .unwrap()
}

fn value(output: Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

const CONTROLS: &str = "validation/rank-n2/controls/reference";
const KNOWN: &str = "validation/rank-n2/known-failures.json";

/// Rank <= n+1 cases: v1 and v2 (`mu^2` blocks), n = 1..7, every family, with
/// a high-precision reference; and benchmark cases with original-Ninja ones.
const CASES: [(&str, &str); 12] = [
    (
        "validation/rank-n2/controls/cases/rn2-int-complex-n1-r2.json",
        CONTROLS,
    ),
    (
        "validation/rank-n2/controls/cases/rn2-bub0-mixed-n2-r3.json",
        CONTROLS,
    ),
    (
        "validation/rank-n2/controls/cases/rn2-int-real-n2-r3-mu2.json",
        CONTROLS,
    ),
    (
        "validation/rank-n2/controls/cases/rn2-irsoft-real-n3-r4.json",
        CONTROLS,
    ),
    (
        "validation/rank-n2/controls/cases/rn2-shifted-mixed-n4-r5-mu2.json",
        CONTROLS,
    ),
    (
        "validation/rank-n2/controls/cases/rn2-phys-complex-n5-r6.json",
        CONTROLS,
    ),
    (
        "validation/rank-n2/controls/cases/rn2-int-mixed-n6-r7.json",
        CONTROLS,
    ),
    (
        "validation/rank-n2/controls/cases/rn2-int-complex-n7-r8.json",
        CONTROLS,
    ),
    (
        "validation/rank-n2/controls/cases/rn2-int-real-n7-r8-mu2.json",
        CONTROLS,
    ),
    ("validation/benchmark/synthetic/syn-n5-r6.json", CONTROLS),
    (
        "validation/benchmark/synthetic/syn-n4-r5.json",
        "validation/benchmark/ninja",
    ),
    (
        "validation/benchmark/cases/gg-hhh-top-pentagon-26.json",
        "validation/benchmark/ninja",
    ),
];

fn lanes(case: &str) -> usize {
    let text = std::fs::read_to_string(root().join(case)).unwrap();
    serde_json::from_str::<Value>(&text).unwrap()["lanes"]
        .as_array()
        .unwrap()
        .len()
}

/// The fast-engine reference check of `case-check` equals `case-bench`'s at
/// the same batch width.
#[test]
fn reference_check_equals_case_bench() {
    for (case, references) in CASES {
        for width in [lanes(case), 32] {
            let width = width.to_string();
            let bench = value(run(
                "case-bench",
                &[
                    case,
                    "--widths",
                    &width,
                    "--blocks",
                    "1",
                    "--min-block-ms",
                    "0",
                    "--reference-dir",
                    references,
                ],
            ));
            let check = value(run(
                "case-check",
                &[
                    case,
                    "--engines",
                    "fast",
                    "--width",
                    &width,
                    "--reference-dir",
                    references,
                    "--no-fail",
                ],
            ));
            let expected = &bench["cases"][0]["per_point_timing"][0]["reference_check"];
            let run = &check["cases"][0]["runs"][0];
            assert!(expected["lanes_checked"].as_u64().unwrap() > 0, "{case}");
            assert_eq!(
                &run["case_bench_reference_check"], expected,
                "{case} width {width}"
            );
            assert_eq!(run["repeats_bitwise"], true, "{case} width {width}");
            // The per-lane differences are those of the summary.
            let worst = run["lanes"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|l| l["scaled_difference"].as_f64())
                .fold(0., f64::max);
            let summary = expected["max_scaled_laurent_difference"].as_f64().unwrap();
            assert!(
                (worst - summary).abs() <= 1e-3 * summary,
                "{case}: lanes {worst:e} vs case-bench {summary:e}"
            );
        }
    }
}

fn verdicts(report: &Value) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    for case in report["cases"].as_array().unwrap() {
        for run in case["runs"].as_array().unwrap() {
            assert!(run["error"].is_null(), "{}: {}", case["case"], run["error"]);
            let label = format!(
                "{}/{}",
                run["engine"].as_str().unwrap(),
                run["precision"].as_str().unwrap()
            );
            for lane in run["lanes"].as_array().unwrap() {
                out.push((
                    label.clone(),
                    lane["id"].as_str().unwrap().to_string(),
                    lane["verdict"].as_str().unwrap().to_string(),
                ));
            }
        }
    }
    out
}

/// Both engines (fast; generic in f64 and Float(256)) pass every checked
/// control lane, declared at the case rank and one rank higher. The lanes of
/// `rn2-int-real-n7-r8-mu2` whose references are COLLIER's (tier C, from the
/// oneloopmaster D0 bug fixed in 09b2c74) are report-only; lane 1 of
/// `rn2-int-mixed-n6-r7` (the D0 bug of finding 3) passes, in binary64 with
/// original Ninja's error corrected for OneLOop's D0 (`ninja_unreliable`).
#[test]
fn controls_pass_with_both_engines() {
    let cases: Vec<&str> = CASES
        .iter()
        .filter(|(_, r)| *r == CONTROLS)
        .map(|(c, _)| *c)
        .collect();
    for declare in [None, Some("n+1")] {
        let mut args = vec![
            "--engines",
            "fast,generic",
            "--precision",
            "f64,float:256",
            "--reference-dir",
            CONTROLS,
            "--known-failures",
            KNOWN,
            "--twins",
            "validation/rank-n2/summary/controls.json",
            "--twins",
            "validation/rank-n2/summary/synthetic-twins.json",
            "--twins",
            "validation/rank-n2/summary/n7-controls.json",
        ];
        if let Some(rank) = declare {
            args.extend(["--declare-rank", rank]);
        }
        args.extend(cases.iter().copied());
        let report = value(run("case-check", &args));
        assert_eq!(report["failed_lanes"], 0);
        for (label, id, verdict) in verdicts(&report) {
            let tier_c =
                id == "rn2-int-real-n7-r8-mu2-lane0" || id == "rn2-int-real-n7-r8-mu2-lane1";
            let expected = if tier_c { "report-only" } else { "pass" };
            assert_eq!(verdict, expected, "{label} {id} ({declare:?})");
        }
        let corrected = report["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|case| case["case"] == "rn2-int-mixed-n6-r7")
            .flat_map(|case| case["runs"].as_array().unwrap())
            .filter(|run| run["precision"] == "f64")
            .map(|run| &run["lanes"][1])
            .collect::<Vec<_>>();
        assert_eq!(corrected.len(), 2);
        for lane in corrected {
            assert_eq!(lane["id"], "rn2-int-mixed-n6-r7-lane1");
            assert!(
                lane["reason"]
                    .as_str()
                    .unwrap()
                    .contains("without OneLOop's error"),
                "{}",
                lane["reason"]
            );
        }
    }
}

/// A copy of the control reference of `case` with the real finite part of
/// lane `lane` (and of its COLLIER value) scaled by `1 + 1e-6`, in a fresh
/// directory.
fn perturbed_reference(name: &str, case: &str, lane: usize) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let file = format!("{case}.json");
    let mut reference: Value =
        serde_json::from_str(&std::fs::read_to_string(root().join(CONTROLS).join(&file)).unwrap())
            .unwrap();
    let entry = &mut reference["lanes"][lane];
    let value = entry["eps0"][0].as_f64().unwrap() * (1. + 1e-6);
    entry["eps0"][0] = value.into();
    entry["laurent_decimal"]["eps0"][0] = format!("{value:e}").into();
    entry["collier_coli_quad_decimal"]["eps0"][0] = format!("{value:e}").into();
    std::fs::write(dir.join(file), reference.to_string()).unwrap();
    dir
}

/// A failing lane makes the command fail (and is reported as `fail`): every
/// engine and precision misses a reference moved by 1e-6.
#[test]
fn failures_are_reported() {
    let references = perturbed_reference("case-check-fail", "rn2-int-complex-n3-r4", 1);
    let output = run(
        "case-check",
        &[
            "validation/rank-n2/controls/cases/rn2-int-complex-n3-r4.json",
            "--engines",
            "fast,generic",
            "--precision",
            "f64,float:256",
            "--reference-dir",
            references.to_str().unwrap(),
        ],
    );
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["failed_lanes"], 3);
    for run in report["cases"][0]["runs"].as_array().unwrap() {
        let lanes = run["lanes"].as_array().unwrap();
        assert_eq!(lanes[0]["verdict"], "pass");
        assert_eq!(lanes[1]["verdict"], "fail");
        assert_eq!(lanes[2]["verdict"], "pass");
    }
}

/// A known failure limited to some engines and precisions (`engines`,
/// `precisions`) is judged as such only there: elsewhere the lane is judged
/// as any other, so a lane that fails only in those precisions does not make
/// the others exit non-zero, and the entry does not go stale in them.
#[test]
fn known_failure_scope() {
    let references = perturbed_reference("case-check-scope", "rn2-int-complex-n3-r4", 1);
    let dir = references.parent().unwrap().join("case-check-scope-known");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let known = dir.join("known-failures.json");
    // The moved lane differs from the engines' values by 5.8e-7 of its scale,
    // so it is a known failure at a known-failure tolerance of 1e-8.
    std::fs::write(
        &known,
        r#"{"format": "ninjars-known-failures-v1", "lanes": [{"id": "rn2-int-complex-n3-r4-lane1", "reason": "test entry", "engines": ["generic"], "precisions": ["float:256"]}]}"#,
    )
    .unwrap();
    let check = |engines: &str, precision: &str, references: &PathBuf| {
        run(
            "case-check",
            &[
                "validation/rank-n2/controls/cases/rn2-int-complex-n3-r4.json",
                "--engines",
                engines,
                "--precision",
                precision,
                "--reference-dir",
                references.to_str().unwrap(),
                "--known-failures",
                known.to_str().unwrap(),
                "--known-failure-tolerance",
                "1e-8",
            ],
        )
    };
    let verdict = |output: &Output| {
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        report["cases"][0]["runs"][0]["lanes"][1]["verdict"]
            .as_str()
            .unwrap()
            .to_string()
    };
    // In scope, the moved reference is a known failure (exit 0), and the
    // correct reference makes the entry stale (exit 1).
    let output = check("generic", "float:256", &references);
    assert!(output.status.success());
    assert_eq!(verdict(&output), "known-failure");
    let output = check("generic", "float:256", &root().join(CONTROLS));
    assert!(!output.status.success());
    assert_eq!(verdict(&output), "stale-known-failure");
    // Out of scope (the fast engine, the generic engine in f64), the lane is
    // judged as any other: it passes the correct reference without going
    // stale, and fails the moved one.
    for (engines, precision) in [("fast", "f64"), ("generic", "f64")] {
        let output = check(engines, precision, &root().join(CONTROLS));
        assert!(output.status.success(), "{engines} {precision}");
        assert_eq!(verdict(&output), "pass", "{engines} {precision}");
        let output = check(engines, precision, &references);
        assert!(!output.status.success(), "{engines} {precision}");
        assert_eq!(verdict(&output), "fail", "{engines} {precision}");
    }
}

/// A known-failure entry whose lane agrees with COLLIER's value (here a
/// lane without any upstream bug) is stale: it is reported as
/// `stale-known-failure` and fails the command, so that a fixed upstream bug
/// is noticed.
#[test]
fn stale_known_failures_are_reported() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("case-check-stale");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let known = dir.join("known-failures.json");
    std::fs::write(
        &known,
        r#"{"format": "ninjars-known-failures-v1", "lanes": [{"id": "rn2-int-complex-n3-r4-lane1", "reason": "test entry"}]}"#,
    )
    .unwrap();
    for engine in ["fast", "generic"] {
        let output = run(
            "case-check",
            &[
                "validation/rank-n2/controls/cases/rn2-int-complex-n3-r4.json",
                "--engines",
                engine,
                "--precision",
                "f64",
                "--reference-dir",
                CONTROLS,
                "--known-failures",
                known.to_str().unwrap(),
            ],
        );
        assert!(!output.status.success(), "{engine}");
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["failed_lanes"], 1, "{engine}");
        let lane = &report["cases"][0]["runs"][0]["lanes"][1];
        assert_eq!(lane["id"], "rn2-int-complex-n3-r4-lane1");
        assert_eq!(lane["verdict"], "stale-known-failure", "{engine}");
        assert!(
            lane["collier_difference"].as_f64().unwrap() <= 1e-6,
            "{engine}"
        );
    }
}

/// `--bits 256 --out-dir DIR` writes `DIR/<case>.json` with 45-digit
/// `lanes[].laurent_decimal` that reproduce the reference.
#[test]
fn generic_runner_interface() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("case-check-runner");
    let _ = std::fs::remove_dir_all(&dir);
    let case = "validation/rank-n2/controls/cases/rn2-int-complex-n3-r4.json";
    let output = run(
        "case-check",
        &["--bits", "256", "--out-dir", dir.to_str().unwrap(), case],
    );
    let report = value(output);
    assert_eq!(report["cases"][0]["runs"][0]["engine"], "generic");
    assert_eq!(report["cases"][0]["runs"][0]["precision"], "float:256");
    let written: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("rn2-int-complex-n3-r4.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(written["format"], "ninjars-generic-float-run-v1");
    assert_eq!(written["bits"], 256);
    let reference: Value = serde_json::from_str(
        &std::fs::read_to_string(root().join(CONTROLS).join("rn2-int-complex-n3-r4.json")).unwrap(),
    )
    .unwrap();
    for (lane, expected) in written["lanes"]
        .as_array()
        .unwrap()
        .iter()
        .zip(reference["lanes"].as_array().unwrap())
    {
        assert_eq!(lane["id"], expected["id"]);
        for key in ["eps0", "eps_m1", "eps_m2"] {
            for part in 0..2 {
                let text = lane["laurent_decimal"][key][part].as_str().unwrap();
                let got: f64 = text.parse().unwrap();
                assert!(got == 0. || text.len() > 40, "{text}");
                let want = expected[key][part].as_f64().unwrap();
                assert!(
                    (got - want).abs() <= 1e-15 * want.abs().max(1e-300) + 1e-300,
                    "{}: {key} {got} vs {want}",
                    lane["id"]
                );
            }
        }
    }
}

/// The rank n+2 suite through both engines (n = 1..7, a sample of every
/// family and of the `mu^2` blocks).
#[test]
fn rank_n2_cases_pass() {
    let cases = [
        "validation/rank-n2/cases/rn2-int-complex-n1-r3-mu2.json",
        "validation/rank-n2/cases/rn2-bub0-mixed-n2-r4.json",
        "validation/rank-n2/cases/rn2-irml-zero-n3-r5-mu2.json",
        "validation/rank-n2/cases/rn2-int-real-n4-r6-muall.json",
        "validation/rank-n2/cases/rn2-phys-complex-n5-r7.json",
        "validation/rank-n2/cases/rn2-shifted-mixed-n6-r8-mu2.json",
        "validation/rank-n2/cases/rn2-int-complex-n7-r9.json",
        "validation/rank-n2/cases/rn2-int-mixed-n7-r9.json",
    ];
    let mut args = vec![
        "--engines",
        "fast,generic",
        "--precision",
        "f64,float:256",
        "--reference-dir",
        "validation/rank-n2/reference",
        "--known-failures",
        KNOWN,
        "--twins",
        "validation/rank-n2/summary/controls.json",
        "--twins",
        "validation/rank-n2/summary/n7-controls.json",
    ];
    args.extend(cases);
    let report = value(run("case-check", &args));
    assert_eq!(report["failed_lanes"], 0);
    assert_eq!(report["case_errors"], 0);
}

/// `--stability-test` (generic engine, opt-in as for `reduce`) flags the rank
/// n+2 lanes whose new terms cancel in binary64: lane 1 of
/// rn2-phys-{real,complex}-n6-r8 (scaled differences 1.4e-6 real, 4.4e-5
/// complex). Real lane 1 cancels between the cut-constructible and rational
/// parts of single cuts. Lane 2 (1.3e-6 real, 6.0e-6 complex) cancelled
/// between cuts beyond half the bits only with the earlier eight-reference
/// bubble score (1.1e-3 and 5.4e-4); with the 24 references of
/// `cuts.rs::bubble_reference` it stays below, so the test passes it. Real
/// lane 2 is then a false negative: its 1.3e-6 exceeds its binary64 budget
/// (5.9e-7; the lane is report only, cond' > 1e6) because the remaining loss
/// is inside single cuts, which the test does not measure (TODO_NEXT_PLAN.md
/// 2.4). DoubleFloat keeps more than half of its bits on the same lanes and
/// passes.
#[test]
fn stability_test_flags_rank_n2_cancellations() {
    let statuses = |extra: &[&str]| {
        generic_statuses(
            extra,
            &[
                "validation/rank-n2/cases/rn2-phys-real-n6-r8.json",
                "validation/rank-n2/cases/rn2-phys-complex-n6-r8.json",
            ],
        )
    };
    let success = vec![vec!["Success".to_string(); 3]; 2];
    assert_eq!(statuses(&["--precision", "f64"]), success);
    assert_eq!(
        statuses(&["--precision", "f64", "--stability-test"]),
        [
            ["Success", "Unstable", "Success"],
            ["Success", "Unstable", "Success"]
        ]
    );
    assert_eq!(
        statuses(&["--precision", "double", "--stability-test"]),
        success
    );
}

/// Generic-engine lane statuses (`[case][lane]`) of `case-check` with `extra`.
fn generic_statuses(extra: &[&str], cases: &[&str]) -> Vec<Vec<String>> {
    let mut args = vec!["--engines", "generic", "--no-fail"];
    args.extend(extra);
    args.extend(cases);
    let report = value(run("case-check", &args));
    report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| {
            case["runs"][0]["lanes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|lane| lane["status"].as_str().unwrap().to_string())
                .collect()
        })
        .collect()
}

/// `--stability-test` measures the terms inside each cut contribution, not
/// only whole contributions (final review of the rank n+2 work, finding 4):
/// - `tests/data/repro-stability-test-miss-n6-r8.json` (`fr-light-zero-n6-r8`
///   lane 2) is 24 % wrong in generic binary64 with a vanishing amplitude:
///   cut-constructible and rational parts of single cuts cancel (about -0.086
///   and +0.086 to 1e-12), and the terms of its cuts reach 3e15 times the
///   result. Before 22c332c it was reported `Success`. DoubleFloat (error
///   1e-17, below its half-precision 1.1e-16) passes.
/// - `rn2-phys-complex-n6-r7` lane 1 (a rank n+1 control) is off by 3.5e-7 in
///   binary64. Its master-integral terms cancel inside cut contributions by
///   1.2e9 (2^26 = 6.7e7 is the binary64 limit), while the whole contributions
///   and their cut-constructible and rational parts pass. Lanes 0 and 2 (off
///   by 1.1e-10 and 1.0e-7, terms 7.9e5 and 2.5e7 times the result) pass.
#[test]
fn stability_test_flags_cancellation_inside_cut_contributions() {
    let cases = [
        "crates/ninjars-cli/tests/data/repro-stability-test-miss-n6-r8.json",
        "validation/rank-n2/controls/cases/rn2-phys-complex-n6-r7.json",
    ];
    let success = |lanes: usize| vec!["Success".to_string(); lanes];
    assert_eq!(
        generic_statuses(&["--precision", "f64"], &cases),
        [success(1), success(3)]
    );
    assert_eq!(
        generic_statuses(&["--precision", "f64", "--stability-test"], &cases),
        [vec!["Unstable"], vec!["Success", "Unstable", "Success"]]
    );
    assert_eq!(
        generic_statuses(&["--precision", "double", "--stability-test"], &cases),
        [success(1), success(3)]
    );
}
