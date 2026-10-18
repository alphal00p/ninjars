//! `ninjars case-check` on bench cases with high-precision and original-Ninja
//! references (`validation/rank-n2`, `validation/benchmark/ninja`).
//!
//! - On rank <= n+1 cases its fast-engine reference check is exactly
//!   `case-bench`'s (same batch, same lanes, same numbers).
//! - The controls pass with both engines (known failures excepted), also
//!   declared at a higher rank.
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
/// control lane, declared at the case rank and one rank higher.
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
            let expected_failure = id == "rn2-int-mixed-n6-r7-lane1"
                || id.starts_with("rn2-int-real-n7-r8-mu2-lane0")
                || id.starts_with("rn2-int-real-n7-r8-mu2-lane1");
            if expected_failure {
                assert_eq!(verdict, "known-failure", "{label} {id} ({declare:?})");
            } else {
                assert_eq!(verdict, "pass", "{label} {id} ({declare:?})");
            }
        }
    }
}

/// A failing lane makes the command fail (and is reported as `fail`).
#[test]
fn failures_are_reported() {
    let output = run(
        "case-check",
        &[
            "validation/rank-n2/controls/cases/rn2-int-mixed-n6-r7.json",
            "--reference-dir",
            CONTROLS,
        ],
    );
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["failed_lanes"], 1);
    let lane = &report["cases"][0]["runs"][0]["lanes"][1];
    assert_eq!(lane["verdict"], "fail");
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
