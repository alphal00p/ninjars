use serde_json::{Value, json};
use std::{
    path::PathBuf,
    process::{Command, Output},
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ninjars"))
        .current_dir(root())
        .arg("reduce")
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
fn component(row: &Value, coefficient: usize, part: usize) -> f64 {
    row["values"][coefficient][part]
        .as_str()
        .unwrap()
        .parse()
        .unwrap()
}
fn near(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-9 * (1. + expected.abs()),
        "{actual:.17e} != {expected:.17e}"
    );
}
fn original_checks(actual: &Value, fixture: &str) {
    let evidence: Value = serde_json::from_str(include_str!(
        "../../../validation/evidence/prepared-graphs.json"
    ))
    .unwrap();
    for check in evidence["checks"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|check| check["fixture"] == fixture)
    {
        let row = &actual["results"][check["lane"].as_u64().unwrap() as usize];
        assert_eq!(row["status"]["kind"], "success");
        for coefficient in 0..3 {
            for part in 0..2 {
                near(
                    component(row, coefficient, part),
                    check["reference"]["epsilon"][coefficient][part]
                        .as_f64()
                        .unwrap(),
                );
            }
        }
        for part in 0..2 {
            near(
                component(row, 0, part),
                component(row, 3, part) + component(row, 4, part),
            );
        }
    }
}

#[test]
fn bubble_and_triangle_match_original_ninja_with_batches_and_tails() {
    let fixture = "examples/graphs/bubble.json";
    let single = value(run(&[fixture, "--batch-size", "1"]));
    let tail = value(run(&[fixture, "--batch-size", "2", "--diagnostics"]));
    original_checks(&single, fixture);
    original_checks(&tail, fixture);
    assert_eq!(tail["results"].as_array().unwrap().len(), 3);
    assert_eq!(tail["results"][0]["cut_counts"]["bubbles"], 1);
    assert_eq!(tail["results"][0]["cut_counts"]["tadpoles"], 2);
    let fixture = "examples/graphs/triangle-reference.json";
    original_checks(&value(run(&[fixture])), fixture);
}

#[test]
fn double_float_and_arbitrary_precision_match_original_ninja() {
    for precision in ["double-float", "float"] {
        for fixture in [
            "examples/graphs/bubble.json",
            "examples/graphs/triangle-reference.json",
        ] {
            let actual = value(run(&[
                fixture,
                "--precision",
                precision,
                "--batch-size",
                "2",
            ]));
            original_checks(&actual, fixture);
            assert_eq!(actual["precision"]["carrier"], precision);
            assert!(actual["precision"]["bits"].as_u64().unwrap() >= 106);
        }
    }
}

#[cfg(feature = "native-kernels")]
#[test]
fn native_denominator_selection_matches_original_in_all_precisions() {
    for selection in ["native-denominators", "native"] {
        for precision in ["f64", "double-float", "float"] {
            for fixture in [
                "examples/graphs/bubble.json",
                "examples/graphs/triangle-reference.json",
            ] {
                let actual = value(run(&[
                    fixture,
                    "--precision",
                    precision,
                    "--batch-size",
                    "2",
                    "--reduction-kernels",
                    selection,
                ]));
                original_checks(&actual, fixture);
                let kernels = &actual["reduction_kernels"];
                assert_eq!(kernels["selection"], selection);
                for name in [
                    "bubble_source_projection",
                    "triangle_bubble_correction",
                    "tadpole_source_projection",
                ] {
                    assert_eq!(
                        kernels[name],
                        if selection == "native" {
                            "native-rust"
                        } else {
                            "symbolica-interpreter"
                        }
                    );
                }
                for name in ["t2_denominators", "t3_denominators"] {
                    assert_eq!(kernels[name], "native-rust");
                }
                for name in ["other_reduction", "integration", "accumulation"] {
                    assert_eq!(kernels[name], "symbolica-interpreter");
                }
                assert_eq!(actual["numerator_provider"]["evaluator"], "symbolica");
            }
        }
    }
}

#[cfg(not(feature = "native-kernels"))]
#[test]
fn native_denominator_selection_requires_its_feature_before_preparing() {
    for selection in ["native-denominators", "native"] {
        let output = run(&["missing-fixture.json", "--reduction-kernels", selection]);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("--features native-kernels"));
        assert!(output.stdout.is_empty());
    }
}

fn modified(label: &str, expression: &str, mut fixture: Value, options: &[&str]) -> Output {
    let directory =
        std::env::temp_dir().join(format!("ninjars-reduce-{label}-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let source = include_str!("../../../examples/graphs/bubble.dot").replace(
        "1 + ninjars::q0 + 2*ninjars::q1^2 + ninjars::mu2",
        expression,
    );
    std::fs::write(directory.join("graph.dot"), source).unwrap();
    fixture["graph"] = "graph.dot".into();
    std::fs::write(
        directory.join("input.json"),
        serde_json::to_vec(&fixture).unwrap(),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_ninjars"))
        .arg("reduce")
        .arg(directory.join("input.json"))
        .args(options)
        .output()
        .unwrap();
    std::fs::remove_dir_all(directory).unwrap();
    output
}
fn bubble() -> Value {
    serde_json::from_str(include_str!("../../../examples/graphs/bubble.json")).unwrap()
}

#[test]
fn numerator_external_parameters_are_batched_and_failed_lanes_remain_identified() {
    // The scalar product routes through FeynKit, then varies by phase-space lane.
    let actual = value(modified(
        "parameters",
        "spenso::dot(gammalooprs::P(0,spenso::mink(4)),gammalooprs::P(0,spenso::mink(4)))",
        bubble(),
        &["--batch-size", "2"],
    ));
    assert_eq!(actual["results"].as_array().unwrap().len(), 3);
    for (lane, pole) in [-1., -4., -37. / 16.].into_iter().enumerate() {
        near(component(&actual["results"][lane], 1, 0), pole);
    }
    let mut invalid = bubble();
    invalid["samples"][1]["external_momenta"] =
        json!({"0":["0","0","0","0"],"1":["0","0","0","0"]});
    let output = modified("repeated", "1", invalid, &[]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("repeated loop propagators"));
    let mut mixed = bubble();
    mixed["samples"][1]["external_momenta"] =
        json!({"0":["0","1+𝑖","0","0"],"1":["0","1+𝑖","0","0"]});
    let output = modified("failed-lane", "1", mixed, &["--batch-size", "2"]);
    assert!(!output.status.success());
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["results"][0]["status"]["kind"], "success");
    assert_eq!(result["results"][1]["status"]["kind"], "integration_failed");
    assert!(result["results"][1]["values"].is_null());
    assert_eq!(result["results"][2]["status"]["kind"], "success");
}

#[test]
fn exact_numerator_coefficients_retain_more_than_binary64_accuracy() {
    use symbolica::domains::float::Float;
    let mut fixture = bubble();
    fixture["samples"] = json!([fixture["samples"][0].clone()]);
    for (precision, bits, power) in [("double-float", "106", 28), ("float", "192", 45)] {
        let expression = format!("1+1/10^{power}");
        let options = if precision == "float" {
            vec!["--precision", precision, "--bits", bits]
        } else {
            vec!["--precision", precision]
        };
        let actual = value(modified(precision, &expression, fixture.clone(), &options));
        let pole = Float::parse(
            actual["results"][0]["values"][1][0].as_str().unwrap(),
            Some(256),
        )
        .unwrap();
        let delta = Float::parse(&format!("1e-{power}"), Some(256)).unwrap();
        let residual = (pole - Float::with_val(256, 1)) / delta - Float::with_val(256, 1);
        let tolerance = Float::parse("0.001", Some(256)).unwrap();
        assert!(
            residual > -tolerance.clone() && residual < tolerance,
            "{precision}: {residual}"
        );
    }
}

#[test]
fn precision_validation_empty_input_and_unsupported_rank_are_explicit() {
    let invalid = run(&["examples/graphs/bubble.json", "--bits", "128"]);
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("only to --precision float"));
    let invalid = modified("rank", "ninjars::q0^4", bubble(), &[]);
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("rank"));
    let mut empty = bubble();
    empty["samples"] = json!([]);
    let empty = value(modified("empty", "1", empty, &[]));
    assert_eq!(empty["results"], json!([]));
    assert_eq!(empty["success"], true);
}

#[test]
fn rank_selection_requires_supported_topology_and_native_feature_only_when_needed() {
    let output = run(&[
        "examples/graphs/bubble.json",
        "--reduction-kernels",
        "interpreted-rank",
    ]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("requires n=5 and rank=4"));
    #[cfg(feature = "native-kernels")]
    {
        let output = run(&[
            "examples/graphs/bubble.json",
            "--reduction-kernels",
            "native-rank",
        ]);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("requires n=5 and rank=4"));
    }
    #[cfg(not(feature = "native-kernels"))]
    {
        let output = run(&["missing-fixture.json", "--reduction-kernels", "native-rank"]);
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("requires a build with --features native-kernels")
        );
    }
}

#[test]
fn rank_selections_preserve_laurent_values_and_report_conditional_coverage() {
    let fixture = "examples/graphs/pentagon-rank4.json";
    let mut selections = vec!["interpreted-rank"];
    if cfg!(feature = "native-kernels") {
        selections.push("native-rank");
    }
    for precision in ["f64", "double-float", "float"] {
        let reference = value(run(&[
            fixture,
            "--precision",
            precision,
            "--batch-size",
            "2",
        ]));
        assert!(
            reference["reduction_kernels"]
                .get("conditional_corrections")
                .is_none()
        );
        assert_eq!(reference["results"].as_array().unwrap().len(), 3);
        for selection in &selections {
            let report = value(run(&[
                fixture,
                "--precision",
                precision,
                "--batch-size",
                "2",
                "--reduction-kernels",
                selection,
            ]));
            let info = &report["reduction_kernels"];
            let backend = if *selection == "native-rank" {
                "native-rust"
            } else {
                "symbolica-interpreter"
            };
            assert_eq!(info["selection"], *selection);
            assert_eq!(info["topology"], json!({"propagators":5,"rank":4}));
            assert_eq!(
                info["conditional_corrections"],
                json!({"backend":backend,"guard":"all-active-source-kind-structural-zeros","fallback":"reported-generic-families"})
            );
            for field in [
                "t2_denominators",
                "t3_denominators",
                "bubble_source_projection",
                "triangle_bubble_correction",
                "tadpole_source_projection",
            ] {
                assert_eq!(info[field], backend);
            }
            for field in ["other_reduction", "integration", "accumulation"] {
                assert_eq!(info[field], "symbolica-interpreter");
            }
            assert_eq!(report["results"].as_array().unwrap().len(), 3);
            // Serialized output parity across the supported carriers; this
            // binary64 comparison does not measure additional precision.
            for (actual, expected) in report["results"]
                .as_array()
                .unwrap()
                .iter()
                .zip(reference["results"].as_array().unwrap())
            {
                assert_eq!(actual["status"]["kind"], "success");
                for head in 0..5 {
                    for part in 0..2 {
                        near(
                            component(actual, head, part),
                            component(expected, head, part),
                        );
                    }
                }
            }
        }
    }
}
