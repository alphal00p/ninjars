use serde_json::{Value, json};
use std::{
    path::PathBuf,
    process::{Command, Output},
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn fixture() -> Value {
    serde_json::from_str(include_str!("../../../examples/graphs/bubble.json")).unwrap()
}
fn run(
    label: &str,
    expression: &str,
    mut fixture: Value,
    operation: &str,
    args: &[&str],
) -> Output {
    let directory =
        std::env::temp_dir().join(format!("ninjars-controls-{label}-{}", std::process::id()));
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
        .arg(operation)
        .arg(directory.join("input.json"))
        .args(args)
        .output()
        .unwrap();
    std::fs::remove_dir_all(directory).unwrap();
    output
}
fn value(output: Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
fn component(row: &Value, index: usize) -> f64 {
    row["values"][index][0].as_str().unwrap().parse().unwrap()
}
fn near(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9 * (1. + b.abs()), "{a} != {b}");
}

#[test]
fn cut_stopping_retains_requested_contributions_and_reports_effective_cut() {
    // q²-mu² = D_zero_shift+1, so this integral is A0(1)+B0(s;1,1).
    let numerator = "ninjars::q0^2-ninjars::q1^2-ninjars::q2^2-ninjars::q3^2-ninjars::mu2";
    let full = value(run(
        "full",
        numerator,
        fixture(),
        "reduce",
        &["--diagnostics"],
    ));
    let zero = value(run(
        "zero",
        numerator,
        fixture(),
        "reduce",
        &["--cut-stop", "0"],
    ));
    let bubble = value(run(
        "bubble",
        numerator,
        fixture(),
        "reduce",
        &["--cut-stop", "2", "--diagnostics", "--batch-size", "2"],
    ));
    let empty = value(run(
        "empty",
        numerator,
        fixture(),
        "reduce",
        &["--cut-stop", "3", "--diagnostics"],
    ));
    assert_eq!(zero["cut_stop"], 1);
    assert_eq!(zero["requested_cut_stop"], 0);
    assert_eq!(bubble["cut_stop"], 2);
    for lane in 0..3 {
        let f = &full["results"][lane];
        let b = &bubble["results"][lane];
        for index in 0..5 {
            near(
                component(f, index),
                component(&zero["results"][lane], index),
            );
            near(component(&empty["results"][lane], index), 0.);
        }
        near(component(f, 0) - component(b, 0), 1.);
        near(component(f, 1), 2.);
        near(component(b, 1), 1.);
        near(component(f, 2), 0.);
        assert_eq!(f["cut_counts"]["tadpoles"], 2);
        assert_eq!(b["cut_counts"]["tadpoles"], 0);
        assert_eq!(b["cut_counts"]["bubbles"], 1);
        assert_eq!(empty["results"][lane]["cut_counts"]["bubbles"], 0);
    }
}

#[test]
fn supplied_matrix_reaches_masters_and_rational_weights_in_mixed_batches() {
    let numerator = "1+ninjars::mu2";
    let baseline = value(run("baseline", numerator, fixture(), "reduce", &[]));
    let mut inputs = fixture();
    inputs["samples"][0]["s_matrix"] = json!([["0", "-4*UFO::mass^2"], ["-4*UFO::mass^2", "0"]]);
    let prepared = value(run("prepared", numerator, inputs.clone(), "prepare", &[]));
    assert_eq!(
        prepared["samples"][0]["s_matrix"],
        json!([["0", "-4"], ["-4", "0"]])
    );
    for (label, batch) in [("single", "1"), ("tail", "2")] {
        let result = value(run(
            label,
            numerator,
            inputs.clone(),
            "reduce",
            &["--batch-size", batch],
        ));
        assert_eq!(result["results"][0]["s_matrix_source"], "supplied");
        assert_eq!(result["results"][1]["s_matrix_source"], "momenta");
        for index in 0..5 {
            near(
                component(&result["results"][0], index),
                component(&baseline["results"][1], index),
            );
            for lane in 1..3 {
                near(
                    component(&result["results"][lane], index),
                    component(&baseline["results"][lane], index),
                );
            }
        }
        near(component(&baseline["results"][0], 4), 7. / 6.);
        near(component(&result["results"][0], 4), 5. / 3.);
    }
}

#[test]
fn matrix_shape_and_exact_numerical_validation_preserve_other_lanes() {
    let mut inputs = fixture();
    inputs["samples"][0]["s_matrix"] = json!([["0"]]);
    let malformed = run("shape", "1", inputs, "prepare", &[]);
    assert!(!malformed.status.success());
    assert!(String::from_utf8_lossy(&malformed.stderr).contains("exactly 2 rows of 2 entries"));
    let mut inputs = fixture();
    inputs["samples"][0]["s_matrix"] = json!([["0", "-1-1/10^40"], ["-1", "0"]]);
    inputs["samples"][1]["s_matrix"] = json!([["1/10^40", "-4"], ["-4", "0"]]);
    let output = run(
        "invalid-lanes",
        "1",
        inputs,
        "reduce",
        &["--precision", "float", "--bits", "192", "--batch-size", "3"],
    );
    assert!(!output.status.success());
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    for lane in 0..2 {
        assert_eq!(
            result["results"][lane]["status"]["kind"],
            "integration_failed"
        );
        assert!(result["results"][lane]["values"].is_null());
    }
    assert_eq!(result["results"][2]["status"]["kind"], "success");
}

#[test]
fn exact_matrix_rational_contribution_retains_more_than_binary64_accuracy() {
    use symbolica::domains::float::Float;
    for (precision, power) in [("double-float", 28), ("float", 40)] {
        let mut inputs = fixture();
        inputs["samples"] = json!([inputs["samples"][0].clone()]);
        let s = format!("-6/10^{power}");
        inputs["samples"][0]["s_matrix"] = json!([["0", s], [s, "0"]]);
        let result = value(run(
            precision,
            "ninjars::mu2",
            inputs,
            "reduce",
            &["--precision", precision],
        ));
        let row = &result["results"][0];
        let finite = Float::parse(row["values"][0][0].as_str().unwrap(), Some(256)).unwrap();
        let normalized = (finite - Float::with_val(256, 1))
            / Float::parse(&format!("1e-{power}"), Some(256)).unwrap();
        assert!(
            normalized > Float::parse("0.999", Some(256)).unwrap()
                && normalized < Float::parse("1.001", Some(256)).unwrap(),
            "{precision}: {normalized}"
        );
        near(component(row, 1), 0.);
        near(component(row, 2), 0.);
        near(component(row, 3), 0.);
    }
}

#[test]
fn stopped_box_does_not_fail_for_an_unused_overflowing_automatic_scale() {
    let mut fixture: Value =
        serde_json::from_str(include_str!("../../../examples/graphs/box.json")).unwrap();
    fixture["graph"] = root()
        .join("examples/graphs/box.dot")
        .to_string_lossy()
        .into_owned()
        .into();
    fixture["samples"] = json!([{"external_momenta":{"0":["0","10^200","0","0"],"1":["0","0","10^200","0"],"2":["0","0","0","10^200"]}}]);
    let path =
        std::env::temp_dir().join(format!("ninjars-stopped-box-{}.json", std::process::id()));
    std::fs::write(&path, serde_json::to_vec(&fixture).unwrap()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_ninjars"))
        .arg("reduce")
        .arg(&path)
        .args([
            "--cut-stop",
            "5",
            "--mu-extraction",
            "sampling",
            "--full-box-coefficients",
            "--diagnostics",
        ])
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    let result = value(output);
    let row = &result["results"][0];
    assert_eq!(row["status"]["kind"], "success");
    assert_eq!(row["cut_counts"]["boxes"], 0);
    assert_eq!(row["mu_sampling_scale"]["used"], false);
    assert!(row["mu_sampling_scale"].get("reason").is_none());
    for index in 0..5 {
        near(component(row, index), 0.);
    }
}
