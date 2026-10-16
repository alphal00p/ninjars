use std::{path::PathBuf, process::Command};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn bubble_fixture_emits_ordered_exact_sample_inputs() {
    let output = Command::new(env!("CARGO_BIN_EXE_ninjars"))
        .current_dir(root())
        .args(["prepare", "examples/graphs/bubble.json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["operation"], "prepare");
    assert_eq!(value["propagator_edges"].as_array().unwrap().len(), 2);
    assert_eq!(value["samples"].as_array().unwrap().len(), 3);
    assert_eq!(
        value["samples"][0]["mass_squared"],
        serde_json::json!(["1", "1"])
    );
    assert_eq!(
        value["input_fixture"]["samples"][2]["external_momenta"]["0"][1],
        "3/2"
    );
}

#[test]
fn triangle_dot_accepts_matching_sample_fixture() {
    let output = Command::new(env!("CARGO_BIN_EXE_ninjars"))
        .current_dir(root())
        .args([
            "prepare",
            "examples/graphs/triangle.dot",
            "--samples",
            "examples/graphs/triangle.json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["samples"].as_array().unwrap().len(), 2);
    assert_eq!(value["denominators"].as_array().unwrap().len(), 3);
}

#[test]
fn rejects_sample_fixture_from_another_graph() {
    let output = Command::new(env!("CARGO_BIN_EXE_ninjars"))
        .current_dir(root())
        .args([
            "prepare",
            "examples/graphs/bubble.dot",
            "--samples",
            "examples/graphs/triangle.json",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("different graph"));
}

fn run_modified_fixture(label: &str, mut fixture: serde_json::Value) -> std::process::Output {
    fixture["graph"] = root()
        .join("examples/graphs/bubble.dot")
        .to_string_lossy()
        .into_owned()
        .into();
    let path = std::env::temp_dir().join(format!("ninjars-{label}-{}.json", std::process::id()));
    std::fs::write(&path, serde_json::to_vec(&fixture).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_ninjars"))
        .arg("prepare")
        .arg(&path)
        .output();
    std::fs::remove_file(path).unwrap();
    result.unwrap()
}

#[test]
fn exact_mass_override_retains_a_correction_below_binary64_precision() {
    let mut fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../examples/graphs/bubble.json")).unwrap();
    fixture["parameter_overrides"]["UFO::mass"] = "1+1/10^40".into();
    let output = run_modified_fixture("exact-mass", fixture);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_ne!(value["samples"][0]["mass_squared"][0], "1");
    assert!(
        value["samples"][0]["mass_squared"][0]
            .as_str()
            .unwrap()
            .contains('/')
    );
}

#[test]
fn inconsistent_dependent_external_momentum_is_rejected() {
    let mut fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../examples/graphs/bubble.json")).unwrap();
    fixture["samples"][0]["external_momenta"]["1"][0] = "1".into();
    let output = run_modified_fixture("bad-routing", fixture);
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("violates FeynKit's dependent-momentum routing")
    );
}
