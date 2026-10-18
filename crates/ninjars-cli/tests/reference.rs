#![cfg(feature = "reference-codegen")]

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

struct FixtureDirectory(PathBuf);

impl FixtureDirectory {
    fn new(label: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("ninjars-export-{label}-{}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn prepare(
        &self,
        dot: &str,
        overrides: serde_json::Value,
        samples: serde_json::Value,
    ) -> Output {
        fs::write(self.0.join("graph.dot"), dot).unwrap();
        fs::write(
            self.0.join("fixture.json"),
            serde_json::to_vec(&serde_json::json!({
                "schema_version": 1, "graph": "graph.dot", "model": "phi3", "scale_squared": "1",
                "parameter_overrides": overrides, "samples": samples,
            }))
            .unwrap(),
        )
        .unwrap();
        Command::new(env!("CARGO_BIN_EXE_ninjars"))
            .args(["prepare"])
            .arg(self.0.join("fixture.json"))
            .arg("--reference-dir")
            .arg(self.0.join("export"))
            .output()
            .unwrap()
    }

    fn json(&self, name: &str) -> serde_json::Value {
        read_json(&self.0.join("export").join(name))
    }
}

impl Drop for FixtureDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn bubble_sample(energy: &str) -> serde_json::Value {
    serde_json::json!([{ "external_momenta": {"0": [energy, "1", "0", "0"], "1": [energy, "1", "0", "0"]}}])
}

#[test]
fn reference_export_marks_binary64_conversion_and_preserves_exact_inputs() {
    let dir = FixtureDirectory::new("rounding");
    let result = dir.prepare(
        include_str!("../../../examples/graphs/bubble.dot"),
        serde_json::json!({"UFO::mass": "1+1/10^40"}),
        bubble_sample("2"),
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let manifest = dir.json("manifest.json");
    assert_eq!(manifest["reference_precision"], "binary64");
    assert_eq!(manifest["overall_factor_included_in_callback"], true);
    assert_eq!(
        manifest["samples"][0]["mass_squared_binary64"][0],
        serde_json::json!([1.0, 0.0])
    );
    let prepared = dir.json("prepared.json");
    assert_ne!(prepared["samples"][0]["mass_squared"][0], "1");
    assert_eq!(
        prepared["input_fixture"]["parameter_overrides"]["UFO::mass"],
        "1+1/10^40"
    );
    assert!(dir.0.join("export/sample-000000.hh").exists());
    assert!(dir.0.join("export/sample-000000.inputs").exists());
}

#[test]
fn reference_export_applies_graph_factor_and_bridge_denominator_once() {
    let dir = FixtureDirectory::new("normalization");
    let result = dir.prepare(r#"digraph {
        overall_factor="7"; num="5";
        ext [style=invis]; ext -> a [particle="phi"];
        a -> b [particle="phi", lmb_id=0]; b -> a [particle="phi"];
        b -> c [particle="phi"];
        c -> ext [particle="phi"]; c -> ext [particle="phi"];
    }"#, serde_json::json!({"UFO::mass": "1"}),
        serde_json::json!([{ "external_momenta": {"0": ["2", "1", "0", "0"], "1": ["1", "1/2", "0", "0"]}}]));
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let manifest = dir.json("manifest.json");
    assert_eq!(manifest["samples"][0]["propagator_count"], 2);
    // FeynKit's canonical basis routes this bridge as P(0)+P(1),
    // whose squared momentum is 27/4 and unit-mass denominator is 23/4.
    assert_eq!(manifest["samples"][0]["full_numerator"], "140/23");
    let prepared = dir.json("prepared.json");
    assert_eq!(prepared["samples"][0]["numerator"], "5");
    assert_eq!(prepared["samples"][0]["overall_factor"], "28/23");
}

#[test]
fn reference_export_rejects_complex_external_momenta() {
    let dir = FixtureDirectory::new("complex");
    let result = dir.prepare(
        include_str!("../../../examples/graphs/bubble.dot"),
        serde_json::json!({"UFO::mass": "1"}),
        bubble_sample("𝑖/10^400"),
    );
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("complex values are unsupported"));
    assert!(!dir.0.join("export").exists());
}

#[test]
fn reference_export_rejects_unresolved_numerator_parameters() {
    let dir = FixtureDirectory::new("unbound");
    let graph = include_str!("../../../examples/graphs/bubble.dot")
        .replace("1 + ninjars::q0", "ninjars::unbound + ninjars::q0");
    let result = dir.prepare(
        &graph,
        serde_json::json!({"UFO::mass": "1"}),
        bubble_sample("2"),
    );
    assert!(!result.status.success());
    // Preparation rejects the unbound parameter before any export.
    let message = String::from_utf8_lossy(&result.stderr);
    assert!(
        message.contains("cannot be evaluated") && message.contains("ninjars::unbound"),
        "{message}"
    );
    assert!(!dir.0.join("export").exists());
}

#[test]
fn reference_export_keeps_complex_squared_masses() {
    let dir = FixtureDirectory::new("complex-mass");
    let result = dir.prepare(
        include_str!("../../../examples/graphs/bubble.dot"),
        serde_json::json!({"UFO::mass": "1-𝑖/10"}),
        bubble_sample("2"),
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let manifest = dir.json("manifest.json");
    assert_eq!(
        manifest["samples"][0]["mass_squared_binary64"][0],
        serde_json::json!([0.99, -0.2])
    );
}

#[test]
fn reference_export_rejects_numerator_coefficient_underflow() {
    let dir = FixtureDirectory::new("coefficient-underflow");
    let graph = include_str!("../../../examples/graphs/bubble.dot")
        .replace("2*ninjars::q1^2", "ninjars::q1^2/10^400");
    let result = dir.prepare(
        &graph,
        serde_json::json!({"UFO::mass": "1"}),
        bubble_sample("2"),
    );
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("underflows binary64"));
    assert!(!dir.0.join("export").exists());
}
