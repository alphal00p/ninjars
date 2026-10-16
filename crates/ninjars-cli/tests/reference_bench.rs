#![cfg(feature = "reference-codegen")]

use serde_json::{Value, json};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct Temporary(PathBuf);
impl Temporary {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "ninjars-reference-bench-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn run(input: &std::path::Path, output: &std::path::Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_ninjars"))
        .current_dir(root())
        .arg("prepare")
        .arg(input)
        .arg("--reference-bench-dir")
        .arg(output)
        .output()
        .unwrap()
}

#[test]
fn exports_one_template_and_complete_mixed_s_matrix_pool() {
    let temporary = Temporary::new();
    let mut fixture: Value = serde_json::from_slice(
        &fs::read(root().join("examples/graphs/bubble-parameterized.json")).unwrap(),
    )
    .unwrap();
    fixture["graph"] = root()
        .join("examples/graphs/bubble-parameterized.dot")
        .to_str()
        .unwrap()
        .into();
    fixture["samples"][0]["s_matrix"] = json!([["0", "-2"], ["-2", "0"]]);
    let input = temporary.0.join("fixture.json");
    fs::write(&input, serde_json::to_vec(&fixture).unwrap()).unwrap();
    let directory = temporary.0.join("export");
    let output = run(&input, &directory);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let manifest: Value =
        serde_json::from_slice(&fs::read(directory.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["performance_acceptance"], false);
    assert_eq!(manifest["rank"], 2);
    let prepared: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(prepared["samples"][0]["numerator"], "1");
    assert_eq!(manifest["samples"].as_array().unwrap().len(), 3);
    assert_eq!(
        manifest["samples"][0]["s_matrix_binary64"],
        json!([0., -2., -2., 0.])
    );
    assert_eq!(
        manifest["samples"][1]["s_matrix_binary64"],
        json!([0., -4., -4., 0.])
    );
    let count = manifest["parameter_count"].as_u64().unwrap() as usize;
    assert!(count > 0);
    assert_ne!(
        manifest["samples"][0]["parameters_binary64"],
        manifest["samples"][1]["parameters_binary64"]
    );
    let text = fs::read_to_string(directory.join("pool.inputs")).unwrap();
    let mut words = text.split_whitespace();
    assert_eq!(words.next(), Some("NINJARS_BENCH_V1"));
    for expected in [2, 2, count, 3] {
        assert_eq!(words.next().unwrap().parse::<usize>().unwrap(), expected);
    }
    // Three rows, each containing scale/flag, two six-number propagators,
    // ordered complex parameters and a complete four-number matrix; no jobs.
    assert_eq!(words.count(), 3 * (2 + 12 + 2 * count + 4));
    assert_eq!(fs::read_dir(directory).unwrap().count(), 4);
}

#[test]
fn rejects_complex_reference_shifts_before_binary64_underflow() {
    let temporary = Temporary::new();
    let mut fixture: Value =
        serde_json::from_slice(&fs::read(root().join("examples/graphs/bubble.json")).unwrap())
            .unwrap();
    fixture["graph"] = root()
        .join("examples/graphs/bubble.dot")
        .to_str()
        .unwrap()
        .into();
    for leg in ["0", "1"] {
        fixture["samples"][0]["external_momenta"][leg][1] = "1+𝑖*10^-400".into();
    }
    let input = temporary.0.join("fixture.json");
    fs::write(&input, serde_json::to_vec(&fixture).unwrap()).unwrap();
    let directory = temporary.0.join("export");
    let output = run(&input, &directory);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("requires real external momenta"));
    assert!(!directory.exists());
}
