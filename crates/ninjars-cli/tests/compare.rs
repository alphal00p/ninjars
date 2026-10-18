#![cfg(feature = "reference-codegen")]

use serde_json::{Value, json};
use std::{fs, path::PathBuf, process::Command};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
#[ignore = "builds original Ninja runners with C++/Fortran reference dependencies"]
fn prepared_polynomials_compare_with_original_ninja_in_all_laurent_coefficients() {
    let directory = std::env::temp_dir().join(format!("ninjars-compare-{}", std::process::id()));
    // The seven-propagator rank-8 heptagon exercises the reference export and
    // runner beyond six propagators.
    for fixture in ["bubble", "triangle-reference", "box", "heptagon-rank8"] {
        let output = Command::new(env!("CARGO_BIN_EXE_ninjars"))
            .current_dir(root())
            .args([
                "compare",
                &format!("examples/graphs/{fixture}.json"),
                "--batch-size",
                "2",
                "--diagnostics",
            ])
            .arg("--reference-dir")
            .arg(directory.join(fixture))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["operation"], "compare");
        assert_eq!(report["success"], true);
        assert_eq!(
            report["comparison"]["timings_are_acceptance_evidence"],
            false
        );
        let checks = report["comparison"]["checks"].as_array().unwrap();
        assert_eq!(checks.len(), report["results"].as_array().unwrap().len());
        for check in checks {
            assert_eq!(check["passed"], true);
            for error in check["laurent_scaled_errors"].as_array().unwrap() {
                assert!(error.as_f64().unwrap() <= 1e-9);
            }
            assert!(check["artifact"]["generated_numerator_sha256"].is_string());
        }
        assert!(directory.join(fixture).join("comparison.json").exists());
        println!(
            "NINJARS_COMPARISON {}",
            json!({"fixture":fixture,"precision":report["precision"],"cut_stop":report["cut_stop"],"comparison":report["comparison"]})
        );
    }
    fs::remove_dir_all(directory).unwrap();
}

#[test]
#[ignore = "builds original Ninja runners with C++/Fortran reference dependencies"]
fn cut_stop_and_supplied_matrix_are_applied_to_both_reductions() {
    let directory =
        std::env::temp_dir().join(format!("ninjars-compare-controls-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    let mut fixture: Value =
        serde_json::from_str(include_str!("../../../examples/graphs/bubble.json")).unwrap();
    fixture["graph"] = root()
        .join("examples/graphs/bubble.dot")
        .canonicalize()
        .unwrap()
        .to_str()
        .unwrap()
        .into();
    fixture["samples"] = json!([fixture["samples"][0].clone()]);
    fixture["samples"][0]["s_matrix"] = json!([["0", "-2"], ["-2", "0"]]);
    let input = directory.join("fixture.json");
    fs::write(&input, serde_json::to_vec(&fixture).unwrap()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_ninjars"))
        .current_dir(root())
        .arg("compare")
        .arg(input)
        .args([
            "--cut-stop",
            "2",
            "--precision",
            "double-float",
            "--diagnostics",
        ])
        .arg("--reference-dir")
        .arg(directory.join("artifacts"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["success"], true);
    assert_eq!(report["results"][0]["cut_counts"]["tadpoles"], 0);
    assert_eq!(
        report["reference_export"]["samples"][0]["s_matrix_binary64"],
        json!([0., -2., -2., 0.])
    );
    assert!(directory.join("artifacts/sample-000000.s-matrix").exists());
    println!(
        "NINJARS_COMPARISON {}",
        json!({"fixture":"bubble with supplied s=-2 and cut_stop=2","precision":report["precision"],"cut_stop":report["cut_stop"],"comparison":report["comparison"]})
    );
    fs::remove_dir_all(directory).unwrap();
}
