//! `ninjars export-bench` and `ninjars numerator-bench` on the physical
//! benchmark representatives.

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use serde_json::Value;

const REPRESENTATIVES: [(&str, &str); 7] = [
    ("hgg", "top-triangle-0"),
    ("four-photon", "top-box-0"),
    ("gg-hh", "top-triangle-0"),
    ("gg-hh", "top-box-2"),
    ("gg-hhh", "top-triangle-0"),
    ("gg-hhh", "top-box-8"),
    ("gg-hhh", "top-pentagon-26"),
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ninjars"))
        .current_dir(root())
        .args(arguments)
        .output()
        .unwrap()
}

fn success(output: Output) -> Output {
    assert!(
        output.status.success(),
        "CLI exited with {}; stderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn scratch(label: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "ninjars-export-bench-{label}-{}",
        std::process::id()
    ));
    fs::create_dir_all(&directory).unwrap();
    directory
}

fn read(path: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn ncoeff(rank: usize) -> usize {
    (rank + 1) * (rank + 2) * (rank + 3) * (rank + 4) / 24
}

/// Shared-format checks independent of the producing implementation.
fn check_format(case: &Value, lanes: usize) {
    assert_eq!(case["format"], "ninjars-bench-case-v1");
    let n = case["n"].as_u64().unwrap() as usize;
    let rank = case["rank"].as_u64().unwrap() as usize;
    let rows = case["lanes"].as_array().unwrap();
    assert_eq!(rows.len(), lanes);
    for lane in rows {
        assert!(lane["id"].is_string());
        assert_eq!(lane["shifts"].as_array().unwrap().len(), n);
        assert_eq!(lane["masses_squared"].as_array().unwrap().len(), n);
        assert!(lane["mu_squared"].as_f64().unwrap() > 0.);
        assert_eq!(lane["tensor"].as_array().unwrap().len(), ncoeff(rank));
        let s = lane["s_matrix"].as_array().unwrap();
        assert_eq!(s.len(), n);
        for (i, row) in s.iter().enumerate() {
            assert_eq!(row[i].as_f64(), Some(0.));
            for (j, value) in row.as_array().unwrap().iter().enumerate() {
                assert_eq!(*value, s[j][i]);
            }
        }
    }
}

fn export(fixture: &str, output: &Path, report: &Path, extra: &[&str]) -> (Value, Value) {
    let mut arguments = vec![
        "export-bench",
        fixture,
        "--output",
        output.to_str().unwrap(),
        "--report",
        report.to_str().unwrap(),
        "--check-points",
        "6",
    ];
    arguments.extend_from_slice(extra);
    success(run(&arguments));
    (read(output), read(report))
}

fn check_verification(report: &Value, label: &str) {
    let verification = &report["verification"];
    for key in [
        "max_relative_error_vs_high_precision_polynomial",
        "max_relative_error_vs_f64_polynomial",
    ] {
        let error = verification[key].as_f64().unwrap();
        assert!(error < 1e-12, "{label}: {key} = {error}");
    }
}

#[test]
fn pool_fixtures_export_the_committed_bench_cases() {
    let directory = scratch("pool");
    for (process, graph) in REPRESENTATIVES {
        let name = format!("{process}-{graph}");
        let output = directory.join(format!("{name}.json"));
        let fixture = format!("examples/graphs/physical/{process}/{graph}-pool.json");
        let (case, report) = export(
            &fixture,
            &output,
            &directory.join(format!("{name}.report.json")),
            &["--name", &name],
        );
        check_format(&case, 3);
        check_verification(&report, &name);
        assert_eq!(
            report["tensor"]["rank"],
            report["tensor"]["numerator_degree"]
        );
        // The committed case is reproduced bit for bit from the exact fixture.
        let committed = root().join(format!("validation/benchmark/cases/{name}.json"));
        assert!(
            fs::read(&output).unwrap() == fs::read(&committed).unwrap(),
            "{name}: exported case differs from {}",
            committed.display()
        );
    }
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn original_physical_fixtures_match_the_polynomial_numerator() {
    let directory = scratch("original");
    for (process, graph) in REPRESENTATIVES {
        let name = format!("{process}-{graph}");
        let fixture = format!("examples/graphs/physical/{process}/{graph}.json");
        let samples = read(&root().join(&fixture))["samples"]
            .as_array()
            .unwrap()
            .len();
        let (case, report) = export(
            &fixture,
            &directory.join(format!("{name}.json")),
            &directory.join(format!("{name}.report.json")),
            &[],
        );
        check_format(&case, samples);
        check_verification(&report, &name);
    }
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn export_selection_rank_padding_and_rejections() {
    let directory = scratch("options");
    let fixture = "examples/graphs/physical/hgg/top-triangle-0-pool.json";
    let output = directory.join("case.json");
    let report = directory.join("report.json");
    let (case, _) = export(
        fixture,
        &output,
        &report,
        &["--select", "2,0", "--rank", "4", "--s-matrix", "none"],
    );
    assert_eq!(case["rank"], 4);
    let lanes = case["lanes"].as_array().unwrap();
    assert_eq!(lanes.len(), 2);
    assert_eq!(lanes[0]["id"], "boosted_b");
    assert_eq!(lanes[1]["id"], "rest_frame");
    assert!(lanes.iter().all(|lane| lane["s_matrix"].is_null()));
    let committed = read(&root().join("validation/benchmark/cases/hgg-top-triangle-0.json"));
    // Padding to rank 4 keeps the rank-2 slots and appends zeros.
    for (lane, source) in [(0, 2), (1, 0)] {
        let padded = lanes[lane]["tensor"].as_array().unwrap();
        let original = committed["lanes"][source]["tensor"].as_array().unwrap();
        assert_eq!(&padded[..original.len()], &original[..]);
        assert!(
            padded[original.len()..]
                .iter()
                .all(|c| c[0] == 0. && c[1] == 0.)
        );
    }
    for (arguments, message) in [
        (&["--rank", "1"][..], "below the numerator degree"),
        (&["--select", "3"][..], "outside 0..3"),
    ] {
        let mut all = vec![
            "export-bench",
            fixture,
            "--output",
            output.to_str().unwrap(),
        ];
        all.extend_from_slice(arguments);
        let result = run(&all);
        assert!(!result.status.success());
        assert!(
            String::from_utf8_lossy(&result.stderr).contains(message),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn numerator_bench_reproduces_the_case_and_times_warm_batches() {
    let directory = scratch("numerator");
    let output = directory.join("report.json");
    let backend = if cfg!(feature = "numerator-jit") {
        "symjit-o2"
    } else {
        "interpreter"
    };
    success(run(&[
        "numerator-bench",
        "examples/graphs/physical/hgg/top-triangle-0-pool.json",
        "--points",
        "validation/benchmark/points/hgg.json",
        "--feynman-rules",
        "--backend",
        backend,
        "--check-case",
        "validation/benchmark/cases/hgg-top-triangle-0.json",
        "--widths",
        "1,3",
        "--blocks",
        "3",
        "--min-block-ms",
        "0.5",
        "--output",
        output.to_str().unwrap(),
    ]));
    let report = read(&output);
    let check = &report["check_case"];
    assert!(
        check["max_tensor_error_relative_to_lane_max"]
            .as_f64()
            .unwrap()
            < 1e-12
    );
    assert!(
        check["max_shift_error_relative_to_energy"]
            .as_f64()
            .unwrap()
            < 1e-15
    );
    assert_eq!(check["max_mass_squared_error_relative"].as_f64(), Some(0.));
    assert!(
        check["max_s_matrix_error_relative_to_energy_squared"]
            .as_f64()
            .unwrap()
            < 1e-14
    );
    assert_eq!(
        (report["rank"].as_u64(), report["ncoeff"].as_u64()),
        (Some(2), Some(15))
    );
    let timings = report["per_point_timing"].as_array().unwrap();
    assert_eq!(timings.len(), 2);
    for timing in timings {
        for key in [
            "tensor_coefficients",
            "propagator_data",
            "tensor_and_propagator_data",
        ] {
            assert!(timing[key]["median_ns_per_point"].as_f64().unwrap() > 0.);
            assert_eq!(timing[key]["blocks"], 3);
        }
    }
    let generation = &report["generation_ns"];
    assert!(generation["total"].as_f64().unwrap() > 0.);
    assert!(generation["feynman_rules"].as_f64().unwrap() > 0.);
    fs::remove_dir_all(directory).unwrap();
}
