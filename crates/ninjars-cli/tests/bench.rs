use serde_json::{Value, json};
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

#[test]
fn warmed_runtime_matches_reduce_for_all_precisions_and_both_workloads() {
    for precision in ["f64", "double-float", "float"] {
        let fixture = "examples/graphs/bubble.json";
        let reduced = value(run(
            "reduce",
            &[fixture, "--precision", precision, "--batch-size", "2"],
        ));
        let report = value(run(
            "bench",
            &[
                fixture,
                "--precision",
                precision,
                "--batch-sizes",
                "1,2",
                "--points",
                "3",
                "--rounds",
                "2",
                "--iterations",
                "2",
            ],
        ));
        assert_eq!(report["success"], true);
        assert_eq!(report["performance_acceptance"], false);
        assert_eq!(
            report["cache_policy"]["numerical_master_results"],
            "not cached"
        );
        assert_eq!(report["cases"].as_array().unwrap().len(), 4);
        for case in report["cases"].as_array().unwrap() {
            let lane = if case["workload"] == "repeated" { 0 } else { 2 };
            assert_eq!(case["representative_source_lane"], lane);
            assert_eq!(
                case["representative_result"],
                reduced["results"][lane]["values"]
            );
            assert_eq!(case["round_ns"].as_array().unwrap().len(), 2);
            assert_eq!(case["timed_points_per_round"], 6);
            assert_eq!(case["diagnostic"]["points"], 3);
            assert_eq!(case["diagnostic"]["failed_lanes"], 0);
            assert_eq!(case["diagnostic"]["additivity_checked"], true);
            let stages = &case["diagnostic"]["same_pass_nanoseconds"];
            let ns = |name: &str| stages[name].as_u64().unwrap();
            assert_eq!(
                ns("total"),
                ns("numerator_coefficients")
                    + ns("numerator_callbacks")
                    + ns("actual_master_provider")
                    + ns("residual_reduction")
            );
            assert_eq!(
                ns("reduction_excluding_masters_and_coefficients"),
                ns("numerator_callbacks") + ns("residual_reduction")
            );
            assert!(ns("actual_master_provider") > 0);
            assert_eq!(case["precision_rescue"]["attempts"], 0);
            assert!(case["diagnostic"]["masters"]["calls"].as_u64().unwrap() > 0);
            assert!(
                case["diagnostic"]["numerator_expansion"]["rows"]
                    .as_u64()
                    .unwrap()
                    > 0
            );
            assert!(case["median_ns"].as_f64().unwrap() > 0.);
            if case["batch_size"] == 2 {
                assert_eq!(case["tail_lanes"], 1);
            }
        }
    }
}

#[test]
fn stopped_reduction_reports_exactly_zero_actual_master_work() {
    let report = value(run(
        "bench",
        &[
            "examples/graphs/bubble.json",
            "--cut-stop",
            "3",
            "--points",
            "3",
            "--batch-sizes",
            "2",
            "--rounds",
            "1",
        ],
    ));
    for case in report["cases"].as_array().unwrap() {
        let diagnostic = &case["diagnostic"];
        assert_eq!(diagnostic["masters"]["calls"], 0);
        assert_eq!(diagnostic["masters"]["rows"], 0);
        assert_eq!(
            diagnostic["same_pass_nanoseconds"]["actual_master_provider"],
            0
        );
        assert_eq!(diagnostic["additivity_checked"], true);
    }
}

#[test]
fn default_widths_cover_full_batches_and_incomplete_tails() {
    let report = value(run(
        "bench",
        &[
            "examples/graphs/bubble.json",
            "--workload",
            "repeated",
            "--rounds",
            "1",
        ],
    ));
    let cases = report["cases"].as_array().unwrap();
    for (case, width) in cases.iter().zip([1, 8, 32, 128, 1024]) {
        assert_eq!(case["batch_size"], width);
        assert_eq!(case["points_per_iteration"], 1025);
        assert_eq!(case["tail_lanes"], if width == 1 { 0 } else { 1 });
        assert!(case["full_batches"].as_u64().unwrap() > 0);
        assert_eq!(case["diagnostic"]["successful_lanes"], 1025);
    }
    assert_eq!(cases.len(), 5);
}

#[cfg(feature = "native-kernels")]
#[test]
fn benchmark_honors_partial_native_selection() {
    let fixture = "examples/graphs/bubble.json";
    let reduced = value(run("reduce", &[fixture]));
    for selection in ["native-denominators", "native"] {
        let report = value(run(
            "bench",
            &[
                fixture,
                "--reduction-kernels",
                selection,
                "--batch-sizes",
                "2",
                "--points",
                "3",
                "--rounds",
                "1",
            ],
        ));
        assert_eq!(report["reduction_kernels"]["selection"], selection);
        for family in [
            "bubble_source_projection",
            "triangle_bubble_correction",
            "tadpole_source_projection",
        ] {
            assert_eq!(
                report["reduction_kernels"][family],
                if selection == "native" {
                    "native-rust"
                } else {
                    "symbolica-interpreter"
                }
            );
        }
        assert_eq!(
            report["reduction_kernels"]["other_reduction"],
            "symbolica-interpreter"
        );
        assert_eq!(report["performance_acceptance"], false);
        for case in report["cases"].as_array().unwrap() {
            assert_eq!(case["diagnostic"]["failed_lanes"], 0);
            assert_eq!(case["tail_lanes"], 1);
            let lane = case["representative_source_lane"].as_u64().unwrap() as usize;
            for (actual, expected) in case["representative_result"]
                .as_array()
                .unwrap()
                .iter()
                .zip(reduced["results"][lane]["values"].as_array().unwrap())
            {
                for part in 0..2 {
                    let a: f64 = actual[part].as_str().unwrap().parse().unwrap();
                    let b: f64 = expected[part].as_str().unwrap().parse().unwrap();
                    assert!((a - b).abs() <= 1e-9 * (1. + b.abs()));
                }
            }
        }
    }
}

#[test]
fn failed_lanes_are_reported_without_claiming_acceptance_or_rescue() {
    let directory =
        std::env::temp_dir().join(format!("ninjars-bench-failure-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let mut fixture: Value =
        serde_json::from_str(include_str!("../../../examples/graphs/bubble.json")).unwrap();
    fixture["graph"] = root()
        .join("examples/graphs/bubble.dot")
        .to_string_lossy()
        .into_owned()
        .into();
    fixture["samples"][1]["external_momenta"] =
        json!({"0":["0","1+𝑖","0","0"],"1":["0","1+𝑖","0","0"]});
    let path = directory.join("input.json");
    std::fs::write(&path, serde_json::to_vec(&fixture).unwrap()).unwrap();
    let output = run(
        "bench",
        &[
            path.to_str().unwrap(),
            "--workload",
            "changing",
            "--batch-sizes",
            "2",
            "--points",
            "3",
            "--rounds",
            "1",
        ],
    );
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    let case = &report["cases"][0];
    assert_eq!(report["success"], false);
    assert_eq!(report["performance_acceptance"], false);
    assert_eq!(case["diagnostic"]["failed_lanes"], 1);
    assert_eq!(case["diagnostic"]["successful_lanes"], 2);
    assert_eq!(case["warmup_final_pass_failed_lanes"], 1);
    assert_eq!(case["precision_rescue"]["enabled"], false);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn invalid_benchmark_controls_are_rejected_before_timing() {
    for extra in [
        vec!["--warmup", "0"],
        vec!["--batch-sizes", "1,1"],
        vec!["--bits", "192"],
        vec!["--rounds", "0"],
        vec!["--workload", "changing", "--points", "1"],
    ] {
        let mut args = vec!["examples/graphs/bubble.json"];
        args.extend(extra);
        assert!(!run("bench", &args).status.success());
    }
}

#[test]
fn rank_selection_reports_keep_master_and_reduction_accounting_separate() {
    let mut selections = vec!["interpreted-rank"];
    if cfg!(feature = "native-kernels") {
        selections.push("native-rank");
    }
    for selection in selections {
        let report = value(run(
            "bench",
            &[
                "examples/graphs/pentagon-rank4.json",
                "--reduction-kernels",
                selection,
                "--batch-sizes",
                "2",
                "--points",
                "3",
                "--rounds",
                "1",
            ],
        ));
        assert_eq!(report["performance_acceptance"], false);
        assert_eq!(report["reduction_kernels"]["selection"], selection);
        assert_eq!(
            report["reduction_kernels"]["topology"],
            json!({"propagators":5,"rank":4})
        );
        assert_eq!(
            report["reduction_kernels"]["conditional_corrections"]["backend"],
            if selection == "native-rank" {
                "native-rust"
            } else {
                "symbolica-interpreter"
            }
        );
        for case in report["cases"].as_array().unwrap() {
            assert_eq!(case["tail_lanes"], 1);
            assert_eq!(case["diagnostic"]["failed_lanes"], 0);
            assert_eq!(case["diagnostic"]["additivity_checked"], true);
            let stages = &case["diagnostic"]["same_pass_nanoseconds"];
            let ns = |name: &str| stages[name].as_u64().unwrap();
            assert_eq!(
                ns("total"),
                ns("numerator_coefficients")
                    + ns("numerator_callbacks")
                    + ns("actual_master_provider")
                    + ns("residual_reduction")
            );
            assert_eq!(
                ns("reduction_excluding_masters_and_coefficients"),
                ns("numerator_callbacks") + ns("residual_reduction")
            );
            assert!(case["diagnostic"]["masters"]["calls"].as_u64().unwrap() > 0);
        }
    }
}
