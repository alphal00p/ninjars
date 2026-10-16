use std::{
    path::PathBuf,
    process::{Command, Output},
};

fn run(command: &str, arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ninjars"))
        .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
        .arg(command)
        .args(arguments)
        .output()
        .unwrap()
}

#[test]
fn incompatible_coefficient_backends_fail_before_graph_preparation() {
    for backend in ["symjit-o2", "symjit-o2-factored"] {
        for command in ["reduce", "bench"] {
            for precision in ["double-float", "float"] {
                let result = run(
                    command,
                    &[
                        "missing-graph.json",
                        "--numerator-evaluator",
                        backend,
                        "--precision",
                        precision,
                    ],
                );
                assert!(!result.status.success());
                assert!(
                    String::from_utf8_lossy(&result.stderr).contains("requires --precision f64")
                );
            }
            #[cfg(not(feature = "numerator-jit"))]
            {
                let result = run(
                    command,
                    &["missing-graph.json", "--numerator-evaluator", backend],
                );
                assert!(!result.status.success());
                assert!(
                    String::from_utf8_lossy(&result.stderr).contains("--features numerator-jit")
                );
            }
        }
    }
}

#[cfg(feature = "numerator-jit")]
mod enabled {
    use super::*;
    use serde_json::Value;

    fn value(output: Output) -> Value {
        assert!(
            output.status.success(),
            "CLI exited with {}; stderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice(&output.stdout).unwrap()
    }

    fn near(actual: &Value, expected: &Value) {
        assert_eq!(actual.as_array().unwrap().len(), 5);
        for (a, e) in actual
            .as_array()
            .unwrap()
            .iter()
            .zip(expected.as_array().unwrap())
        {
            let a = [
                a[0].as_str().unwrap().parse::<f64>().unwrap(),
                a[1].as_str().unwrap().parse::<f64>().unwrap(),
            ];
            let e = [
                e[0].as_str().unwrap().parse::<f64>().unwrap(),
                e[1].as_str().unwrap().parse::<f64>().unwrap(),
            ];
            let error = (a[0] - e[0]).hypot(a[1] - e[1]) / 1_f64.max(e[0].hypot(e[1]));
            assert!(
                error <= 1e-9,
                "coefficient mismatch: {a:?}, {e:?}, scaled error {error:e}"
            );
        }
    }

    #[test]
    fn compiled_coefficients_preserve_synthetic_and_physical_reductions() {
        for fixture in [
            "examples/graphs/bubble-parameterized.json",
            "examples/graphs/triangle-reference.json",
            "examples/graphs/physical/four-photon/top-box-0.json",
        ] {
            let control = value(run("reduce", &[fixture, "--batch-size", "1"]));
            for backend in ["symjit-o2", "symjit-o2-factored"] {
                for mode in ["expansion", "sampling"] {
                    let actual = value(run(
                        "reduce",
                        &[
                            fixture,
                            "--batch-size",
                            "3",
                            "--mu-extraction",
                            mode,
                            "--numerator-evaluator",
                            backend,
                        ],
                    ));
                    assert_eq!(actual["numerator_provider"]["evaluator"], backend);
                    assert_eq!(
                        actual["numerator_provider"]["coefficients_factored"],
                        backend == "symjit-o2-factored"
                    );
                    assert_eq!(actual["numerator_provider"]["optimization_level"], 2);
                    assert!(
                        actual["numerator_provider"]["coefficient_outputs"]
                            .as_u64()
                            .unwrap()
                            > 0
                    );
                    let rows = actual["results"].as_array().unwrap();
                    let expected = control["results"].as_array().unwrap();
                    assert_eq!(rows.len(), expected.len());
                    for (a, e) in rows.iter().zip(expected) {
                        assert_eq!(a["status"]["kind"], "success");
                        assert_eq!(e["status"]["kind"], "success");
                        near(&a["values"], &e["values"]);
                    }
                }
            }
        }
    }

    #[test]
    fn benchmark_evaluates_coefficient_heads_for_each_sample_including_tails() {
        let fixture = "examples/graphs/bubble-parameterized.json";
        for backend in ["symjit-o2", "symjit-o2-factored"] {
            let reduced = value(run("reduce", &[fixture, "--numerator-evaluator", backend]));
            let report = value(run(
                "bench",
                &[
                    fixture,
                    "--numerator-evaluator",
                    backend,
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
            assert_eq!(report["performance_acceptance"], false);
            assert_eq!(report["numerator_provider"]["evaluator"], backend);
            let cases = report["cases"].as_array().unwrap();
            assert_eq!(cases.len(), 4);
            for case in cases {
                let width = case["batch_size"].as_u64().unwrap();
                let timing = &case["diagnostic"]["numerator_coefficients"];
                assert_eq!(timing["calls"].as_u64().unwrap(), 3_u64.div_ceil(width));
                assert_eq!(timing["rows"], 3);
                assert!(timing["nanoseconds"].as_f64().unwrap() > 0.);
                assert_eq!(case["diagnostic"]["failed_lanes"], 0);
                let lane = case["representative_source_lane"].as_u64().unwrap() as usize;
                near(
                    &case["representative_result"],
                    &reduced["results"][lane]["values"],
                );
            }
        }
    }
}
