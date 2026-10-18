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
fn arbitrary_precision_light_like_invariants_are_exact_at_every_working_precision() {
    use symbolica::domains::float::{Float, Real};
    // The outgoing photons have non-dyadic momentum components. Differencing
    // their rounded momenta left p^2 = +2^-114 (2 tracked bits) at 128 bits and
    // similar noise at 140 bits, which oneloopmaster's four-mass D0 evaluated as
    // an O(1) wrong value with 8 claimed bits (finite -4.91 instead of -8.26e-4).
    // Expected finite and cut-constructible parts from a 400-bit evaluation.
    let expected = [
        [
            "-8.26343963451607199327232131951182815e-4",
            "0",
            "2.29373690232880160538221151266970926e-2",
            "0",
        ],
        [
            "-9.91094989096783794136374014407567277e-3",
            "4.97389490220958867010833792423716985e-4",
            "-4.29270331561046487337059330930346585e-3",
            "4.97389490220958867010833792423716985e-4",
        ],
        [
            "-1.27555267800816381919520138064847241e-2",
            "-4.72886849400088475610167227847162747e-3",
            "-1.25747183247289297700359623790057446e-2",
            "-4.72886849400088475610167227847162747e-3",
        ],
    ];
    let fixture = "examples/graphs/physical/four-photon/top-box-0-pool.json";
    let parse = |text: &str| Float::parse(text, Some(320)).unwrap();
    for bits in ["128", "140"] {
        let actual = value(run(&[fixture, "--precision", "float", "--bits", bits]));
        for (lane, expected) in expected.iter().enumerate() {
            let row = &actual["results"][lane];
            assert_eq!(row["status"]["kind"], "success");
            assert_eq!(row["s_matrix_source"], "momenta");
            for (index, (coefficient, part)) in
                [(0, 0), (0, 1), (3, 0), (3, 1)].into_iter().enumerate()
            {
                let value = parse(row["values"][coefficient][part].as_str().unwrap());
                let error = (value - parse(expected[index])).norm();
                assert!(
                    error < parse("1e-30"),
                    "bits={bits} lane={lane} coefficient={coefficient} part={part} error={error}"
                );
            }
        }
    }
}

#[test]
fn tied_bubble_references_keep_the_rational_split_at_every_working_precision() {
    use symbolica::domains::float::{Float, Real};
    // The four-photon top boxes lie in the x-z plane, so on several double
    // cuts rational bubble references related by y -> -y score exactly alike.
    // The reference fixes how the bubble residue splits between B1/B11, its
    // mu^2 rational term and the tadpoles' A0 subtraction; only the sum is
    // reference independent. Round-off used to break the tie, and lane 3
    // (generic polarizations) moved 1375.2-1373.5i between
    // cut_constructible_finite and rational at 68, 80, 88, 92, 100, 104,
    // 116, 128, 140, 148, ... bits while the finite part stayed right.
    // Expected lane-3 finite, cut-constructible and rational parts from a
    // 400-bit evaluation (identical for both routings).
    let expected = [
        [
            "-1.71910721957404256939336081182328086e-1",
            "5.48242285239929453402606817224109084e-2",
        ],
        [
            "5.26619768189329847649199775374183929e2",
            "-5.89552044813902443475672873641628372e2",
        ],
        [
            "-5.26791678911287251906139111455366257e2",
            "5.89606869042426436421013134323350783e2",
        ],
    ];
    let parse = |text: &str| Float::parse(text, Some(320)).unwrap();
    let fixtures = [
        "examples/graphs/physical/four-photon/top-box-1.json",
        "examples/graphs/physical/four-photon/top-box-2.json",
    ];
    // Bit counts at which the tie used to go the other way; the reductions
    // are independent, so run them concurrently.
    let bits = ["100", "116", "140"];
    let children: Vec<_> = fixtures
        .iter()
        .flat_map(|fixture| bits.iter().map(move |bits| (fixture, bits)))
        .map(|(fixture, bits)| {
            let child = Command::new(env!("CARGO_BIN_EXE_ninjars"))
                .current_dir(root())
                .args(["reduce", fixture, "--precision", "float", "--bits", bits])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .unwrap();
            (fixture, bits, child)
        })
        .collect();
    for (fixture, bits, child) in children {
        let actual = value(child.wait_with_output().unwrap());
        let row = &actual["results"][3];
        assert_eq!(row["status"]["kind"], "success", "{fixture} bits={bits}");
        for (index, coefficient) in [0, 3, 4].into_iter().enumerate() {
            for part in 0..2 {
                let value = parse(row["values"][coefficient][part].as_str().unwrap());
                let error = (value - parse(expected[index][part])).norm();
                assert!(
                    error < parse("1e-18"),
                    "{fixture} bits={bits} coefficient={coefficient} part={part} error={error}"
                );
            }
        }
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

#[test]
fn native_block_rows_reject_invalid_bounds_selections_and_builds() {
    for bound in ["0", "-1", "1.5", "184467440737095516160"] {
        assert!(
            !run(&[
                "missing.json",
                "--reduction-kernels",
                "native",
                &format!("--native-block-rows={bound}")
            ])
            .status
            .success()
        );
    }
    for selection in ["symbolica", "interpreted-rank"] {
        let output = run(&[
            "missing.json",
            "--reduction-kernels",
            selection,
            "--native-block-rows",
            "8",
        ]);
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("requires a native reduction-kernel selection")
        );
    }
    for selection in ["native-denominators", "native", "native-rank"] {
        if !cfg!(feature = "native-kernels") {
            let output = run(&[
                "missing.json",
                "--reduction-kernels",
                selection,
                "--native-block-rows",
                "8",
            ]);
            assert!(!output.status.success());
            assert!(
                String::from_utf8_lossy(&output.stderr)
                    .contains("requires a build with --features native-kernels")
            );
        }
    }
    #[cfg(feature = "native-kernels")]
    {
        let output = run(&[
            "examples/graphs/bubble.json",
            "--reduction-kernels",
            "native-rank",
            "--native-block-rows",
            "8",
        ]);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("requires n=5 and rank=4"));
    }
}

#[cfg(feature = "native-kernels")]
#[test]
fn native_block_rows_preserve_serialized_values_tail_and_kernel_coverage() {
    for selection in ["native-denominators", "native", "native-rank"] {
        for precision in ["f64", "double-float", "float"] {
            let args = [
                "examples/graphs/pentagon-rank4.json",
                "--reduction-kernels",
                selection,
                "--precision",
                precision,
                "--batch-size",
                "2",
            ];
            let whole = value(run(&args));
            assert_eq!(whole["results"].as_array().unwrap().len(), 3);
            assert_eq!(
                whole["reduction_kernels"]["native_execution"],
                json!({"policy":"whole","block_rows":null,"scope":"native-kernel-rows"})
            );
            for bound in ["1", "7", "1024"] {
                let mut blocked_args = args.to_vec();
                blocked_args.extend(["--native-block-rows", bound]);
                let blocked = value(run(&blocked_args));
                assert_eq!(blocked["results"].as_array().unwrap().len(), 3);
                assert_eq!(
                    blocked["reduction_kernels"]["native_execution"],
                    json!({"policy":"fixed-rows","block_rows":bound.parse::<usize>().unwrap(),"scope":"native-kernel-rows"})
                );
                let mut coverage = blocked["reduction_kernels"].clone();
                coverage["native_execution"] =
                    whole["reduction_kernels"]["native_execution"].clone();
                assert_eq!(coverage, whole["reduction_kernels"]);
                // Serialized parity; binary64 parsing is not an additional-accuracy test.
                for (actual, expected) in blocked["results"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(whole["results"].as_array().unwrap())
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
}

/// Seven propagators, rank n+1 (`heptagon-rank8.json`): the generic engine
/// reduces beyond six propagators in every precision. Its results agree with
/// the binary64 fast engine, and DoubleFloat and Float at two working
/// precisions agree to their own accuracy. The single pole vanishes (no UV or
/// IR divergence at rank 8 < 2n-4 with massive propagators).
#[test]
fn seven_propagators_reduce_in_every_precision_and_engine() {
    let fixture = "examples/graphs/heptagon-rank8.json";
    let laurent = |document: &Value| -> Vec<[(f64, f64); 3]> {
        assert_eq!(document["success"], true);
        assert_eq!(document["propagator_edges"].as_array().unwrap().len(), 7);
        assert_eq!(document["rank"], 8);
        document["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                assert_eq!(row["status"]["kind"], "success");
                std::array::from_fn(|k| (component(row, k, 0), component(row, k, 1)))
            })
            .collect()
    };
    let scaled = |actual: &[[(f64, f64); 3]], expected: &[[(f64, f64); 3]]| {
        actual
            .iter()
            .zip(expected)
            .map(|(a, b)| {
                let scale = b.iter().map(|z| z.0.hypot(z.1)).fold(0., f64::max);
                (0..3)
                    .map(|k| (a[k].0 - b[k].0).hypot(a[k].1 - b[k].1) / scale)
                    .fold(0., f64::max)
            })
            .fold(0., f64::max)
    };
    let reference = value(run(&[fixture, "--precision", "float", "--bits", "300"]));
    let exact = laurent(&reference);
    assert_eq!(exact.len(), 2);
    for lane in &exact {
        assert!(lane[1].0.hypot(lane[1].1) < 1e-80 * lane[0].0.hypot(lane[0].1));
    }
    for (options, tolerance) in [
        (&["--precision", "f64"][..], 1e-10),
        (&["--engine", "fast"][..], 1e-9),
        (&["--precision", "double-float"][..], 1e-27),
        (&["--precision", "float", "--bits", "200"][..], 1e-54),
        (&["--precision", "float"][..], 1e-70),
    ] {
        let mut args = vec![fixture, "--batch-size", "1"];
        args.extend_from_slice(options);
        let document = value(run(&args));
        let error = scaled(&laurent(&document), &exact);
        assert!(error <= tolerance, "{options:?}: scaled error {error:e}");
        eprintln!("NINJARS_HEPTAGON {options:?} scaled_error={error:e}");
    }
}

/// `--stability-test` is opt-in. The H -> gg top triangle with FDH R2 has a
/// double-longitudinal lane whose amplitude vanishes exactly (a Ward
/// identity): its result is rounding noise of the summed cut contributions,
/// which no relative accuracy test can certify, so that lane alone is
/// reported unstable (and the run fails) only when the test is requested.
#[test]
fn stability_test_is_opt_in_and_flags_exact_cancellations() {
    let fixture = "examples/graphs/physical/hgg/top-triangle-0.json";
    let default = value(run(&[fixture, "--r2", "fdh"]));
    let statuses = |document: &Value| -> Vec<Value> {
        document["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["status"]["kind"].clone())
            .collect()
    };
    assert_eq!(statuses(&default), vec![json!("success"); 3]);
    let tested = run(&[fixture, "--r2", "fdh", "--stability-test"]);
    assert!(!tested.status.success());
    let tested: Value = serde_json::from_slice(&tested.stdout).unwrap();
    assert_eq!(
        statuses(&tested),
        [json!("success"), json!("success"), json!("unstable")]
    );
}
