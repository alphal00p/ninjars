//! `ninjars case-bench` options used by the rank n+2 benchmark
//! (`validation/benchmark/collier`): the generic-engine breakdown
//! (`--engines generic`) and the fixed-work mode for instruction counting
//! (`--total-calls`).

use serde_json::Value;
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
        .arg("case-bench")
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

/// Rank n+2 cases without and with integrand `mu^2` blocks.
const CASES: [&str; 2] = [
    "validation/benchmark/synthetic-rank-n2/syn-n3-r5.json",
    "validation/rank-n2/cases/rn2-int-complex-n4-r6-mu2.json",
];

#[test]
fn generic_engine_breakdown() {
    for case in CASES {
        let report = value(run(&[
            case,
            "--engines",
            "fast,generic",
            "--widths",
            "1,4",
            "--blocks",
            "1",
            "--min-block-ms",
            "0",
        ]));
        let entry = &report["cases"][0];
        assert_eq!(entry["per_point_timing"].as_array().unwrap().len(), 2);
        let generic = entry["generic_per_point_timing"].as_array().unwrap();
        assert_eq!(generic.len(), 2, "{case}");
        for width in generic {
            assert_eq!(width["failed_lanes"], 0, "{case}");
            assert_eq!(width["replay_bitwise"], true, "{case}");
            let summary = &width["summary_ns_per_point"];
            for key in ["numerator", "reduction", "masters", "total"] {
                assert!(summary[key].as_f64().unwrap().is_finite(), "{case} {key}");
            }
            let rows = width["master_rows_per_batch"].as_object().unwrap();
            assert!(rows.contains_key("B1111"), "{case}: rank n+2 bubbles");
        }
        assert!(report["generic_boundary"].is_string());
    }
    // Generic only: no fast-engine entry.
    let report = value(run(&[
        CASES[0],
        "--engines",
        "generic",
        "--widths",
        "1",
        "--blocks",
        "1",
        "--min-block-ms",
        "0",
    ]));
    assert!(report["cases"][0].get("per_point_timing").is_none());
    assert!(!run(&[CASES[0], "--engines", "slow"]).status.success());
}

#[test]
fn total_calls_is_fixed_work() {
    let report = value(run(&[
        CASES[1],
        "--widths",
        "1,3",
        "--total-calls",
        "2",
        "--reference-dir",
        "validation/rank-n2/reference",
    ]));
    for width in report["cases"][0]["per_point_timing"].as_array().unwrap() {
        assert_eq!(width["total_calls"], 2);
        assert_eq!(width["failed_lanes"], 0);
        assert!(width.get("summary_ns_per_point").is_none());
        assert!(width["reference_check"]["lanes_checked"].as_u64().unwrap() > 0);
    }
}
