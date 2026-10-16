use serde_json::Value;
use std::{path::PathBuf, process::Command};

#[test]
fn prepared_box_matches_reference_across_precisions_modes_and_batch_tails() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let evidence: Value =
        serde_json::from_str(include_str!("../../../validation/evidence/box-cli.json")).unwrap();
    for precision in ["f64", "double-float", "float"] {
        for mode in ["expansion", "sampling"] {
            for batch in ["1", "2"] {
                let mut command = Command::new(env!("CARGO_BIN_EXE_ninjars"));
                command.current_dir(&root).args([
                    "reduce",
                    "examples/graphs/box.json",
                    "--precision",
                    precision,
                    "--batch-size",
                    batch,
                    "--mu-extraction",
                    mode,
                    "--full-box-coefficients",
                    "--mu-sampling-scale",
                    "2",
                ]);
                if precision == "float" {
                    command.args(["--bits", "256"]);
                }
                let output = command.output().unwrap();
                assert!(
                    output.status.success(),
                    "{precision}/{mode}/{batch}: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                let result: Value = serde_json::from_slice(&output.stdout).unwrap();
                assert_eq!(result["success"], true);
                assert_eq!(result["box_reduction"]["mu_extraction"], mode);
                assert_eq!(result["box_reduction"]["coefficient_mode"], "full");
                assert_eq!(result["results"].as_array().unwrap().len(), 3);
                for (row, reference) in result["results"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(evidence["reference_checks"].as_array().unwrap())
                {
                    assert_eq!(row["lane"], reference["lane"]);
                    assert_eq!(row["status"]["kind"], "success");
                    let value = |coefficient: usize, part: usize| {
                        row["values"][coefficient][part]
                            .as_str()
                            .unwrap()
                            .parse::<f64>()
                            .unwrap()
                    };
                    for coefficient in 0..3 {
                        for part in 0..2 {
                            let expected = reference["reference"]["epsilon"][coefficient][part]
                                .as_f64()
                                .unwrap();
                            let actual = value(coefficient, part);
                            assert!(
                                (actual - expected).abs() <= 1e-9 * (1. + expected.abs()),
                                "{precision}/{mode}/{batch},lane{},component{coefficient}/{part}: {actual} != {expected}",
                                row["lane"]
                            );
                        }
                    }
                    for part in 0..2 {
                        assert!(
                            (value(0, part) - value(3, part) - value(4, part)).abs()
                                <= 1e-9 * (1. + value(0, part).abs())
                        );
                    }
                }
            }
        }
    }
}
