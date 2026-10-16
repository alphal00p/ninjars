use serde_json::Value;
use std::{
    path::PathBuf,
    process::{Command, Output},
};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ninjars"))
        .current_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
        .arg("reduce")
        .args(args)
        .output()
        .unwrap()
}
fn result(args: &[&str]) -> Value {
    let output = run(args);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
fn compare(a: &Value, b: &Value) {
    for (a, b) in a["results"]
        .as_array()
        .unwrap()
        .iter()
        .zip(b["results"].as_array().unwrap())
    {
        for coefficient in 0..5 {
            for part in 0..2 {
                let x: f64 = a["values"][coefficient][part]
                    .as_str()
                    .unwrap()
                    .parse()
                    .unwrap();
                let y: f64 = b["values"][coefficient][part]
                    .as_str()
                    .unwrap()
                    .parse()
                    .unwrap();
                assert!((x - y).abs() < 1e-9 * (1. + y.abs()), "{x} != {y}");
            }
        }
    }
}

#[test]
fn automatic_box_scales_have_lane_provenance_and_match_manual_results_with_tails() {
    let fixture = "examples/graphs/box.json";
    let single = result(&[
        fixture,
        "--mu-extraction",
        "sampling",
        "--full-box-coefficients",
        "--batch-size",
        "1",
    ]);
    let tail = result(&[
        fixture,
        "--mu-extraction",
        "sampling",
        "--full-box-coefficients",
        "--batch-size",
        "2",
        "--diagnostics",
    ]);
    let expansion = result(&[
        fixture,
        "--mu-extraction",
        "expansion",
        "--full-box-coefficients",
    ]);
    let manual = result(&[
        fixture,
        "--mu-extraction",
        "sampling",
        "--full-box-coefficients",
        "--mu-sampling-scale",
        "2",
    ]);
    compare(&single, &tail);
    compare(&tail, &expansion);
    compare(&tail, &manual);
    assert_eq!(tail["box_reduction"]["mu_sampling_scale_mode"], "automatic");
    let scales: Vec<_> = tail["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let selected = &row["mu_sampling_scale"];
            assert_eq!(selected["mode"], "automatic");
            assert_eq!(selected["used"], true);
            assert_eq!(selected["source"]["kind"], "invariant");
            assert_eq!(
                selected["source"]["propagators"].as_array().unwrap().len(),
                2
            );
            selected["value"][0]
                .as_str()
                .unwrap()
                .parse::<f64>()
                .unwrap()
        })
        .collect();
    assert_eq!(scales.len(), 3);
    assert!(scales.windows(2).all(|v| v[0] != v[1]));
    for row in manual["results"].as_array().unwrap() {
        assert_eq!(row["mu_sampling_scale"]["mode"], "manual");
        assert_eq!(row["mu_sampling_scale"]["expression"], "2");
    }
}

#[test]
fn exact_manual_scale_preserves_extra_precision_and_zero_override_is_rejected() {
    use symbolica::domains::float::Float;
    for precision in ["double-float", "float"] {
        let actual = result(&[
            "examples/graphs/bubble.json",
            "--precision",
            precision,
            "--mu-sampling-scale",
            "1+1/10^28",
        ]);
        let selected = &actual["results"][0]["mu_sampling_scale"];
        assert_eq!(selected["mode"], "manual");
        assert_eq!(selected["used"], false);
        let value = Float::parse(selected["value"][0].as_str().unwrap(), Some(256)).unwrap();
        let normalized =
            (value - Float::with_val(256, 1)) / Float::parse("1e-28", Some(256)).unwrap();
        assert!(
            normalized > Float::parse("0.999", Some(256)).unwrap()
                && normalized < Float::parse("1.001", Some(256)).unwrap(),
            "{normalized}"
        );
    }
    let output = run(&["examples/graphs/box.json", "--mu-sampling-scale", "0"]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("must be nonzero"));
}

#[test]
fn default_scale_uses_physical_lane_values_even_when_lower_cuts_do_not_need_it() {
    let actual = result(&["examples/graphs/bubble.json", "--batch-size", "2"]);
    for (row, expected) in actual["results"]
        .as_array()
        .unwrap()
        .iter()
        .zip([1., 4., 37. / 16.])
    {
        assert_eq!(row["mu_sampling_scale"]["mode"], "automatic");
        assert_eq!(row["mu_sampling_scale"]["used"], false);
        let value: f64 = row["mu_sampling_scale"]["value"][0]
            .as_str()
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(value, expected);
    }
}
