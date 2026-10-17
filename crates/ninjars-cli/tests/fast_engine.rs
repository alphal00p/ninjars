//! `reduce`/`bench --engine fast` against the generic engine and original Ninja.
//!
//! Laurent coefficients use the CLI's scaled criterion
//! `|actual - expected| / (1 + |expected|) <= 1e-9` (as `compare`). The
//! cut-constructible/rational split depends on the cut basis, so it is compared
//! with original Ninja's values (same real basis as the fast engine) rather
//! than with the generic engine, where the repository has them:
//! `validation/evidence/prepared-graphs.json` and
//! `validation/benchmark/ninja/*.json` for the `*-pool.json` fixtures. Those
//! comparisons are also checked relative to each lane's largest Ninja value.

use serde_json::{Value, json};
use std::{
    path::PathBuf,
    process::{Command, Output},
};

const TOLERANCE: f64 = 1e-9;

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

fn reduce(fixture: &str, options: &[&str]) -> Value {
    value(run("reduce", &[&[fixture], options].concat()))
}

type C = (f64, f64);

fn parse(pair: &Value) -> C {
    let part = |index: usize| match &pair[index] {
        Value::String(text) => text.parse::<f64>().unwrap(),
        number => number.as_f64().unwrap(),
    };
    (part(0), part(1))
}

fn components(row: &Value) -> Vec<C> {
    row["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(parse)
        .collect()
}

fn abs((re, im): C) -> f64 {
    re.hypot(im)
}

fn difference(a: C, b: C) -> f64 {
    abs((a.0 - b.0, a.1 - b.1))
}

/// The CLI's scaled error of one complex coefficient.
fn scaled(actual: C, expected: C) -> f64 {
    difference(actual, expected) / (1. + abs(expected))
}

fn assert_laurent_agrees(label: &str, actual: &[C], expected: &[C]) {
    for k in 0..3 {
        let error = scaled(actual[k], expected[k]);
        assert!(
            error <= TOLERANCE,
            "{label} coefficient {k}: {:?} != {:?} (scaled error {error:e})",
            actual[k],
            expected[k]
        );
    }
}

/// All five components against original Ninja, with the CLI criterion and
/// relative to the lane's largest Ninja Laurent coefficient.
fn assert_matches_ninja(label: &str, actual: &[C], expected: &[C]) {
    assert_laurent_agrees(label, actual, expected);
    let lane_scale = expected[..3].iter().copied().map(abs).fold(0., f64::max);
    assert!(lane_scale > 0., "{label}: vanishing reference");
    for k in 0..5 {
        let cli = scaled(actual[k], expected[k]);
        let relative = difference(actual[k], expected[k]) / lane_scale;
        assert!(
            cli <= TOLERANCE && relative <= TOLERANCE,
            "{label} component {k}: {:?} != Ninja {:?} (scaled {cli:e}, lane-relative {relative:e})",
            actual[k],
            expected[k]
        );
    }
}

/// The fast engine's finite part is its cut-constructible plus rational part.
fn assert_split_consistent(label: &str, values: &[C]) {
    let sum = (values[3].0 + values[4].0, values[3].1 + values[4].1);
    assert!(
        scaled(values[0], sum) <= 1e-12,
        "{label}: finite {:?} != cut-constructible + rational {sum:?}",
        values[0]
    );
}

/// Run both engines and compare every lane; returns the fast document.
fn compare_engines(fixture: &str, options: &[&str]) -> Value {
    compare_engines_except(fixture, options, &[])
}

/// As [`compare_engines`]; `cancellation_lanes` are known binary64 pole
/// cancellations (a vanishing pole beside a large finite part), compared only
/// relative to the lane's largest Laurent coefficient.
fn compare_engines_except(fixture: &str, options: &[&str], cancellation_lanes: &[u64]) -> Value {
    let generic = reduce(fixture, options);
    let fast = reduce(fixture, &[options, &["--engine", "fast"]].concat());
    assert_eq!(generic["engine"], "generic");
    assert_eq!(fast["engine"], "fast");
    assert_eq!(fast["success"], true);
    assert_eq!(fast["precision"], json!({"carrier":"f64","bits":53}));
    assert_eq!(fast["rank"], generic["rank"]);
    assert_eq!(fast["result_order"], generic["result_order"]);
    assert_eq!(fast["number_format"], generic["number_format"]);
    assert_eq!(fast["reduction_kernels"]["selection"], "fast");
    assert_eq!(fast["box_reduction"]["mu_extraction"], "expansion");
    let (generic_rows, fast_rows) = (
        generic["results"].as_array().unwrap(),
        fast["results"].as_array().unwrap(),
    );
    assert_eq!(generic_rows.len(), fast_rows.len());
    for (expected, actual) in generic_rows.iter().zip(fast_rows) {
        let label = format!("{fixture} lane {}", actual["lane"]);
        assert_eq!(actual["lane"], expected["lane"]);
        assert_eq!(actual["status"], json!({"kind":"success"}), "{label}");
        assert_eq!(expected["status"]["kind"], "success", "{label}");
        assert_eq!(actual["s_matrix_source"], expected["s_matrix_source"]);
        assert_eq!(actual["component_precision_bits"], json!(vec![[53, 53]; 5]));
        let values = components(actual);
        assert_eq!(values.len(), 5);
        let expected = components(expected);
        if cancellation_lanes.contains(&actual["lane"].as_u64().unwrap()) {
            let lane_scale = expected[..3].iter().copied().map(abs).fold(0., f64::max);
            for k in 0..3 {
                let relative = difference(values[k], expected[k]) / lane_scale;
                assert!(
                    relative <= TOLERANCE,
                    "{label} coefficient {k}: {relative:e}"
                );
            }
        } else {
            assert_laurent_agrees(&label, &values, &expected);
        }
        assert_split_consistent(&label, &values);
    }
    fast
}

/// Original Ninja's prepared-graph evidence (bubble and timelike triangle).
fn prepared_graph_checks(fixture: &str, fast: &Value) -> usize {
    let evidence: Value = serde_json::from_str(include_str!(
        "../../../validation/evidence/prepared-graphs.json"
    ))
    .unwrap();
    let mut checked = 0;
    for check in evidence["checks"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|check| check["fixture"] == fixture)
    {
        let reference = &check["reference"];
        assert_eq!(reference["status"], 0);
        let mut expected: Vec<C> = reference["epsilon"]
            .as_array()
            .unwrap()
            .iter()
            .map(parse)
            .collect();
        expected.push(parse(&reference["cut_constructible"]));
        expected.push(parse(&reference["rational"]));
        let lane = check["lane"].as_u64().unwrap() as usize;
        assert_matches_ninja(
            &format!("{fixture} lane {lane} vs original Ninja"),
            &components(&fast["results"][lane]),
            &expected,
        );
        checked += 1;
    }
    checked
}

/// A temporary fixture next to a copy of an existing graph.
struct TemporaryFixture {
    directory: PathBuf,
}

impl TemporaryFixture {
    fn new(label: &str, graph: &str, mut fixture: Value) -> Self {
        let directory =
            std::env::temp_dir().join(format!("ninjars-fast-{label}-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let source = root().join("examples/graphs").join(graph);
        std::fs::copy(source, directory.join(graph)).unwrap();
        fixture["graph"] = graph.into();
        std::fs::write(
            directory.join("input.json"),
            serde_json::to_vec(&fixture).unwrap(),
        )
        .unwrap();
        Self { directory }
    }

    fn path(&self) -> String {
        self.directory
            .join("input.json")
            .to_string_lossy()
            .into_owned()
    }
}

impl Drop for TemporaryFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

#[test]
fn fast_engine_matches_generic_and_original_ninja_on_synthetic_fixtures() {
    // Every synthetic numerator has explicit mu^2 terms, so the fast engine
    // runs the polynomial numerator behind its generic adapter.
    let mut checked = 0;
    for fixture in [
        "examples/graphs/bubble.json",
        "examples/graphs/box.json",
        "examples/graphs/triangle-reference.json",
    ] {
        let fast = compare_engines(fixture, &["--batch-size", "2"]);
        assert_eq!(
            fast["numerator_provider"]["interface"],
            "generic-polynomial"
        );
        assert_eq!(fast["numerator_provider"]["evaluator"], "symbolica");
        checked += prepared_graph_checks(fixture, &fast);
    }
    assert_eq!(checked, 4, "bubble and triangle original-Ninja evidence");

    // The pentagon graph and numerator (mu2*q0^2 and mu2^2 terms) at timelike
    // points; see the spacelike test below for pentagon-rank4.json itself.
    let pentagon = TemporaryFixture::new(
        "pentagon",
        "pentagon-rank4.dot",
        json!({
            "schema_version": 1, "model": "phi3", "scale_squared": "1",
            "parameter_overrides": {"UFO::mass": "1"},
            "samples": [
                {"external_momenta": {"0": ["-3","1","-1","0"], "1": ["-4","0","1","-1"],
                    "2": ["-9","0","0","-1"], "3": ["-6","1","2","1/2"]}},
                {"external_momenta": {"0": ["-5/2","1","-1/2","0"], "1": ["-3","1/2","1","-1"],
                    "2": ["-8","0","1/2","-1"], "3": ["-11/2","1","2","1"]}},
                {"external_momenta": {"0": ["-2","1","-1","0"], "1": ["-3","0","1","-1"],
                    "2": ["-7","0","0","-1"], "3": ["-5","1","2","3/2"]}}
            ]
        }),
    );
    let fast = compare_engines(&pentagon.path(), &["--batch-size", "2"]);
    assert_eq!(fast["rank"], 4);
    assert_eq!(fast["propagator_edges"].as_array().unwrap().len(), 5);
}

#[test]
fn fast_engine_reports_spacelike_cut_planes_as_failed_lanes() {
    // Original Ninja's real cut basis does not exist for purely spacelike cut
    // planes (validation/reference/README.md, exceptional-spacelike-triangle):
    // the Euclidean triangle and every pentagon-rank4.json point have them.
    // The generic engine reduces these points; the fast engine, a port of
    // original Ninja, reports failed lanes without values and exits nonzero.
    for (fixture, status) in [
        (
            "examples/graphs/triangle.json",
            json!({"kind":"nonfinite_result"}),
        ),
        (
            "examples/graphs/pentagon-rank4.json",
            json!({"kind":"cut_failed","cut_index":null,"reason":"Unstable"}),
        ),
    ] {
        let output = run("reduce", &[fixture, "--engine", "fast"]);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("failed reduction"));
        let document: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(document["engine"], "fast");
        assert_eq!(document["success"], false);
        let rows = document["results"].as_array().unwrap();
        assert!(!rows.is_empty());
        for row in rows {
            assert_eq!(row["status"], status, "{fixture}");
            assert!(row["values"].is_null());
            assert!(row["component_precision_bits"].is_null());
        }
        assert_eq!(reduce(fixture, &[])["success"], true);
    }
}

/// Physical `*-pool.json` fixtures and their original-Ninja results.
const POOLS: [(&str, &str); 7] = [
    ("hgg/top-triangle-0", "hgg-top-triangle-0"),
    ("gg-hh/top-triangle-0", "gg-hh-top-triangle-0"),
    ("gg-hh/top-box-2", "gg-hh-top-box-2"),
    ("gg-hhh/top-triangle-0", "gg-hhh-top-triangle-0"),
    ("gg-hhh/top-box-8", "gg-hhh-top-box-8"),
    ("gg-hhh/top-pentagon-26", "gg-hhh-top-pentagon-26"),
    ("four-photon/top-box-0", "four-photon-top-box-0"),
];

fn pool(name: &str) -> String {
    format!("examples/graphs/physical/{name}-pool.json")
}

/// Compare a fast document with original Ninja's results, matched by sample id.
fn ninja_pool_checks(fast: &Value, case: &str) {
    let reference: Value = serde_json::from_slice(
        &std::fs::read(root().join(format!("validation/benchmark/ninja/{case}.json"))).unwrap(),
    )
    .unwrap();
    let samples = fast["input_fixture"]["samples"].as_array().unwrap();
    for (row, sample) in fast["results"].as_array().unwrap().iter().zip(samples) {
        let id = &sample["id"];
        let lane = reference["lanes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|lane| lane["id"] == *id)
            .unwrap_or_else(|| panic!("{case}: no Ninja result for {id}"));
        assert_eq!(lane["return_status"], 0);
        let expected: Vec<C> = ["eps0", "eps_m1", "eps_m2", "cut_constructible", "rational"]
            .iter()
            .map(|key| parse(&lane[key]))
            .collect();
        assert_matches_ninja(
            &format!("{case} {id} vs original Ninja"),
            &components(row),
            &expected,
        );
    }
}

#[test]
fn fast_engine_matches_generic_and_original_ninja_on_physical_fixtures() {
    for (name, case) in POOLS {
        let fast = compare_engines(&pool(name), &[]);
        assert_eq!(fast["numerator_provider"]["interface"], "tensor", "{name}");
        assert_eq!(fast["results"].as_array().unwrap().len(), 3);
        ninja_pool_checks(&fast, case);
    }
    // Explicit polarizations, including a vanishing helicity amplitude (lane
    // 1). Lane 2 is the known binary64 cancellation of tests/physical.rs: its
    // pole coefficients vanish beside a finite part of 1.4e6, so both engines
    // leave ~1e-9 round-off there.
    let fast = compare_engines_except(
        "examples/graphs/physical/hgg/top-triangle-1.json",
        &["--batch-size", "3"],
        &[2],
    );
    assert_eq!(fast["numerator_provider"]["interface"], "tensor");
    // A larger box sample set.
    let fast = compare_engines(
        "examples/graphs/physical/gg-hh/top-box-3.json",
        &["--batch-size", "3"],
    );
    assert_eq!(fast["results"].as_array().unwrap().len(), 8);
}

#[test]
fn fast_engine_honours_cut_stop_and_is_independent_of_batching() {
    // A box-only reduction keeps the basis-independent box coefficients.
    let fast = compare_engines("examples/graphs/box.json", &["--cut-stop", "4"]);
    assert_eq!(fast["cut_stop"], 4);
    assert_eq!(fast["requested_cut_stop"], 4);
    let full = reduce("examples/graphs/box.json", &["--engine", "fast"]);
    assert_eq!(full["cut_stop"], 1);
    assert_ne!(full["results"][0]["values"], fast["results"][0]["values"]);
    // Stops above the cut sizes leave zero contributions, as in the generic engine.
    let empty = reduce(
        "examples/graphs/bubble.json",
        &["--engine", "fast", "--cut-stop", "3"],
    );
    for row in empty["results"].as_array().unwrap() {
        assert!(components(row).iter().all(|value| *value == (0., 0.)));
    }
    // Lanes are reduced independently: batches and tails are bitwise identical.
    let fixture = pool("gg-hh/top-box-2");
    let single = reduce(&fixture, &["--engine", "fast", "--batch-size", "1"]);
    for width in ["2", "32"] {
        let batched = reduce(&fixture, &["--engine", "fast", "--batch-size", width]);
        assert_eq!(batched["results"], single["results"]);
        assert_eq!(batched["batch_size"], width.parse::<u64>().unwrap());
    }
    let diagnostics = reduce(&fixture, &["--engine", "fast", "--diagnostics"]);
    assert!(diagnostics["numerator_template"].is_string());
    assert!(diagnostics["results"][0]["cut_counts"].is_null());
}

#[cfg(feature = "numerator-jit")]
#[test]
fn fast_engine_symjit_coefficient_heads_match_interpreted_heads() {
    for fixture in [
        pool("hgg/top-triangle-0"),
        pool("gg-hh/top-box-2"),
        "examples/graphs/bubble.json".into(),
    ] {
        let interpreted = reduce(&fixture, &["--engine", "fast"]);
        let interface = interpreted["numerator_provider"]["interface"].clone();
        for evaluator in ["symjit-o2", "symjit-o2-factored"] {
            let jit = reduce(
                &fixture,
                &["--engine", "fast", "--numerator-evaluator", evaluator],
            );
            let provider = &jit["numerator_provider"];
            assert_eq!(provider["evaluator"], evaluator);
            assert_eq!(provider["interface"], interface);
            assert_eq!(provider["optimization_level"], 2);
            assert_eq!(
                provider["coefficients_factored"],
                evaluator == "symjit-o2-factored"
            );
            assert!(provider["coefficient_outputs"].as_u64().unwrap() > 0);
            for (actual, expected) in jit["results"]
                .as_array()
                .unwrap()
                .iter()
                .zip(interpreted["results"].as_array().unwrap())
            {
                let (actual, expected) = (components(actual), components(expected));
                let label = format!("{fixture} {evaluator}");
                assert_laurent_agrees(&label, &actual, &expected);
                for k in 3..5 {
                    assert!(scaled(actual[k], expected[k]) <= TOLERANCE, "{label}");
                }
            }
        }
    }
}

#[cfg(not(feature = "numerator-jit"))]
#[test]
fn fast_engine_symjit_heads_require_their_feature() {
    let output = run(
        "reduce",
        &[
            "missing-fixture.json",
            "--engine",
            "fast",
            "--numerator-evaluator",
            "symjit-o2",
        ],
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--features numerator-jit"));
}

#[test]
fn fast_engine_rejects_generic_only_options_before_preparing() {
    for (options, needle) in [
        (&["--precision", "double-float"][..], "--precision f64"),
        (&["--precision", "float"], "--precision f64"),
        (&["--bits", "64"], "--precision f64"),
        (&["--mu-extraction", "sampling"], "--mu-extraction sampling"),
        (&["--full-box-coefficients"], "--full-box-coefficients"),
        (&["--mu-sampling-scale", "2"], "--mu-sampling-scale"),
        (
            &["--reduction-kernels", "interpreted-rank"],
            "--reduction-kernels interpreted-rank",
        ),
        (
            &["--reduction-kernels", "native"],
            "--reduction-kernels native",
        ),
        (&["--native-block-rows", "4"], "--native-block-rows"),
    ] {
        for command in ["reduce", "bench"] {
            let output = run(
                command,
                &[&["missing-fixture.json", "--engine", "fast"], options].concat(),
            );
            assert!(!output.status.success(), "{command} {options:?}");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                stderr.contains("--engine fast") && stderr.contains(needle),
                "{command} {options:?}: {stderr}"
            );
            assert!(output.stdout.is_empty());
        }
    }
    // Explicit default values remain accepted.
    let accepted = reduce(
        "examples/graphs/bubble.json",
        &[
            "--engine",
            "fast",
            "--precision",
            "f64",
            "--mu-extraction",
            "expansion",
            "--reduction-kernels",
            "symbolica",
        ],
    );
    assert_eq!(accepted["success"], true);
}

#[test]
fn fast_engine_rejects_complex_external_kinematics() {
    let fixture = TemporaryFixture::new(
        "complex",
        "bubble.dot",
        json!({
            "schema_version": 1, "model": "phi3", "scale_squared": "1",
            "parameter_overrides": {"UFO::mass": "1"},
            "samples": [
                {"external_momenta": {"0": ["0","1","0","0"], "1": ["0","1","0","0"]}},
                {"external_momenta": {"0": ["0","1+𝑖","0","0"], "1": ["0","1+𝑖","0","0"]}}
            ]
        }),
    );
    let output = run("reduce", &[&fixture.path(), "--engine", "fast"]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("requires real external kinematics") && stderr.contains("sample 1 shift"),
        "{stderr}"
    );
    assert!(output.stdout.is_empty());
}

#[test]
fn bench_fast_engine_runs_and_matches_reduce() {
    for (fixture, interface) in [
        (
            "examples/graphs/bubble.json".to_string(),
            "generic-polynomial",
        ),
        (pool("hgg/top-triangle-0"), "tensor"),
    ] {
        let reduced = reduce(&fixture, &["--engine", "fast"]);
        let report = value(run(
            "bench",
            &[
                &fixture,
                "--engine",
                "fast",
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
        assert_eq!(report["engine"], "fast");
        assert_eq!(report["success"], true);
        assert_eq!(report["performance_acceptance"], false);
        assert_eq!(report["precision"], json!({"carrier":"f64","bits":53}));
        assert_eq!(report["numerator_provider"]["interface"], interface);
        assert_eq!(report["reduction_kernels"]["selection"], "fast");
        assert_eq!(report["rank"], reduced["rank"]);
        let cases = report["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 4);
        for case in cases {
            let lane = if case["workload"] == "repeated" { 0 } else { 2 };
            assert_eq!(case["representative_source_lane"], lane);
            assert_eq!(
                case["representative_result"],
                reduced["results"][lane]["values"]
            );
            assert_eq!(case["round_ns"].as_array().unwrap().len(), 2);
            let diagnostic = &case["diagnostic"];
            assert_eq!(diagnostic["failed_lanes"], 0);
            assert_eq!(case["warmup_final_pass_failed_lanes"], 0);
            assert!(diagnostic["masters"]["rows"].as_u64().unwrap() > 0);
            assert!(diagnostic["numerator_expansion"]["rows"].as_u64().unwrap() > 0);
            assert_eq!(
                diagnostic["numerator_coefficients"]["calls"]
                    .as_u64()
                    .unwrap()
                    > 0,
                interface == "tensor"
            );
            let stages = &diagnostic["same_pass_nanoseconds"];
            let ns = |name: &str| stages[name].as_u64().unwrap();
            assert_eq!(
                ns("total"),
                ns("numerator_coefficients")
                    + ns("numerator_callbacks")
                    + ns("actual_master_provider")
                    + ns("residual_reduction")
            );
        }
    }
    let generic = value(run(
        "bench",
        &[
            "examples/graphs/bubble.json",
            "--batch-sizes",
            "2",
            "--points",
            "2",
            "--rounds",
            "1",
            "--workload",
            "repeated",
        ],
    ));
    assert_eq!(generic["engine"], "generic");
}
