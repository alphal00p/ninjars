//! Usability and robustness of graph preparation on hand-written DOT input.

use std::{
    path::PathBuf,
    process::{Command, Output},
};

use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn graph(name: &str) -> String {
    format!("crates/ninjars-cli/tests/graphs/{name}")
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ninjars"))
        .current_dir(root())
        .args(args)
        .output()
        .unwrap()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn success(args: &[&str]) -> (Value, String) {
    let output = run(args);
    assert!(output.status.success(), "{args:?}: {}", stderr(&output));
    (
        serde_json::from_slice(&output.stdout).unwrap(),
        stderr(&output),
    )
}

fn failure(args: &[&str]) -> String {
    let output = run(args);
    assert!(!output.status.success(), "{args:?} unexpectedly succeeded");
    stderr(&output)
}

fn complex(value: &Value) -> (f64, f64) {
    (
        value[0].as_str().unwrap().parse().unwrap(),
        value[1].as_str().unwrap().parse().unwrap(),
    )
}

/// Five result components agree up to binary64 round-off of the largest, or
/// of unity for an amplitude that vanishes analytically.
fn assert_close(left: &Value, right: &Value) {
    let values = |row: &Value| (0..5).map(|i| complex(&row[i])).collect::<Vec<_>>();
    let (left, right) = (values(left), values(right));
    let scale = left
        .iter()
        .map(|(re, im)| re.hypot(*im))
        .fold(1.0_f64, f64::max);
    for (a, b) in left.iter().zip(&right) {
        assert!(
            (a.0 - b.0).hypot(a.1 - b.1) <= 1e-12 * scale,
            "{left:?} != {right:?}"
        );
    }
}

#[test]
fn unknown_models_name_the_missing_file_and_the_built_ins() {
    let message = failure(&[
        "prepare",
        &graph("gluon-bubble.dot"),
        "--model",
        "qcdx",
        "--feynman-rules",
    ]);
    assert!(message.contains("unknown model \"qcdx\""), "{message}");
    assert!(message.contains("no model file at ./qcdx"), "{message}");
    assert!(message.contains("yang_mills"), "{message}");
    let message = failure(&["prepare", &graph("missing.json")]);
    assert!(message.contains("missing.json"), "{message}");
}

#[test]
fn standard_model_sectors_are_built_in_models() {
    for (model, dot) in [
        ("yang_mills", "gluon-bubble.dot"),
        ("qcd", "gluon-bubble.dot"),
        ("qcd_qed", "gluon-bubble.dot"),
        ("qed", "eea-vertex.dot"),
        ("electroweak", "haa-wloop.dot"),
    ] {
        let (prepared, _) = success(&["prepare", &graph(dot), "--model", model, "--feynman-rules"]);
        assert_eq!(prepared["model"], model);
    }
    // Each sector keeps only its particles.
    let message = failure(&[
        "prepare",
        &graph("hgg-top.dot"),
        "--model",
        "yang_mills",
        "--feynman-rules",
    ]);
    assert!(
        message.contains("particle 'H' was not found in model 'yang_mills'"),
        "{message}"
    );
}

#[test]
fn symbolica_warnings_stay_off_the_json_output() {
    // The second sample puts the tree propagator of the external bubble on
    // shell: its exact factor 1/0 makes Symbolica warn while preparing.
    let output = run(&["reduce", &graph("phi3-leg-bubble.json"), "--feynman-rules"]);
    let message = stderr(&output);
    assert!(message.contains("Created infinity"), "{message}");
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["results"][0]["status"]["kind"], "success");
    assert_ne!(result["results"][1]["status"]["kind"], "success");
}

#[test]
fn coincident_propagators_fail_only_their_lane() {
    let fixture = graph("phi3-box-zero-momentum.json");
    let output = run(&["reduce", &fixture, "--feynman-rules", "--engine", "generic"]);
    assert!(!output.status.success());
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["results"][0]["status"]["kind"], "success");
    assert_eq!(
        result["results"][1]["status"],
        serde_json::json!({
            "kind": "repeated_propagator",
            "propagators": [2, 3],
            "reason": "raised or repeated loop propagators are not supported"
        })
    );
    // The valid lane is unaffected by its neighbor, alone or in a batch.
    let alone = run(&[
        "reduce",
        &fixture,
        "--feynman-rules",
        "--engine",
        "generic",
        "--batch-size",
        "1",
    ]);
    let alone: Value = serde_json::from_slice(&alone.stdout).unwrap();
    assert_eq!(
        alone["results"][0]["values"],
        result["results"][0]["values"]
    );
    assert_eq!(alone["results"][1]["status"]["kind"], "repeated_propagator");
    let fast = run(&["reduce", &fixture, "--feynman-rules", "--engine", "fast"]);
    let fast: Value = serde_json::from_slice(&fast.stdout).unwrap();
    assert_eq!(fast["results"][0]["status"]["kind"], "success");
    assert_eq!(fast["results"][1]["status"], result["results"][1]["status"]);
}

#[test]
fn reduction_requires_samples() {
    for engine in ["generic", "fast"] {
        let message = failure(&[
            "reduce",
            &graph("phi3-box.dot"),
            "--feynman-rules",
            "--engine",
            engine,
        ]);
        assert!(
            message.contains("at least one phase-space sample"),
            "{message}"
        );
    }
}

#[test]
fn explicit_mode_rejects_a_graph_without_numerators() {
    let message = failure(&["prepare", &graph("haa-wloop.dot"), "--model", "sm"]);
    assert!(
        message.contains("no `num` attribute") && message.contains("--feynman-rules"),
        "{message}"
    );
    // Stated numerators, including the stable dialect's, remain explicit.
    success(&["prepare", "examples/graphs/bubble.json"]);
    success(&[
        "prepare",
        "examples/graphs/physical/hgg/top-triangle-0.json",
    ]);
}

#[test]
fn hand_written_graphs_receive_the_generators_graph_factors() {
    let factors = |args: &[&str]| {
        let (prepared, _) = success(args);
        (
            prepared["graph_factors"].clone(),
            prepared["overall_factor"].clone(),
        )
    };
    let gluon = factors(&[
        "prepare",
        &graph("gluon-bubble.dot"),
        "--model",
        "yang_mills",
        "--feynman-rules",
    ]);
    assert_eq!(
        gluon,
        (
            serde_json::json!({"automorphisms": 2, "closed_fermion_loops": 0, "open_fermion_lines": 0,
                "factor": "1/2", "applied": true}),
            "1/2".into()
        )
    );
    let ghost = factors(&[
        "prepare",
        &graph("ghost-bubble.dot"),
        "--model",
        "yang_mills",
        "--feynman-rules",
    ]);
    assert_eq!(ghost.0["closed_fermion_loops"], 1);
    assert_eq!(ghost.1, "-1");
    let top = factors(&[
        "prepare",
        &graph("hgg-top.dot"),
        "--model",
        "sm",
        "--feynman-rules",
    ]);
    assert_eq!(top.0["closed_fermion_loops"], 1);
    assert_eq!(top.1, "-1");
    // A stated overall_factor is authoritative, in either dialect. The
    // generated graph records AutG(1) and one internal fermion loop.
    let stated = factors(&[
        "prepare",
        &graph("gg-gluon-loop.dot"),
        "--model",
        "sm",
        "--feynman-rules",
    ]);
    assert_eq!(
        (stated.0["applied"].clone(), stated.1),
        (false.into(), "1/2".into())
    );
    let generated = factors(&[
        "prepare",
        "examples/graphs/physical/hgg/top-triangle-0.json",
    ]);
    assert_eq!(generated.0["automorphisms"], 1);
    assert_eq!(generated.0["closed_fermion_loops"], 1);
    assert_eq!(generated.0["applied"], false);
    assert_eq!(generated.1, "-1");
    // Explicit mode only reports what it does not apply.
    let (prepared, message) = success(&["prepare", "examples/graphs/bubble.json"]);
    assert_eq!(prepared["graph_factors"]["factor"], "1/2");
    assert_eq!(prepared["graph_factors"]["applied"], false);
    assert!(message.contains("automorphism order 2"), "{message}");
}

#[test]
fn one_open_color_pair_receives_the_delta_projector() {
    let (automatic, _) = success(&["reduce", &graph("hgg-top.json"), "--feynman-rules"]);
    let (supplied, _) = success(&[
        "reduce",
        &graph("hgg-top-projector.json"),
        "--feynman-rules",
    ]);
    assert!(
        automatic["color_projector"]
            .as_str()
            .unwrap()
            .starts_with("1/8*spenso::"),
        "{}",
        automatic["color_projector"]
    );
    assert_eq!(supplied["color_projector"], Value::Null);
    for lane in 0..2 {
        assert_close(
            &automatic["results"][lane]["values"],
            &supplied["results"][lane]["values"],
        );
    }
    // With the derived fermion-loop sign, the hand-written graph reproduces
    // the FeynKit-generated diagram with the same orientation and samples.
    let (generated, _) = success(&[
        "reduce",
        "examples/graphs/physical/hgg/top-triangle-0.json",
        "--feynman-rules",
    ]);
    for lane in 0..2 {
        assert_close(
            &automatic["results"][lane]["values"],
            &generated["results"][lane]["values"],
        );
    }
    // Every external leg reports the half-edge a projector refers to.
    assert_eq!(
        automatic["external_legs"][1],
        serde_json::json!({"leg": 1, "momentum_key": 1, "dependent": false, "edge": 1,
            "half_edge": 1, "particle": "g", "state": "outgoing"})
    );
    // Other open indices name their slot and external leg.
    let message = failure(&[
        "prepare",
        &graph("gg-color-only-projector.dot"),
        "--model",
        "sm",
        "--feynman-rules",
    ]);
    assert!(
        message.contains("open indices") && message.contains("at external leg 0 (g, half-edge 0)"),
        "{message}"
    );
    // Dangling color indices of a hand-written numerator are not external
    // legs: they are not joined.
    let message = failure(&["prepare", &graph("dangling-color-indices.dot")]);
    assert!(
        message.contains("open indices") && message.contains("hedge(50,1)"),
        "{message}"
    );
}

#[test]
fn unbound_external_states_and_invariants_fail_in_preparation() {
    let message = failure(&[
        "prepare",
        &graph("haa-wloop-one-polarization.json"),
        "--feynman-rules",
    ]);
    assert!(
        message.contains("external_polarizations` for legs 2 (a)"),
        "{message}"
    );
    let message = failure(&["prepare", &graph("gg-gluon-loop.json"), "--feynman-rules"]);
    assert!(
        message.contains("--color-dimension-invariants"),
        "{message}"
    );
    let (prepared, _) = success(&[
        "reduce",
        &graph("gg-gluon-loop.json"),
        "--feynman-rules",
        "--color-dimension-invariants",
    ]);
    assert_eq!(prepared["unbound"], serde_json::json!([]));
    let message = failure(&["prepare", &graph("eea-vertex-trace.json")]);
    assert!(message.contains("no `num` attribute"), "{message}");
    let fixture = graph("eea-vertex-spinors.json");
    let source = std::fs::read_to_string(root().join(&fixture)).unwrap();
    let directory = root().join("target/test-robustness");
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::copy(
        root().join(graph("eea-vertex.dot")),
        directory.join("eea-vertex.dot"),
    )
    .unwrap();
    let mut without: Value = serde_json::from_str(&source).unwrap();
    for sample in without["samples"].as_array_mut().unwrap() {
        sample.as_object_mut().unwrap().remove("external_spinors");
    }
    let path = directory.join("eea-vertex-no-spinors.json");
    std::fs::write(&path, without.to_string()).unwrap();
    let message = failure(&["prepare", path.to_str().unwrap(), "--feynman-rules"]);
    assert!(
        message.contains("external_spinors` for legs 0 (e-), 1 (e+)"),
        "{message}"
    );
    // Without samples, preparation only warns and reports the same reason.
    let (prepared, message) = success(&[
        "prepare",
        &graph("eea-vertex.dot"),
        "--model",
        "sm",
        "--feynman-rules",
    ]);
    assert!(message.contains("cannot be evaluated yet"), "{message}");
    assert_eq!(prepared["unbound"].as_array().unwrap().len(), 2);
}

#[test]
fn widths_warn_or_enter_complex_mass_propagators() {
    let (ignored, message) = success(&["prepare", &graph("hgg-top-width.json"), "--feynman-rules"]);
    assert!(
        message.contains("propagator widths UFO::{}::WT=3/2 are ignored"),
        "{message}"
    );
    assert_eq!(ignored["samples"][0]["mass_squared"][0], "29929");
    assert_eq!(ignored["propagator_widths"]["mode"], "ignored");
    let (complex, message) = success(&[
        "reduce",
        &graph("hgg-top-width.json"),
        "--feynman-rules",
        "--complex-mass-propagators",
    ]);
    assert!(!message.contains("are ignored"), "{message}");
    assert_eq!(complex["propagator_widths"]["mode"], "complex_mass");
    assert_eq!(
        complex["samples"][0]["mass_squared"],
        serde_json::json!(["29929-519𝑖/2", "29929-519𝑖/2", "29929-519𝑖/2"])
    );
    assert_eq!(complex["results"][0]["status"]["kind"], "success");
}

#[test]
fn disconnected_graphs_are_rejected() {
    let message = failure(&["prepare", &graph("disconnected.dot"), "--feynman-rules"]);
    assert!(
        message.contains("disconnected: it has 2 connected components"),
        "{message}"
    );
}

#[test]
fn external_legs_report_both_fixture_numberings() {
    let (prepared, _) = success(&[
        "prepare",
        &graph("hgg-top-projector.json"),
        "--feynman-rules",
    ]);
    for leg in prepared["external_legs"].as_array().unwrap() {
        // FeynKit lists external edges in leg order: P(i) is leg i's momentum.
        assert_eq!(leg["leg"], leg["momentum_key"]);
    }
    assert_eq!(prepared["external_legs"][0]["dependent"], false);
    assert_eq!(prepared["external_legs"][2]["dependent"], true);
    assert_eq!(
        prepared["polarization_basis"]["1"],
        "gammalooprs::{spenso::rank1,spenso::tensor}::ϵbar(1)"
    );
}

/// `γ^μ` in the chiral basis documented by `ninjars_graph::spinors`.
fn gamma(mu: usize) -> [[(f64, f64); 4]; 4] {
    let mut matrix = [[(0.0, 0.0); 4]; 4];
    let entries: [(usize, usize, (f64, f64)); 4] = match mu {
        0 => [
            (0, 2, (1., 0.)),
            (1, 3, (1., 0.)),
            (2, 0, (1., 0.)),
            (3, 1, (1., 0.)),
        ],
        1 => [
            (0, 3, (1., 0.)),
            (1, 2, (1., 0.)),
            (2, 1, (-1., 0.)),
            (3, 0, (-1., 0.)),
        ],
        2 => [
            (0, 3, (0., -1.)),
            (1, 2, (0., 1.)),
            (2, 1, (0., 1.)),
            (3, 0, (0., -1.)),
        ],
        _ => [
            (0, 2, (1., 0.)),
            (1, 3, (-1., 0.)),
            (2, 0, (-1., 0.)),
            (3, 1, (1., 0.)),
        ],
    };
    for (row, column, value) in entries {
        matrix[row][column] = value;
    }
    matrix
}

fn multiply(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn slash(p: [f64; 4]) -> [[(f64, f64); 4]; 4] {
    let mut matrix = [[(0.0, 0.0); 4]; 4];
    for (mu, component) in p.iter().enumerate() {
        let weight = if mu == 0 { *component } else { -component };
        for (row, entries) in gamma(mu).iter().enumerate() {
            for (column, value) in entries.iter().enumerate() {
                matrix[row][column].0 += weight * value.0;
                matrix[row][column].1 += weight * value.1;
            }
        }
    }
    matrix
}

fn product(a: &[[(f64, f64); 4]; 4], b: &[[(f64, f64); 4]; 4]) -> [[(f64, f64); 4]; 4] {
    let mut matrix = [[(0.0, 0.0); 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            for k in 0..4 {
                let term = multiply(a[row][k], b[k][column]);
                matrix[row][column].0 += term.0;
                matrix[row][column].1 += term.1;
            }
        }
    }
    matrix
}

/// The product of slashed vectors.
fn slashes(vectors: &[[f64; 4]]) -> [[(f64, f64); 4]; 4] {
    let identity = std::array::from_fn(|row| {
        std::array::from_fn(|column| (if row == column { 1. } else { 0. }, 0.))
    });
    vectors
        .iter()
        .fold(identity, |matrix, vector| product(&matrix, &slash(*vector)))
}

/// Unit spinors give the amputated matrix element by element: lane 4a+b of
/// `explicit` is vbar=e_a and u=e_b. Contracting it with `projector` must
/// reproduce the spin-trace projection, which Spenso builds independently of
/// the Dirac basis.
fn assert_spin_trace(explicit: &Value, trace: &Value, projector: &[[(f64, f64); 4]; 4]) {
    let expected = &trace["results"][0]["values"];
    for component in 0..5 {
        let mut sum = (0.0, 0.0);
        for bar in 0..4 {
            for ket in 0..4 {
                let row = &explicit["results"][4 * bar + ket];
                assert_eq!(row["status"]["kind"], "success");
                let term = multiply(complex(&row["values"][component]), projector[ket][bar]);
                sum.0 += term.0;
                sum.1 += term.1;
            }
        }
        let reference = complex(&expected[component]);
        let scale = reference.0.hypot(reference.1).max(1.);
        assert!(
            (sum.0 - reference.0).hypot(sum.1 - reference.1) < 1e-9 * scale,
            "component {component}: {sum:?} vs {reference:?}"
        );
    }
}

#[test]
fn explicit_spinors_reproduce_a_spin_trace_projection() {
    // The trace projector closes the line with p0-slash p1-slash.
    let (explicit, _) = success(&[
        "reduce",
        &graph("eea-vertex-spinors.json"),
        "--feynman-rules",
    ]);
    let (trace, _) = success(&["reduce", &graph("eea-vertex-trace.json"), "--feynman-rules"]);
    assert_eq!(
        explicit["spinor_basis"],
        serde_json::json!({
            "0": "gammalooprs::{spenso::rank1,spenso::tensor}::u(0)",
            "1": "gammalooprs::{spenso::rank1,spenso::tensor}::vbar(1)"
        })
    );
    let projector = slashes(&[[5., 0., 3., 4.], [5., 3., 0., -4.]]);
    assert_spin_trace(&explicit, &trace, &projector);
}

#[test]
fn explicit_spinors_cover_single_vertex_lines_and_open_vectors() {
    // e- mu- -> e- mu-: a vertex correction (an electron chain and a
    // single-vertex muon line sharing a Lorentz index) and the vacuum
    // polarization (two single-vertex lines contracted with loop vectors).
    // Unit spinors give the amputated amplitude elements. Closing each line by
    // the slashed momentum of its incoming leg must match the trace projector;
    // with momenta along z only four elements contribute.
    let (electron, muon) = (slash([5., 0., 0., 5.]), slash([6., 0., 0., -6.]));
    for name in ["emu-vertex", "emu-vp"] {
        let (explicit, _) = success(&[
            "reduce",
            &graph(&format!("{name}-spinors.json")),
            "--feynman-rules",
        ]);
        let (trace, _) = success(&[
            "reduce",
            &graph(&format!("{name}-trace.json")),
            "--feynman-rules",
        ]);
        let fixture: Value = serde_json::from_str(
            &std::fs::read_to_string(root().join(graph(&format!("{name}-spinors.json")))).unwrap(),
        )
        .unwrap();
        let mut sum = [(0.0, 0.0); 5];
        for (row, sample) in explicit["results"]
            .as_array()
            .unwrap()
            .iter()
            .zip(fixture["samples"].as_array().unwrap())
        {
            assert_eq!(row["status"]["kind"], "success");
            // "ubar2-e{a}-u0-e{b}-ubar3-e{c}-u1-e{d}"
            let unit = |position: usize| -> usize {
                sample["id"]
                    .as_str()
                    .unwrap()
                    .split('-')
                    .nth(position)
                    .unwrap()[1..]
                    .parse()
                    .unwrap()
            };
            let weight = multiply(electron[unit(3)][unit(1)], muon[unit(7)][unit(5)]);
            for (component, total) in sum.iter_mut().enumerate() {
                let term = multiply(complex(&row["values"][component]), weight);
                total.0 += term.0;
                total.1 += term.1;
            }
        }
        let expected = &trace["results"][0]["values"];
        let scale = (0..5)
            .map(|component| {
                let (re, im) = complex(&expected[component]);
                re.hypot(im)
            })
            .fold(1e-300_f64, f64::max);
        for (component, total) in sum.iter().enumerate() {
            let reference = complex(&expected[component]);
            assert!(
                (total.0 - reference.0).hypot(total.1 - reference.1) < 1e-9 * scale,
                "{name} component {component}: {total:?} vs {reference:?}"
            );
        }
        assert_ne!(sum[0], (0.0, 0.0), "{name}: the projection must not vanish");
    }
}

/// A rational fixture component such as `-1/3`.
fn rational(value: &Value) -> f64 {
    let text = value.as_str().unwrap();
    match text.split_once('/') {
        Some((numerator, denominator)) => {
            numerator.parse::<f64>().unwrap() / denominator.parse::<f64>().unwrap()
        }
        None => text.parse().unwrap(),
    }
}

#[test]
fn levi_civita_tensors_reproduce_explicit_axial_traces() {
    // Idenso traces the hand-written numerator Tr[g5 p0 p2 p3 p4] (slashed)
    // to 4 epsilon(p0,p2,p3,p4). Its components must reproduce the trace of
    // explicit chiral-basis matrices, gamma5 = diag(-1,-1,1,1), times the
    // scalar pentagon with the same propagators.
    let (axial, _) = success(&["reduce", &graph("pentagon-axial-trace.json")]);
    let (scalar, _) = success(&["reduce", &graph("pentagon-scalar.json")]);
    // 4i det[p0; p2; p3; p4] for the first sample.
    assert_eq!(axial["samples"][0]["numerator"], "-2𝑖");
    let fixture: Value = serde_json::from_str(
        &std::fs::read_to_string(root().join(graph("pentagon-axial-trace.json"))).unwrap(),
    )
    .unwrap();
    let samples = fixture["samples"].as_array().unwrap();
    for (lane, sample) in samples.iter().enumerate() {
        let momenta = ["0", "2", "3", "4"].map(|key| {
            let components = &sample["external_momenta"][key];
            std::array::from_fn(|mu| rational(&components[mu]))
        });
        let mut matrix = slashes(&momenta);
        for row in &mut matrix[..2] {
            for entry in row.iter_mut() {
                *entry = (-entry.0, -entry.1);
            }
        }
        let trace = (0..4).fold((0.0, 0.0), |sum, i| {
            (sum.0 + matrix[i][i].0, sum.1 + matrix[i][i].1)
        });
        assert!(
            trace.1.abs() > 0.1,
            "lane {lane}: the trace must not vanish"
        );
        let left = &axial["results"][lane];
        let right = &scalar["results"][lane];
        assert_eq!(left["status"]["kind"], "success");
        assert_eq!(right["status"]["kind"], "success");
        let expected = (0..5)
            .map(|component| {
                let (re, im) = multiply(trace, complex(&right["values"][component]));
                serde_json::json!([re.to_string(), im.to_string()])
            })
            .collect::<Value>();
        assert_close(&left["values"], &expected);
    }
}

#[test]
fn levi_civita_indices_are_summed_with_spinor_chains() {
    // e+ e- -> Z a through a photon and a top triangle: the axial trace leaves
    // epsilon tensors whose index joins the open electron line. Unit spinors
    // sum it over components with the chain; the spin-trace projector
    // p0-slash p1-slash p2-slash lets Idenso contract it symbolically.
    let (explicit, _) = success(&["reduce", &graph("eeza-top-spinors.json"), "--feynman-rules"]);
    let (trace, _) = success(&["reduce", &graph("eeza-top-trace.json"), "--feynman-rules"]);
    assert_eq!(explicit["unbound"], serde_json::json!([]));
    let projector = slashes(&[
        [125., 0., 0., 125.],
        [125., 0., 0., -125.],
        [4425953209. / 31250000., -3386546791. / 31250000., 0., 0.],
    ]);
    assert_spin_trace(&explicit, &trace, &projector);
}

#[test]
fn axial_top_loops_reduce_in_both_engines() {
    // H -> Z a through one top loop orientation. Lane 0 contains only the
    // epsilon(pH, pa, eps_a, eps_Z) structure, lane 1 only the vector part;
    // lane 2 uses helicity states.
    let reduce = |fixture: &str, engine: &str| {
        let (result, _) = success(&[
            "reduce",
            &graph(fixture),
            "--feynman-rules",
            "--engine",
            engine,
        ]);
        result
    };
    let generic = reduce("hza-top.json", "generic");
    let fast = reduce("hza-top.json", "fast");
    let reversed = reduce("hza-top-reversed.json", "generic");
    for lane in 0..3 {
        for result in [&generic, &fast, &reversed] {
            assert_eq!(result["results"][lane]["status"]["kind"], "success");
        }
        assert_close(
            &generic["results"][lane]["values"],
            &fast["results"][lane]["values"],
        );
    }
    // Charge conjugation: reversing the loop flips the axial (epsilon) part
    // and keeps the vector part.
    let axial = &generic["results"][0]["values"];
    assert!(complex(&axial[0]).1.abs() > 0.1, "{axial}");
    let flipped = (0..5)
        .map(|component| {
            let (re, im) = complex(&reversed["results"][0]["values"][component]);
            serde_json::json!([(-re).to_string(), (-im).to_string()])
        })
        .collect::<Value>();
    assert_close(axial, &flipped);
    assert_close(
        &generic["results"][1]["values"],
        &reversed["results"][1]["values"],
    );
    // Without samples only the polarizations are missing.
    let (prepared, _) = success(&[
        "prepare",
        &graph("hza-top.dot"),
        "--model",
        "sm",
        "--feynman-rules",
    ]);
    assert_eq!(
        prepared["unbound"],
        serde_json::json!([
            "external polarizations are unbound: give every sample `external_polarizations` for legs 1 (a), 2 (Z)"
        ])
    );
}
