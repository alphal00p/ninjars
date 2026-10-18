//! `--r2 fdh`: R2 rational terms from the FDH numerator's explicit `mu^2` terms.
//!
//! - H -> gg top triangles: R2 equals MadGraph5 `loop_sm`'s `R2_GGHt` vertex;
//! - the SU(3) gluon self-energy (gluon, ghost, down- and top-quark loops) and
//!   the sum of the six top-quark four-photon boxes satisfy their Ward
//!   identities only with `--r2 fdh`;
//! - fast/generic engine parity and generic f64/DoubleFloat/Float(200) agreement.
//!
//! Closed fermion loops with external bosons have scheme-independent R2, and
//! the gluon self-energy is transverse in FDH, so these are exact checks.

use std::{
    collections::HashMap,
    f64::consts::PI,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use serde_json::{Value, json};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::float::Complex,
    id::{Pattern, Replacement},
    parser::ParseSettings,
};

type C = (f64, f64);

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

fn value(output: Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn reduce(fixture: &str, options: &[&str]) -> Value {
    value(run(&[&["reduce", fixture], options].concat()))
}

/// Reductions run concurrently, one process each.
fn reduce_all(jobs: &[(String, Vec<String>)]) -> Vec<Value> {
    std::thread::scope(|scope| {
        let handles: Vec<_> = jobs
            .iter()
            .map(|(fixture, options)| {
                scope.spawn(move || {
                    let options: Vec<&str> = options.iter().map(String::as_str).collect();
                    reduce(fixture, &options)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

fn parse(pair: &Value) -> C {
    let part = |index: usize| match &pair[index] {
        Value::String(text) => text.parse::<f64>().unwrap(),
        number => number.as_f64().unwrap(),
    };
    (part(0), part(1))
}

/// `[finite, single pole, double pole, cut-constructible, rational]` of a lane.
fn components(document: &Value, lane: usize) -> Vec<C> {
    let row = &document["results"][lane];
    assert_eq!(row["status"]["kind"], "success", "{row}");
    row["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(parse)
        .collect()
}

fn add(a: C, b: C) -> C {
    (a.0 + b.0, a.1 + b.1)
}

fn sub(a: C, b: C) -> C {
    (a.0 - b.0, a.1 - b.1)
}

fn mul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn abs(a: C) -> f64 {
    a.0.hypot(a.1)
}

/// Exact Symbolica expression (fixture syntax) evaluated in binary64.
fn number(expression: &str) -> C {
    let value: Complex<f64> = Atom::parse(expression, "ninjars", ParseSettings::default())
        .unwrap()
        .evaluate_with_prec(&HashMap::<Atom, Complex<f64>>::new(), 53)
        .unwrap();
    (value.re, value.im)
}

fn vector(components: &Value) -> [C; 4] {
    std::array::from_fn(|mu| number(components[mu].as_str().unwrap()))
}

/// Minkowski `(+---)` product without conjugation.
fn dot(a: &[C; 4], b: &[C; 4]) -> C {
    let mut total = mul(a[0], b[0]);
    for mu in 1..4 {
        total = sub(total, mul(a[mu], b[mu]));
    }
    total
}

fn fixture(path: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(root().join(path)).unwrap()).unwrap()
}

/// MadGraph5 `loop_sm` `R2_GGHt` from the fixture's model inputs, following
/// `models/loop_sm/parameters.py` and `CT_couplings.py` verbatim:
/// `R2_GGHt = 4 (-i yt/sqrt2) (1/2) G^2/(8 pi^2) MT` with `yt = sqrt2 ymt/v`,
/// `v = 2 MW sw/ee` and `G = 2 sqrt(aS pi)`.
fn r2_gght(overrides: &Value) -> C {
    let input = |name: &str| number(overrides[name].as_str().unwrap()).0;
    let (mt, ymt, gf, a_s) = (input("MT"), input("ymt"), input("Gf"), input("aS"));
    let (aewm1, mz) = (input("aEWM1"), input("MZ"));
    let aew = 1. / aewm1;
    let g = 2. * a_s.sqrt() * PI.sqrt();
    let mw =
        (mz * mz / 2. + (mz.powi(4) / 4. - aew * PI * mz * mz / (gf * 2_f64.sqrt())).sqrt()).sqrt();
    let ee = 2. * aew.sqrt() * PI.sqrt();
    let sw = (1. - mw * mw / (mz * mz)).sqrt();
    let v = 2. * mw * sw / ee;
    let yt = ymt * 2_f64.sqrt() / v;
    // 4 * (-i yt / sqrt2) * (1/2) * G^2 / (8 pi^2) * MT
    (
        0.,
        -4. * yt / 2_f64.sqrt() * 0.5 * g * g / (8. * PI * PI) * mt,
    )
}

/// NinjaRS normalization of MadLoop's `V_GGH` R2 vertex amplitude.
///
/// NinjaRS integrates `d^d q / (i pi^(d/2))`, so a loop amplitude `M` in
/// MadGraph's convention (`i M` = product of Feynman rules; ALOHA amplitudes
/// carry `-i`) is `NinjaRS / (16 pi^2)`. This is the documented MadLoop
/// conversion: raw `ML5_0_LOOP` Laurent coefficients (with CT normalization
/// `1/(16 pi^2)`) times the linear color projection `-1/2` equal
/// `NinjaRS / (16 pi^2)`, i.e. NinjaRS = `-8 pi^2` raw
/// (`validation/evidence/physical-hgg-madloop.json`). The R2 vertex
/// `R2_GGHt Tr(Ta Tb) g^{mu nu}` contributes `M = -i R2_GGHt (1/2) eps1bar.eps2bar`
/// to the `delta_ab` coefficient that the fixtures project.
fn expected_hgg_r2(r2_gght: C, polarization_product: C) -> C {
    let amplitude = mul(
        mul((0., -1.), r2_gght),
        mul((0.5, 0.), polarization_product),
    );
    mul((16. * PI * PI, 0.), amplitude)
}

#[test]
fn hgg_r2_matches_madgraph_r2_gght_vertex() {
    let paths = [0, 1].map(|o| format!("examples/graphs/physical/hgg/top-triangle-{o}.json"));
    let jobs: Vec<_> = paths
        .iter()
        .flat_map(|path| {
            ["off", "fdh"].map(|r2| (path.clone(), vec!["--r2".to_string(), r2.to_string()]))
        })
        .collect();
    let documents = reduce_all(&jobs);
    let input = fixture(&paths[0]);
    assert_eq!(
        fixture(&paths[1])["parameter_overrides"],
        input["parameter_overrides"]
    );
    let r2 = r2_gght(&input["parameter_overrides"]);
    for (document, r2) in documents.iter().zip(["off", "fdh"].iter().cycle()) {
        assert_eq!(document["r2"]["mode"], *r2);
        assert_eq!(document["rank"], 2, "mu^2 counts as rank two");
    }
    let samples = input["samples"].as_array().unwrap();
    for (lane, sample) in samples.iter().enumerate() {
        let polarizations = &sample["external_polarizations"];
        let product = dot(&vector(&polarizations["1"]), &vector(&polarizations["2"]));
        let mut total = [(0., 0.); 5];
        let mut off_finite = (0., 0.);
        for orientation in 0..2 {
            let off = components(&documents[2 * orientation], lane);
            let fdh = components(&documents[2 * orientation + 1], lane);
            let scale = off.iter().chain(&fdh).copied().map(abs).fold(1., f64::max);
            // R2 changes only the rational part: poles and the
            // cut-constructible part are those of the four-dimensional numerator.
            for k in [1, 2, 3] {
                assert!(
                    abs(sub(fdh[k], off[k])) <= 1e-12 * scale,
                    "orientation {orientation} lane {lane} component {k}"
                );
            }
            for k in 0..5 {
                total[k] = add(total[k], sub(fdh[k], off[k]));
            }
            off_finite = add(off_finite, off[0]);
            // The double-longitudinal sample: transverse only with R2.
            if lane == 2 {
                assert!(abs(fdh[0]) <= 1e-12 * abs(off[0]), "{fdh:?} vs {off:?}");
                assert!(abs(off[0]) > 1e3);
            }
        }
        // (+-) has eps1bar.eps2bar = 0 and no R2.
        let expected = expected_hgg_r2(r2, product);
        let reference = abs(expected).max(abs(expected_hgg_r2(r2, (1., 0.))));
        for k in [0, 4] {
            let error = abs(sub(total[k], expected)) / reference;
            eprintln!(
                "hgg lane {lane} component {k}: NinjaRS R2 {:?}, MadGraph R2_GGHt {expected:?}, relative error {error:e}",
                total[k]
            );
            assert!(error <= 1e-12, "lane {lane} component {k}");
        }
        assert!(abs(off_finite) > 0.);
    }
}

/// Sum over graphs of the per-lane components of several reductions.
fn summed(documents: &[Value], lane: usize) -> (Vec<C>, f64) {
    let mut total = vec![(0., 0.); 5];
    let mut scale = 0_f64;
    for document in documents {
        for (k, value) in components(document, lane).into_iter().enumerate() {
            total[k] = add(total[k], value);
            scale = scale.max(abs(value));
        }
    }
    (total, scale)
}

const SELF_ENERGIES: [&str; 4] = ["ghost-loop", "gluon-loop", "down-loop", "top-loop"];

#[test]
fn gluon_self_energy_is_transverse_only_with_fdh() {
    let jobs: Vec<_> = ["off", "fdh"]
        .iter()
        .flat_map(|r2| {
            SELF_ENERGIES.map(|graph| {
                (
                    format!("examples/graphs/physical/gluon-self-energy/{graph}.json"),
                    vec!["--r2".to_string(), r2.to_string()],
                )
            })
        })
        .collect();
    let documents = reduce_all(&jobs);
    let (off, fdh) = documents.split_at(SELF_ENERGIES.len());
    let samples = fixture("examples/graphs/physical/gluon-self-energy/gluon-loop.json")["samples"]
        .as_array()
        .unwrap()
        .clone();
    for (lane, sample) in samples.iter().enumerate() {
        let id = sample["id"].as_str().unwrap();
        let (fdh_sum, fdh_scale) = summed(fdh, lane);
        let (off_sum, off_scale) = summed(off, lane);
        if id.starts_with("ward") {
            // p_mu Pi^{mu nu} = 0 for the complete self-energy, and separately
            // for the gauge (gluon + ghost) part and each quark flavor.
            for (k, value) in fdh_sum.iter().enumerate() {
                assert!(
                    abs(*value) <= 1e-11 * fdh_scale,
                    "{id} component {k}: {value:?} (scale {fdh_scale:e})"
                );
            }
            for (subset, graphs) in [
                ("gauge", &fdh[..2]),
                ("down", &fdh[2..3]),
                ("top", &fdh[3..4]),
            ] {
                let (sum, _) = summed(graphs, lane);
                assert!(
                    abs(sum[0]) <= 1e-11 * fdh_scale,
                    "{id} {subset}: {:?} (scale {fdh_scale:e})",
                    sum[0]
                );
            }
            // Without the mu^2 terms the rational part is not transverse,
            // neither for the gauge part nor for the quark loops.
            assert!(abs(off_sum[0]) > 1e-3 * off_scale, "{id}: {off_sum:?}");
            for (subset, graphs) in [
                ("gauge", &off[..2]),
                ("down", &off[2..3]),
                ("top", &off[3..4]),
            ] {
                let (sum, _) = summed(graphs, lane);
                assert!(
                    abs(sum[0]) > 1e-6 * off_scale,
                    "{id} {subset}: {:?}",
                    sum[0]
                );
            }
            eprintln!(
                "gluon self-energy {id}: |finite| off {:e}, fdh {:e} (largest graph {fdh_scale:e})",
                abs(off_sum[0]),
                abs(fdh_sum[0])
            );
        } else {
            assert!(
                abs(fdh_sum[0]) > 1e-3 * fdh_scale,
                "{id}: control is nonzero"
            );
        }
    }
}

/// Fixtures of the six four-photon boxes with one polarization replaced by
/// its momentum, written next to absolute graph paths.
fn four_photon_fixtures(directory: &Path) -> Vec<String> {
    std::fs::create_dir_all(directory).unwrap();
    (0..6)
        .map(|graph| {
            let source = format!("examples/graphs/physical/four-photon/top-box-{graph}.json");
            let mut input = fixture(&source);
            input["graph"] = root()
                .join(format!(
                    "examples/graphs/physical/four-photon/top-box-{graph}.dot"
                ))
                .canonicalize()
                .unwrap()
                .to_string_lossy()
                .into_owned()
                .into();
            let samples = input["samples"].as_array().unwrap().clone();
            let (helicity, generic) = (&samples[0], &samples[3]);
            let longitudinal = |sample: &Value, leg: &str, id: &str| {
                let mut sample = sample.clone();
                sample["external_polarizations"][leg] = sample["external_momenta"][leg].clone();
                sample["id"] = id.into();
                sample
            };
            input["samples"] = json!([
                longitudinal(generic, "0", "ward-leg-0"),
                longitudinal(generic, "2", "ward-leg-2"),
                longitudinal(helicity, "1", "ward-leg-1-pppp"),
                generic.clone(),
            ]);
            let path = directory.join(format!("top-box-{graph}.json"));
            std::fs::write(&path, serde_json::to_vec_pretty(&input).unwrap()).unwrap();
            path.to_string_lossy().into_owned()
        })
        .collect()
}

#[test]
fn four_photon_sum_is_gauge_invariant_only_with_fdh() {
    let directory = std::env::temp_dir().join(format!("ninjars-r2-photons-{}", std::process::id()));
    let fixtures = four_photon_fixtures(&directory);
    let jobs: Vec<_> = ["off", "fdh"]
        .iter()
        .flat_map(|r2| {
            fixtures
                .iter()
                .map(|path| (path.clone(), vec!["--r2".to_string(), r2.to_string()]))
        })
        .collect();
    let documents = reduce_all(&jobs);
    let (off, fdh) = documents.split_at(6);
    for (lane, id) in ["ward-leg-0", "ward-leg-2", "ward-leg-1-pppp", "generic"]
        .iter()
        .enumerate()
    {
        let (fdh_sum, fdh_scale) = summed(fdh, lane);
        let (off_sum, _) = summed(off, lane);
        // Largest finite part of a single box.
        let finite_scale = off
            .iter()
            .chain(fdh)
            .map(|document| abs(components(document, lane)[0]))
            .fold(0., f64::max);
        if id.starts_with("ward") {
            assert!(
                abs(fdh_sum[0]) <= 1e-11 * fdh_scale,
                "{id}: {:?} (scale {fdh_scale:e})",
                fdh_sum[0]
            );
            assert!(
                abs(off_sum[0]) > 1e-3 * finite_scale,
                "{id}: {:?}",
                off_sum[0]
            );
            eprintln!(
                "four photons {id}: |sum finite| off {:e}, fdh {:e} (largest box {fdh_scale:e})",
                abs(off_sum[0]),
                abs(fdh_sum[0])
            );
        } else {
            assert!(abs(fdh_sum[0]) > 1e-3 * finite_scale);
        }
        // Poles cancel in the sum independently of R2.
        for k in [1, 2] {
            assert!(abs(fdh_sum[k]) <= 1e-9 * fdh_scale.max(1.), "{id} pole {k}");
        }
    }
    let _ = std::fs::remove_dir_all(directory);
}

/// Largest component magnitude over every lane of a document.
fn document_scale(document: &Value) -> f64 {
    (0..document["results"].as_array().unwrap().len())
        .flat_map(|lane| components(document, lane))
        .map(abs)
        .fold(1., f64::max)
}

/// Errors relative to the document's largest value: lanes with vanishing
/// amplitudes (helicity zeros, Ward samples) are cancellation-dominated.
fn assert_close(label: &str, actual: &[C], expected: &[C], scale: f64, tolerance: f64) {
    for k in 0..5 {
        let error = abs(sub(actual[k], expected[k])) / scale;
        assert!(
            error <= tolerance,
            "{label} component {k}: {:?} != {:?} ({error:e})",
            actual[k],
            expected[k]
        );
    }
}

#[test]
fn fdh_fast_engine_matches_generic_engine() {
    let mut fixtures: Vec<String> = [0, 1]
        .map(|o| format!("examples/graphs/physical/hgg/top-triangle-{o}.json"))
        .into();
    fixtures.extend(
        SELF_ENERGIES.map(|g| format!("examples/graphs/physical/gluon-self-energy/{g}.json")),
    );
    fixtures.push("examples/graphs/physical/four-photon/top-box-0.json".into());
    let jobs: Vec<_> = fixtures
        .iter()
        .flat_map(|path| {
            [&[][..], &["--engine", "fast"][..]].map(|engine| {
                let mut options = vec!["--r2".to_string(), "fdh".to_string()];
                options.extend(engine.iter().map(|s| s.to_string()));
                (path.clone(), options)
            })
        })
        .collect();
    let documents = reduce_all(&jobs);
    for (path, pair) in fixtures.iter().zip(documents.chunks(2)) {
        let (generic, fast) = (&pair[0], &pair[1]);
        assert_eq!(fast["engine"], "fast");
        // The ghost loop has no loop-momentum self-contraction (no mu^2 term)
        // and keeps Ninja's plain tensor.
        let interface = if path.contains("ghost") {
            "tensor"
        } else {
            "mu-tensor"
        };
        assert_eq!(fast["numerator_provider"]["interface"], interface, "{path}");
        assert_eq!(fast["rank"], generic["rank"]);
        let scale = document_scale(generic);
        for lane in 0..generic["results"].as_array().unwrap().len() {
            let (expected, actual) = (components(generic, lane), components(fast, lane));
            // Laurent coefficients; the cc/rational split depends on the cut basis.
            for k in 0..3 {
                let error = abs(sub(actual[k], expected[k])) / scale;
                assert!(
                    error <= 1e-9,
                    "{path} lane {lane} coefficient {k}: {error:e}"
                );
            }
        }
    }
}

#[test]
fn fdh_generic_precisions_agree() {
    let fixtures = [
        "examples/graphs/physical/hgg/top-triangle-0.json",
        "examples/graphs/physical/gluon-self-energy/gluon-loop.json",
        "examples/graphs/physical/gluon-self-energy/top-loop.json",
    ];
    let precisions: [&[&str]; 3] = [
        &[],
        &["--precision", "double-float"],
        &["--precision", "float", "--bits", "200"],
    ];
    let jobs: Vec<_> = fixtures
        .iter()
        .flat_map(|path| {
            precisions.map(|precision| {
                let mut options = vec!["--r2".to_string(), "fdh".to_string()];
                options.extend(precision.iter().map(|s| s.to_string()));
                (path.to_string(), options)
            })
        })
        .collect();
    let documents = reduce_all(&jobs);
    for (path, triple) in fixtures.iter().zip(documents.chunks(3)) {
        assert_eq!(triple[2]["precision"]["bits"], 200);
        let scale = document_scale(&triple[2]);
        for lane in 0..triple[0]["results"].as_array().unwrap().len() {
            let reference = components(&triple[2], lane);
            assert_close(
                &format!("{path} lane {lane} f64"),
                &components(&triple[0], lane),
                &reference,
                scale,
                1e-9,
            );
            assert_close(
                &format!("{path} lane {lane} double-float"),
                &components(&triple[1], lane),
                &reference,
                scale,
                1e-20,
            );
        }
    }
}

#[test]
fn r2_option_defaults_to_off_and_binds_kk_minus_mu2() {
    let path = "examples/graphs/physical/hgg/top-triangle-0.json";
    let default = value(run(&["prepare", path]));
    let off = value(run(&["prepare", path, "--r2", "off"]));
    let fdh = value(run(&["prepare", path, "--r2", "fdh"]));
    assert_eq!(default, off);
    assert_eq!(off["r2"]["mode"], "off");
    assert_eq!(fdh["r2"]["mode"], "fdh");
    let mu2 = Atom::parse("ninjars::mu2", "ninjars", ParseSettings::default()).unwrap();
    let mut dimensional_samples = 0;
    for (a, b) in off["samples"]
        .as_array()
        .unwrap()
        .iter()
        .zip(fdh["samples"].as_array().unwrap())
    {
        let parse = |text: &Value| {
            Atom::parse(text.as_str().unwrap(), "ninjars", ParseSettings::default()).unwrap()
        };
        let (four, dimensional) = (parse(&a["numerator"]), parse(&b["numerator"]));
        assert!(!four.contains(&mu2));
        // (+-) has no mu^2 term: it multiplies eps1bar.eps2bar = 0.
        dimensional_samples += usize::from(dimensional.contains(&mu2));
        // mu^2 -> 0 recovers the four-dimensional numerator exactly.
        let zero = dimensional.replace_multiple(&[Replacement::new(
            Pattern::Literal(mu2.clone()),
            Pattern::Literal(Atom::Zero),
        )]);
        assert!((zero - four).expand().is_zero());
        assert_eq!(a["shifts"], b["shifts"]);
        assert_eq!(a["mass_squared"], b["mass_squared"]);
    }
    assert_eq!(dimensional_samples, 2);
    let rejected = run(&["prepare", path, "--r2", "hv"]);
    assert!(!rejected.status.success());
}

#[cfg(feature = "numerator-jit")]
#[test]
fn pipeline_bench_fdh_stays_on_the_fast_path_and_adds_r2() {
    use ninjars_graph::vector_wavefunction;
    let directory =
        std::env::temp_dir().join(format!("ninjars-r2-pipeline-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let reports: Vec<Value> = ["off", "fdh"]
        .iter()
        .map(|r2| {
            let output = directory.join(format!("{r2}.json"));
            let result = run(&[
                "pipeline-bench",
                "examples/graphs/physical/hgg/top-triangle-0-pool.json",
                "--points",
                "validation/benchmark/points/hgg.json",
                "--feynman-rules",
                "--r2",
                r2,
                "--widths",
                "1,3",
                "--blocks",
                "1",
                "--min-block-ms",
                "1",
                "--output",
                output.to_str().unwrap(),
            ]);
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            serde_json::from_str(&std::fs::read_to_string(output).unwrap()).unwrap()
        })
        .collect();
    let (off, fdh) = (&reports[0], &reports[1]);
    assert_eq!(
        (off["r2"].as_str(), off["mu_powers"].as_u64()),
        (Some("off"), Some(0))
    );
    assert_eq!(
        (fdh["r2"].as_str(), fdh["mu_powers"].as_u64()),
        (Some("fdh"), Some(1))
    );
    assert_eq!(fdh["rank"], off["rank"]);
    assert_eq!(fdh["ncoeff"], 16, "rank-2 tensor plus the mu^2 constant");
    let pool = fixture("validation/benchmark/points/hgg.json");
    let input = fixture("examples/graphs/physical/hgg/top-triangle-0-pool.json");
    let r2 = r2_gght(&input["parameter_overrides"]);
    let width = &fdh["per_point_timing"][1];
    assert_eq!(width["width"], 3);
    for (lane, point) in pool["points"].as_array().unwrap().iter().enumerate() {
        let momenta: Vec<[f64; 4]> = serde_json::from_value(point["momenta"].clone()).unwrap();
        let helicities: Vec<i8> = serde_json::from_value(point["helicities"].clone()).unwrap();
        // Both gluons are outgoing: conjugated polarizations, as in the pipeline.
        let wave = |leg: usize| {
            vector_wavefunction(&momenta[leg], helicities[leg], true)
                .unwrap()
                .map(|z| (z.re, z.im))
        };
        let product = dot(&wave(1), &wave(2));
        let result = |report: &Value, key: &str| {
            let entry = &report["per_point_timing"][1]["lane_results"][lane];
            assert_eq!(entry["id"], point["id"]);
            entry[key].clone()
        };
        let (off_rational, fdh_rational) = (
            parse(&result(off, "rational")),
            parse(&result(fdh, "rational")),
        );
        let (off_cc, fdh_cc) = (
            parse(&result(off, "cut_constructible")),
            parse(&result(fdh, "cut_constructible")),
        );
        // One orientation carries half of the vertex.
        let expected = mul((0.5, 0.), expected_hgg_r2(r2, product));
        let delta = sub(fdh_rational, off_rational);
        let error = abs(sub(delta, expected)) / abs(expected);
        eprintln!("pipeline hgg lane {lane}: R2 {delta:?}, expected {expected:?}, error {error:e}");
        assert!(error <= 1e-12);
        assert!(abs(sub(fdh_cc, off_cc)) <= 1e-12 * abs(off_cc).max(1.));
    }
    let _ = std::fs::remove_dir_all(directory);
}

#[test]
fn export_bench_fdh_writes_mu_block_cases_for_case_and_numerator_bench() {
    let directory = std::env::temp_dir().join(format!("ninjars-r2-export-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let (case_path, report_path) = (directory.join("case.json"), directory.join("report.json"));
    let pool_fixture = "examples/graphs/physical/hgg/top-triangle-0-pool.json";
    let exported = run(&[
        "export-bench",
        pool_fixture,
        "--r2",
        "fdh",
        "--output",
        case_path.to_str().unwrap(),
        "--report",
        report_path.to_str().unwrap(),
    ]);
    assert!(
        exported.status.success(),
        "{}",
        String::from_utf8_lossy(&exported.stderr)
    );
    let read = |path: &Path| -> Value {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    };
    let (case, report) = (read(&case_path), read(&report_path));
    assert_eq!(case["format"], "ninjars-bench-case-v2");
    assert_eq!(
        (case["rank"].as_u64(), case["mu_powers"].as_u64()),
        (Some(2), Some(1))
    );
    assert_eq!(case["r2"], "fdh");
    let committed = fixture("validation/benchmark/cases/hgg-top-triangle-0.json");
    for (lane, original) in case["lanes"]
        .as_array()
        .unwrap()
        .iter()
        .zip(committed["lanes"].as_array().unwrap())
    {
        let tensor = lane["tensor"].as_array().unwrap();
        assert_eq!(tensor.len(), 15 + 1);
        // The mu^0 block is the four-dimensional (committed v1) tensor.
        assert_eq!(&tensor[..15], &original["tensor"].as_array().unwrap()[..]);
        assert_eq!(lane["shifts"], original["shifts"]);
    }
    let verification = &report["verification"];
    assert!(
        verification["max_relative_error_vs_high_precision_polynomial"]
            .as_f64()
            .unwrap()
            < 1e-13
    );
    let timed = value(run(&[
        "case-bench",
        case_path.to_str().unwrap(),
        "--widths",
        "1,3",
        "--blocks",
        "1",
        "--min-block-ms",
        "1",
    ]));
    let timing = &timed["cases"][0];
    assert_eq!(timing["mu_powers"], 1);
    assert!(
        timing["per_point_timing"]
            .as_array()
            .unwrap()
            .iter()
            .all(|width| width["failed_lanes"] == 0)
    );
    let checked = value(run(&[
        "numerator-bench",
        pool_fixture,
        "--points",
        "validation/benchmark/points/hgg.json",
        "--feynman-rules",
        "--r2",
        "fdh",
        "--backend",
        "interpreter",
        "--check-case",
        case_path.to_str().unwrap(),
        "--generation-only",
    ]));
    assert_eq!(checked["mu_powers"], 1);
    assert!(
        checked["check_case"]["max_tensor_error_relative_to_lane_max"]
            .as_f64()
            .unwrap()
            < 1e-12
    );
    let _ = std::fs::remove_dir_all(directory);
}
