//! Shared three-point phase-space pools and pool fixtures for the benchmark
//! representatives of every physical process.
//!
//! ```sh
//! cargo run --profile perf -p ninjars-cli --example benchmark_pools -- .
//! ```
//!
//! Momenta are exact rationals (gg->HHH reuses the radicals of the existing
//! changing pool). Symbolica checks momentum conservation, mass shells and
//! physical thresholds exactly; every component is evaluated at 256 bits and
//! rounded once to binary64, and these binary64 rows are what NinjaRS and
//! MadLoop both consume. Gluon/photon polarizations are the `+` helicity
//! states of HELAS `vxxxxx` (FeynKit `Epsilon` for incoming, `EpsilonBar` for
//! outgoing legs), written exactly into the fixtures and checked against
//! `ninjars_graph::vector_wavefunction` on the rounded momenta.

use std::{collections::HashMap, fs, path::Path, sync::Arc};

use ninjars_graph::{ExternalState, FeynmanDiagram, FourMomentum, Model, vector_wavefunction};
use serde_json::{Value, json};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::float::{Complex, Float, RealLike},
    parser::ParseSettings,
};

fn high(expression: &Atom) -> Complex<f64> {
    let value: Complex<Float> = expression
        .evaluate_with_prec(&HashMap::<Atom, Complex<Float>>::new(), 256)
        .unwrap();
    let value = Complex::new(value.re.to_f64(), value.im.to_f64());
    assert!(value.re.is_finite() && value.im.is_finite());
    value
}
fn rounded(expression: &Atom) -> f64 {
    let value = high(expression);
    assert!(value.im == 0., "{expression} is not real");
    value.re
}
fn q(text: &str) -> Atom {
    Atom::parse(text, "ninjars", ParseSettings::default()).unwrap()
}
fn massless(energy: &Atom, direction: [&str; 3]) -> FourMomentum<Atom> {
    let [x, y, z] = direction.map(q);
    FourMomentum::from_args(energy.clone(), energy * &x, energy * &y, energy * &z)
}
fn components(p: &FourMomentum<Atom>) -> [Atom; 4] {
    let [e, x, y, z] = p.components();
    [e.clone(), x.clone(), y.clone(), z.clone()]
}
fn text(atom: &Atom) -> String {
    atom.to_canonical_string()
}

/// HELAS `vxxxxx(p, 0, +1, nsv)` exactly: `epsilon(+)` for incoming legs and
/// its conjugate for outgoing legs (massless vectors only).
fn plus_polarization(p: &FourMomentum<Atom>, incoming: bool) -> [Atom; 4] {
    let [e, x, y, z] = components(p);
    let i = Atom::i();
    let root2 = q("2^(1/2)");
    let pt2 = (&x * &x + &y * &y).expand();
    let (a, b) = if pt2.is_zero() {
        let sign = if rounded(&z) > 0. {
            Atom::num(1)
        } else {
            Atom::num(-1)
        };
        (
            [Atom::Zero, Atom::num(1), Atom::Zero, Atom::Zero],
            [Atom::Zero, Atom::Zero, sign, Atom::Zero],
        )
    } else {
        let pt = q("ninjars::pt2^(1/2)").replace(q("ninjars::pt2")).with(pt2);
        (
            [
                Atom::Zero,
                &x * &z / (&pt * &e),
                &y * &z / (&pt * &e),
                -(&pt / &e),
            ],
            [Atom::Zero, -(&y / &pt), &x / &pt, Atom::Zero],
        )
    };
    let sign = if incoming { -1 } else { 1 };
    std::array::from_fn(|k| ((-&a[k] + &b[k] * &i * sign) / &root2).expand())
}

struct Leg {
    pdg: i64,
    incoming: bool,
    mass: &'static str,
}

struct Graph {
    name: &'static str,
    madloop_filter_id: u32,
    denominator_map: &'static [usize],
}

struct Process {
    name: &'static str,
    madgraph: &'static str,
    legs: Vec<Leg>,
    runtime: Vec<(&'static str, &'static str)>,
    fixed: Vec<(&'static str, &'static str)>,
    scale_squared: &'static str,
    graphs: Vec<Graph>,
    raw_factor: &'static str,
    laurent_factor: &'static str,
    points: Vec<(String, Vec<FourMomentum<Atom>>)>,
    construction: &'static str,
    /// Recorded FeynKit/MadLoop graph matching (filter id, denominator map,
    /// loop-momentum rule and normalization) the graph metadata is taken from.
    evidence: &'static str,
}

const SM7: [(&str, &str); 7] = [
    ("MT", "173"),
    ("ymt", "173"),
    ("MH", "125"),
    ("Gf", "116639/10000000000"),
    ("aS", "59/500"),
    ("aEWM1", "132507/1000"),
    ("MZ", "22797/250"),
];

fn gluon(incoming: bool) -> Leg {
    Leg {
        pdg: 21,
        incoming,
        mass: "0",
    }
}
fn photon(incoming: bool) -> Leg {
    Leg {
        pdg: 22,
        incoming,
        mass: "0",
    }
}
fn higgs(incoming: bool) -> Leg {
    Leg {
        pdg: 25,
        incoming,
        mass: "125",
    }
}

/// Two massless momenta along `n1`, `n2` with invariant mass 125.
fn decay(e1: &str, n1: [&str; 3], n2: [&str; 3]) -> Vec<FourMomentum<Atom>> {
    let e1 = q(e1);
    let g1 = massless(&e1, n1);
    let dot = n1
        .iter()
        .zip(n2)
        .fold(Atom::Zero, |sum, (a, b)| sum + q(a) * q(b));
    let e2 = (q("125^2") / (Atom::num(2) * &e1 * (Atom::num(1) - dot))).expand();
    let g2 = massless(&e2, n2);
    let h = g1.clone() + g2.clone();
    vec![h, g1, g2]
}

/// Beams along +-z with energy `e`, outgoing pair along +-n with momentum `k`.
fn two_to_two(e: &str, k: &str, n: [&str; 3]) -> Vec<FourMomentum<Atom>> {
    let (e, k) = (q(e), q(k));
    let [x, y, z] = n.map(q);
    vec![
        FourMomentum::from_args(e.clone(), Atom::Zero, Atom::Zero, e.clone()),
        FourMomentum::from_args(e.clone(), Atom::Zero, Atom::Zero, -&e),
        FourMomentum::from_args(e.clone(), &k * &x, &k * &y, &k * &z),
        FourMomentum::from_args(e.clone(), -(&k * &x), -(&k * &y), -(&k * &z)),
    ]
}

fn processes(root: &Path) -> Vec<Process> {
    let changing: Value = serde_json::from_str(
        &fs::read_to_string(root.join("validation/evidence/gg-hhh-changing-pool.json")).unwrap(),
    )
    .unwrap();
    let hhh_points = changing["points"]
        .as_array()
        .unwrap()
        .iter()
        .map(|point| {
            let momenta: Vec<_> = point["exact_momenta"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| {
                    let [e, x, y, z] = [0, 1, 2, 3].map(|k| q(p[k].as_str().unwrap()));
                    FourMomentum::from_args(e, x, y, z)
                })
                .collect();
            // The shared binary64 rows must be reproduced bit for bit.
            for (p, expected) in momenta.iter().zip(point["momenta"].as_array().unwrap()) {
                for (component, value) in components(p).iter().zip(expected.as_array().unwrap()) {
                    assert_eq!(rounded(component), value.as_f64().unwrap());
                }
            }
            (point["id"].as_str().unwrap().to_string(), momenta)
        })
        .collect();
    vec![
        Process {
            name: "hgg",
            madgraph: "h > g g QED=1 [virt=QCD]",
            legs: vec![higgs(true), gluon(false), gluon(false)],
            runtime: SM7.to_vec(),
            fixed: vec![("WT", "0"), ("WH", "0")],
            scale_squared: "15625",
            graphs: vec![Graph {
                name: "top-triangle-0",
                madloop_filter_id: 3,
                denominator_map: &[2, 0, 1],
            }],
            raw_factor: "-1/2",
            laurent_factor: "-8*pi^2",
            points: vec![
                (
                    "rest_frame".into(),
                    decay("125/2", ["2/3", "1/3", "2/3"], ["-2/3", "-1/3", "-2/3"]),
                ),
                (
                    "boosted_a".into(),
                    decay("100", ["2/7", "3/7", "6/7"], ["-6/7", "2/7", "-3/7"]),
                ),
                (
                    "boosted_b".into(),
                    decay("150", ["4/9", "-4/9", "7/9"], ["1/3", "2/3", "-2/3"]),
                ),
            ],
            construction: "Gluons are massless rational momenta E_i (1, n_i) with rational unit directions; E_2 = MH^2/(2 E_1 (1 - n_1.n_2)) and p_H = p_1 + p_2. The first point is the Higgs rest frame, the other two are boosted Higgs bosons with generic gluon directions.",
            evidence: "validation/evidence/physical-hgg-madloop.json (denominator_order, routing, ninjars_conversion)",
        },
        Process {
            name: "four-photon",
            madgraph: "a a > a a [virt=QED]",
            legs: vec![photon(true), photon(true), photon(false), photon(false)],
            runtime: vec![("MT", "173"), ("aEWM1", "132507/1000")],
            fixed: vec![("WT", "0")],
            scale_squared: "10000",
            graphs: vec![Graph {
                name: "top-box-0",
                madloop_filter_id: 182,
                denominator_map: &[3, 0, 2, 1],
            }],
            raw_factor: "-3",
            laurent_factor: "-48*pi^2",
            points: vec![
                (
                    "sqrt_s_200".into(),
                    two_to_two("100", "100", ["2/3", "1/3", "2/3"]),
                ),
                (
                    "sqrt_s_500".into(),
                    two_to_two("250", "250", ["2/7", "-3/7", "6/7"]),
                ),
                (
                    "sqrt_s_800".into(),
                    two_to_two("400", "400", ["-4/9", "4/9", "7/9"]),
                ),
            ],
            construction: "Photon beams along +-z with energy sqrt(s)/2; outgoing photons back to back along rational unit directions with all components nonzero. sqrt(s)=200 lies below the top-pair threshold 2*MT=346, the other two above.",
            evidence: "validation/evidence/four-photon-raw.json",
        },
        Process {
            name: "gg-hh",
            madgraph: "g g > h h QCD=2 QED=2 [virt=QCD]",
            legs: vec![gluon(true), gluon(true), higgs(false), higgs(false)],
            runtime: SM7.to_vec(),
            fixed: vec![("WT", "0"), ("WH", "0")],
            scale_squared: "10000",
            graphs: vec![
                Graph {
                    name: "top-triangle-0",
                    madloop_filter_id: 11,
                    denominator_map: &[2, 0, 1],
                },
                Graph {
                    name: "top-box-2",
                    madloop_filter_id: 10,
                    denominator_map: &[3, 0, 2, 1],
                },
            ],
            raw_factor: "-1/2",
            laurent_factor: "-8*pi^2",
            points: vec![
                (
                    "sqrt_s_312.5".into(),
                    two_to_two("625/4", "375/4", ["2/3", "1/3", "2/3"]),
                ),
                (
                    "sqrt_s_416.7".into(),
                    two_to_two("625/3", "500/3", ["2/7", "-3/7", "6/7"]),
                ),
                (
                    "sqrt_s_650".into(),
                    two_to_two("325", "300", ["-4/9", "4/9", "7/9"]),
                ),
            ],
            construction: "Gluon beams along +-z with energy E=sqrt(s)/2; Higgs bosons back to back along rational unit directions with momentum k, (E,k)=125*(5/4,3/4), 125*(5/3,4/3), 125*(13/5,12/5). The first point lies between the HH threshold 250 and the top-pair threshold 346, the other two above both.",
            evidence: "validation/evidence/gg-hh-raw.json",
        },
        Process {
            name: "gg-hhh",
            madgraph: "g g > h h h QCD=2 QED=3 [virt=QCD]",
            legs: vec![
                gluon(true),
                gluon(true),
                higgs(false),
                higgs(false),
                higgs(false),
            ],
            runtime: SM7.to_vec(),
            fixed: vec![("WT", "0"), ("WH", "0")],
            scale_squared: "10000",
            graphs: vec![
                Graph {
                    name: "top-triangle-0",
                    madloop_filter_id: 66,
                    denominator_map: &[2, 0, 1],
                },
                Graph {
                    name: "top-box-8",
                    madloop_filter_id: 58,
                    denominator_map: &[3, 0, 2, 1],
                },
                Graph {
                    name: "top-pentagon-26",
                    madloop_filter_id: 55,
                    denominator_map: &[4, 0, 2, 1, 3],
                },
            ],
            raw_factor: "-1/2",
            laurent_factor: "-8*pi^2",
            points: hhh_points,
            construction: "Reuses validation/evidence/gg-hhh-changing-pool.json unchanged (exact radicals; binary64 rows reproduced bit for bit): sqrt(s)=450,525,600 with equal-energy Higgs bosons forming an equilateral triangle in the transverse plane.",
            evidence: "validation/evidence/gg-hhh-raw.json",
        },
    ]
}

fn main() {
    let root = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    let root = Path::new(&root);
    let model = Arc::new(Model::standard_model());
    for process in processes(root) {
        let legs = process.legs.len();
        // Exact checks: conservation, mass shells, positive energies, threshold.
        for (id, momenta) in &process.points {
            assert_eq!(momenta.len(), legs);
            let mut total = FourMomentum::from_args(Atom::Zero, Atom::Zero, Atom::Zero, Atom::Zero);
            for (p, leg) in momenta.iter().zip(&process.legs) {
                total = if leg.incoming {
                    total + p.clone()
                } else {
                    total - p.clone()
                };
                let mass = q(leg.mass);
                assert!(
                    (p.mass_squared() - &mass * &mass).expand().is_zero(),
                    "{} {id}: off shell",
                    process.name
                );
                assert!(
                    rounded(&components(p)[0]) > 0.,
                    "{} {id}: energy",
                    process.name
                );
            }
            assert!(
                components(&total).iter().all(|c| c.expand().is_zero()),
                "{} {id}: momentum conservation",
                process.name
            );
            let mut incoming =
                FourMomentum::from_args(Atom::Zero, Atom::Zero, Atom::Zero, Atom::Zero);
            let mut threshold = Atom::Zero;
            for (p, leg) in momenta.iter().zip(&process.legs) {
                if leg.incoming {
                    incoming += p.clone();
                } else {
                    threshold += q(leg.mass);
                }
            }
            let s = rounded(&incoming.mass_squared().expand());
            let threshold = rounded(&threshold);
            assert!(
                s >= threshold * threshold * (1. + 1e-12),
                "{} {id}: below threshold",
                process.name
            );
        }
        let helicities: Vec<i8> = process
            .legs
            .iter()
            .map(|leg| if leg.pdg == 25 { 0 } else { 1 })
            .collect();
        let model_values: Vec<f64> = process
            .runtime
            .iter()
            .map(|(_, v)| rounded(&q(v)))
            .collect();
        let scale_squared = rounded(&q(process.scale_squared));
        let points: Vec<Value> = process
            .points
            .iter()
            .map(|(id, momenta)| {
                let incoming = momenta
                    .iter()
                    .zip(&process.legs)
                    .filter(|(_, leg)| leg.incoming)
                    .fold(FourMomentum::from_args(Atom::Zero, Atom::Zero, Atom::Zero, Atom::Zero), |s, (p, _)| s + p.clone());
                json!({
                    "id": id,
                    "sqrt_s": rounded(&incoming.mass_squared().expand()).sqrt(),
                    "momenta": momenta.iter().map(|p| components(p).iter().map(rounded).collect::<Vec<_>>()).collect::<Vec<_>>(),
                    "exact_momenta": momenta.iter().map(|p| components(p).iter().map(text).collect::<Vec<_>>()).collect::<Vec<_>>(),
                    "helicities": helicities,
                    "model": model_values,
                    "scale_squared": scale_squared,
                    "mu": scale_squared.sqrt(),
                })
            })
            .collect();
        let mut graphs = Vec::new();
        for graph in &process.graphs {
            let directory = root.join("examples/graphs/physical").join(process.name);
            let template: Value = serde_json::from_str(
                &fs::read_to_string(directory.join(format!("{}.json", graph.name))).unwrap(),
            )
            .unwrap();
            let dot = fs::read_to_string(directory.join(format!("{}.dot", graph.name))).unwrap();
            let diagram = FeynmanDiagram::from_dot(model.clone(), &dot).unwrap();
            // Map FeynKit external-basis positions to physical legs.
            let mut position_of_leg = vec![usize::MAX; legs];
            let mut vector_legs = Vec::new();
            for (position, edge) in diagram
                .loop_momentum_basis()
                .external_edges
                .iter()
                .enumerate()
            {
                let (_, _, data) = diagram.edges().find(|(id, _, _)| id == edge).unwrap();
                let external = data.external.as_ref().unwrap();
                let particle = diagram.model().particle_by_id(data.particle).unwrap();
                let leg = &process.legs[external.index];
                assert_eq!(
                    particle.pdg_code, leg.pdg,
                    "{} {}",
                    process.name, graph.name
                );
                assert_eq!(external.state == ExternalState::Incoming, leg.incoming);
                position_of_leg[external.index] = position;
                if particle.spin == 3 {
                    vector_legs.push(external.index);
                }
            }
            vector_legs.sort();
            let samples: Vec<Value> = process
                .points
                .iter()
                .map(|(id, momenta)| {
                    let mut external_momenta = serde_json::Map::new();
                    for (leg, p) in momenta.iter().enumerate() {
                        external_momenta.insert(
                            position_of_leg[leg].to_string(),
                            json!(components(p).iter().map(text).collect::<Vec<_>>()),
                        );
                    }
                    let mut polarizations = serde_json::Map::new();
                    for &leg in &vector_legs {
                        let incoming = process.legs[leg].incoming;
                        let exact = plus_polarization(&momenta[leg], incoming);
                        let binary64 = components(&momenta[leg]).map(|c| rounded(&c));
                        let reference = vector_wavefunction(&binary64, 1, !incoming).unwrap();
                        for (value, reference) in exact.iter().zip(reference) {
                            let value = high(value);
                            assert!(
                                (value.re - reference.re).hypot(value.im - reference.im) < 4e-16,
                                "{} {id} leg {leg}: polarization {value:?} != {reference:?}",
                                process.name
                            );
                        }
                        polarizations.insert(
                            leg.to_string(),
                            json!(exact.iter().map(text).collect::<Vec<_>>()),
                        );
                    }
                    json!({"id": id, "external_momenta": external_momenta, "external_polarizations": polarizations})
                })
                .collect();
            let fixture = json!({
                "schema_version": 1,
                "graph": format!("{}.dot", graph.name),
                "model": template["model"],
                "scale_squared": process.scale_squared,
                "color_dimension_invariants": template["color_dimension_invariants"],
                "parameter_overrides": template["parameter_overrides"],
                "samples": samples,
            });
            let path = directory.join(format!("{}-pool.json", graph.name));
            fs::write(
                &path,
                serde_json::to_string_pretty(&fixture).unwrap() + "\n",
            )
            .unwrap();
            println!("{}", path.display());
            graphs.push(json!({
                "graph": graph.name,
                "fixture": format!("examples/graphs/physical/{}/{}-pool.json", process.name, graph.name),
                "bench_case": format!("validation/benchmark/cases/{}-{}.json", process.name, graph.name),
                "bench_case_command": format!(
                    "ninjars export-bench examples/graphs/physical/{p}/{g}-pool.json --name {p}-{g} --output validation/benchmark/cases/{p}-{g}.json",
                    p = process.name,
                    g = graph.name
                ),
                "madloop_evidence": process.evidence,
                "madloop_loop_filter_id": graph.madloop_filter_id,
                "madloop_denominator_map": graph.denominator_map,
                "loop_momentum": "q_FeynKit = -q of the MadLoop LOOPNUM/CutTools numerator (recorded momentum rule K(0) -> -K(0))",
                "raw_numerator_factor": process.raw_factor,
                "integrated_laurent_factor": process.laurent_factor,
            }));
        }
        let pool = json!({
            "format": "ninjars-phase-space-pool-v1",
            "process": process.name,
            "madgraph_process": process.madgraph,
            "legs": process.legs.iter().enumerate().map(|(i, leg)| json!({"index": i, "pdg": leg.pdg, "state": if leg.incoming { "incoming" } else { "outgoing" }, "mass": leg.mass})).collect::<Vec<_>>(),
            "model": "sm",
            "model_parameters": process.runtime.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
            "exact_model_inputs": process.runtime.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
            "fixed_parameters": process.fixed.iter().map(|(n, v)| (n.to_string(), Value::String(v.to_string()))).collect::<serde_json::Map<_, _>>(),
            "scale_squared": process.scale_squared,
            "helicities": helicities,
            "helicity_convention": {
                "label": "pp (all vector bosons helicity +1)",
                "madloop": "HELAS VXXXXX(P,0,NHEL,-1) for incoming and VXXXXX(P,0,NHEL,+1) for outgoing vectors (IC=1); NHEL is the physical helicity of each leg in MadGraph leg order. The row equals one line of MadLoop5_resources/ML5_0_HelConfigs.dat; in the generated reference exports (+1,+1,...) is helicity configuration 4 of 4 for h>gg, gg>hh and gg>hhh and 16 of 16 for aa>aa (1-based).",
                "feynkit": "FourMomentum::wavefunction(Epsilon, +1) for incoming and (EpsilonBar, +1) for outgoing vectors; ninjars_graph::vector_wavefunction reproduces it without allocation. Scalars have helicity 0 and wavefunction 1.",
                "fixtures": "external_polarizations hold these vectors exactly; external_momenta keys are FeynKit external-basis positions."
            },
            "widths": process.fixed.iter().map(|(n, v)| format!("{n}={v}")).collect::<Vec<_>>().join(","),
            "momentum_convention": "MadGraph leg order, incoming then outgoing, physical positive-energy four-momenta (E,px,py,pz); binary64 values are 256-bit evaluations of exact_momenta rounded once.",
            "construction": process.construction,
            "generator": "cargo run --profile perf -p ninjars-cli --example benchmark_pools -- .",
            "graphs": graphs,
            "points": points,
        });
        let path = root.join(format!("validation/benchmark/points/{}.json", process.name));
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, serde_json::to_string_pretty(&pool).unwrap() + "\n").unwrap();
        println!("{}", path.display());
    }
}
