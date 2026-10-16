use feynkit_graph::{ExternalState, VertexId};
use ninjars_graph::{
    FourMomentum, IntegralFamily, Kinematics, Model, NumeratorMode, PreparedIntegrand, prepare_dot,
    symbols,
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{atom::AtomField, float::Complex, rational::Rational},
    parse,
};
fn atom_real(v: f64) -> Atom {
    assert!(v.is_finite());
    Atom::num(Rational::try_from(v).unwrap())
}
fn atom_complex(v: &Value) -> Atom {
    atom_real(v[0].as_f64().unwrap()) + atom_real(v[1].as_f64().unwrap()) * Atom::i()
}
fn four(v: Vec<Atom>) -> FourMomentum<Atom> {
    FourMomentum::from_args(v[0].clone(), v[1].clone(), v[2].clone(), v[3].clone())
}
// Topological tree labels preserve nested Higgs vertices as well as leaves.
// This performs no numerator or momentum algebra.
fn cycle(g: &PreparedIntegrand) -> Vec<String> {
    let mut next = BTreeMap::new();
    let mut adjacent: BTreeMap<VertexId, Vec<(Option<VertexId>, Option<usize>)>> = BTreeMap::new();
    for (_, ends, data) in g.diagram.edges() {
        let pdg = g
            .diagram
            .model()
            .particle_by_id(data.particle)
            .unwrap()
            .pdg_code;
        if pdg.abs() == 6 {
            next.insert(ends.source.unwrap(), ends.target.unwrap());
        } else if let Some(external) = &data.external {
            adjacent
                .entry(ends.source.or(ends.target).unwrap())
                .or_default()
                .push((None, Some(external.index + 1)));
        } else {
            assert_eq!(pdg, 25);
            let a = ends.source.unwrap();
            let b = ends.target.unwrap();
            adjacent.entry(a).or_default().push((Some(b), None));
            adjacent.entry(b).or_default().push((Some(a), None));
        }
    }
    fn tree(
        v: VertexId,
        parent: Option<VertexId>,
        adjacent: &BTreeMap<VertexId, Vec<(Option<VertexId>, Option<usize>)>>,
    ) -> String {
        let mut children = adjacent[&v]
            .iter()
            .filter(|(neighbor, _)| neighbor.is_none() || *neighbor != parent)
            .map(|(neighbor, leaf)| match (neighbor, leaf) {
                (_, Some(index)) => index.to_string(),
                (Some(child), None) => tree(*child, Some(v), adjacent),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        children.sort();
        if parent.is_none() {
            assert_eq!(children.len(), 1);
            children.pop().unwrap()
        } else {
            assert!(children.len() == 2 || children.len() == 3);
            format!("H[{}]", children.join(","))
        }
    }
    let legs = next
        .keys()
        .map(|v| (*v, tree(*v, None, &adjacent)))
        .collect::<BTreeMap<_, _>>();
    let start = *legs.iter().find(|(_, l)| l.as_str() == "1").unwrap().0;
    let mut v = start;
    let mut result = Vec::new();
    for _ in 0..next.len() {
        result.push(legs[&v].clone());
        v = next[&v];
    }
    assert_eq!(v, start);
    result
}

fn main() {
    let input = std::env::args()
        .nth(1)
        .expect("supply the archived MadLoop reference JSON");
    let summary: Value = serde_json::from_slice(&std::fs::read(input).unwrap()).unwrap();
    let fixture_dir =
        std::path::PathBuf::from(std::env::args().nth(2).expect("FeynKit fixtures directory"));
    let metadata: Value =
        serde_json::from_slice(&std::fs::read(fixture_dir.join("feynkit-graphs.json")).unwrap())
            .unwrap();
    assert_eq!(summary["partial"], false);
    assert_eq!(summary["graphs"].as_array().unwrap().len(), 50);
    assert_eq!(metadata["graphs"].as_array().unwrap().len(), 50);
    let graphs = metadata["graphs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| {
            let dot = std::fs::read_to_string(
                fixture_dir.join(format!("{}.dot", g["name"].as_str().unwrap())),
            )
            .unwrap();
            prepare_dot(
                Model::standard_model(),
                &dot,
                &Kinematics::new(),
                NumeratorMode::Explicit,
            )
            .unwrap()
            .with_color_dimension_invariants()
            .unwrap()
        })
        .collect::<Vec<_>>();
    let variables = [
        parse!("ninjars::q0"),
        parse!("ninjars::q1"),
        parse!("ninjars::q2"),
        parse!("ninjars::q3"),
    ];
    let q = four(variables.to_vec());
    let mut result = Vec::new();
    let mut total = 0;
    let mut maximum = 0.0_f64;
    let mut matched_ids = BTreeSet::new();
    for (position, g) in graphs.into_iter().enumerate() {
        let orientation = metadata["graphs"][position]["index"].as_u64().unwrap();
        let directed = cycle(&g);
        let matched = summary["graphs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| {
                let mut order = r["wavefunction_order"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| {
                        r["wavefunction_trees"][x.as_u64().unwrap().to_string()]
                            .as_str()
                            .unwrap()
                            .to_string()
                    })
                    .collect::<Vec<_>>();
                if r["fermion_seed"] == "LCUT_AF" {
                    order[1..].reverse();
                }
                order == directed
            })
            .unwrap();
        let id = matched["graph_filter_id"].as_u64().unwrap();
        assert!(matched_ids.insert(id), "duplicate physical graph mapping");
        let reference = matched;
        let order = reference["wavefunction_order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        let mut physical = BTreeMap::new();
        for (edge, _, data) in g.diagram.edges() {
            if let Some(e) = &data.external {
                let routed = g
                    .diagram
                    .loop_momentum_basis()
                    .route_expression(&symbols::momentum().call(edge.0));
                physical.insert(
                    e.index + 1,
                    if e.state == ExternalState::Incoming {
                        routed
                    } else {
                        -routed
                    },
                );
            }
        }
        let loop_name = &g.family.loop_momenta()[0];
        let mut shift = Atom::Zero;
        let mut denominators = Vec::new();
        for leg in &order {
            let momentum = loop_name + &shift;
            denominators.push(
                g.family
                    .kinematics()
                    .scalar_product(&momentum, &momentum)
                    .unwrap()
                    - parse!("UFO::MT^2"),
            );
            for external in reference["wavefunction_leaves"][leg.to_string()]
                .as_array()
                .unwrap()
            {
                shift += &physical[&(external.as_u64().unwrap() as usize)];
            }
        }
        assert!(shift.expand().is_zero());
        let target = IntegralFamily::new(
            g.family.loop_momenta().to_vec(),
            g.family.external_momenta().to_vec(),
            denominators,
            g.family.kinematics(),
        )
        .unwrap();
        let mapping = g.family.find_mapping(&target, 100).unwrap().unwrap();
        for (i, j) in mapping.denominator_map().iter().enumerate() {
            assert!(
                (mapping.apply(&g.family.denominators()[i]) - &target.denominators()[*j])
                    .expand()
                    .is_zero()
            );
        }
        eprintln!(
            "orientation {orientation} cycle {directed:?} -> {id}, map {:?}, permutation {:?}",
            mapping.momentum_rules(),
            mapping.denominator_map()
        );
        // Independent exact normalization control: all momenta zero, temporal
        // gluon polarizations. Dirac trace four times Tr(TaTb)=delta_ab/2,
        // one closed-fermion sign, and the existing Higgs bridge denominator.
        let zero = four(vec![Atom::Zero; 4]);
        let temporal = four(vec![Atom::num(1), Atom::Zero, Atom::Zero, Atom::Zero]);
        let mut controls = BTreeMap::new();
        for (position, edge) in g
            .diagram
            .loop_momentum_basis()
            .external_edges
            .iter()
            .enumerate()
        {
            controls.insert(symbols::external_momentum().call(position), zero.clone());
            let data = g.diagram.edges().find(|(i, _, _)| i == edge).unwrap().2;
            if data.external.as_ref().unwrap().index < 2 {
                controls.insert(symbols::epsilon().call(edge.0), temporal.clone());
            }
        }
        let normalization = g
            .component_kinematics(&zero, &controls)
            .unwrap()
            .apply(&(&g.numerator * &g.overall_factor))
            .together();
        let normalization_expected = match (
            g.quadratics.len(),
            reference["tree_couplings"].as_array().unwrap().len(),
        ) {
            (5, 0) => parse!("-2𝑖*UFO::GC_11^2*UFO::GC_94^3*UFO::MT^5"),
            (4, 1) => parse!("2𝑖*UFO::GC_11^2*UFO::GC_94^2*UFO::GC_69*UFO::MT^4/UFO::MH^2"),
            (3, 1) => parse!("2*UFO::GC_11^2*UFO::GC_94*UFO::GC_33*UFO::MT^3/UFO::MH^2"),
            (3, 2) => parse!("-2𝑖*UFO::GC_11^2*UFO::GC_94*UFO::GC_69^2*UFO::MT^3/UFO::MH^4"),
            other => panic!("unexpected scalar-tree class {other:?}"),
        };
        assert!(
            (&normalization - &normalization_expected)
                .together()
                .is_zero(),
            "exact normalization control: {normalization}"
        );
        let mut cases = Vec::new();
        for case in reference["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["ctmode"] == 4)
        {
            let mut components = BTreeMap::new();
            for (position, edge) in g
                .diagram
                .loop_momentum_basis()
                .external_edges
                .iter()
                .enumerate()
            {
                let data = g.diagram.edges().find(|(i, _, _)| i == edge).unwrap().2;
                let e = data.external.as_ref().unwrap();
                let key = (e.index + 1).to_string();
                components.insert(
                    symbols::external_momentum().call(position),
                    four(
                        case["external_momenta"][&key]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|x| atom_real(x.as_f64().unwrap()))
                            .collect(),
                    ),
                );
                if e.index >= 2 {
                    continue;
                }
                let carrier = if e.state == ExternalState::Incoming {
                    symbols::epsilon()
                } else {
                    symbols::epsilonbar()
                };
                components.insert(
                    carrier.call(edge.0),
                    four(
                        case["external_polarizations"][&key]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(atom_complex)
                            .collect(),
                    ),
                );
            }
            let mut mapped_q = four(vec![Atom::Zero; 4]);
            for (momentum, coefficient) in g
                .family
                .kinematics()
                .linear_terms(&mapping.momentum_rules()[0].1)
                .unwrap()
            {
                mapped_q += if momentum == *loop_name {
                    q.clone() * coefficient
                } else {
                    components[&momentum].clone() * coefficient
                };
            }
            let inputs = case["inputs_mt_mh_muR_wh_reimGC5_reimGC37_reimGC30_reimGCHHHH"]
                .as_array()
                .unwrap();
            let kin = g.component_kinematics(&mapped_q, &components).unwrap();
            let expression = kin
                .apply(&(&g.numerator * &g.overall_factor))
                .replace(parse!("UFO::MT"))
                .with(atom_real(inputs[0].as_f64().unwrap()))
                .replace(parse!("UFO::MH"))
                .with(atom_real(inputs[1].as_f64().unwrap()))
                .replace(parse!("UFO::GC_11"))
                .with(
                    atom_real(inputs[4].as_f64().unwrap())
                        + atom_real(inputs[5].as_f64().unwrap()) * Atom::i(),
                )
                .replace(parse!("UFO::GC_94"))
                .with(
                    atom_real(inputs[6].as_f64().unwrap())
                        + atom_real(inputs[7].as_f64().unwrap()) * Atom::i(),
                )
                .replace(parse!("UFO::GC_69"))
                .with(
                    atom_real(inputs[8].as_f64().unwrap())
                        + atom_real(inputs[9].as_f64().unwrap()) * Atom::i(),
                )
                .replace(parse!("UFO::GC_33"))
                .with(
                    atom_real(inputs[10].as_f64().unwrap())
                        + atom_real(inputs[11].as_f64().unwrap()) * Atom::i(),
                );
            let polynomial = expression.to_polynomial_in_vars_with_field::<u16>(
                &variables,
                &AtomField {
                    statistical_zero_test: false,
                    ..AtomField::new()
                },
            );
            assert!(
                (&polynomial)
                    .into_iter()
                    .all(|term| term.exponents.iter().sum::<u16>() <= 4
                        && variables.iter().all(|v| !term.coefficient.contains(v)))
            );
            assert!(!expression.contains(&parse!("ninjars::mu2")));
            let degree = (&polynomial)
                .into_iter()
                .map(|t| t.exponents.iter().sum::<u16>())
                .max()
                .unwrap_or(0);
            assert!(degree <= if g.quadratics.len() == 3 { 2 } else { 4 });
            let rows = case["numerator_samples"].as_array().unwrap();
            assert_eq!(rows.len(), 70);
            assert_eq!(
                rows.iter()
                    .map(|r| r["id"].as_u64().unwrap())
                    .collect::<BTreeSet<_>>()
                    .len(),
                70
            );
            let mut max = 0.0_f64;
            for row in case["numerator_samples"].as_array().unwrap() {
                let bindings: HashMap<_, _> =
                    variables
                        .iter()
                        .cloned()
                        .zip(
                            row["q"].as_array().unwrap().iter().map(|x| {
                                Complex::new(x[0].as_f64().unwrap(), x[1].as_f64().unwrap())
                            }),
                        )
                        .collect();
                let actual = expression.evaluate_with_prec(&bindings, 53).unwrap();
                let expected = Complex::new(
                    -0.5 * row["raw"][0].as_f64().unwrap(),
                    -0.5 * row["raw"][1].as_f64().unwrap(),
                );
                let error = (actual.re - expected.re).hypot(actual.im - expected.im)
                    / 1.0_f64.max(expected.re.hypot(expected.im));
                assert!(error.is_finite(), "nonfinite raw comparison");
                if error > 1e-9 {
                    eprintln!(
                        "FAIL {orientation} {id} {} point{} actual{actual:?} expected{expected:?} error{error:e}",
                        case["state"], row["id"]
                    );
                }
                max = max.max(error);
                total += 1;
            }
            maximum = maximum.max(max);
            eprintln!("{} maxerror {max:e}", case["state"]);
            cases.push(json!({"state":case["state"],"checks":case["numerator_samples"].as_array().unwrap().len(),"maximum_scaled_error":max,"loop_degree":degree}));
        }
        result.push(json!({"feynkit_orientation":orientation,"directed_physical_cycle":directed,"madloop_filter_id":id,"momentum_rules":mapping.momentum_rules().iter().map(|(a,b)|[a.to_plain_string(),b.to_plain_string()]).collect::<Vec<_>>(),"denominator_map":mapping.denominator_map(),"exact_normalization_control":normalization.to_canonical_string(),"cases":cases}));
    }
    assert_eq!(matched_ids.len(), 50);
    assert_eq!(total, 28000);
    assert_eq!(
        total,
        summary["expected_raw_checks"].as_u64().unwrap() as usize
    );
    println!("{}",serde_json::to_string_pretty(&json!({"checks":total,"maximum_scaled_error":maximum,"pass":maximum<=1e-9,"graphs":result})).unwrap());
    if maximum > 1e-9 {
        std::process::exit(1);
    }
}
