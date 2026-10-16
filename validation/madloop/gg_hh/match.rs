use feynkit_graph::ExternalState;
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
fn cycle(g: &PreparedIntegrand) -> Vec<usize> {
    let mut legs = BTreeMap::new();
    let mut next = BTreeMap::new();
    for (_, ends, data) in g.diagram.edges() {
        if g.diagram
            .model()
            .particle_by_id(data.particle)
            .unwrap()
            .pdg_code
            .abs()
            == 6
        {
            next.insert(ends.source.unwrap(), ends.target.unwrap());
        }
    }
    for (_, ends, data) in g.diagram.edges() {
        if let Some(e) = &data.external {
            let v = ends.source.or(ends.target).unwrap();
            if next.contains_key(&v) {
                legs.insert(v, e.index + 1);
            }
        } else if g
            .diagram
            .model()
            .particle_by_id(data.particle)
            .unwrap()
            .pdg_code
            == 25
        {
            let v = [ends.source.unwrap(), ends.target.unwrap()]
                .into_iter()
                .find(|v| next.contains_key(v))
                .unwrap();
            legs.insert(v, 5);
        }
    }
    let start = *legs.iter().find(|(_, l)| **l == 1).unwrap().0;
    let mut v = start;
    let mut result = Vec::new();
    for _ in 0..next.len() {
        result.push(legs[&v]);
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
                    .map(|x| x.as_u64().unwrap() as usize)
                    .collect::<Vec<_>>();
                if r["fermion_seed"] == "LCUT_AF" {
                    order[1..].reverse();
                }
                order == directed
            })
            .unwrap();
        let id = matched["graph_filter_id"].as_u64().unwrap();
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
        if g.quadratics.len() == 3 {
            physical.insert(5, -&physical[&1] - &physical[&2]);
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
            shift += &physical[leg];
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
        let normalization_expected = if g.quadratics.len() == 3 {
            parse!("2*UFO::GC_11^2*UFO::GC_94*UFO::GC_69*UFO::MT^3/UFO::MH^2")
        } else {
            parse!("-2*UFO::GC_11^2*UFO::GC_94^2*UFO::MT^4")
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
            let inputs = case["inputs_mt_mh_muR_wh_reimGC5_reimGC37_reimGC30"]
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
            cases.push(json!({"state":case["state"],"checks":case["numerator_samples"].as_array().unwrap().len(),"maximum_scaled_error":max}));
        }
        result.push(json!({"feynkit_orientation":orientation,"directed_physical_cycle":directed,"madloop_filter_id":id,"momentum_rules":mapping.momentum_rules().iter().map(|(a,b)|[a.to_plain_string(),b.to_plain_string()]).collect::<Vec<_>>(),"denominator_map":mapping.denominator_map(),"exact_normalization_control":normalization.to_canonical_string(),"cases":cases}));
    }
    assert_eq!(
        total,
        summary["expected_raw_checks"].as_u64().unwrap() as usize
    );
    println!("{}",serde_json::to_string_pretty(&json!({"checks":total,"maximum_scaled_error":maximum,"pass":maximum<=1e-9,"graphs":result})).unwrap());
    if maximum > 1e-9 {
        std::process::exit(1);
    }
}
