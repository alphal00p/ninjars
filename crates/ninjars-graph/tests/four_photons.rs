use std::collections::{BTreeMap, BTreeSet};

use feynkit_graph::ExternalState;
use ninjars_graph::{
    FourMomentum, Kinematics, Model, PhysicalPreparationError, prepare_top_four_photons, symbols,
};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::atom::AtomField,
    parse,
};

#[test]
fn six_oriented_boxes_keep_exact_rules_color_and_rank_four() {
    let prepared = prepare_top_four_photons(Model::standard_model(), &Kinematics::new()).unwrap();
    assert_eq!(prepared.len(), 6);
    let variables = ["q0", "q1", "q2", "q3"].map(|name| {
        Atom::parse(format!("ninjars::{name}"), "ninjars", Default::default()).unwrap()
    });
    let q = FourMomentum::from_args(
        variables[0].clone(),
        variables[1].clone(),
        variables[2].clone(),
        variables[3].clone(),
    );
    let zero = FourMomentum::from_args(Atom::Zero, Atom::Zero, Atom::Zero, Atom::Zero);
    let temporal = FourMomentum::from_args(Atom::one(), Atom::Zero, Atom::Zero, Atom::Zero);
    let mut cycles = BTreeSet::new();
    for graph in prepared {
        assert_eq!(graph.overall_factor, Atom::num(-1));
        assert_eq!(graph.family.denominators().len(), 4);
        assert_eq!(graph.squared_masses(), vec![parse!("UFO::MT^2"); 4]);
        let mut components = BTreeMap::new();
        let mut external_vertices = BTreeMap::new();
        let mut successors = BTreeMap::new();
        for (edge, endpoints, data) in graph.diagram.edges() {
            let particle = graph.diagram.model().particle_by_id(data.particle).unwrap();
            if let Some(external) = &data.external {
                assert_eq!(particle.pdg_code, 22);
                external_vertices.insert(
                    endpoints.source.or(endpoints.target).unwrap(),
                    external.index,
                );
                let carrier = match external.state {
                    ExternalState::Incoming => symbols::epsilon(),
                    ExternalState::Outgoing => symbols::epsilonbar(),
                };
                components.insert(carrier.call(edge.0), temporal.clone());
            } else {
                assert_eq!(particle.pdg_code, 6);
                successors.insert(endpoints.source.unwrap(), endpoints.target.unwrap());
            }
        }
        let start = *external_vertices
            .iter()
            .find(|(_, leg)| **leg == 0)
            .unwrap()
            .0;
        let mut vertex = start;
        let mut cycle = Vec::new();
        for _ in 0..4 {
            cycle.push(external_vertices[&vertex]);
            vertex = successors[&vertex];
        }
        assert_eq!(vertex, start);
        assert_eq!(
            cycle.iter().copied().collect::<BTreeSet<_>>(),
            BTreeSet::from([0, 1, 2, 3])
        );
        assert!(cycles.insert(cycle));
        for momentum in graph.family.external_momenta() {
            components.insert(momentum.clone(), zero.clone());
        }
        // An off-shell temporal-polarization control avoids accidental helicity
        // zeros. At q=p=0 the trace of gamma0^4 is four; the closed color trace
        // contributes three, and the graph has exactly one fermion-loop sign.
        let constant =
            graph.component_numerator(&zero, &components).unwrap() * &graph.overall_factor;
        assert!(
            (constant - parse!("-12*UFO::GC_2^4*UFO::MT^4"))
                .expand()
                .is_zero()
        );
        let numerator = graph.component_numerator(&q, &components).unwrap();
        assert!(!numerator.contains(parse!("ninjars::mu2")));
        let polynomial = numerator.to_polynomial_in_vars_with_field::<u16>(
            &variables,
            &AtomField {
                statistical_zero_test: false,
                ..AtomField::new()
            },
        );
        assert_eq!(
            (&polynomial)
                .into_iter()
                .map(|term| term.exponents.iter().sum::<u16>())
                .max(),
            Some(4)
        );
        assert!((&polynomial).into_iter().all(|term| {
            variables
                .iter()
                .all(|variable| !term.coefficient.contains(variable))
        }));
    }
    assert_eq!(cycles.len(), 6);
}

#[test]
fn four_photon_generation_rejects_non_four_dimensions_and_missing_top_vertex() {
    assert!(matches!(
        prepare_top_four_photons(
            Model::standard_model(),
            &Kinematics::in_dimension(&parse!("D")).unwrap()
        ),
        Err(PhysicalPreparationError::Dimension)
    ));
    assert!(matches!(
        prepare_top_four_photons(Model::phi3(), &Kinematics::new()),
        Err(PhysicalPreparationError::Model(_))
    ));
}
