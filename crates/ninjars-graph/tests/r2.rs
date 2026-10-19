//! FDH binding (`R2Scheme::Fdh`): the bound numerator is the four-dimensional
//! (`Off`) numerator with `K.K -> K.K - mu^2`, nothing else changes, and
//! numerators the binding cannot treat exactly are rejected.

use std::collections::BTreeMap;

use feynkit_graph::ExternalState;
use idenso::{dirac::AGS, epsilon::EPSILON_SYMBOL};
use ninjars_graph::{
    FourMomentum, Kinematics, Model, NumeratorMode, PolynomialCoefficientPlan, PreparationError,
    PreparedIntegrand, R2Scheme, TensorCoefficientPlan, prepare_dot, prepare_gluon_self_energy,
    prepare_top_four_photons, prepare_top_higgs_gluons, symbols,
};
use spenso::{
    network::library::symbolic::ETS,
    structure::representation::{LibraryRep, Minkowski},
};
use symbolica::{
    atom::{Atom, AtomCore},
    function,
    id::{Pattern, Replacement},
    symbol,
};

fn variable(name: &str) -> Atom {
    Atom::parse(format!("ninjars::{name}"), "ninjars", Default::default()).unwrap()
}

/// Symbolic components for every external momentum and vector carrier.
fn external_components(graph: &PreparedIntegrand) -> BTreeMap<Atom, FourMomentum<Atom>> {
    let mut carriers: Vec<Atom> = graph.family.external_momenta().to_vec();
    for (edge, _, data) in graph.diagram.edges() {
        let Some(external) = &data.external else {
            continue;
        };
        if graph
            .diagram
            .model()
            .particle_by_id(data.particle)
            .unwrap()
            .spin
            == 3
        {
            carriers.push(match external.state {
                ExternalState::Incoming => symbols::epsilon().call(edge.0),
                ExternalState::Outgoing => symbols::epsilonbar().call(edge.0),
            });
        }
    }
    carriers
        .into_iter()
        .enumerate()
        .map(|(index, carrier)| {
            let [e, x, y, z] =
                std::array::from_fn(|mu| Atom::var(symbol!(format!("r2_test::c{index}_{mu}"))));
            (carrier, FourMomentum::from_args(e, x, y, z))
        })
        .collect()
}

fn loop_components() -> FourMomentum<Atom> {
    FourMomentum::from_args(
        variable("q0"),
        variable("q1"),
        variable("q2"),
        variable("q3"),
    )
}

/// Returns whether the graph's FDH numerator has `mu^2` terms.
fn check_kk_substitution(label: &str, graph: &PreparedIntegrand) -> bool {
    let mu2 = variable("mu2");
    let external = external_components(graph);
    let q = loop_components();
    let off = graph.component_numerator(&q, &external).unwrap();
    let fdh = graph
        .dimensional_numerator(&q, R2Scheme::Fdh.mu_squared(&mu2), &external)
        .unwrap();
    assert_eq!(
        graph
            .dimensional_numerator(&q, R2Scheme::Off.mu_squared(&mu2), &external)
            .unwrap(),
        off,
        "{label}: Off is the four-dimensional binding"
    );
    // Independently: K.K -> KK - mu^2 for a fresh KK, then the 4D binding,
    // then KK -> q.q (four-dimensional).
    let k = &graph.family.loop_momenta()[0];
    let kk = Atom::var(symbol!("r2_test::kk"));
    let shifted = Kinematics::new()
        .with_scalar_product(k, k, &kk - &mu2)
        .unwrap()
        .apply(&graph.numerator);
    let reference = graph
        .component_kinematics(&q, &external)
        .unwrap()
        .apply(&shifted)
        .replace_multiple(&[Replacement::new(
            Pattern::Literal(kk),
            Pattern::Literal(q.dot(&q)),
        )])
        .expand();
    assert!(
        (&fdh - &reference).expand().is_zero(),
        "{label}: FDH binding differs from K.K -> K.K - mu^2"
    );
    // mu^2 -> 0 recovers the four-dimensional numerator (up to any explicit
    // mu^2 terms of the graph's own numerator).
    let zero = |expression: &Atom| {
        expression.replace_multiple(&[Replacement::new(
            Pattern::Literal(mu2.clone()),
            Pattern::Literal(Atom::Zero),
        )])
    };
    assert!((zero(&fdh) - zero(&off)).expand().is_zero(), "{label}");
    // mu^2 has rank weight two: the rank is unchanged.
    let variables = ["q0", "q1", "q2", "q3", "mu2"].map(variable);
    let parameters: Vec<_> = external
        .values()
        .flat_map(|v| v.components().map(Clone::clone))
        .collect();
    let rank = |expression: &Atom| {
        PolynomialCoefficientPlan::new(expression.clone(), variables.clone(), parameters.clone())
            .unwrap()
            .rank()
    };
    assert_eq!(rank(&fdh), rank(&off), "{label}");
    fdh.contains(&mu2)
}

#[test]
fn fdh_binding_is_the_four_dimensional_numerator_with_kk_minus_mu2() {
    let kinematics = Kinematics::new();
    let mut with_mu = 0;
    for graph in prepare_top_higgs_gluons(Model::standard_model(), &kinematics).unwrap() {
        assert!(check_kk_substitution("hgg", &graph));
        with_mu += 1;
    }
    for graph in prepare_top_four_photons(Model::standard_model(), &kinematics).unwrap() {
        assert!(check_kk_substitution("four photons", &graph));
        with_mu += 1;
    }
    let mut self_energies = Vec::new();
    for graph in prepare_gluon_self_energy(Model::standard_model(), &kinematics, &[1, 6]).unwrap() {
        self_energies.push(check_kk_substitution("gluon self-energy", &graph));
    }
    // The ghost loop has no K.K; the gluon and quark loops do.
    assert_eq!(self_energies.iter().filter(|mu| **mu).count(), 3);
    assert_eq!(with_mu, 8);
    // Explicit component numerators keep their own mu^2 and are unchanged.
    let bubble = prepare_dot(
        Model::phi3(),
        include_str!("../../../examples/graphs/bubble.dot"),
        &kinematics,
        NumeratorMode::Explicit,
    )
    .unwrap();
    check_kk_substitution("explicit bubble", &bubble);
}

#[test]
fn fdh_tensor_layout_adds_mu_blocks_without_changing_the_mu_free_layout() {
    let graph = prepare_top_higgs_gluons(Model::standard_model(), &Kinematics::new())
        .unwrap()
        .remove(0);
    let mu2 = variable("mu2");
    let external = external_components(&graph);
    let parameters: Vec<_> = external
        .values()
        .flat_map(|v| v.components().map(Clone::clone))
        .collect();
    let variables = ["q0", "q1", "q2", "q3", "mu2"].map(variable);
    let q = loop_components();
    let plans = [None, Some(&mu2)].map(|mu| {
        let numerator = graph.dimensional_numerator(&q, mu, &external).unwrap();
        PolynomialCoefficientPlan::new(numerator, variables.clone(), parameters.clone()).unwrap()
    });
    let off = TensorCoefficientPlan::new(&plans[0]).unwrap();
    let off_mu = TensorCoefficientPlan::with_mu_squared(&plans[0], None).unwrap();
    assert_eq!(off_mu.mu_powers(), 0);
    assert_eq!(off_mu.entries(), off.entries());
    assert_eq!(off_mu.zero_slots(), off.zero_slots());
    assert_eq!(off_mu.ncoeff(), off.ncoeff());
    let fdh = TensorCoefficientPlan::with_mu_squared(&plans[1], None).unwrap();
    assert_eq!((fdh.rank(), fdh.mu_powers()), (2, 1));
    assert_eq!(fdh.ncoeff(), 15 + 1);
    assert!(
        fdh.entries()
            .iter()
            .any(|entry| entry.mu_power == 1 && entry.slot == 15)
    );
    assert!(TensorCoefficientPlan::new(&plans[1]).is_err());
}

fn fdh_error(graph: &PreparedIntegrand) -> String {
    match graph.check_fdh() {
        Err(PreparationError::UnsupportedFdh(reason)) => reason.to_string(),
        other => panic!("expected an FDH rejection, got {other:?}"),
    }
}

#[test]
fn fdh_rejects_four_dimensional_only_structures() {
    let graph = prepare_top_higgs_gluons(Model::standard_model(), &Kinematics::new())
        .unwrap()
        .remove(0);
    graph.check_fdh().unwrap();
    let with_projector = |extra: Atom| {
        let mut modified = graph.clone();
        modified.diagram = modified
            .diagram
            .clone()
            .with_projector(
                feynkit_graph::NumeratorInterface::admit(feynkit_graph::NumeratorInterface::export(modified.diagram.projector()).0 * extra, None)
                    .unwrap(),
            );
        modified
    };
    let minkowski = LibraryRep::from(Minkowski {}).symbol();
    let mink = |index: i64| function!(minkowski, Atom::num(4), Atom::num(index));
    let bispinor = LibraryRep::from(idenso::representations::Bispinor {}).symbol();
    let bis = |index: i64| function!(bispinor, Atom::num(4), Atom::num(index));
    for (extra, reason) in [
        (function!(AGS.gamma5, bis(1), bis(2)), "gamma5"),
        (function!(AGS.projm, bis(1), bis(2)), "gamma5"),
        (
            function!(*EPSILON_SYMBOL, mink(1), mink(2), mink(3), mink(4)),
            "Levi-Civita",
        ),
        (function!(symbols::ubar(), Atom::num(7), bis(1)), "spinor"),
        (function!(ETS.metric, mink(11), mink(12)), "metric"),
    ] {
        let modified = with_projector(extra);
        assert!(fdh_error(&modified).contains(reason), "{reason}");
        let mu2 = variable("mu2");
        assert!(matches!(
            modified.dimensional_kinematics(&loop_components(), Some(&mu2), &BTreeMap::new()),
            Err(PreparationError::UnsupportedFdh(_))
        ));
    }
    // Color metrics in the projector are fine (the gluon delta_ab / 8).
    assert!(graph.diagram.projector().to_string().contains("coad"));
}
