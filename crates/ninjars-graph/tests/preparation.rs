use ninjars_graph::{
    FourMomentum, Kinematics, Model, NumeratorMode, PreparationError, prepare_dot,
};
use std::collections::BTreeMap;
use symbolica::{
    atom::{Atom, AtomCore},
    parse,
};

#[test]
fn compact_dot_prefactor_is_included_exactly_once() {
    let prepared = prepare_dot(
        Model::phi3(),
        include_str!("../../../examples/graphs/bubble.dot"),
        &Kinematics::new(),
        NumeratorMode::Explicit,
    )
    .unwrap();
    assert_eq!(prepared.family.denominators().len(), 2);
    assert_eq!(prepared.quadratics.len(), 2);
    assert_eq!(prepared.overall_factor, Atom::one());
    let expected = parse!("1+ninjars::q0+2*ninjars::q1^2+ninjars::mu2");
    assert!((prepared.numerator - expected).expand().is_zero());
    assert!(prepared.propagator_edges.windows(2).all(|w| w[0] < w[1]));
}

#[test]
fn triangle_keeps_physical_propagators_only() {
    let prepared = prepare_dot(
        Model::phi3(),
        include_str!("../../../examples/graphs/triangle.dot"),
        &Kinematics::new(),
        NumeratorMode::Explicit,
    )
    .unwrap();
    assert_eq!(prepared.family.denominators().len(), 3);
    assert_eq!(prepared.quadratics.len(), 3);
    assert_eq!(prepared.family.loop_momenta().len(), 1);
}

#[test]
fn rejects_tree_before_reduction() {
    let result = prepare_dot(
        Model::phi3(),
        r#"digraph {
        ext [style=invis]; ext -> a [particle="phi"];
        a -> ext [particle="phi"]; a -> ext [particle="phi"];
    }"#,
        &Kinematics::new(),
        NumeratorMode::Explicit,
    );
    assert!(matches!(result, Err(PreparationError::LoopCount(0))));
}

#[test]
fn exact_components_reuse_minkowski_products_and_routing() {
    let graph = include_str!("../../../examples/graphs/bubble.dot");
    let mut prepared = prepare_dot(
        Model::phi3(),
        graph,
        &Kinematics::new(),
        NumeratorMode::Explicit,
    )
    .unwrap();
    let q = prepared.family.loop_momenta()[0].clone();
    let p = prepared.family.external_momenta()[0].clone();
    prepared.numerator = prepared.family.kinematics().scalar_product(&q, &p).unwrap();
    let q_components = FourMomentum::from_args(
        parse!("ninjars::q0"),
        parse!("ninjars::q1"),
        parse!("ninjars::q2"),
        parse!("ninjars::q3"),
    );
    let p_components =
        FourMomentum::from_args(Atom::num(5), Atom::num(3), Atom::num(2), Atom::num(1));
    let external = BTreeMap::from([(p, p_components)]);
    let component = prepared
        .component_numerator(&q_components, &external)
        .unwrap();
    assert!(
        (component - parse!("5*ninjars::q0-3*ninjars::q1-2*ninjars::q2-ninjars::q3"))
            .expand()
            .is_zero()
    );
    assert_eq!(prepared.shifts(&external).unwrap().len(), 2);
    assert_eq!(prepared.squared_masses().len(), 2);
}

#[test]
fn loop_independent_bridge_moves_into_overall_factor() {
    let prepared = prepare_dot(
        Model::phi3(),
        r#"digraph {
            overall_factor="7"; num="5";
            ext [style=invis]; ext -> a [particle="phi"];
            a -> b [particle="phi", lmb_id=0]; b -> a [particle="phi"];
            b -> c [particle="phi"];
            c -> ext [particle="phi"]; c -> ext [particle="phi"];
        }"#,
        &Kinematics::new(),
        NumeratorMode::Explicit,
    )
    .unwrap();
    let full = prepared
        .diagram
        .propagator_family(&Kinematics::new())
        .unwrap();
    assert_eq!(full.denominators().len(), 3);
    assert_eq!(prepared.family.denominators().len(), 2);
    assert_eq!(prepared.numerator, Atom::num(5));
    let bridge = (0..full.denominators().len())
        .find(|index| full.quadratic_denominator(*index).unwrap().is_none())
        .unwrap();
    assert!(
        (&prepared.overall_factor * &full.denominators()[bridge] - Atom::num(7))
            .together()
            .is_zero()
    );
}

#[test]
fn dot_preparation_applies_generator_graph_factors_to_hand_written_graphs() {
    // A gluon bubble: 1/AutG = 1/2. Every DOT
    // preparation path, including the physical tensor numerators of
    // numerator-bench and pipeline-bench, goes through `prepare_dot_timed`.
    let dot = include_str!("../../ninjars-cli/tests/graphs/gluon-bubble.dot");
    let rules = prepare_dot(
        Model::yang_mills(),
        dot,
        &Kinematics::new(),
        NumeratorMode::FeynmanRules,
    )
    .unwrap();
    assert_eq!(rules.graph_factors.automorphisms, 2);
    assert!(rules.graph_factors_applied);
    assert_eq!(rules.overall_factor, parse!("1/2"));
    // A stated overall_factor is kept as written.
    let stated = dot.replace(
        "ext [style=invis];",
        "overall_factor=\"3\"; ext [style=invis];",
    );
    let stated = prepare_dot(
        Model::yang_mills(),
        &stated,
        &Kinematics::new(),
        NumeratorMode::FeynmanRules,
    )
    .unwrap();
    assert!(!stated.graph_factors_applied);
    assert_eq!(stated.overall_factor, parse!("3"));
    // A diagram prepared directly only reports its factors.
    let diagram = ninjars_graph::FeynmanDiagram::from_dot(Model::yang_mills(), dot).unwrap();
    let direct =
        ninjars_graph::prepare_diagram(diagram, &Kinematics::new(), NumeratorMode::FeynmanRules)
            .unwrap();
    assert_eq!(direct.graph_factors, rules.graph_factors);
    assert!(!direct.graph_factors_applied);
    assert_eq!(direct.overall_factor, parse!("1"));
}

#[test]
fn dependent_external_momentum_routes_like_its_edge() {
    // In this box, edge 1 is the dependent external: P(1) = -P(0) + P(2) + P(3).
    let graph = |head: &str, label: &str| {
        let vector = |head: &str, label: &str, i: usize| {
            format!(
                "gammalooprs::{{spenso::rank1,spenso::tensor,spenso::component-labels:subscript,\
                 spenso::tensor-label:{label}}}::{head}({i},spenso::{{spenso::representation,\
                 spenso::self_dual}}::mink(4,gammalooprs::{{spenso::index}}::hedge(202,1)))"
            )
        };
        format!(
            r#"digraph box_effective {{
    ext [style=invis];
    ext -> a [id=0, particle="phi", num="1"];
    ext -> b [id=1, particle="phi", num="1"];
    c -> ext [id=2, particle="phi", num="1"];
    d -> ext [id=3, particle="phi", num="1"];
    a -> b [id=4, particle="phi", lmb_id=0, num="1"];
    b -> c [id=5, particle="phi", num="1"];
    c -> d [id=6, particle="phi", num="1"];
    d -> a [id=7, particle="phi", num="1"];
    b [num="{}*{}"];
}}"#,
            vector("P", "p", 0),
            vector(head, label, 1)
        )
    };
    let prepare = |dot: &str| {
        prepare_dot(
            Model::phi3(),
            dot,
            &Kinematics::new(),
            NumeratorMode::Explicit,
        )
        .unwrap()
    };
    let external = prepare(&graph("P", "p"));
    let edge = prepare(&graph("Q", "q"));
    let basis = external.diagram.loop_momentum_basis();
    assert_eq!(basis.dependent_externals, vec![basis.external_edges[1]]);
    assert!((&external.numerator - &edge.numerator).expand().is_zero());
    assert!(!external.numerator.to_canonical_string().contains("::P(1,"));
}
