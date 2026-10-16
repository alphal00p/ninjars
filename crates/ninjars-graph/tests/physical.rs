use std::collections::BTreeMap;

use ninjars_graph::{
    FourMomentum, Kinematics, Model, PhysicalPreparationError, prepare_top_higgs_gluons, symbols,
};
use symbolica::{
    atom::{Atom, AtomCore},
    parse,
};

#[test]
fn two_top_orientations_have_four_dimensional_color_projected_numerators() {
    let prepared = prepare_top_higgs_gluons(Model::standard_model(), &Kinematics::new()).unwrap();
    assert_eq!(prepared.len(), 2);
    let loop_components = FourMomentum::from_args(
        parse!("ninjars::q0"),
        parse!("ninjars::q1"),
        parse!("ninjars::q2"),
        parse!("ninjars::q3"),
    );
    for graph in &prepared {
        assert_eq!(graph.overall_factor, Atom::num(-1));
        assert_eq!(graph.family.denominators().len(), 3);
        assert_eq!(graph.squared_masses(), vec![parse!("UFO::MT^2"); 3]);
        let mut components = BTreeMap::new();
        for (position, edge) in graph
            .diagram
            .loop_momentum_basis()
            .external_edges
            .iter()
            .enumerate()
        {
            let data = graph
                .diagram
                .edges()
                .find(|(id, _, _)| id == edge)
                .unwrap()
                .2;
            let leg = data.external.as_ref().unwrap().index;
            let momentum = match leg {
                0 => FourMomentum::from_args(Atom::num(125), Atom::Zero, Atom::Zero, Atom::Zero),
                1 => FourMomentum::from_args(
                    parse!("125/2"),
                    Atom::Zero,
                    Atom::Zero,
                    parse!("125/2"),
                ),
                2 => FourMomentum::from_args(
                    parse!("125/2"),
                    Atom::Zero,
                    Atom::Zero,
                    parse!("-125/2"),
                ),
                _ => panic!("unexpected external leg"),
            };
            components.insert(symbols::external_momentum().call(position), momentum);
            if leg != 0 {
                components.insert(
                    symbols::epsilonbar().call(edge.0),
                    FourMomentum::from_args(Atom::Zero, Atom::one(), Atom::Zero, Atom::Zero),
                );
            }
        }
        let numerator = graph
            .component_numerator(&loop_components, &components)
            .unwrap();
        assert!(!numerator.is_zero());
        assert!(
            !numerator.to_plain_string().contains("spenso::"),
            "{numerator}"
        );
        assert!(
            !numerator.to_plain_string().contains("gammalooprs::"),
            "{numerator}"
        );
        assert!(
            numerator
                .replace(parse!("UFO::MT"))
                .with(Atom::Zero)
                .expand()
                .is_zero(),
            "the chirality trace must vanish at zero pole mass with Yukawa fixed"
        );
    }
}

#[test]
fn physical_generation_rejects_incompatible_dimensions_and_models() {
    assert!(matches!(
        prepare_top_higgs_gluons(
            Model::standard_model(),
            &Kinematics::in_dimension(&parse!("D")).unwrap()
        ),
        Err(PhysicalPreparationError::Dimension)
    ));
    assert!(matches!(
        prepare_top_higgs_gluons(Model::phi3(), &Kinematics::new()),
        Err(PhysicalPreparationError::Model(_))
    ));
}
