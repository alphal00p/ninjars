//! Orientation of Dirac factors in explicit spinor chains.

use std::collections::BTreeMap;

use idenso::dirac::AGS;
use ninjars_graph::{FourMomentum, expand_components, symbols};
use spenso::network::tags::SPENSO_TAG;
use symbolica::atom::{Atom, FunctionBuilder, Symbol};

fn chain(left: Atom, right: Atom, factor: Atom) -> Atom {
    FunctionBuilder::new(SPENSO_TAG.chain)
        .add_arg(left)
        .add_arg(right)
        .add_arg(factor)
        .finish()
}

fn slashed(first: Symbol, second: Symbol) -> Atom {
    FunctionBuilder::new(AGS.gamma)
        .add_arg(Atom::var(first))
        .add_arg(Atom::var(second))
        .add_arg(symbols::external_momentum().call((0, Atom::num(4))))
        .finish()
}

#[test]
fn reversed_chain_markers_transpose_and_unknown_markers_are_rejected() {
    let x = symbols::ubar().call((0, Atom::num(4)));
    let y = symbols::u().call((1, Atom::num(4)));
    let vectors = BTreeMap::from([(
        symbols::external_momentum().call(0),
        FourMomentum::from([3, 1, 2, -1].map(Atom::num)),
    )]);
    let spinors = BTreeMap::from([
        (symbols::ubar().call(0), [1, 2, -1, 3].map(Atom::num)),
        (symbols::u().call(1), [2, 0, 1, -2].map(Atom::num)),
    ]);
    let (input, output) = (SPENSO_TAG.chain_in, SPENSO_TAG.chain_out);
    let expand = |expression: &Atom| expand_components(expression, &vectors, &spinors).ok();
    // x (M^T) y = y M x.
    let transposed = expand(&chain(x.clone(), y.clone(), slashed(output, input))).unwrap();
    let swapped = expand(&chain(y.clone(), x.clone(), slashed(input, output))).unwrap();
    let direct = expand(&chain(x.clone(), y.clone(), slashed(input, output))).unwrap();
    assert_eq!(transposed, swapped);
    assert_ne!(transposed, direct);
    assert!(expand(&chain(x, y, slashed(input, input))).is_none());
}
