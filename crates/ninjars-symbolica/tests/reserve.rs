//! `ninjars_symbolica::reserve`: the variable-slot regime, every master
//! integral's symbol registered by name (the table `src/master_symbols.txt`
//! complete and its symbols plain), and a loud failure when the process
//! registered too many symbols before.

use ninjars::{DoubleFloat, MasterIntegral};
use ninjars_symbolica::{PositionalError, Regime, SymbolicaMasters, master_symbols, reserve};
use std::collections::BTreeSet;
use symbolica::{
    atom::{Atom, UserData},
    domains::float::Float,
    state::State,
};

const CHILD: &str = "NINJARS_RESERVE_CHILD";

/// Every master-integral family; the match fails to compile when NinjaRS gains
/// one, so that it is added here (and to `masters::exact_families`).
fn families() -> Vec<MasterIntegral> {
    let listed = |integral: MasterIntegral| match integral {
        MasterIntegral::A0
        | MasterIntegral::B0
        | MasterIntegral::DB0
        | MasterIntegral::C0
        | MasterIntegral::D0
        | MasterIntegral::B1
        | MasterIntegral::B11
        | MasterIntegral::B111
        | MasterIntegral::B1111
        | MasterIntegral::An { .. } => integral,
    };
    [
        MasterIntegral::A0,
        MasterIntegral::B0,
        MasterIntegral::DB0,
        MasterIntegral::C0,
        MasterIntegral::D0,
        MasterIntegral::B1,
        MasterIntegral::B11,
        MasterIntegral::B111,
        MasterIntegral::B1111,
    ]
    .into_iter()
    .chain((0..=4).map(|rank| MasterIntegral::An { rank }))
    .map(listed)
    .collect()
}

fn names() -> BTreeSet<String> {
    State::symbol_iter()
        .map(|(_, name)| name.to_owned())
        .collect()
}

/// After `reserve`, the pool holds variable slots and building and compiling
/// every master family in double-double and arbitrary precision registers no
/// further symbol: their IDs are fixed by `reserve`. Every listed symbol is
/// still a plain one afterwards (oneloopmaster registered the same symbol).
/// A failure names what to add to `src/master_symbols.txt`.
#[test]
fn reserve_registers_every_master_symbol() {
    let before_reserve = names();
    assert!(matches!(reserve(), Ok(Regime::Variables)));
    let listed: Vec<&str> = master_symbols().collect();
    let reserved: Vec<String> = names().difference(&before_reserve).cloned().collect();
    for name in &listed {
        assert!(
            reserved.iter().any(|r| r == name) || before_reserve.contains(*name),
            "{name}"
        );
    }
    assert_eq!(ninjars_positional::regime(), Regime::Variables);
    let slot = ninjars_positional::slots(1).unwrap();
    assert!(matches!(slot[0], Atom::Var(_)), "{slot:?}");
    let before = names();
    let mut double = SymbolicaMasters::new(DoubleFloat::from(0.));
    let mut float = SymbolicaMasters::new(Float::with_val(190, 0));
    for integral in families() {
        double
            .prepare(integral, 1)
            .unwrap_or_else(|error| panic!("{integral:?} (double-double): {error}"));
        float
            .prepare(integral, 1)
            .unwrap_or_else(|error| panic!("{integral:?} (arbitrary precision): {error}"));
    }
    let new: Vec<_> = names().difference(&before).cloned().collect();
    assert!(
        new.is_empty(),
        "symbols registered after reserve(): {new:?}"
    );
    for (symbol, name) in State::symbol_iter() {
        if listed.contains(&name) {
            let plain = symbol.get_attributes().is_empty()
                && symbol.get_tags().is_empty()
                && symbol.get_normalization_function().is_none()
                && symbol.get_evaluation_info().is_none()
                && symbol.get_derivative_function().is_none()
                && symbol.get_print_function().is_none()
                && matches!(symbol.get_data(), UserData::None);
            assert!(plain, "{name} is not a plain symbol");
        }
    }
}

#[test]
#[ignore = "child process of reserve_fails_loudly_after_too_many_symbols"]
fn reserve_child() {
    if std::env::var_os(CHILD).is_some() {
        for i in 0..70_000 {
            let _ = symbolica::symbol!(format!("ninjars_reserve_foreign::s{i}"));
        }
        match reserve() {
            Err(PositionalError::FunctionSlots { registered }) if registered >= 70_000 => {
                println!("FUNCTION-SLOTS {registered}")
            }
            other => println!("UNEXPECTED {other:?}"),
        }
    }
}

/// `reserve` after 70,000 foreign symbols returns the typed error instead of
/// silently taking function slots (in a child process, which owns its symbol
/// registry).
#[test]
fn reserve_fails_loudly_after_too_many_symbols() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "reserve_child", "--ignored", "--nocapture"])
        .env(CHILD, "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("FUNCTION-SLOTS "), "{stdout}");
}
