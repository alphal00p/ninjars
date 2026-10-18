//! Coefficients of a prepared numerator template (export-bench, the SymJIT
//! numerator evaluators of `reduce` and `bench`, the fast runtime) must not
//! depend on what the process prepared before. Symbolica sorts terms, breaks
//! ties in its Horner schemes and orders polynomial variables (hence exact
//! factors) by the symbols' encoded IDs, which are registration indices.
//!
//! The reference is a fresh child process that prepares only the probe
//! fixture. The parent first registers a few hundred unrelated symbols and
//! prepares other fixtures, then the probe in this thread and concurrently in
//! two threads, and every result must be bitwise equal to the child's.

use std::path::PathBuf;

use ninjars_graph::PolynomialCoefficientPlan;
use symbolica::{
    atom::Atom,
    domains::float::{Complex, Float},
    parser::ParseSettings,
    symbol,
};

use crate::{prepare, runtime::numerical};

const CHILD: &str = "NINJARS_CLI_DETERMINISM_CHILD";

/// Bits of the coefficient heads of a pool fixture's template at its exact
/// samples, in Float(256) as export-bench evaluates them and in binary64,
/// optionally after exact factoring (`--factor`, `symjit-o2-factored`).
fn bits(fixture: &str, factor: bool) -> Vec<String> {
    let input = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(fixture);
    let prepared =
        prepare::prepare(&prepare::Args::for_input(input, None, Default::default())).unwrap();
    let variables =
        ["q0", "q1", "q2", "q3", "mu2"].map(|name| Atom::var(symbol!(format!("ninjars::{name}"))));
    let plan = PolynomialCoefficientPlan::new(
        prepared.numerator_template.clone(),
        variables,
        prepared.parameters.clone(),
    )
    .unwrap();
    let plan = if factor {
        plan.factor_coefficients().unwrap()
    } else {
        plan
    };
    let rows = prepared.samples.len();
    let heads = plan.coefficient_expressions().len();
    let parameters = || {
        prepared
            .samples
            .iter()
            .flat_map(|sample| &sample.parameters)
    };

    let high = Float::new(256);
    let input: Vec<_> = parameters()
        .map(|atom| numerical(atom, &high, "parameter").unwrap())
        .collect();
    let mut output = vec![Complex::new(high.clone(), high.clone()); rows * heads];
    plan.compile(high.clone())
        .unwrap()
        .evaluate_batch(&input, &mut output, rows)
        .unwrap();
    let mut bits: Vec<_> = output
        .iter()
        .map(|z| format!("{}|{}", z.re.to_rational(), z.im.to_rational()))
        .collect();

    let input: Vec<_> = parameters()
        .map(|atom| numerical(atom, &0_f64, "parameter").unwrap())
        .collect();
    let mut output = vec![Complex::new(0., 0.); rows * heads];
    plan.compile(0_f64)
        .unwrap()
        .evaluate_batch(&input, &mut output, rows)
        .unwrap();
    bits.extend(
        output
            .iter()
            .map(|z| format!("{:x}|{:x}", z.re.to_bits(), z.im.to_bits())),
    );
    bits
}

/// The probe: a top box of gg -> hh, plain and exactly factored.
fn probe() -> Vec<String> {
    let fixture = "examples/graphs/physical/gg-hh/top-box-2-pool.json";
    let mut values = bits(fixture, false);
    values.extend(bits(fixture, true));
    values
}

/// Unrelated preparations: other graphs and processes first.
fn unrelated() {
    bits(
        "examples/graphs/physical/hgg/top-triangle-0-pool.json",
        true,
    );
    bits(
        "examples/graphs/physical/four-photon/top-box-0-pool.json",
        false,
    );
}

#[test]
#[ignore = "child process of templates_do_not_depend_on_process_history"]
fn determinism_child() {
    if std::env::var_os(CHILD).is_some() {
        println!("BITS {}", probe().join(";"));
    }
}

#[test]
fn templates_do_not_depend_on_process_history() {
    for i in 0..300 {
        drop(Atom::parse(
            format!("filler{i}"),
            "ninjars_determinism",
            ParseSettings::default(),
        ));
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "determinism_tests::determinism_child",
            "--ignored",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let reference: Vec<String> = stdout
        .lines()
        .find_map(|line| line.strip_prefix("BITS "))
        .unwrap_or_else(|| {
            panic!(
                "child failed: {stdout}{}",
                String::from_utf8_lossy(&output.stderr)
            )
        })
        .split(';')
        .map(str::to_owned)
        .collect();
    unrelated();
    let first = probe();
    let changed = first.iter().zip(&reference).filter(|(a, b)| a != b).count();
    assert!(
        first == reference,
        "after unrelated preparations, {changed} of {} values differ",
        reference.len()
    );
    let threads: Vec<_> = (0..2)
        .map(|thread| {
            std::thread::spawn(move || {
                if thread == 1 {
                    unrelated();
                }
                probe()
            })
        })
        .collect();
    for thread in threads {
        assert!(thread.join().unwrap() == reference, "concurrent threads");
    }
}
