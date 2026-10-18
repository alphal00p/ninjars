//! Physical numerator coefficients must not depend on what the process
//! prepared before: Symbolica sorts terms and breaks ties in its Horner
//! schemes by their encoded bytes, which contain symbol IDs (registration
//! indices), and orders polynomial variables, hence exact factors, by symbol
//! too. Coefficient and mass evaluators built after unrelated preparations
//! could otherwise round differently.
//!
//! The reference is a fresh child process that prepares only the probe
//! graphs. The parent first registers a few hundred unrelated symbols (every
//! later symbol ID needs two bytes) and prepares other graphs (other
//! numerators, registering the model's symbols in another order), then the
//! probe in this thread and concurrently in two threads, and every result
//! must be bitwise equal to the child's.

use std::{collections::BTreeMap, fs, path::PathBuf, sync::Arc};

use ninjars_graph::{
    CoefficientBackend, Model, NumeratorMode, PhysicalNumeratorSpec, PhysicalTensorNumerator,
    R2Scheme,
};
use serde_json::Value;
use symbolica::{
    atom::Atom,
    domains::float::{Complex, Float},
    parser::ParseSettings,
};

const CHILD: &str = "NINJARS_GRAPH_DETERMINISM_CHILD";

const BACKEND: CoefficientBackend = if cfg!(feature = "numerator-jit") {
    CoefficientBackend::SymJitO2
} else {
    CoefficientBackend::Interpreter
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn json(path: &str) -> Value {
    serde_json::from_str(&fs::read_to_string(root().join(path)).unwrap()).unwrap()
}

/// Uniform on [-1, 1] rounded to 2^-12: exact in every carrier.
fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let x = ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.;
    (x * 4096.).round() / 4096.
}

/// Bits of a physical numerator's binary64 tensor coefficients and squared
/// masses at its benchmark points, and of its coefficient heads in
/// Float(256) (as `export-bench` evaluates them) at exact random inputs.
fn bits(process: &str, graph: &str, r2: R2Scheme, factor_coefficients: bool) -> Vec<String> {
    let directory = format!("examples/graphs/physical/{process}");
    let fixture = json(&format!("{directory}/{graph}-pool.json"));
    let pool = json(&format!("validation/benchmark/points/{process}.json"));
    let fixed_parameters: BTreeMap<_, _> = pool["fixed_parameters"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, value)| {
            let value =
                Atom::parse(value.as_str().unwrap(), "ninjars", ParseSettings::default()).unwrap();
            (name.clone(), value)
        })
        .collect();
    let spec = PhysicalNumeratorSpec {
        model: Arc::new(Model::standard_model()),
        dot: fs::read_to_string(root().join(format!("{directory}/{graph}.dot"))).unwrap(),
        numerator_mode: NumeratorMode::FeynmanRules,
        color_dimension_invariants: fixture["color_dimension_invariants"].as_bool().unwrap(),
        runtime_parameters: pool["model_parameters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|name| name.as_str().unwrap().to_string())
            .collect(),
        fixed_parameters,
        factor_coefficients,
        tensor_rank: None,
        backend: BACKEND,
        r2,
    };
    let mut numerator = PhysicalTensorNumerator::new(&spec).unwrap();

    let points = pool["points"].as_array().unwrap();
    let rows = points.len();
    let (mut momenta, mut helicities, mut model) = (Vec::new(), Vec::new(), Vec::new());
    for point in points {
        for p in point["momenta"].as_array().unwrap() {
            momenta.push(std::array::from_fn(|k| p[k].as_f64().unwrap()));
        }
        for h in point["helicities"].as_array().unwrap() {
            helicities.push(h.as_i64().unwrap() as i8);
        }
        for x in point["model"].as_array().unwrap() {
            model.push(x.as_f64().unwrap());
        }
    }
    let (n, ncoeff) = (numerator.propagators(), numerator.ncoeff());
    let mut tensor = vec![Complex::new(0., 0.); rows * ncoeff];
    numerator
        .prepare(&momenta, &helicities, &model, rows, &mut tensor)
        .unwrap();
    let mut shifts = vec![[0.; 4]; rows * n];
    let mut masses = vec![Complex::new(0., 0.); rows * n];
    numerator
        .kinematics(&momenta, &model, rows, &mut shifts, &mut masses, None, 1e-8)
        .unwrap();
    let mut bits: Vec<String> = tensor
        .iter()
        .chain(&masses)
        .map(|z| format!("{:x}|{:x}", z.re.to_bits(), z.im.to_bits()))
        .collect();

    let plan = numerator.coefficient_plan();
    let mut state = 17;
    let mut exact = || Float::with_val(256, uniform(&mut state));
    let input: Vec<_> = plan
        .parameters()
        .iter()
        .map(|_| Complex::new(exact(), exact()))
        .collect();
    let zero = Complex::new(Float::new(256), Float::new(256));
    let mut heads = vec![zero; plan.coefficient_expressions().len()];
    plan.compile(Float::new(256))
        .unwrap()
        .evaluate_batch(&input, &mut heads, 1)
        .unwrap();
    bits.extend(
        heads
            .iter()
            .map(|z| format!("{}|{}", z.re.to_rational(), z.im.to_rational())),
    );
    bits
}

/// The probe: the five-propagator rank-four top pentagon of gg -> hhh, with
/// the mu^2 terms of the four-dimensional-helicity binding, and a top box of
/// gg -> hh with exactly factored coefficient heads.
fn probe() -> Vec<String> {
    let mut values = bits("gg-hhh", "top-pentagon-26", R2Scheme::Fdh, false);
    values.extend(bits("gg-hh", "top-box-2", R2Scheme::Off, true));
    values
}

/// Unrelated preparations: other graphs and processes first.
fn unrelated() {
    bits("hgg", "top-triangle-0", R2Scheme::Off, true);
    bits("four-photon", "top-box-0", R2Scheme::Fdh, false);
}

#[test]
#[ignore = "child process of physical_numerators_do_not_depend_on_process_history"]
fn determinism_child() {
    if std::env::var_os(CHILD).is_some() {
        println!("BITS {}", probe().join(";"));
    }
}

#[test]
fn physical_numerators_do_not_depend_on_process_history() {
    for i in 0..300 {
        drop(Atom::parse(
            format!("filler{i}"),
            "ninjars_determinism",
            ParseSettings::default(),
        ));
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "determinism_child", "--ignored", "--nocapture"])
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
