//! Per-point physical tensor numerators against the exported bench cases,
//! FeynKit's wavefunctions, and the zero-allocation contract of warm calls.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::Arc,
};

use feynkit_kinematics::{Helicity, WavefunctionKind};
use ninjars_graph::{
    CoefficientBackend, FourMomentum, LegState, Model, NumeratorMode, PhysicalNumeratorError,
    PhysicalNumeratorSpec, PhysicalTensorNumerator, R2Scheme, vector_wavefunction,
};
use serde_json::Value;
use symbolica::{
    atom::Atom,
    domains::float::{Complex, Real},
    parser::ParseSettings,
};

/// Counts allocations of the current thread while `COUNTING` is set.
struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = COUNTING.try_with(|counting| {
            if counting.get() {
                ALLOCATIONS.with(|count| count.set(count.get() + 1));
            }
        });
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let _ = COUNTING.try_with(|counting| {
            if counting.get() {
                ALLOCATIONS.with(|count| count.set(count.get() + 1));
            }
        });
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocations(body: impl FnOnce()) -> usize {
    ALLOCATIONS.with(|count| count.set(0));
    COUNTING.with(|counting| counting.set(true));
    body();
    COUNTING.with(|counting| counting.set(false));
    ALLOCATIONS.with(Cell::get)
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn json(path: &str) -> Value {
    serde_json::from_str(&fs::read_to_string(root().join(path)).unwrap()).unwrap()
}

const BACKEND: CoefficientBackend = if cfg!(feature = "numerator-jit") {
    CoefficientBackend::SymJitO2
} else {
    CoefficientBackend::Interpreter
};

struct Case {
    numerator: PhysicalTensorNumerator,
    pool: Value,
    case: Value,
}

fn case(process: &str, graph: &str) -> Case {
    let directory = format!("examples/graphs/physical/{process}");
    let fixture = json(&format!("{directory}/{graph}-pool.json"));
    let pool = json(&format!("validation/benchmark/points/{process}.json"));
    let case = json(&format!(
        "validation/benchmark/cases/{process}-{graph}.json"
    ));
    assert_eq!(fixture["model"], "sm");
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
        factor_coefficients: false,
        tensor_rank: None,
        backend: BACKEND,
        r2: R2Scheme::Off,
    };
    Case {
        numerator: PhysicalTensorNumerator::new(&spec).unwrap(),
        pool,
        case,
    }
}

/// Row-major inputs for `width` points cycling through the pool.
fn inputs(pool: &Value, width: usize) -> (Vec<[f64; 4]>, Vec<i8>, Vec<f64>) {
    let points = pool["points"].as_array().unwrap();
    let mut momenta = Vec::new();
    let mut helicities = Vec::new();
    let mut model = Vec::new();
    for point in points.iter().cycle().take(width) {
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
    (momenta, helicities, model)
}

fn number(value: &Value) -> f64 {
    value.as_f64().unwrap()
}

fn check_against_case(
    Case {
        numerator,
        pool,
        case,
    }: &mut Case,
) {
    let rows = pool["points"].as_array().unwrap().len();
    let (n, ncoeff) = (numerator.propagators(), numerator.ncoeff());
    assert_eq!(case["n"].as_u64().unwrap() as usize, n);
    assert_eq!(case["rank"].as_u64().unwrap() as usize, numerator.rank());
    let (momenta, helicities, model) = inputs(pool, rows);
    let mut tensor = vec![Complex::new(0., 0.); rows * ncoeff];
    numerator
        .prepare(&momenta, &helicities, &model, rows, &mut tensor)
        .unwrap();
    let mut shifts = vec![[0.; 4]; rows * n];
    let mut masses = vec![Complex::new(0., 0.); rows * n];
    let mut s_matrix = vec![0.; rows * n * n];
    numerator
        .kinematics(
            &momenta,
            &model,
            rows,
            &mut shifts,
            &mut masses,
            Some(&mut s_matrix),
            1e-8,
        )
        .unwrap();
    for (row, lane) in case["lanes"].as_array().unwrap().iter().enumerate() {
        assert_eq!(lane["id"], pool["points"][row]["id"]);
        let expected: Vec<_> = lane["tensor"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| Complex::new(number(&c[0]), number(&c[1])))
            .collect();
        let largest = expected.iter().fold(0_f64, |m, c| m.max(c.norm().re));
        for (actual, expected) in tensor[row * ncoeff..(row + 1) * ncoeff]
            .iter()
            .zip(&expected)
        {
            assert!(
                (*actual - *expected).norm().re <= 1e-12 * largest,
                "row {row}: {actual} != {expected}"
            );
        }
        let energy = shifts[row * n..(row + 1) * n]
            .iter()
            .fold(1_f64, |m, v| m.max(v[0].abs()));
        for i in 0..n {
            for k in 0..4 {
                let expected = number(&lane["shifts"][i][k]);
                assert!((shifts[row * n + i][k] - expected).abs() <= 1e-15 * energy);
            }
            let mass = &lane["masses_squared"][i];
            assert_eq!(
                masses[row * n + i],
                Complex::new(number(&mass[0]), number(&mass[1]))
            );
            for j in 0..n {
                let expected = number(&lane["s_matrix"][i][j]);
                let actual = s_matrix[row * n * n + i * n + j];
                assert!((actual - expected).abs() <= 1e-14 * energy * energy);
                assert_eq!(actual, s_matrix[row * n * n + j * n + i]);
            }
        }
    }
}

#[test]
fn hgg_and_four_photon_numerators_reproduce_their_bench_cases() {
    for (process, graph) in [("hgg", "top-triangle-0"), ("four-photon", "top-box-0")] {
        let mut case = case(process, graph);
        let legs = case.numerator.legs_info();
        let vectors = legs
            .iter()
            .filter(|leg| leg.state != LegState::Scalar)
            .count();
        assert_eq!(vectors, if process == "hgg" { 2 } else { 4 });
        check_against_case(&mut case);
    }
}

#[test]
fn pentagon_preparation_is_warm_allocation_free_and_matches_its_bench_case() {
    let mut case = case("gg-hhh", "top-pentagon-26");
    assert_eq!(
        (
            case.numerator.propagators(),
            case.numerator.rank(),
            case.numerator.ncoeff()
        ),
        (5, 4, 70)
    );
    check_against_case(&mut case);
    let Case {
        numerator, pool, ..
    } = &mut case;
    let (n, ncoeff) = (numerator.propagators(), numerator.ncoeff());
    // The counter observes this thread's allocations.
    assert!(allocations(|| drop(std::hint::black_box(vec![0_u8; 16]))) > 0);
    for width in [1, 8, 32] {
        let (momenta, helicities, model) = inputs(pool, width);
        let mut tensor = vec![Complex::new(0., 0.); width * ncoeff];
        let mut shifts = vec![[0.; 4]; width * n];
        let mut masses = vec![Complex::new(0., 0.); width * n];
        let mut s_matrix = vec![0.; width * n * n];
        let mut run = || {
            numerator
                .prepare(&momenta, &helicities, &model, width, &mut tensor)
                .unwrap();
            numerator
                .kinematics(
                    &momenta,
                    &model,
                    width,
                    &mut shifts,
                    &mut masses,
                    Some(&mut s_matrix),
                    1e-8,
                )
                .unwrap();
        };
        run();
        assert_eq!(allocations(&mut run), 0, "width {width}");
        assert_eq!(allocations(&mut run), 0, "width {width}");
    }
}

#[test]
fn vector_wavefunctions_are_feynkits_bit_for_bit() {
    let momenta: [[f64; 4]; 5] = [
        [100., 100. * 2. / 3., 100. / 3., 100. * 2. / 3.],
        [225., 0., 0., 225.],
        [225., 0., 0., -225.],
        [300., 30., -40., 120.],
        [91.1876, 0., 0., 0.],
    ];
    for p in momenta {
        let momentum = FourMomentum::from(p);
        for helicity in [-1_i8, 0, 1] {
            for bar in [false, true] {
                let kind = if bar {
                    WavefunctionKind::EpsilonBar
                } else {
                    WavefunctionKind::Epsilon
                };
                let fixed = match helicity {
                    -1 => Helicity::MINUS,
                    0 => Helicity::ZERO,
                    _ => Helicity::PLUS,
                };
                let reference = momentum.wavefunction(kind, fixed);
                match (vector_wavefunction(&p, helicity, bar), reference) {
                    (Some(actual), Ok(reference)) => {
                        for (a, r) in actual.iter().zip(reference.components()) {
                            assert_eq!(a.re.to_bits(), r.re.to_bits(), "{p:?} {helicity} {bar}");
                            assert_eq!(a.im.to_bits(), r.im.to_bits(), "{p:?} {helicity} {bar}");
                        }
                    }
                    (None, Err(_)) => {}
                    (actual, reference) => panic!("{p:?} {helicity}: {actual:?} vs {reference:?}"),
                }
            }
        }
    }
    assert!(vector_wavefunction(&[1., 0., 0., 1.], 2, false).is_none());
}

#[test]
fn invalid_physical_inputs_are_rejected_without_touching_the_output() {
    let Case {
        mut numerator,
        pool,
        ..
    } = case("hgg", "top-triangle-0");
    let (mut momenta, mut helicities, mut model) = inputs(&pool, 2);
    let ncoeff = numerator.ncoeff();
    let sentinel = Complex::new(7., -7.);
    let mut tensor = vec![sentinel; 2 * ncoeff];
    // Leg 0 is the Higgs boson; scalars only admit helicity zero.
    helicities[3] = 1;
    assert!(matches!(
        numerator.prepare(&momenta, &helicities, &model, 2, &mut tensor),
        Err(PhysicalNumeratorError::Helicity {
            row: 1,
            leg: 0,
            helicity: 1
        })
    ));
    helicities[3] = 0;
    momenta[4][2] = f64::NAN;
    assert!(matches!(
        numerator.prepare(&momenta, &helicities, &model, 2, &mut tensor),
        Err(PhysicalNumeratorError::NonFinite { row: 1 })
    ));
    momenta[4][2] = 0.5;
    model.pop();
    assert!(matches!(
        numerator.prepare(&momenta, &helicities, &model, 2, &mut tensor),
        Err(PhysicalNumeratorError::Shape { .. })
    ));
    assert!(tensor.iter().all(|c| *c == sentinel));
}
