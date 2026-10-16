use std::{collections::BTreeMap, fs, path::PathBuf};

use feynkit_kinematics::{Helicity, WavefunctionKind};
use ninjars_graph::{
    FourMomentum, Kinematics, Model, NumeratorMode, PhysicalInputBatch, PhysicalInputEvaluator,
    PhysicalInputPlan, PhysicalScalar, PreparedIntegrand, prepare_dot,
};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::float::{Complex, DoubleFloat, Float, FloatLike, Real, SingleFloat},
    evaluate::EvaluationDomain,
    parse, symbol,
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn graph(path: &str) -> PreparedIntegrand {
    prepare_dot(
        Model::standard_model(),
        &fs::read_to_string(root().join(path)).unwrap(),
        &Kinematics::new(),
        NumeratorMode::Explicit,
    )
    .unwrap()
    .with_color_dimension_invariants()
    .unwrap()
}
fn variables() -> [Atom; 5] {
    [
        parse!("ninjars::q0"),
        parse!("ninjars::q1"),
        parse!("ninjars::q2"),
        parse!("ninjars::q3"),
        parse!("ninjars::mu2"),
    ]
}

#[test]
fn physical_evaluators_preserve_thread_safety_with_optional_jit() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PhysicalInputEvaluator<f64>>();
    assert_send_sync::<PhysicalInputEvaluator<DoubleFloat>>();
    assert_send_sync::<PhysicalInputEvaluator<Float>>();
    #[cfg(feature = "numerator-jit")]
    {
        assert_send_sync::<symbolica::evaluate::JITCompiledEvaluator<Complex<f64>>>();
        assert_send_sync::<ninjars_graph::SymJitCoefficientEvaluator>();
    }
}

fn plan(path: &str) -> PhysicalInputPlan {
    PhysicalInputPlan::new(
        &graph(path),
        variables(),
        ["MT", "ymt", "MH", "Gf", "aS", "aEWM1", "MZ"]
            .map(|s| symbol!(format!("UFO::{s}")))
            .to_vec(),
        BTreeMap::from([
            (symbol!("UFO::WT"), Atom::Zero),
            (symbol!("UFO::WH"), Atom::Zero),
        ]),
        true,
    )
    .unwrap()
}
fn exact<T: Real + SingleFloat>(text: &str, prototype: &T) -> T
where
    Complex<T>: Real + EvaluationDomain,
{
    let atom = Atom::parse(text, "physical_input_test", Default::default()).unwrap();
    let result: Complex<T> = atom
        .evaluate_with_prec(
            &std::collections::HashMap::<Atom, Complex<T>>::new(),
            prototype.get_precision(),
        )
        .unwrap();
    assert!(result.im.is_zero());
    result.re
}
fn hgg<T: PhysicalScalar>(
    plan: &PhysicalInputPlan,
    prototype: T,
    tolerance: &str,
    mut evaluator: PhysicalInputEvaluator<T>,
) where
    Complex<T>: Real + EvaluationDomain,
{
    let tol = exact(tolerance, &prototype);
    let zero = prototype.zero();
    let mut momenta = Vec::new();
    let mut parameters = Vec::new();
    for (mh, mt, ymt) in [(125, 173, 173), (130, 180, 190), (125, 173, 346)] {
        let mass = prototype.from_usize(mh);
        let half = mass.clone() / prototype.from_usize(2);
        momenta.extend([
            FourMomentum::from([mass, zero.clone(), zero.clone(), zero.clone()]),
            FourMomentum::from([half.clone(), zero.clone(), zero.clone(), half.clone()]),
            FourMomentum::from([half.clone(), zero.clone(), zero.clone(), -half]),
        ]);
        parameters.extend(
            [
                mt.to_string(),
                ymt.to_string(),
                mh.to_string(),
                "116639/10^10".into(),
                "59/500".into(),
                "132507/1000".into(),
                "22797/250".into(),
            ]
            .iter()
            .map(|s| Complex::new(exact(s, &prototype), zero.clone())),
        );
    }
    let helicities = [0, 1, 1, 0, 1, 1, 0, 1, 1].map(|h| Helicity::try_from(h).unwrap());
    let scales = vec![prototype.from_usize(10000); 3];
    let values = evaluator
        .evaluate_batch(PhysicalInputBatch {
            lanes: 3,
            momenta: &momenta,
            helicities: &helicities,
            model_parameters: &parameters,
            scales_squared: &scales,
        })
        .unwrap()
        .clone();
    assert_eq!(values.lanes, 3);
    assert_eq!(values.shifts.len(), 9);
    assert_eq!(values.s_matrix.len(), 27);
    for (row, mt) in [173, 180, 173].into_iter().enumerate() {
        for mass in &values.masses_squared[3 * row..3 * row + 3] {
            assert!((mass.re.clone() - prototype.from_usize(mt * mt)).norm() < tol);
            assert!(mass.im.is_zero());
        }
    }
    let columns = plan.coefficients().coefficient_symbols().len();
    assert!(columns > 0);
    for (a, b) in values.coefficients[..columns]
        .iter()
        .zip(&values.coefficients[2 * columns..])
    {
        let expected = a.clone() * prototype.from_usize(2);
        let delta = b.clone() - expected.clone();
        assert!(delta.norm().re < tol.clone() * (prototype.one() + expected.norm().re));
    }
    for row in [2, 0, 1] {
        let single = evaluator
            .evaluate_batch(PhysicalInputBatch {
                lanes: 1,
                momenta: &momenta[3 * row..3 * row + 3],
                helicities: &helicities[3 * row..3 * row + 3],
                model_parameters: &parameters[7 * row..7 * row + 7],
                scales_squared: &scales[row..row + 1],
            })
            .unwrap();
        assert_eq!(single.shifts, values.shifts[3 * row..3 * row + 3]);
        assert_eq!(
            single.coefficients,
            values.coefficients[columns * row..columns * (row + 1)]
        );
    }
    let before = evaluator.values().clone();
    assert!(
        evaluator
            .evaluate_batch(PhysicalInputBatch {
                lanes: 3,
                momenta: &momenta[..8],
                helicities: &helicities,
                model_parameters: &parameters,
                scales_squared: &scales
            })
            .is_err()
    );
    assert_eq!(evaluator.values(), &before);
    let mut bad = momenta.clone();
    bad[3].temporal.value += prototype.one();
    assert!(
        evaluator
            .evaluate_batch(PhysicalInputBatch {
                lanes: 3,
                momenta: &bad,
                helicities: &helicities,
                model_parameters: &parameters,
                scales_squared: &scales
            })
            .is_err()
    );
    assert_eq!(evaluator.values(), &before);
    let mut bad_helicities = helicities;
    bad_helicities[3] = Helicity::try_from(1).unwrap();
    assert!(
        evaluator
            .evaluate_batch(PhysicalInputBatch {
                lanes: 3,
                momenta: &momenta,
                helicities: &bad_helicities,
                model_parameters: &parameters,
                scales_squared: &scales
            })
            .is_err()
    );
    assert_eq!(evaluator.values(), &before);
    let empty = evaluator
        .evaluate_batch(PhysicalInputBatch {
            lanes: 0,
            momenta: &[],
            helicities: &[],
            model_parameters: &[],
            scales_squared: &[],
        })
        .unwrap();
    assert_eq!(empty.lanes, 0);
    assert!(empty.shifts.is_empty() && empty.coefficients.is_empty());
}

#[test]
fn hgg_raw_physical_batch_rebinding_precision_and_transactional_errors() {
    let plan = plan("examples/graphs/physical/hgg/top-triangle-0.dot");
    assert_eq!(plan.propagator_edges().len(), 3);
    assert_eq!(plan.external_legs().len(), 3);
    assert!(
        plan.external_legs()[1..]
            .iter()
            .all(|leg| leg.wavefunction == WavefunctionKind::EpsilonBar)
    );
    hgg(&plan, 0_f64, "1e-11", plan.compile(0_f64, 1e-11).unwrap());
    let dd = DoubleFloat::from(0.);
    hgg(
        &plan,
        dd,
        "1e-26",
        plan.compile(dd, exact("1e-26", &dd)).unwrap(),
    );
    let high = Float::new(256);
    let evaluator = plan.compile(high.clone(), exact("1e-65", &high)).unwrap();
    hgg(&plan, high, "1e-65", evaluator);
}

#[cfg(feature = "numerator-jit")]
#[test]
fn symjit_hgg_rebinding_and_transactional_errors_share_preparation() {
    let plan = plan("examples/graphs/physical/hgg/top-triangle-0.dot");
    hgg(
        &plan,
        0_f64,
        "1e-11",
        plan.compile_symjit_o2(1e-11).unwrap(),
    );
}

#[test]
fn model_scalar_names_are_known_distinct_and_disjoint() {
    let graph = graph("examples/graphs/physical/hgg/top-triangle-0.dot");
    let mt = symbol!("UFO::MT");
    let typo = symbol!("UFO::Mtt");
    for (free, fixed, expected) in [
        (vec![typo], BTreeMap::new(), "unknown model scalar"),
        (
            vec![mt],
            BTreeMap::from([(typo, Atom::Zero)]),
            "unknown model scalar",
        ),
        (vec![mt, mt], BTreeMap::new(), "distinct and disjoint"),
        (
            vec![mt],
            BTreeMap::from([(mt, Atom::Zero)]),
            "distinct and disjoint",
        ),
    ] {
        match PhysicalInputPlan::new(&graph, variables(), free, fixed, false) {
            Err(error) => assert!(error.to_string().contains(expected), "{error}"),
            Ok(_) => panic!("invalid model input declaration was accepted"),
        }
    }
}

#[test]
fn pentagon_changing_physical_points_match_independent_madloop_raw_grid() {
    pentagon_raw_grid(|plan| plan.compile(0_f64, 1e-10).unwrap());
}

#[cfg(feature = "numerator-jit")]
#[test]
fn symjit_pentagon_changing_points_match_independent_madloop_raw_grid() {
    pentagon_raw_grid(|plan| plan.compile_symjit_o2(1e-10).unwrap());
}

fn pentagon_raw_grid(compile: impl FnOnce(&PhysicalInputPlan) -> PhysicalInputEvaluator<f64>) {
    let plan = plan("examples/graphs/physical/gg-hhh/top-pentagon-26.dot");
    assert_eq!(plan.propagator_edges().len(), 5);
    assert_eq!(plan.external_legs().len(), 5);
    assert_eq!(plan.coefficients().rank(), 4);
    let reference: serde_json::Value = serde_json::from_slice(
        &fs::read(root().join("validation/evidence/gg-hhh-changing-madloop.json")).unwrap(),
    )
    .unwrap();
    let points = reference["points"].as_array().unwrap();
    let mut momenta = Vec::new();
    let mut helicities = Vec::new();
    let mut model = Vec::new();
    let mut scales = Vec::new();
    for point in points {
        momenta.extend(
            point["momenta"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| FourMomentum::from(std::array::from_fn(|i| v[i].as_f64().unwrap()))),
        );
        helicities.extend(
            point["helicities"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| Helicity::try_from(v.as_i64().unwrap() as i8).unwrap()),
        );
        model.extend(
            point["model"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| Complex::new(v.as_f64().unwrap(), 0.)),
        );
        scales.push(point["scale_squared"].as_f64().unwrap());
    }
    let mut evaluator = compile(&plan);
    let prepared = evaluator
        .evaluate_batch(PhysicalInputBatch {
            lanes: points.len(),
            momenta: &momenta,
            helicities: &helicities,
            model_parameters: &model,
            scales_squared: &scales,
        })
        .unwrap();
    let coefficients = plan.coefficients();
    let parameters: Vec<_> = variables()
        .into_iter()
        .chain(coefficients.coefficient_symbols().iter().cloned())
        .collect();
    let mut numerator =
        Atom::evaluator_multiple(std::slice::from_ref(coefficients.template()), &parameters)
            .cores(1)
            .build()
            .unwrap()
            .map_coeff_with_prec(
                &|v| Complex::new(0_f64.from_rational(&v.re), 0_f64.from_rational(&v.im)),
                53,
            )
            .into_batch_evaluator();
    let columns = coefficients.coefficient_symbols().len();
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for (row, point) in points.iter().enumerate() {
        let case = reference["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["point_id"] == point["id"] && c["ctmode"] == 4)
            .unwrap();
        for sample in case["numerator_samples"].as_array().unwrap() {
            for q in sample["q"].as_array().unwrap() {
                inputs.push(Complex::new(
                    -q[0].as_f64().unwrap(),
                    -q[1].as_f64().unwrap(),
                ));
            }
            inputs.push(Complex::new(0., 0.));
            inputs.extend_from_slice(&prepared.coefficients[row * columns..(row + 1) * columns]);
            expected.push(Complex::new(
                -0.5 * sample["raw"][0].as_f64().unwrap(),
                -0.5 * sample["raw"][1].as_f64().unwrap(),
            ));
        }
    }
    assert_eq!(expected.len(), 210);
    let mut actual = vec![Complex::new(0., 0.); expected.len()];
    numerator
        .evaluate_batch(&inputs, &mut actual, expected.len())
        .unwrap();
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!(
            (actual - expected).norm().re <= 1e-9 * (1_f64.max(expected.norm().re)),
            "{actual} != {expected}"
        );
    }
}

#[test]
fn exact_non_binary64_input_is_not_rounded_before_preparation() {
    let proto = Float::new(256);
    let plan = plan("examples/graphs/physical/hgg/top-triangle-0.dot");
    let mut evaluator = plan.compile(proto.clone(), exact("1e-65", &proto)).unwrap();
    let momenta = [
        ["125", "0", "0", "0"],
        ["125/2", "0", "0", "125/2"],
        ["125/2", "0", "0", "-125/2"],
    ]
    .map(|p| FourMomentum::from(p.map(|v| exact(v, &proto))));
    let parameters = [
        "173+1/10^50",
        "173",
        "125",
        "116639/10^10",
        "59/500",
        "132507/1000",
        "22797/250",
    ]
    .map(|v| Complex::new(exact(v, &proto), proto.zero()));
    let helicities = [0_i8, 1, 1].map(|h| Helicity::try_from(h).unwrap());
    let scales = [proto.from_usize(10000)];
    let output = evaluator
        .evaluate_batch(PhysicalInputBatch {
            lanes: 1,
            momenta: &momenta,
            helicities: &helicities,
            model_parameters: &parameters,
            scales_squared: &scales,
        })
        .unwrap();
    let expected = exact("(173+1/10^50)^2", &proto);
    for mass in &output.masses_squared {
        assert!(mass.re > proto.from_usize(29929));
        assert!((mass.re.clone() - expected.clone()).norm() < exact("1e-65", &proto));
    }
    let mut low_precision = parameters;
    low_precision[0].re = Float::new(64).from_usize(173);
    let output = evaluator
        .evaluate_batch(PhysicalInputBatch {
            lanes: 1,
            momenta: &momenta,
            helicities: &helicities,
            model_parameters: &low_precision,
            scales_squared: &scales,
        })
        .unwrap();
    for mass in &output.masses_squared {
        assert!(
            mass.re.get_precision() < 256,
            "represented input precision must not be inflated"
        );
        assert!((mass.re.clone() - proto.from_usize(29929)).norm() < exact("1e-12", &proto));
    }
}
