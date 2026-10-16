//! Numerical preparation boundary, independent of any reduction engine.
//! This is a correctness probe; no measured duration is a performance result.
use std::{
    collections::{BTreeMap, HashMap},
    error::Error,
    fs,
    path::PathBuf,
};

use feynkit_kinematics::{FourMomentum, Helicity, WavefunctionKind};
use ninjars_graph::{
    Kinematics, Model, PolynomialCoefficientPlan, prepare_top_four_photons, symbols,
};
use serde_json::{Value, json};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        atom::AtomField,
        float::{Complex, Float, RealLike, SingleFloat},
        rational::Rational,
    },
    id::{Pattern, Replacement},
    parser::ParseSettings,
    symbol,
};

type C = Complex<f64>;
type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn atom(text: &str) -> Atom {
    Atom::parse(text, "physical_probe", ParseSettings::default()).unwrap()
}

fn close(a: C, b: C) -> f64 {
    (a.re - b.re).hypot(a.im - b.im) / 1_f64.max(b.re.hypot(b.im))
}

fn point(energy: f64, x: f64, y: f64, z: f64) -> [FourMomentum<f64>; 4] {
    [
        FourMomentum::from_args(energy, 0., 0., energy),
        FourMomentum::from_args(energy, 0., 0., -energy),
        FourMomentum::from_args(energy, x, y, z),
        FourMomentum::from_args(energy, -x, -y, -z),
    ]
}

fn main() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let fixture: Value = serde_json::from_slice(&fs::read(
        root.join("examples/graphs/physical/four-photon/top-box-0.json"),
    )?)?;
    let model = Model::standard_model();
    let graph = prepare_top_four_photons(model.clone(), &Kinematics::new())?.remove(0);
    let q = ["q0", "q1", "q2", "q3", "mu2"].map(|s| atom(&format!("physical_probe::{s}")));
    let loop_components =
        FourMomentum::from([q[0].clone(), q[1].clone(), q[2].clone(), q[3].clone()]);
    let model_parameters = [
        atom("physical_probe::mt"),
        atom("physical_probe::alpha_inverse"),
    ];
    let overrides = BTreeMap::from([
        (symbol!("UFO::MT"), model_parameters[0].clone()),
        (symbol!("UFO::aEWM1"), model_parameters[1].clone()),
        (symbol!("UFO::WT"), Atom::Zero),
    ]);
    let bindings = model.scalar_bindings(None, &overrides)?;
    let replacements: Vec<_> = bindings
        .into_iter()
        .map(|(key, value)| {
            Replacement::new(Pattern::Literal(Atom::var(key)), Pattern::Literal(value))
        })
        .collect();
    let mut carriers = Vec::new();
    let mut sources = Vec::new();
    let mut waves = Vec::new();
    let basis = graph.diagram.loop_momentum_basis();
    // The graph owns physical-leg to external-momentum ordering. No routing is guessed.
    for (position, edge) in basis.external_edges.iter().enumerate() {
        let data = graph
            .diagram
            .edges()
            .find(|(id, _, _)| id == edge)
            .unwrap()
            .2;
        let leg = data.external.as_ref().unwrap().index;
        let momentum = symbols::external_momentum().call(position);
        if graph.family.external_momenta().contains(&momentum) {
            carriers.push(momentum);
            sources.push(leg);
        }
        let candidates: Vec<_> = [
            (symbols::epsilon(), WavefunctionKind::Epsilon),
            (symbols::epsilonbar(), WavefunctionKind::EpsilonBar),
        ]
        .into_iter()
        .filter(|(head, _)| {
            let pattern = head
                .call((edge.0, atom("physical_probe::rep_")))
                .to_pattern();
            graph
                .numerator
                .pattern_match(&pattern, None, None)
                .next()
                .is_some()
        })
        .collect();
        assert_eq!(
            candidates.len(),
            1,
            "one existing vector carrier per external leg"
        );
        waves.push((leg, candidates[0].1, candidates[0].0.call(edge.0)));
    }
    let momentum_inputs = carriers.len();
    carriers.extend(waves.iter().map(|(_, _, carrier)| carrier.clone()));
    let mut parameters: Vec<_> = (0..carriers.len() * 4)
        .map(|i| atom(&format!("physical_probe::p{i}")))
        .collect();
    let symbolic: BTreeMap<_, _> = carriers
        .iter()
        .cloned()
        .zip(
            parameters
                .as_chunks::<4>()
                .0
                .iter()
                .cloned()
                .map(FourMomentum::from),
        )
        .collect();
    parameters.extend(model_parameters);
    let kinematics = graph.component_kinematics(&loop_components, &symbolic)?;
    let expression = kinematics
        .apply(&(&graph.numerator * &graph.overall_factor))
        .replace_multiple(&replacements)
        .expand();
    let plan = PolynomialCoefficientPlan::new(expression.clone(), q.clone(), parameters.clone())?;
    assert_eq!(plan.rank(), 4);
    let mut jit = plan.compile_symjit_o2()?;
    let mut interpreted = plan.compile(0_f64)?;
    let propagator_heads: Vec<_> = graph
        .shifts(&symbolic)?
        .iter()
        .flat_map(|p| p.components().map(Clone::clone))
        .chain(graph.squared_masses())
        .map(|p| p.replace_multiple(&replacements))
        .collect();
    let mut propagators = Atom::evaluator_multiple(&propagator_heads, &parameters)
        .cores(1)
        .build()?
        .map_coeff_with_prec(
            &|v| C::new(0_f64.from_rational(&v.re), 0_f64.from_rational(&v.im)),
            53,
        )
        .into_batch_evaluator();
    let points = [
        point(100., 60., 0., 80.),
        point(130., 50., 120., 0.),
        point(85., 0., 68., 51.),
    ];
    for p in &points {
        for v in p {
            assert!(v.mass_squared().abs() < 1e-9);
        }
        let conservation = p[0] + p[1] - p[2] - p[3];
        assert!(conservation.components().iter().all(|x| x.abs() < 1e-12));
    }
    // Raw model inputs vary as well as momenta. Couplings remain analytic in the
    // evaluator, so model closure is setup and numerical coupling work is per call.
    let model_rows = [[173., 132.507], [179., 133.], [167., 131.5]];
    let helicities = [
        [Helicity::PLUS; 4],
        [
            Helicity::PLUS,
            Helicity::PLUS,
            Helicity::MINUS,
            Helicity::MINUS,
        ],
        [
            Helicity::MINUS,
            Helicity::PLUS,
            Helicity::MINUS,
            Helicity::PLUS,
        ],
    ];
    let bind = |point_index: usize, state: usize| -> Result<Vec<C>> {
        let p = &points[point_index];
        let mut values = Vec::with_capacity(parameters.len());
        for &leg in &sources {
            values.extend(p[leg].components().map(|x| C::new(*x, 0.)));
        }
        for &(leg, kind, _) in &waves {
            values.extend_from_slice(
                p[leg]
                    .wavefunction(kind, helicities[state][leg])?
                    .components(),
            );
        }
        values.extend(model_rows[point_index].map(|v| C::new(v, 0.)));
        Ok(values)
    };
    // Independently retained MadLoop HELAS fixture values fix the shared phase.
    let mut wave_error = 0_f64;
    for state in 0..3 {
        let values = bind(0, state)?;
        for (wave, (leg, _, _)) in waves.iter().enumerate() {
            for component in 0..4 {
                let expected = atom(
                    fixture["samples"][state]["external_polarizations"][leg.to_string()][component]
                        .as_str()
                        .unwrap(),
                )
                .evaluate_with_prec(&HashMap::<Atom, C>::new(), 53)?;
                wave_error = wave_error.max(close(
                    values[(momentum_inputs + wave) * 4 + component],
                    expected,
                ));
            }
        }
    }
    assert!(
        wave_error < 1e-14,
        "shared wavefunction/HELAS phase mismatch {wave_error}"
    );
    let mut records = Vec::new();
    let mut coefficient_error = 0_f64;
    let mut reconstruction_error = 0_f64;
    for rows in [0, 1, 8, 32, 128, 1024, 3, 31] {
        let mut input = Vec::new();
        for row in 0..rows {
            input.extend(bind((rows - row) % 3, row % 3)?);
        }
        let mut actual = vec![C::new(0., 0.); rows * plan.coefficient_expressions().len()];
        let mut control = actual.clone();
        jit.evaluate_batch(&input, &mut actual, rows)?;
        interpreted.evaluate_batch(&input, &mut control, rows)?;
        for (a, b) in actual.iter().zip(&control) {
            coefficient_error = coefficient_error.max(close(*a, *b));
        }
        let mut kinematics = vec![C::new(0., 0.); rows * propagator_heads.len()];
        propagators.evaluate_batch(&input, &mut kinematics, rows)?;
        assert!(
            kinematics
                .iter()
                .all(|v| v.re.is_finite() && v.im.is_finite())
        );
        for row in 0..rows.min(9) {
            let mut values: HashMap<_, _> = parameters
                .iter()
                .cloned()
                .zip(
                    input[row * parameters.len()..(row + 1) * parameters.len()]
                        .iter()
                        .copied(),
                )
                .collect();
            let loop_values = [
                C::new(0.7, 0.2),
                C::new(-1.3, 0.4),
                C::new(0.5, -0.6),
                C::new(2., 0.1),
                C::new(0.3, 0.),
            ];
            values.extend(q.iter().cloned().zip(loop_values));
            let expected: C = expression.evaluate_with_prec(&values, 53)?;
            let mut reconstructed: HashMap<_, _> = q.iter().cloned().zip(loop_values).collect();
            reconstructed.extend(
                plan.coefficient_symbols().iter().cloned().zip(
                    actual[row * plan.coefficient_symbols().len()
                        ..(row + 1) * plan.coefficient_symbols().len()]
                        .iter()
                        .copied(),
                ),
            );
            let value: C = plan.template().evaluate_with_prec(&reconstructed, 53)?;
            reconstruction_error = reconstruction_error.max(close(value, expected));
        }
        records.push(json!({"rows":rows,"coefficient_components":actual.len(),"propagator_components":kinematics.len()}));
    }
    assert!(coefficient_error <= 1e-9 && reconstruction_error <= 1e-9);
    // A higher-precision oracle distinguishes disagreement between two binary64
    // execution orders from the accuracy of each. Inputs are the exact binary64
    // wavefunction/parameter components used above, not independently rounded states.
    let mut unique_inputs = Vec::new();
    for point_index in 0..3 {
        for state in 0..3 {
            unique_inputs.extend(bind(point_index, state)?);
        }
    }
    let prototype = Float::new(256);
    let promote = |x: f64| prototype.from_rational(&Rational::try_from(x).unwrap());
    let exact_inputs: Vec<_> = unique_inputs
        .iter()
        .map(|v| Complex::new(promote(v.re), promote(v.im)))
        .collect();
    let mut high = plan.compile(prototype)?;
    let mut oracle = vec![
        Complex::new(Float::new(256), Float::new(256));
        9 * plan.coefficient_expressions().len()
    ];
    high.evaluate_batch(&exact_inputs, &mut oracle, 9)?;
    let mut jit_unique = vec![C::new(0., 0.); oracle.len()];
    let mut interpreter_unique = jit_unique.clone();
    jit.evaluate_batch(&unique_inputs, &mut jit_unique, 9)?;
    interpreted.evaluate_batch(&unique_inputs, &mut interpreter_unique, 9)?;
    let mut high_error = [0_f64; 2];
    let mut worst = [Value::Null, Value::Null];
    for (index, ((a, b), expected)) in jit_unique
        .iter()
        .zip(&interpreter_unique)
        .zip(&oracle)
        .enumerate()
    {
        let rounded = C::new(expected.re.to_f64(), expected.im.to_f64());
        for (provider, value) in [a, b].into_iter().enumerate() {
            let error = close(*value, rounded);
            if error > high_error[provider] {
                high_error[provider] = error;
                worst[provider] = json!({"row":index/plan.coefficient_symbols().len(),"head":index%plan.coefficient_symbols().len(),"actual":[value.re,value.im],"expected":[expected.re.to_string(),expected.im.to_string()],"error":error});
            }
        }
    }
    eprintln!(
        "high precision coefficient controls {}",
        serde_json::to_string(&worst)?
    );
    // Stability experiment uses Symbolica's existing factorization only. The
    // public Atom API/docs, poly factor implementation/examples and FeynKit's
    // symbolic factorization consumers were checked before this composition.
    let factored_heads: Vec<_> = plan
        .coefficient_expressions()
        .iter()
        .map(|head| {
            let factored = head.factor();
            assert!((&factored - head).expand().is_zero());
            factored
        })
        .collect();
    let mut factored = Atom::evaluator_multiple(&factored_heads, &parameters)
        .cores(1)
        .build()?
        .map_coeff_with_prec(
            &|v| C::new(0_f64.from_rational(&v.re), 0_f64.from_rational(&v.im)),
            53,
        )
        .into_batch_evaluator();
    let mut factored_values = jit_unique.clone();
    factored.evaluate_batch(&unique_inputs, &mut factored_values, 9)?;
    let factor_error = factored_values
        .iter()
        .zip(&oracle)
        .map(|(v, e)| close(*v, C::new(e.re.to_f64(), e.im.to_f64())))
        .fold(0_f64, f64::max);
    let factored_expression = plan
        .polynomial()
        .map_coeff(
            |head| head.factor(),
            AtomField {
                statistical_zero_test: false,
                ..AtomField::new()
            },
        )
        .flatten(false);
    let factored_plan =
        PolynomialCoefficientPlan::new(factored_expression, q.clone(), parameters.clone())?;
    assert_eq!(
        factored_plan.coefficient_expressions().len(),
        factored_heads.len()
    );
    for (a, b) in factored_plan
        .coefficient_expressions()
        .iter()
        .zip(&factored_heads)
    {
        assert!((a - b).expand().is_zero(), "coefficient order changed");
    }
    let mut factored_jit = factored_plan.compile_symjit_o2()?;
    let mut factored_jit_values = jit_unique.clone();
    factored_jit.evaluate_batch(&unique_inputs, &mut factored_jit_values, 9)?;
    let factored_jit_error = factored_jit_values
        .iter()
        .zip(&oracle)
        .map(|(v, e)| close(*v, C::new(e.re.to_f64(), e.im.to_f64())))
        .fold(0_f64, f64::max);
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"passed":high_error.into_iter().all(|e|e<=1e-9),"provider_parity_passed":true,"performance_acceptance":false,
        "scope":"FeynKit wavefunctions and routed propagators plus Symbolica/SymJIT coefficient preparation from numerical momenta, helicities, pole mass and inverse electromagnetic coupling; no reduction or timings",
        "graph":"four-photon/top-box-0","distinct_momentum_points":3,"helicity_states":3,"model_input_rows":model_rows,
        "coefficient_heads":plan.coefficient_expressions().len(),"parameter_count":parameters.len(),
        "coefficient_float256_oracle":{"scope":"all nine distinct momentum/helicity bindings, exact promotion of binary64 component inputs; outputs rounded to binary64 for comparison","criterion":1e-9,"jit_max_scaled_error":high_error[0],"interpreter_max_scaled_error":high_error[1],"worst_cases":worst},
        "factorization_experiment":{"scope":"Symbolica factor() on every coefficient head with exact identity checks, existing interpreter and existing PolynomialCoefficientPlan/SymJIT through map_coeff/flatten; no production changes","interpreter_max_scaled_error":factor_error,"jit_max_scaled_error":factored_jit_error,"passed":factor_error<=1e-9 && factored_jit_error<=1e-9},
        "wavefunction_fixture_max_scaled_error":wave_error,"coefficient_max_scaled_error":coefficient_error,
        "reconstruction_max_scaled_error":reconstruction_error,"batches":records})
        )?
    );
    Ok(())
}
