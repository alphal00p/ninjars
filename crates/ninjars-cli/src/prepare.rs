use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use clap::Args as ClapArgs;
use ninjars_graph::{
    FourMomentum, Kinematics, Model, NumeratorMode, PreparedIntegrand, prepare_dot, symbols,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use symbolica::{
    atom::{Atom, AtomCore, AtomView, Symbol},
    id::{Pattern, Replacement},
    parser::ParseSettings,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(ClapArgs)]
pub struct Args {
    /// DOT graph, or a schema-version-1 JSON fixture resolving its own graph path.
    pub input: PathBuf,
    /// Sample fixture accompanying a DOT input.
    #[arg(long)]
    pub samples: Option<PathBuf>,
    /// Built-in phi3/phi4, or path to a FeynKit model JSON. Defaults to fixture model or phi3.
    #[arg(long)]
    pub model: Option<String>,
    /// Construct local numerators and default external states from model rules.
    #[arg(long)]
    pub feynman_rules: bool,
    /// Write JSON to a file; default writes JSON to standard output.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: u32,
    graph: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default = "one")]
    scale_squared: String,
    #[serde(default)]
    parameter_overrides: BTreeMap<String, String>,
    #[serde(default)]
    samples: Vec<Sample>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Sample {
    external_momenta: BTreeMap<usize, [String; 4]>,
}

fn one() -> String {
    "1".into()
}

fn parse(expression: &str) -> Result<Atom> {
    Atom::parse(expression, "ninjars", ParseSettings::default())
        .map_err(|error| format!("invalid Symbolica expression {expression:?}: {error}").into())
}

fn canonical(expression: &Atom) -> String {
    expression.to_canonical_string()
}

fn substitute(expression: &Atom, bindings: &[Replacement]) -> Atom {
    expression.replace_multiple(bindings).together()
}

fn bind_external(
    sample: &Sample,
    prepared: &PreparedIntegrand,
    bindings: &[Replacement],
) -> Result<BTreeMap<Atom, FourMomentum<Atom>>> {
    let count = prepared.diagram.loop_momentum_basis().external_edges.len();
    let mut momenta = BTreeMap::new();
    for (index, components) in &sample.external_momenta {
        if *index >= count {
            return Err(format!("external momentum index {index} is outside 0..{count}").into());
        }
        let values = components
            .iter()
            .map(|text| parse(text).map(|atom| substitute(&atom, bindings)))
            .collect::<Result<Vec<_>>>()?;
        momenta.insert(
            symbols::external_momentum().call(*index),
            FourMomentum::from_args(
                values[0].clone(),
                values[1].clone(),
                values[2].clone(),
                values[3].clone(),
            ),
        );
    }
    // Check any supplied dependent momenta against the same routing used for
    // denominators, with Symbolica doing the exact equality test.
    for index in sample.external_momenta.keys() {
        let name = symbols::external_momentum().call(*index);
        let supplied = &momenta[&name];
        let edge = prepared.diagram.loop_momentum_basis().external_edges[*index];
        let routed = prepared
            .diagram
            .loop_momentum_basis()
            .route_expression(&symbols::momentum().call(edge.0));
        let mut expected = FourMomentum::from_args(Atom::Zero, Atom::Zero, Atom::Zero, Atom::Zero);
        for (term, coefficient) in prepared.family.kinematics().linear_terms(&routed)? {
            let vector = momenta
                .get(&term)
                .ok_or_else(|| format!("missing routed external momentum {term}"))?;
            expected = expected + vector.clone() * coefficient;
        }
        if expected
            .components()
            .iter()
            .zip(supplied.components())
            .any(|(a, b)| (*a - b).together() != Atom::Zero)
        {
            return Err(format!(
                "supplied momentum {name} violates FeynKit's dependent-momentum routing"
            )
            .into());
        }
    }
    Ok(momenta)
}

pub fn run(args: Args) -> Result<()> {
    let input_is_json = args
        .input
        .extension()
        .is_some_and(|extension| extension == "json");
    if input_is_json && args.samples.is_some() {
        return Err("a JSON input already provides samples; do not also pass --samples".into());
    }
    let fixture_path = if input_is_json {
        Some(args.input.clone())
    } else {
        args.samples.clone()
    };
    let fixture = fixture_path
        .as_ref()
        .map(|path| -> Result<Fixture> {
            let fixture: Fixture = serde_json::from_str(&fs::read_to_string(path)?)?;
            if fixture.schema_version != 1 {
                return Err(format!(
                    "unsupported fixture schema_version {}",
                    fixture.schema_version
                )
                .into());
            }
            Ok(fixture)
        })
        .transpose()?;
    let base = fixture_path
        .as_deref()
        .and_then(Path::parent)
        .unwrap_or_else(|| Path::new("."));
    let graph_path = if input_is_json {
        base.join(&fixture.as_ref().unwrap().graph)
    } else {
        if let Some(fixture) = &fixture {
            if fs::canonicalize(base.join(&fixture.graph))? != fs::canonicalize(&args.input)? {
                return Err("sample fixture refers to a different graph than the DOT input".into());
            }
        }
        args.input.clone()
    };
    let model_name = args
        .model
        .as_deref()
        .or_else(|| fixture.as_ref().and_then(|f| f.model.as_deref()))
        .unwrap_or("phi3");
    let model = Arc::new(match model_name {
        "phi3" => Model::phi3(),
        "phi4" => Model::phi4(),
        path => {
            let path = if args.model.is_some() {
                PathBuf::from(path)
            } else {
                base.join(path)
            };
            Model::from_json(&fs::read_to_string(path)?)?
        }
    });
    let mut overrides = BTreeMap::<Symbol, Atom>::new();
    if let Some(fixture) = &fixture {
        for (name, value) in &fixture.parameter_overrides {
            let key = Atom::parse(name, "UFO", ParseSettings::default())?;
            let AtomView::Var(symbol) = key.as_view() else {
                return Err(format!("parameter override key {name:?} is not a symbol").into());
            };
            overrides.insert(symbol.get_symbol(), parse(value)?);
        }
    }
    let bindings = model.scalar_bindings(None, &overrides)?;
    let replacements = bindings
        .iter()
        .map(|(key, value)| {
            Replacement::new(
                Pattern::Literal(Atom::var(*key)),
                Pattern::Literal(value.clone()),
            )
        })
        .collect::<Vec<_>>();
    let dot = fs::read_to_string(&graph_path)?;
    let prepared = prepare_dot(
        model.clone(),
        &dot,
        &Kinematics::new(),
        if args.feynman_rules {
            NumeratorMode::FeynmanRules
        } else {
            NumeratorMode::Explicit
        },
    )?;
    let loop_components = FourMomentum::from_args(
        parse("ninjars::q0")?,
        parse("ninjars::q1")?,
        parse("ninjars::q2")?,
        parse("ninjars::q3")?,
    );
    let mut samples = Vec::<Value>::new();
    if let Some(fixture) = &fixture {
        for (lane, sample) in fixture.samples.iter().enumerate() {
            let external = bind_external(sample, &prepared, &replacements)
                .map_err(|error| format!("sample {lane}: {error}"))?;
            let kinematics = prepared.component_kinematics(&loop_components, &external)?;
            let specialize = |atom: &Atom| substitute(&kinematics.apply(atom), &replacements);
            let numerator = specialize(&prepared.component_numerator(&loop_components, &external)?);
            let shifts = prepared
                .shifts(&external)?
                .iter()
                .map(|vector| {
                    vector
                        .components()
                        .iter()
                        .map(|component| canonical(&specialize(component)))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            samples.push(json!({
                "lane": lane,
                "numerator": canonical(&numerator),
                "shifts": shifts,
                "mass_squared": prepared.squared_masses().iter().map(|mass| canonical(&specialize(mass))).collect::<Vec<_>>(),
                "overall_factor": canonical(&specialize(&prepared.overall_factor)),
            }));
        }
    }
    let scale = parse(
        fixture
            .as_ref()
            .map(|f| f.scale_squared.as_str())
            .unwrap_or("1"),
    )?;
    let result = json!({
        "schema_version": 1,
        "operation": "prepare",
        "ninjars_version": env!("CARGO_PKG_VERSION"),
        "expression_format": "symbolica_canonical",
        "graph": graph_path,
        "graph_source": dot,
        "model": model_name,
        "model_fingerprint": model.fingerprint().to_string(),
        "numerator_mode": if args.feynman_rules { "feynman_rules" } else { "explicit" },
        "metric": "+---",
        "scale_squared": canonical(&substitute(&scale, &replacements)),
        "propagator_edges": prepared.propagator_edges.iter().map(|edge| edge.0).collect::<Vec<_>>(),
        "denominators": prepared.family.denominators().iter().map(|denominator| canonical(&substitute(denominator, &replacements))).collect::<Vec<_>>(),
        "external_basis": prepared.family.external_momenta().iter().map(canonical).collect::<Vec<_>>(),
        "momentum_basis": prepared.diagram.loop_momentum_basis(),
        "numerator": canonical(&substitute(&prepared.numerator, &replacements)),
        "overall_factor": canonical(&substitute(&prepared.overall_factor, &replacements)),
        "input_fixture": fixture,
        "samples": samples,
    });
    let output = serde_json::to_string_pretty(&result)? + "\n";
    if let Some(path) = args.output {
        fs::write(path, output)?;
    } else {
        print!("{output}");
    }
    Ok(())
}
