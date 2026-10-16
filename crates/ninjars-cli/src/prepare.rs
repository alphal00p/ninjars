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
    symbol,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub(crate) struct PreparedSample {
    pub lane: usize,
    pub numerator: Atom,
    pub overall_factor: Atom,
    pub shifts: Vec<FourMomentum<Atom>>,
    pub mass_squared: Vec<Atom>,
    /// Optional row-major n×n matrix in prepared propagator order.
    pub s_matrix: Option<Vec<Atom>>,
    pub parameters: Vec<Atom>,
}

pub(crate) struct PreparedDocument {
    pub document: Value,
    pub propagators: usize,
    pub samples: Vec<PreparedSample>,
    pub scale_squared: Atom,
    pub numerator_template: Atom,
    pub parameters: Vec<Atom>,
}

#[derive(ClapArgs)]
pub struct Args {
    /// DOT graph, or a schema-version-1 JSON fixture resolving its own graph path.
    pub input: PathBuf,
    /// Sample fixture accompanying a DOT input.
    #[arg(long)]
    pub samples: Option<PathBuf>,
    /// Built-in phi3/phi4/sm, or path to a FeynKit model JSON. Defaults to fixture model or phi3.
    #[arg(long)]
    pub model: Option<String>,
    /// Construct local numerators and default external states from model rules.
    #[arg(long)]
    pub feynman_rules: bool,
    /// Write JSON to a file; default writes JSON to standard output.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Export per-sample C++ callbacks and binary64 inputs for original Ninja validation.
    #[cfg(feature = "reference-codegen")]
    #[arg(long)]
    pub reference_dir: Option<PathBuf>,
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
    /// Explicit Idenso policy for fundamental color invariants.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    color_dimension_invariants: bool,
    #[serde(default)]
    samples: Vec<Sample>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Sample {
    external_momenta: BTreeMap<usize, [String; 4]>,
    /// Exact polarization components, keyed by the graph's physical leg index.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    external_polarizations: BTreeMap<usize, [String; 4]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    s_matrix: Option<Vec<Vec<String>>>,
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
    polarizations: &BTreeMap<usize, Atom>,
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
            expected += vector.clone() * coefficient;
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
    for (leg, carrier) in polarizations {
        let values = sample.external_polarizations[leg]
            .iter()
            .map(|text| parse(text).map(|atom| substitute(&atom, bindings)))
            .collect::<Result<Vec<_>>>()?;
        momenta.insert(
            carrier.clone(),
            FourMomentum::from_args(
                values[0].clone(),
                values[1].clone(),
                values[2].clone(),
                values[3].clone(),
            ),
        );
    }
    Ok(momenta)
}

fn polarization_carriers(
    prepared: &PreparedIntegrand,
    samples: &[Sample],
) -> Result<BTreeMap<usize, Atom>> {
    let Some(first) = samples.first() else {
        return Ok(BTreeMap::new());
    };
    let mut carriers = BTreeMap::new();
    for leg in first.external_polarizations.keys() {
        let (edge, _, data) = prepared
            .diagram
            .edges()
            .find(|(_, _, data)| {
                data.external
                    .as_ref()
                    .is_some_and(|external| external.index == *leg)
            })
            .ok_or_else(|| format!("unknown external polarization leg {leg}"))?;
        if prepared.diagram.model().particle_by_id(data.particle)?.spin != 3 {
            return Err(format!("external polarization leg {leg} is not a vector particle").into());
        }
        // Read the actual prepared projector carrier. Incoming/outgoing
        // conventions remain owned by FeynKit, including explicit projectors.
        let candidates: Vec<_> = [symbols::epsilon(), symbols::epsilonbar()]
            .into_iter()
            .filter(|symbol| {
                let pattern = symbol
                    .call((edge.0, Atom::var(symbol!("ninjars_cli_components::rep_"))))
                    .to_pattern();
                prepared
                    .numerator
                    .pattern_match(&pattern, None, None)
                    .next()
                    .is_some()
            })
            .map(|symbol| symbol.call(edge.0))
            .collect();
        if candidates.len() != 1 {
            return Err(format!("external polarization leg {leg} must have exactly one prepared epsilon carrier; found {}", candidates.len()).into());
        }
        carriers.insert(*leg, candidates[0].clone());
    }
    for (lane, sample) in samples.iter().enumerate() {
        if !sample.external_polarizations.keys().eq(carriers.keys()) {
            return Err(format!("sample {lane}: external_polarizations must bind the same physical legs in every sample").into());
        }
    }
    Ok(carriers)
}

pub(crate) fn prepare(args: &Args) -> Result<PreparedDocument> {
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
        if let Some(fixture) = &fixture
            && fs::canonicalize(base.join(&fixture.graph))? != fs::canonicalize(&args.input)?
        {
            return Err("sample fixture refers to a different graph than the DOT input".into());
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
        "sm" => Model::standard_model(),
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
    let mut prepared = prepare_dot(
        model.clone(),
        &dot,
        &Kinematics::new(),
        if args.feynman_rules {
            NumeratorMode::FeynmanRules
        } else {
            NumeratorMode::Explicit
        },
    )?;
    if fixture
        .as_ref()
        .is_some_and(|f| f.color_dimension_invariants)
    {
        prepared = prepared.with_color_dimension_invariants()?;
    }
    let polarizations = polarization_carriers(
        &prepared,
        fixture.as_ref().map_or(&[], |f| f.samples.as_slice()),
    )?;
    let carriers: Vec<_> = prepared
        .family
        .external_momenta()
        .iter()
        .cloned()
        .chain(polarizations.values().cloned())
        .collect();
    let loop_components = FourMomentum::from_args(
        parse("ninjars::q0")?,
        parse("ninjars::q1")?,
        parse("ninjars::q2")?,
        parse("ninjars::q3")?,
    );
    let mut samples = Vec::<Value>::new();
    let mut exact_samples = Vec::new();
    // These are evaluator parameters, not a new symbolic momentum representation.
    // Pick names absent from the bound integrand, including parameter overrides.
    let bound_numerator = substitute(&prepared.numerator, &replacements);
    let bound_factor = substitute(&prepared.overall_factor, &replacements);
    let mut generation = 0;
    let parameters = loop {
        let parameters: Vec<_> = (0..carriers.len() * 4)
            .map(|index| {
                Atom::var(symbol!(format!(
                    "ninjars_cli_components::p{generation}_{index}"
                )))
            })
            .collect();
        if parameters.iter().all(|parameter| {
            !bound_numerator.contains(parameter) && !bound_factor.contains(parameter)
        }) {
            break parameters;
        }
        generation += 1;
    };
    let symbolic_external = carriers
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
    let template_kinematics =
        prepared.component_kinematics(&loop_components, &symbolic_external)?;
    let numerator_template = substitute(
        &template_kinematics.apply(&(&prepared.numerator * &prepared.overall_factor)),
        &replacements,
    )
    .expand();
    if let Some(fixture) = &fixture {
        for (lane, sample) in fixture.samples.iter().enumerate() {
            let external = bind_external(sample, &prepared, &replacements, &polarizations)
                .map_err(|error| format!("sample {lane}: {error}"))?;
            let kinematics = prepared.component_kinematics(&loop_components, &external)?;
            let specialize = |atom: &Atom| substitute(&kinematics.apply(atom), &replacements);
            let numerator = specialize(&prepared.component_numerator(&loop_components, &external)?);
            let shifts = prepared
                .shifts(&external)?
                .iter()
                .map(|vector| {
                    let [e, x, y, z] = vector.components().map(&specialize);
                    FourMomentum::from_args(e, x, y, z)
                })
                .collect::<Vec<_>>();
            let mass_squared = prepared
                .squared_masses()
                .iter()
                .map(&specialize)
                .collect::<Vec<_>>();
            let overall_factor = specialize(&prepared.overall_factor);
            let s_matrix = sample.s_matrix.as_ref().map(|matrix| -> Result<Vec<Atom>> {
                let n = shifts.len();
                if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
                    return Err(format!("sample {lane}: s_matrix must contain exactly {n} rows of {n} entries in prepared propagator order").into());
                }
                matrix.iter().flatten().map(|expression| parse(expression).map(|atom| specialize(&atom))).collect()
            }).transpose()?;
            let exact = PreparedSample {
                lane,
                numerator,
                overall_factor,
                shifts,
                mass_squared,
                s_matrix,
                parameters: carriers
                    .iter()
                    .flat_map(|name| external[name].components().map(Clone::clone))
                    .collect(),
            };
            samples.push(json!({
                "lane": exact.lane,
                "numerator": canonical(&exact.numerator),
                "shifts": exact.shifts.iter().map(|vector| vector.components().map(canonical)).collect::<Vec<_>>(),
                "mass_squared": exact.mass_squared.iter().map(canonical).collect::<Vec<_>>(),
                "s_matrix": exact.s_matrix.as_ref().map(|matrix| matrix.chunks(exact.shifts.len()).map(|row| row.iter().map(canonical).collect::<Vec<_>>()).collect::<Vec<_>>()),
                "overall_factor": canonical(&exact.overall_factor),
            }));
            exact_samples.push(exact);
        }
    }
    let scale = parse(
        fixture
            .as_ref()
            .map(|f| f.scale_squared.as_str())
            .unwrap_or("1"),
    )?;
    let scale = substitute(&scale, &replacements);
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
        "scale_squared": canonical(&scale),
        "propagator_edges": prepared.propagator_edges.iter().map(|edge| edge.0).collect::<Vec<_>>(),
        "denominators": prepared.family.denominators().iter().map(|denominator| canonical(&substitute(denominator, &replacements))).collect::<Vec<_>>(),
        "external_basis": prepared.family.external_momenta().iter().map(canonical).collect::<Vec<_>>(),
        "polarization_basis": polarizations.iter().map(|(leg, carrier)| (leg.to_string(), canonical(carrier))).collect::<BTreeMap<_,_>>(),
        "momentum_basis": prepared.diagram.loop_momentum_basis(),
        "numerator": canonical(&substitute(&prepared.numerator, &replacements)),
        "overall_factor": canonical(&substitute(&prepared.overall_factor, &replacements)),
        "input_fixture": fixture,
        "samples": samples,
        "reference_export": null,
    });
    Ok(PreparedDocument {
        document: result,
        propagators: prepared.quadratics.len(),
        samples: exact_samples,
        scale_squared: scale,
        numerator_template,
        parameters,
    })
}

pub fn run(args: Args) -> Result<()> {
    let mut prepared = prepare(&args)?;
    #[cfg(feature = "reference-codegen")]
    if let Some(directory) = &args.reference_dir {
        prepared.document["reference_export"] =
            crate::reference::export(directory, &prepared.samples, &prepared.scale_squared)?;
    }
    // Keep this mutation available in builds without the optional reference exporter.
    prepared.document["operation"] = "prepare".into();
    let output = serde_json::to_string_pretty(&prepared.document)? + "\n";
    #[cfg(feature = "reference-codegen")]
    if let Some(directory) = args.reference_dir {
        fs::write(directory.join("prepared.json"), &output)?;
    }
    if let Some(path) = args.output {
        fs::write(path, output)?;
    } else {
        print!("{output}");
    }
    Ok(())
}
