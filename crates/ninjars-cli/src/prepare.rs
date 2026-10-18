use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    error::Error,
    fs, io,
    path::{Path, PathBuf},
    sync::Arc,
};

use clap::{Args as ClapArgs, ValueEnum};
use feynkit_graph::{DiagramEndpoint, DiagramHalfEdge};
use ninjars_graph::{
    ExternalState, FeynmanDiagram, FourMomentum, Kinematics, Model, NumeratorMode,
    PreparationOptions, PreparedIntegrand, R2Scheme, dot_attributes, edge_width,
    expand_spinor_chains, has_spinor_chains, prepare_dot, symbols,
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
    /// Optional fixture label of the phase-space point.
    pub id: Option<String>,
    pub numerator: Atom,
    pub overall_factor: Atom,
    pub shifts: Vec<FourMomentum<Atom>>,
    pub mass_squared: Vec<Atom>,
    /// Optional row-major n×n matrix in prepared propagator order.
    pub s_matrix: Option<Vec<Atom>>,
    pub parameters: Vec<Atom>,
}

/// Whether the physical numerator carries the `mu^2` terms that give R2.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum R2 {
    /// Four-dimensional loop momentum, `K.K -> q.q`: the reduction gives R1 but no R2.
    #[default]
    Off,
    /// FDH: d_s = 4 internal algebra with a d-dimensional loop momentum,
    /// `K.K -> q.q - mu^2` (`K.p`, `K.eps` unchanged); the mu^2 terms give R2 in FDH.
    Fdh,
}

impl R2 {
    pub(crate) fn scheme(self) -> R2Scheme {
        match self {
            Self::Off => R2Scheme::Off,
            Self::Fdh => R2Scheme::Fdh,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        self.scheme().name()
    }

    /// Prepared-document metadata of the selected scheme.
    pub(crate) fn metadata(self) -> Value {
        match self {
            Self::Off => {
                json!({"mode":"off","loop_momentum":"four-dimensional: K.K -> q0^2-q1^2-q2^2-q3^2","rational_terms":"R1 from the d-dimensional denominators; no R2"})
            }
            Self::Fdh => {
                json!({"mode":"fdh","scheme":"FDH: d_s = 4 internal algebra, d-dimensional loop momentum","loop_momentum":"K.K -> q0^2-q1^2-q2^2-q3^2-mu2; K.P and K.eps four-dimensional","rational_terms":"R1 from the denominators and R2 (FDH) from the numerator mu^2 terms"})
            }
        }
    }
}

pub(crate) struct PreparedDocument {
    pub document: Value,
    pub propagators: usize,
    pub samples: Vec<PreparedSample>,
    pub scale_squared: Atom,
    pub numerator_template: Atom,
    pub parameters: Vec<Atom>,
}

/// Model and integrand options shared by every command that prepares a graph.
#[derive(ClapArgs, Clone, Default)]
pub struct GraphOptions {
    /// Built-in model (phi3, phi4, sm, qcd, qed, qcd_qed, yang_mills,
    /// electroweak), or a FeynKit model JSON path. Defaults to the fixture
    /// model, else phi3.
    #[arg(long)]
    pub model: Option<String>,
    /// Construct local numerators and default external states from model rules.
    /// A hand-written (compact) DOT without `overall_factor` also receives the
    /// generator's symmetry factor and closed fermion/ghost-loop sign.
    #[arg(long)]
    pub feynman_rules: bool,
    /// Evaluate fundamental color invariants such as cas(2,coad(8)) with
    /// Idenso's dimension formulas, like the fixture's `color_dimension_invariants`.
    #[arg(long)]
    pub color_dimension_invariants: bool,
    /// Use m² − i·m·Γ in every propagator whose particle has a nonzero model
    /// width. Numerators and couplings keep their real masses.
    #[arg(long)]
    pub complex_mass_propagators: bool,
    /// R2 from the numerator: `off` binds the loop momentum four-dimensionally
    /// (no mu^2 terms, R1 only); `fdh` binds K.K -> q.q - mu^2 (FDH: d_s = 4
    /// algebra, d-dimensional loop momentum), so the mu^2 terms give R2.
    #[arg(long, value_enum, default_value = "off")]
    pub r2: R2,
}

#[derive(ClapArgs)]
pub struct Args {
    /// DOT graph, or a schema-version-1 JSON fixture resolving its own graph path.
    pub input: PathBuf,
    /// Sample fixture accompanying a DOT input.
    #[arg(long)]
    pub samples: Option<PathBuf>,
    #[command(flatten)]
    pub graph: GraphOptions,
    /// Write JSON to a file; default writes JSON to standard output.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Export per-sample C++ callbacks and binary64 inputs for original Ninja validation.
    #[cfg(feature = "reference-codegen")]
    #[arg(long)]
    pub reference_dir: Option<PathBuf>,
    /// Export one parameterized callback and a numerical pool for reference benchmarks.
    #[cfg(feature = "reference-codegen")]
    #[arg(long, conflicts_with = "reference_dir")]
    pub reference_bench_dir: Option<PathBuf>,
}

impl Args {
    /// Preparation of `input` alone, as every reduction command performs it.
    pub(crate) fn for_input(input: PathBuf, samples: Option<PathBuf>, graph: GraphOptions) -> Self {
        Self {
            input,
            samples,
            graph,
            output: None,
            #[cfg(feature = "reference-codegen")]
            reference_dir: None,
            #[cfg(feature = "reference-codegen")]
            reference_bench_dir: None,
        }
    }
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
    /// Optional label, e.g. the id of a shared phase-space pool point.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    external_momenta: BTreeMap<usize, [String; 4]>,
    /// Exact polarization components, keyed by the graph's physical leg index.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    external_polarizations: BTreeMap<usize, [String; 4]>,
    /// Exact chiral-basis components of each external spinor carrier, keyed
    /// by physical leg index: columns for u and v, rows for ubar and vbar.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    external_spinors: BTreeMap<usize, [String; 4]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    s_matrix: Option<Vec<Vec<String>>>,
}

fn one() -> String {
    "1".into()
}

/// Built-in models accepted by `--model` and a fixture's `model`.
pub(crate) const BUILTIN_MODELS: [&str; 8] = [
    "phi3",
    "phi4",
    "sm",
    "qcd",
    "qed",
    "qcd_qed",
    "yang_mills",
    "electroweak",
];

/// A built-in model by name, else the FeynKit model JSON at `base/name`.
pub(crate) fn resolve_model(name: &str, base: &Path) -> Result<Model> {
    Ok(match name {
        "phi3" => Model::phi3(),
        "phi4" => Model::phi4(),
        "sm" => Model::standard_model(),
        "qcd" => Model::qcd(),
        "qed" => Model::qed(),
        "qcd_qed" => Model::qcd_qed(),
        "yang_mills" => Model::yang_mills(),
        "electroweak" => Model::electroweak(),
        file => {
            let path = base.join(file);
            let text = fs::read_to_string(&path).map_err(|error| -> Box<dyn Error> {
                if error.kind() == io::ErrorKind::NotFound {
                    format!(
                        "unknown model {file:?}: not a built-in model ({}) and no model file at {}",
                        BUILTIN_MODELS.join(", "),
                        path.display()
                    )
                    .into()
                } else {
                    format!("cannot read model file {}: {error}", path.display()).into()
                }
            })?;
            Model::from_json(&text)
                .map_err(|error| format!("invalid model file {}: {error}", path.display()))?
        }
    })
}

fn read(path: &Path, what: &str) -> Result<String> {
    fs::read_to_string(path)
        .map_err(|error| format!("cannot read {what} {}: {error}", path.display()).into())
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

fn components(values: &[String; 4], bindings: &[Replacement]) -> Result<[Atom; 4]> {
    let [e, x, y, z] = values
        .each_ref()
        .map(|text| parse(text).map(|atom| substitute(&atom, bindings)));
    Ok([e?, x?, y?, z?])
}

/// One external leg in both of the numberings used by sample fixtures.
struct Leg {
    /// Physical leg index: the key of `external_polarizations` and `external_spinors`.
    index: usize,
    /// Position in the momentum basis: the key `i` of `external_momenta`, `P(i)`.
    momentum: usize,
    dependent: bool,
    edge: usize,
    /// Half-edge at the leg's vertex, as used by open indices and projectors.
    half_edge: usize,
    particle: String,
    spin: i64,
    state: ExternalState,
}

fn external_legs(diagram: &FeynmanDiagram) -> Result<Vec<Leg>> {
    let basis = diagram.loop_momentum_basis();
    let mut legs = diagram
        .edges()
        .filter_map(|(edge, endpoints, data)| {
            let external = data.external.as_ref()?;
            Some((edge, endpoints, data, external))
        })
        .map(|(edge, endpoints, data, external)| -> Result<Leg> {
            let endpoint = if endpoints.source.is_some() {
                DiagramEndpoint::Source
            } else {
                DiagramEndpoint::Target
            };
            let particle = diagram.model().particle_by_id(data.particle)?;
            Ok(Leg {
                index: external.index,
                momentum: basis
                    .external_edges
                    .iter()
                    .position(|candidate| *candidate == edge)
                    .ok_or_else(|| {
                        format!("external edge {} is not in the momentum basis", edge.0)
                    })?,
                dependent: basis.dependent_externals.contains(&edge),
                edge: edge.0,
                half_edge: diagram
                    .half_edge_id(DiagramHalfEdge { edge, endpoint })
                    .ok_or_else(|| format!("external edge {} has no vertex half-edge", edge.0))?
                    .0,
                particle: particle.name.clone(),
                spin: particle.spin,
                state: external.state,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    legs.sort_by_key(|leg| leg.index);
    Ok(legs)
}

fn leg_json(leg: &Leg) -> Value {
    json!({
        "leg": leg.index,
        "momentum_key": leg.momentum,
        "dependent": leg.dependent,
        "edge": leg.edge,
        "half_edge": leg.half_edge,
        "particle": leg.particle,
        "state": match leg.state { ExternalState::Incoming => "incoming", ExternalState::Outgoing => "outgoing" },
    })
}

fn bind_external(
    sample: &Sample,
    prepared: &PreparedIntegrand,
    bindings: &[Replacement],
    polarizations: &BTreeMap<usize, Atom>,
) -> Result<BTreeMap<Atom, FourMomentum<Atom>>> {
    let count = prepared.diagram.loop_momentum_basis().external_edges.len();
    let mut momenta = BTreeMap::new();
    for (index, values) in &sample.external_momenta {
        if *index >= count {
            return Err(format!("external momentum index {index} is outside 0..{count}").into());
        }
        momenta.insert(
            symbols::external_momentum().call(*index),
            FourMomentum::from(components(values, bindings)?),
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
        momenta.insert(
            carrier.clone(),
            FourMomentum::from(components(&sample.external_polarizations[leg], bindings)?),
        );
    }
    Ok(momenta)
}

/// Whether the numerator contains `head(edge, rep)` for a rank-one carrier head.
fn contains_carrier(numerator: &Atom, head: Symbol, edge: usize) -> bool {
    let pattern = head
        .call((edge, Atom::var(symbol!("ninjars_cli_components::rep_"))))
        .to_pattern();
    numerator
        .pattern_match(&pattern, None, None)
        .next()
        .is_some()
}

/// Rank-one external-state carriers bound by the samples, by physical leg.
///
/// `keys` selects the sample field; FeynKit owns the incoming/outgoing
/// conventions, including those of explicit projectors, so the actual
/// prepared carrier is read from the numerator.
fn state_carriers(
    prepared: &PreparedIntegrand,
    legs: &[Leg],
    samples: &[Sample],
    kind: StateKind,
) -> Result<BTreeMap<usize, Atom>> {
    let keys = |sample: &Sample| -> Vec<usize> {
        match kind {
            StateKind::Polarization => sample.external_polarizations.keys().copied().collect(),
            StateKind::Spinor => sample.external_spinors.keys().copied().collect(),
        }
    };
    let Some(first) = samples.first() else {
        return Ok(BTreeMap::new());
    };
    let mut carriers = BTreeMap::new();
    for leg in keys(first) {
        let record = legs
            .iter()
            .find(|candidate| candidate.index == leg)
            .ok_or_else(|| format!("unknown external {} leg {leg}", kind.name()))?;
        let heads = match kind {
            StateKind::Polarization if record.spin == 3 => {
                vec![symbols::epsilon(), symbols::epsilonbar()]
            }
            StateKind::Polarization => {
                return Err(
                    format!("external polarization leg {leg} is not a vector particle").into(),
                );
            }
            StateKind::Spinor if record.spin == 2 => {
                vec![symbols::u(), symbols::ubar(), symbols::v(), symbols::vbar()]
            }
            StateKind::Spinor => {
                return Err(format!("external spinor leg {leg} is not a spin-1/2 fermion").into());
            }
        };
        let candidates: Vec<_> = heads
            .into_iter()
            .filter(|head| contains_carrier(&prepared.numerator, *head, record.edge))
            .map(|head| head.call(record.edge))
            .collect();
        if candidates.len() != 1 {
            return Err(format!(
                "external {} leg {leg} must have exactly one prepared {} carrier; found {}",
                kind.name(),
                kind.carrier(),
                candidates.len()
            )
            .into());
        }
        carriers.insert(leg, candidates[0].clone());
    }
    for (lane, sample) in samples.iter().enumerate() {
        if !keys(sample).iter().eq(carriers.keys()) {
            return Err(format!(
                "sample {lane}: {} must bind the same physical legs in every sample",
                kind.field()
            )
            .into());
        }
    }
    Ok(carriers)
}

#[derive(Clone, Copy)]
enum StateKind {
    Polarization,
    Spinor,
}

impl StateKind {
    fn name(self) -> &'static str {
        match self {
            Self::Polarization => "polarization",
            Self::Spinor => "spinor",
        }
    }
    fn carrier(self) -> &'static str {
        match self {
            Self::Polarization => "epsilon",
            Self::Spinor => "u, ubar, v or vbar",
        }
    }
    fn field(self) -> &'static str {
        match self {
            Self::Polarization => "external_polarizations",
            Self::Spinor => "external_spinors",
        }
    }
}

/// Explain the symbols a numerator template still needs, or nothing.
///
/// A bound template registers no Symbolica symbol here: evaluators built
/// later order their operations by symbol identity, so a new symbol would
/// perturb results at round-off level. Loop variables are matched by name.
fn unbound_symbols(
    template: &Atom,
    parameters: &[Atom],
    legs: &[Leg],
    polarizations: &BTreeMap<usize, Atom>,
    spinors: &BTreeMap<usize, Atom>,
) -> Vec<String> {
    let allowed = parameters
        .iter()
        .filter_map(|atom| match atom.as_view() {
            AtomView::Var(variable) => Some(variable.get_symbol()),
            _ => None,
        })
        .collect::<HashSet<_>>();
    let loop_variables = ["q0", "q1", "q2", "q3", "mu2"].map(|name| format!("ninjars::{name}"));
    let symbols = template
        .get_all_symbols(true)
        .into_iter()
        .filter(|symbol| {
            !symbol.is_builtin()
                && !allowed.contains(symbol)
                && !loop_variables.iter().any(|name| name == symbol.get_name())
        })
        .collect::<BTreeSet<_>>();
    if symbols.is_empty() {
        return Vec::new();
    }
    let named = |names: &[Symbol]| names.iter().any(|name| symbols.contains(name));
    let unlisted = |spin: i64, bound: &BTreeMap<usize, Atom>| {
        legs.iter()
            .filter(|leg| leg.spin == spin && !bound.contains_key(&leg.index))
            .map(|leg| format!("{} ({})", leg.index, leg.particle))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut messages = Vec::new();
    let spinor_heads = [symbols::u(), symbols::ubar(), symbols::v(), symbols::vbar()];
    if named(&spinor_heads) {
        let fermions = unlisted(2, spinors);
        messages.push(if fermions.is_empty() {
            // Every leg is bound: name the structure the expansion left.
            let mut leftover = None;
            template.visitor(&mut |part| {
                if leftover.is_none()
                    && let AtomView::Fun(function) = part
                    && function.iter().any(|argument| {
                        matches!(argument, AtomView::Fun(carrier) if spinor_heads.contains(&carrier.get_symbol()))
                    })
                {
                    leftover = Some(part.to_canonical_string());
                }
                leftover.is_none()
            });
            let mut leftover = leftover.unwrap_or_default();
            if leftover.len() > 300 {
                let end = (0..=300).rev().find(|end| leftover.is_char_boundary(*end)).unwrap_or(0);
                leftover.truncate(end);
                leftover.push_str("...");
            }
            format!("explicit spinors cannot expand the spinor structure {leftover}")
        } else {
            format!(
                "external fermion spinors are unbound: give every sample `external_spinors` for legs {fermions} \
                 (four chiral-basis components of each carrier), or close the fermion lines with a spin-trace `projector`"
            )
        });
    }
    let vectors = unlisted(3, polarizations);
    if named(&[symbols::epsilon(), symbols::epsilonbar()]) && !vectors.is_empty() {
        messages.push(format!(
            "external polarizations are unbound: give every sample `external_polarizations` for legs {vectors}"
        ));
    }
    let rest = symbols
        .iter()
        .filter(|symbol| {
            ![
                symbols::u(),
                symbols::ubar(),
                symbols::v(),
                symbols::vbar(),
                symbols::epsilon(),
                symbols::epsilonbar(),
            ]
            .contains(symbol)
        })
        .map(|symbol| symbol.get_name())
        .collect::<Vec<_>>();
    if rest.contains(&"spenso::cas") || rest.contains(&"spenso::idx") {
        messages.push(
            "color invariants such as cas(2,coad(8)) or idx(2,cof(3)) remain: pass \
             --color-dimension-invariants (or set the fixture's `color_dimension_invariants`)"
                .into(),
        );
    }
    if rest.contains(&"spenso::epsilon") {
        messages.push(
            "Levi-Civita tensors (from gamma5 traces) remain; they have no component binding"
                .into(),
        );
    }
    // Vector, spinor-chain and representation heads are implied by the
    // messages above.
    let implied = [
        "spenso::cas",
        "spenso::idx",
        "spenso::epsilon",
        "spenso::dot",
        "spenso::chain",
        "spenso::in",
        "spenso::out",
        "spenso::gamma",
        "spenso::mink",
        "spenso::bis",
        "spenso::coad",
        "spenso::cof",
        "spenso::cos",
        "spenso::dind",
    ];
    let other = rest
        .iter()
        .filter(|name| !implied.contains(name) && !name.ends_with("::P") && !name.ends_with("::K"))
        .copied()
        .collect::<Vec<_>>();
    if !other.is_empty() || messages.is_empty() {
        messages.push(format!(
            "unbound symbols remain: {}",
            if other.is_empty() {
                rest.join(", ")
            } else {
                other.join(", ")
            }
        ));
    }
    messages
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
            let fixture: Fixture = serde_json::from_str(&read(path, "sample fixture")?)
                .map_err(|error| format!("invalid sample fixture {}: {error}", path.display()))?;
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
            let canonical = |path: &Path| {
                fs::canonicalize(path)
                    .map_err(|error| format!("cannot read DOT graph {}: {error}", path.display()))
            };
            if canonical(&base.join(&fixture.graph))? != canonical(&args.input)? {
                return Err("sample fixture refers to a different graph than the DOT input".into());
            }
        }
        args.input.clone()
    };
    let options = &args.graph;
    let model_name = options
        .model
        .as_deref()
        .or_else(|| fixture.as_ref().and_then(|f| f.model.as_deref()))
        .unwrap_or("phi3");
    // A command-line model path is relative to the working directory, a
    // fixture's to the fixture.
    let model = Arc::new(resolve_model(
        model_name,
        if options.model.is_some() {
            Path::new(".")
        } else {
            base
        },
    )?);
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
    let dot = read(&graph_path, "DOT graph")?;
    let mode = if options.feynman_rules {
        NumeratorMode::FeynmanRules
    } else {
        NumeratorMode::Explicit
    };
    // FeynKit reports malformed DOT itself, during preparation.
    let attributes = dot_attributes(&dot).ok();
    if mode == NumeratorMode::Explicit
        && attributes.is_some_and(|attributes| attributes.compact && !attributes.numerator)
    {
        return Err(format!(
            "{} has no `num` attribute, so explicit mode would use the numerator 1; pass \
             --feynman-rules for the model's Feynman rules, or state the numerator with `num` \
             (num=\"1\" on the graph for a scalar integrand)",
            graph_path.display()
        )
        .into());
    }
    // `prepare_dot` applies the generator's graph factors to a compact DOT
    // prepared with Feynman rules and without an overall_factor.
    let mut prepared = prepare_dot(
        model.clone(),
        &dot,
        &Kinematics::new(),
        PreparationOptions {
            numerator: mode,
            complex_mass_propagators: options.complex_mass_propagators,
        },
    )?;
    let factors = prepared.graph_factors;
    let compact_without_factor =
        attributes.is_some_and(|attributes| attributes.compact && !attributes.overall_factor);
    if compact_without_factor && !prepared.graph_factors_applied && !factors.factor().is_one() {
        eprintln!(
            "ninjars: note: {} states no overall_factor; its topology has automorphism order {} \
             and {} closed fermion/ghost loops (factor {}). Explicit mode applies only stated \
             factors: set overall_factor if the numerator does not already include them.",
            graph_path.display(),
            factors.automorphisms,
            factors.closed_grassmann_loops,
            canonical(&factors.factor())
        );
    }
    if compact_without_factor && factors.open_fermion_lines > 1 {
        eprintln!(
            "ninjars: warning: {} has {} open fermion lines; the external-fermion ordering sign \
             relative to other diagrams of the process is not derived: set overall_factor",
            graph_path.display(),
            factors.open_fermion_lines
        );
    }
    let color_dimension_invariants = options.color_dimension_invariants
        || fixture
            .as_ref()
            .is_some_and(|f| f.color_dimension_invariants);
    if color_dimension_invariants {
        prepared = prepared.with_color_dimension_invariants()?;
    }
    let mut widths = Vec::new();
    for (edge, _, data) in prepared.diagram.edges() {
        if data.external.is_some() {
            continue;
        }
        if let Some((_, width)) = edge_width(&prepared.diagram, edge)? {
            let value = substitute(&width, &replacements);
            if !value.is_zero() {
                widths.push(json!({
                    "edge": edge.0,
                    "particle": prepared.diagram.model().particle_by_id(data.particle)?.name,
                    "width": canonical(&width),
                    "value": canonical(&value),
                }));
            }
        }
    }
    if !widths.is_empty() && !options.complex_mass_propagators {
        let names = widths
            .iter()
            .map(|width| {
                format!(
                    "{}={}",
                    width["width"].as_str().unwrap(),
                    width["value"].as_str().unwrap()
                )
            })
            .collect::<BTreeSet<_>>();
        eprintln!(
            "ninjars: warning: propagator widths {} are ignored; pass --complex-mass-propagators \
             to use m^2 - i*m*width",
            names.into_iter().collect::<Vec<_>>().join(", ")
        );
    }
    let legs = external_legs(&prepared.diagram)?;
    let samples_in = fixture.as_ref().map_or(&[][..], |f| f.samples.as_slice());
    let polarizations = state_carriers(&prepared, &legs, samples_in, StateKind::Polarization)?;
    let spinors = state_carriers(&prepared, &legs, samples_in, StateKind::Spinor)?;
    let carriers: Vec<_> = prepared
        .family
        .external_momenta()
        .iter()
        .cloned()
        .chain(polarizations.values().cloned())
        .chain(spinors.values().cloned())
        .collect();
    let spinor_carriers = spinors.values().cloned().collect::<BTreeSet<_>>();
    let loop_components = FourMomentum::from_args(
        parse("ninjars::q0")?,
        parse("ninjars::q1")?,
        parse("ninjars::q2")?,
        parse("ninjars::q3")?,
    );
    // Create the mu^2 symbol only when it is bound: registering it earlier
    // would change Symbolica's symbol order, and thereby the evaluation order
    // and rounding of every `--r2 off` result.
    let mu_squared = match options.r2 {
        R2::Off => None,
        R2::Fdh => Some(parse("ninjars::mu2")?),
    };
    let mu_squared = mu_squared.as_ref();
    let loop_momentum = prepared.family.loop_momenta()[0].clone();
    // Expand spinor chains once every external fermion has bound components;
    // otherwise they remain and are reported as unbound below.
    let chains = has_spinor_chains(&prepared.numerator)
        && legs
            .iter()
            .all(|leg| leg.spin != 2 || spinors.contains_key(&leg.index));
    // Explicit spinor components for the chains, from the same vectors.
    let expand = |numerator: &Atom,
                  vectors: &BTreeMap<Atom, FourMomentum<Atom>>,
                  spinor_values: &BTreeMap<Atom, [Atom; 4]>|
     -> Result<Atom> {
        let mut vectors = vectors.clone();
        vectors.insert(loop_momentum.clone(), loop_components.clone());
        Ok(expand_spinor_chains(numerator, &vectors, spinor_values)?)
    };
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
    let mut symbolic_external = BTreeMap::new();
    let mut symbolic_spinors = BTreeMap::new();
    for (name, chunk) in carriers.iter().zip(parameters.as_chunks::<4>().0) {
        if spinor_carriers.contains(name) {
            symbolic_spinors.insert(name.clone(), chunk.clone());
        } else {
            symbolic_external.insert(name.clone(), FourMomentum::from(chunk.clone()));
        }
    }
    let template_kinematics =
        prepared.dimensional_kinematics(&loop_components, mu_squared, &symbolic_external)?;
    // FDH rejects external spinors in `dimensional_kinematics` above, so
    // chains are only expanded with a four-dimensional loop momentum.
    let numerator = if chains {
        expand(&prepared.numerator, &symbolic_external, &symbolic_spinors)?
    } else {
        prepared.numerator.clone()
    };
    let numerator_template = substitute(
        &template_kinematics.apply(&(&numerator * &prepared.overall_factor)),
        &replacements,
    )
    .expand();
    let unbound = unbound_symbols(
        &numerator_template,
        &parameters,
        &legs,
        &polarizations,
        &spinors,
    );
    if !unbound.is_empty() {
        let message = unbound.join("; ");
        if samples_in.is_empty() {
            eprintln!(
                "ninjars: warning: the prepared numerator cannot be evaluated yet: {message}"
            );
        } else {
            return Err(format!("the prepared numerator cannot be evaluated: {message}").into());
        }
    }
    let momentum_keys_differ = legs.iter().any(|leg| leg.index != leg.momentum);
    if momentum_keys_differ && (!polarizations.is_empty() || !spinors.is_empty()) {
        eprintln!(
            "ninjars: warning: in this graph, external_momenta keys (momentum-basis positions) and \
             external_polarizations/external_spinors keys (physical legs) differ: {}",
            legs.iter()
                .filter(|leg| leg.index != leg.momentum)
                .map(|leg| format!("leg {} has momentum P({})", leg.index, leg.momentum))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if let Some(fixture) = &fixture {
        for (lane, sample) in fixture.samples.iter().enumerate() {
            let external = bind_external(sample, &prepared, &replacements, &polarizations)
                .map_err(|error| format!("sample {lane}: {error}"))?;
            let spinor_values = spinors
                .iter()
                .map(|(leg, carrier)| {
                    Ok((
                        carrier.clone(),
                        components(&sample.external_spinors[leg], &replacements)?,
                    ))
                })
                .collect::<Result<BTreeMap<_, _>>>()
                .map_err(|error| format!("sample {lane}: {error}"))?;
            let kinematics =
                prepared.dimensional_kinematics(&loop_components, mu_squared, &external)?;
            let specialize = |atom: &Atom| substitute(&kinematics.apply(atom), &replacements);
            let numerator = if chains {
                // The model-bound numerator with exact sample components needs
                // no further normalization: a rational-function `together` of
                // the expanded chains would dominate preparation.
                kinematics
                    .apply(&expand(&bound_numerator, &external, &spinor_values)?)
                    .expand()
            } else {
                specialize(&prepared.dimensional_numerator(
                    &loop_components,
                    mu_squared,
                    &external,
                )?)
            };
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
                id: sample.id.clone(),
                numerator,
                overall_factor,
                shifts,
                mass_squared,
                s_matrix,
                parameters: carriers
                    .iter()
                    .flat_map(|name| match external.get(name) {
                        Some(vector) => vector.components().map(Clone::clone),
                        None => spinor_values[name].clone(),
                    })
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
        "numerator_mode": if options.feynman_rules { "feynman_rules" } else { "explicit" },
        "r2": options.r2.metadata(),
        "metric": "+---",
        "scale_squared": canonical(&scale),
        "propagator_edges": prepared.propagator_edges.iter().map(|edge| edge.0).collect::<Vec<_>>(),
        "denominators": prepared.family.denominators().iter().map(|denominator| canonical(&substitute(denominator, &replacements))).collect::<Vec<_>>(),
        "propagator_widths": {
            "mode": if options.complex_mass_propagators { "complex_mass" } else { "ignored" },
            "nonzero": widths,
        },
        "external_basis": prepared.family.external_momenta().iter().map(canonical).collect::<Vec<_>>(),
        "external_legs": legs.iter().map(leg_json).collect::<Vec<_>>(),
        "polarization_basis": polarizations.iter().map(|(leg, carrier)| (leg.to_string(), canonical(carrier))).collect::<BTreeMap<_,_>>(),
        "spinor_basis": spinors.iter().map(|(leg, carrier)| (leg.to_string(), canonical(carrier))).collect::<BTreeMap<_,_>>(),
        "momentum_basis": prepared.diagram.loop_momentum_basis(),
        "graph_factors": {
            "automorphisms": factors.automorphisms,
            "closed_fermion_loops": factors.closed_grassmann_loops,
            "open_fermion_lines": factors.open_fermion_lines,
            "factor": canonical(&factors.factor()),
            "applied": prepared.graph_factors_applied,
        },
        "color_dimension_invariants": color_dimension_invariants,
        "color_projector": prepared.color_projector.as_ref().map(canonical),
        "numerator": canonical(&substitute(&prepared.numerator, &replacements)),
        "overall_factor": canonical(&substitute(&prepared.overall_factor, &replacements)),
        "unbound": unbound,
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
    #[cfg(feature = "reference-codegen")]
    if let Some(directory) = &args.reference_bench_dir {
        prepared.document["reference_bench_export"] =
            crate::reference_bench::export(directory, &prepared)?;
    }
    // Keep this mutation available in builds without the optional reference exporter.
    prepared.document["operation"] = "prepare".into();
    let output = serde_json::to_string_pretty(&prepared.document)? + "\n";
    #[cfg(feature = "reference-codegen")]
    if let Some(directory) = args.reference_dir {
        fs::write(directory.join("prepared.json"), &output)?;
    }
    #[cfg(feature = "reference-codegen")]
    if let Some(directory) = args.reference_bench_dir {
        fs::write(directory.join("prepared.json"), &output)?;
    }
    if let Some(path) = args.output {
        fs::write(path, output)?;
    } else {
        print!("{output}");
    }
    Ok(())
}
