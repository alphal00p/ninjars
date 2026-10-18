//! Topology-derived factors that FeynKit's generator records for generated
//! diagrams, for graphs written by hand without an `overall_factor`.

use std::collections::BTreeMap;

use feynkit_graph::{FeynmanDiagram, VertexId};
use symbolica::{atom::Atom, graph::Graph};

use crate::PreparationError;

/// Combinatorial factors of a diagram topology.
///
/// FeynKit's generator multiplies `1/AutG` and one `-1` per closed fermion or
/// ghost loop into a generated diagram's overall factor; an imported DOT keeps
/// whatever `overall_factor` it states. The generator also derives a sign per
/// diagram from the order of the external fermions along its open fermion
/// lines. Only the global sign of a process is a convention: relative signs
/// between diagrams with two or more open lines are physical (e.g. the s- and
/// t-channel diagrams of Bhabha scattering). That sign is not derived here, so
/// [`Self::open_fermion_lines`] is reported for callers to check.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphFactors {
    /// Order of the automorphism group with fixed, labelled external legs.
    pub automorphisms: u64,
    /// Closed loops of fermions or ghosts, each contributing `-1`.
    pub closed_grassmann_loops: usize,
    /// Open fermion lines between external legs. With two or more, the
    /// external-fermion ordering sign relative to other diagrams is not derived.
    pub open_fermion_lines: usize,
}

impl GraphFactors {
    /// `(-1)^closed_grassmann_loops / automorphisms`.
    pub fn factor(&self) -> Atom {
        let sign = if self.closed_grassmann_loops % 2 == 1 {
            -1
        } else {
            1
        };
        Atom::num(sign) / Atom::num(self.automorphisms)
    }
}

/// Derive the generator's automorphism order and closed-loop sign.
///
/// Vertices are distinguished by their interaction, external legs by their
/// index, and edges by their particle and fermion-flow orientation, as in
/// [`FeynmanDiagram::canonical_key`]. Symbolica counts multi-edge permutations
/// and undirected self-loop reversals, as it does during generation.
pub fn graph_factors(diagram: &FeynmanDiagram) -> Result<GraphFactors, PreparationError> {
    let model = diagram.model();
    let mut graph = Graph::<(bool, usize), usize>::new();
    let mut nodes = BTreeMap::new();
    for (vertex, data) in diagram.vertices() {
        let rule = data.interaction.map_or(usize::MAX, |rule| rule.index());
        nodes.insert(vertex, graph.add_node((false, rule)));
    }
    let mut grassmann = Vec::new();
    for (_, endpoints, edge) in diagram.edges() {
        let particle = model.particle_by_id(edge.particle)?;
        let leg = edge.external.as_ref().map_or(usize::MAX, |leg| leg.index);
        let mut node = |vertex: Option<VertexId>| match vertex {
            Some(vertex) => nodes[&vertex],
            None => graph.add_node((true, leg)),
        };
        let (mut source, mut target) = (node(endpoints.source), node(endpoints.target));
        let mut base = edge.particle;
        if edge.directed && particle.is_antiparticle() {
            std::mem::swap(&mut source, &mut target);
            base = particle.antiparticle;
        }
        graph
            .add_edge(source, target, edge.directed, base.index())
            .map_err(|error| PreparationError::Topology(error.into()))?;
        if particle.is_fermion() || particle.is_ghost() {
            grassmann.push((source, target));
        }
    }
    let automorphisms = graph.canonize().automorphism_group_size;
    let automorphisms = automorphisms.to_string().parse::<u64>().map_err(|_| {
        PreparationError::Topology(format!("automorphism order {automorphisms} exceeds u64"))
    })?;
    let (closed_grassmann_loops, open_fermion_lines) =
        grassmann_lines(graph.nodes().len(), &grassmann);
    Ok(GraphFactors {
        automorphisms,
        closed_grassmann_loops,
        open_fermion_lines,
    })
}

/// Closed and open connected components of the fermion and ghost edges: a
/// closed one has every vertex of degree two.
fn grassmann_lines(nodes: usize, edges: &[(usize, usize)]) -> (usize, usize) {
    let mut degrees = vec![0; nodes];
    let mut parent = (0..nodes).collect::<Vec<_>>();
    for &(a, b) in edges {
        degrees[a] += 1;
        degrees[b] += 1;
        let (a, b) = (find(&mut parent, a), find(&mut parent, b));
        parent[a] = b;
    }
    let mut open = BTreeMap::<usize, bool>::new();
    for node in (0..nodes).filter(|node| degrees[*node] != 0) {
        let root = find(&mut parent, node);
        *open.entry(root).or_default() |= degrees[node] != 2;
    }
    let closed = open.values().filter(|open| !**open).count();
    (closed, open.len() - closed)
}

fn find(parent: &mut [usize], mut node: usize) -> usize {
    while parent[node] != node {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    node
}

/// Number of connected components formed by the vertices and internal edges.
pub fn connected_components(diagram: &FeynmanDiagram) -> usize {
    let vertices = diagram
        .vertices()
        .enumerate()
        .map(|(position, (vertex, _))| (vertex, position))
        .collect::<BTreeMap<_, _>>();
    let mut parent = (0..vertices.len()).collect::<Vec<_>>();
    for (_, endpoints, _) in diagram.edges() {
        if let (Some(source), Some(target)) = (endpoints.source, endpoints.target) {
            let (a, b) = (
                find(&mut parent, vertices[&source]),
                find(&mut parent, vertices[&target]),
            );
            parent[a] = b;
        }
    }
    (0..vertices.len())
        .filter(|&vertex| find(&mut parent, vertex) == vertex)
        .count()
}

/// What a DOT source states about itself, beyond FeynKit's import.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DotAttributes {
    /// FeynKit's compact, hand-writable dialect rather than its stable export.
    pub compact: bool,
    /// Any graph, vertex or edge numerator attribute.
    pub numerator: bool,
    pub overall_factor: bool,
}

/// Read [`DotAttributes`] with FeynKit's DOT parser.
pub fn dot_attributes(dot: &str) -> Result<DotAttributes, PreparationError> {
    let parsed: linnet::parser::DotGraph = linnet::parser::DotGraph::from_string(dot)
        .map_err(|error| PreparationError::Topology(format!("invalid DOT: {error}")))?;
    let global = &parsed.global_data.statements;
    // The dialect test of `FeynmanDiagram::from_dot`.
    let compact = ![
        "model_fingerprint",
        "feynkit_name",
        "half_edge_order",
        "loop_momentum_basis",
        "cuts",
        "topology_threshold_candidates",
    ]
    .iter()
    .any(|key| global.contains_key(*key))
        && !parsed
            .iter_edges()
            .any(|(_, _, edge)| edge.data.statements.contains_key("particle_id"));
    let key = if compact { "num" } else { "numerator" };
    let numerator = global.contains_key(key)
        || parsed
            .iter_nodes()
            .any(|(_, _, node)| node.statements.contains_key(key))
        || parsed
            .iter_edges()
            .any(|(_, _, edge)| edge.data.statements.contains_key(key));
    Ok(DotAttributes {
        compact,
        numerator,
        overall_factor: global.contains_key("overall_factor"),
    })
}
