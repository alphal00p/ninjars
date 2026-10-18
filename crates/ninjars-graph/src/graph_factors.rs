//! Topology-derived factors that FeynKit's generator records for generated
//! diagrams, for graphs written by hand without an `overall    let automorphisms = automorphisms.to_string().parse::<u64>().map_err(|_| {factor`.

use std::collections::BTreeMap;

use feynkit_graph::{FeynmanDiagram, VertexId};
use symbolica::{atom::Atom, graph::Graph};

use crate::PreparationError;

/// Combinatorial factors of a diagram topology.
///
/// FeynKit's generator multiplies `1/AutG` and one `-1` per closed fermion or
/// ghost loop into a generated diagram's overall factor; an imported DOT keeps
/// whatever `overall_factor` it states. External-fermion ordering signs are a
/// convention of the whole process and are not derived per diagram.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphFactors {
    /// Order of the automorphism group with fixed, labelled external legs.
    pub automorphisms: u64,
    /// Closed loops of fermions or ghosts, each contributing `-1`.
    pub closed_grassmann_loops: usize,
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
    Ok(GraphFactors {
        automorphisms,
        closed_grassmann_loops: closed_loops(graph.nodes().len(), &grassmann),
    })
}

/// Count connected components whose every vertex has degree two.
fn closed_loops(nodes: usize, edges: &[(usize, usize)]) -> usize {
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
    open.values().filter(|open| !**open).count()
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
