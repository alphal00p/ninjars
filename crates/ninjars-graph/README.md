# FeynKit input preparation for NinjaRS

This crate prepares one-loop graph inputs without depending on the reduction
engine. It uses FeynKit's DOT parser, routing, propagator families and momentum
types, and Idenso/Symbolica for numerator contraction and exact algebra.

```rust
use ninjars_graph::{Kinematics, Model, NumeratorMode, prepare_dot};

let dot = std::fs::read_to_string("examples/graphs/bubble.dot")?;
let prepared = prepare_dot(Model::phi3(), &dot, &Kinematics::new(), NumeratorMode::Explicit)?;
assert_eq!(prepared.family.denominators().len(), 2);
assert_eq!(prepared.quadratics.len(), 2);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Run the integration checks from the repository root:

```console
nix develop --command cargo test -p ninjars-graph -- --test-threads=1
```

The standalone preparation command accepts the supplied graph/sample fixtures:

```console
nix develop --command cargo run --locked -p ninjars-cli -- prepare examples/graphs/bubble.json
nix develop --command cargo run --locked -p ninjars-cli -- prepare examples/graphs/triangle.dot --samples examples/graphs/triangle.json
```

Add `--output prepared.json` to save the result, `--model` to select a built-in
model or a FeynKit model JSON, or `--feynman-rules` to derive local factors. JSON
contains canonical Symbolica expressions, original inputs, graph/model metadata,
propagator edge order, and exact per-sample shifts, squared masses and numerators.

`prepare_dot` accepts compact or stable FeynKit DOT. `NumeratorMode::Explicit`
preserves graph/vertex/edge `num` attributes. `NumeratorMode::FeynmanRules`
replaces local factors using the model; it supplies external wavefunctions if
the stored projector is one and retains any nontrivial explicit projector.
Models can also be loaded with FeynKit's `Model::from_json`.

The result exposes the shared `IntegralFamily`, exact `QuadraticMomentum`
decompositions, numerator `Atom`, original edge identities, and the overall
factor. Apply the overall factor once after reduction. Graph numerator
prefactors and projectors are already included in the numerator. Auxiliary
denominators are never appended. Loop-independent propagators are moved to
the overall factor; repeated loop propagators are rejected, and so are
disconnected graphs.

`prepare_dot` also takes `PreparationOptions`, whose
`complex_mass_propagators` replaces `m^2` by `m^2 - i*m*width` in every
propagator with a model width (`edge_width` names both symbols). When
contraction leaves exactly one joinable pair of open color indices, the numerator
is projected onto `delta/N` and `color_projector` records it; other open indices
are an error naming their external legs. `graph_factors` derives the generator's
automorphism order and closed fermion/ghost-loop sign for graphs written without
an `overall_factor`; `expand_components` replaces open spinor chains by their
chiral-basis components given explicit spinors, and four-dimensional
Levi-Civita tensors from `gamma5` traces by determinants of vector components
in Idenso's normalization (see the CLI guide's
[hand-written DOT graphs](../ninjars-cli/README.md#hand-written-dot-graphs)).

`component_numerator`, `component_kinematics`, `shifts`, and `squared_masses`
prepare exact component expressions from a map keyed by the literal atoms in
`prepared.family.external_momenta()`. They reuse `FourMomentum<Atom>` arithmetic
and `Kinematics` scalar-product bindings. Numerical precision is chosen later
by the evaluator, without intermediate binary64 conversion.

For model parameters, use `Model::scalar_bindings(None, &overrides)` with exact
Atom overrides. To retain a parameter for numerical or Dual evaluation, map
its `UFO::` name to a distinct free symbol; a self-reference is an error.
Model defaults may originate in binary64, so override every parameter whose
input precision matters.

These APIs and fixtures cover preparation. Complete physical graph matching,
MadLoop comparisons, and reduction/performance acceptance remain separate
integration milestones.
