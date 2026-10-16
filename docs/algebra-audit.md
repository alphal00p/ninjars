# Algebra API checks

## Laurent division and coefficient subtraction

Before implementation, checked three independent sources:

1. Symbolica's public `AtomCore::series`, `Series::coefficient`/`to_atom`,
   `to_polynomial_in_vars_with_field` and `MultivariatePolynomial::coefficient`
   support the required series and finite Laurent coefficient extraction.
2. Implementations in `symbolica/src/poly/series.rs` include Laurent/Puiseux
   coefficient tests, absolute versus relative truncation and coefficient cleanup;
   `examples/series.rs` demonstrates the public interface. Signed polynomial
   exponents already exist. `quot_rem` also exists, but rational series are a more
   direct match for the expansion at infinity required here.
3. FeynKit's `feynkit-cff/src/algebra.rs::laurent_coefficient` and
   `feynkit-graph/src/{power_counting,uv}.rs` already use these series facilities
   for residues and asymptotic expansions.

NinjaRS therefore builds exact Atom formulas, substitutes `z=1/t`, uses Symbolica
series at `z=0`, and extracts polynomial coefficients with Symbolica. Coefficient
ordering is reduction metadata, not a replacement polynomial representation.
Numerical arithmetic is executed by Symbolica instruction graphs; Dual graphs
are derived by Symbolica's Dualizer. No custom series inversion, polynomial
division recurrence or differentiation implementation is introduced.

## Invariants, master contraction and accumulation

The integration coordinator reuses FeynKit `FourMomentum::dot` and momentum
subtraction on Symbolica Atoms. These public operations, their implementations
in `feynkit-kinematics/src/momentum.rs`, and the existing symbolic cut-geometry
usages provide the three API checks. No new momentum arithmetic is introduced.
Master weights and rational terms reuse `MasterContraction`'s exact graphs.
Symbolica's existing multi-output evaluator and Dualizer compile those graphs,
including the sum over all cuts and the previous accumulated result. Core Rust
code schedules rows, copies values and checks statuses; numerical sums and
products execute in Symbolica evaluators. A master weight is skipped only when
every scalar or Dual component is zero.

## Automatic sampling scales and supplied invariants

The public FeynKit `FourMomentum::dot` API, its `momentum.rs` implementation,
and existing graph/cut usages supply invariant construction. Symbolica's public
`AtomCore::abs`, evaluator implementation and Numerica complex norm/`hypot`
implementation supply magnitudes without overflow-prone squaring in NinjaRS.
Sampling providers compile these exact expressions through the existing
`ExactKernel` and `BatchExpressionEvaluator`. Selecting the greatest magnitude
is numerical control flow; it uses the primal at working precision and returns
a constant with the original Dual shape. No derivative algebra is introduced.
Only an all-zero kinematic scale receives the unit fallback.

An optional supplied S-matrix replaces the generated invariant columns before
both master arguments and rational weights are evaluated. Copying those columns
adds no algebra. Matrix symmetry, zero diagonals and finiteness include all Dual
components; primal-only equality is deliberately insufficient.

## Off-axis bubble reference vectors

The existing public FeynKit `FourMomentum<Atom>::from_weyl_spinors` constructs
null vectors from exact spinor components. Its `feynkit-kinematics/src/weyl.rs`
implementation and reconstruction/null-vector tests were checked, together with
the existing NinjaRS tadpole and cut-basis usages. Symbolica's existing exact
multi-output evaluator and Dualizer support these constant vectors and their
dot products; no new spinor algebra or numerical square-root implementation is
needed.

Original Ninja's `basis.cc::Basis(const RealMomentum&)` uses a generic null
reference with nonzero components along every spatial axis. The earlier NinjaRS
axis references could place both transverse vectors outside a planar physical
graph, making an uncut denominator's leading Laurent coefficient identically
zero. Eight rational off-axis references now follow the same principle, with
the largest primal dot product selected at working precision. Every reference
component and dot product comes from Symbolica arithmetic. This fixes the
equal-mass lightlike bubbles in the H→gg triangle geometry; complement-aware
reference retries and broader numerical rescue remain outstanding. The existing
zero-leading-denominator rejection is retained.

## Reference callback constants

The public `ExpressionEvaluator::map_coeff_with_prec` API, its implementation
in `symbolica/src/evaluate/evaluator.rs`, and Symbolica's
`examples/nested_evaluation.rs` establish that registered constants must be
initialized before C++ export. Exact rational evaluators retain placeholder
slots for constants such as π; exporting those slots directly emits zero.
NinjaRS already maps constants for Rust evaluation. The reference exporter now
uses the same supported operation with `Complex<Float>` at 200 bits, then
delegates C++ emission to Symbolica. The compiler rounds the emitted literals
at the reference's binary64 boundary. The exact prepared expressions remain in
the reference manifest. This changes neither production constants nor original
Ninja, and introduces no custom constant evaluation or algebra.
