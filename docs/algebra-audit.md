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
