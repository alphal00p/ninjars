# Real-domain arithmetic audit

This source audit identifies a bounded experiment, not a measured bottleneck or
speedup. It follows the [algorithm audit](reduction-algorithm-audit.md) and the
[bounded transpose result](transpose-probe.md). Snapshot: NinjaRS `dbaaadf`,
Symbolica `2c06f245`, and oneloopmaster `53106429`. No production behavior changes.

## What differs in the actual comparison

The n5/r4 physical harness constructs original Ninja's
`Amplitude<ComplexMasses>` with `RealMomentum` shifts. Both implementations
therefore already use complex mass carriers. The verified distinction is real
external shifts and S-matrix/scale data: NinjaRS converts these into complex
carriers in `validation/performance/shared_provider/src/session.rs`.

Original Ninja's `basis.hh`/`basis.cc` retain real longitudinal basis vectors
`e1/e2` and real Gram-related coefficients, with complex transverse `e3/e4`.
Its `momentum.hh` has mixed real/complex products. NinjaRS's cut geometry uses
`Complex<T>` throughout, including quantities that can remain real when the
external shifts are real. Complex cut vectors, masses and discriminant roots
must remain supported. A real external shift contract alone does not establish
that a square root or logarithm is real.

## Existing algebra and numerical facilities checked

1. Symbolica's public `ExpressionEvaluator<Complex<T>>::set_real_params`, in
   `src/evaluate/optimize.rs`, propagates phase information and annotates mixed
   Add/Mul instructions. It must run after final optimization and merging.
   Default `ComplexEvaluatorSettings` preserve complex square-root, logarithm
   and general power behavior.
2. Symbolica tests this API in `tests/cpp_transcendentals.rs`; its exported
   instruction graph contains Add/Mul real-operand counts. Its current scalar
   and batch interpreters discard the phase metadata during arithmetic. The
   oneloopmaster scalar and batch Rust emitters likewise ignore Add/Mul's
   real-operand counts. Annotation alone will not accelerate those interpreters.
3. FeynKit exposes the same annotation in
   `crates/spynso3/src/evaluator.rs`. Numerica already provides `Complex<T> + T`
   and `Complex<T> * T` in `lib/numerica/src/domains/float/complex.rs`. A native
   lowering experiment can use those operations without implementing new
   symbolic algebra or a parallel complex type.

The native reciprocal primitive already has finite real-axis shortcuts. Count
neither those existing shortcuts nor changing the benchmark mass carrier as new
savings. Real-parameter annotation can reorder operands; numerical equivalence
must be checked rather than assuming bit-identical floating-point evaluation.
Compare the original generic, annotated-but-generic and annotated/mixed bodies
separately. Exported `n_real` describes a prefix of operands; it is not a bitmask
or a declaration of a slot's scalar type. An initial lowerer can keep complex
storage, use mixed operations through the real prefix and its first complex
operand, then retain full-complex tail operations.

## Smallest useful experiment

Inventory mixed and fully real operations in two existing instruction graphs:

- T2 denominator expansion: under an explicit real-shift contract, zero-based
  columns 19 through 22 and 24 through 27 are real. The masses and cut vectors
  remain complex. See `crates/ninjars-symbolica/src/denominators.rs`.
- `TadpoleGeometryProbe`: all eight inputs are external shift components. See
  `crates/ninjars-symbolica/src/tadpole_plans.rs`.

An experimental oneloopmaster emitter can then consume the annotations using
Numerica's mixed operations while retaining existing output heads. Inventory
scalar or emitted mixed operations: a cheaper complex multiplication may leave
the instruction count unchanged. A full public momentum-carrier redesign is
not needed to evaluate this candidate.

Annotate after Dual transformation, which currently rebuilds phase metadata as
`Any`. Real eligibility requires a zero imaginary part in **every** derivative
component. A real primal with an imaginary derivative direction must use the
generic path. Real assumptions must form part of the native semantic identity;
neither primal-only Dual equality nor a cached numerical value establishes that
identity. Preserve the explicit scalar precision prototype independently of
the Dual shape prototype.

Before any production use, cover derivative-only imaginary inputs, higher and
mixed derivatives, reordered unsupported native shapes, complex masses, negative
discriminants, signed zeros and extreme ranges with independent high-precision
controls. Retain finite checks, generic complex fallback and transactional lane
failure semantics. Changed rounding, intermediate overflow or nonfinite behavior
must be diagnosed explicitly.
Eliding zero products can hide `0 * Inf`, and mixed addition can change signed
imaginary zeros that influence later complex branches. Finite inputs alone do
not guarantee finite intermediates. Guarded primitives must fall back to existing
complex operations for nonfinite operands; failures caused by operand reordering
need the untouched whole-kernel fallback. Initially restrict the experiment to
straight-line graphs with literal constants, without inferring realness through
shared helpers, registered constants or general branch graphs.

## Relation to the other candidates

The bounded n3/r3 experiment found 1954 separated-forward instructions versus
1075 fused-forward instructions, but its five-output graph omits intermediate
heads required for failure/reconstruction contracts. Real-domain specialization
could retain those heads and complement fusion; its operation inventory and
runtime are not yet measured. It would leave packing, validation, coordination
and numerator callbacks intact. Historical geometry profiles cannot establish
its priority or expected saving in the current native reduction pipeline.
