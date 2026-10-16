# Rust instruction export audit

Source audit on 2026-10-16. This records an implementation route and a proposed
experiment, not measured NinjaRS performance. No numerical implementation was
changed for this audit. Line numbers below refer to the inspected local files.

## Existing mechanism and exact APIs

| Capability | Existing implementation |
| --- | --- |
| Exact master evaluator | `OneLoopExpressions::evaluator(&[Atom], &[Atom]) -> Result<ExpressionEvaluator<Complex<Rational>>, EvaluationError>`, [expressions.rs](../references/oneloopmaster/src/expressions.rs):58. Its compiler retains the `FunctionMap`, uses `direct_translation(true)` and `horner_iterations(0)` at lines 161–180. |
| Portable instruction graph | `ExpressionEvaluator::export_instructions() -> ExportedInstructions<T>`, [instruction.rs](../references/symbolica/src/evaluate/instruction.rs):219. Public fields at line 144 include input/output counts, instructions, temporary count, constants, constant-function metadata and recursive shared bodies. This API does not require the optional C++ native-code-generation feature. |
| Checked exact export | `export_checked`, [generate_native.rs](../references/oneloopmaster/examples/generate_native.rs):73, checks an exact export against a 128-bit mapped export recursively, detecting unresolved constants. |
| Generic Rust lowering | Private `Generator::{module,helper,body}`, same file at lines 216, 239 and 263, emits borrowed inputs, immutable arithmetic locals, helper functions and exact cached constants. `module` currently requires exactly three outputs; helpers require one output. |
| Lazy control flow | `liveness` and `Emitter::range`, same file at lines 537 and 663. They reconstruct structured `if` branches, merge every live slot, restrict reuse to dominating definitions and outline large branches without evaluating inactive arms. |
| Supported instruction operations | The emitter handles `Add`, `Mul`, `Pow`, `Powf`, `Assign`, shared or recognized `Fun`, and structured `IfElse`/`Goto`/`Label`/`Join` at lines 663–896. Unsupported external functions or control-flow shapes return errors. |
| Constant initialization | `Context::new`, [native/mod.rs](../references/oneloopmaster/src/native/mod.rs):65, constructs rational constants directly at working precision and resolves registered pi/polylog constants once. Exact export placeholders must never be emitted as actual zero constants. |
| Numeric interface | `NativeFloat`, [primitives.rs](../references/oneloopmaster/src/native/primitives.rs):43, has implementations for `f64`, `DoubleFloat` and `Float` at lines 71, 122 and 182. Emitted values are `Complex<T>`. |
| Generator regression controls | [generate_native.rs](../references/oneloopmaster/examples/generate_native.rs):1004 onward tests registered constants, inactive division by zero, branch liveness, aliases, captures, tuple ordering and scoped common-expression reuse by compiling emitted Rust probes. |

The generation entry point is `generate` at line 899; its example `main` at
line 961 allocates a 128 MiB stack. It generates all files before replacing the
output set. The owning oneloopmaster code is the available Rust-emission
implementation. Searches of Symbolica's public export APIs, implementation, and
FeynKit/gammaloop usages found C++/SIMD/CUDA exporters and the portable instruction
API, but no second generic Rust emitter to reuse.

## What transfers to NinjaRS

The source of every arithmetic kernel should remain its existing exact
Symbolica evaluator. [kernel.rs](../crates/ninjars-symbolica/src/kernel.rs):48
already builds that evaluator from `ExactKernel::{expressions,parameters}`.
Export that same object before numeric mapping; do not reconstruct Laurent
division, residue subtraction, cut algebra or differentiation in the emitter.

The Laurent, denominator, projection and subtraction kernels are rational
straight-line programs. Their basic emitted operations can call Symbolica's
existing numerical arithmetic, integer powers and inverse. Cut kernels also
need square roots and retain the existing lane-local basis/chart selection.
The selection logic should initially remain in the existing cut engine; export
each selected arithmetic kernel rather than rebuilding chart decisions in
generated code. Reusing oneloopmaster's instruction lowering does not require
copying its custom complex numerical primitives into NinjaRS.

There are two distinct batch mechanisms:

- `NativeEvaluator::evaluate_batch`, [native/mod.rs](../references/oneloopmaster/src/native/mod.rs):165,
  validates rows and then loops over `evaluate_row`. Each complete scalar
  formula finishes before the next row. **This outer row loop does not meet
  NinjaRS's requirement to carry batches through arithmetic and shared calls.**
- `BatchExpressionEvaluator`, [batch.rs](../references/symbolica/src/evaluate/batch.rs):107,
  retains component-contiguous workspaces, executes each instruction across
  active lanes, propagates those lanes through shared functions, and preserves
  lazy branches. `evaluate_batch` at line 130 calls the scalar evaluator directly
  for `rows == 1` at line 158, avoiding batch packing and batch-workspace
  allocations. Tests at lines 486 and 536 cover instruction-major calls and
  inactive shared-function branches.

A first compiled specialization can handle batch size one and keep the existing
instruction-major evaluator for larger batches. A later compiled batch emitter
must distribute each instruction over lanes, with reusable storage and active
lane subsets for branches/shared calls. A scalar row-loop wrapper is only a
measurement control. SIMD remains a separate specialization with independent
accuracy checks; generic batched scalar arithmetic is sufficient for `Float`.

For Dual values, first apply the existing
`ExpressionEvaluator::vectorize(&Dualizer::new(shape, ...))`,
[dual.rs](../references/symbolica/src/evaluate/dual.rs):207, then export the
resulting scalar-component graph. Shared bodies are recursively transformed;
`Dualizer::vectorize_function` at line 753 uses Symbolica differentiation for
registered functions. This preserves the current arbitrary higher/mixed shape
semantics and avoids inventing derivative primitives. Emitted variants would
be keyed by the complete ordered shape, with the existing interpreter as the
fallback for shapes not generated ahead of time. `NativeFloat` itself is
real-carrier-specific and has no Dual implementation; extending it blindly
would also choose the wrong `Complex<Dual<T>>` layout for NinjaRS's
`Dual<Complex<T>>` inputs.

Constants and outputs need the same precision and shape validation as current
`ExactKernel::compile_dual` at [kernel.rs](../crates/ninjars-symbolica/src/kernel.rs):73.
For arbitrary precision, generated immutable locals may create more temporary
allocations than the current reusable evaluator stack. For fixed Dual shapes,
flattened arithmetic can increase generated source and compile time. Neither
tradeoff has been measured here.

## Smallest useful upstream change

No missing Symbolica instruction-export capability blocks a first probe. The
smallest focused reuse change is to extract oneloopmaster's existing emitter
from its example into a feature-gated reusable module, retaining its regression
tests, and parameterize the entry output count and numeric primitive names.
Keep master-specific constant/function policy in oneloopmaster. A proposed API
would accept `&ExportedInstructions<Complex<Rational>>` and emission settings,
and return generated source plus exact constant initialization data; this API
does not exist yet.

Do not fork the existing emitter into NinjaRS or introduce another symbolic
graph representation. If its owner prefers a general upstream home, move the
same implementation to Symbolica or FeynKit's `symbolica-utils` in a focused PR
and make oneloopmaster use it too. A truly instruction-major native batch
exporter is a separate extension; its lazy-branch lowering must be tested before
replacing the current Symbolica batch path.

## Next reproducible experiment

The proposed probe should be implemented only after current correctness gates
pass. It is not an existing CLI command or benchmark result.

1. Export the same exact graphs for `LaurentDivision::new(4,5,3)`,
   `TriangleBubbleSubtraction::new(true)` and `DenominatorExpansion::new(2)`.
   These cover reciprocal Laurent coefficients, the 13 higher-rank bubble
   correction outputs, and the seven denominator coefficients. Add one existing
   cut seed/finish kernel containing a square root after those pass.
2. Generate scalar Rust through the extracted existing emitter. Keep an
   interpreter control built from the exact same graph and numeric constants.
   Add a flattened Dual variant with shape
   `[[0,0],[1,0],[0,1],[2,0],[1,1]]`; do not approximate derivatives numerically
   in the implementation.
3. Reuse the exact fixture construction in `tests/{laurent,subtraction,denominators}.rs`
   and the subtraction C++ oracle controls. Check every output/component,
   including complex inputs, zero coefficients, permutations, invalid shapes
   and unchanged outputs on errors. Validate `f64`, `DoubleFloat` and 256-bit
   `Float` separately, against higher-precision controls where appropriate.
4. Measure three routes with the same preallocated row-major buffers: current
   batch evaluator; generated scalar entry for size one; generated scalar row
   loop as a labeled control. Add generated instruction-major batches only in
   the subsequent experiment. Use batches 1, 8, 32, 128 and 1024, with 31/129
   tails, repeated inputs and changing fixtures.
5. Build release mode on the same host, without fast-math, using one thread and
   one recorded allowed CPU. Run no concurrent builds/tests during timing.
   Construct evaluators/constants and warm every route before timing; retain
   outputs through `black_box`. Record at least seven repetitions, each long
   enough to exceed timer noise, and report median, spread, calls, rows and
   nanoseconds per row. Record revisions, compiler, profile, CPU, input hashes,
   generated source hash/size, compilation time and precision/shape.
6. Measure the prepacked arithmetic alone first, then the coordinator including
   request packing. The latter still creates temporary vectors in
   [triangles.rs](../crates/ninjars/src/triangles.rs):108; generated arithmetic
   alone cannot establish the warmed size-one allocation or latency gate.
   Compare complete reduction with original Ninja only once all required cut
   stages and matching master providers are present.

The existing oneloopmaster benchmark protocol is in
[performance_survey.rs](../references/oneloopmaster/examples/performance_survey.rs):1;
it excludes construction and allocation from steady-state timings. Its
[recorded report](../references/oneloopmaster/performance/2026-09-08-native/README.md)
explicitly retains target misses. Those measurements are not evidence for the
new NinjaRS kernels, and this audit claims no speedup or acceptance-gate result.
