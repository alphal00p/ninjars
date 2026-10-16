# Rust instruction export audit

Source audit and implementation checkpoint on 2026-10-16. The reusable emitter
and scalar/Dual correctness probes are implemented. The owning emitter now also
provides instruction-major batch execution; production dispatch and performance
acceptance remain outstanding. Line numbers in the
original source audit refer to its inspected revisions.

## Existing mechanism and exact APIs

| Capability | Existing implementation |
| --- | --- |
| Exact master evaluator | `OneLoopExpressions::evaluator(&[Atom], &[Atom]) -> Result<ExpressionEvaluator<Complex<Rational>>, EvaluationError>`, [expressions.rs](../references/oneloopmaster/src/expressions.rs):58. Its compiler retains the `FunctionMap`, uses `direct_translation(true)` and `horner_iterations(0)` at lines 161–180. |
| Portable instruction graph | `ExpressionEvaluator::export_instructions() -> ExportedInstructions<T>`, [instruction.rs](../references/symbolica/src/evaluate/instruction.rs):219. Public fields at line 144 include input/output counts, instructions, temporary count, constants, constant-function metadata and recursive shared bodies. This API does not require the optional C++ native-code-generation feature. |
| Checked exact export | `export_checked`, [rust_codegen.rs](../references/oneloopmaster/src/rust_codegen.rs), checks an exact export against a 128-bit mapped export recursively, detecting unresolved constants. |
| Generic Rust lowering | Public `RustEmitter::emit`, in the same module, accepts the existing exact evaluator and `EmitOptions`, returning source and layout metadata. Its private `Generator` emits borrowed inputs, immutable arithmetic locals, helper functions and exact cached constants. Entry output counts are arbitrary; helpers require one output. |
| Lazy control flow | `liveness` and `Emitter::range`, in the same module, reconstruct structured `if` branches, merge every live slot, restrict reuse to dominating definitions and outline large branches without evaluating inactive arms. |
| Supported instruction operations | The emitter handles `Add`, `Mul`, `Pow`, `Powf`, `Assign`, shared or recognized `Fun`, and structured `IfElse`/`Goto`/`Label`/`Join`. Unsupported external functions or control-flow shapes return errors. |
| Constant initialization | `ConstantContext::new`, [native/constants.rs](../references/oneloopmaster/src/native/constants.rs), constructs rational constants directly at working precision and resolves registered pi/polylog constants once. Exact export placeholders must never be emitted as actual zero constants. It is also exported as `rust_codegen::runtime::Context`. |
| Numeric interface | `NativeFloat`, [primitives.rs](../references/oneloopmaster/src/native/primitives.rs):43, has implementations for `f64`, `DoubleFloat` and `Float` at lines 71, 122 and 182. Emitted values are `Complex<T>`. |
| Generator regression controls | [rust_codegen.rs](../references/oneloopmaster/src/rust_codegen.rs) tests registered constants, inactive division by zero, branch liveness, aliases, captures, tuple ordering, scoped common-expression reuse, output counts and transactional constant pools by compiling emitted Rust probes. |

The master generation example is now a thin client of the public emitter; its
`main` allocates a 128 MiB stack. It generates all files before replacing the
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

There are three distinct batch mechanisms:

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
- `RustEmitter::emit_batch`, [rust_codegen/batch.rs](../references/oneloopmaster/src/rust_codegen/batch.rs),
  emits one loop over lanes per instruction and invokes shared helper bodies
  once for their batch. Its `BatchWorkspace` retains disjoint slot-major storage,
  validates sizes before writes, and bypasses packing for a single row. The
  pinned main revision `5091653` includes lazy active-lane execution, validated
  with conditional shared helpers, higher/mixed Dual components, inactive
  singular arms, retained workspaces and allocation controls.

A compiled specialization can use the scalar fast path at batch size one and
the same emitter's instruction-major entry point for larger batches. Lazy
branches require the validated active-lane extension. A scalar row-loop wrapper is only a
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

## Completed upstream extraction

No missing Symbolica instruction-export capability blocked the first probe.
[oneloopmaster PR #6](https://github.com/alphal00p/oneloopmaster/pull/6) extracts
the existing emitter behind `rust-codegen`, preserves its regression tests,
supports arbitrary entry output counts and exposes the existing numerical
runtime. `RustEmitter::emit` accepts `ExpressionEvaluator<Complex<Rational>>`
and `EmitOptions`; `constants_source` exports the shared exact constant table.
Master-specific constant/function policy stays in oneloopmaster. All six
generated master modules are byte-identical before and after extraction.

Do not fork the existing emitter into NinjaRS or introduce another symbolic
graph representation. If its owner prefers a general upstream home, move the
same implementation to Symbolica or FeynKit's `symbolica-utils` in a focused PR
and make oneloopmaster use it too. A truly instruction-major native batch
exporter is now available with tested lazy-branch lowering. Native dispatch
still requires integration and comparison before replacing the current
Symbolica batch path.

## Correctness probe and remaining measurements

The [probe runner](../validation/rust_codegen/README.md) implements the first
three steps below for the Laurent, subtraction and denominator plans. All 36
checks pass: 1,044 scalar outputs and 5,220 flattened Dual components, including
empty and malformed batches. Maximum scalar/Dual differences are
8.13e-16/1.83e-15 for f64, 6.96e-32/3.10e-31 for DoubleFloat, and
9.86e-77/2.56e-76 for Float256. Generated-source hashes and exact plan parameters
are retained in `validation/evidence/rust-kernel-codegen.json`. These graphs
have no branches or shared functions; those features are tested upstream and
are not newly validated by this straight-line probe. Cut square roots,
permutations, timings and production dispatch remain to be measured.

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
4. Measure four routes with the same preallocated row-major buffers: current
   batch evaluator; generated scalar entry for size one; generated scalar row
   loop as a labeled control; generated instruction-major batches. Use batches
   1, 8, 32, 128 and 1024, with incomplete
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

The expanded instruction-major batch probe passes all 108 groups on the final
pins, checking 118,581 scalar outputs and 592,905 flattened higher/mixed Dual
components. It includes permutations, changing workspace sizes, malformed and
overflowing shapes, and zero warmed/fresh-single-row f64 allocations. Source,
executable and dependency hashes are in
`validation/evidence/rust-kernel-batch-codegen.json`.

Its first monolithic dev-profile compilation was stopped after 1,364 seconds
and about 5.6 GB resident memory, before any numerical result. The retained
report is `validation/evidence/rust-kernel-batch-unoutlined-build.json`. The
owning emitter now extracts bounded functions at instruction boundaries without
changing scalar source or arithmetic order. The isolated build completed in
57.10 seconds with about 2.4 GB peak RSS; these shared-host observations establish
build feasibility, not a controlled speed ratio. The final pinned probe also
builds and passes. Native production dispatch and runtime performance acceptance
are still outstanding.
