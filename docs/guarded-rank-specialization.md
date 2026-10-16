# Guarded correction specialization

The optional `with_rank_specialized_corrections()` builder on the Symbolica
bubble, tadpole and amplitude adapters specializes corrections for five
propagators and rank four. Other topologies are rejected explicitly. The
default path and the public standalone subtraction plans keep their existing
contracts. No numerator callback layouts, cut geometries, master calls or core
coordinator behavior change.

The implementation derives structural zero slots from the existing exact
`ResidueProjection`: triangle slots 3, 6, 8 and 9; bubble slots 2, 4, 6, 7, 8 and
9. It uses Symbolica literal replacement to construct alternate correction
bodies with the same input ABI. Both bubble orientations still return their
complete three-head quotient. Tadpoles select the constant head only after
checking that the actual quotient is exactly `T3[(0,0)]`.

Each call first performs the existing shape, finite-input, source-projection
and denominator checks. The alternate runs only if every structural-zero slot
of every active parent is zero, using `IntegrationChecks::is_zero` across all
Dual components. A single nonzero coefficient chooses the generic batch for
that entire source kind. Invalid/inactive requests cannot force another lane
onto a different path. The generic tadpole fallback retains both output heads
and checks both for finiteness before copying the constant. Standalone public
stage users may therefore supply arbitrary coefficients without losing data.

`with_rank_specialized_corrections()` selects interpreted alternates. With the
`native-kernels` feature, `with_native_rank_specialized_corrections()` instead
selects generated alternates for f64, DoubleFloat and arbitrary-precision Float.
The existing ordered Dual shape `[[0,0],[1,0],[0,1],[2,0],[1,1]]` also has generated
bodies; all other shapes, including reordered components, retain interpreted
alternates. Stage adapters report the actually installed alternate backend.

Generic native selection and rank specialization are independent: either order
with `with_native_kernels()` or `with_native_denominators()` preserves both.
Calling the interpreted-specialization builder after the native-specialization
builder explicitly returns to interpreted alternates, and vice versa. Native
constructors retain their existing generic fallback. Callers must consult both
`kernel_backends()` (generic fallback selections) and
`rank_specialized_correction_backend()` (conditional alternate). The report
identifies the eligible alternate; the guard decides whether each active batch
uses it. No CLI selection or report schema is changed by this prototype.

The generator `generate_native_rank_specializations` consumes three explicit
semantic plans: triangle-to-bubble n5/r4 full quotient (28 inputs, 3 heads),
triangle-to-tadpole n5/r4 constant (25 inputs, 1 head), and bubble-to-tadpole n5/r4
constant (24 inputs, 1 head). The interpreted stages and the generator use the
same exact constructor. The oneloopmaster emitter lowers Symbolica instruction
graphs, including Symbolica's Dual transform; no formulas are handwritten. The
full original input ABI is retained for every generated body. The generator
requires only `rust-codegen`, so it can reproduce native files without first
building the native catalog.

Dual stages retain the scalar precision supplied explicitly to their constructors.
The optional correction bodies recover that scalar prototype from the existing
compiled division kernel; they do not infer it from the Dual carrier. For example,
a Float256 scalar prototype and a Float128 Dual carrier still compile evaluator
constants at 256 bits. The mixed-prototype regression covers direct stages and
amplitude factories and checks actual mapped evaluator constants.

Algebra reuse was checked in three places before implementation:

- Symbolica's public `AtomCore::replace_multiple` API and example in
  `src/atom/core.rs`;
- its `AtomView::replace_multiple` implementation and replacement regression
  tests in `src/id.rs`;
- FeynKit's existing replacement uses in `feynkit-amplitude/src/spin.rs` and
  `feynkit-amplitude/src/lib.rs`, alongside NinjaRS's existing exact projection
  and subtraction plans.

There is no handwritten polynomial simplification, differentiation or new
residue formula. Numerical evaluation and higher/mixed differentiation use
the same shared Symbolica compiler as other kernels.

Validation is recorded in `validation/evidence/guarded-rank-specialization.json`.
The subsequent mixed-prototype precision fix and refreshed checks are recorded in
`validation/evidence/guarded-rank-specialization-precision.json`.
Tests compare complete lower-cut residue coefficients, ordinary f64,
DoubleFloat, Float192 and every component of an ordered five-component
HyperDual. They exercise every masked-slot violation, derivative-only
violations, complex masses and differing origins, changing batch sizes,
permutations, invalid rows and retries. A separate Float256 control distinguishes
an exceptional intermediate-overflow change from a finite numerical mismatch.
Independent exact box and pentagon numerators also verify that all lower-stage
coefficients vanish in the n=5/r=4 topology, for both generic and specialized
corrections; those numerators do not use the box or pentagon reducers.

Native-extension validation is recorded separately in
`validation/evidence/native-rank-specialization.json`: 21 focused controls pass,
including full lower-stage coefficients and independent parent cancellation.
Eligible Dual fixtures use distinct complex seeds by derivative component and
lane on mass, routing and residue inputs. A discriminator requires the two first
derivatives to differ in an output head, so a generated component permutation
cannot pass through symmetric seeds. Every component is compared against the
interpreted alternate at f64, DoubleFloat and Float192 precision. Exact masks,
whole-kind fallback, empty/tail/permuted batches, invalid inputs and retries,
unsupported/reordered Dual shapes, both builder orders and actual native
Float256 constants with Float128 Dual carriers are covered. An independent
Float256 oracle also controls the exceptional intermediate-overflow case.

The six generated bodies, shared constant table and metadata reproduce
byte-for-byte. Strict Clippy for the library, tests and examples, feature-off
compilation, formatting and whitespace checks pass. The initial build failure
was a missing type annotation in test-only precision instrumentation; its log
is retained. There is no native timing result or performance acceptance claim.
End-to-end CLI selection and any reduction of the swapped numerator callback
remain separate work.
