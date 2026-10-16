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

The alternate currently uses the Symbolica interpreter. Native constructors
retain their native generic fallback, and amplitude builders compose in either
order. Callers using this option must consult both `kernel_backends()` (generic
fallback selections) and `rank_specialized_correction_backend()` (conditional
alternate). No CLI selection or report schema is changed by this prototype.

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
Tests compare complete lower-cut residue coefficients, ordinary f64,
DoubleFloat, Float192 and every component of an ordered five-component
HyperDual. They exercise every masked-slot violation, derivative-only
violations, complex masses and differing origins, changing batch sizes,
permutations, invalid rows and retries. A separate Float256 control distinguishes
an exceptional intermediate-overflow change from a finite numerical mismatch.
Independent exact box and pentagon numerators also verify that all lower-stage
coefficients vanish in the n=5/r=4 topology, for both generic and specialized
corrections; those numerators do not use the box or pentagon reducers.

This is a correctness prototype. It has no timing result or performance
acceptance claim. Generated native specialized bodies, end-to-end CLI selection,
and any reduction of the swapped numerator callback remain separate work.
