# Symbolica adapters for NinjaRS

`PolynomialNumerator<T>` implements the batched Ninja callback contract over
Symbolica expressions and shared FeynKit momenta. `T` sets the real coefficient
precision; cut values use Symbolica `Complex<T>`. Tested carriers include `f64`,
`DoubleFloat`, and arbitrary-precision `Float`.

Construct the adapter from an `Atom`, the five loop variables in
`[q0,q1,q2,q3,mu_squared]` order, external parameter symbols, and a value at the
desired working precision. Call `set_parameter_batch` with row-major external
parameters and the phase-space lane count. Every callback request selects its
external parameters through `SampleId::lane`; request order can differ from
phase-space order.

The adapter rejects negative powers and non-polynomial functions of the loop
variables. It counts `mu_squared` with rank weight two. The reducer is responsible
for the separate topology constraint `rank <= propagator_count + 1`.
Expansion requests contain explicit **raw numerator** powers of `t`, `x`, and
`mu_squared`; original Ninja's `mindeg` convention denotes subleading depth.
All substitutions, finite Laurent polynomial extraction and numerical evaluator
construction use Symbolica. Evaluators and input buffers are reused.

`DualPolynomialNumerator<T,D>` accepts an existing static Symbolica Dual or dynamic
`HyperDual`, preserving its derivative shape. It transforms the exact evaluator
with Symbolica's `Dualizer`, then executes flattened components in batches.
This propagates higher and mixed derivatives through external parameters, cut
momenta, cut parameters and explicit `mu_squared` terms. Components follow
Symbolica's Taylor coefficient convention: a multi-index component is the
corresponding derivative divided by the product of the component factorials.

`SymbolicaMasters<T>` implements `MasterProvider<Complex<T>>` using oneloopmaster's
exact expressions for A0, B0, DB0, C0, D0, B1, B11, B111 and the available tadpole
tensor coefficients. It maps constants directly at the working precision and
uses Symbolica's batch interpreter through shared helper calls and lazy branches.
Results are `[finite, single pole, double pole]`.

For arbitrary-precision `Float`, the configured working precision controls
constant construction. Symbolica tracks significant bits separately in each
numerical component, so cancellation can reduce a component's reported precision.
The adapters preserve that information instead of resetting it to the nominal
precision. Finite output and domain checks do not by themselves certify accuracy;
analytic controls and precision-convergence checks remain necessary.

`SymbolicaDualMasters<T>` currently enables derivatives for positive real masses
and scales in tadpoles and strictly spacelike bubbles with real seeds. It rejects
triangle/box derivatives, zero-invariant removable limits and other unvalidated
derivative regions. These restrictions are tracked in the project plan. Invalid
master requests through `evaluate` leave the entire output buffer unchanged.
`evaluate_batch` reports a status for each row and leaves failed rows unchanged,
while evaluating all eligible rows through one batched instruction graph.

## Integrated amplitudes

`SymbolicaAmplitudeKernels::new(n, rank, prototype)?.into_amplitude(lanes)`
constructs a core `Amplitude` for one through six propagators and rank at most
n+1. Supply shared lane-major shifts/masses, one squared scale per lane, a
numerator, `SymbolicaCuts` and a master provider to `evaluate_batch`. Successful
lanes accumulate into `[finite, single pole, double pole, cut-constructible finite,
rational]`; failed lanes retain all previous components. Inspect `statuses()`
for the last attempt and call `reset()` or `reset_batch(lanes)` to clear results.
`SymbolicaDualAmplitudeKernels` provides the same interface for static and dynamic
Dual carriers; supported integrated derivatives remain limited by the master
provider's validated domains.

For four or more propagators, use `evaluate_batch_with_high_cuts` with a
`SymbolicaHighCuts` provider and one nonzero mu-squared sampling scale per lane.
The scale should be comparable to the largest squared mass or invariant.
`with_box_options(BoxReductionOptions { coefficients, mu_extraction })` selects
integrated or full box coefficients, and expansion or finite-mu sampling.
The default is integrated coefficients with mu expansion; it avoids pentagons.
Other modes construct pentagons only when needed for subtraction. Box integration
is `c0*D0-c4/6`; pentagon residues have no direct master contribution. Box requests
to `ResidueIntegrator` use `basis: None`; lower cuts require their extraction basis.

Run `cargo run -p ninjars-symbolica --example bubble` for a complete batched
rank-three example. The example provides a 128 MiB worker stack for master
expression preparation. Numerical coefficients, geometry and master values
retain the requested precision throughout; all arithmetic is generated through
Symbolica. Zero-weight masters are skipped only if every Dual component is zero.

## Laurent reduction stages

The core `TriangleReducer` connects `SymbolicaCuts`, numerator callbacks and
`SymbolicaTriangleKernels`. Inputs are lane-major shared `FourMomentum` shifts
and squared masses. Outputs are cut-major triangle residues with original Ninja
coefficient ordering and per-cut/lane stability. This is the triple-cut stage;
it does not return an integrated amplitude. `SymbolicaDualTriangleKernels` uses
the same coordinator with static or dynamic Dual carriers.

Run a standalone residue-stage example with
`cargo run -p ninjars-symbolica --example triangle`; it returns `c0=c7=1`
for `1+mu_squared` and the three specified unit-mass propagators.

`SymbolicaCuts` supplies triangle, bubble and tadpole parameterizations and
batched cut sampling. Complex square roots and lane-local reference charts
support spatial and null kinematics. Chart decisions use primal values, while
the selected arithmetic propagates every derivative component. Degenerate Gram
determinants and singular sampling parameters produce per-lane failures.

`SymbolicaHighCuts` and `SymbolicaDualHighCuts` generate four/five-cut solutions
with Symbolica's exact matrix solver over FeynKit momenta. Four coordinate charts
cover the box linear constraints; the remaining quadratic equation generates
paired samples. `PentagonReducer` computes the residue `coefficient*mu_squared`.
`BoxReducer` subtracts containing pentagons where needed and keeps an explicit
computed-coefficient mask. Scalar and higher/mixed Dual implementations share
the generated expressions. The independent [reference audit](../../docs/box-reference-audit.md)
records original Ninja reconstruction and caching defects relevant to validation.

`LaurentDivision`, `DenominatorExpansion`, `ResidueProjection`,
`TriangleBubbleSubtraction` and `TadpoleSubtraction` expose the constituent
arithmetic kernels. Their plans contain coefficient layouts, while all
substitution, series expansion, coefficient extraction and differentiation use
Symbolica. `MasterContraction` produces tensor-master weights and contracts
Laurent results while retaining the explicit rational part. Its decomposition
depends on the selected residue basis, as in original Ninja.

Kernel inputs and preconditions are documented on their public types. Low-level
division and projection expect nonsingular active rows. The triangle coordinator
compacts invalid rows and restores cut/lane order. A scaled small-denominator
test, precision rescue, full reconstruction diagnostics and S-matrix/cut controls
remain unfinished. Bubble coordination
subtracts every containing triangle; tadpole coordination subtracts triangle and
corrected bubble parents. Tadpoles expose an explicit mask for the integrated
coefficients c0 and, at higher rank, c14/c15; other entries are uncomputed. The
current coordinator still allocates request/geometry buffers and is not the
final performance implementation.

Run adapter probes serially with a larger test-thread stack for oneloopmaster's
expression construction:

```sh
RUST_MIN_STACK=134217728 cargo test -p ninjars-symbolica -- --test-threads=1
```

The optional `reference-codegen` feature exports original Ninja C++ callbacks via
Symbolica's code generator. It is used only by validation tooling. The generated
reference adapter currently computes complete coefficient lists before copying
requested prefixes, so its timings are not the final optimized Ninja benchmark.
