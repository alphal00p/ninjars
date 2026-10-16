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

`SymbolicaDualMasters<T>` currently enables derivatives for positive real masses
and scales in tadpoles and strictly spacelike bubbles with real seeds. It rejects
triangle/box derivatives, zero-invariant removable limits and other unvalidated
derivative regions. These restrictions are tracked in the project plan. Invalid
master requests leave the entire output buffer unchanged; per-lane rescue is
still part of the future reduction engine.

Run adapter probes serially with a larger test-thread stack for oneloopmaster's
expression construction:

```sh
RUST_MIN_STACK=134217728 cargo test -p ninjars-symbolica -- --test-threads=1
```

The optional `reference-codegen` feature exports original Ninja C++ callbacks via
Symbolica's code generator. It is used only by validation tooling. The generated
reference adapter currently computes complete coefficient lists before copying
requested prefixes, so its timings are not the final optimized Ninja benchmark.
