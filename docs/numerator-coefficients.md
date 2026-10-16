# Phase-space polynomial coefficients

The optional CLI mode `--numerator-evaluator symjit-o2` prepares all distinct
polynomial coefficient heads in one Symbolica evaluator and compiles it with
SymJIT O2. The default `symbolica` mode keeps the original numerator preparation
and callbacks. The JIT mode is binary64 only; DoubleFloat and arbitrary precision
requests are rejected explicitly, and the generic library and Dual callbacks are
unchanged. Build the CLI with `--features numerator-jit` to enable this mode.

```sh
cargo run --release -p ninjars-cli --features numerator-jit -- \
  reduce examples/graphs/physical/four-photon/top-box-0.json \
  --numerator-evaluator symjit-o2 --batch-size 3
```

The same option is accepted by `bench`. Its report includes the coefficient
plan/compilation setup costs and a separate `diagnostic.numerator_coefficients`
measurement; the primary total includes coefficient evaluation on every pass.

`--numerator-evaluator symjit-o2-factored` additionally factors each external
coefficient exactly with Symbolica before compiling the same O2 evaluator.
`PolynomialCoefficientPlan::factor_coefficients` exposes this optional setup
operation for either the generic interpreter or JIT. It rebuilds the coefficient
template because factorization can make previously distinct heads identical.
Reports record `coefficients_factored` and `coefficient_factoring_ns`; the latter
is part of the total coefficient preparation cost, not an additional timed
per-point operation. Both existing modes and their defaults are unchanged.

Factoring changes floating-point evaluation order. In a changing-phase-space
four-photon preparation probe, a vanishing unfactored coefficient missed the
fixed `1e-9` Float256 comparison gate at `1.72e-9`; the factored SymJIT candidate
reduced the worst discrepancy to `5.04e-12`. This motivates an explicit optional
policy, not a general guarantee about conditioning. Keep the original form and
high-precision controls when comparing a new physical workload.

`ninjars_graph::PolynomialCoefficientPlan` owns the existing Symbolica polynomial
and composes `expand_in`, `map_coeff` and `flatten(false)`. Its five loop variables
are `(q0,q1,q2,q3,mu_squared)`, with rank weights `(1,1,1,1,2)`. External parameter
sums remain coefficient expressions. Identical exact coefficient heads share an
output, and Symbolica's multi-output evaluator can share their arithmetic. A
fresh parameter stands for each output in the polynomial passed to the existing
NinjaRS callbacks. This is a preparation adapter, not a new polynomial format.

The input expression must include the graph factor and normalization exactly
once. The coefficient plan adds no physical factors. The CLI uses the same
prepared complete numerator as its original Symbolica path.

The generic coefficient interpreter constructs constants at the supplied
prototype's working precision; it does not route DoubleFloat or Float through
binary64. The JIT maps exact and registered constants at 53 bits before compiling.
It uses O2 with fast-math, lane SIMD and threading disabled. SymJIT's separate
`fast_complex` setting enables its scalar complex instruction implementation
(using vector registers for a real/imaginary pair); it does not enable fast-math.
The implementation rejects external SymJIT configuration overrides so those
settings are reproducible. Dynamic shared-function arities are measured from the
exported evaluator DAG and respect the pinned SymJIT argument limit.

Each call accepts exactly `rows * parameter_count` row-major inputs and returns
`rows * coefficient_count` outputs. Zero rows, zero polynomials and constant-only
polynomials are supported. Both input and output components must be finite.
**The initial preparation error policy is transactional for the whole batch:**
one nonfinite coefficient rejects that binding before any row is passed to the
reduction engine. The error identifies the row and coefficient column. Core
per-lane stability handling is unchanged; this mode does not yet compact valid
preparation rows around a failed coefficient row. Callers needing that policy
can split/retry their preparation batches explicitly.

The CLI keeps raw external parameters in its reusable numerical pool. It
computes the coefficient heads inside each complete-evaluation timed iteration,
then binds the result once for all cut callbacks in that phase-space batch.
Coefficient preparation is also reported separately. Compilation, exact plan
construction and buffer warmup are setup costs, not steady-state evaluation.
A reduction-only comparison may precompute coefficients for both implementations
using the same exact coefficient basis and numerical rows. Comparing precomputed
Rust coefficients to a reference callback that recomputes physical coefficients
would not be a matched numerator-provider performance test.

Before composing the plan, the algebra audit covered Symbolica's public
`AtomCore::expand_in`, `to_polynomial_in_vars_with_field`,
`MultivariatePolynomial::map_coeff`, `flatten` and `Atom::evaluator_multiple` APIs;
their polynomial/evaluator implementations and existing tests; and FeynKit,
Spenso and NinjaRS's existing exact-coefficient preservation usages. No new
algebraic transformation was implemented. The SymJIT API audit covered
`evaluate/backend.rs`, the pinned SymJIT 2.27.3 configuration/complex generator
and complex-division tests, and gammaloop's existing multi-output JIT wrapper.
The gammaloop-specific unsafe carrier casts are not used here.

The optional factorization audit additionally checked `AtomCore::factor` and
its exact-identity documentation in `atom/core.rs`, the implementation in
`poly/factor.rs`, `examples/factorization.rs`, and FeynKit's installed FeynCalc
WW/Higgs/Z checks that use Symbolica's factor operation. Coefficient mapping,
factorization and polynomial reconstruction all remain existing Symbolica calls.

Focused controls live in `crates/ninjars-graph/tests/coefficients.rs` and the
CLI `numerator_coefficients` module: exact reconstruction and factor-once,
complex multi-output batches, mixed lazy branches with inactive singularities,
registered constants, zero-input/zero-output plans, permutations and tails,
transactional shape/nonfinite failures, and higher-precision interpreter
controls. These are correctness controls, not a performance claim.
