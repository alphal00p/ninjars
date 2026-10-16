# Matched numerator and master benchmark boundary

The current complete-evaluation harness correctly keeps process setup, input
parsing, compilation, packing, JSON and IPC outside its timers. It does not yet
provide the strict matched-provider comparison requested for reduction.

`reference_bench.rs:54` exports the complete physical expression with raw external
parameters. `benchmark_runner.cc:100` rebinds those parameters and invokes compiled
C++ numerator callbacks. The new Rust SymJIT mode instead computes physical
coefficient heads once in `runtime.rs::begin_batch`, then calls the Symbolica
polynomial adapter. Passing only the new Rust flag through `compare.py` would
compare different numerator preparation scopes and different callback backends.

The original harness had no in-process shared-provider bridge.
Symbolica's C++ API exposes polynomial simplification, not its JIT evaluator;
the pinned SymJIT package is an rlib. The smallest strict control is a validation
bridge around existing public Rust providers, kept outside the production Cargo
workspace. Both reducers must use the same out-of-line provider functions if
bridge overhead is included in an eventual latency claim. A Rust-native call on
one side and an FFI call on the other is a correctness probe, not a matched
latency control.

The first probe under `validation/performance/shared_provider` covers the
numerator boundary. A C++ `ninja::Numerator` subclass calls the same Rust
`PolynomialNumerator<f64>` implementation through explicit dimensioned arrays of
`{double re, double im}`. No `std::complex`/Rust layout cast is permitted. Handles
own their provider and scratch; callers retain input buffers only for a call.
Errors and caught Rust panics become status codes, with no unwinding across the
ABI. The probe compares the direct, mu, T3 and T2 responses with safe Rust calls,
including both ordinary and higher-rank mu layouts, then lets original Ninja
reduce an analytic triangle control. A second correctness probe captures the
actual master-request union of both reducers for ordinary/higher-rank bubbles,
then freezes and replays it without executing master expressions. Neither is a
timed benchmark. The first integrated Rust control constructs its topology and
geometry for each call and uses direct Rust providers. A separate retained
session now calls the same non-inlined provider entry points as C++, retaining
its numerical wire buffers and topology/geometry. It is checked against that
direct control. Remaining descriptor/master staging allocations and the
underlying providers must be measured before accepting its warmed cost.

The audited original virtual interface is `ninja/num_defs.hh`: direct receives
`q,mu_squared,cut,partition`; mu receives one vector for ordinary rank and two for
rank n+1 and has no cut-count argument; T3 receives three vectors and beta; T2
receives four vectors and three betas. T3 `mindeg=0..4` output counts are
`1,2,4,6,9`, and T2 `mindeg=0..3` counts are `1,3,7,13`. The existing reference
exporter is the authoritative adapter-order control. The probe carries lane and
sample identity explicitly and binds coefficient rows before invoking callbacks.

The next reduction-only mode will bind identical precomputed coefficient rows
and query an immutable master table populated outside timing. Its scope must be
named **reduction, contraction and shared master lookup**, since lookup overhead
is still included. Original Ninja's public `Amplitude::setIntegralLibrary`
accepts an `IntegralLibrary` subclass. Override real and complex A0, B0,
rank-two/rank-three bubbles, C0 and D0; rank-three must be overridden rather than
falling back to the original lower-rank derivation. Preserve `[finite,single
pole,double pole]`, B1/B11/B111 conventions and the D0 invariant order.

The correctness prototype builds the table from the union of both reducers' requests, using the shared
provider to evaluate all entries once before timing. Keys contain the integral
kind, scale, ordered invariant and squared-mass values. Use the same explicit
S-matrix rows to avoid independently rounded invariant keys, and fail on a miss
rather than computing or approximating a missing value. Do not cache by cut
number alone, drop scale, or reorder masses. Replay is an explicit reduction
experiment; neither production master backend may silently acquire result
caching. The initial prototype uses exact f64 component bits, including signed
zero, reports source-specific and shared key counts, and explicitly rejects
unknown replay keys. Its integrated controls cover A0/B0/B1/B11/B111; integrated
C0/D0 controls, live-master timing and allocation measurements are still
required. Separate frozen-key checks
cover C0/D0 real/complex-mass virtual argument order and scale propagation.

For complete physical evaluation, retain the raw phase-space parameter pool and
invoke the same multi-output SymJIT coefficient evaluator inside every timed
point/batch in both pipelines. Compilation remains setup. The original scalar
pipeline uses rows=1; the Rust pipeline uses its requested batch width. Bind the
result once per phase-space lane, then perform all callbacks, reduction and
production master evaluation. Keep each production master backend named in the
report; a shared live-master mode can be added separately. Precomputed
coefficients are permitted only in the explicitly reduction-only mode.

The existing four-photon fixture rows vary helicities/polarizations at one
momentum point. They must not be called changing phase space. A physical timing
pool must include distinct momentum/invariant points at fixed helicity, generated
with shared FeynKit wavefunction APIs and independently checked against the
reference. Separate repeated-point, changing-phase-space, and helicity-variation
workloads; record distinct kinematic and polarization hashes. Static graph/rule
construction and parsing are setup, while the numerical coefficient evaluation
is mandatory timed work. The full physical acceptance boundary starts with momenta plus helicities and
model bindings: per-point wavefunction/polarization and numerical parameter
evaluation must also occur inside timing on both sides. The initial existing
fixture benchmark starts later, with prebound component vectors and couplings,
and must be labeled accordingly. Static graph generation stays outside either
timer; prebound wavefunction/coupling preparation is never reported as timed.

Before retained timing: verify every Laurent coefficient and finite split,
callback counts and coefficient binding counts, ordinary/higher-rank layouts,
empty/single/tail/permuted batches, invalid lengths and error propagation,
master key completeness and scale variation, and unchanged result checksums.
Run optimized builds on a quiet pinned CPU with identical thread/fast-math/cache
policies, reverse process order in independent runs, retain raw rounds and
uncertainty, and keep busy-host observations diagnostic. The validation bridge
and shared master replay now have correctness controls; measured reduction
throughput and a changing-kinematics large physical benchmark remain separate
milestones.
