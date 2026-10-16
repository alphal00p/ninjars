# Matched numerator and master benchmark boundary

The validation harness in `validation/performance/shared_provider` compares
original Ninja and NinjaRS with the same numerator provider and an explicitly
selected master provider. It covers synthetic bubbles and a verified top-quark
gg→HHH pentagon (graph 26). It is separate from the production Rust workspace.
The executable scopes, commands, retained measurements and limitations are
documented in [`TIMING.md`](../validation/performance/shared_provider/TIMING.md).

## What each measurement includes

| Scope | Work inside the measured region |
|---|---|
| `preparation` | Raw momentum/model staging, FeynKit wavefunctions, dependent parameters, factored multi-output SymJIT O2 coefficient evaluation, propagators and numerator binding |
| `shared-reduction` | Prepared-input staging, reduction, the common numerator callbacks, frozen master lookup, contraction and output staging |
| `shared-live-complete` | Preparation and reduction with the common numerator and live Symbolica master providers |
| `production-complete` | Preparation and reduction with the common numerator, using original Ninja/OneLOop or NinjaRS/SymbolicaMasters |

The reduction scope excludes master expression execution. Frozen master lookup
and transport remain timed work and must not be described as computing masters.
Complete evaluation bypasses the frozen table, including for repeated points.
Original OneLOop's result cache must be disabled. Ordinary allocations, copies,
descriptor construction and callback transport are included in every applicable
scope; graph import, symbolic preparation, evaluator compilation and topology
construction are setup.

Master computation and reduction require separate reported times. Primary
aggregate timings stay uninstrumented. A separate diagnostic pass must measure
actual backend execution and identifies numerator/preparation/transport spans;
inclusive master transport must not be added to its nested backend duration.
Time outside master execution still includes lookup and transport. Do not infer
master cost by subtracting measurements from different runs or confuse a cached
lookup with an actual backend call. Diagnostic instrumentation overhead is not
removed from primary timings, because those builds contain no such timers.

## Identical provider boundaries

Both reducers invoke the same out-of-line Rust numerator entry points. The
validation ABI uses dimensioned arrays of `{double re, double im}`, with explicit
lane and sample identity; there are no `std::complex`/Rust layout casts. Retained
handles own providers and scratch, while input borrows last only for a call.
Errors and caught Rust panics become status codes without unwinding across the
ABI. No production core FFI or additional algebra is introduced.

In the current physical control, SymJIT O2 compiles the external polynomial
coefficient heads. The direct/Mu/T3/T2 cut callbacks still use the same
interpreted `PolynomialNumerator<f64>` on both sides. Whole-process interpreter
CPU samples therefore include numerator callbacks as well as reduction kernels;
separate callback and kernel spans are needed to attribute those costs. This is
not yet a fully compiled symmetric-tensor callback provider.

Original Ninja's `Numerator` callbacks and Rust callbacks retain their distinct
batch widths while sharing the arithmetic provider. The original interface has
one vector for ordinary-rank mu expansion and two for rank n+1; T3 takes three
vectors and beta, and T2 takes four vectors and three betas. The reference
exporter and direct callback probes check coefficient ordering and expansion
depths separately from integrated results.

Equal callback row totals do not establish equal numerator arithmetic. Original
Ninja requests a shallower expansion for the swapped bubble orientation; the
current Rust coordinator requests the full depth for both orientations. The
common provider builds different evaluators for these layouts. This is part of
the measured algorithm difference. Comparisons of coefficient-pruning changes
must record requested heads and powers as well as rows and callback time.

The frozen master table contains the union of requests actually made by both
reducers. Its keys retain integral kind, scale, ordered invariants and squared
masses, using exact f64 component bits, including signed zero. Both reducers use
the same explicit S-matrix rows. An unknown key fails instead of executing a
master or approximating a match. Controls cover A0/B0/B1/B11/B111 and C0/D0,
including real/complex virtual argument ordering. The original rank-three bubble
override prevents a hidden lower-rank fallback from changing this boundary.

The older standalone reference benchmark uses compiled C++ callbacks over raw
parameters, whereas the production Rust SymJIT path prepares coefficient heads
before reduction. It remains useful for validation but does not establish a
matched-provider reduction comparison. The shared bridge supplies that control.

## Physical workload and evidence

Both paths receive physical preparation at the same batch width, including
coefficient evaluation. Original Ninja then reduces lanes individually; NinjaRS
receives the whole batch. Complete records start from momenta, helicities and
model bindings. These records include validation FFI and staging and are not
unmodified production CLI latency.

The fixed physical pool has three distinct phase-space points at fixed `pp`
helicity. Changing-point records rotate this pool; repeated records use its
first point. Helicity changes at fixed momentum are a different workload and
must not be labeled changing phase space. Warmup exercises every pool point and
callback layout, then checks Laurent coefficients against original Ninja and
independent MadLoop values. It also checks each backend's internal finite split;
cross-backend rational/cut-constructible component parity remains incomplete.

The separate physical probe checks independent raw numerator samples, a Float256
coefficient oracle, transactional preparation and empty/tail batches. Its source,
artifact and dependency identities are retained alongside timing evidence.
Historical correctness evidence is not silently promoted to current sources.

Performance acceptance requires aligned optimized builds without fast-math,
matching cache policies, contiguous repeated timing blocks, balanced execution
order, retained raw samples and uncertainty, and reviewed host conditions.
Widths 1, 8, 32, 128 and 1024, incomplete tails and changing inputs remain required.
CPU affinity alone does not reserve a CPU or its SMT sibling. Current retained
timings are diagnostic and do not meet the reduction performance gates. The
concrete historical sampling/profile limitations remain recorded in `TIMING.md`
and the immutable evidence indexes.
