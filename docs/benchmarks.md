# Rust benchmark command

`bench` measures the same numerical execution session used by `reduce`. Build an
optimized binary before collecting timings:

```sh
cargo run --release -p ninjars-cli -- bench examples/graphs/bubble.json \
  --output bubble-bench.json
```

The default runs both repeated and changing workloads at batch sizes
`1,8,32,128,1024`. Each iteration contains 1025 points, so every default batch
width greater than one includes a one-point tail. Repeated workloads use the
first fixture sample for every lane. Changing workloads cycle through the fixture
rows in their original order; they require at least two fixture rows and two
points per iteration. The report
records the source-pool size and the policy, since repeated or equivalent rows in
the supplied fixture do not become new kinematics.

For a shorter run or a single-point fixture:

```sh
cargo run --release -p ninjars-cli -- bench examples/graphs/bubble.json \
  --batch-sizes 1,8,32 --points 65 --rounds 5 --iterations 2 \
  --workload repeated --warmup 1
```

The numerical controls match `reduce`: `--precision f64|double-float|float`,
`--bits`, `--cut-stop`, `--mu-extraction`, `--full-box-coefficients`, and exact
`--mu-sampling-scale`. Fixture S-matrices and exact parameter bindings follow the
same preparation path. DoubleFloat and Float inputs are evaluated directly in
their working precision. The benchmark CLI currently covers these three scalar
carriers; Dual benchmark configuration remains future work.

## Timing boundaries

Graph and numerator preparation, exact-expression numerical binding, batch
packing, session construction, and warmup are measured separately. Every full
and tail batch is warmed at least once. Lazy numerator expansion compilation,
master-expression compilation, and workspace growth encountered in those passes
are outside the primary timed rounds. The session and compiled evaluator caches
are retained between cases.

The primary loop includes numeric numerator-parameter rebinding, automatic
sampling-scale evaluation, amplitude reset, cut geometry, numerator callbacks,
reduction, master integrals, and contraction. It calls the existing batched
interfaces without callback timers. It excludes parsing, graph preparation,
prepacking, JSON encoding, setup processes, and IPC. Results and statuses are
consumed through `black_box` to retain the work.

A separate diagnostic pass wraps the numerator and master-provider interfaces.
It preserves every request batch and records calls, rows, and elapsed time.
`residual_reduction_ns` is the diagnostic total minus coefficient preparation,
callback and master time;
its scope includes scale selection, binding, reset, geometry, contraction,
bookkeeping, and timer overhead. It is an attribution measurement, not an isolated
kernel benchmark. The instrumented pass cannot affect primary round timings.

`same_pass_nanoseconds` records integer durations for the total, numerator
coefficients, numerator callbacks, actual master-provider execution, and residual
reduction. The four stages must sum exactly to the total; inconsistent nested
timings produce an error instead of clamping a negative remainder to zero.
`reduction_excluding_masters_and_coefficients` adds callbacks to the residual,
giving the inclusive reduction boundary with actual master computation excluded.
The CLI's direct master provider has no numerical result cache. Shared frozen
master lookup is a separate validation-harness scope, never master computation.

The report retains all round timings, observed quantiles, per-point latency, and
throughput. These observed ranges are not confidence intervals. Numerical failure
counts come from the final warmup pass and the separate diagnostic pass; they are
not inferred to be measurements of every primary iteration. Failed lanes make
the command exit unsuccessfully after writing its report. Precision rescue is
currently disabled and its attempts are reported as zero.

## Current scope of the evidence

Every report sets `performance_acceptance` to `false`. The command measures Rust
execution with warmed compiled evaluators and no numerical master-result cache.
The separate [matched runner](../validation/performance/README.md) compares this
command with original Ninja. Neither tool currently establishes the project's
5% latency or batched-throughput acceptance gates.

The reference comparison must use matching cache policies. Original Ninja's
default OneLOop integration can cache numerical master results, unlike the
current Rust provider. The matched runner uses a dedicated reference build with
`--disable-avholo_cache`, a persistent in-process sample loop, and a common
parameterized numerator for its complete-evaluation baseline.
Reduction with a shared master provider is a separate experiment. Compiler and
dependency revisions, host load, thread settings, accuracy, timing uncertainty,
and rescue rates must accompany those acceptance measurements.

Before retaining an evidence run, the recorder must save the exact build and run
commands, compiler version (`rustc -Vv`), checkout and dependency revisions,
uncommitted-patch status, compiler flags, CPU model, CPU affinity, and active thread
settings beside the JSON report. The current report embeds operating system,
architecture, available parallelism, and debug/release status; it does not embed
compiler identity, revision hashes, CPU model, or affinity. Available parallelism
alone does not establish that two runs used the same processor placement or
thread settings. Record host contention and any processor-frequency controls as
well before comparing timings from separate runs.

## Physical graphs and compiled coefficient preparation

The acceptance suite must include a large physical graph, in addition to
reduction-only comparisons using the same supplied numerator provider. Report
both cold preparation/compilation and warmed end-to-end evaluation. Warmed
evaluation must include wavefunction and dependent-model-parameter evaluation
from the supplied momenta, helicities and model bindings, physical numerator
coefficient evaluation for every phase-space point, cut callbacks, reduction
and masters. Prebound polarization fixtures measure a narrower boundary;
precomputed coefficients alone cannot establish this end-to-end result.

The optional `--numerator-evaluator symjit-o2` provider collects the loop
polynomial's coefficient Atoms with Symbolica and compiles all coefficient
outputs into one SymJIT evaluator at O2. Enable it with the `numerator-jit`
feature. Each batch evaluates every sample's coefficient heads once and
populates the existing reduction callback's parameters. This belongs to the
CLI's numerator-preparation package;
the NinjaRS reduction engine remains independent of graphs, models and JIT
compilation. The ordinary provider remains a correctness and timing control,
with precision-preserving evaluation for carriers unsupported by SymJIT.
The explicit `symjit-o2-factored` variant additionally factors external
coefficients with Symbolica at setup. Record that policy when comparing providers;
its factoring time is included in preparation time and must not be added twice.

Before timing, compare every compiled coefficient and complete reduction with
the existing provider on the same physical inputs. Record the compilation and
cache policy, coefficient evaluation, callback, reduction, master and total
times separately. Compare f64 with original Ninja under matching accuracy,
cache and thread settings; report higher precision and Dual workloads
separately. The compiled coefficient mode is validated on synthetic inputs and
all six physical four-photon boxes, with both mu modes and the independent
exact pole controls. Controlled large-graph end-to-end performance
acceptance remains outstanding; the existing timing evidence predates this
mode. See [coefficient provider details](numerator-coefficients.md) for its
precision and preparation-error policy.

## Optional generated reduction kernels

Build with `--features native-kernels` and select
`--reduction-kernels native-denominators` on `reduce`, `compare` or `bench`
to execute T2/T3 denominator expansions with Rust arithmetic generated from
the existing Symbolica instruction graphs. The default is `symbolica`.
The JSON `reduction_kernels` record identifies this partial selection: other
reduction kernels, integration and accumulation still use the interpreter.
Numerator callbacks, geometry and masters retain their separate providers.

This option supports f64, DoubleFloat and arbitrary precision in the CLI.
It is independent of `--numerator-evaluator`; for example, build with
`--features native-kernels,numerator-jit` to combine generated denominator
execution with `--numerator-evaluator symjit-o2-factored`. Record both choices
when comparing timings. Numerical parity tests do not establish a speedup;
controlled performance comparisons for this selection remain outstanding.

`--reduction-kernels native` selects the growing partial catalog. It adds
ordinary bubble source-axis projections and triangle-to-bubble correction
kernels for quotient degrees 0, 1, 2 and the rank n+1 degree 3. Selection follows
the existing exact plan's rank class and explicit Laurent powers. The JSON
report names `bubble_source_projection` and `triangle_bubble_correction`
separately; unused cut stages need not execute. Other reduction kernels,
integration and accumulation remain interpreted. The library equivalent is
`SymbolicaAmplitudeKernels::with_native_kernels()`.

The library Dual factory supports its existing ordered five-component native
denominator catalog. New bubble projection/correction kernels use the shared
flattened Symbolica evaluator for all Dual shapes. Calling
`with_native_denominators()` after `with_native_kernels()` restores the narrower
selection. See [catalog details](native-reduction-catalog.md) for generation and
validation; neither mode claims complete native execution or acceptance parity.

Guarded n=5/r=4 corrections have two explicit selectors:
`--reduction-kernels interpreted-rank` keeps generic kernels and alternates
interpreted, while `native-rank` selects the native generic catalog and native
alternates. The interpreted selector works in a feature-off build. Both reject
other topologies; existing selectors retain their previous behavior. Reports
keep generic/fallback family coverage and add separate `topology` and
`conditional_corrections` fields for the guarded alternate. The guard can select
the generic path for an entire source-kind batch, so alternate availability is
not an execution-count claim.

The CLI/harness controls use the synthetic
`examples/graphs/pentagon-rank4.json` fixture and the independently matched
physical pentagon. Feature-off and native selection/accounting tests pass;
CLI f64/DoubleFloat/Float checks compare serialized values and do not establish
additional precision. All five physical selectors agree within `4.00e-18`
scaled error over widths 0, 1, 3 and 8. See the
[validation record](../validation/evidence/rank-specialization-harness.json).
Master computation, numerator coefficients, callbacks, preparation and residual
reduction retain their existing separate accounting. These correctness and
accounting controls provide no new timing or performance acceptance result.

Native row scheduling is an independent explicit option:
`--native-block-rows N`. It bounds complete native kernel rows, not phase-space
samples, and reports `native_execution` separately from generic and conditional
backend coverage. Omission retains Whole execution. See
[the execution-policy contract](native-block-execution.md) for supported modes,
compatibility and bounded development validation. No timing scopes or
master/preparation/callback accounting change.
