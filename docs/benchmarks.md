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
`residual_reduction_ns` is the diagnostic total minus callback and master time;
its scope includes scale selection, binding, reset, geometry, contraction,
bookkeeping, and timer overhead. It is an attribution measurement, not an isolated
kernel benchmark. The instrumented pass cannot affect primary round timings.

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
