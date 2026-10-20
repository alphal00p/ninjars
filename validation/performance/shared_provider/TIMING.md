# Physical timing boundaries

This harness compares the same verified gg→HHH graph26, coefficient basis and numerator provider, using the frozen three-point physical pool. It must first pass its default correctness-only mode; timing is an explicit separate invocation. Timing output is diagnostic until the host/CPU conditions and the project acceptance gates are independently reviewed.

Four records distinguish numerical work instead of inferring a stage by subtracting two noisy measurements:

| Scope | Included in each measured call | Excluded |
|---|---|---|
| `preparation` baseline | Raw input staging, FeynKit wavefunctions, dependent parameters, factored multihead SymJIT coefficients, shifts/masses/S matrix, numerator binding | Reduction and masters |
| `shared-reduction` | Ordinary prepared-input staging, original/Rust reduction, the identical out-of-line Rust numerator provider, frozen union master lookup, contraction and output staging | Physical numerical preparation and master expression execution |
| `shared-live-complete` | Full physical numerical preparation plus reduction, the identical out-of-line numerator provider and identical live Symbolica master provider, output staging | Only setup and driver I/O |
| `production-complete` | Full physical numerical preparation and common numerator provider, original Ninja/OneLOop versus NinjaRS/direct SymbolicaMasters, output staging | Only setup and driver I/O |

`production-complete` compares production master backends while retaining a deliberately common validation numerator provider. It is not the latency of the unmodified production CLI: the common validation FFI for numerator callbacks and the normal C++/Rust boundary staging remain included. There is no production core FFI or new algebra. SymJIT O2 compiles the physical coefficient heads only. The shared `PolynomialNumerator<f64>` direct/Mu/T3/T2 callbacks still use Symbolica's generic batched interpreter. Interpreter CPU in a whole-process profile therefore includes numerator callbacks and cannot be called pure reduction-kernel time. The separate execution pass records each callback kind independently, including calls, rows and provider ABI time.

Both paths receive the same batched physical preparation at the record's width, including coefficient-head evaluation. Original Ninja subsequently reduces one lane at a time; NinjaRS receives the whole reduction batch. This intentionally gives original Ninja the same preparation batching benefit. Each result records `original_preparation_batching`, `shared_preparation_width` and the separate reduction batch widths.

Graph import, Feynman rules/contraction, exact coefficient preparation/factoring, evaluator construction/compilation and topology construction are cold setup. Warmup exercises every actual phase-space point and callback layout. Each scope uses retained handles and buffers. Ordinary allocations inside the measured numerical path remain included: original Ninja's per-evaluation work, Rust reduction/workspace allocations, descriptor/key vectors, copies, validation, FFI and output materialization are never subtracted. Neither path is claimed allocation-free. Buffer capacities may be retained; numerical values and coefficient bindings must change with inputs.

The shared reduction record freezes the union of requests actually made by both reducers. Complete shared-live calls unconditionally bypass this table. Production OneLOop must be built with Ninja's master-result cache disabled; direct SymbolicaMasters does not use the validation table. No result-cache hit may stand in for a complete repeated-point evaluation. Separate counter deltas report preparation calls/rows/wavefunctions, numerator calls/rows, wire master calls/rows, and successful live backend rows where observable.

Repeated workload: the first physical point in every lane and iteration. Changing workload: actual momentum/model rows cycle over all three distinct points, with fixed `pp` helicity; the starting phase rotates between iterations. This is phase-space variation, not helicity variation. Required widths are 1, 8, 32, 128, 1024, with 31 as an incomplete tail. Width zero is exercised by `physical_probe.cc`, not by the timing driver. Every reduction scope is checked against retained Laurent references before sampling. No rescue backend or implicit precision change is enabled; failures abort and are reported rather than omitted.

Timing warmup checks integrated values and internal finite-split identities; it does not execute the independent raw numerator grid or Float256 coefficient oracle. `run_timing.py` records the separate physical probe report, its hash and dependency revisions, plus whether its runtime sources match the current run. The archived `shared-physical-provider-probe.json` preserves the earlier width-zero, raw-grid and oracle controls at its original pins. A current `run_physical.py` result can be supplied with `--physical-controls`; historical evidence is not silently promoted to a current-pin validation result.

Paired blocks use equal rounds, the same input order, and alternating original-first/Rust-first order with balanced counts. Each scalar original call in a width-N block processes one lane; Rust receives the complete width-N batch. Driver I/O, correctness comparisons and statistics calculation stay outside sampled regions. For shared reduction, three immutable coefficient/input states are prepared once before warmup. Every measured call selects one state, then passes its changed shifts/masses/rows through the same retained original amplitude or Rust reducer/session. There is one reducer workspace per implementation, not three per-point workspaces. State selection, geometry recomputation, callbacks and ordinary input staging remain inside both clocks. The other scopes time complete repeated calls, with preparation inside. Raw times are retained. Summaries report median/MAD latency, throughput and a paired bootstrap95% interval for the Rust/original ratio. The interval describes observed blocks and does not prove host isolation.

The current driver puts one monotonic clock pair around the entire repeated block. Provider counters and checksum reads are outside that block. Round counts must be multiples of three, so each phase-space point occurs equally often, including when balanced process order changes. The default is 384 rounds; `minimum_observed_block_ns` is the measured contiguous block duration. Ordinary loop/selection overhead is included, and clock-pair overhead is reported without subtraction. These changes have a new source/evidence identity; the retained `dba7411` preliminary diagnostic used one clock pair per call and is not reinterpreted as contiguous-block evidence.

The standalone release profile now matches the main workspace: `opt-level=3`, thin LTO and one codegen unit. The earlier archived diagnostic used the standalone default release profile without explicit LTO/codegen overrides; its results remain immutable and separate. This is still a validation bridge, not unmodified production CLI latency.

The runner records compiler versions/flags, complete runtime source and artifact hashes, dependency revisions, CPU model, allowed/chosen affinity, SMT sibling mapping, load averages, `/proc/stat` snapshots and process names/CPU usage without command-line arguments. It never terminates unrelated workloads. Pinning one thread does not reserve its CPU or SMT sibling; unrelated work or frequency/host interference keeps any resulting measurements diagnostic. No fast-math is used. Default execution performs correctness checks only; no acceptance timing is run implicitly.

The executable and runner are `physical_timing.cc` and `run_timing.py`. The default invocation runs correctness only. For example:

```sh
python3 validation/performance/shared_provider/run_timing.py \
  --release --output-dir target/performance/workload-check --widths 1,8
python3 validation/performance/shared_provider/run_timing.py \
  --release --allocation-diagnostics --output-dir target/performance/workload-allocations --widths 1,8
```

An explicit `--timing --cpu CPU_NUMBER` invocation builds optimized Rust and C++ artifacts and defaults to the shared-reduction scope only. Correctness, allocation and execution-diagnostic modes default to all four scopes. Explicit `--scopes` is always honored and recorded; complete-provider timing must be requested deliberately because master evaluation is much more expensive. It must be scheduled separately from builds and host workloads; the runner always keeps `performance_acceptance=false`. `--scopes 0,1,2,3`, `--widths`, `--rounds` and an even `--trials` control the recorded workload. Blocks below20ms are flagged. No timing run is performed by the correctness or allocation commands.

Allocation diagnostics use a separate feature build with a System-delegating Rust `GlobalAlloc` wrapper. Primary timing builds contain no wrapper or diagnostic allocation scopes. The recorded columns are allocation calls/requested bytes, reallocation calls/requested bytes, and deallocation calls. Totals and nonoverlapping numerator/master/preparation scopes are reported; the residual contains Rust reduction and validation transport/staging. These are Rust allocator observations only: original C++ allocations and native GMP/MPFR allocations that bypass Rust are unobserved. Consequently, the diagnostic helps locate Rust work and cannot establish whole-process allocation parity.

## Separate execution diagnostics

`--execution-diagnostics` builds clocks/counters into a separate artifact. That artifact refuses primary `timing` mode, and ordinary timing builds contain none of these spans. One fully warmed call per implementation, width and workload is observed at phase index 1; the preceding correctness warmup covers all three phases. This is an attribution diagnostic, not a second source of acceptance latency. Clock/counter instrumentation perturbs both stages and their remainder; its cost is not subtracted or extrapolated into the uninstrumented timing result.

Each backend family records actual calls, evaluated rows and integer monotonic nanoseconds. `A0`, `B0`, `B1`, `B11`, `B111`, `C0`, `D0`, `DB0` , `An0..4` and a generic `An_other` higher-rank bucket have independent Rust records for shared-original, shared-Rust and direct-Rust dispatch. Original OneLOop rank-two/three bubble methods return multiple heads together, so they retain grouped records `B0+B1+B11` and `B0+B1+B11+B111`. Their costs are not invented individual-family timings. Original scale initialization has its own record and remains in the unassigned remainder.

The additive breakdown is measured within the same outer call:

- `preparation_total_ns` surrounds all of `Physical::prepare`, including loading raw momenta/helicities/model rows, Rust preparation, and output staging. Its inner `preparation_provider_ns` and their difference are nested details, not additional summands.
- `numerator_provider_abi_ns` surrounds common numerator FFI decoding, callbacks and output writing.
- `master_backend_ns` surrounds actual provider execution, including ordinary provider validation/conversion and expression arithmetic. Frozen replay must have exactly zero backend calls, rows and nanoseconds.
- `master_lookup_transport_ns` is the measured shared Rust master ABI interval minus its own nested backend interval. It includes exact-key lookup and data conversion/copying. Caller-side packing is in the remainder; frozen lookup is never labeled master computation.
- `remainder_ns` contains core reduction, contraction, caller-side staging/transport, initialization and instrumentation overhead. It is not a claim of isolated arithmetic time.

These five nonoverlapping values sum exactly to `overall_ns`. `non_master_elapsed_ns` subtracts actual backend execution only; `reduction_excluding_master_and_preparation_ns` additionally subtracts full numerical preparation. They retain numerator callbacks and fixed-provider overhead, matching the inclusive reduction control. No stage is inferred by subtracting separate noisy runs. `execution_report.py` checks the dimensions, nesting, sums and zero-backend invariant before publishing a report.

```sh
python3 validation/performance/shared_provider/run_timing.py \
  --release --execution-diagnostics --widths 1,8 --scopes 1,2,3 \
  --output-dir target/performance/workload-execution
```

## Retained preliminary diagnostic

The October 16, 2026 run used Symbolica `2c06f245`, oneloopmaster `bf7632a9`, CPU 35 and one thread, with four balanced pairs at widths 1 and 8. Other host jobs remained active and the CPU/SMT sibling was not reserved. That original source used default release profile settings and one round per measured interval; the later methodology changes above do not alter its identity. This is bottleneck evidence, not an acceptance result. Medians below are per batch for changing phase-space points; paired ratios are calculated from paired samples, not from the ratio of displayed medians.

| Scope | Width | Original | NinjaRS | Paired Rust/original median |
|---|---:|---:|---:|---:|
| Preparation | 1 | 3.90 µs | 3.90 µs | 1.000 |
| Preparation | 8 | 29.66 µs | 29.63 µs | 0.998 |
| Shared reduction/replay | 1 | 108.68 µs | 374.03 µs | 3.442 |
| Shared reduction/replay | 8 | 850.35 µs | 2674.82 µs | 3.146 |
| Complete shared-live | 1 | 49.11 ms | 50.34 ms | 1.027 |
| Complete shared-live | 8 | 356.14 ms | 344.12 ms | 0.966 |
| Complete production backends | 1 | 0.304 ms | 50.484 ms | 161.800 |
| Complete production backends | 8 | 1.749 ms | 343.194 ms | 200.678 |

Production master execution dominates the complete Rust path in this control, while the common-provider reduction control independently exposes a reduction cost gap. Preparation is a small part of both complete paths. The allocation run observes 162 Rust allocations at width one for complete production evaluation: 25 in the numerator, 6 in preparation, none in direct master execution, and 131 in the residual reduction/transport scope. At width eight the corresponding total is 559, with 30 numerator, 48 preparation and 481 residual allocations. These counts are allocation-inclusive diagnostics, not whole-process allocation comparisons.

The independent release physical probe passes 210 raw numerator samples, the Float256 coefficient oracle and 3,681 Laurent components across widths 0, 1, 3, 8, 31, 32, 128 and 1024. The workload probe passes all 16 scope/workload/width combinations. The standalone seven Rust tests include the repaired exceptional Euclidean C0 point, and eight statistics tests pass; strict all-feature Clippy and formatting pass.

Evidence indexes are [`shared-physical-release-probe.json`](../../evidence/shared-physical-release-probe.json), [`shared-workload-release-correctness.json`](../../evidence/shared-workload-release-correctness.json), [`shared-workload-allocations.json`](../../evidence/shared-workload-allocations.json), and [`shared-workload-diagnostic-timing.json`](../../evidence/shared-workload-diagnostic-timing.json). Each workload index links a deterministic gzip archive retaining the complete raw samples, source hashes and host snapshots with compressed/uncompressed hashes. Earlier failing or successful evidence remains unchanged. Final gates still require new evidence from the aligned release profile and contiguous longer blocks, all requested widths, independent runs, and suitable host conditions; rational-separation component parity remains incomplete.

## Methodology validation before optimized sampling

The new driver passes all 16 developer-profile scope/workload/width controls at widths 1 and 8, both without instrumentation and with execution accounting. Maximum scaled Laurent error is `2.236549047241113e-16`; both source guards pass, frozen union misses are zero, and backend call/row deltas are zero. Named source assertions verify the intended provider for every scope. Nine Rust tests, eight accounting tests, eight statistics tests, strict Clippy and formatting pass. This validates the measurement machinery; it does not establish an optimized timing result.

For changing inputs, both reducers request 10 direct, 25 T3 and 20 T2 numerator rows per point. At width one, original Ninja makes 10/25/20 callback calls and NinjaRS makes 5/15/10. At width eight, original makes 80/200/160 calls while NinjaRS retains 5/15/10 calls over the same 80/200/160 rows. Frozen master lookup calls are 50 versus 5 at width one and 400 versus 5 at width eight; actual master backend calls, rows and time are exactly zero. These provider costs remain included in the reduction total.

The [methodology validation index](../../evidence/shared-workload-methodology-validation.json) retains exact source identities and deterministic archives. Its execution durations are developer-profile accounting controls and must not be used for speed comparisons. The separate pool/static-predicate core checkpoint and optimized contiguous sampling are subsequent validation stages.


## Reduction selection

`--reduction-kernels symbolica|native-denominators|native` selects reduction
factories only, independently of the fixed shared numerator and master scope.
The default is interpreted and the standalone Cargo feature is optional.
`--native-catalog` enables that feature while selecting interpreted execution,
so a matched three-way experiment can use identical native-enabled library and
C++ artifacts. The C++ runner accepts the same optional last argument; omitting
it preserves interpreted behavior. Session selection and coverage validation
happen during setup, before warmup or any block.

Every JSON report includes the actual eight-family backend report and the
compiled feature flag. Both frontends validate those values. The currently
available `native` catalog additionally executes ordinary bubble source
projections and triangle-parent corrections; other reduction, integration and
accumulation remain interpreted. No metadata calls occur inside timed blocks.
The `run_physical.py --all-reduction-kernels` check supplies a single-artifact
frozen-master parity control. A separate controlled timing window and explicit
artifact/source identity are still required for performance conclusions.

For exact same-artifact timing comparisons, reuse one copied `physical-timing`
executable and its library, changing only its final selection argument. Separate
`run_timing.py` output directories rebuild the small C++ launcher with different
library search paths; do not label those executable hashes as identical.

Guarded rank selectors are explicit and topology-limited: `interpreted-rank`
uses interpreted generic kernels and alternates; `native-rank` uses the generic
native catalog and native alternates. The latter requires the native feature.
Their reports add conditional-alternate metadata alongside the existing generic
family coverage. The rank-mode wiring passes bounded correctness controls
recorded in `validation/evidence/rank-specialization-harness.json`, including
the five-mode physical comparison. It has no timing evidence and changes no
scopes, counters, clock boundaries, master policy, or preparation/numerator
provider. Use a separately coordinated same-artifact comparison before any
performance conclusion.

The independent `--native-block-rows N` option passes a positive complete-kernel
row bound through session setup and records the configured `native_execution`
policy. Omission retains Whole. It changes no timing boundaries, counters,
master policy or numerator provider. Before sampling, use the physical runner's
same-artifact Whole/FixedRows control for the explicitly selected native mode.
Development execution controls are recorded in
`validation/evidence/native-block-harness.json`; they provide no timing evidence.
Bounds are never selected by timed autotuning.

`run_timing.py` continues to accept historical separate physical-control files.
For a future fixed-row measurement, explicitly supply a new physical report
matching the current source, selected mode and exact bound, including its
same-artifact Whole/FixedRows parity rows. Timing warmup and configuration
metadata do not establish that parity. The preselected performance bound is
32; the bound 7 used by development correctness tests stresses tails and is
not evidence for bound 32. Release correctness must validate the selected bound
on the immutable artifact used for the subsequent comparison.

## Optional swapped arithmetic configuration

The validation runner accepts `--prune-swapped-arithmetic` independently of
`--reduction-kernels` and `--native-block-rows`. It defaults off and is passed
only when constructing the retained session. The timing loops and scopes are
unchanged: preparation, frozen-master reduction, shared live complete
evaluation, and production complete evaluation remain separate; diagnostic
master timing stays separate from primary wall time. Selected swapped graphs
remain interpreted under every native selection and policy. Their JSON field
reports configured coverage, not execution counts; existing family metadata
continues to describe full-head paths.

The matching `run_physical.py --prune-swapped-arithmetic` control must pass
before timing this option. Its full/pruned comparison uses the same executable,
prepared numerator and frozen masters and checks all five result heads.
When pruning is enabled, the runner rejects physical controls with missing or
mismatched full/pruned rows, selection, native policy, feature state or runtime
source map before building. Every requested timing width must have been checked
by those physical controls. Fixed-row requests additionally require both native
policy controls. Historical default-off control handling remains unchanged.
The optimized physical controls and actual-report preflight pass at frozen
`8a33a59`; see
[`pruned-swapped-harness.json`](../../evidence/pruned-swapped-harness.json).
The timing launcher also compiles with strict C++ warnings, but was never run.
Pruning timings remain pending. Existing archived timing evidence is unchanged
and cannot be attributed to pruning.
