# Physical timing boundaries

This harness compares the same verified gg→HHH graph26, coefficient basis and numerator provider, using the frozen three-point physical pool. It must first pass its default correctness-only mode; timing is an explicit separate invocation. Timing output is diagnostic until the host/CPU conditions and the project acceptance gates are independently reviewed.

Four records distinguish numerical work instead of inferring a stage by subtracting two noisy measurements:

| Scope | Included in each measured call | Excluded |
|---|---|---|
| `preparation` baseline | Raw input staging, FeynKit wavefunctions, dependent parameters, factored multihead SymJIT coefficients, shifts/masses/S matrix, numerator binding | Reduction and masters |
| `shared-reduction` | Ordinary prepared-input staging, original/Rust reduction, the identical out-of-line Rust numerator provider, frozen union master lookup, contraction and output staging | Physical numerical preparation and master expression execution |
| `shared-live-complete` | Full physical numerical preparation plus reduction, the identical out-of-line numerator provider and identical live Symbolica master provider, output staging | Only setup and driver I/O |
| `production-complete` | Full physical numerical preparation and common numerator provider, original Ninja/OneLOop versus NinjaRS/direct SymbolicaMasters, output staging | Only setup and driver I/O |

`production-complete` compares production master backends while retaining a deliberately common validation numerator provider. It is not the latency of the unmodified production CLI: the common validation FFI for numerator callbacks and the normal C++/Rust boundary staging remain included. There is no production core FFI or new algebra.

Both paths receive the same batched physical preparation at the record's width, including coefficient-head evaluation. Original Ninja subsequently reduces one lane at a time; NinjaRS receives the whole reduction batch. This intentionally gives original Ninja the same preparation batching benefit. Each result records `original_preparation_batching`, `shared_preparation_width` and the separate reduction batch widths.

Graph import, Feynman rules/contraction, exact coefficient preparation/factoring, evaluator construction/compilation and topology construction are cold setup. Warmup exercises every actual phase-space point and callback layout. Each scope uses retained handles and buffers. Ordinary allocations inside the measured numerical path remain included: original Ninja's per-evaluation work, Rust reduction/workspace allocations, descriptor/key vectors, copies, validation, FFI and output materialization are never subtracted. Neither path is claimed allocation-free. Buffer capacities may be retained; numerical values and coefficient bindings must change with inputs.

The shared reduction record freezes the union of requests actually made by both reducers. Complete shared-live calls unconditionally bypass this table. Production OneLOop must be built with Ninja's master-result cache disabled; direct SymbolicaMasters does not use the validation table. No result-cache hit may stand in for a complete repeated-point evaluation. Separate counter deltas report preparation calls/rows/wavefunctions, numerator calls/rows, wire master calls/rows, and successful live backend rows where observable.

Repeated workload: the first physical point in every lane and iteration. Changing workload: actual momentum/model rows cycle over all three distinct points, with fixed `pp` helicity; the starting phase rotates between iterations. This is phase-space variation, not helicity variation. Required widths are 1, 8, 32, 128, 1024, with 31 as an incomplete tail. Width zero is exercised by `physical_probe.cc`, not by the timing driver. Every reduction scope is checked against retained Laurent references before sampling. No rescue backend or implicit precision change is enabled; failures abort and are reported rather than omitted.

Timing warmup checks integrated values and internal finite-split identities; it does not execute the independent raw numerator grid or Float256 coefficient oracle. `run_timing.py` records the separate physical probe report, its hash and dependency revisions, plus whether its runtime sources match the current run. The archived `shared-physical-provider-probe.json` preserves the earlier width-zero, raw-grid and oracle controls at its original pins. A current `run_physical.py` result can be supplied with `--physical-controls`; historical evidence is not silently promoted to a current-pin validation result.

Paired blocks use equal rounds, the same input order, and alternating original-first/Rust-first order with balanced counts. Each scalar original call in a width-N block processes one lane; Rust receives the complete width-N batch. Driver I/O, correctness comparisons and statistics calculation stay outside sampled regions. For shared reduction, preparation is performed outside each sampled call; the small clock-pair overhead is retained and reported separately, not subtracted. The other scopes time complete repeated calls, with preparation inside. Raw times are retained. Summaries report median/MAD latency, throughput and a paired bootstrap95% interval for the Rust/original ratio. The interval describes observed blocks and does not prove host isolation.

The preliminary driver clocks each call separately and sums those durations within a reported block, with counter snapshots between calls. Consequently, its `minimum_observed_block_ns` is a sum of measured call intervals, not a contiguous block duration when rounds exceed one. An acceptance runner must move counters outside a contiguous repeated block and clock the complete block. The first diagnostic uses one round per block and four balanced blocks; this is deliberately insufficient for final performance acceptance.

The standalone Cargo workspace currently uses the default release profile (`opt-level=3`, no explicit LTO or codegen-unit override). The main CLI instead requests thin LTO and one codegen unit. Both actual manifests and compiler commands are retained; this profile difference must be aligned or independently controlled before a final gate. These preliminary measurements are not unmodified production CLI latency.

The runner records compiler versions/flags, complete runtime source and artifact hashes, dependency revisions, CPU model, allowed/chosen affinity, SMT sibling mapping, load averages, `/proc/stat` snapshots and process names/CPU usage without command-line arguments. It never terminates unrelated workloads. Pinning one thread does not reserve its CPU or SMT sibling; unrelated work or frequency/host interference keeps any resulting measurements diagnostic. No fast-math is used. Default execution performs correctness checks only; no acceptance timing is run implicitly.

The executable and runner are `physical_timing.cc` and `run_timing.py`. The default invocation runs correctness only. For example:

```sh
python3 validation/performance/shared_provider/run_timing.py \
  --release --output-dir target/performance/workload-check --widths 1,8
python3 validation/performance/shared_provider/run_timing.py \
  --release --allocation-diagnostics --output-dir target/performance/workload-allocations --widths 1,8
```

An explicit `--timing --cpu CPU_NUMBER` invocation builds optimized Rust and C++ artifacts and samples the four scopes. It must be scheduled separately from builds and host workloads; the runner always keeps `performance_acceptance=false`. `--scopes 0,1,2,3`, `--widths`, `--rounds` and an even `--trials` control the recorded workload. Blocks below20ms are flagged. No timing run is performed by the correctness or allocation commands.

Allocation diagnostics use a separate feature build with a System-delegating Rust `GlobalAlloc` wrapper. Primary timing builds contain no wrapper or diagnostic allocation scopes. The recorded columns are allocation calls/requested bytes, reallocation calls/requested bytes, and deallocation calls. Totals and nonoverlapping numerator/master/preparation scopes are reported; the residual contains Rust reduction and validation transport/staging. These are Rust allocator observations only: original C++ allocations and native GMP/MPFR allocations that bypass Rust are unobserved. Consequently, the diagnostic helps locate Rust work and cannot establish whole-process allocation parity.

## Retained preliminary diagnostic

The October 16, 2026 run used Symbolica `2c06f245`, oneloopmaster `bf7632a9`, CPU 35 and one thread, with four balanced pairs at widths 1 and 8. Other host jobs remained active, the CPU/SMT sibling was not reserved, and the profile and sampling limitations above apply. This is bottleneck evidence, not an acceptance result. Medians below are per batch for changing phase-space points; paired ratios are calculated from paired samples, not from the ratio of displayed medians.

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

Evidence indexes are [`shared-physical-release-probe.json`](../../evidence/shared-physical-release-probe.json), [`shared-workload-release-correctness.json`](../../evidence/shared-workload-release-correctness.json), [`shared-workload-allocations.json`](../../evidence/shared-workload-allocations.json), and [`shared-workload-diagnostic-timing.json`](../../evidence/shared-workload-diagnostic-timing.json). Each workload index links a deterministic gzip archive retaining the complete raw samples, source hashes and host snapshots with compressed/uncompressed hashes. Earlier failing or successful evidence remains unchanged. Final gates still require the production release profile, contiguous longer blocks, all requested widths, independent runs, and suitable host conditions; rational-separation component parity remains incomplete.
