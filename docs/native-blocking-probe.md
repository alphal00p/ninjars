# Native executor blocking probe

This is a test-only scheduling experiment. Production dispatch, existing
reduction expressions and their generated bodies are unchanged. No performance
result or acceptance claim is established by the source audit.

The n5/r4 coordinator expands a phase-space lane into many cut/parent rows.
Even width one therefore enters the generated instruction-major batch path:
bubble correction evaluates 60 rows, not one. The current degree-one correction
has 56 retained slots. Its scratch alone occupies 52.5 KiB at phase width one
and 52.5 MiB at width 1024 for complex f64. The executor packs row-major inputs
into contiguous slot columns and unpacks outputs at each kernel boundary.

For the seven selected ordinary native executor instances, the pre-reuse
high-water scratch estimate is 240.47 KiB per phase-space lane:

| Executor | Slots | Retained rows per phase-space lane |
|---|---:|---:|
| Bubble T2 denominators | 50 | 60 |
| Triangle T3 denominators | 39 | 40 |
| Tadpole T3 denominators | 39 | 60 |
| Bubble source projection | 48 | 60 |
| Bubble degree-one correction | 56 | 60 |
| Tadpole triangle source projection | 41 | 30 |
| Tadpole bubble source projection | 51 | 20 |

Same-call denominator reuse lowers the tadpole denominator maximum to 20 rows,
reducing this estimate to 216.09 KiB per lane. These estimates exclude the
coordinator's input, output and intermediate buffers, other interpreted
workspaces, constants, allocations' spare capacity and numerical payloads beyond
plain complex f64. They do not measure memory traffic or predict speedup.

The source locations are the `workspace.prepare` literals in
`crates/ninjars-symbolica/src/native_kernels/generated`, together with
`references/oneloopmaster/src/native/batch.rs`. The latter retains the largest
row count as its stride; merely slicing an already enlarged workspace cannot
recover the bounded footprint.

The prototype decorates newly installed native executors only in `cfg(test)`.
It validates the complete input/output lengths, then passes contiguous whole-row
slices to the unchanged executor. Block sizes are 8, 16, 32 and 64, with an
unchanged whole-batch control. Natural tails are retained, including a one-row
tail that invokes the existing scalar generated body. Each variant owns a fresh
workspace. A block size of 32 bounds degree-one correction scratch to 28 KiB;
the seven arenas together require at most 162 KiB before allocator overhead.

The controls compare exact ordinary f64/DoubleFloat components, signed zero,
nonfinite classification, every ordered higher/mixed Dual component and Float
values with tracked precision. They exercise empty batches, cut-expanded sizes,
permutations, shrink/retry sequences and all degree classes. Separate full
reducer controls compare residue coefficients and lane failure statuses with a
deterministic master-value stub, so actual master arithmetic is zero. The stub
is only a contraction/control input and does not validate physical master values.

A new fixture is generated mechanically by `generate_blocking_fixture` from
Symbolica's conditional evaluator and a non-inlined inverse helper. Its counter
checks that each row executes exactly one reciprocal, while the inactive arm
contains a zero denominator. It covers mixed branches, shared helpers, changing
batch sizes and natural scalar tails. This adds no handwritten algebra and does
not modify the existing generated reduction catalog.

The ignored `native_blocking_microbenchmark` requires release mode,
`NINJARS_BLOCKING_QUIET_WINDOW=approved`, and a new `NINJARS_BLOCKING_OUTPUT`
JSONL path. Run it only in a separately coordinated quiet window. It compares the
seven n5/r4 executor instances after denominator reuse at phase widths
1/8/32/128/1024/31. Setup is excluded, every variant is warmed independently,
and all raw timings and measured workspace capacities/cell counts are retained.
Each timing record is flushed immediately; a final completion record distinguishes
a complete run. Source hashes, executable hash, dependency revisions/status,
compiler, host and CPU affinity are recorded before and after the run. The
blocked variants include the test decorator dispatch and per-chunk shape checks;
this overhead is part of the measured scheduling experiment.
Synthetic row-major matrices contain distinct deterministic rows but repeat
between rounds. This isolates kernel packing, arithmetic and unpacking; it does
not measure changing physical points, complete reduction, numerator preparation,
callbacks or actual masters. Subsequent end-to-end shared-provider measurements
are required before selecting any production policy.

A second architectural candidate remains repeated packing/validation across
stage boundaries. Bubble source projection, correction, parent summation and
residue projection each repack their inputs; public request validation also
rechecks topology and full lane data per parent. Those costs are separate from
this blocking experiment. Any later fusion or trusted internal path must retain
public standalone contracts, full-component validation and failed-lane isolation.

Validation of source commit `7e68928099b1027f8694f9c7627431d59a566a78`
passed eight focused controls, with the microbenchmark ignored. Each of f64,
DoubleFloat and Float192 compared 421,872 complex components exactly across
whole/chunked execution. Strict package Clippy (including the generator example),
formatting and the feature-off library check passed. Raw logs, source/dependency
identities and the test executable hash are retained in
[the evidence record](../validation/evidence/native-blocking/summary.json).

Native Dual controls use the existing ordered shape
`[[0,0],[1,0],[0,1],[2,0],[1,1]]`. The conditional fixture covers ordinary
f64/DoubleFloat/Float; the supported native Dual reduction bodies tested here
are straight-line. Validation was performed in the debug profile. Optimized
correctness controls and coordinated timings remain required before interpreting
performance or selecting a production policy.
