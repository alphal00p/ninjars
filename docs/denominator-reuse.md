# Same-call denominator expansion reuse

Bubble and tadpole kernels expose an additive
`divide_with_corrections_batch` operation. Existing standalone division and
correction methods remain available and recompute their denominators. Existing
custom kernel implementations inherit a default implementation that divides,
compacts successful target groups, and makes one batched correction call.

The core coordinator constructs borrowed parent descriptors before division;
these contain no evaluated arithmetic. Parent provenance and numerical
validation are still deferred until every required orientation of the target
has divided successfully. Failed targets do not cause previously skipped parent
errors to be evaluated. Output order is unchanged.

The Symbolica implementation retains the raw division's denominator buffer and
indexes the relevant uncut lines during corrections. The bubble uses seven
T2 coefficients per line; the tadpole uses four T3 coefficients. The correction
requests must share the current division request's live vectors, parameters,
shifts and mass slices, with the same target anchor and ordered layout. Live
reference identity is only a conservative sufficient condition for reuse;
distinct allocations trigger ordinary recomputation even if values compare
equal. The parent residue's origin remains a separate source-projection input.
No numerical equality, primal-only Dual equality, persistent cache, epoch,
unsafe lifetime conversion, or new algebra is involved.

The buffer is reused only inside this one call. Packed row offsets are reset
at the next division; standalone corrections cannot consume retained results.
All existing per-value shape/finiteness checks, leading-coefficient checks,
active-lane compaction and output checks remain. Numerical arithmetic still
passes through the existing Symbolica plans or their generated native bodies.

For a fully stable n5/rank4 lane, bubble denominator rows decrease from 120 to
60, and tadpole rows from 100 to 20. For n2/rank2 and n2/rank3, there are no
bubble denominators; tadpole rows decrease from four to two and eight to four,
respectively. These are executed-row counts, not timing or speedup claims.
The exact row counters exist only in tests, with no production instrumentation.

Combined-call buffers are stage scratch. Buffer-shape errors are rejected before
writes, but a later selected-parent error may leave successful raw quotients and
statuses written, as with the previous sequence of separate calls. Published
reducer residues and accumulated amplitude results retain their existing
transactionality. Tests distinguish selected malformed parents from identical
malformed descriptors in failed targets, and exercise successful retries.

The default compatibility implementation allocates compact temporary request and
output buffers. Production Symbolica overrides use existing retained buffers
and a reusable integer-offset vector. No runtime-performance acceptance is
claimed before matched measurements with fixed callbacks and master policies.

## Validation checkpoint

Seven new unit groups compare the optimized operation with a wrapper that uses
the inherited compatibility default. They check all residue components, exact
executed denominator counts, f64/DoubleFloat/Float metadata, higher and mixed
Dual coefficients, standalone calls, distinct-borrow fallback, selected and
skipped malformed parents, failed-lane retries, and empty/zero-parent batches.
Eight existing suites add 38 passing regression tests, including late callback
failure, workspace reuse, high-parent cancellation and native dispatch.

Strict package Clippy, feature-off library compilation, formatting and whitespace
checks pass. Existing dependency warnings remain in Numerica's `double.rs` and
Symbolica's `atom.rs`; no dependency sources are changed here. The reproducible
commands, dependency revisions, source and executable hashes, and retained logs
are recorded in
[`denominator-reuse.json`](../validation/evidence/denominator-reuse.json).

Optimized physical controls and a subsequent frozen-source timing schedule are
retained in [the matched reduction report](../validation/performance/shared_provider/REUSE-TIMING-553f406.md).
All 576 individual blocks passed source/artifact guards and had zero actual
master computation or frozen-union misses. The native catalog remains 2.10–2.73
times original Ninja on the changing-point graph26 workload. These shared-host
diagnostics do not meet the performance target. Older pre-reuse runs were not
paired with this schedule and do not establish a causal before/after speedup.
