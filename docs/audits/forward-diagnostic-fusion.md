# Forward fusion with retained diagnostic outputs

Status: source-only candidate based on `b2de05e`. No build, symbolic inventory,
numerical evaluator or timing has run for this follow-up. Source review precedes
execution. This extends the bounded n3/r3 forward control from the
[transpose experiment](transpose-probe.md), without running another transpose
experiment or extending the topology.

The historical five-output graph used 1,075 instructions, compared with 1,954
summed instructions for separate forward blocks. Those counts omit many outputs
inspected by the current reducer. The new ignored test asks whether that
instruction-count saving survives a conservative diagnostic superset: **retain
every output head of every existing forward block**.

The comparison uses one unchanged exact network for all three inventories:

1. Separate forward blocks, including all their currently generated heads.
2. Fused forward expressions for the five integrated outputs.
3. Fused forward expressions for those same five outputs first, followed by all
   remaining block output heads exactly once.

The report records each retained symbol, block/category, and half-open output
range. Categories cover division, parent correction, subtraction sums, residue
projection, master weights, individual master contractions, and the final
amplitude sum. The retained set deliberately includes more than a minimal
diagnostic interface; this experiment does not optimize that interface. The
current ordinary tadpole has five output slots but only its constant coefficient
is computed; its four zero slots do not become new reconstructed coefficients.

## Exact graph boundary

The implementation is a nested ignored-test module,
`algorithm_transpose_probe::diagnostic_forward`, using the existing
`forward_network`, `Network::compose`, instruction inventory, and oneloopmaster
emitter. There is no new polynomial manipulation, reduction formula, optimizer,
or generated arithmetic. The algebra reuse audit in the original probe applies:
Symbolica performs substitutions, exact expansion and instruction optimization.

All three graphs keep the same 60 independent Laurent callback heads, the same
39 independent symbolic master Laurent components, and the same complete
external input list. Cut/edge/orientation identities and the same-call shared
denominator identities stay unchanged. Masses and pair invariants retain their
existing physical propagator identities. All other geometry remains abstract.
Denominator coefficients and source projections are inputs: neither their
calculation nor physical geometry construction is composed into this network.

The first five retained expressions must equal the five-output baseline exactly.
For every retained block, the test independently rebinds that block's existing
exact expressions to retained upstream values and external inputs. Each
difference must expand to literal Symbolica zero. It also rejects unresolved
intermediate symbols and any abstract master dependence before a contraction.
The complete category and exact-identity inventories are retained in the report.
These controls check symbolic graph wiring; they do not independently validate
the original exact reduction formulas, which are shared by all three graphs.

Master values remain independent symbols. No master provider is called or
compiled into the graph. Retained weights can support later lazy request
selection, but this experiment does not implement that control flow or authorize
eager evaluation of a zero-weight master.

## What output retention can and cannot preserve

| Current contract | Arithmetic values retained | Remaining control outside this probe |
| --- | --- | --- |
| Division result finiteness | Every quotient head, including unused/swapped heads | Callback/input shape and all-component finite checks; denominator construction and leading-coefficient checks before division |
| Parent correction validation | Every parent correction head and every source residue coefficient | Source geometry/projection finiteness, topology/provenance/containment, computed masks, and failed-parent eligibility |
| Corrected quotient and residue validation | Every subtraction result and all current residue slots | Validation order, active-lane compaction, intermediate numerical overflow and transactionality |
| Integration | All weights, every cut contribution, and all five amplitude outputs | Lazy master selection, derivative-only weights, provider failures, nonanalytic points, accumulation status |
| Geometry and diagnostics | Existing abstract geometry inputs remain unchanged | Gram/pivot/branch selection, cut-stop, reconstruction execution, precision rescue |

Relevant owning code remains `triangle_kernels.rs::Engine::divide`,
`bubble_kernels.rs::Engine::{divide,corrections}`,
`tadpole_kernels.rs::Engine::{divide,correction}`, the corresponding core reducers
in `crates/ninjars/src`, and `ResidueIntegrator::evaluate_batch`'s weight-based
request selection. Keeping an algebraically equal quotient as an output does
not reproduce a failure caused by a different intermediate rounding/overflow
path. Exact identities do not establish signed-zero, conditioning, full Dual,
branch, batching, or failure-status equivalence.

## Bounded execution plan

Both compositions retain the original cumulative **8 MiB expression-text cap**.
Rebinding all local identity expressions has a separate cumulative cap of the
same size. Initial, separated, five-output, identity and optimization checkpoints
remain clearly incomplete until every stage succeeds. Per-stage and per-block
log messages locate failures. A byte cap cannot interrupt a single expensive
Symbolica operation; the ignored executable must therefore run under the same
**180-second external timeout inside the numerical lock**.

After source review and explicit release, compile the test executable outside
the numerical lock, with build jobs at most four:

```sh
cargo test -p ninjars-symbolica --features rust-codegen --lib --no-run --message-format=json
```

Then run only the new ignored test, with the executable path returned by Cargo,
the configured Symbolica license, and a retained output path:

```sh
export NINJARS_FORWARD_DIAGNOSTICS_OUTPUT=/tmp/ninjars-forward-diagnostics/inventory.json
flock /tmp/ninjars-symbolica-tests.lock timeout --signal=TERM --kill-after=10s 180s \
  "$test_executable" --ignored --exact \
  algorithm_transpose_probe::diagnostic_forward::inventory_forward_diagnostics_n3_r3 \
  --test-threads=1 --nocapture
```

Retain source/dependency revisions, executable hash, command, raw log, partial or
complete JSON, and exact exit status. If a cap or timeout is reached, retain it as
an incomplete result and stop; do not enlarge bounds or extend to n3/r4. Strict
focused Clippy and package formatting follow build validation, outside any timing
window. The prior transpose test is not part of this execution plan.

The result will compare instructions, additions, multiplications, inversions,
local temporaries, expression size, and generated source size. Signed instruction
differences make loss of the original saving visible. Separate temporary sums
are not peak storage, and output stores/packing are not timed. No production
dispatch, default policy, performance acceptance, or runtime speedup follows
from these counts.
