# Explicit native batch execution

The `native-kernels` feature exposes oneloopmaster's `BatchExecutionPolicy` through
`ninjars_symbolica`. The default is `Whole`: each native executor calls its existing
generated batch entry directly. Blocking is selected explicitly at amplitude
construction, independently of the native catalog and guarded rank specialization:

```rust,ignore
use std::num::NonZeroUsize;
use ninjars_symbolica::{BatchExecutionPolicy, SymbolicaAmplitudeKernels};

let kernels = SymbolicaAmplitudeKernels::new(5, 4, 0.0)?
    .with_native_kernels()
    .with_native_batch_execution(BatchExecutionPolicy::FixedRows(
        NonZeroUsize::new(32).unwrap(),
    ));
let amplitude = kernels.into_amplitude(0)?;
```

The Dual amplitude factory exposes the same builder. Either ordering with native
catalog or rank-specialization builders preserves the requested policy; it is
applied after each stage has been constructed. `native_batch_execution()` reports
the request, while the existing backend reports identify native coverage and
unsupported-Dual interpreter fallbacks. Interpreted kernels, numerator callbacks,
geometry providers and master providers keep their own execution. There is no
automatic block selection or CLI default change.

The reusable policy lives in oneloopmaster's `native::batch` and is re-exported by
`rust_codegen::runtime`. It validates the complete row-major input/output lengths
and overflow before resetting scratch or writing output, then invokes unchanged
generated bodies on natural whole-row chunks. Each row contains all flattened Dual
components. Empty batches still reach the generated empty path; singleton tails
use the existing scalar entry. Each generated call initializes its lazy branch
and shared-helper state.

Before a bounded call, an earlier oversized workspace releases its numerical,
active-lane and branch-control allocations while retaining the numerical
prototype. Growing, shrinking and tightening calls therefore respect the selected
maximum stride. Slot counts remain properties of the generated graph. Directly
calling a generated body or workspace preparation method bypasses this opt-in.
Only the policy's complete-buffer checks are transactional: a generic callback
error can leave earlier blocks and the failing block written. The scalar-master
API's all-row domain prevalidation stays separate.

Block32 is an experiment choice. The retained frozen diagnostic at
`validation/evidence/native-blocking-release` measured individual unchanged native
kernels on a shared host; it does not establish complete-reduction acceptance or
an automatic policy. The supported path must pass its own correctness and runtime
checks before production conclusions.

Implementation dependency: oneloopmaster main commit
`c9dd82fd1a6e3fc638b01df95bd0cf09ceda3460`, published directly after independent
review and validation. NinjaRS pins this revision. Historical measurement
worktrees retain the previous dependency revision and their original artifacts.

Validation covers full shape rejection before mutation, workspace
grow/tighten/shrink, bounds 8/16/32/64, empty/single/natural tails, exact f64,
DoubleFloat and 192-bit Float results, all supported Dual components, mixed input
precision, signed zero and nonfinite classification, shared lazy helpers,
ordinary/Dual reducer states and retry, explicit builder order, and native rank
alternates. The executed Dual n5/r4 factory control observes both generic and rank
alternate workspace strides: Whole exceeds 32 while the explicit bound retains at
most 32, with exact equality for every asymmetric higher/mixed derivative
component. Complete-buffer checks and callback error/order behavior have separate
owner tests, including writes in a failing callback block.

Focused owner/consumer tests, strict Clippy, formatting, and a feature-off build
are recorded in `validation/evidence/native-batch-policy`. Generated reduction
bodies remain byte-identical to the consumer base revision. No performance
benchmark has been run for this supported implementation; the separate earlier
diagnostic does not measure this API or complete reduction.
A separate CLI/harness execution-policy option and optimized physical controls
remain to be added before comparing this opt-in with Whole on the physical workload.
