# Projection-only swapped bubble arithmetic

`SymbolicaAmplitudeKernels` and `SymbolicaDualAmplitudeKernels` accept an
explicit `.with_pruned_swapped_arithmetic()` configuration. The corresponding
bubble stage builders return `Result<Self>` because they prepare extra exact
Symbolica evaluators. The default remains the full arithmetic path.

The opt-in selects positive-power heads from the existing swapped-orientation
division and triangle-correction expressions. It preserves the original rank,
full parameter arrays, numerator callbacks, complete bubble residues and
same-call denominator reuse. No numerator preparation, master provider or CLI
selector changes. See the [source audit](audits/swapped-bubble-heads.md) for the
algebra API checks and reference implementation comparison.

```rust
use ninjars_symbolica::SymbolicaAmplitudeKernels;

let factory = SymbolicaAmplitudeKernels::new(5, 4, 0.0)?
    .with_pruned_swapped_arithmetic();
let configured = factory.pruned_swapped_arithmetic().unwrap();
assert_eq!(configured.backend(), "symbolica-interpreter");
assert_eq!((configured.full_columns, configured.swapped_columns), (3, 1));
let amplitude = factory.into_amplitude(8)?;
```

`pruned_swapped_arithmetic()` describes configuration, not actual execution.
Its counts are quotient/correction heads per orientation before flattening Dual
components. The first orientation retains all heads; the swapped orientation
uses only positive powers. Tadpole-only and rank-inactive bubble stages retain
their existing construction support and report zero heads. A degree-zero bubble
has no positive swapped heads; no empty selected graph is compiled or invoked.

Selected graphs always use the Symbolica interpreter, including their guarded
rank-specialized variant. Generic native coverage remains separately reported.
The configured native row policy reaches newly constructed executors but does
not schedule these interpreted graphs. Rank specialization and native settings
can be selected in either builder order. A rank guard still checks every active
parent's complete derivative components before either orientation is grouped;
one violation selects generic correction graphs for the whole active kind.

The core uses the additive
`BubbleReductionKernels::divide_with_corrections_for_projection_batch` method.
Its default delegates to the original combined method. The selected Symbolica
override can omit unused swapped scratch heads, setting them to zero for the
existing subtraction/projection buffers. These zeros are not raw physical
coefficients. The old `divide_batch`, `triangle_corrections_batch` and
`divide_with_corrections_batch` continue computing full raw outputs even on an
opted-in object.

All full input shape/finiteness checks, every denominator head, leading pivot,
source coefficient and source projection remain validated. Dual checks examine
every component and the complete ordered shape, and compilation retains the
explicit scalar precision prototype. Failed dependent lanes and transactional
residue/amplitude publication keep their existing rules.

There is an intentional numerical-status difference: overflow confined to an
unused generated swapped head no longer rejects that lane because the head is
not evaluated. A nonfinite full callback input, denominator/source projection,
or live output still rejects the affected lane. The default and existing raw
APIs retain their original status behavior.

The implementation groups first/swapped arithmetic across all active cuts and
lanes, with retained packing/scatter buffers. Numerator callback kinds, rows,
heads and request identities remain unchanged. Scratch widths, full-width
subtraction and source projection also remain unchanged. Any savings must
therefore be measured against additional arithmetic dispatch and copying.
This is the arithmetic-only control; callback trimming is a separate future
candidate because it increases homogeneous callback invocations.

The bounded correctness checkpoint passes 87 tests: 31 with `rust-codegen` and
56 with `rust-codegen,native-kernels`, with no failed or ignored tests. These
cover exact complete-residue liveness, independently sampled cut reconstruction,
raw API preservation, scratch reuse/retry, input and overflow status, callback
identity, precision and higher/mixed Dual components. The actual-master amplitude
control also verifies accumulation, failed-lane isolation, reset and empty input.
Strict Clippy for core/adapter libraries and tests passes in both feature
variants, and workspace formatting passes. The retained failed lint attempt was
an unused test import; removing it changed no numerical implementation.

See the [validation record](../validation/evidence/swapped-bubble-arithmetic/README.md)
for exact source snapshots, executable hashes, commands and logs. No performance
measurement or improvement is claimed. Later measurements must retain separate
actual master-computation timing and the shared-master reduction-only control.
