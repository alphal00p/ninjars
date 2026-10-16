Shared Dual helper transformations re-import an evaluator's flattened scalar constants into vector-sized storage groups. Importing the zero padding repeatedly makes the constant pool and helper captures grow with each level of nesting. Reuse `InstructionList::add_constant_in_first_component` for known exact Dual literals so the same scalar literal occupies one existing group.

The generic `Vectorize<T>` API retains its `T: Default + Clone` compatibility through a defaulted scalar-import hook. Registered constants are identified before importing literals, retain distinct slots even when their placeholders compare equal, and are immediately marked unknown so literals cannot reuse their slots. Constants remain exact `Complex<Rational>` values until the existing precision-aware evaluator mapping resolves them.

Validation:

- The published PR82 branch at `6af8de565eb3908eacbb93390355f2bdb98bfa9e` passes all 16 Dual tests and five batch interpreter tests. The separate tested integration `58b72d438e7bcfce158d132612db7d35fdefa4ba` (also containing the independently submitted PR83/84 changes) passes all 17 Dual tests and five batch tests, including shared captures, nested lazy branches, differing Dual shapes, native export/merge/serialization, and higher/mixed complex derivatives.
- New nested-helper regression: five-component Dual shape `[00,10,01,20,11]`, depths 1/2/4/8, three complex/branch inputs, comparison with independently inlined evaluator. Patched stored root constant slots remain 30 and maximum helper parameters remain 40 for every depth. Disabling only the interning override reproduces 110 slots at depth 1 and 510 at depth 2 and fails the regression. Counts refer to stored scalar slots, not semantic liveness analysis.
- New registered-constant regression verifies distinct symbolic placeholders, literal-zero exclusion, resolved values and component precision at 192 bits. A custom carrier without `PartialEq` verifies the unchanged generic bound.
- All 65 active oneloopmaster unit tests pass (one pre-existing ignored test), including all 11 removable-log, quotient, and coincident-dilog neighborhood tests. These include independently seeded higher/mixed components, degree-eight log controls, Float comparisons, nonlinear symbolic controls, and explicit nonanalytic rejection.
- The C0 root scalar constant pool falls from 46,106,630 stored slots to 110. The exact same oneloopmaster C0 unit regression passes on baseline integration `8e23925` and patched integration `58b72d4`, checking f64 and Float256 primal agreement and zero derivative seeds. Diagnostic debug-build measurements were 132.74 s / 15,038,696 KiB peak RSS before, and 5.19 s / 406,324 KiB after. This is the complete regression (construction plus mappings/evaluations), not an isolated constructor benchmark. Unrelated host work was present; timings do not establish controlled performance acceptance or general C0 derivative-domain support.

Reproduction (Symbolica):

```sh
cargo test --no-default-features \
  --features integer-gmp,float-mpfr,native_code_generation,bincode \
  --lib evaluate::dual::test -- --test-threads=1
cargo test --no-default-features \
  --features integer-gmp,float-mpfr,native_code_generation,bincode \
  --lib evaluate::batch -- --test-threads=1
```

Reproduction (oneloopmaster 384b637, equivalent constructor code in consolidated main): patch Symbolica, Numerica and Graphica together to the tested source tree, then run:

```sh
cargo test --lib removable_log_tests -- --test-threads=1
cargo test --lib quotient_tests -- --test-threads=1
cargo test --lib coincident_dilog_tests -- --test-threads=1
```

This patch changes constant storage only. Per-component helper-body copies, dead captures, external-context copies and broad function-map rebuilding remain separate optimization opportunities. No algebra, numerical master-result cache, or production derivative-domain extension is added.

Formatting and diff checks pass. Clippy completes with the existing `clippy::never_loop` denial allowed; existing upstream warnings remain. Strict `-D warnings` is blocked by pre-existing diagnostics across the crate. No diagnostics point to added or changed lines.
