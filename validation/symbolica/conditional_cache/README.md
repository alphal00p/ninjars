# Completed conditional reuse

[Symbolica PR 88](https://github.com/symbolica-dev/symbolica/pull/88) caches a
completed `if` result in its parent translation scope. Branch-local caches remain
separate. Repeated uses reuse that result without evaluating inactive branches
or changing arithmetic order. The focused upstream commit is
`f629ebc4283bfeca6104e889028f841a7e52ca6a`; the downstream integration commit is
`2c06f245835d57535418b5d78f711fe191db49cc`, following `fd7bced5`.

The archived checkpoint uses oneloopmaster `e9d14f7`. It passes 51 upstream and
69 integration evaluator tests, 293 original master fixtures with scalar and
batch execution in f64/DoubleFloat/Float256, existing JIT and Rust batch-emitter
controls, and higher/mixed C0 derivative tests. The new graph-shape tests fail
on the old Symbolica revision. Original fixture inputs are promoted binary64;
they do not establish arbitrary-precision input accuracy. The separate regular
Euclidean C0 failure is outside this patch and is recorded in its own evidence.

To reproduce the supplementary checks, use an isolated oneloopmaster checkout
at the recorded revision and patch Symbolica, Numerica and Graphica to the same
integration tree. Copy `conditional_cache.rs` into its `tests/` directory and
`inspect_conditional_cache.rs` into `examples/`, then run:

```sh
RUST_MIN_STACK=134217728 cargo test --features generated-evaluators \
  --test conditional_cache -- --test-threads=1
RUST_MIN_STACK=134217728 cargo run --features rust-codegen \
  --example inspect_conditional_cache -- baseline target/conditional-layout
```

Use the `baseline` argument only: the archived generation script retains labels
from a separate optimization experiment but does not enable those optimizations.
Count all shared instruction bodies once per signature. Root-only instruction
counts do not describe the full graph.

The D0 graph shrinks from 209,051 to 78,040 instructions, and its generated batch
source from 57,591,828 to 14,349,679 bytes. These generated master batches were
not compiled or enabled in production. Construction timings are diagnostic;
there is no runtime performance acceptance. Commands, raw test logs, layouts,
source identities and generated hashes are retained in
[`conditional-cache-upstream-checks.json`](../../evidence/conditional-cache-upstream-checks.json).
