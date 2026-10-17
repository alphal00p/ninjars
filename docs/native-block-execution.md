# Explicit native row blocks

`--native-block-rows N` selects a positive bound on complete native kernel rows
per generated call. Omission preserves whole-batch execution. This setting is
independent of `--batch-size`, which counts phase-space samples: cut sampling and
corrections can expand one sample into several kernel rows. A row retains all
components, including the complete flattened Dual payload supported by the
library. The CLI currently exposes ordinary scalar carriers.

The option requires `native-kernels` and one of `native-denominators`, `native`
or `native-rank`. `symbolica` and `interpreted-rank` reject it. The existing
`native-rank` restriction to five propagators and rank four still applies. Bounds
larger than the active row count are valid and yield one block; zero, negative,
fractional and overflowing values are rejected.

The CLI applies `with_native_batch_execution` after backend and rank selection,
then reads `native_batch_execution()` back from the configured factory. Reports
keep generic/fallback family coverage and conditional rank coverage unchanged,
and add a separate object:

```json
"native_execution": {
  "policy": "fixed-rows",
  "block_rows": 7,
  "scope": "native-kernel-rows"
}
```

Whole execution reports `"policy":"whole"` and `"block_rows":null`. This is
configuration metadata: it does not prove that an unused native stage or a
conditional alternate executed. Native generic fallbacks and native alternates
receive the policy. Interpreter kernels, geometry, numerator callbacks,
preparation and master providers retain their separate execution.

For example:

```sh
cargo run -p ninjars-cli --features native-kernels -- reduce \
  examples/graphs/pentagon-rank4.json --reduction-kernels native-rank \
  --native-block-rows 7 --batch-size 2
```

The shared-provider bridge preserves the existing constructors and ten-cell
family/four-cell rank queries. Its additive constructor
`sp_session_create_with_execution` accepts `Whole=(0,0)` and
`FixedRows=(1,N>0)`. `sp_session_native_execution_info` requires exactly two
output cells and reports the configured tag/bound. Malformed policies, unsupported
selections/builds/topologies and invalid output buffers are rejected before
writing the caller's handle or query buffer. Existing constructors select Whole.
This boundary guarantee does not add rollback for runtime callback failures.

The physical runner accepts the same option independently of kernel selection.
`--all-reduction-kernels` retains its existing Whole behavior and cannot be
combined with a fixed bound. A selected native mode with a bound records both
Whole and FixedRows controls against the same frozen master union in the same
executable, checking all five complex outputs for every integrated lane. Old
reports without `native_execution` remain historical Whole reports. Explicit
fixed requests require exact policy/bound metadata; new parity rows must report
both policies explicitly.

Development correctness is recorded in
`validation/evidence/native-block-harness.json`: 33 Rust/Python test executions
cover CLI serialized parity, ABI transactionality, metadata corruption and
unchanged accounting. One copied physical executable/library checks all three
native selections with bound 7 over widths 0, 1, 3, 8, 31 and 32. Each Whole and
FixedRows policy checks 375 complex outputs with zero scaled discrepancy.
The same artifact also passes all five selectors with Whole execution, with a
maximum scaled difference of `4.00e-18` between selections.

CLI precision comparisons parse serialized values to binary64; they do not
demonstrate additional accuracy. These development controls provide no timing
or performance acceptance evidence; bound 7 remains a correctness-tail stress
setting. No automatic tuning or default change occurs.

Optimized controls have also passed for the preselected bound 32 on a separate
frozen release artifact. All three native selections agree exactly between
Whole and fixed32 across widths 0/1/3/8/31/32/128/1024, and all five Whole
selections agree within `4.00e-18` scaled error. The full set checks 67,485
complex parity components. Original Ninja agreement is within `2.2e-16`.
See `validation/evidence/native-block-release32.json` for source, dependency,
compiler, executable and control identities. These controls do not establish
complete production-backend or rational-component agreement. The
[same-artifact rank/block timing comparison](../validation/performance/shared_provider/RANK-BLOCK-TIMING-8b86822.md)
has completed all 96 processes and 3,072 individual blocks. Improvements are
modest and width-dependent; no default changes and the performance gate remains
unmet. Actual master computation and preparation are being accounted for in a
separate instrumented pass.
