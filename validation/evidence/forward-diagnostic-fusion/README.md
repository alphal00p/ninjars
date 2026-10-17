# Retained-output forward fusion: bounded symbolic result

The one authorized n3/r3 symbolic experiment passed at source
`bf6bde16388fb1ab5e9e9b70f3f2eb54fbaff759`, based on
`b2de05e49a91d750c5976729bc5a967cc587ebc2`. It executed only
`algorithm_transpose_probe::diagnostic_forward::inventory_forward_diagnostics_n3_r3`,
under the shared numerical lock with a 180-second timeout. Exit status was zero;
the test reports 1.30 seconds. No numerical evaluator, production dispatch,
transpose rerun, n3/r4 experiment, or timing benchmark ran.

The conservative diagnostic superset loses the five-output fusion's instruction
advantage under the existing composition/optimizer settings:

| Inventory | Outputs | Instructions | Additions | Multiplications | Inversions | Local temporary slots | Generated source bytes |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Separate forward | 283 summed heads | 1,954 | 636 | 1,500 | 81 | largest 30; sum 520 | 517,266 summed callsites |
| Fused five-output | 5 | 1,075 | 429 | 991 | 36 | 132 | 249,531 |
| Fused retained-output | 283 | 3,177 | 1,110 | 2,437 | 39 | 470 | 643,684 |

The retained graph has 1,223 more instructions than the separate blocks and
2,102 more than five-output fusion. Both old control counts reproduce exactly.
All three inventories use the same 313 external inputs, including 60 independent
callback heads and 39 abstract master Laurent components. All 283 retained block
heads and five integrated identities pass exact Symbolica expansion checks.
`inventory.json` retains each input and output symbol, all 57 block identities,
categories, output ranges, arithmetic counts, source sizes and limitations.

The retained expression text is 1,716,393 bytes, below the original cumulative
8 MiB cap. All masters stay symbolic; counted function calls are zero. Actual
denominator/source-projection construction, geometry checks, lazy master
requests, active-lane compaction and transactionality are outside this graph.
This is a negative instruction-count result for the tested conservative
full-network candidate. It does not rule out smaller fusion regions, other
output sets, or constrained optimization, and establishes no numerical or
runtime equivalence.

`ready.json` records 38 source hashes, dependency identities and the preserved
executable. All source hashes were verified unchanged after execution; canonical
dependencies remained clean. `root-verification.json` is a separate independent
reviewer's source/artifact/count/identity verification, without a numerical
rerun. `manifest.json` records exact commands, environment, caps and raw file
hashes. The executable is retained outside Git at:

```text
/tmp/ninjars-forward-diagnostics/ninjars-symbolica-probe-test
SHA256 cba1002ece8087b85613beed3983410d9852ab9dca1342941716ab7be8cddb8c
```

The build took 42.92 seconds. Strict package Clippy with
`--features rust-codegen,native-kernels --lib --tests -- -D warnings` passes,
as do package formatting and whitespace checks. The initial strict
`rust-codegen`-only check returned 101 because the existing test-only
`KernelExecutor::probe_workspace` method is unused when the native feature is
off. That complete log is retained as `clippy.log`; no unrelated source was
edited. `clippy-native.log` records the passing strict check covering the new
ignored-test source. Compiler and symbolic-test elapsed values here are
validation metadata, not performance samples.

See the source-level scope and reproduction protocol in
[`docs/audits/forward-diagnostic-fusion.md`](../../../docs/audits/forward-diagnostic-fusion.md).
