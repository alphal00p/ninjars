# Same-artifact rank and native-block timing comparison

The reduction performance gate remains unmet. The complete preselected 2×2 comparison shows modest rank/block improvements, while NinjaRS still takes about twice original Ninja’s time or more on this physical case. No default policy is changed. These are shared-host diagnostics, not final acceptance.

## Fixed measurement boundary

Runtime source `8b868220f3a438a7f78e0e5329cd011646fef71c`, one copied optimized executable/library, and the same top-quark gg→HHH pentagon (graph26, n5/r4) serve all four settings. Rust uses optimization level 3, thin LTO and one codegen unit; C++ timing uses `-O3`, without fast-math. oneloopmaster is `c9dd82fd`; exact compilers, sources, dependencies, input and resolved runtime libraries are in the retained manifests.

Scope1 includes numerator callbacks, geometry/reduction, frozen master lookup/contraction and staging. Physical coefficient preparation and actual one-loop master computation are excluded. Every record has zero actual backend calls/rows and zero frozen-union misses. Master lookup traffic is retained and must not be interpreted as master computation. Separate instrumented preparation/live-master accounting remains pending.

Widths are 1/8/31/32/128/1024 with four cyclic process-position sweeps, repeated and changing workloads, and eight balanced original-first/Rust-first pairs per record. All 96 processes, 192 records, 1,536 pairs and 3,072 individual blocks are retained. The shortest individual block is 39.016648 ms, exceeding the preselected 20 ms floor. No adaptive sampling, selective reruns or discarded blocks occurred. Maximum scaled Laurent error is 2.089e-16.

CPU75 was pinned, but its sibling and other host workloads were not reserved. Build/numerical work under this task was held for the full window. These conditions do not establish host isolation.

## Rust/original ratios

Each cell is the median of four process-level paired-ratio medians, followed by their unscaled MAD. Values above one mean NinjaRS took longer. The median/MAD is descriptive; it is not a pooled confidence interval. Every individual process retains its own stratified paired-bootstrap 95% interval in [the complete review](../../evidence/native-rank-block-timing-v2/root-review.json).

### Changing phase-space points

| Batch | native Whole | native fixed32 | native-rank Whole | native-rank fixed32 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.6627 ± 0.0193 | 2.6394 ± 0.0073 | 2.5737 ± 0.0108 | 2.5556 ± 0.0158 |
| 8 | 2.0649 ± 0.0097 | 2.0887 ± 0.0080 | 1.9256 ± 0.0110 | 1.9643 ± 0.0132 |
| 31 | 2.2021 ± 0.0440 | 2.1148 ± 0.0114 | 2.0259 ± 0.0070 | 1.9869 ± 0.0315 |
| 32 | 2.2301 ± 0.0827 | 2.1654 ± 0.0548 | 2.0376 ± 0.0079 | 1.9627 ± 0.0200 |
| 128 | 2.2230 ± 0.0281 | 2.1497 ± 0.0204 | 2.1106 ± 0.0110 | 2.0270 ± 0.0073 |
| 1024 | 2.6115 ± 0.0098 | 2.4851 ± 0.0140 | 2.5164 ± 0.0227 | 2.4028 ± 0.0182 |

### Repeated phase-space point

| Batch | native Whole | native fixed32 | native-rank Whole | native-rank fixed32 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.6603 ± 0.0072 | 2.6527 ± 0.0107 | 2.5564 ± 0.0023 | 2.5427 ± 0.0106 |
| 8 | 2.0640 ± 0.0295 | 2.0416 ± 0.0167 | 1.9419 ± 0.0269 | 1.9590 ± 0.0152 |
| 31 | 2.1607 ± 0.0160 | 2.1191 ± 0.0237 | 2.0778 ± 0.0348 | 1.9572 ± 0.0302 |
| 32 | 2.2436 ± 0.1019 | 2.1400 ± 0.0300 | 2.0597 ± 0.0132 | 2.0516 ± 0.0182 |
| 128 | 2.2169 ± 0.0203 | 2.1497 ± 0.0029 | 2.0997 ± 0.0060 | 2.0369 ± 0.0026 |
| 1024 | 2.6501 ± 0.0112 | 2.5240 ± 0.0257 | 2.5173 ± 0.0291 | 2.3929 ± 0.0229 |

## Conditional rank/block effects

These ratios compare Rust process-median times within each of the four rotations. They are across different processes, unlike the original/Rust pairs above. Each entry is the median and MAD of four rotation ratios; no bootstrap interval is claimed. The changing-point results are:

| Batch | fixed32 / Whole, native | fixed32 / Whole, rank | rank / generic, Whole | rank / generic, fixed32 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9902 ± 0.0031 | 0.9926 ± 0.0046 | 0.9616 ± 0.0014 | 0.9620 ± 0.0033 |
| 8 | 1.0214 ± 0.0054 | 1.0089 ± 0.0152 | 0.9481 ± 0.0143 | 0.9391 ± 0.0027 |
| 31 | 0.9475 ± 0.0128 | 0.9803 ± 0.0147 | 0.9255 ± 0.0131 | 0.9431 ± 0.0159 |
| 32 | 0.9551 ± 0.0068 | 0.9639 ± 0.0086 | 0.9129 ± 0.0308 | 0.9237 ± 0.0237 |
| 128 | 0.9667 ± 0.0153 | 0.9614 ± 0.0017 | 0.9426 ± 0.0129 | 0.9423 ± 0.0091 |
| 1024 | 0.9505 ± 0.0030 | 0.9559 ± 0.0106 | 0.9686 ± 0.0020 | 0.9741 ± 0.0103 |

The effects depend on width and policy; fixed32 is not uniformly faster. Configuration metadata does not count native block invocations or successful rank guards. No causal comparison to the older denominator-reuse timings follows from this run.

All four Rust settings keep the same callback call/row counts. Source inspection gives original/Rust T2 output heads of 40/60 per point, respectively; this is not an observed head counter or a claim of identical head work. The same shared provider still exposes this port inefficiency.

## Evidence and review

[Evidence manifest](../../evidence/native-rank-block-timing-v2/manifest.json) binds the complete raw archive, root review and run log. The archive includes all stdout/stderr, command order/status, validated reports, raw paired blocks, host snapshots, copied driver, schedule and pre/postflight identities. Root rechecked every archived file hash, all process/record/block counts and zero-backend records; reproduced the deterministic paired summaries and cross-setting contrasts; and verified identical source/dependency/artifact/loader identities before and after sampling.

The earlier stopped attempt remains in [protocol v2](../../evidence/native-rank-block-protocol-v2): its external validator expected 50 master lookup rows instead of the source-correct 40. That attempt was not spliced into this complete restart. The correction did not change the executable, rounds, trials, schedule or duration floor.

The audit’s paired-orientation and staged master-weight probes, and the physical pruning harness, are being validated after this window. Instruction counts alone will not establish a speedup.
