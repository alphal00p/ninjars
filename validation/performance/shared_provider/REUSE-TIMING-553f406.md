# Matched denominator-reuse reduction timings at 553f406

The observed reduction parity target is **not met**. All three selections remain more than twice as slow as original Ninja on this diagnostic graph. These are shared-host measurements, not performance acceptance. The [full evidence index](../../evidence/denominator-reuse-timing-553f406.json) retains all 18 raw reports, source/artifact guards, counters and host snapshots.

Values below are paired median NinjaRS/original-Ninja ratios with stratified bootstrap 95% intervals. Lower is better; each mode is paired with original Ninja, not directly with the other modes.

## Changing phase-space workload

| Width | Interpreted | Native denominators | Native catalog |
|---:|---:|---:|---:|
| 1 | 2.777 [2.761, 2.796] | 2.737 [2.728, 2.741] | 2.726 [2.720, 2.729] |
| 8 | 2.247 [2.246, 2.289] | 2.203 [2.172, 2.226] | 2.103 [2.091, 2.181] |
| 32 | 2.469 [2.411, 2.558] | 2.286 [2.282, 2.380] | 2.217 [2.198, 2.240] |
| 128 | 2.448 [2.437, 2.474] | 2.332 [2.327, 2.336] | 2.302 [2.261, 2.356] |
| 1024 | 2.867 [2.859, 2.919] | 2.808 [2.728, 2.864] | 2.638 [2.611, 2.650] |
| 31 | 2.403 [2.358, 2.537] | 2.337 [2.296, 2.380] | 2.289 [2.190, 2.344] |

## Repeated phase-space workload

| Width | Interpreted | Native denominators | Native catalog |
|---:|---:|---:|---:|
| 1 | 2.754 [2.749, 2.763] | 2.708 [2.701, 2.720] | 2.730 [2.723, 2.742] |
| 8 | 2.264 [2.221, 2.301] | 2.145 [2.139, 2.153] | 2.085 [2.069, 2.114] |
| 32 | 2.405 [2.374, 2.422] | 2.268 [2.261, 2.291] | 2.209 [2.176, 2.283] |
| 128 | 2.450 [2.449, 2.458] | 2.349 [2.336, 2.356] | 2.257 [2.224, 2.287] |
| 1024 | 2.871 [2.867, 2.931] | 2.834 [2.810, 2.855] | 2.695 [2.682, 2.714] |
| 31 | 2.352 [2.347, 2.447] | 2.405 [2.390, 2.426] | 2.173 [2.155, 2.183] |

All 576 individual raw blocks exceed 20 ms; the minimum is 38.015 ms. Each of the 36 records has eight contiguous paired blocks, balanced four per process order. Rounds per block are 384/48/12/3/3/15 for widths 1/8/32/128/1024/31. The three selections reuse one saved native-enabled Rust library and one C++ runner, built at optimization level 3 with thin LTO and one codegen unit. No build or link runs inside the measurement schedule.

The input is the independently validated graph26 top pentagon for gg→HHH. Changing workload means three distinct physical momentum points with fixed helicities; it is not merely helicity variation. One retained reducer/workspace switches among three immutable prepared numerator/input states. Numerical geometry, callbacks, reduction and contraction execute in each sampled call. Coefficient preparation is outside this reduction-only scope.

Actual master backend calls and rows are zero in every measured record; the 85-entry frozen union has no misses. Lookup/transport remains included. The [prior execution controls](CATALOG-de0f190.md) at de0f190 distinguish actual backend computation, preparation, numerator ABI and the remaining work. Those instrumented reports retain their older source identity; no new full production-master timing is claimed here.

Both reducers call the same interpreted PolynomialNumerator callback provider, with equal sampled rows but different requested expansion heads/depths. Original Ninja requests reduced depth for the swapped bubble orientation; this Rust baseline requests full depth for both orientations. SymJIT compiles coefficient heads only, outside this timed scope.

Native denominators covers T2/T3 denominator kernels. Native catalog additionally covers bubble source projection, triangle-to-bubble correction and tadpole source projection. Geometry and the remaining reduction/integration/accumulation stay interpreted. The names therefore do not describe a fully compiled reduction engine.

CPU75 was pinned, but its SMT sibling331 was not reserved and unrelated host activity remained. Confidence intervals capture observed paired-block variation only. No isolated-host or full-suite performance gate is satisfied. Earlier archives, including the superseded pre-fix native controls, remain unchanged.

The measured runtime adds within-call denominator reuse between quotient and correction stages in bubbles and tadpoles. There is no persistent numerical-result cache. Native geometry, rank specialization and blocking experiments are excluded. Dependencies and benchmark runtime API match the older de0f190 schedule.

The older de0f190 timings and these 553f406 timings were collected in separate schedules. They are **not directly paired before/after measurements**; differences must not be presented as causal speedups or paired before/after confidence intervals. The older source, binaries and evidence remain unchanged.
