# Matched native catalog reduction timings at de0f190

The observed reduction parity target is **not met**. All three selections remain more than twice as slow as original Ninja on this diagnostic graph. These are shared-host measurements, not performance acceptance. The [full evidence index](../../evidence/shared-reduction-catalog-timing-de0f190.json) retains all 18 raw reports, source/artifact guards, counters and host snapshots.

Values below are paired median NinjaRS/original-Ninja ratios with stratified bootstrap 95% intervals. Lower is better; each mode is paired with original Ninja, not directly with the other modes.

## Changing phase-space workload

| Width | Interpreted | Native denominators | Native catalog |
|---:|---:|---:|---:|
| 1 | 2.920 [2.914, 2.934] | 2.868 [2.854, 2.882] | 2.915 [2.884, 2.929] |
| 8 | 2.436 [2.421, 2.531] | 2.235 [2.221, 2.356] | 2.559 [2.524, 2.615] |
| 32 | 2.603 [2.595, 2.814] | 2.434 [2.423, 2.438] | 2.325 [2.255, 2.366] |
| 128 | 2.629 [2.571, 2.634] | 2.498 [2.441, 2.562] | 2.478 [2.461, 2.480] |
| 1024 | 3.779 [3.763, 3.841] | 2.999 [2.996, 3.022] | 2.982 [2.949, 3.160] |
| 31 | 2.717 [2.698, 2.731] | 2.599 [2.569, 2.603] | 2.306 [2.291, 2.334] |

## Repeated phase-space workload

| Width | Interpreted | Native denominators | Native catalog |
|---:|---:|---:|---:|
| 1 | 2.931 [2.924, 2.951] | 2.875 [2.865, 2.879] | 2.907 [2.885, 2.929] |
| 8 | 2.779 [2.776, 2.799] | 2.218 [2.206, 2.229] | 2.226 [2.207, 2.240] |
| 32 | 2.648 [2.599, 2.707] | 2.568 [2.433, 2.585] | 2.428 [2.364, 2.485] |
| 128 | 2.677 [2.640, 2.789] | 2.521 [2.498, 2.553] | 2.505 [2.491, 2.552] |
| 1024 | 4.091 [3.963, 4.307] | 3.082 [3.081, 3.094] | 2.984 [2.976, 2.994] |
| 31 | 2.608 [2.601, 2.633] | 2.544 [2.482, 2.607] | 2.336 [2.309, 2.394] |

All 576 individual raw blocks exceed 20 ms; the minimum is 39.139 ms. Each of the 36 records has eight contiguous paired blocks, balanced four per process order. Rounds per block are 384/48/12/3/3/15 for widths 1/8/32/128/1024/31. The three selections reuse one saved native-enabled Rust library and one C++ runner, built at optimization level 3 with thin LTO and one codegen unit. No build or link runs inside the measurement schedule.

The input is the independently validated graph26 top pentagon for gg→HHH. Changing workload means three distinct physical momentum points with fixed helicities; it is not merely helicity variation. One retained reducer/workspace switches among three immutable prepared numerator/input states. Numerical geometry, callbacks, reduction and contraction execute in each sampled call. Coefficient preparation is outside this reduction-only scope.

Actual master backend calls and rows are zero in every measured record; the 85-entry frozen union has no misses. Lookup/transport remains included. Separate [execution controls](CATALOG-de0f190.md) distinguish actual backend computation, preparation, numerator ABI and the remaining work without subtracting timings from separate noisy runs.

Both reducers call the same interpreted PolynomialNumerator callback provider, with equal sampled rows but different requested expansion heads/depths. Original Ninja requests reduced depth for the swapped bubble orientation; this Rust baseline requests full depth for both orientations. SymJIT compiles coefficient heads only, outside this timed scope.

Native denominators covers T2/T3 denominator kernels. Native catalog additionally covers bubble source projection, triangle-to-bubble correction and tadpole source projection. Geometry and the remaining reduction/integration/accumulation stay interpreted. The names therefore do not describe a fully compiled reduction engine.

CPU75 was pinned, but its SMT sibling331 was not reserved and unrelated host activity remained. Confidence intervals capture observed paired-block variation only. No isolated-host or full-suite performance gate is satisfied. Earlier archives, including the superseded pre-fix native controls, remain unchanged.
