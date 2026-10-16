# Interpreted matched reduction baseline at `0c77597`

This diagnostic measures the remaining reduction gap with the same numerator
provider and frozen master table in both implementations. NinjaRS has not met
performance parity: the observed changing-input median time ratio is 3.004 at
width one and 2.638–3.303 across the other tested widths. This shared-host run is
**not performance acceptance**, and it is not compared numerically with the old
`dba7411` methodology.

The [provenance index](../../evidence/shared-reduction-timing-0c77597.json)
links six deterministic gzip archives containing every raw block, host snapshot,
source and artifact hash, counter delta, and statistical result. The separate
[release controls](../../evidence/shared-reduction-methodology-release-0c77597.json)
retain the physical numerator/Float256 oracle, all-width correctness controls,
and separately instrumented numerator/master/preparation accounting.

## Paired results

Ratios are median paired `NinjaRS/original` times; brackets show the stratified
paired-bootstrap 95% intervals. Latencies are medians per phase-space point,
including normal allocations and transport inside the stated reduction scope.
Intervals describe these blocks only, not uncontrolled host variability.

| Width | Repeated ratio [95% interval] | Changing ratio [95% interval] | Changing original µs/point | Changing NinjaRS µs/point |
|---:|---:|---:|---:|---:|
| 1 | 3.000 [2.997, 3.004] | 3.004 [2.993, 3.013] | 103.17 | 309.73 |
| 8 | 2.588 [2.564, 2.610] | 2.638 [2.494, 2.714] | 103.35 | 273.56 |
| 32 | 2.900 [2.778, 2.963] | 2.864 [2.612, 2.883] | 102.57 | 293.46 |
| 128 | 2.871 [2.767, 2.951] | 2.745 [2.710, 2.830] | 102.63 | 281.41 |
| 1024 | 3.345 [3.313, 3.372] | 3.303 [3.287, 3.350] | 101.88 | 338.44 |
| 31 | 2.719 [2.660, 2.793] | 2.768 [2.685, 2.835] | 102.18 | 287.38 |

## Measurement boundary

- Source is frozen at `0c77597`; reduction kernels and numerator callbacks are
  interpreted. SymJIT O2 evaluates factored coefficient heads only during
  preparation, which is outside this reduction timer. The optional generated
  reduction catalog is not selected here.
- One retained reducer/session per implementation receives changing shifts,
  masses and prepared coefficient rows. Three immutable prepared provider
  states are selected by pointer inside each call. Geometry, reduction and
  callbacks run again; there is no numerical reduction-result cache.
- Both use the same out-of-line Rust `PolynomialNumerator<f64>` ABI and the same
  frozen union of master requests. The timer includes callbacks, frozen lookup,
  contraction, ordinary allocation/staging and input-state selection. Actual
  master arithmetic and physical coefficient preparation are excluded.
- The same physical pool contains three genuine phase-space points at fixed
  helicities, not a helicity-only workload. Repeated inputs use point zero;
  changing inputs cycle all three points inside the same workspace.
- Each record has eight paired contiguous blocks, four in each process order.
  Rounds per block are 384/48/12/3/3/15 for widths 1/8/32/128/1024/31. One clock
  pair surrounds every block; counters and checksum reads are outside it.
  All 192 individual blocks exceed 20 ms; the smallest is 38.884 ms.
- The Rust profile matches the main workspace: optimization level three,
  thin LTO, one codegen unit. Original Ninja and OneLOop use the recorded
  uncached reference build with no fast-math. Existing copied binaries were
  reused, with no compiler or linker invoked during sampling.
- Runtime was pinned to CPU75. Its SMT sibling331 was not reserved. Team-owned
  heavy workloads paused during sampling, but unrelated host work continued.
  Host isolation and performance acceptance therefore remain false.

All twelve records passed the paired and independent Laurent checks. Source and
artifact hashes stayed unchanged. Callback row totals match exactly: 10 direct,
25 T3 and 20 T2 rows per phase-space sample, with no Mu callback for this graph.
NinjaRS batches those rows into fewer ABI calls. Equal rows do not imply equal
requested coefficient heads: original Ninja uses a shallower swapped bubble
expansion, whereas this Rust checkpoint requests full depth in both orientations.
Both use the same provider, with different evaluators for those layouts.
Preparation and actual master backend counters are zero inside every timed
block, and the frozen union has no misses. Frozen master lookup time is not
called master computation time.

The separately instrumented report measures actual live/production master
execution directly rather than subtracting unrelated timing runs. Its additive
accounting separates full physical preparation, numerator ABI, actual master
backend execution, frozen/live lookup transport and remainder. It observes one
call per implementation/record and is not the primary paired timing run.

This is one `n=5, rank=4` top-loop `gg→HHH` graph, so neither the complete
correctness matrix nor the full performance acceptance suite is claimed.
Next comparisons must preserve this provider/master policy while explicitly
selecting the generated reduction catalog, then cover the broader fixed suite.
