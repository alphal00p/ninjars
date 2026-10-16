# Reduction kernel profile and next catalog

The retained diagnostic evidence contains two external CPU profiles and six
instrumented kernel runs. Every run passed pre/post total-Laurent checks,
source/header guards and frozen-master replay with zero misses. Eleven harness
regressions pass. These results prioritize work; they do not measure performance
acceptance or establish a speedup.

The physical case is gg→HHH graph26, n=5 and rank4, at the first archived changing
pool point. Default integrated box coefficients do not require explicit pentagon
residues for this rank. Each reduction recomputes geometry for five boxes, ten
triangles, ten bubbles and five tadpoles. The counters prove one call for each
geometry family with those rows per lane. No geometry cache hides that work.

The shared physical provider JITs external coefficient heads with SymJIT O2;
**the direct and Laurent numerator callbacks still use interpreted
PolynomialNumerator**. The same callback provider serves both original and Rust.
The generic bubble controls also use that interpreted provider. No native
reduction kernels were enabled in these profiles.

## External CPU samples

The n5 width-one profile retained 10,643 samples over 30,000 reductions, with no
lost samples. Symbolica `evaluate_program` accounts for 52.78% exclusive CPU and
`evaluate_batch` for 9.10%. Their sum includes numerator callbacks as well as
reduction kernels. It cannot be attributed wholly to reduction arithmetic.
Other exclusive entries include tadpole/bubble scalar-predicate closures,
validation iterators, geometry, correction preparation, copies and allocations.
The optimized immutable library has incomplete unwind chains, so the observed
inclusive percentages do not provide a complete owner map.

The n2/rank2 bubble retained 1,476 samples over 100,000 reductions. The same two
interpreter functions account for 18.77% and 14.70%. This is a substantially
different composition; it does not resolve the historical bubble 10–16× gap.
That gate remains required on its original matched suite.

## Instrumented exact kernels

Times below are microseconds per full reduction call in the copied diagnostic
build. They include timer effects and are not comparable parity timings.

| Case | Width | Full call | Exact reduction/geometry/integration kernels |
|---|---:|---:|---:|
| n5, rank4 | 1 | 377.44 | 153.22 |
| n5, rank4 | 8 | 2487.27 | 1072.47 |
| bubble, rank2 | 1 | 20.93 | 5.49 |
| bubble, rank2 | 8 | 88.77 | 29.42 |
| bubble, rank3 | 1 | 28.67 | 9.63 |
| bubble, rank3 | 8 | 139.62 | 59.86 |

The remainder includes numerator callbacks, coordinator and FFI work, frozen
master lookup, validation/packing, and instrumentation overhead. The separate
measurement harness must compare numerator spans for original and Rust at
widths1/8. Equal callback row counts alone do not imply equal batch overhead.

For n5 width1, the largest individual exact plans are:

| Existing plan | Inputs→outputs | Calls/rows per reduction | Diagnostic time |
|---|---:|---:|---:|
| Selected triangle→bubble subtraction, ordinary degree1 | 28→3 | 1/60 | 17.64 |
| T2 uncut denominators for bubble division | 29→7 | 1/60 | 15.13 |
| T2 uncut denominator for triangle→bubble correction | 29→7 | 1/60 | 14.41 |
| T3 uncut denominators for tadpole corrections | 23→4 | 2/80 | 13.04 |
| Bubble correction source-axis projections | 28→8 | 1/60 | 10.52 |
| Tadpole geometry finish | 8→28 | 1/20 | 9.87 |
| T3 uncut denominators for triangle division | 23→4 | 1/40 | 7.45 |
| Two tadpole source-axis projection plans | 24→6, 28→9 | 1/30, 1/20 | 5.08, 3.99 |
| Triangle Laurent division | 12→4 | 1/20 | 3.32 |
| Bubble Laurent division | 24→3 | 1/20 | 2.49 |

At width8 the same selected subtraction/source projection process480 rows in
131.36/82.05µs. These plans merit the next ordinary native catalog increment
after measuring the existing T2/T3 native option. Division alone is a smaller
part of this workload.

Ordinary bubble geometry and non-arithmetic overhead remain material. The
higher-rank bubble additionally exercises bubble→tadpole subtraction34→4:
2.84µs over4 rows at width1 and20.30µs over32 rows at width8. The broader catalog
should include this lower-cut path; n5-only improvements cannot establish the
full performance gate. Empty triangle-correction calls are counted but their
intrusive timings do not justify a cost claim or optimization priority.

## Existing API and next increment

The existing native catalog names T2/T3 denominator semantics explicitly and
exports unchanged instruction-major Rust from Symbolica through oneloopmaster's
`RustEmitter`. Ordinary f64, DoubleFloat and Float use generic generated bodies;
the tested five-component Dual shape has separately generated denominator
bodies. Other Dual shapes and other kernels remain interpreted.

The next proposed increment reuses the exact expressions currently constructed
in `bubble_kernels::Engine::new`: source-axis projection28→8 and the explicit
power subset of `TriangleBubbleSubtraction`. Move construction/subset selection
to their owning plans so interpreter and emitter consume one source of truth.
Name correction variants by ordinary quotient degree0/1/2 or higher degree3,
with output layouts1/3/7/13. A validated quotient layout selects semantics;
matching dimensions only verifies the host boundary.

Keep `with_native_denominators()` unchanged. Add a growing opt-in
`with_native_kernels()` (CLI `--reduction-kernels native`) with explicit family
coverage for source projections and triangle→bubble corrections. The name means
all currently available generated kernels, not complete native coverage.
Integration, accumulation and other reduction plans remain reported separately.
New ordinary bodies support generic precision; new correction/projection Dual
paths initially retain the interpreter, with explicit fallback reporting.

Tests must compare every selected output against the original exact plans,
including arbitrary source origins, all four degree classes, empty/single/tail
batches, permutations, f64/DD/Float, and all derivative components on fallback
paths. Then rerun complete n5 and bubble controls. Generated code is mechanical;
no handwritten recurrence or physics representation is introduced.
