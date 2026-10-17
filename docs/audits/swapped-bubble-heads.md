# Swapped bubble expansion heads: bounded implementation design

This source audit is based on NinjaRS `3793cbac1d901a19c11482db75eda4e273134deb`
and the pinned original Ninja reference. It proposes a small, explicit reduction
opt-in. No production code, numerical tests or timings were run for this audit.
Implementation remains pending the rank/block validation and performance
checkpoint. This follows the [algorithm audit](reduction-algorithm-audit.md).

The first candidate should restrict arithmetic heads while keeping numerator
callbacks unchanged. A later, separately measured candidate can trim callback
heads. The existing raw quotient APIs must retain their complete outputs in both
cases. This restores work pruning already present in original Ninja; it is not
an algorithmic improvement over the reference itself.

## Exact reference and port behavior

Write `u=n-2` for the number of uncut lines and `d=R-u` for the bubble quotient
degree. [Original Ninja](../../references/ninja/src/ninja_implem.cxx#L1015)
requests `t2Expansion(..., d, 2, partition, output)` in the first orientation.
Its [swapped call](../../references/ninja/src/ninja_implem.cxx#L1074) requests
depth `d-1`, divides to that reduced depth at line 1084, and passes
`allcoeffs=false` to triangle subtraction at line 1096. Here `mindeg` is the
depth below the numerator's leading power `t^R`, not the lowest requested
absolute power. [TensorNumerator](../../references/ninja/src/tensor_ninja.cc#L419)
computes 1, 3, 7 or 13 heads at depths 0, 1, 2 or 3.

The reference's swapped call is additionally conditional on `d>=1` and
`spurious_terms_needed`, defined at
[lines 952–958](../../references/ninja/src/ninja_implem.cxx#L952): nonzero cut
masses, requested global/local reconstruction, or coefficient printing. There
are separate scaleless-bubble skips at line 921. The proposed first candidate
does **not** adopt these mass/diagnostic-dependent skips; it retains every
complete residue coefficient required by NinjaRS's current interface.

In NinjaRS, [LaurentDivision](../../crates/ninjars-symbolica/src/laurent.rs#L43)
orders quotient tuples `(t,x,mu2)` by descending `t`, then ascending `x`, then
ascending `mu2`, with `t+x+2*mu2<=d` and `t>=0`. Numerator tuples have the same
order with `u` added to their `t` powers. The core constructs adjacent first and
swapped requests at [bubbles.rs:429](../../crates/ninjars/src/bubbles.rs#L429),
then sends **both** through one full `ExpansionKind` at line 455. Their IDs are
`{lane, sample: 2*cut_index+orientation}`; their partitions and all numerical
inputs are explicit.

The [bubble residue projection](../../crates/ninjars-symbolica/src/projection.rs#L187)
reads first-orientation `t^0` heads and only positive `t` powers from the
swapped orientation. The latter supply ordinary slots `c3,c4,c7` and, at
`R=n+1`, slots `c11,c14,c16,c18`. It reads no swapped `t^0` head. All 10 ordinary
or 20 higher-rank residue coefficients therefore remain available after this
pruning, including coefficients needed by tadpoles and reconstruction.

| d | Full heads per orientation | Live swapped heads | Current paired heads | Required paired heads |
|---|---:|---:|---:|---:|
| 0 | 1 | 0 | 2 | 1 |
| 1 | 3 | 1 | 6 | 4 |
| 2 | 7 | 3 | 14 | 10 |
| 3, R=n+1 | 13 | 7 | 26 | 20 |

The swapped raw numerator needs only tuples with `t>u`; the swapped quotient
and correction need tuples with `t>0`. The retained tuples are prefixes of the
current ordered layouts, but selection should still verify explicit powers.
Do not construct `LaurentDivision(n,R-1,2)`: that changes the numerator's leading
power and the higher-rank classification. Preserve `n`, `R`, the 10/15-component
triangle source layout and the 10/20-component published bubble layout.

For all-stable n5/r4 input, ten bubbles per phase-space lane currently produce
60 division and 180 triangle-correction heads. Selecting live heads gives 40
and 120. The n2/r2 and n2/r3 cases give 14→10 and 26→20 division heads and have no
triangle parents. These are head counts, not runtime estimates.

## First candidate: arithmetic only, complete callbacks

Use an explicit opt-in, provisionally named
`with_pruned_swapped_arithmetic()`, on ordinary/Dual bubble kernels and their
amplitude factories. Leave the default unchanged. Keep `Numerator`, its one
common-kind batch contract, and all callback request identities/layouts intact.

One additive, defaulted method on `BubbleReductionKernels` is justified:
`divide_with_corrections_for_projection_batch`, with the same arguments as the
existing combined method. Its default delegates directly to
`divide_with_corrections_batch`. The core changes its single call at
[bubbles.rs:500](../../crates/ninjars/src/bubbles.rs#L500) to this method.

The new contract defines the full first-orientation scratch heads and the
swapped heads consumed by `project_batch`; it permits other swapped scratch
slots to be omitted. The default may still compute them. The opted-in Symbolica
implementation writes zeros into omitted slots of successful rows so the
existing full-width subtraction/projection can run unchanged. Those zeros are
scratch placeholders, not reported physical coefficients.

This separate entry point matters: the existing combined method is public and
returns observable raw quotient/correction arrays. It, `divide_batch` and
`triangle_corrections_batch` must continue returning full arrays even on an
opted-in object. Internal selection in the old combined method alone would
silently change that API. No new numerator or physics representation is needed.

Inside the projection-specific path:

1. Derive positive-power output indices once from the existing exact quotient
   layout. Select the corresponding existing `Atom` expressions from the
   division and triangle-correction graphs. Keep their complete parameter arrays
   in this first candidate. Compile with the existing ordinary/Dual kernel
   machinery; do not write new Laurent recurrences.
2. Retain all complete-buffer and request-topology validation. Keep denominator
   preparation once over the same eligible adjacent orientation rows. Preserve
   `division_offsets`, ordered complement lookup and live borrowed-input identity
   used by [same-call reuse](../../crates/ninjars-symbolica/src/bubble_kernels.rs#L353).
   Do not call the standalone correction path and recompute denominators.
3. Partition valid arithmetic rows into first/swapped homogeneous batches using
   the documented **combined-row ordering**, not `SampleId` parity inferred in
   arbitrary standalone calls. Evaluate the full first graph and selected
   swapped graph, then scatter to their original full scratch positions. Use
   retained index/value buffers. No per-lane arithmetic loop or scalar callback.
4. Preserve the existing requirement that both divisions succeed before either
   orientation's parent corrections run. Keep source projection and all its
   validation unchanged. At `d=0`, the swapped arithmetic graph has no outputs;
   avoid invoking it, while retaining the original full callbacks, input checks
   and denominator/source validation for this arithmetic-only control.
5. Keep complete residue projection, parent propagation and the final staged
   output swap at [bubbles.rs:551](../../crates/ninjars/src/bubbles.rs#L551).
   API errors may affect stage scratch as today, but must not publish residues
   or change an accumulated failed lane.

Existing `KernelExecutor` accepts contiguous row-major inputs at
[kernel.rs:31](../../crates/ninjars-symbolica/src/kernel.rs#L31); grouping the two
arithmetic kinds needs packing and scatter, which must be measured. The original
full-width subtraction remains in place initially. Consequently the reduced
head counts apply to division/correction execution, not callback outputs,
transported scratch width, or subtraction outputs.

For n5/r4, this control retains 20 bubble callback rows, 60 callback heads,
60 denominator rows and 60 triangle source-projection rows per lane. Only the
selected division/correction arithmetic is reduced. The existing
[symbolic inventory](../../validation/evidence/rank-specialization-a908e70.json)
reports 102→19 instructions for full→positive-only generic n5/r4 correction,
and 34→6 for the guarded rank-specialized version. These historical graph counts
are motivation, not new timing evidence; regenerate/verify for the candidate.

## Status, Dual and native-backend constraints

Keep every current input shape/finiteness check, including full numerator
outputs in the arithmetic-only control, source coefficients, all denominator
heads and source projections. Reuse
[ValueChecks/DualChecks](../../crates/ninjars-symbolica/src/value_checks.rs#L94):
all derivative components must be finite and the complete ordered shape must
match. Pivots retain their existing primal nonzero rule. Preserve the explicit
scalar precision prototype; do not recover it from a Dual primal value.

There is one unavoidable opt-in status difference: a discarded generated
`t^0` output can overflow even when all live heads are finite. The old path
rejects that row because [division](../../crates/ninjars-symbolica/src/bubble_kernels.rs#L297)
and [correction](../../crates/ninjars-symbolica/src/bubble_kernels.rs#L483) check
every full output. A path that does not evaluate that dead head cannot preserve
this particular failure. State explicitly that the opt-in validates all retained
inputs and executed live arithmetic, and can remove failures caused only by
discarded arithmetic. Exact old status equivalence would require computing the
discarded outputs and defeat this optimization. The default and old raw APIs
retain their old behavior. Malformed inputs, bad live heads, failed parents and
singular denominators must still fail in the same dependent lanes.

If rank specialization is combined with pruning, retain the current exact-zero
guard over **all active parent rows before orientation grouping**, including
every Dual component. A guard violation keeps both orientations on their
generic correction graphs, as today. Select positive heads from the prepared
specialized graph only after preserving its original semantic plan and guard;
this optimization does not justify assuming structural zeros for arbitrary
public stage inputs.

Do not select a native body merely by output count or leading degree.
[BubblePlan::correction](../../crates/ninjars-symbolica/src/native_kernels/bubbles.rs#L44)
currently identifies full 1/3/7/13-head bodies; the same highest degree with
restricted outputs is a different semantic plan. An interpreted restricted
graph is a useful first control. Any later native version needs explicitly
named generated plans, exact export checks and honest separate backend metadata.
Native row policy must reach all new generic/guarded executors after construction
and in either builder order; unsupported Dual shapes keep their existing
interpreter fallback. Do not claim that configured native-family coverage proves
the restricted graph was executed natively.

## Later candidate: trim callback heads independently

After validating the arithmetic-only control, add a separately selectable
callback-trimming mode. Preserve the existing full callback API and existing
raw kernel APIs. A defaulted orientation-specific kind accessor plus a new
projection-path input contract can carry separate first/swapped numerator slices;
the default remains the full adjacent layout. Do not reinterpret the old full
numerator argument as a ragged buffer.

Request the full kind across all eligible first orientations and the restricted
kind across all eligible swapped orientations. At `d=0`, omit the swapped
callback entirely. Preserve lane, partition, vectors, parameters and the original
`2*cut_index+orientation` sample IDs, regardless of request compaction or call
order. Cache identity already includes the explicit kind under the
[Numerator contract](../../crates/ninjars/src/numerator.rs#L46). Both
[PolynomialNumerator](../../crates/ninjars-symbolica/src/polynomial.rs#L310) and
[DualPolynomialNumerator](../../crates/ninjars-symbolica/src/dual_polynomial.rs#L232)
already prepare and cache evaluators for arbitrary requested tuple lists.

Prove through Symbolica that retained quotient expressions do not depend on
removed numerator parameters before narrowing their input ABI. Do not zero-pad
a shortened callback and execute the old full division: dead generated outputs
would still cost work and could introduce unrelated instability. Retain original
rank and higher-rank source semantics when constructing the narrow plans.

Splitting callbacks has a concrete overhead. The core currently makes one T2
call for both orientations. The shared-provider wrapper additionally groups
requests by partition at
[foreign.rs:169](../../validation/performance/shared_provider/src/foreign.rs#L169).
For all-stable n5/r4 at phase width L, the current wrapper makes ten provider ABI
calls with 2L rows each; trimmed homogeneous callbacks make twenty with L rows
each. Rows remain 20L, while heads fall from 60L to 40L. At L=1 this may offset
arithmetic savings. The two evaluator caches also increase preparation cost.

## Algebra-policy checks

The three required checks find existing facilities for every proposed operation:

1. Public APIs: Symbolica's
   [series](../../references/symbolica/src/atom/core.rs#L734),
   [polynomial conversion](../../references/symbolica/src/atom/core.rs#L1497),
   [coefficient lookup](../../references/symbolica/src/poly/polynomial.rs#L2302)
   and [multiple-output evaluator](../../references/symbolica/src/atom/core.rs#L1048)
   already cover Laurent extraction and compiling selected heads.
2. Implementations/tests/examples: the
   [series implementation](../../references/symbolica/src/derivative.rs#L377),
   [exact series tests](../../references/symbolica/src/derivative.rs#L903),
   [series example](../../references/symbolica/examples/series.rs#L7), and
   [multi-output evaluator tests](../../references/symbolica/tests/evaluation.rs#L84)
   exercise those facilities. NinjaRS already uses them in
   [LaurentDivision](../../crates/ninjars-symbolica/src/laurent.rs#L131) and
   [TriangleBubbleSubtraction](../../crates/ninjars-symbolica/src/subtraction.rs#L186).
   Its [existing selection helper](../../crates/ninjars-symbolica/src/subtraction.rs#L108)
   and [bounded rank probe](../../crates/ninjars-symbolica/src/algorithm_rank_probe.rs#L108)
   demonstrate explicit tuple-based head selection without new algebra.
3. FeynKit/gammaloop usage: FeynKit's
   [Laurent coefficient extraction](../../references/gammaloop_feynkit/crates/feynkit-cff/src/algebra.rs#L170)
   uses Symbolica series; Spenso's
   [tensor series adapter](../../references/gammaloop_feynkit/crates/spenso/src/tensors/parametric/atomcore.rs#L1123)
   delegates to the same API. No additional shared momentum/polynomial type or
   upstream algebra implementation is needed for this candidate.

## Focused acceptance controls

- Prove selected exact heads equal the corresponding full heads for d=0/1/2/3,
  including `R=n+1` and explicit mu powers. Prove the omitted heads do not occur
  in any complete projected residue coefficient. Verify retained numerator
  dependence before the later trimmed mode.
- Compare every residue component, tadpole dependency, Laurent result,
  supported reconstruction and rational split with the full path. Cover n2/r2,
  n2/r3, a d=0 bubble and n5/r4; retain original-Ninja integrated controls.
- Instrument callback kinds, IDs, calls, rows and heads. Arithmetic-only must
  keep them identical. Trimmed mode must produce the exact counts above without
  scalarizing lanes. Include empty/single/tail batches and lane permutations.
- Check legacy implementers through the new default method, and verify old
  standalone **and combined** APIs still return full raw heads on an opted-in
  object. Short buffers and malformed parent identities must reject before
  publishing output. Inject an error after the first internal arithmetic group
  and verify unchanged external residues/accumulated amplitudes.
- Test invalid input in a currently dead raw numerator head, nonfinite live
  outputs, zero leading denominators, failed parent isolation, and overflow
  confined to discarded arithmetic. Document the latter's expected opt-in status
  difference. Poison previous scratch contents to catch incomplete scatter/zero
  initialization on repeated calls.
- Cover complex f64, DoubleFloat and arbitrary precision with direct-precision
  constants, plus asymmetric higher/mixed static and dynamic Dual shapes.
  Check every derivative component and wrong/reordered shapes; test rank-guard
  fallback across the combined batch rather than separately by orientation.
- Retain same-call denominator row counts/reuse identity checks and exercise
  standalone recomputation. If native plans are added, verify generated exports,
  fallback metadata, row-bound tails, builder order and warmed workspace reuse.

Compare full, arithmetic-only and arithmetic-plus-trimmed modes within a fresh
matched performance schedule. Charge extra packing, scatter, dispatch, scratch
validation and callback invocations; separate cold graph/evaluator preparation.
Use widths 1/8/32/128/1024 plus incomplete tails, repeated/changing physical
points, and report uncertainty. Reduction-only controls use one frozen master
union and verify zero actual master calls/rows/misses. Separate live-provider
controls must report actual master computation, callback cost, reduction
remainder and end-to-end time. No speedup follows from this audit or the head
counts alone.
