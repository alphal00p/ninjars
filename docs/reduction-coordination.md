# Bubble and tadpole coordination

This design documents the implemented bubble/tadpole coordinators and integrated
amplitude path, without coupling the core to DOT import, models, symbolic
numerators, or a particular master provider. The first integrated path covers
one, two, and three propagators,
including rank `n+1`; more propagators require the box and pentagon stages before
an integrated result can be advertised as complete.

## Boundaries and shared data

Core stages accept lane-major `&[FourMomentum<T>]`, `&[T]` squared masses,
one scale per lane, `Numerator<T>`, `CutProvider<T>`, and reduction-specific
numerical kernel traits. All momenta remain FeynKit types. The symbolic adapter
implements the kernel traits with the existing `ExactKernel`, Laurent,
denominator, subtraction, projection, and contraction plans. Core code packs
indices and values; arithmetic, including correction sums and amplitude sums,
executes in generated Symbolica instruction graphs.

Keep `TriangleResidue<T>`. Add `BubbleResidue<T>` containing cut index, `SampleId`,
existing `BubbleCut<T>`, and coefficients in original Ninja order (10 or 20).
Add a distinctly named integrated-tadpole result with existing `TadpoleCut<T>`
and coefficients in original 5/16-entry indexing. It records which coefficients
were computed. Filling the other entries with working-precision zeros is buffer
initialization, not evidence that the full tadpole residue vanishes there.
These are reduction outputs, not new momentum or polynomial representations.

The complete reducer owns one `Topology` and cached stage cut-index lists.
Each stage stores dense cut-major, lane-minor outcomes. Cache parent rows and
set differences once: bubble-to-triangle parents have one extra propagator;
tadpole-to-triangle and tadpole-to-bubble parents have two and one respectively.
The existing `Cut::parents` also contains boxes/pentagons: filter by the required
source cut size rather than subtracting every parent.

Cache explicit `ExpansionKind` power-to-column maps for ordinary, higher, and
lower ranks. A requested degree `rank - (n - cut_size) < 0` is structurally zero;
the corresponding stage is absent by construction. Do not confuse an absent
stage of this kind with a missing required parent.

## Bubble requests and execution

`BubbleReducer<T,K>` and `BubbleReductionKernels<T>` run parallel to the
triangle APIs. A division request contains:

- `partition: &[usize]`, `complement: &[usize]` and one lane's shared shifts/masses;
- `vectors: &[FourMomentum<T>;4]`, ordered `[a0,a1,growing,shrinking_effective]`;
- `parameters: &[T;3]`, ordered `[beta0,beta1,beta2]`.

Build both orientations for every geometry-active cut/lane:

1. `[a0,a1,e3,e4_effective]`;
2. `[a0,a1,e4,-e3/(2*e12)]`.

The swap arithmetic is a generated kernel. Pack both orientations consecutively,
as in the triangle stage. `ExpansionRequest.partition` identifies the target cut;
the request retains its original phase-space `lane`. An orientation sample ID
can continue to use `2*cut_index+orientation`. Compaction must not renumber lanes.
Both orientations initially use the same full T2 expansion depth. Optimizing the
second orientation to fewer columns can follow after parity is established.

For each orientation, batch numerator expansion and uncut-denominator expansion,
then divide using `LaurentDivision::new(n,rank,2)`. Validate input shapes, all Dual
components, finiteness, and leading denominator conditioning before arithmetic.
Keep invalid quotient rows unchanged and mark their stage status unstable.

A triangle-to-bubble correction request contains the target row and orientation,
the source triangle residue and its source cut index, the unique extra line,
the source origin `V[source.partition[0]]`, the target four callback vectors,
and `[beta0,beta1,beta2]`. This is reference/index metadata over existing types.
The adapter generates the eight source-axis projections and denominator block
required by `TriangleBubbleSubtraction`:

`[(a0+V_source).E, a1.E, growing.E, shrinking_effective.E]`, for `E=E3,E4`.

The source origin is essential when the bubble does not contain the triangle's
first propagator. Expand the unique extra denominator relative to the target
anchor with `DenominatorExpansion::new(2)`. A public request must verify source
containment and the unique set difference, not merely index bounds.

Batch all active target/source/orientation corrections together. The existing
subtraction plan returns degree-two/three columns. Gather the exact powers needed
by the target rank through the cached map. A separate generated kernel computes
`raw quotient - sum(parent corrections)` using the topology's fixed parent count;
do not perform a Rust scalar loop of numerical subtraction. Then apply the
existing `ResidueProjection` to both corrected orientations. The full bubble
coefficient array is needed by subsequent tadpole subtraction even when only a
subset contributes directly to bubble masters.

## Parent coverage and failure propagation

A stage input is validated against the common topology, lane count, coefficient
layout, and ordered cut list. Stable parent entries must carry the expected
`cut_index` and `id.lane`. Missing, duplicate, wrong-topology, or wrong-rank input
is an API error; leave caller outputs unchanged. A required unstable parent makes
the dependent target cut unstable in that same phase-space lane. It must never be
treated as a zero correction. Unrelated lanes remain eligible.

Keep this status propagation separate from numerical correction execution.
Precompute dependency eligibility before numerator callbacks so work for doomed
rows can be omitted. Source corrections can be skipped as exactly zero only when
all coefficient components are zero, including every Dual component. Primal-only
Dual equality is insufficient. First implementation may simply evaluate these
zero corrections to avoid adding a value-dependent optimization.

Projection receives only rows for which both orientations and every required
parent correction succeeded. A final integrated lane succeeds only if every
required stage and master contribution for that lane succeeded. Cut-local status
is useful for diagnostics even when the final lane must be rescued.

## Tadpole requests and minimal integrated coefficients

`TadpoleReducer<T,K>` and `TadpoleReductionKernels<T>` use T3 division
requests: target partition/complement, lane shifts/masses, three callback vectors,
and beta. Reuse `CutProvider::tadpole_batch` and its existing reference-selection
input. Geometry supplies `a=-V_target`, `beta=m_target_squared`, `e2_effective`,
and `e4_effective`.

Ordinary rank needs one transverse orientation `[a,e3,e4_effective]`. Higher rank
needs two: longitudinal `[a,e1,e2_effective]`, then transverse
`[a,e3,e4_effective]`. Divide each numerator expansion with
`LaurentDivision::new(n,rank,1)`. Subtract every containing triangle and every
containing, already corrected bubble using `TadpoleSubtraction::new(3/2,higher)`.
Triangle and bubble corrections are separate additive contributions; subtracting
a corrected bubble does not replace the explicit triangle contribution.

For each source axis E, prepare the three projections
`[(a+V_source).E, growing.E, shrinking_effective.E]`. Axis order is `[E3,E4]` for
a triangle and `[E2,E3,E4]` for a bubble. Denominators are the source lines outside
the target single cut, in source partition order. Their T3 blocks come from
`DenominatorExpansion::new(1)`. Batch source types separately, then combine all
corrections in a generated fixed-width sum kernel.

Let L and T denote the corrected longitudinal and transverse quotient rows,
and `g=e12`, so `e34=-g`. A new generated tadpole projection kernel extracts:

| Rank | Original Ninja coefficients needed for integration |
|---|---|
| `R<=n` | `c0 = T[t^0 mu^0]` |
| `R=n+1` | `c0 = L[t^0 mu^0]`, `c14 = L[t^0 mu^1]`, `c15 = -2*(T[t^0 mu^1]-c14)/g` |

Here `mu` denotes the variable mu-squared. Keep the reserved higher-rank `c9`
slot. The formula follows `ninja_implem.cxx:1534-1664`; it is also obtained by
substituting the two paths into `polys_implem.cxx:64-71`. The longitudinal path
has both transverse coordinates zero. Their product on the transverse path is
`e34*(beta+mu_squared)/2`. Full tadpole reconstruction requires the additional
spurious coefficients and must be an explicit later mode, not inferred from
this minimal integrated result.

For plain scalar integrated evaluation, massless tadpoles and scaleless bubbles
can be skipped with a documented zero-integral status. Do not skip them during
requested residue reconstruction. For Dual inputs, a zero primal mass/invariant
with nonzero derivative components is not an identically scaleless family;
report the currently unsupported/nonanalytic derivative point explicitly.

## Master batches and accumulation

Generate numerical invariants through a shared-momentum Symbolica kernel:
`s_ij=(V_i-V_j)^2`, `ke2=(V_second-V_first).e2`, and the existing e12. Cache only
topology mappings unless full numerical input identity, including all Dual
components, has been established. Supplied S-matrix values must use the same
lane ordering, precision, and derivative domain.

Use `MasterContraction` to prepare cut weights and explicit rational terms.
Group all eligible cut/lane rows by existing `MasterIntegral`: A0; B0/B1/B11
and B111 for higher rank; C0. Call the provider once per family batch, not once
per cut/lane. Bubble invariants are `[s21]`; triangle invariants are
`[s21,s32,s13]`. Squared masses retain source partition order, and scale is copied
from the original lane. A retained index map scatters master results into each
cut's contraction row. Preserve the direct OneLOop B1/B111 sign convention.

`MasterProvider::evaluate` returns one error for a whole batch.
`MasterProvider::evaluate_batch` adds per-row `MasterStatus`, implemented
directly by the Symbolica providers. Structural/API errors remain `Err` and leave
all outputs unchanged; numerical/domain failures become row statuses with
unchanged failed rows. There is no default scalar loop: prevalidation and
compaction retain batched arithmetic and lazy branch execution.

Contract each cut to `[finite,single,double,cut_constructible_finite,rational]`.
Reduce cut contributions per original lane with generated fixed-topology sum
kernels, stage the complete lane result, and only then accumulate it. A failed
lane leaves all previously accumulated numerical fields unchanged. Store the
last-attempt status separately. Reset zeroes all numerical fields from a domain
prototype, preserving Float precision and Dual shape. No f64 constants or
primal-only equality enter result creation, validation, or cache identity.

The complete core amplitude object depends on numerical-kernel and master-provider
traits. The Symbolica adapter offers convenient constructors; the CLI supplies
its already prepared numerator and shared propagator data. Neither the core nor
its master provider receives a graph/model object.

## Workspace and validation sequence

Retain owned geometry, coefficient, correction, status, and index buffers across
calls. The present triangle coordinator still allocates borrowed request vectors
per call. Avoid copying this as a claimed final fast path: use stack request
arrays for supported singleton paths, and consider a borrowed batch-view callback
over retained index descriptors for general allocation reuse. Such a view is
scheduling metadata over existing FourMomentum slices, not a new physics type.
Measure allocations and timings before claiming the warmed-path requirement.

Implement and validate in this order:

1. Bubble coordination: all parent corrections, both orientations, original
   coefficients, lane permutations/tails, malformed-parent rejection, and Dual
   shape/unstable-parent isolation. Reconstruct on the double cut independently.
2. Minimal tadpoles: ordinary and higher-rank paths, all containing parents,
   unequal origins, c0/c14/c15 controls from the original Ninja runner, and
   symbolic higher/mixed derivative controls.
3. Master status extension and transactional accumulation/reset. Include mixed
   valid/invalid lanes and verify every stored component remains unchanged after
   a failed attempt.
4. Full one-, two-, and three-propagator CLI comparisons for polynomials, explicit
   mu-squared/mu-fourth terms, complex masses, DoubleFloat/Float, and currently
   supported Dual master domains. Compare all Laurent coefficients and both
   rational/cut-constructible pieces with compatible residue bases.

Scaled leading-denominator tests, reconstruction diagnostics, precision rescue,
allocation removal, and measured throughput remain acceptance work. The existing
triangle kernel's exact-zero check alone does not establish numerical stability.

Symbolica `Float` tracks significant precision per component through arithmetic.
A small component produced by cancellation may have few significant bits while
the other complex component retains high precision. Intermediate precision need
not equal the evaluator's nominal constant precision. Keep those values and
their reported precision; do not pad bits, zero residuals, or reject them solely
for different bit counts. Domain/finiteness and Dual shape validation remain
mandatory. Converged higher-precision and independent analytic controls assess
accuracy; precision metadata alone is not a reconstruction certificate.
