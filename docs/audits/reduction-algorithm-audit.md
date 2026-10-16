# Reduction algorithm audit

This is a source and literature audit, not a measured optimization result. The
recommended next experiment is a reduction dependency graph that evaluates each
needed denominator expansion once and computes only live Laurent coefficients.
A second, more ambitious experiment should transpose the lower-cut subtraction
maps to project directly onto integrated outputs. Neither requires changing the
public numerator contract or moving graph preparation into the reducer.

There is no evidence yet that replacing Laurent reduction wholesale would solve
the current performance gap. There **is** concrete unnecessary arithmetic in
the port, and there are credible opportunities beyond the original algorithm.
Keep those two conclusions separate.

## Scope and evidence

Audit snapshot: NinjaRS `ec1f10fc2bd02e9ff6c6b468c2e5f6105b064da3`, original
Ninja `05fb0d3a0f47f216eceef411a09ac19e1b9a19f0`. The optional native catalog is
under further correctness validation; this audit did not compile, benchmark,
change production code, or change dependencies.

A later isolated [rank-specialization inventory](rank-specialization-probe.md)
compiled and passed its symbolic test. It quantifies selected-head and
structural-zero simplifications without changing production dispatch or claiming
numerical/performance acceptance.

The [immutable interpreted baseline](../../validation/evidence/shared-reduction-timing-0c77597.json)
measures graph26, n=5/rank4, with the same numerator callback provider and frozen
masters. Changing-point batch-one medians are approximately 309.7 microseconds
for Rust and 103.2 for original Ninja, with a paired ratio of 3.004. Actual master
execution is exactly zero in that scope. The shared host is uncontrolled; this
is not performance acceptance. The historical [kernel profile](../reduction-kernel-profile.md)
attributes substantial time to interpreter execution, including numerator
callbacks. It cannot establish the remaining algorithmic bottleneck after
native kernels are enabled.

For scale, that older instrumented n5/r4 run spends 15.13 and 14.41 microseconds
in separate bubble denominator passes, 13.04 in tadpole correction denominators,
17.64 in triangle-to-bubble corrections, and 10.52 in their source projections.
Triangle and bubble Laurent division account for only 3.32 and 2.49 microseconds.
These are prioritization clues, not additive speedup predictions for current
code. The [Laurent inventory](../laurent-catalog-inventory.md) likewise shows
only 5/32/45 instructions in this case's single/double/triple quotient plans.
An FFT or large linear solver replacing these short quotients is unlikely to
address the dominant work.

The original algorithm already uses diagonal coefficient extraction, avoids
box/pentagon subtraction in lower cuts, and omits unneeded pentagon residues in
ordinary integrated evaluation. These are central improvements over traditional
OPP fitting, not opportunities still waiting to be applied. The extensions for
rank n+1 require additional coefficients and must be considered separately.
[Laurent paper, sections 2–4](https://arxiv.org/html/1203.0291v3),
[Ninja paper, section 2 and appendix B](https://arxiv.org/html/1403.1229v2).

## Priority and expected scope

| Candidate | What actually changes | Single changing point | Several numerators at fixed kinematics | Priority |
|---|---|---|---|---|
| Live coefficient scheduling | Avoid dead expansions and structurally zero source terms | Definite operation reduction; some restores original behavior | Same saving per numerator | First |
| Shared denominator/projection graph | Avoid recomputing identical geometry-dependent quantities | Definite operation reduction; small bounded storage | Geometry can additionally be reused | First |
| Fused integrated projection / transposed lower-cut maps | Replace materialized residue propagation with output-directed linear maps | Uncertain; symbolic cancellation may win or expression growth may lose | Stronger opportunity to amortize maps | Bounded research experiment |
| Prepared tensor reduction operator | Reuse a kinematic linear operator across numerators | Generally unattractive if built from numerical basis probes | Potentially large gain above a measured break-even | Later optional API |
| Direct tensor/PV/projective reduction or on-the-fly rank reduction | Different mathematical reduction formulation | No demonstrated win under current API and coverage | Potential benefit in particular workloads | Research, not a rewrite now |
| Better cut charts / small-Gram expansions | Reduce exceptional-point failure and precision rescue | Helps difficult points, adds selection cost | Can amortize selection | Correctness/rescue program |

## 1. Compute only live coefficients

### A concrete port inefficiency: the swapped bubble orientation

The bubble branch of `ResidueProjection::new` in
[projection.rs](../../crates/ninjars-symbolica/src/projection.rs) reads the
unswapped t^0 coefficients and only **positive t powers** from the swapped
orientation. The latter supply ordinary coefficients c3/c4/c7 and higher-rank
c11/c14/c16/c18. No swapped t^0 coefficient enters any of the complete 10/20
residue outputs.

Nevertheless, [bubbles.rs](../../crates/ninjars/src/bubbles.rs) sends both
orientations through the same full numerator, division, correction and
subtraction layouts. Original Ninja already requests one less expansion depth
for the swapped orientation in `src/ninja_implem.cxx:1075`, and passes
`allcoeffs=false` to its correction routine. Thus this first change corrects
extra work in the port; it is not a new improvement over Ninja's method.

Let d=R−n+2 be the bubble quotient degree. Exact layout counts are:

| d | Current heads, both orientations | Required heads, first + swapped | Reduction |
|---|---:|---:|---:|
| 0 | 2 | 1+0 | 50% |
| 1 | 6 | 3+1 | 33% |
| 2 | 14 | 7+3 | 29% |
| 3, rank n+1 | 26 | 13+7 | 23% |

These counts apply to quotient and selected correction outputs, not to total
runtime. Positive-power division depends only on the corresponding highest
numerator powers. Generate the restricted expression heads with Symbolica;
do not merely drop columns after evaluating the original full plan. For d=0,
remove the second request entirely. For d<0 the stage is already absent.

For n5/r4, ten bubbles currently produce 60 quotient heads and 180
triangle-correction heads per lane; the required counts are 40 and 120.
For n2/r2 the bubble count changes from 14 to 10, and for n2/r3 from 26 to 20.
Neither n2 case has triangle corrections.

This pruning preserves all complete bubble residue coefficients, hence their
lower-cut uses, rational terms and reconstruction values. It is valid for
complex masses and every Dual component because it follows symbolic dependence,
not numerical zero tests. It does change callback request counts/shapes, which
is permitted by an expansion contract but must be documented and tested. Raw
quotient diagnostics, if introduced, need an explicit full mode. A discarded
head must not be left as a secretly missing physical coefficient.

There is an interface cost to test: `Numerator::expansion_batch` accepts one
common `ExpansionKind` for its entire request slice. `PolynomialNumerator`
follows that contract. Different depths therefore need two homogeneous batches
over all eligible cuts and lanes, or a separately justified request-layout
extension. They must not turn into scalar callbacks. Two smaller batches can
cost more than one larger batch, especially at width one. Measure both a
reduction-only pruning variant that retains the original full callback request
and a variant that also trims callback heads; this isolates arithmetic savings
from extra callback/packing overhead. Kernel interfaces with one common output
width need the corresponding explicit layout treatment too.

### Structural rank zeros and integrated-output liveness

`TriangleBubbleSubtraction::for_quotient` selects output powers but its input
triangle ansatz still contains the full ordinary cubic or higher quartic
residue. `ResidueProjection` makes several of these source coefficients
identically zero at lower ranks. For example, n5/r4 triangle degree is two, so
c3/c6/c8/c9 vanish identically. Substitute those **structural** zeros before
compiling the correction expressions. This can remove denominator powers and
source projections that multiplying numerical zero at runtime does not remove.
The original ordinary correction routine also computes generic intermediate
expressions before some rank-conditional assignments, so this can go beyond
restoring parity. Never infer such zeros from a particular sample or primal
Dual value.

Apply backward liveness from every requested output through the existing cut
containment graph. Integrated output, full residue diagnostics, reconstruction,
and cut-stop settings have different live sets. Spurious triangle and bubble
coefficients cannot generally be discarded: they feed lower cuts. Triangle
t^0 averaging must also remain because it cancels higher-cut contamination.

Original Ninja additionally skips massless tadpoles and scaleless bubbles and
sometimes the second bubble orientation when no lower-cut spurious terms are
needed (`ninja_implem.cxx:922–970,1135`). NinjaRS deliberately retains massless
cuts for derivative semantics. A future liveness policy may skip them only when
the entire carrier and requested derivative directions describe a scaleless
family; zero primal masses are insufficient, and genuinely nonanalytic mass
derivatives must still be reported. This is not a safe unconditional shortcut.

## 2. One denominator graph per evaluation

[bubble_kernels.rs](../../crates/ninjars-symbolica/src/bubble_kernels.rs)
calls `append_denominator` in both `divide` and `corrections`. In the second
pass, the line is the unique member of the parent triangle outside the target
bubble. It was already among that bubble's uncut lines in the first pass.
The target vectors, beta parameters, anchor and numerical propagators are the
same. Every containing triangle supplies exactly one such line.

[tadpole_kernels.rs](../../crates/ninjars-symbolica/src/tadpole_kernels.rs)
similarly expands all uncut lines during `divide`, then repeats subsets for
each triangle and bubble source in `correction`. For a target tadpole, triangle
parents repeatedly use pairs of the same n−1 uncut lines. This is reusable
work within a single evaluation even when every phase-space point changes.

With all cuts stable and live, and before pruning orientations:

| Case | Bubble T2 rows: current → unique | Tadpole T3 rows: current → unique |
|---|---:|---:|
| n5/r4 | 60 division + 60 correction → 60 | 20 division + 60 triangle + 20 bubble → 20 |
| n2/r2 | 0 → 0 | 2 division + 2 bubble → 2 |
| n2/r3 | 0 → 0 | 4 division + 4 bubble → 4 |

Generally bubble rows are twice `2*C(n,2)*(n−2)` versus that quantity once.
For o tadpole orientations, the current count is `o*n*n*(n−1)`, whereas the
unique count is `o*n*(n−1)`; o=1 ordinarily and o=2 at rank n+1. These formulas
assume the tadpole stage is active, hence R>=n−1. If R<n−1 it is absent. For n=1
there are no uncut denominators to share. Counts scale linearly with active
lanes, with actual compaction reducing them at failures.

For n5/r4 the retained payload is just 60*7+20*4=500 numerical carriers per lane
for these two families. Float and Dual carriers make memory/copy costs larger;
store one reusable contiguous buffer and index it, rather than clone the entire
table for each parent. This does not claim that 80% fewer tadpole denominator
rows means 80% less total tadpole time.

### Identity and lifetime are part of correctness

The identity must include evaluation epoch, original phase-space lane, target
cut, chosen chart/basis, orientation, anchor index, uncut line, denominator
expansion layout, precision and full derivative structure. Within one immutable
evaluation, the supplied target geometry fixes all vectors and beta parameters,
and the lane fixes all shifts and masses. The parent triangle's own origin is
**not** the denominator anchor; it remains an independent input to the source
projection and must not be replaced with the bubble origin.

Existing division request structs do not carry `SampleId`; implicit reuse based
on call order or pointers would be fragile. Add explicit internal prepared
geometry/expansion handles or a combined stage operation with a well-defined
epoch. Public standalone kernel calls must remain correct without a previous
division call. Clear or invalidate on every new evaluation, error, retry,
reordering, precision change, and derivative-shape change. Do not compare Dual
values using primal-only equality.

Preserve per-row input validation, denominator-leading-coefficient checks and
finite checks. A bad denominator invalidates only dependent cuts in that lane.
Sharing computed values is safe; bypassing a failed dependency is not.

### Joint orientations and fusion

After exact reuse, generate both transverse orientations together. They share
a0/a1 and beta; the growing and effective shrinking directions are related by
the existing basis swap and e12 normalization. The first denominator expansion
already contains projections on both directions. Likewise a pair of bubble
source-projection rows repeats the a0/a1 projections and exchanges/scales the
remaining transverse projections. A multi-output Symbolica plan can discover
these common expressions before code generation.

This removes genuine dot products, not just interpreter dispatch. It also
applies to paired triangle denominator expansions. Do not hand-code the derived
relations or assume exact floating-point equivalence under reassociation.
Compare every generated head, finite-status behavior and full Dual derivative
array, including very small/large scales and near singular charts. Preserve
checks on each required leading coefficient after the joint evaluation.

Next fuse source projection, restricted correction, parent sum and residue
projection in bounded per-cut plans. This allows common-expression elimination
across today's kernel boundaries and avoids storing dead intermediate arrays.
The topology supplies the dependency graph; the chosen numeric chart remains
lane-local. One monolithic all-topology expression risks excessive setup,
register pressure and code size. Benchmark small fused regions first.

## 3. A genuine alternative: transpose the subtraction network

This is an inference from the linearity of the existing exact maps, not a claim
of a published or tested NinjaRS speedup. For fixed geometry and branch choices,
all numerator-dependent reduction steps are linear. Write the uncorrected
triangle, bubble and tadpole coefficient blocks as u3, u2, u1, and absorb their
local projections into maps Aij. Schematically,

```
c3 = u3
c2 = u2 - A23*c3
c1 = u1 - A12*c2 - A13*c3
S  = w3*c3 + w2*c2 + w1*c1
```

The integrated result can instead use

```
v1 = w1
v2 = w2 - v1*A12
v3 = w3 - v2*A23 - v1*A13
S  = v1*u1 + v2*u2 + v3*u3
```

Here the w rows may be symbolic master-integral weights, including rational
terms. Boxes remain an independent block. Symbolica can compose and simplify
these exact linear expressions. This potentially eliminates materialized
spurious residues and exposes cancellation between direct triangle-to-tadpole
and indirect triangle-to-bubble-to-tadpole paths. **No such cancellation has
yet been proved or measured.** Transposition alone does not guarantee fewer
arithmetic operations; densely composing A12*A23 can be much worse.

For one numerator at one new phase-space point, every kinematic map must be
constructed or evaluated inside the reduction timer. There is no honest
amortization over changing points. A successful implementation must either
reduce the symbolic operation graph through cancellation/liveness, or retain
an efficient factored reverse traversal. Applying a generic dense matrix to a
single vector is not automatically faster than the current sparse hierarchy.

For several numerators with identical denominator kinematics, reverse weights
can be reused. If building weights costs B, current per-numerator work is F,
and applying the new weights costs A, the break-even is `H > B/(F-A)`, provided
F>A. Measure B, A and F separately. When masters are included in w, their
computation belongs in the separately reported master stage, once per actual
backend invocation. Reduction map construction and application remain reduction
time. Alternatively retain symbolic master heads so no master arithmetic enters
operator preparation.

An integrated-only fast path may use this approach, but requested residue and
reconstruction diagnostics must execute the original or an equivalent full
path. Cut-stop must select a corresponding live output graph. The current
failure contract also matters: algebraic cancellation cannot silently turn a
failed required cut into a successful accumulated lane. Keep geometry/failure
provenance and commit outputs transactionally. Near singularities, a formally
simpler expression can have worse cancellation or overflow behavior.

## 4. Reusable tensor operators and more distant alternatives

For a shared symmetric tensor/polynomial numerator, the complete reduction is
`c=L(kinematics)*a`, where a contains numerator coefficients. This is distinct
from generating those coefficients or compiling the numerator callback. The
tensor interface paper builds Laurent heads by contracting tensors with
cut-dependent projectors and reuses lower-rank intermediate tensors. Its timing
definition includes masters and Laurent numerator evaluation, so published
reduction timings cannot be identified with our isolated core timings.
[Tensor interface paper, sections 4–5 and 7.1](https://arxiv.org/html/1604.01363v1).

The operator idea can be tested with already supplied synthetic coefficient
arrays; it does not require completing the physical SymJIT provider first.
However, reconstructing L by numerically reducing each unit numerator is an
expensive control, not a promising production construction. A four-component
polynomial through rank4 has 70 coefficients, or 86 including explicit mu² of
weight two; rank7 has 330 or 496 respectively. Basis probing repeats reduction
that many times before the first physical numerator. Derive a factored operator
from Symbolica expressions if reuse measurements justify it.

An explicit prepared-kinematics object can safely amortize cut geometry and
denominator expansions over helicities or other numerator heads. It must bind
all precision and derivative data, routing, branch choices and scale-dependent
weights. Equal primal kinematics with different derivative seeds are different
operators. An opt-in reuse API is clearer than an automatic numeric hash cache.
Reuse never helps the single-changing-point acceptance gate unless preparation
itself is also improved. Existing tensor libraries likewise distinguish local
reduction-tree reuse from explicitly enabled event-wide reuse.
[COLLIER, section 5.5](https://arxiv.org/html/1604.06792v2#S5.SS5).

Other formulations worth retaining as research comparisons:

- **On-the-fly rank reduction.** OpenLoops interleaves numerator segment
  multiplication with reduction and keeps intermediate tensor rank small.
  This is a real algorithmic change, but its benefit depends on access to the
  factorized numerator construction. An opaque Ninja callback or an already
  assembled polynomial does not expose those segments. Importing this design
  into the default reducer would change the architectural contract; an optional
  FeynKit-side research pipeline is the appropriate place later. Its R1 and
  Gram-expansion treatments also need a complete compatible derivation.
  [OpenLoops paper, sections 4–5](https://arxiv.org/html/1710.11452v2).
- **Direct tensor reduction.** PV/Denner–Dittmaier or newer generating-function
  coefficient formulas could avoid sampling/projecting numerous cut residues.
  They bring different Gram/Cayley denominators, dimension dependence and
  exceptional-point algorithms. The 2025 rational generating-function paper
  reports comparisons with PV/recursive symbolic methods, not a matched
  batched native Ninja implementation. It is evidence of an alternative, not
  evidence of winning this task's latency gate. An isolated n3/n4 polynomial
  experiment would need all epsilon/rational terms and complex-mass controls.
  [Denner–Dittmaier](https://arxiv.org/abs/hep-ph/0509141),
  [Hu–Hu–Shen](https://arxiv.org/html/2501.11150v1).
- **Cut-chart selection and analytic exceptional-point expansions.** They can
  lower precision-rescue cost and thus improve total throughput even if regular
  points are not faster. Prefer a conditioning-aware chart policy plus local
  series to blindly taking smaller denominators or increasing precision.
  Branch selection is lane-local; higher/mixed derivatives must be evaluated
  within a valid analytic chart, and nonanalytic points reported. This is a
  separate rescue benchmark, not a regular-point speedup claim.
- **Global fitting or generic Groebner reduction.** These usually replace
  already diagonal, shallow Laurent work with larger solves or polynomial
  machinery. They may be useful for exact identities, unusual ranks or a
  topology-specific precomputation, but no evidence here favors a wholesale
  runtime replacement for n<=6 and R<=n+1. Similarly, lane tiling/SIMD can improve
  throughput and working-set size but are execution changes, not fewer
  mathematical reduction operations.

## Two bounded experiments

### Experiment A: pruned and shared lower-cut graph

Use n5/r4 graph26 plus n2/r2, n2/r3 and d=0 synthetic bubbles. Implement separate
switches for swapped-output pruning, exact within-evaluation denominator reuse,
and joint-orientation common expressions. Keep the numerator implementation
identical between variants. Prepare no numeric reduction results outside the
timer. Report requested heads, denominator rows, exact evaluator instructions,
live storage and allocations in addition to timing.

First prove equality of every complete residue coefficient and all integrated
components against current exact plans, original Ninja, and higher precision.
Cover empty/single/tail batches, permutations, changing phase points, late
failures and retries, complex masses, explicit mu²/mu⁴, arbitrary origins,
static/dynamic Dual shapes with higher/mixed components, cut-stop and
reconstruction modes. Include deliberately nonfinite or singular inputs to
detect changed failure masking.

Then compare matched native builds at widths1/8/32/128/1024 and incomplete
tails, separating numerator, reduction and actual masters. Counter predictions
above are falsifiable before timing. If removed arithmetic yields negligible
reduction improvement, profile remaining execution/packing costs instead of
expanding this transformation blindly. The 5% batch-one parity gate remains
against original Ninja; a local percentage gain is not acceptance.

### Experiment B: integrated-output projection

Start with one complete n3/r3 lower-cut network and n3/r4 extension. Form the
forward and transposed/composed expressions using existing Symbolica plans,
treating callback heads and master weights as independent symbols. Compare
exact symbolic identities, operation count, peak temporaries and generated
size before building a new runtime path. Do not assume unphysical independent
geometry symbols satisfy cut identities; compose through the existing exact
geometric parameterizations when testing cancellations that require them.

Evaluate a bounded prototype for H=1,2,4,8,16 numerators at fixed kinematics and
for H=1 with every point changing. Report map construction, application,
callback and actual master times separately, with the measured break-even H.
Include exceptional points and all-component derivative comparisons. Extend
to n5/r4 only if the exact graph remains manageable and either H=1 reduction
arithmetic shrinks substantially or realistic H shows a clear amortized gain.
Reject the default-path change if setup exceeds its savings, expressions grow
rapidly, failure semantics cannot be preserved, or numerical rescue increases.

## Algebra/API audit and implementation constraints

No new algebra implementation is required for either experiment. The three
required checks have concrete existing precedents:

| Operation | Symbolica public API / implementation and tests | Existing FeynKit/NinjaRS use |
|---|---|---|
| Restrict coefficients, structural zeros, composition | `AtomCore` replacement and polynomial coefficient APIs; `poly/polynomial.rs`, division tests | `projection.rs`, `subtraction.rs`, `laurent.rs`; FeynKit generator replacement maps |
| Laurent series and exceptional-point expansion | `AtomCore::series`, `poly/series.rs` | NinjaRS `LaurentDivision`; FeynKit `feynkit-cff/src/algebra.rs` and gammaloop series tests |
| Linear-map extraction/differentiation | `AtomCore::derivative`, `derivative.rs` tests; polynomial coefficient extraction | FeynKit CFF residue derivatives; Spenso parametric tensor derivative/series methods |
| Shared expression evaluation | `Atom::evaluator_multiple`; evaluator optimization and `ExpressionEvaluator::merge` tests in `tests/evaluation.rs` | NinjaRS `ExactKernel::evaluator`; existing oneloopmaster RustEmitter |
| Higher/mixed derivatives | Symbolica `Dualizer` and existing numerical carriers | NinjaRS `ExactKernel::dual_evaluator` and all-component validation |

`ExpressionEvaluator::merge` concatenates outputs with the same parameter
layout; it is not an arbitrary evaluator-composition or reverse-AD API. Use
symbolic substitution/coefficient extraction for composition, followed by the
existing evaluator and emitter. No public general reverse evaluator API was
identified in this audit; do not create an independent handwritten AD engine.
If an upstream capability is needed after this bounded prototype, add it in
its owning Symbolica/FeynKit crate with focused tests and the required PR.

All plans should continue using FeynKit momenta and Symbolica atoms/polynomials,
constants at working precision, batched arithmetic, lane-local chart decisions,
and transactional accumulation. An optimization that only works for ordinary
f64 can be an explicit specialization with a validated fallback; it cannot
silently weaken generic precision, full Dual support, R1 or diagnostics.
