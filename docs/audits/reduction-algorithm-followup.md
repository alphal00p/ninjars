# Remaining reduction algorithm options

Source-only follow-up to [the original audit](reduction-algorithm-audit.md).
Inspected main: `015622d098b4df9e157d09130cf56ea3fce595b9`; original Ninja:
`05fb0d3a0f47f216eceef411a09ac19e1b9a19f0`. No build, symbolic execution,
numerical control, profile, or timing was performed. The paired-orientation and
master-weight inventories are still pending; no additional runtime implementation
is introduced. Historical timings do not identify the current dominant cost.

## Ranked decision

| Priority | Candidate | Feasibility and stopping criterion |
| --- | --- | --- |
| First, already prepared | Paired orientations and staged master weights | Finish their bounded inventories. They address duplicate geometry work and output-directed reduction without introducing a new reduction formula. Stop an unfavorable graph-count result before a runtime implementation; favorable counts still need numerical and timing evidence. |
| Next new inventory | Preserve the one-line Laurent recurrence | Strongest additional candidate for n6/r7, whose expanded bubble quotient has 712 instructions. Compare a factored chain of existing one-line plans. Stop if summed operations/storage/source size offer no useful tradeoff, or preserving truncation requires a changed callback contract. This restores reference algorithm structure, not a novel method. |
| Next new inventory | Share reciprocal factors across parents of one target cut | Current denominator reuse stops before arithmetic inversions. This can preserve every existing correction output. Stop if preparation, packing, and retained state erase the operation reduction, or exceptional-domain behavior cannot be validated. This goes beyond the reference's independent parent corrections. |
| Conditional, chiefly n6/r7 | Specialize to verified source/target geometry | Some source-origin projections vanish on physical containing cuts, already exploited in Ninja. Prove them through existing exact geometry before pruning. Reject unconditional application to public arbitrary cut providers; stop if proof/provenance needs a broad geometry rewrite. |
| Later workload-specific | Prepared kinematics for several numerators | Reuse geometry and denominator factors across helicities at one point, with all preparation charged. It does not improve the H=1 changing-point gate by amortization. Stop without a demonstrated relevant workload and measured break-even. |
| Defer | Different cut ordering, dense tensor/PV/IBP/global fitting, reverse reduction | Ordering does not remove dependency edges; different formulations add solves, conditioning problems or numerator-contract changes. The existing transpose inventory is already negative at one changing numerator. No new evidence justifies a rewrite. |

## 1. A factored Laurent recurrence, without a new divider

`LaurentDivision::new` constructs the product of all uncut denominators, then
series-expands and extracts the complete quotient
([laurent.rs:124](../../crates/ninjars-symbolica/src/laurent.rs#L124)). Original
Ninja instead repeatedly calls `divpolyby` on one denominator in complement
order ([ninja_implem.cxx:1195](../../references/ninja/src/ninja_implem.cxx#L1195)).
Symbolica can optimize the expanded product, but expansion can obscure a cheap
recurrence. The retained [catalog](../laurent-catalog-inventory.md) reports 712
instructions for n6/r7/cut2, versus only 5/32/45 for n5/r4 cuts1/2/3. This is
an instruction-size motivation for the higher-rank case, not a timing result or
an assertion that recurrence wins at n5/r4.

The exact parameter map can use existing plans only. Write `k=cut_size`,
`u=n-k`, `delta=R-n`, and `d=k+delta`. For `u>0` and nonempty quotients, each step uses
`LaurentDivision::new(k+1, k+1+delta, k)`: exactly one uncut line and the same
degree `d` and ordinary/higher-rank classification. Check the explicit ordered
power tuples, rather than assuming column positions:

- Original numerator tuple `(t+u,x,mu)` maps to one-line input `(t+1,x,mu)`;
  T3 omits `x`. The coefficient itself is unchanged.
- After step `j`, output tuple `(t,x,mu)` represents physical power
  `(t+u-j,x,mu)`. Before the next step it is relabeled `(t+1,x,mu)` in the
  next one-line input. After `u` steps its physical power is exactly `t`.
- At every step retain the full existing weighted degree layout, including
  `mu²` weight two, and the same seven T2/four T3 denominator heads. No `x` or
  `mu²` power is changed or dropped. Formal series below the retained depth
  cannot feed higher powers in subsequent division by a normalized denominator
  with nonzero constant; verify this through exact Symbolica composition.

For n6/r7/cut2 this is four applications of the existing n3/r4/cut2 plan,
retaining all 13 quotient heads. For n5/r4/cut2 it is three applications of
n3/r2/cut2, retaining all three heads. Higher-rank single/triple cuts would use
five n2/r3/cut1 and three n4/r5/cut3 applications respectively, but need not be
part of the first experiment.

An independent source review confirms that with `z=1/t`, both input layouts
normalize to `z^(d-t)`. Denominator terms obey
`x_degree+2*mu_degree <= z_degree`, so multiplication and formal inversion of a
nonzero leading constant cannot bring discarded higher z powers back into the
retained cone. Compute `delta` and `d` using checked signed arithmetic; handle
`u=0` as identity and `d<0` as inactive before constructing the one-line plan.
For active plans its rank is simply `d+1`. Check tuple bijections explicitly,
including T3 layouts and degree zero. For Dual denominators, invertibility
requires a nonzero primal; the all-component nonzero rule used for lazy master
weights is insufficient here.

Smallest useful experiment: those two bubble controls, with abstract numerator
and denominator heads, original expanded plans versus sums of the actual
one-line evaluators. Check every final coefficient by substituting each stage
using Symbolica; retain maps, instructions/operations, intermediate buffer size
and source size, with the existing 8 MiB/180-second bounds. Do not flatten the
candidate back into one expanded evaluator or report an already recomposed
graph as a recurrence. A runtime implementation would need one batched outer
call and reusable coefficient buffers/shared generated bodies; per-line FFI,
packing allocations or scalar callback fallback would invalidate the comparison.

Input checks, all original denominator pivots, callback requests and cut stages
stay unchanged. Additional intermediate finite checks must not silently change
the present final-quotient contract. Different arithmetic order can change
rounding and exceptional behavior; compare complete coefficients, complex
masses, explicit mu terms, DoubleFloat/Float precision and higher/mixed Dual
components before any performance claim. Lazy masters remain downstream.

## 2. Reuse denominator reciprocals, not just denominator coefficients

The combined tadpole path stores each target/orientation/uncut-line denominator
once ([tadpole_kernels.rs:310](../../crates/ninjars-symbolica/src/tadpole_kernels.rs#L310))
and copies its four coefficients into each correction's input
([line 541](../../crates/ninjars-symbolica/src/tadpole_kernels.rs#L541)). Each
correction evaluator still owns its inverse arithmetic.

The current emitted n5/r4 rank-specialized scalar bodies show this directly:
[triangle correction](../../crates/ninjars-symbolica/src/native_kernels/generated/rank_triangle_to_tadpole_n5_r4.rs#L35)
has `powi(a17,-1)` and `powi(a21,-1)`; the
[bubble correction](../../crates/ninjars-symbolica/src/native_kernels/generated/rank_bubble_to_tadpole_n5_r4.rs#L34)
has `powi(a20,-1)`. With all five tadpoles and all their parents eligible and
the whole-kind rank guards succeeding, there are 30 triangle correction rows
and 20 bubble rows: **80 source-level reciprocal evaluations**, referring to
only `5*(5-1)=20` distinct leading denominator values. Per target, six triangle
parents use 12 reciprocals and four bubble parents use four; only four values
are distinct. These are predictions from generated bodies and topology, not
observed dispatch counters, hardware division counts, or a speedup.
They concern the ordinary complex-carrier bodies; Dual lowering has its own
instruction counts.

The numerator quotient already uses the same leading-denominator product:
n5/r4/cut1 has degree zero, so the exact series divides its single numerator
head by those four leading coefficients. However, the current divider exports
only quotient heads, and its evaluator scratch is private. The retained
five-instruction catalog record is insufficient by itself to establish its
particular internal inverse schedule. There is **no supported reusable table
of those reciprocal values today**. A conservative first candidate must charge
a separate 20-value preparation pass while leaving division unchanged. The
80-to-20 comparison then concerns correction reciprocals only. A later combined
division/preparation graph could expose the factors, but its counts and storage
must be measured separately; do not claim free reuse of inaccessible scratch.

Smallest exact inventory: one n5/r4 tadpole target, its six independently supplied
triangle residues and four bubble residues, all ten selected constant correction
outputs retained. Compare the ten existing guarded plans with four explicit
inverse heads plus ten plans rebound to those heads using Symbolica. Preserve
each source identity, independent source projections and all original input
columns/checks. Count preparation and applications separately, then their sum.
Do not include source-projection fusion, numerator changes or master values.

For n6/r7, two target orientations give 60 distinct leading denominator values,
120 triangle-parent rows and 60 bubble-parent rows. These are only topology
counts: higher-rank kernels involve more denominator powers, so no inverse-count
or performance extrapolation follows from n5. If the small probe succeeds,
Symbolica's series API can derive reusable truncated reciprocal factors from
the existing denominator polynomial; no handwritten recurrence is needed.

A production cache would belong to the immutable combined-call provenance,
indexed by target, orientation, lane and uncut line. Compute only validated
eligible factors, inspect every carrier component and preserve error scattering.
Carry complete Dual rows and working precision; never key on primal equality.
Changing reciprocal powers/order can expose or avoid overflow: the previous
[reciprocal-domain regression](../native-domain-audit.md) is mandatory evidence,
not permission to treat finite output as proof of accuracy. This sharing does
not authorize eager masters or change the order in which callbacks are invoked.

## 3. Geometry identities and ordering: real opportunities with boundaries

The public correction API accepts arbitrary supplied bases and origins. The
current tadpole source projections therefore compute every
`(a+V_source).dot(E)`; source residues and projections are validated before
correction dispatch. For the built-in tadpole geometry `a=-V_target`, containment
and a correctly constructed parent transverse basis imply that the two
transverse origin offsets vanish for triangle and bubble sources. Ninja's
ordinary and higher-rank triangle-to-tadpole routines take no origin argument;
bubble-to-tadpole retains only the longitudinal offset
([coefficient_level_subs.cc:338,366,1557,1647](../../references/ninja/src/coefficient_level_subs.cc#L338)).
This is source-backed motivation to prove a smaller physical graph, not a
license to zero arbitrary inputs. Compose the existing geometry and projection
plans for one fixed nonsingular chart; prove exact identities and differentiate
that composition with Symbolica. In particular, n5/r4 guarded corrections already
use only two of six triangle and three of nine bubble projection heads; further
origin pruning has little arithmetic scope there. Higher-rank n6/r7 has more
scope, but requires private geometry provenance, full public fallbacks, and
controls for roundoff, signed zeros, complex masses and all Dual components.
This largely restores simplifications present in original Ninja.

Across different target cuts, transverse axes and denominator leading values
are generally different. A shared pairwise shift/invariant table can reduce
geometry recomputation, but cannot replace those cut-dependent dot products.
Triangle geometry recomputes its difference squares/Gram probe in
[cuts.rs:181](../../crates/ninjars-symbolica/src/cuts.rs#L181); integration even
computes invariants before overriding them from a supplied S-matrix
([integration.rs:140](../../crates/ninjars-symbolica/src/integration.rs#L140)).
Those are bounded common-subexpression/dispatch cleanups, not evidence for a
different reduction order or the current dominant cost.

The existing topology caches containment, and execution is already grouped by
cut size. Bubbles require successful triangle parents; tadpoles require both
triangle and bubble parents before geometry and callbacks
([tadpoles.rs:456](../../crates/ninjars/src/tadpoles.rs#L456)). Source-major
reordering may improve locality but removes no edges. Calling children early,
transposing to evaluated master values, or skipping a zero-primal coefficient
changes callback/failure or lazy-master semantics. Dense tensor reduction and
on-the-fly numerator reduction additionally need capabilities the opaque
`Numerator` contract does not supply. Keep the existing candidates first.

## Algebra ownership checked

No algebraic operation was added. The proposed recurrence and reciprocal probes
reuse existing `LaurentDivision`, `TadpoleSubtraction`, projection-derived rank
zeros and exact parameter maps. Symbolica public `series`, `replace_multiple`,
polynomial coefficients and evaluator inventories already cover the operations
(`atom/core.rs:391,734,2193`). Its public series example, evaluator optimization
implementation and evaluation tests cover these facilities; gammaloop already
uses `series` in `uv/uv_graph.rs:320` and `utils/symbols.rs:1413`. The existing
rank and forward probes demonstrate the same substitutions and exact expanded
identity checks. Any implementation must retain these shared APIs and generated
arithmetic, not add a local polynomial divider or numerical algebra primitive.
