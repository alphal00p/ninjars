# Box and pentagon stage design

This note records the inspected interfaces and algebra behind the implemented
box/pentagon stages. Its original proposals are retained below; tests and
acceptance evidence are recorded separately in `references/PLAN.md` and the
[reference audit](box-reference-audit.md). Reference source is
original Ninja revision `05fb0d3a0f47f216eceef411a09ac19e1b9a19f0`.

## Residue conventions to preserve

Let `u = (q + V[first]) · Vort` and let `mu` below denote **mu squared**.
Original `src/cuts.hh`, `src/Xcuts.hh`, and `src/polys_implem.cxx:10` define:

| Cut | Ordinary rank | Rank n+1 | Polynomial |
| --- | --- | --- | --- |
| Pentagon | 1 coefficient | 1 coefficient | `c0 * mu` |
| Box | 5 coefficients | 6 coefficients | `c0 + c2*mu + c4*mu² + (c1+c3*mu+c5*mu²)*u`, with `c5=0` at ordinary rank |

The integrated box contribution is `c0 * D0 - c4/6`, in both rank branches.
The explicit rational part is `-c4/6`; `c1,c2,c3,c5` are needed for full
residue reconstruction, but do not add integrated scalar masters. Pentagon
coefficients have no direct master contribution in the original four-dimensional
integrated result. Do not add an E0 provider family merely because five-cut
residues exist. See `src/ninja.cc:318` and `src/Xninja.cc:213`.

For a sorted box partition `[i1,i2,i3,i4]`, the D0 request uses invariants
`[s21,s32,s43,s14,s31,s42]` and masses in that partition order, where
`sij = (V[i]-V[j])²`. Existing `MasterIntegral::D0` already has six invariants
and four masses and maps to oneloopmaster's exact D0 expression in this order.

## Two box extraction modes

Original `src/ninja_implem.cxx:226` has an integrated-only path. It averages
the two quadruple-cut solutions at `mu=0` to determine c0. Pentagon residues
vanish at `mu=0`, so this step needs no pentagon subtraction.

With `use_mu_exp=true` and rank at least n, it obtains c4 through the existing
numerator mu-expansion callback:

- Ordinary rank: request the highest coefficient for `q=t*Vort`,
  `mu=t²*Vort²`. Divide by the leading uncut-denominator factors
  `2*Vort·(Vuncut-Vfirst)`, then by `(Vort²)²`.
- Rank n+1: request the highest **two** coefficients for
  `q=t*Vort+a`, where `a=(l_plus+l_minus)/2-Vfirst`. The powers are `[n+1,n]`.
  Divide the two-term series by each uncut denominator's leading and constant
  coefficients. The resulting t4 coefficient gives c4 after division by
  `(Vort²)²`. The t5 coefficient gives the full-residue c5 after division by
  `(Vort²)³`.

NinjaRS already exposes this substitution through `ExpansionKind::Mu` and
`ExpansionRequest` with vectors `[Vort,a]`; the polynomial and Dual adapters
already accept these requests. Add a focused mu-division/projection plan using
Symbolica series and coefficients. Existing `LaurentDivision` explicitly
supports T2/T3 only; silently reinterpreting one of its layouts would be wrong.

With `use_mu_exp=false`, sample both box roots at `mu=+s` and `mu=-s`, where s
is a lane-local nonzero sampling scale. Subtract every containing pentagon
residue times its complementary denominators **before** dividing by the box's
uncut denominator product. The even root combinations determine c2/c4; odd
root combinations determine c3/c5. Full reconstruction also requires the
odd root combination at `mu=0` for c1.

Original full extraction (`src/ninja_implem.cxx:480`) also samples `mu=+s`
when mu expansion is enabled to recover c2/c3. Minimal integrated extraction
must carry an explicit computed-coefficient mask, just as the integrated
tadpole residue does. Uncomputed spurious coefficients are not measured zeros.

Original pentagon evaluation solves all five cut equations, samples N once,
and stores `N/(product of uncut D * mu)` (`src/ninja_implem.cxx:147`). Thus
the five-cut geometry must return both q and mu; a zero-mu solution is an
explicit failed geometry for this extraction. Original Ninja only constructs
pentagons for diagnostics or when mu expansion is disabled
(`src/ninja.cc:169`). Preserve this optimization without dropping the explicit
pentagon diagnostic mode.

## Suggested package and request boundaries

Keep `FourMomentum<T>`, `Atom`, and all numerical carriers as the current
shared types. Reduction-specific geometry records are appropriate; new
momentum, denominator, polynomial, or spinor representations are not.

Suggested core additions are `BoxReducer`, `PentagonReducer`, their kernel
traits, and residue records carrying `cut_index`, `SampleId`, source topology
provenance and indexed coefficients. Introduce a separate four/five-cut
provider extension trait rather than requiring unrelated downstream custom
triangle/bubble providers to implement new methods immediately.

- A box geometry request carries lane/cut identity, four ordered shifts and
  masses. Its result carries Vort, the shared origin, and data needed to
  generate the two q solutions at a requested mu. Sampling returns shared
  `FourMomentum<T>` pairs with distinct sample identities.
- A pentagon geometry request carries five ordered shifts and masses. Its
  result carries shared q and mu. The callback remains `Numerator::evaluate_batch`.
- Box mode is explicit: integrated-only or complete residue, independently
  selecting mu expansion or finite-mu sampling.
- Sampling-box parent input includes `&Topology` and dense ordered pentagon
  outcomes, using the existing topology-provenance checks. Required failed
  parents invalidate the affected lane before any denominator arithmetic.

Retain topology-cut-major/lane-minor outcome storage. Compact active rows for
geometry, numerator evaluation, division and extraction, restore caller order,
and retain the lane/cut/expansion/sample identity at every callback boundary.
Validate every Dual component and shape, including every uncut propagator.
Generated arithmetic handles denominator products, parent sums, parity
combinations, normalization, and rational contraction. Rust coordinates
requests, indices and lane-local branch decisions.

The existing triangle Laurent path divides the original numerator by all
uncut denominators without explicit box/pentagon subtraction, matching
`src/ninja_implem.cxx:762`. Do not add such subtraction just because box
residues are now available. Verify the cancellation independently with
constructed box/pentagon residue numerators, for ordinary and higher rank,
before expanding integrated amplitudes past n=3. Bubble and tadpole paths
retain their existing triangle/bubble coefficient corrections.

## Algebra reuse check

Three checks were made before proposing any new algebraic machinery:

1. Public Symbolica APIs: `AtomCore::system_to_matrix`, `AtomView::solve`,
   `AtomCore::series`, `coefficient`, evaluator construction and Dualizer.
2. Implementations and examples: `lib/numerica/src/tensors/matrix.rs:1949`
   implements matrix solve and has solve tests; `examples/solve_linear_system.rs`
   shows exact rational-polynomial matrix solving; `examples/series.rs` and
   `src/poly/series.rs` provide series infrastructure. Existing NinjaRS
   `ExactKernel` already compiles these expressions into scalar-batched and
   flattened-Dual evaluators.
3. FeynKit usage: `crates/feynkit-tensor/src/weingarten.rs:496` uses
   `Atom::system_to_matrix(...).solve(...)` and converts the result back to
   expressions. FeynKit already supplies exact momentum dot products and the
   upstream exact Weyl helpers used by current cut geometry.

Generate box/pentagon geometry from these exact facilities; do not introduce
handwritten Gaussian elimination, polynomial long division, interpolation,
or derivative recurrences. A five-cut solve is a four-component linear system
from denominator differences followed by `mu=(q+Vfirst)²-mfirst²`. A box
uses three such constraints plus the remaining quadratic equation. Use
Symbolica to derive selected pivot charts and the quadratic-root arithmetic.
Primal-only pivot and root-branch choices remain lane-local; derivatives must
flow through the selected analytic expressions. Degenerate root differences
and branch singularities need explicit failure statuses and validation.

## Master and derivative dependencies

Scalar D0 is already available for f64, DoubleFloat and Float. Before using it
in integrated n4/n5 amplitudes, add independently normalized original-Ninja
D0 controls, complex-mass controls, mixed lazy-branch batches, and precision
convergence. Reuse the existing Laurent order `[finite,single,double]` and
per-row `MasterStatus`; no additional scalar master is required.

Current `SymbolicaDualMasters` explicitly excludes C0 and D0. Four/five-cut
Dual geometry and coefficient tests may proceed independently, but integrated
Dual box/triangle amplitudes cannot be advertised until continuation helpers,
dilogarithm derivative primitives, and removable-limit neighborhoods are
validated. Extend the shared evaluator/helper bodies, preserving lazy
branches, rather than bypassing the current rejection. Keep all components in
cache keys and failure checks; primal equality is insufficient.

## Reference audit hypothesis and subsequent confirmation

In original `src/ninja_implem.cxx:717-724`, the higher-rank full-box branch with
`use_mu_exp=false` reconstructs c5 from the two nonzero-mu samples, then updates
`d.c[3] = d.c[3] - d.c[5]`. At the preceding positive sample, the stored c3
appears to be `c3_true + c5*scale_choice`; dimensional substitution therefore
suggests the final subtraction needs `c5*scale_choice`. The current source
only subtracts c5. Confirm this by Symbolica substitution of the residue
polynomial and a synthetic c5-only numerator at `scale_choice != 1`, without
modifying the reference. Integrated box contributions depend only on c0/c4,
so this hypothesis does not imply an integrated-result discrepancy.
The independent unchanged-reference probe now confirms it; see
`validation/evidence/box-full-reconstruction.json` and the reference audit.

Original higher-rank global reconstruction is already incomplete; a future
spurious-coefficient mismatch should be checked against the exact residue
identity before attributing it to NinjaRS. No acceptance gate is satisfied by
this design note alone.
