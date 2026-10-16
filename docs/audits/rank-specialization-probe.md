# Bounded rank-specialization probe

The ignored unit test `inventory_rank_specialization_n5_r4` builds existing
Symbolica plans, selects output heads and substitutes structural source zeros.
It emits Rust text to count its size, but installs no backend and executes no
numerical kernels. This draft has not yet been compiled or run; no operation
counts or performance result are claimed.

After the coordinated build lock is released:

```sh
cargo test --locked -j4 -p ninjars-symbolica --features rust-codegen --lib \
  inventory_rank_specialization_n5_r4 -- --ignored --nocapture
```

The JSON destination defaults to `target/rank-specialization-inventory.json` at
the worktree root and can be overridden with `NINJARS_RANK_INVENTORY_OUTPUT`.
Use the normal licensed Symbolica development environment and stack setting.

The test derives zero coefficients from `ResidueProjection::new` using
`LaurentDivision::new(5,4,source_size)`; no hand-maintained algebraic recurrence
is introduced. The expected masks are triangle c3/c6/c8/c9 and bubble
c2/c4/c6/c7/c8/c9. It compares:

- Current triangle-to-bubble degree-one output with its rank-specialized form.
- The generic positive-t swapped head with its rank-specialized form.
- Full ordinary triangle/bubble-to-tadpole plans, selected constant heads, and
  selected heads with rank-zero substitution.

The source checks whether specialized tadpole heads depend only on each
denominator's leading coefficient, and whether the specialized swapped bubble
head has the same property. It records live parameter columns, separate
addition/multiplication/inversion counts, instructions, temporaries and generated
source bytes. Keeping the original input ABI isolates algebraic liveness from
packing changes. It refuses to silently undercount hidden helper bodies.

These substitutions are not valid for arbitrary externally constructed source
residues. A production specialization needs trusted same-topology projected
residues, or an explicit all-component structural-zero validation with a generic
fallback. The existing public `for_quotient` plan only selects outputs and must
keep accepting its documented generic source data. The proposed probe changes
neither that contract nor the standalone evaluators.

At n5/r4, degree-two triangles remove cubic numerator terms from the correction.
Symbolically this can lower the largest leading-denominator inverse from the
third to second power in unswapped bubble output, and to first power in its
swapped positive-t output. Tadpole constant corrections only need the leading
homogeneous source terms. These are source-derived expectations; numerical
overflow, derivative and failure behavior still need separate validation.

## Concrete plan for the later n3/r3 transpose experiment

A crate-internal ignored test is sufficient: the needed exact plans already
expose `pub(crate)` expression/parameter arrays. There is no need to widen the
public runtime API or manually implement a transposed evaluator.

For topology `{0,1,2}`, use one triangle, bubbles `{0,1}`, `{0,2}`, `{1,2}`, and
three tadpoles. Keep every physical source/target index in symbol names so
different edges cannot accidentally share a parameter. The relevant plans are:

| Block | Existing expression source |
|---|---|
| Triangle u3 | `ResidueProjection::new(LaurentDivision::new(3,3,3))` |
| Bubble raw u2 and correction A23 | `ResidueProjection::new(LaurentDivision::new(3,3,2))` composed with both orientations of `TriangleBubbleSubtraction::for_quotient` |
| Tadpole A12/A13 | The `(t,mu²)=(0,0)` heads of `TadpoleSubtraction::new(2,false)` and `new(3,false)` |
| Integration weights | `MasterContraction::new(cut_size,false).weights`, then `.contraction` |
| Optional geometry composition | `BubbleSourceProjection`, `TadpoleSourceProjection`, and `DenominatorExpansion` |

The first probe can treat geometry projections and denominator coefficients as
independent symbolic inputs. Rename each plan's generic parameters into a fresh
edge-specific namespace using Symbolica replacements before composing it. Use
independent coefficient symbols for the raw u blocks and independent symbols
for each master's finite/pole components; call no master provider. At this
level the forward and transposed expressions must be identical for arbitrary
parameter values. It will not expose cancellations that rely on relations
between different cuts' geometric parameters.

The fixed-kinematics network has ten triangle coefficients, three ten-component
raw bubble blocks and three tadpole constants. Symbolica polynomial coefficient
extraction or differentiation in those coefficient symbols constructs the sparse
A maps. Compose the forward and reverse expressions through Symbolica, then
build three controls with identical evaluator settings: current separated
forward kernels, fused forward expressions, and reverse/composed expressions.
Otherwise a benefit from ordinary cross-kernel common-expression elimination
could be misidentified as a benefit of transposition.

Keep every master's three Laurent coefficients and the explicit rational
weight, and compare all five integrated outputs. Report graph construction
cost and size as well as operation counts. Only after the abstract-geometry
graph is manageable should the probe insert the existing geometric expressions
to search for physical cancellations. Do not assert independently named cut
vectors satisfy cut identities without that composition.

For runtime work later, separate constructing the reverse weights from applying
them for H=1,2,4,8,16 numerators. All numerical weight construction stays inside
the H=1 changing-point timer. A factored transpose with no arithmetic reduction
is not automatically faster; dense A12*A23 formation may be much worse. The
generic callback provider and production master implementation can remain
unchanged throughout this isolated experiment.
