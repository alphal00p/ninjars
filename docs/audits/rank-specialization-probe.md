# Bounded rank-specialization probe

The ignored unit test `inventory_rank_specialization_n5_r4` builds existing
Symbolica plans, selects output heads and substitutes structural source zeros.
It emits Rust text to count its size, but installs no backend and executes no
numerical kernels. The inventory passed on 2026-10-16 at source commit
`a908e7081d463cef97f7e3ce06550ecbeadc69b1`. Its results and dependency revisions
are preserved in
[`rank-specialization-a908e70.json`](../../validation/evidence/rank-specialization-a908e70.json).
These are expression operation counts, not performance or numerical acceptance.
The ignored inventory test, workspace formatting check and strict adapter Clippy
check (`--features rust-codegen --lib --tests -- -D warnings`) passed. Existing
dependency warnings remain outside the package-level strict lint result.

Compile outside the shared Symbolica execution lock, then run the produced
test executable under that lock:

```sh
cargo test --locked -j4 -p ninjars-symbolica --features rust-codegen --lib \
  --no-run --message-format=json
flock /tmp/ninjars-symbolica-tests.lock /path/to/produced/ninjars_symbolica-test \
  inventory_rank_specialization_n5_r4 --ignored --nocapture
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

| Correction expression | Outputs | Instructions | Additions | Multiplications | Inversions | Rust source bytes |
|---|---:|---:|---:|---:|---:|---:|
| Triangle to bubble, current degree one | 3 | 102 | 33 | 84 | 3 | 22,603 |
| Triangle to bubble, rank specialized | 3 | 34 | 9 | 29 | 2 | 10,773 |
| Swapped triangle to bubble, selected positive t | 1 | 19 | 5 | 18 | 2 | 8,117 |
| Swapped triangle to bubble, rank specialized | 1 | 6 | 1 | 6 | 1 | 5,769 |
| Triangle to tadpole, current degree one | 2 | 30 | 7 | 29 | 4 | 9,900 |
| Triangle to tadpole, selected constant | 1 | 25 | 6 | 23 | 4 | 8,846 |
| Triangle to tadpole, rank specialized constant | 1 | 8 | 1 | 7 | 2 | 5,721 |
| Bubble to tadpole, current degree one | 2 | 50 | 18 | 42 | 2 | 13,181 |
| Bubble to tadpole, selected constant | 1 | 35 | 14 | 35 | 2 | 10,914 |
| Bubble to tadpole, rank specialized constant | 1 | 5 | 2 | 6 | 1 | 5,269 |

The specialized swapped bubble, triangle-to-tadpole and bubble-to-tadpole
expressions use only 5/28, 6/25 and 7/24 input columns respectively. All assertions
about denominator-column liveness passed. Symbolica counts arithmetic operations
separately from instruction nodes: an n-ary product is one instruction but can
represent several multiplications. Source size excludes the separately recorded
constant-table source; generated Rust was not compiled or executed.

For an all-active n5/r4 lane there are 30 unswapped and 30 swapped triangle-to-bubble
corrections, 30 triangle-to-tadpole corrections and 20 bubble-to-tadpole
corrections. Weighting these expression counts gives 8,020 current instruction
nodes versus 1,540 after both head selection and rank specialization (80.8% fewer),
and 6,750 versus 1,380 multiplications. This aggregate excludes geometry,
denominator expansion, quotient division, projection, numerical checks, input
packing, callbacks and master evaluation. It is not an estimate of complete
reduction speedup. Each rank-specialized head also needs numerical and derivative
controls before a production selection can be considered.

The shared reference adapter in `validation/performance/shared_provider/numerator.hh`
honors Ninja's shorter swapped expansion depth, and `PolynomialNumerator` caches
different evaluators by `ExpansionKind`. Existing equality of callback row
counts therefore does not imply equal callback coefficient work. Future
comparisons should also report requested heads/powers; separately test trimming
the reducer's arithmetic and trimming its numerator requests.

The rank substitution is distinct from restoring Ninja's shorter swapped
request. In original Ninja revision
`05fb0d3a0f47f216eceef411a09ac19e1b9a19f0`,
`src/coefficient_level_subs.cc:14` computes generic cubic source contributions
and a cubic leading-denominator inverse in the full bubble correction; its
`rmn` checks primarily select final writes. The ordinary triangle-to-tadpole
function at line 338 retains c3/c6 terms and squared denominator factors, and
the bubble-to-tadpole function at line 366 retains quadratic source terms.
Those functions receive numerical source coefficients, so these operations are
present even when the projected coefficients are structurally zero. Rank-aware
specialization can therefore remove source-level arithmetic also present in
original Ninja. A matched native benchmark is still needed to establish any
latency or throughput gain over that implementation.

These substitutions are not valid for arbitrary externally constructed source
residues. A production specialization needs trusted same-topology projected
residues, or an explicit all-component structural-zero validation with a generic
fallback. The existing public `for_quotient` plan only selects outputs and must
keep accepting its documented generic source data. The proposed probe changes
neither that contract nor the standalone evaluators.

An amplitude factory alone does not establish this trust boundary:
`AmplitudeKernels::bubble_kernels` and `tadpole_kernels` are public trait methods,
and their returned kernel types can receive externally constructed residues.
An explicit specialization must therefore validate every component of each
required zero coefficient and fall back to the generic plan, unless the trusted
path is genuinely private. Fold these checks into existing full-component
finite/shape validation where practical. Dual primal equality and the
denominator nonzero predicate cannot establish structural zeros. The test above
derives symbolic zeros from the projection itself and exercises neither factory
nor arbitrary residue execution.

`value_checks.rs` already provides the necessary all-component zero semantics
through `IntegrationChecks::is_zero`; its Dual implementation iterates all
`values()` carriers. Reuse or relocate that predicate instead of introducing
new zero-testing behavior. `ValueChecks::nonzero` intentionally tests only the
primal pivot and serves a different purpose.

At n5/r4, degree-two triangles remove cubic numerator terms from the correction.
Symbolically this lowers the largest leading-denominator inverse from the
third to second power in unswapped bubble output, and to first power in its
swapped positive-t output. Tadpole constant corrections only need the leading
homogeneous source terms. The source liveness assertions confirm these
structural simplifications; numerical
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

`ResidueIntegrator` currently skips a master whose full coefficient carrier is
zero before calling the provider. Seeding a reverse map with eagerly evaluated
masters could request otherwise unused or nonanalytic integrals, changing both
cost and failure semantics. Keep masters symbolic in the first prototype and
preserve the individual master coefficients and existing lazy selection.
Any later numerical reverse-weight cache must account for extra actual backend
calls and must not hide a failed zero-weight master behind multiplication by
zero. This is another reason to distinguish symbolic integrated-expression
counting from a production numerical algorithm.

For runtime work later, separate constructing the reverse weights from applying
them for H=1,2,4,8,16 numerators. All numerical weight construction stays inside
the H=1 changing-point timer. A factored transpose with no arithmetic reduction
is not automatically faster; dense A12*A23 formation may be much worse. The
generic callback provider and production master implementation can remain
unchanged throughout this isolated experiment.
