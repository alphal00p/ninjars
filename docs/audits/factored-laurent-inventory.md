# Factored one-line Laurent division inventory

Status: the single bounded symbolic inventory passed at frozen source
`0988bbd`, based on the validated integrated-weight inventory `c16e09b`.
All 16 quotient identities and five layout guards passed. The only inventoried
cases are **n6/r7/cut2** and **n5/r4/cut2**. There is no production kernel
change, numerical evaluation, or performance claim.

## Result

For n6/r7, four existing one-line evaluator calls use **472 instructions**,
compared with 712 for the expanded-product plan. For n5/r4, the three-call
candidate instead uses **36 instructions versus 32**. The high-rank case has a
useful instruction-count reduction; the ordinary-rank control does not support
a general replacement. No runtime or numerical-stability conclusion follows.

| Case and graph | Calls | Instructions | Additions | Multiplications | Inversions | Local temporaries | Source bytes |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| n6/r7 expanded product | 1 | 712 | 291 | 533 | 16 | 115 | 135,243 |
| n6/r7 reusable one-line body | 1 local body | 118 | 45 | 90 | 4 | 30 | 25,571 unique |
| n6/r7 repeated calls | 4 | 472 | 180 | 360 | 16 | largest 30; sum 120 | 102,284 unrolled callsites |
| n5/r4 expanded product | 1 | 32 | 6 | 22 | 6 | 9 | 9,638 |
| n5/r4 reusable one-line body | 1 local body | 12 | 2 | 8 | 2 | 6 | 4,661 unique |
| n5/r4 repeated calls | 3 | 36 | 6 | 24 | 6 | largest 6; sum 18 | 13,983 unrolled callsites |

All function-call operation counts are zero. The n6/r7/n5/r4 layouts retain
13/three output heads and 41/24 original input columns. A straightforward
reusable-buffer schedule has 26/six coefficient-buffer slots and a 20/ten-slot
argument row, separately from final output and evaluator temporaries. Those
quantities are not measured peak memory, allocation, dispatch, or copying cost.
The exact composed expressions are proof data only; no flattened candidate
evaluator was inventoried.

All 13 high-rank and three ordinary-rank quotient identities passed, including
the full explicit mu-squared layout. The zero-uncut, inactive, degree-zero, and
two T3 mapping guards passed. The shared expression budget consumed 69,512
bytes of its fixed 8 MiB cap. The ignored test completed in 0.11 seconds under
the external 180-second limit; this is symbolic-check elapsed time only. No
failure, cap increase, source fix, or repeated symbolic run was needed.

Package formatting and strict `rust-codegen` library/test Clippy passed. All
182 recorded source hashes, three clean dependency revisions, and the copied
read-only test executable remained unchanged after validation. The previously
retained integrated-weight executable was also unchanged. Commands, raw logs,
maps, counts, source identities and the frozen source snapshot are retained in
[`validation/evidence/factored-laurent-inventory`](../../validation/evidence/factored-laurent-inventory/README.md).

## Algebra reuse check before implementation

1. Symbolica's public `AtomCore::series`, polynomial coefficient extraction,
   `replace_multiple`, `expand`, joint evaluator construction and
   `count_operations` already provide every algebraic operation needed.
2. Their implementations and examples in `src/atom/core.rs`,
   `src/evaluate/evaluator.rs`, `src/evaluate/optimize.rs`, and
   `tests/evaluation.rs` cover simultaneous replacement, evaluator merging,
   exact expressions and operation inventories. Existing NinjaRS
   `LaurentDivision::new` performs the series and coefficient extraction; the
   probe does not implement another divider or series recurrence.
3. FeynKit/gammaloop uses Symbolica evaluator merging in process evaluators and
   Spenso evaluator sets. NinjaRS's existing exact `LaurentDivision`,
   `ExactKernel`, `bind`/literal substitution, exact identity checks, and shared
   oneloopmaster emitter provide the remaining implementation. The existing
   Laurent tests cover known quotients through six lines, rank n+1, explicit mu
   terms and mixed Dual coefficients; those numerical tests are not rerun here.

Only layout metadata and graph orchestration are new. All stage arithmetic is
an application of the existing one-line plan. Symbolica owns substitution,
expansion, optimization, operation counts and emitted Rust instruction graphs.

## Fixed degree and parameter maps

Write `k=cut`, `u=n-k`, `delta=R-n`, and `d=k+delta=R-u`, using checked signed
arithmetic. For `u>0` and `d>=0`, the reusable one-line plan is
`LaurentDivision::new(k+1, d+1, k)`. Its degree is d and its higher-rank flag
matches the original. Original numerator power `(s+u,x,mu)` maps to `(s+1,x,mu)`;
one-line output `(s,x,mu)` maps to the next input `(s+1,x,mu)`. After j lines its
physical power is `s+u-j`; final powers match the original quotient exactly.
T3 omits x. The coefficients are unchanged by this relabeling.

With `z=1/t`, both numerator layouts normalize to `z^(d-s)`. Each normalized
denominator has a nonzero scalar constant and only nonnegative z powers. The
existing layouts retain `x_degree + 2*mu_degree <= z_degree <= d`; denominator
monomials satisfy this same cone, which remains closed through multiplication
and inversion. Terms beyond depth d cannot re-enter the retained region when
dividing another line. This source argument motivates, but does not replace,
the exact all-coefficient identities below.

Every input, interstage and final map checks complete ordered tuple bijections,
including every mu-squared exponent. Denominator power tuples remain in the
owning plan's seven-column order. No positional shortcut is accepted without
those assertions. Complement order is retained. No extra numerator callback or
callback layout change is introduced.

## Preselected probe and controls

- n6/r7/cut2: four applications of n3/r4/cut2, retaining all 13 quotient heads.
- n5/r4/cut2: three applications of n3/r2/cut2, retaining all three heads.

Each case uses independent symbolic original numerator and ordered denominator
coefficients. The probe binds every one-line step through Symbolica and compares
every final coefficient with the existing full-product plan. It records the
physical power map after every step. Every binding covers the complete local
parameter list and introduces no new intermediate atom.
This verifies graph/layout equivalence to the shared existing divider, not an
independent derivation of its formulas.

Layout-only guards are fixed in advance: `(n,R,k)=(2,3,2)` for zero uncut lines,
`(5,0,2)` for inactive negative degree, `(5,3,2)` for degree zero, and
`(4,5,3)` / `(3,4,1)` for T3 maps including mu terms. The first two must bypass
one-line construction; the last three only inspect layouts and mappings. They
have no candidate inventory, composed recurrence, or numerical execution.
These guards construct the existing exact plans (including their normal
Symbolica preparation) and then inspect only their layouts/mappings; they are
not additional inventoried or composed candidates.

The operation comparison counts actual one-line evaluator calls: four or three
copies of the same local evaluator inventory. The exact composition used to
prove identities is **not** optimized as a candidate flattened evaluator.
Unique emitted-body size and instruction count are recorded separately from
summed callsite source bytes and repeated execution counts. Reported storage
separates original input, final output, two reusable coefficient buffers, one
reusable one-line argument row, and local temporary slots. These are a possible
straightforward buffer schedule, not a measured peak-memory or packing cost.

One shared **cumulative 8 MiB expression-text budget** covers all prepared
plan expressions, composed stages and checked identity expressions across both
cases and the layout guards. Partial JSON precedes expensive preparation,
composition and inventory work, and preserves each completed stage. A single
Symbolica operation can exceed the remaining time before a byte check; the
ignored executable therefore needs the fixed external **180-second timeout
after acquiring** `/tmp/ninjars-symbolica-tests.lock`. If a cap or timeout is
reached, preserve the incomplete report and stop. No cap increase or repeat is
allowed without a concrete source defect.

## Limits and later validation

No numerical comparison or runtime sampling is part of this inventory. Changed
arithmetic ordering can change overflow, conditioning and roundoff. Existing
input finiteness, all original denominator pivots, per-lane failure isolation
and callback requests must stay in their owning runtime layer. Dual pivots
require a nonzero primal; derivative-only nonzero values are not invertible.
New intermediate-finiteness rejection must not silently replace today's
final-quotient failure contract. Any later runtime experiment needs one batched
outer entry point and reusable storage, rather than per-line FFI/packing
allocation. Actual masters remain separate and downstream.

The source-proven 712-instruction n6/r7 bubble plan motivates this bounded
experiment. It makes no assumption that the recurrence improves n5/r4.
