# Native reduction numerical-domain audit

The generated ordinary degree-zero triangle-to-bubble correction could return
an incorrect finite zero after an intermediate overflow. The correction is the
existing exact `TriangleBubbleSubtraction::for_quotient(LaurentDivision::new(5,
3, 2))` plan, not a separate symbolic implementation. Set all 28 inputs to zero
except residue `c3 = s`, `projection_e3[2] = 1`, and denominator `d0 = d1 = s`.
For represented binary64 `s = 1e110`, its Float256 evaluation is approximately
one. Before the shared runtime fix, native f64 returned zero and interpreted
f64 returned NaN. Both evaluator calls returned `Ok(())`, which validates shape
only; the reduction engine separately rejects nonfinite correction values.

The generated graph explicitly forms `d0*d0*d0`. Its binary64 intermediate is
infinite with zero imaginary part. The shared native reciprocal's real-axis
shortcut converted that infinity to zero. Symbolica's complex reciprocal keeps
this input nonfinite. Delegating nonfinite operands to Symbolica prevents the
native path from concealing overflow; finite-axis behavior remains unchanged.
This is a conservative domain fix, not a range-extension or precision-rescue
algorithm. Some otherwise recoverable finite answers are now visibly nonfinite.

## Existing APIs checked before the patch

1. Symbolica's public `SingleFloat::is_finite` and `FloatLike::inv` provide the
   required classification and reciprocal operation. `FloatLike::real_classify`
   supplies portable NaN classification for generic regression assertions.
2. Symbolica/Numerica `lib/numerica/src/domains/float/complex.rs` implements
   componentwise finite checks (around line 780), separate-product complex
   multiplication (around line 468), and scaled reciprocal fallback (around
   line 865). `src/evaluate/batch.rs` uses that reciprocal for exponent -1.
3. Existing usage was checked in oneloopmaster `src/native/primitives.rs`, its
   primitive tests, `src/rust_codegen/batch.rs`, and the NinjaRS generated bubble
   corrections. The runtime already imports both traits; no new algebra,
   polynomial representation, or primitive is needed. Finite helpers and exact
   constants are preserved.

The upstream patch adds only a five-line guard to `inverse`. The focused tests
exercise finite axes, signed zeros, infinities and NaNs in f64, DoubleFloat and
Float256. A Symbolica-emitted `c*d1^2/d0^3` graph tests both scalar and row-batched
execution, with exact represented inputs promoted to a Float256 interpreter.
The full NinjaRS plan has a permanent regression in
`crates/ninjars-symbolica/tests/native_domain.rs`: f64, DoubleFloat and Float256
check both single rows and mixed three-row batches. A verbose diagnostic lives
in `validation/performance/shared_provider/examples/native_domain_probe.rs`.

## Retained reproducer

Run from a checkout with the fixed shared runtime:

```sh
cargo run --locked -j4 \
  --manifest-path validation/performance/shared_provider/Cargo.toml \
  --features native-kernels --example native_domain_probe -- --require-safe
```

`validation/evidence/native-domain/` retains before/after output, the original
before-probe source, upstream test logs and a source/hash manifest. All 48
case/width rows use widths 1 and 3 and the same exact plan. Before the fix, 12
recoverable-cubic rows returned incorrect finite zeros; after the fix, none do.
The minimum demonstrated overflowing scale is `8e102`; `4e102` remains a finite
control. The `1e155` and `1e-110` cases remain nonfinite at f64 precision. Their
Float256 references stay finite. No physical amplitude failure or timing result
is inferred from these synthetic kernel controls.

## Other audited boundaries and limits

- The selected native catalog's integer-power instructions are reciprocal
  powers; positive powers were already lowered by Symbolica to multiplies.
  The general runtime exponentiation order is therefore not a demonstrated
  divergence for this catalog.
- Native and interpreted graphs use separate-product complex multiplication.
  Exact constants are built at the chosen prototype precision; input copies
  preserve tracked component precision. Existing finite/precision regressions
  cover the ordinary domain, not every near-singular configuration.
- The native Dual dispatch validates the complete ordered five-component shape.
  Different shapes retain interpreter fallback; bubble corrections remain
  interpreted for Dual. All components remain subject to finite validation.
- Nonzero leading denominators and finite output do not prove accuracy near
  singular cuts. No general conditioning guarantee or complete extreme-range
  coverage is claimed. This patch addresses the demonstrated hidden overflow.

## Validation notes

The upstream suite passed 24 tests across native primitives, generated scalar
and batch graphs, native master fixtures, shape/precision contracts, callbacks
and startup (two existing ignored helper/stress tests). The final strengthened
generated test was rerun separately. Upstream strict Clippy finds four existing
warnings in untouched `src/precision.rs` and `src/masters.rs`; focused Clippy
passes with only `chunks_exact_to_as_chunks` and `clone_on_copy` allowed. These
baseline warnings are retained in the evidence, not fixed by this domain patch.
The new NinjaRS permanent test and diagnostic example pass strict Clippy.
