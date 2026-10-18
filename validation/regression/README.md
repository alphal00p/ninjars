# Regression fixtures for numerator rank <= n+1

Rank n+2 support must leave every existing rank bitwise unchanged. These
fixtures were generated at NinjaRS e4c16a7 (with oneloopmaster 264a6ce),
before any rank n+2 change, and must never be re-blessed to absorb a change of
results for rank <= n+1.

| file | test | content |
|---|---|---|
| `r-le-n1-e4c16a7.json` | `crates/ninjars-symbolica/tests/regression_bitwise.rs` (V0) | Bits of the fast engine (`TensorNumerator` + `OneLoopNativeMasters`, widths 1 and 32) on every case of `validation/fast/ninja_reference/{cases,conditioning/cases}` and `validation/benchmark/{cases,synthetic}`; the generic engine in binary64 on the same cases; explicit-`mu²` numerators through `GenericNumerator`, `MuTensorNumerator` and the generic engine; the generic engine in `Float(256)` on six cases (decimal strings). |
| `plan-fingerprints-e4c16a7.json` | `crates/ninjars-symbolica/src/plan_fingerprints_tests.rs` (V0p) | SHA-256 of the canonical strings of every exact Symbolica plan for n <= 6 and rank <= n+1: standalone plans and the ordered list of plans compiled while constructing each generic-engine stage. |

Run (release; `--features native-masters` as for the other fast tests):

```sh
cargo test --release -p ninjars-symbolica --features native-masters --test regression_bitwise
cargo test --release -p ninjars-symbolica --features native-masters --lib plan_fingerprints
```

`NINJARS_BLESS=1` rewrites a fixture instead of comparing; use it only to
record a reviewed new baseline (for example a new fixture case), never to
accept a changed result.
