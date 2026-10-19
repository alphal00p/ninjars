# Regression fixtures for numerator rank <= n+1

Rank n+2 support must leave every existing rank bitwise unchanged. These
fixtures hold the results of NinjaRS main before any rank n+2 change (with
oneloopmaster 264a6ce; its successor fe550e3, which adds the rank-four bubble
`B1111` for rank n+2, gives the same bits), and must never be re-blessed to
absorb a change of results for rank <= n+1 made by rank n+2 work.

| file | test | content |
|---|---|---|
| `r-le-n1-dec3d55.json` | `crates/ninjars-symbolica/tests/regression_bitwise.rs` (V0) | Bits of the fast engine (`TensorNumerator` + `OneLoopNativeMasters`, widths 1 and 32) on every case of `validation/fast/ninja_reference/{cases,conditioning/cases}` and `validation/benchmark/{cases,synthetic}`; the generic engine in binary64 on the same cases; explicit-`mu²` numerators through `GenericNumerator`, `MuTensorNumerator` and the generic engine; the generic engine in `Float(256)` on six cases (decimal strings). |
| `plan-fingerprints-e4c16a7.json` | `crates/ninjars-symbolica/src/plan_fingerprints_tests.rs` (V0p) | SHA-256 of the canonical strings of every exact Symbolica plan for n <= 6 and rank <= n+1: standalone plans and the ordered list of plans compiled while constructing each generic-engine stage. |

Run (release; `--features native-masters` as for the other fast tests):

```sh
cargo test --release -p ninjars-symbolica --features native-masters --test regression_bitwise
cargo test --release -p ninjars-symbolica --features native-masters --lib plan_fingerprints
```

`NINJARS_BLESS=1` rewrites a fixture instead of comparing; use it only to
record a reviewed new baseline (for example a new fixture case), never to
accept a changed result.

## History

- `plan-fingerprints-e4c16a7.json` was generated at e4c16a7 and still holds
  at main dec3d55: no exact plan changed there (b6b3129's flat accumulation
  sum and truncated numerator expansion produce the same canonical atoms, and
  b11767a's positional slots rename parameters only inside the evaluators).
- The V0 fixture was generated at e4c16a7 as `r-le-n1-e4c16a7.json`. Merging
  main dec3d55 (PR #3) into the rank n+2 integration re-blessed it once, as
  `r-le-n1-dec3d55.json`, because b11767a builds every generic-engine
  evaluator from positional parameter slots, so their operation order no
  longer depends on the symbols the process registered before (PR #3's
  precision-aware singularity checks change no result here). The re-blessed
  file is byte-identical to the one PR #3's author produced on that branch,
  and the changes are at rounding level only:
  - fast engine with `TensorNumerator` (widths 1 and 32) and with
    `MuTensorNumerator`: unchanged (all 2349 and 26 entries);
  - generic engine in binary64: 106 of 237 lanes (36 of 66 cases), by at
    most 2.1e-12 of the lane's largest component (syn-n6-r7 lane 2, which
    cancels); errors against the `Float(256)` values stay at their level
    (e.g. syn-n3-r4 6.6e-15 -> 5.3e-15, 7.7e-15 -> 9.7e-15);
  - explicit-`mu²` numerators: 18 of 26 generic binary64 lanes (at most
    1.0e-11 of the lane maximum, syn-n6-r7 lane 2) and 1 of 26
    `GenericNumerator` fast lanes (4.3e-17; its expansions are Symbolica
    evaluators);
  - generic `Float(256)`: 71 of 170 decimal components (12 of 17 lanes), by at
    most 6.9e-73 of the lane's largest component;
  - no lane status changed.
