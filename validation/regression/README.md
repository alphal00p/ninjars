# Regression fixtures for numerator rank <= n+1

Rank n+2 support must leave every existing rank bitwise unchanged. These
fixtures hold the results of NinjaRS main without any rank n+2 change, at its
current oneloopmaster pin (main d53dd08 with oneloopmaster 09b2c74, which
contains fe550e3's rank-four bubble `B1111`), and must never be re-blessed to
absorb a change of results for rank <= n+1 made by rank n+2 work. They are
regenerated only from a clean checkout of main, when merging main brings
results that main itself changed (History).

| file | test | content |
|---|---|---|
| `r-le-n1-d53dd08.json` | `crates/ninjars-symbolica/tests/regression_bitwise.rs` (V0) | Bits of the fast engine (`TensorNumerator` + `OneLoopNativeMasters`, widths 1 and 32) on every case of `validation/fast/ninja_reference/{cases,conditioning/cases}` and `validation/benchmark/{cases,synthetic}`; the generic engine in binary64 on the same cases; explicit-`mu²` numerators through `GenericNumerator`, `MuTensorNumerator` and the generic engine; the generic engine in `Float(256)` on six cases (decimal strings). |
| `plan-fingerprints-e4c16a7.json` | `crates/ninjars-symbolica/src/plan_fingerprints_tests.rs` (V0p) | SHA-256 of the canonical strings of every exact Symbolica plan for n <= 6 and rank <= n+1: standalone plans and the ordered list of plans compiled while constructing each generic-engine stage. |

Run (release; `--features native-masters` as for the other fast tests):

```sh
cargo test --release -p ninjars-symbolica --features native-masters --test regression_bitwise
cargo test --release -p ninjars-symbolica --features native-masters --lib plan_fingerprints
```

`NINJARS_BLESS=1` rewrites a fixture instead of comparing; use it only to
record a reviewed new baseline (for example a new fixture case), never to
accept a changed result. A new baseline for main's own changes is blessed on a
clean clone of main with this branch's `regression_bitwise.rs` copied in (its
`FIXTURE` and `baseline` naming the main commit), and the branch must then
pass it bitwise.

## History

- `plan-fingerprints-e4c16a7.json` was generated at e4c16a7 and still holds
  at main dec3d55: no exact plan changed there (b6b3129's flat accumulation
  sum and truncated numerator expansion produce the same canonical atoms, and
  b11767a's positional slots rename parameters only inside the evaluators).
  It also holds after merging main d53dd08, which changes no plan.
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
- Merging main d53dd08 into the rank n+2 integration moved the V0 fixture a
  second time, to `r-le-n1-d53dd08.json` (`r-le-n1-dec3d55.json` deleted).
  Main had moved its oneloopmaster pin from 264a6ce to 09b2c74 (via 3023fa6
  and 149ddff). Those revisions change C0 and D0 results in the native and
  exact masters alike: conditioned coincident dilogarithm differences and tiny
  box invariants (3cbe08d), the contour box for real-mass three-mass boxes and
  its edge roots from exact Källén functions (4d438cf, 9c5490b), far-root
  `r1` (261bf3d), a real-mass four-mass D0 independent of the propagator
  labelling (95f8107), collinear few-mass boxes (3023fa6, 149ddff),
  commutative DoubleFloat products (db151cd) and dilogarithm divided
  differences evaluated once in the interpreter (09b2c74). With 09b2c74 the
  wrong real-mass D0 real parts of `validation/rank-n2/README.md` findings 3
  and 5 are gone. Main's own code changes since dec3d55 add the cut-contribution
  cancellation report (af72488, 385c072), a separate accumulator that leaves
  every result unchanged. The fixture was therefore regenerated on a clean
  clone of main d53dd08 as described above; it is byte-identical (apart from
  the `baseline` label) to the one the rank n+2 final review generated
  independently the same way, and the merged branch passes it bitwise. Against
  `r-le-n1-dec3d55.json`, as scaled differences (largest component change
  over the lane's largest component):
  - fast engine with `TensorNumerator`: 69 of 237 width-1 lanes (27 of 66
    cases) and 754 of 2112 width-32 entries changed, median 2.0e-14. The
    largest change is 8.8e-5 on `syn-gghhh-ggHHH-n5-r5` lane 1, the cancelling
    lane next to a box with normalised Gram determinant 8.2e-8, whose D0 moved
    by 4.0e-11 at oneloopmaster 9c5490b; its error against the exact value goes
    from 3.4e-4 to 4.6e-4, as for Ninja's own binary64 reduction (3.0e-4,
    `validation/benchmark/ill-conditioned/README.md`, reported upstream). Next
    are 3.4e-10 (`syn-n6-r5` lane 0, which cancels) and 8.1e-11 (`syn-n6-r7`
    lane 2);
  - generic engine in binary64: 71 of 237 lanes (27 of 66 cases), median
    1.6e-14, largest 7.8e-5 on the same lane, then 3.5e-10 (`syn-n6-r5` lane 0);
  - explicit-`mu²` numerators: 9 (`GenericNumerator`), 12 (`MuTensorNumerator`)
    and 11 (generic binary64) of 26 lanes, in 4 of 9 cases, at most 7.6e-11
    (fast) and 1.5e-11 (generic) on `syn-n6-r7` lane 2;
  - generic `Float(256)`: 26 of 170 decimal components (7 of 17 lanes, cases
    `gg-hhh-top-pentagon-26`, `syn-n4-r5`, `syn-n6-r7`), by at most 7.8e-72 of
    the lane's largest component;
  - no lane status changed.
- The bubble tensor coefficients B1, B11 and B111 now come from
  oneloopmaster's native tensor families for every carrier, as B1111 already
  did (`crates/ninjars-symbolica/src/masters.rs`). The exact
  Passarino-Veltman forms lost 2-4 digits per decade of `m²/|p²|`, which made
  DoubleFloat and `Float(128)` bubbles at small `|p²|` with unequal masses
  silently wrong (rank n+2 final review, finding 3: relative errors 0.63 and
  9.9e-8 at rank 2, 1.3e13 and 2.0e6 at rank 4; now 2e-19 and below 1e-20,
  2.5e-15 and below 1e-20 against `Float(512)`). The fixture was re-blessed in
  place: only the generic `Float(256)` section moved, 43 decimal components of
  `syn-n2-r3`, `syn-n3-r4`, `syn-n4-r5` and `syn-n6-r7` in their last printed
  digit (about 1e-76 relative); the fast engine, generic binary64 and the
  explicit-`mu²` sections are bitwise unchanged.
- B0 now comes from oneloopmaster's native tensor family as well (OneLOop's
  `bub0`, slot 0 of `TensorIntegral::B11`, as `OneLoopNativeMasters` already
  does in binary64), for every carrier of `SymbolicaMasters`
  (`crates/ninjars-symbolica/src/masters.rs`). The exact scalar B0
  (oneloopmaster's generated code in DoubleFloat, its interpreted expression
  in `Float`) loses about `log10(|m0²-m1²|/|p²|)` digits at small `|p²|`:
  DoubleFloat 2.1e-21 relative at `p² = 1.86e-9` with `m² = 25, 2025`, which
  left the review's rank-4 small-`p²` reproducer 2.2e-15 wrong (now 2.8e-24;
  `Float(128)` 5.4e-22 -> 8.3e-31). The fixture was re-blessed in place: only
  the generic `Float(256)` section moved, 37 of 170 decimal components in 12
  of 17 lanes, by at most 2.4e-69 of the lane's largest component
  (`syn-n6-r7` lane 1). Against the same engine in `Float(512)` the changed
  lanes stay at their error level (4 closer, 3 farther, 5 with the same
  largest error; e.g.
  `syn-n6-r7` lane 1 4.3e-67 -> 4.2e-67, `syn-n3-r4` lane 2 3.8e-76 ->
  4.1e-76); the fast engine, generic binary64 and the explicit-`mu²` sections
  are bitwise unchanged.
