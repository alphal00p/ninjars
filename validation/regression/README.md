# Regression fixtures for numerator rank <= n+1

Rank n+2 support must leave every existing rank bitwise unchanged. These
fixtures hold the results of NinjaRS main without any rank n+2 change, at the
current dependency pins (first main d53dd08 with oneloopmaster 09b2c74, which
contains fe550e3's rank-four bubble `B1111`; now oneloopmaster 8ec256fc and
Symbolica 9cdc2949, whose fast sections equal main 0d0d7e7's with these pins),
and must never be re-blessed to absorb a change of results for rank <= n+1
made by rank n+2 work. They are regenerated only from a clean checkout of
main, when merging main brings results that main itself changed (History).
Dependency re-pins carried by a branch before main has them are re-blessed
in place on that branch, with the fast sections checked bitwise against a
fixture blessed on main with only the same re-pins applied (History, the
oneloopmaster 8ec256fc entry). The one exception is a deliberate
accuracy change of the generic engine (the escalation path, whose bitwise
identity at rank <= n+1 is a regression convention, not a compatibility
contract): it is re-blessed in place, in its own commit, with a written
justification in History that quotes a symmetric error statistic against a
high-precision truth, including the lanes that get worse. The fast engine's
sections never move for such a change.

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
- The generic cut provider (`crates/ninjars-symbolica/src/cuts.rs`) now
  chooses bubble references and tadpole bases with the fast engine's scores
  at every rank class: 24 bubble references (the signed permutations of
  `(1, 2/3, 2/3, 1/3)`) scored by `|k.r|/(|k| |r⁰|) min_v |e3.v|/(|e3| |v|)`,
  and the six tadpole bases scored by `min_v min(|e1.v|/|e1|, |e3.v|/|e3|)/|v|`.
  86b378b had adopted them for rank n+2 only (final review, finding 5); below
  it the provider kept its eight references and its earlier scores so that
  this fixture stayed bitwise. The V0 fixture was re-blessed in place (commit
  "cuts: one bubble and tadpole selection for every rank class"); V0p is
  unchanged (no exact plan changed).
  - **Justification: a symmetric statistic.** Generic binary64 against
    `Float(256)` of the same build, before (86b378b, at rank <= n+1 bitwise
    equal to main 0d0d7e7) and after, as scaled differences (largest
    `|x - t|` over the three Laurent coefficients over the largest `|t|`).
    The new build's `Float(256)` agrees with its `Float(512)` to 5.0e-36 or
    better on all 2109 lanes. On the twins the two builds' `Float(256)` agree
    to 1.5e-40. On the controls they differ above 1e-30 on 130 lanes, all
    with a snapped S-matrix and a recorded `input_ambiguity`, by up to 4.3e-6
    (`syn-n6-r6` lane 0): the input ambiguity of
    `validation/rank-n2/README.md` (at most 1.23 times the recorded value), so
    each build is judged against its own `Float(256)`.
    Errors are clamped at 1e-16; "selected" lanes have an error above 1e-13
    before or after; ratios are after/before.

    | set | lanes | selected | geomean ratio (selected) | more than 10x better / worse (selected) | median before -> after (all) | p99 before -> after (all) |
    |---|---|---|---|---|---|---|
    | controls (`validation/rank-n2/controls` and `syn-n{2..6}-r{n,n+1}`) | 615 | 192 | 0.715 | 8 / 0 | 8.3e-15 -> 7.5e-15 | 9.1e-9 -> 3.2e-9 |
    | final-review twins (`references/final-review/twins`) | 166 | 79 | 0.605 | 4 / 0 | 5.7e-14 -> 4.4e-14 | 2.6e3 -> 2.6e3 |
    | fresh twins, review generator seeds 301-303 | 498 | 243 | 0.699 | 18 / 6 | 5.7e-14 -> 5.4e-14 | 1.7e4 -> 7.9e2 |
    | fresh twins, review generator seeds 101-105 | 830 | 396 | 0.649 | 32 / 5 | 5.3e-14 -> 3.4e-14 | 2.5e2 -> 1.3e2 |
    | all twins | 1494 | 718 | 0.661 | 54 / 11 | 5.6e-14 -> 4.4e-14 | 1.3e3 -> 3.0e2 |

    Over all lanes (clamped, no selection) the geometric means are 0.878
    (controls) and 0.841 (twins), with 9 / 0 and 62 / 23 lanes more than 10
    times better / worse; the 12 extra worse ones stay below 1e-13 (e.g.
    1.8e-16 -> 2.6e-15). Against the stored control references, as
    `case-check` judges them, the controls give the same picture (192
    selected lanes, geometric mean 0.716, 8 / 0).
  - **Single lanes can lose.** The 11 selected twin lanes more than 10 times
    worse: `fr-light-mixed-n4-r6-mu2-s301` lane 1 2.1e-3 -> 7.1 (3.3e3 times;
    fast engine 1.5e-4), then at most 26 times, e.g. `fr-phys-complex-n7-r9-mu2-s303`
    lane 0 1.1e-11 -> 2.9e-10 (fast 2.2e-10) and `fr-light-equal-n5-r7-s301`
    lane 0 1.3e-7 -> 2.3e-6 (fast 2.8e-8). The largest gains:
    `fr-light-equal-n7-r9-s102` lane 1 6.9e-4 -> 5.2e-8,
    `fr-light-equal-n7-r9-s301` lane 1 1.2e5 -> 1.0e2.
  - **The controls' outliers** of `validation/rank-n2/README.md`: `rn2-phys-real-n6-r7`
    lane 2 4.26e-7 -> 1.37e-8 (its `inaccuracies` entry in `known-failures.json`
    went stale and was removed), `syn-n2-r3` lane 1 3.0e-10 -> 2.0e-12,
    `rn2-phys-complex-n6-r7` lane 2 1.03e-7 -> 7.3e-10. The `Float(128)`
    inaccuracy of `rn2-irml-zero-n5-r6` lane 1 went stale as well
    (4.3e-32 -> 1.2e-32, tolerance 2.7e-32) and was removed.
  - **Fixture changes**, as scaled differences against the previous fixture:
    - fast engine (`TensorNumerator`, widths 1 and 32, 2349 entries) and the
      explicit-`mu²` fast sections (`GenericNumerator`, `MuTensorNumerator`):
      unchanged;
    - generic engine in binary64: 203 of 237 lanes (58 of 66 cases), median
      2.8e-15. The largest is 0.33 on `syn-n6-r6` lane 0, a lane that
      binary64 cannot resolve (with the full tensor, as in the controls,
      generic f64 is off by 1.4e-2 before and 7.3e-3 after; V0 keeps a
      sixteenth of its monomials), then 8.8e-6 on `syn-gghhh-ggHHH-n5-r5`
      lane 1 (both copies; the cancelling lane next to a box with
      normalised Gram determinant 8.2e-8) and 4.9e-8 on `syn-n6-r7` lane 1.
      The cut-constructible/rational split moves with the bubble reference
      (only the sum is reference independent): median 5.1e-13, up to 1.6e13
      of the lane on `syn-n6-r6` lane 0, whose two parts cancel;
    - explicit-`mu²` numerators, generic binary64: 23 of 26 lanes, at most
      4.6e-8 (`syn-n6-r7` lane 1);
    - generic `Float(256)`: 91 of 170 decimal components (14 of 17 lanes).
      The Laurent coefficients move only on the snapped-S-matrix `syn`
      lanes, by at most 1.3e-11 (`syn-n6-r7` lane 1, recorded
      `input_ambiguity` 2.1e-10), always within the recorded
      `input_ambiguity`; `gg-hhh-top-pentagon-26` moves by 5e-76;
    - generic binary64 against these `Float(256)` values: `syn-n2-r3` lane 1
      8.7e-11 -> 3.9e-12, `syn-n4-r5` lane 0 4.3e-12 -> 2.1e-12, `syn-n6-r7`
      lanes 1 and 2 4.8e-8 -> 2.9e-8 and 1.1e-8 -> 9.2e-9; the other lanes
      stay at 1e-15 to 4e-14 (largest change 1.2e-15 -> 6.4e-15, `syn-n2-r3`
      lane 0);
    - no lane status changed.
  - **Regenerating after a merge.** Another branch re-blessed the generic
    `Float(256)` section for its own change. After merging, bless on the
    merged tree and check that the fast sections are unchanged and that the
    generic sections differ from either branch's fixture only by the other
    branch's change.

- The fixture was re-blessed in place on the rank n+2 integration branch
  (`claude/todo-next-integration` at 2a7df0d, commit "regression: re-bless V0
  for the oneloopmaster and Symbolica re-pins"), once, for four causes
  together; V0p is unchanged. The convention above re-blesses dependency
  moves from a clean checkout of main. Here the two re-pins (oneloopmaster
  09b2c74 → 8ec256fc, Symbolica e679e8c1 → 9cdc2949) are carried by the
  integration branch, main's code is unchanged, and the previous fixture came
  from the generic-selection branch without the B0 change, so the re-bless is
  done in place and its fast part is attributed by a second fixture: blessed
  on main 0d0d7e7 with only the two re-pins applied (uncommitted) and this
  `regression_bitwise.rs`, its fast sections (`TensorNumerator` widths 1 and
  32, `GenericNumerator` and `MuTensorNumerator`: 2349 and 52 entries) are
  bitwise identical to the new fixture's. So the fast engine moves only
  through cause (c), and the generic sections differ from that main fixture
  only by causes (a) and (b). Against the previous fixture, as scaled
  differences (largest change of the three Laurent coefficients over the
  lane's largest one); no lane status changed:

  | section | changed | median | largest |
  |---|---|---|---|
  | fast `TensorNumerator`, width 1 | 62 of 237 lanes (31 of 66 cases) | 4.5e-15 | 4.8e-4, `syn-gghhh-ggHHH-n5-r5` lane 1 (both copies) |
  | fast `TensorNumerator`, width 32 | 607 of 2112 entries | 3.7e-15 | 4.8e-4, the same lane |
  | fast `MuTensorNumerator` / `GenericNumerator` | 6 / 6 of 26 lanes (5 of 9 cases) | 2.5e-14 / 3.0e-14 | 6.0e-9, `syn-n6-r7` lane 1 |
  | generic binary64 | 60 of 237 lanes (32 cases) | 4.3e-15 | 4.8e-4, `syn-gghhh-ggHHH-n5-r5` lane 1 |
  | generic binary64, explicit `mu²` | 8 of 26 lanes | 2.6e-14 | 6.1e-9, `syn-n6-r7` lane 1 |
  | generic `Float(256)` | 87 of 170 decimal components (15 of 17 lanes) | | 3.6e-67, `syn-n6-r7` lane 1 |

  The causes, each counted against the output before it:
  - **(a) B0 from oneloopmaster's native bubble family** (the B0 entry
    above, re-blessed on its own branch but not in the fixture taken from the
    generic branch): `Float(256)` only, 43 of 170 components (14 lanes), at
    most 1.8e-70.
  - **(b) the generic engine's bubble and tadpole selection at every rank
    class** (the entry above): already in the previous fixture. Against
    main's fixture it moves generic binary64 on 203 of 237 lanes,
    explicit-`mu²` generic binary64 on 23 of 26 and `Float(256)` on 91 of
    170 components, and no fast entry.
  - **(c) oneloopmaster 09b2c74 → 8ec256fc** (main; its code is that of the
    verified candidate 3a322d0, 8ec256fc adds documentation): the contour box
    next to vanishing Gram determinants, Denner and Dittmaier's sheets for
    one- and two-mass boxes with complex masses, two-mass mass-pair
    complements, the soft-box relabelling at the numerical D0 entry points.
    D0 changes in binary64 and DoubleFloat; C0 keeps its bits on the 534 and
    4,134 C0 calls recorded from NinjaRS runs. Of the 309 distinct recorded
    D0 calls, the binary64 bits move on 99 (hand-written native), 191
    (generated native), 191 (interpreted) and 219 (SymJIT) calls and the
    DoubleFloat bits on 200; on the 3,510 rank n+2 D0 calls on 1,666 /
    2,737 / 2,711 / 2,786 (DoubleFloat 2,735). The worst D0 error of the 309
    against COLLIER in quadruple precision goes from 3.6e-11 to 1.4e-12, and
    no call is more than ten times worse (oneloopmaster finisher's report).
    In V0 (integration 759abc2 → 8a8bc55) it moves every changed fast and
    binary64 entry of the table and 23 `Float(256)` components (8 lanes, at
    most 1.8e-67). The largest change, 4.8e-4 on `syn-gghhh-ggHHH-n5-r5`
    lane 1, is the D0 next to the box with normalized Gram determinant 8.2e-8
    (`validation/benchmark/ill-conditioned/README.md`): against the exact
    value the lane goes from 4.6e-4 to 3.1e-4 in the fast engine, the error of
    exact masters rounded to binary64 (3.09e-4: the 9c5490b regression of the
    d53dd08 entry is gone), and from 4.7e-4 to 7.8e-6 in the generic engine.
    The next largest are 7.0e-9 (fast) and 6.6e-9 (generic) on `syn-n6-r7`
    lane 1. The 44 complex two-mass sub-boxes of the six edge cases
    (`edge-low-rank-n5-r0` to `-r3`, `edge-many-n7-r6`, `-r7`) take the new
    sheets and stay within 7.0e-15 of COLLIER (09b2c74: 8.2e-15).
  - **(d) Symbolica e679e8c1 → 9cdc2949** (symbolica-dev/symbolica#97,
    `Float` products rounded once): `Float(256)` only, 82 of 170 components
    (14 lanes), at most 5.4e-67 (`syn-n6-r7` lane 1); binary64 and DoubleFloat
    bitwise unchanged.

  The branch's own changes in the same pass, the relabelling of soft D0 boxes
  on `SymbolicaMasters`' `Float` route and the tracked-precision check of
  `Float` master results (`MasterFailure::ImpreciseResult`), change nothing
  in V0: its output at 2a7df0d equals the re-pin's (8a8bc55) bitwise, with no
  status change in `Float(256)`.
