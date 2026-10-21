# Next plan

State after the landing of `claude/todo-next-integration` on NinjaRS main
(2026-10-21). <!-- PASS2: NinjaRS main hash after the landing push. -->
Pins: oneloopmaster `8ec256fc` (main), gammaloop `d1904bf2` (branch
`feynkit`, with PRs #142-#144 and #149-#152 merged), Symbolica `9cdc2949`
(alphal00p/symbolica `ninjars-pin`: upstream `c78cffc2` plus the patches of
`dependencies/patches`, among them symbolica-dev/symbolica#89 and #97).
Details of the NinjaRS items are in `docs/rank-n2.md` (known limitations),
`validation/rank-n2/README.md` and `validation/regression/README.md`; those
of oneloopmaster in its `STATUS.md`.

Sections 1 to 4 are the plan of 2026-10-19 (NinjaRS main `fc1700e`), done.
Section 5 is the open work.

## 1. gammaloop: the NinjaRS PRs on feynkit, then re-pin (done)

- PRs #152 (feynkit CI), #142, #143, #144, #149, #150 and #151 are
  squash-merged into alphal00p/gammaloop `feynkit`, head `d1904bf2`. The
  `nix-ci.nix` flag stays `enable = false`, as the feynkit maintainers set it.
- Re-pin `029af57` (merged in `0f57c89`): `Cargo.toml` and `Cargo.lock`,
  `validation/performance/shared_provider`, the
  `validation/benchmark/ill-conditioned` lockfile,
  `dependencies/manifest.json`, the README table and one sentence of
  `docs/physical-preparation.md`.
  - The diff `1e8063a7..d1904bf2` touches none of the 13 crate directories
    NinjaRS compiles.
  - The 70 physical DOT graphs (the plan's "72" was a miscount) and the 58
    gg→HH(H) sample fixtures regenerate byte-identically. 308 runs (70 graph
    fixtures plus 7 pools, fast and generic, `--r2 off|fdh`) give
    byte-identical results.
- `c740c8c`: `validation/madloop/gg_hh{,h}/prepare_feynkit.py` picks the
  target-profile rlibs; it failed at random before.

## 2. NinjaRS fixes (done)

Each branch is merged into `claude/todo-next-integration`; the merge commit
is given after the branch's commits.

1. **Generic binary64 conditioning on light-like lanes** (final review,
   finding 5): `86b378b` (rank n+2), `5ed3b22` (every rank class), `f9a8b16`
   (cost) and `f8fd2b4` (test precision); merged in `8ce5754`. The cause was
   the generic scores, not the candidate count: the generic cut provider now
   uses the fast engine's scores over 24 bubble references and six tadpole
   bases.
   - Review lanes (42, symmetric selection): worse than fast on 26 → 20,
     more than 10 times worse on 9 → 5, geometric-mean ratio 2.56 → 1.07.
     Fresh lanes: 1.10 to 1.30, depending on the seed set.
   - Rank <= n+1 against Float(256): controls geometric mean 0.715 (8 lanes
     more than 10 times better, 0 worse); 1494 twins 0.661 (54 better, 11
     worse). Single lanes can lose up to 3.3e3 times. V0 is re-blessed with
     this statistic (`validation/regression/README.md`).
   - `rn2-phys-real-n6-r7` lane 2: 4.26e-7 → 1.37e-8. Its known-failures
     entry and the Float(128) one of `rn2-irml-zero-n5-r6` lane 1 went stale
     and were removed.
   - Cost after `f9a8b16`: at most +1.5 % instructions per point on the
     generic bench (+6 % to +17 % on phi3 numerators at `5ed3b22`).
2. **Over-declared rank in the fast engine** (finding 6): `dc8c7e7` and
   `baaa240` (merged in `cdb5c07` and `7f1c324`), then `2d56861` and
   `f33ff8d`.
   - Lanes declared at rank n+2 whose numerator has a lower rank are reduced
     bitwise as declared at their own rank (`fast/declared_rank.rs`).
     `AtDeclaredRank` keeps the rank n+2 engine. Declaring rank n+1 for
     numerators of rank <= n still changes the results, as in original
     Ninja.
   - The lowered engines inherit `FastOptions::conditioned_cuts`, tested
     bitwise in `2d56861` (238 padding cases with 753 lanes; mixed-rank
     batches with 268 lane checks).
   - Given up by default: through the rank n+2 engine these lanes had been
     more than 10 times more accurate on 28 of the review's 166 twins and
     more than 10 times less accurate on 6, from the bases of `bases_x2.rs`.
     `conditioned_cuts` recovers those bases without over-declaring.
3. **`syn-n2-r2` width-1 outlier** (`73f53d9`, merged in `0c5b96c`): heap
   layout of the case-bench worker thread, not code placement.
   - Linking the rank n+2 code out of the hot span, `#[cold]` and
     `-z keep-text-section-prefix` leave 8346699 at 1.08–1.14; case-file
     padding moves it between 462 and 512 ns; with 64-byte aligned heap
     blocks it runs at 0.98 of dec3d55.
   - Guard against main `d53dd08` with `0d0d7e7` as the candidate: geometric
     mean 0.9996, every key <= 1.03 or <= 1.03 rerun alone.
4. **`--stability-test` inside one cut contribution** (`9ff3d0a`, merged in
   `6770a5f`): per-term moduli, each master integral times its weight and
   each rational term. Binary64 confusion on 1286 lanes 54/30/0 → 59/25/2;
   DoubleFloat unchanged at 11/10/0. The lanes still missed have inaccurate
   cut coefficients, not summation cancellation.
5. **Fast binary64 at n = 7**: the rank n+2 box geometry (`611f6dc`), the
   opt-in conditioned geometry for rank <= n+1 (`d0819f0`; a separate engine
   since `ec2df0f`, `cdc1930`, `a0e7b50`; CLI flag `acdcb81`; ignored
   default-geometry variants `414a57a`); merged in `9321240`.
   - `suite_fast_within_ten_times_generic_binary64` and
     `relabelling_invariance_rank_n2_binary64` pass with the default engine;
     the two V6 binary64 tests pass with their pieces through
     `conditioned_cuts`. The default engine's n = 7 gap stays visible in the
     ignored `*_default_geometry` variants.
   - `conditioned_cuts` stays opt-in (HANDOFF.md D3). On the control lanes:
     error 0.46 times the default's in geometric mean, lanes outside
     `max(10 e_generic, 1e-13)` 43 → 6, 20 lanes more than 2 times worse (up
     to 1.1e-10); +29.8 % instructions in the replayed reduction.
6. **DoubleFloat at rank 4, small p²** (`c3a6aba`, merged in `489c34c`): the
   remaining loss was the exact scalar B0; B0 now comes from oneloopmaster's
   native family for every carrier. Rank 4: 2.2e-15 → 2.8e-24 (DoubleFloat),
   5.4e-22 → 8.3e-31 (Float(128)); rank 2: 8.3e-29 and 7.8e-36.
7. **Benchmark report, rank n+2 section** (`6472644`, `4dec120`, `a198e50`,
   `7d9c01f`, `d1293bd`, `b992fbc`; merged in `14bb81c`): section 5 of
   `docs/benchmark/benchmark.typ`, with COLLIER 1.2.10 as the reference. The
   fast engine takes ×0.52, ×1.24, ×1.15, ×0.27, ×0.050, ×0.026 and ×0.011
   of COLLIER COLI's time for n = 1..7 (`docs/benchmark/data/rank_n2.json`,
   main `0d0d7e7`). COLLIER's global cache is wrong at N = 7, rank 9
   (`COLLIER_CACHE_BUG.md`, `fa7a789`).

## 3. oneloopmaster (done: main `8ec256fc`)

oneloopmaster main `8ec256fc` (2026-10-21) has the code of the verified
candidate `3a322d0`; `a8a3b8c` and `8ec256fc` add documentation. It landed
with its remaining tail as open work, by the user's decision ("land now,
track the tail"; `STATUS.md`, "Adversarial sweeps and the remaining tail
(landed 2026-10-21)").

1. **Contour box near a vanishing Gram determinant.** Line-edge pole offsets
   from Vieta's product (`dbb2217`) and the second contour where `K`
   vanishes (`e72b404`), with the later chart checks. The reproducer of
   `tests/d0_vanishing_gram.rs` was off by 3.6e-11; the test asserts the
   1e-13 target on every binary64 route. Worst of its 24 relabellings:
   2.6e-14, 1.9e-14, 3.4e-14 and 3.0e-14 (hand-written, generated,
   interpreted, SymJIT).
2. **One-mass D0 with a wide complex mass.** OneLOop's `boxf1` and
   oneloopmaster were both on a wrong sheet; Denner and Dittmaier's sheets
   for one- and two-mass boxes with complex masses (`9f663c5`, `a5da789`)
   fix it. Of 3000 random one-mass boxes 131 were wrong before and none
   after, against COLLIER 1.2.9 in quadruple precision.
3. **Also landed**: two-mass mass-pair complements (`66d84ec`), the Symbolica
   pin `9cdc2949` (`5cad984`, `f551e9e`), oneloopmaster PRs #9 and #10, the
   soft-box relabelling at the numerical D0 entry points (`d880be8`,
   pyAmpliCol's soft box) and the `d0_two_mass_width` test model.
   - Gates of `3a322d0`: full suite 267 passed, 0 failed; oracle sweeps as at
     09b2c74 except two documented boxes; instructions per D0 call -0.303 %
     in geometric mean, +0.444 % at most; definitions 95.8 MB against the
     gate's 100.7 MB.
   - Adversarial sweeps fs, ns and ns2 (99,119 referenced rows per route),
     binary64 route entries summed over four routes, D0 and C0 rows: finite
     and wrong by more than 1e-6 18,377 → 2,798; non-finite (all rows)
     31,076 → 3,162; DoubleFloat finite and wrong 3,321 → 21. Most of the
     gain is the soft-box relabelling: without the 14,400 soft relabellings,
     6,441 → 2,798 and (referenced rows) 8,056 → 2,534.
   - NinjaRS's 309 recorded master D0 calls: the worst binary64 error against
     COLLIER quad goes from 3.60e-11, 3.59e-11, 8.89e-12, 3.41e-11 to
     1.43e-12, 1.42e-12, 1.29e-12, 5.17e-12 (hand-written, generated,
     interpreted, SymJIT). On the 3,510 rank n+2 D0 calls it is 7.53e-11,
     7.48e-11, 7.51e-11, 2.58e-11 (SymJIT 1.66e-11 at 09b2c74). No call is
     more than ten times worse.
   - Against OneLOop 3.6 and COLLIER 1.2.10 on the same sweeps
     (`references/session-2026-10-20b/oneloop-compare/out/final_tables.md`):
     OneLOop's double and quadruple precision share 09b2c74's soft-box
     failure in non-cyclic labellings and 105 wrong complex-mass values;
     COLLIER COLI is the most accurate on D0 (0 to 0.13 % of the rows wrong),
     but wrong on 14 to 16 % of the ns and ns2 C0 rows, mostly with a
     vanishing Källén function and flagged; COLLIER DD is wrong on 4 to 9 % of
     the D0 and 6 to 21 % of the C0 rows (`collier.txt`, `shared.txt`).
4. **NinjaRS side**: re-pin `8a8bc55`; `SymbolicaMasters` with `Float`
   carriers relabels soft D0 boxes (`8e583f2`); `Float` master results with
   fewer than max(53, bits/4) tracked bits fail with
   `MasterFailure::ImpreciseResult`, so that a rescue escalates them
   (`2a7df0d`); V0 re-blessed once for the four causes (`c24f976`).

## 4. Housekeeping (done at landing)

- Integrated from HANDOFF.md D2: the paired-orientation docs (`725dbec`;
  the frozen probe stays on `archive/codex/paired-orientation-audit`), the
  pruned-validation harness (`4091986`, its control on the integration pins
  `d06503e`) and `examples/n5r5_exact.rs` (`2de63f9`).
  `COLLIER_CACHE_BUG.md` is committed (`fa7a789`).
- Also in the landing: Symbolica `9cdc2949` (`72a273c`), `dis_compare.py`
  matching the CLI binary's symbols (`759abc2`), the plan record (`115fb46`).
  <!-- PASS2: list the pass-2 commits (V0 Float(256) tolerance, known-failures
  re-derivation, tests) and one line with the outcome of the pass-2 checks
  (case-check in five modes, test suites, machine code and instruction
  counts; docs/rank-n2.md 7.2, 8.2, 8.3). -->
- The coordinator deletes the stale branches and worktrees of D2 and this
  effort's worktrees at landing (HANDOFF.md 3.F).

## 5. Follow-ups

### NinjaRS fast engine

- **Cheaper conditioned scoring.** Almost all of `conditioned_cuts`' +29.8 %
  replayed instructions is the per-lane scoring of 25 bubble references and
  9 tadpole bases. Then reconsider the default (HANDOFF.md D3): a default
  change moves V0's fast section and the cut-constructible/rational split
  that `fast_ninja_reference`, V5 and `case-bench` compare with original
  Ninja.
- **Triangle stage**: its `e3`, `e4` still come from Ninja's spinors (rank
  n+2 misses at about 5e-13, `rn2-exact-mu-n4-r6-perm` lanes 1, 5, 6).
- **Cancellation flag**, like the generic `--stability-test`: on the final
  review's fresh lanes, the 21 lanes above 1e-6 in binary64 all have cut
  contributions cancelling by 3e6 or more.
- **Rank n+2 speed**: the dense tensor expansions (`t3_expansion_x2`,
  `t2_expansion_x2`) take a median 64–76 % of the fast engine's time per n.
- **Tadpole scoring on light-like equal-mass kinematics** picks a basis 64
  times worse than Ninja's (`fr-light-equal-n5-r7` lane 1).

### NinjaRS generic engine

- **Small-|p| bubbles with unequal masses** (V9 `rn2-scan-bubp2-complex`):
  every fixed precision loses the same relative accuracy; it needs another
  bubble parametrization for |k| → 0.
- **Float(128) at thresholds**: numerica keeps the discriminant at about 106
  of 128 bits (a fix would be in oneloopmaster's `native/tensor.rs`).
- **Dual B0 at small p²**, and B0/B1/B11 each evaluating the whole B11
  family.
- **Box and triangle stages**: on 14 of the review's 41 lanes the generic
  error sits there, where no basis is chosen.
- **`--stability-test` misses**: binary64 25 of 84 wrong lanes. Options: a
  threshold factor (2^5 catches 77 of 84 with 11 false alarms) or an
  N=N-style test. `rn2-phys-real-n6-r8` lane 2 is a false negative (1.3e-6
  against a budget of 5.9e-7).
- **Remaining cost of the 24-reference selection**: box +1.4 %,
  pentagon-rank4 +1.0 % per point; dropping the constant reference columns
  would save about 0.25 %.
- **From HANDOFF.md D2**: a numerical prototype of paired bubble
  orientations (the symbolic inventory, 294 → 216 instructions,
  `docs/audits/paired-orientation-inventory.md`); regenerate and dispatch the
  native tadpole and bubble probe kernels for the new selection if the
  `native-kernels` work resumes.
- **Generic bench last digits**: representative results differ between
  d53dd08 and 0d0d7e7 on pentagon-rank4, heptagon-rank8 and gg-hh top-box-2;
  bisect, starting with `bc03412` and `6487309`.

### Masters and precision rescue (NinjaRS)

- **The rescue ladder** on the adversarial sweeps
  (`oneloop-compare/out/ladder.txt`): the binary64 failures of `3a322d0` are
  rescued by DoubleFloat in 90-98 % and by `Float` 256 or 512 bits in
  95-99.6 % (09b2c74: 15-33 %). 47 of the 100,680 rows are rescued at no
  level (exact-zero-Gram planar light-like boxes, mostly NaN at every level;
  OneLOop quad is NaN on most). On exact-zero-Gram planar rows higher
  precision can be worse (298 ns and 179 ns2 rows), which the tracked-precision
  check of `2a7df0d` addresses.
- **What the tracked-precision check misses**: 6 sweep rows report 81 to
  1023 tracked bits while wrong (ns 9210, 10378, 10965, 11620, ns2 5278,
  11326); ns 9210 tracks 253 of 256 bits and is wrong by 4.46.

### oneloopmaster

- **The tail** (`STATUS.md`, "Adversarial sweeps and the remaining tail"):
  - 21 silent binary64 boxes (18 ns, 3 ns2) outside the input-conditioning
    classes J and J* (28 with them), wrong by up to 0.36 where 09b2c74 was
    right or NaN; 10 of the 21 were NaN at 09b2c74. DoubleFloat is within
    3e-14 on all 21;
  - 195 silent `Float` rows at 128 and 256 bits, mostly exactly vanishing
    Gram determinants that were NaN at 09b2c74;
  - the non-silent regressions (counts with examples), the four Symbolica-pin
    boxes, the seed-6, seed-5 and seed-7 boxes and two soft boxes of the fs
    7225 kind.
- **The six follow-ups of that section**:
  1. `Float` at a rounding-level `K` or Gram determinant: return NaN, or
     check the chart as binary64 does; rerun ns at 128, 256 and 512 bits.
  2. binary64 NaN where the second contour (`e72b404`) is not resolved.
  3. A per-row `Float` comparison against the previous main in the
     oracle-sweep gate.
  4. Callers check numerica's tracked precision: done on the NinjaRS side
     (`2a7df0d`); rows like ns 9210 remain.
  5. Instruction headroom on syn-n6-r7: 0.04 % to 0.06 % of the 0.5 % gate.
  6. The rest of the tail: fix it, or accept it row by row.
- **Older open items** (`STATUS.md`, "Open D0 limitations" and "Open C0
  limitations"): opposite-mass two-mass boxes near collinear limits (6e-9 /
  1e-8); a real negative m² is not the −i0 limit; SymJIT below
  |Im m²|/Re m² ≈ 1e-150; the corner-singular box is NaN; the Gram
  determinant at the rounding level; `Float` three-mass boxes at 128/192
  bits; C0 with equal masses, on-shell legs and a vanishing invariant
  (`Float`); C0(0,0,0; m_complex, 0, m_real) wrong on every route, binary64
  included; interpreted `Float` C0 at 112–224 bits.
- **Debugging hooks** `ZZ_GEN_LOG` and `ZZ_GEN_DD` in `src/sheet_exact.rs`
  should go.
- **History-independent evaluators**: build oneloopmaster's Symbolica
  evaluators from fixed slot symbols (as `ninjars-positional` does) or after
  a fixed registration order, so that every bitwise test is thread- and
  history-independent. It moves `Float` master bits once (a V0 re-bless).
  Then the remaining `--test-threads=1` conventions can go, and so can the
  `--test-threads=1` advice in READMEs where it only came from the
  unlicensed one-thread limit.

### COLLIER

- The global-cache defect is documented in `COLLIER_CACHE_BUG.md` and not
  reported upstream (HANDOFF.md D4); the user forwards it if wanted.
  Optionally confirm its hypothesis with the instrumentation the note
  describes.
- `validation/rank-n2/harness` aggregates COLLIER's accuracy flag with `max`
  from 0 (`collier_tensor_driver.f90`, `collier_reference.py`); take the
  minimum if the flag is ever used.

### Dependencies

- gammaloop `feynkit` has moved on to `8cb80dfd`; re-pin when needed. The
  `ninjars-combined` branch can be retired.
- Symbolica PR #97 is still open upstream (as is #89): keep
  alphal00p/symbolica `ninjars-pin` `9cdc2949` until it merges. If upstream
  changes the approach, rebuild the pin and redo the `Float` checks.

### Requests from pyAmpliCol

From pyLoopAmpliCol (loop engine v2), received 2026-10-20; none blocks.
<!-- PASS2: if the pass-2 test that lanes of unrelated families (same n and
rank) give bitwise the same result batched as unbatched did not land, add it
here as a follow-up; otherwise name its test. -->

- **Finite-only mode**: what can be skipped when only the finite part is
  needed (pole-only Laurent terms, master pole parts, mu²/eps expansions
  feeding only poles); a cheap option or API if worthwhile (measure first).
- **Reduction-direction check**: a cheap stability check (q → −q, reversed
  cyclic order) reusing masters, or reusing cut solutions and geometry
  between the two passes. Mind oneloopmaster's D0 labelling
  (`regular_d0_labelling`).
- **Conditioning diagnostics** per cut and lane (smallest uncut-denominator
  leading coefficient, Gram determinant) for error budgets.
- **Integer master ids** in master requests (caching without hashing).
- **Upstream `TensorGenericNumerator<T>`** (now only in
  `crates/ninjars-cli/src/case_check/harness.rs`), so that DoubleFloat and
  `Float` ingress needs no Symbolica expression.
- **Later, only if measurements justify it**: a public jets API
  (`fast/tensor_jets.rs` is private) to share cuts between a parent family
  and its pinches; a transposed or multi-numerator reduction (many
  numerators, one geometry).

### Pre-existing small items

- 3 clippy warnings in the `ninjars-symbolica` lib.
- Stale provenance hashes in
  `examples/graphs/physical/gg-hh(h)/feynkit-graphs.json` (since `bc03412`;
  nothing reads them).
