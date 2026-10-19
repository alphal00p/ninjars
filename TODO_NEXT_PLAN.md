# Next plan

Open work after NinjaRS main `fc1700e` (2026-10-19). Pins: oneloopmaster
`09b2c74`, gammaloop `ninjars-combined` `1e8063a7` (feynkit `8959f063` plus
PRs #142-#144 and #149-#151), Symbolica `e679e8c1`. Details of the NinjaRS
items are in `docs/rank-n2.md` (known limitations) and
`validation/rank-n2/README.md`.

## 1. gammaloop: land the NinjaRS PRs on feynkit, then re-pin

- feynkit's own NixCI is red for reasons older than these PRs. PR #152
  (`ci/feynkit-green`, head `ec148cbd2`) fixes what was reached so far:
  - clinnet Typst probes;
  - the spynso3 doc assertion and doctest;
  - stub export;
  - the notebook host wheel;
  - the docs figure;
  - a duplicate docs label;
  - broken source links.

  Jobs never yet completed in CI (core, integration, docs, python-api
  nextest; the rest of alphal00p-docs) may reveal more.
- Once feynkit is green: update #142 → #143 → #144 (stacked) and #149, #150,
  #151 from feynkit, wait for green, and merge them in order (never into
  gammaloop main).
- Then re-pin NinjaRS from `ninjars-combined` to the feynkit head. Update:
  - `Cargo.toml` and `Cargo.lock`;
  - `validation/performance/shared_provider` (manifest and lockfile) and the
    `validation/benchmark/ill-conditioned` lockfile;
  - `dependencies/manifest.json`;
  - the README dependency table.

  The physical DOT graphs should not change. Check by regenerating them
  (`prepare_hgg`, `prepare_four_photons`, `prepare_gluon_self_energy`,
  `validation/madloop/gg_hh{,h}/prepare_feynkit.py`), then compare the 72
  physical fixtures bit for bit (fast and generic, `--r2 off|fdh`).

## 2. NinjaRS fixes (documented as limitations)

1. **Generic binary64 conditioning on light-like lanes** (final review,
   finding 5). On 42 lanes with a binary64 error above 1e-9, generic is worse
   than fast on 25 and more than 10× worse on 9 (`fr-light-zero-n5-r7`
   lane 2: fast 3.7e-10, generic 3.6e-8).
   - Both engines use the same per-lane score. Generic has 8 bubble
     candidates and 6 tadpole bases (`cuts.rs::bubble_reference`,
     `tadpole_plans.rs`); fast has 25 and 9 (`fast/bases_x2.rs`).
   - Plan: first confirm the cause by forcing each engine's choice on those
     lanes. If the candidate count is the cause, widen the generic probe
     kernels to the 24 signed permutations of `(3/4, 1/2, 1/2, 1/4)`,
     preferably only for `RankClass::NPlus2`.
   - Any change to the shared kernels moves V0 bits and V0p fingerprints, so
     re-bless with a written justification.
2. **Over-declared rank in the fast engine** (finding 6). A rank n+1
   numerator declared at rank n+2 loses up to 981× in accuracy
   (`fr-light-equal-n5-r7` lane 1).
   - Plan: detect identically zero top coefficients, either at construction
     for `TensorNumerator` or per batch, and use the rank <= n+1 path
     instead.
   - Alternatively, document the declared rank as an accuracy contract. The
     CLI derives the rank itself and is not affected.
3. **`syn-n2-r2` width-1 reduction about 6 % slower** (code placement, same
   instructions).
   - Plan: move the rank n+2 fast engine into its own crate or codegen unit,
     or cold-mark its entry. Judge with cycle-timed ABBA runs on a quiet
     machine (`validation/performance/rank-n2-ab/scripts/run_all.sh`).
   - Also rerun that performance guard against current main: it last ran
     against dec3d55.
4. **`--stability-test`** does not yet measure cancellation among the terms
   of a single cut-constructible part. It now counts the cut-constructible
   and rational parts separately.
   - Plan: accumulate `sum |c_i M_i|` per cut in the integration stage, then
     compare the confusion table on the final review's lanes (generic f64
     and DoubleFloat) before and after.
5. **Fast binary64 at n = 7.** These tests stay ignored with "needs fast
   accuracy fix", at 5e-9..1e-5 on ill-conditioned lanes:
   - `pieces_reproduce_the_rank_n2_references_binary64`
   - `rank_n2_integrals_equal_their_pieces_binary64`
   - `relabelling_invariance_rank_n2_binary64`
   - `suite_fast_within_ten_times_generic_binary64`
6. **DoubleFloat at rank 4, small p²** (`repro-smallp2-bubble-r4.json`):
   2.5e-15 after the native bubble fix. The remaining loss lies outside the
   bubble masters and has not been traced.
7. **Benchmark report**: add a rank n+2 section to
   `docs/benchmark/benchmark.typ` (cost against the rank n+1 twins, accuracy
   tables from `docs/rank-n2.md`).

## 3. oneloopmaster (from "Exact contour-box discriminants at thresholds")

Commit as ValentinHirschi; fast-forward to oneloopmaster main once tests
pass. For each item, done means:
- a regression test;
- `cargo test --release --features generated-evaluators -- --test-threads=1`
  passes;
- native code is regenerated (`cargo run --release --features rust-codegen
  --example generate_native -- src/native`);
- the D0 SymJIT asset is rebuilt (`cargo run --release --no-default-features
  --example rebuild_evaluators -- assets/evaluators D0`) and its hash updated
  in `assets/evaluators/README.md`;
- D0 oracle sweeps (`tests/oracle_sweep.py`) show no regressions;
- retired instructions per D0 call on
  `validation/benchmark/ninja/*.masters.json` stay within 0.5 %.

Check the interpreted (Symbolica `Float`) route too: NinjaRS uses it at high
precision.

1. **Contour box near a vanishing Gram determinant: 3.6e-11.** This is
   behind NinjaRS `syn-gghhh-ggHHH-n5-r5` lane 1.
   - Reproducer: D0 with p² = (202500, 15625, 15625, 15625),
     s12 = 93305.84790585532, s23 = 62501.31001345166, m² = 29929 ×4,
     mu² = 1e4 (`tests/d0_vanishing_gram.rs`).
   - Exact value: −1.907030378722972e-10 + 8.834687875568863e-10 i.
   - Binary64 error: native and SymJIT 3.4e-11..3.6e-11, interpreter
     7.5e-12. It was 4.4e-12..1.4e-11 before 9c5490b; OneLOop double gives
     7.9e-15.
   - The box itself is well conditioned: a one-ulp input change moves it by
     4.6e-16.
   - Cause: `general` sorts the four λ's and puts the tiny λ(t, p2², p3²) into
     the first tfun. The poles of the linear s3 terms then lie within ~1e-5
     of their zeros (Gram/(p1² p2² p3²) = 1.4e-5), and two terms of ~1.4e4
     cancel to ~25.
   - OneLOop takes the first positive λ, which gives 3.3e-14 in our kernel.
     Over 200 random near-threshold boxes, neither rule is uniformly better.
   - Fix, preferred: (a) exact pole–zero offsets, i.e. y − z of the linear s3
     terms from Vieta and an exact kinematic quantity (Gram or a Cayley
     minor), rounded consistently, as `channel_gap` in 149ddff does. Else
     (b) an orientation rule that provably avoids pole–zero coincidences.
   - Target: 1e-13 on every binary64 route, with `d0_vanishing_gram.rs`
     tightened and a near-threshold regression scan.
2. **One-mass D0 with a wide complex mass: half of OneLOop's value**
   (`box_finite_one_mass`, OneLOop's boxf1).
   - Reproducer row (kind mu² p1²..p4² s12 s23, then re/im of the four m²):
     `4 1.0 P 14.780766530764295 12.25860671016521 -18.322615984332348 15.387619905549073 -17.481128979282158 0 0 0 0 0 0 3.0184355478828198 -0.301843554788282`
   - For P = −1, −0.1 and −1e-3 the results agree. Below P ≈ −1e-6 ours is
     almost exactly half of OneLOop's, at every precision:
     - P = −1e-6: OneLOop 6.112 + 383.0 i, ours 3.026 + 191.5 i;
     - P = −2.3e-10: OneLOop 5.577 + 431.0 i, ours 2.745 + 215.5 i.
   - With width −0.003 instead of −0.30 both agree. OneLOop's double and
     quadruple precision agree to 2e-9. The difference predates the
     collinear fixes (95f8107).
   - Second case: `4 1.0 8.584082897821794e-06 -6.169323358511654 13.537171246725286 -14.7771269138752 9.266576935067057 17.99194250487595 0 0 0 0 0 0 4.905727793442548 -0.4905727793442548`
     (OneLOop −7.553 − 7.551 i, ours −7.439 − 2.341 i).
   - Plan: compare our transcription with OneLOop's boxf1 (`avh_olo.f90`
     ~5399; called ~10387) term by term at these points: root lips, the
     sheet phases of X r34/r13 and X r24/r12, and the complements of
     3023fa6/149ddff. Settle which side is right with an independent
     Feynman-parameter integration at moderate P, or by continuity in the
     width.
   - If ours is wrong, fix it and add a test against OneLOop qp. If
     OneLOop's is wrong, document it with the evidence (as in
     `tests/data/real_mass_limits.txt`).

## 4. Housekeeping

- Local Codex branches with unpushed commits (keep or drop):
  - oneloopmaster: `codex/native-batch-outlining`, `codex/c0-dual-controls`,
    `codex/jit-shared-arity`, `codex/native-batch-masters`.
  - NinjaRS: the `references/ninjars_*` Codex worktrees whose branches are
    not in main (evidence and audit work).
- `references/ninjars_claude_n5r5` has untracked files. The scratch
  checkouts of the gammaloop re-pin preparation are dirty and can be deleted
  once item 1 is done.
