# Rank n+2 high-precision validation suite

This directory holds reference values, control cases and exact identity data for numerator
rank r = n+2. It is stream S5a of the rank n+2 design, extended by S5b with n = 7 cases
(section "n = 7"). Nothing here is Rust: the consumers are `ninjars case-check` and
`crates/ninjars-symbolica/tests/rank_n2_*.rs` (section "Checking the engines"). The algorithm and a summary of
the results are in [`docs/rank-n2.md`](../../docs/rank-n2.md).

| directory | content |
|---|---|
| `cases/` | 105 rank n+2 bench cases (315 lanes, n = 1..7), `ninjars-bench-case-v1` or `-v2` (integrand mu^2) |
| `reference/` | one reference file per suite case, plus `syn-n{1..6}-r{n+2}` (cases in `validation/benchmark/synthetic-rank-n2/`) |
| `controls/` | `cases/`: 195 rank <= n+1 twins on the suite kinematics. `reference/`: their references, plus references for `syn-n{2..6}-r{n,n+1}` of `validation/benchmark/synthetic/`. `ninja/`: original-Ninja results for 174 twins |
| `identity/` | `cases/`, `reference/`: 75 exact identity and invariance cases (n = 1..7). `manifest.json`: which lanes must agree. `padding-manifest.json`: the V5 cases. `residuals.json`: the identities on the references |
| `scan/` | 6 conditioning-scan cases (V9): a bubble p^2 ladder, and triangle and box Gram-determinant ladders |
| `scalars/` | exact scalar C0/D0 of every 3- and 4-propagator subset of every lane (inputs of the reduction, see below); `shift-invariants/` holds the same with the invariants taken from the shifts |
| `summary/` | one row per lane and set: source, tier, uncertainty and every cross-check |
| `provenance/` | the review's `hpref` values and the comparison with them |
| `harness/` | every script that produced the above, and the COLLIER driver |
| `known-failures.json` | lanes expected to fail for an upstream bug, optionally per engine and precision (`lanes`; empty since oneloopmaster 09b2c74 fixed findings 3 and 5), known fixed-precision inaccuracies of one engine (`inaccuracies`), and kinematics whose original-Ninja errors are no binary64 budget (`ninja_unreliable`), read by `case-check --known-failures` and the tests |

## Conventions

- **Integral.** Each lane holds the Laurent coefficients `eps0`, `eps_m1` and `eps_m2` of
  `I = mu_R^(2 eps) / (i pi^(D/2) r_Gamma) int d^D qbar N(q, mu^2) / prod_k D_k`, where
  `D_k = (qbar + V_k)^2 - m_k^2` and `qbar^2 = q^2 - mu^2`. The normalisation is Ninja's
  (OneLOop's), with `mu_R^2` = the lane's `mu_squared` and `eps_UV = eps_IR`.
- **Numerator.** `N(q, mu^2) = sum_k (mu^2)^k T_k(q)`. Each `T_k` is a dense tensor of rank
  `rank - 2k` in the four contravariant components of `q`, in Ninja's TensorNumerator order
  (`crates/ninjars/src/fast/tensor_layout.rs`).
  - Cases without mu^2 use `ninjars-bench-case-v1`.
  - Cases with mu^2 use `ninjars-bench-case-v2`, the layout of `export-bench --r2 fdh`
    (`crates/ninjars-cli/src/export_bench.rs`): a `mu_powers` field K and
    `tensor = T_0 ++ ... ++ T_K`. Their `r2` field is `"synthetic"`, because the numerators
    are random and come from no scheme.
  - Rank counts mu^2 twice, so a rank n+2 case has every block at its full rank n+2-2k.
  - `ninjars case-bench` (main) reads these v2 files with `MuTensorNumerator`.
- **Exact inputs.** Every input is taken exactly as the binary64 number its JSON value
  denotes.
- **Scaled differences.** These follow `tests/fast_ninja_reference.rs`: the largest `|x - ref|`
  over the three Laurent coefficients, divided by the lane scale `max_k |ref_k|` (the `scale`
  field). The scale is 1 for the one lane whose value vanishes exactly, a massless tadpole.

## Reference files

`reference/<case>.json` keeps the phase-1 format string `ninjars-reference-collier-v1`.
`case-bench --reference-dir` reads its `lanes[].eps0/eps_m1/eps_m2`. Each lane carries:

| field | meaning |
|---|---|
| `eps0`, `eps_m1`, `eps_m2` | binary64 values. `laurent_decimal` holds the 40-digit decimals; use those at high precision |
| `source` | the published evaluation: `mp`, `reduction-exact`, `coli-quad` or `dd-quad` (below) |
| `uncertainty` | scaled uncertainty of the published value (definition below) |
| `verified_to` | scaled difference to the closest valid COLLIER evaluation, i.e. agreement with a fully independent code |
| `uncertainty_independent` | `max(uncertainty, verified_to)` |
| `input_ambiguity`, `input_ambiguity_parts` | only for lanes whose `s_matrix` differs from the invariants of the shifts (snapped MadLoop-style S-matrices: `phys`, `syn`); see below. Included in `uncertainty` |
| `tier` | **A**: `verified_to <= 1e-8`, scalar masters verified, `uncertainty <= 1e-25`. **B**: `verified_to <= 1e-8`, `uncertainty <= 1e-9`. **C**: anything else (report only) |
| `checks` | scaled difference of every other evaluation to the published value |
| `scalar_check` | the worst relative difference between the lane's exact scalar C0/D0 and the closer of COLLIER's COLI and DD quad values |
| `collier_coli_quad_decimal` | the COLLIER COLI quad value |
| `condition`, `absum_over_scale` | cancellation in the COLLIER contraction: the sum of absolute terms over the result |
| `reduction` | `new_terms_over_scale` and `parts_over_scale`: the size of the new rank n+2 integrated terms relative to the lane (box mu^6, B1111, bubble mu^2 (q.e2)^2 and mu^4) |
| `ninja` (controls) | original Ninja's scaled error against the reference |

**Tolerances** (as implemented by `case-check` and the tests; section "Checking the engines").
- **Generic Float(256) with exact masters (V4).** `max(3 uncertainty, 1e-30)` on tier A and
  B lanes; tier C is report only.
  - This tolerance assumes that oneloopmaster's exact C0/D0, which NinjaRS and the reduction
    share, are correct. `verified_to` is the statement that does not depend on those masters.
  - The lanes of findings 3 and 5 failed until oneloopmaster 09b2c74 fixed that D0 (see
    "Known failures" below); their published values are COLLIER's.
  - Tier-B lanes whose uncertainty comes from `generic_engine_extrapolated` or
    `generic_engine_on_twins` (32 suite lanes: `phys` and `syn`) are report only: that
    uncertainty estimates the input ambiguity from the generic engine's own rank twins and is
    not a conservative bound. Only the part that sets `uncertainty` is examined: the other 30
    snapped tier-B lanes also carry a generic-derived part (up to 0.92 of their uncertainty,
    16 above 0.1 of it), but there an independent part (COLLIER, or the reduction, with the
    shift invariants) is larger and sets the uncertainty, so they are checked.
- **Binary64 (V4f).** The design's `1e-14 max(1, condition) + 3 uncertainty` does not hold:
  COLLIER's contraction condition does not predict the reduction's binary64 loss (on the
  controls the fast engine needs up to 1e5 times that at n = 6, e.g. 2.6e-6 on
  `rn2-phys-real-n6-r7` lane 1, condition 1.7). Original Ninja's own error does. On the 543
  control lanes with their own Ninja result (115fb46: oneloopmaster 8ec256fc, Symbolica
  9cdc2949; no known failure among them):
  - the fast engine is within 12.5 times Ninja's error, with or without `--conditioned-cuts`;
  - the generic engine in f64 is within 12.7 times it, apart from `rn2-phys-real-n6-r7` lane 2
    at 20 times (1.4e-8 against 6.7e-10). Before the generic engine took the fast engine's
    bubble and tadpole selection (5ed3b22), on the 531 lanes of the time (known failures
    excluded), both engines were within 12.6 times it, apart from three `phys`/`syn` lanes for
    the generic engine: 632 times on `rn2-phys-real-n6-r7` lane 2, 42 on `syn-n2-r3` lane 1, 17
    on `rn2-phys-complex-n6-r7` lane 2;
  - with Ninja's error estimated from the rank twins (the mu^2 twins, and the suite), the
    generic engine is within 56 times it on the checked control lanes and 63 times on the
    checked suite lanes (`rn2-int-complex-n6-r8` lane 1).

  The binary64 tolerance is therefore `3 unc + max(1e-14 max(1, cond'), min(A e_ninja, 1e-4))`:
  - `cond' = max(condition, reduction.new_terms_over_scale)`. The new rank n+2 terms can
    cancel against the rest of the lane far beyond COLLIER's condition: 5e7 on
    `rn2-phys-real-n6-r8` lane 2 (condition 5.6), where the generic engine in f64 is off by
    1.1e-3 while Float(256) agrees to 5e-9, and 1.4e6 on `rn2-phys-complex-n6-r8` lane 2
    (generic f64 5.4e-4). Both are report only (`cond' > 1e6`). `case-check --stability-test`
    marks both `Unstable` in binary64, and lane 1 of the complex case (tier C, new terms
    9e10 times the lane).
  - `A` = 30 for the fast engine and 100 for the generic engine, for every lane. The one lane
    that the generic engine missed by more, `rn2-phys-real-n6-r7` lane 2 (4.3e-7, against
    6.7e-10 for original Ninja and 3.7e-10 for the fast engine), was listed individually in
    `known-failures.json` (`inaccuracies`, below) instead of widening `A` for every lane.
    Its loss was not the snapped S-matrix (the generic engine gave the same 4.3e-7 with the
    shift invariants) nor cancellation between cut contributions (`--stability-test` passes
    it). The generic engine's bubble and tadpole selection (5ed3b22) brought it to 1.4e-8,
    within its tolerance (6.7e-8), and the entry was removed. (Before that entry, `A` was 1000
    for the generic engine, set by this lane alone.)
  - `e_ninja` is Ninja's error on the lane (`ninja` field), or for lanes without one (rank
    n+2, mu^2 twins) on the rank twins of the same kinematics (`summary/{controls,
    synthetic-twins,n7-controls}.json`), extrapolated geometrically by one rank from the two
    highest twin ranks. It is not used on the kinematics of a known failure where that entry
    applies. On the kinematics listed in `ninja_unreliable` (`known-failures.json`), original
    Ninja's errors contain OneLOop's own D0 error of findings 3 and 5 (0.10 to 3.4), so they
    are replaced by Ninja's error against the exact reduction with oneloopmaster fe550e3's
    identically wrong D0 (`errors`, by rank excess): 1.3e-12, 4.9e-12 and 1.5e-11 at excess
    -1, 0 and 1 for `rn2-int-mixed-n6` lane 1, instead of 0.10, 0.30 and 0.24. The same
    comparison reproduces Ninja's published errors on lanes 0 and 2 of those cases to 4
    digits. `rn2-int-real-n7` lane 0 has no Ninja budget: OneLOop's wrong D0 values differ
    from fe550e3's there (Ninja against that reduction: 0.41 to 1.5).
  - Generic-engine-derived uncertainties count 100 times; `cond' > 1e6` and tier C are report
    only.
- **Resolution of the new terms (every precision).** A tolerance at least as large as the
  smallest new rank n+2 term of the lane (`reduction.parts_over_scale`; parts below 1e-30 count
  as zero) cannot detect that term being wrong or missing. Such a lane is report only while it
  is within the tolerance, and still fails outside it. In binary64 this concerns 23 suite
  lanes for the generic engine and 22 for the fast engine (mostly `B1111` parts of 1e-14 to
  3e-6 of the lane on `-mu2` and `-muall` lanes); in Float(256) none.
- **Known failures.** A lane of `known-failures.json` (`lanes`) is compared with COLLIER's COLI
  quad value (`collier_coli_quad_decimal`) at a scaled 1e-6 (`case-check
  --known-failure-tolerance`): far above COLLIER's error on such lanes, far below an upstream
  bug's effect (the D0 bug of findings 3 and 5 moved its lanes by at least 3 %, 3.1e-2 on
  `rn2-int-mixed-n6-r8` lane 1). While it differs by more, it is `known-failure` (with the
  measured difference); once it agrees, or passes its own tolerance, it is
  `stale-known-failure`, which fails `case-check` and the tests until the entry is removed.
  - **Scope.** An entry may list the `engines` (`fast`, `generic`) and `precisions` (`f64`,
    `double`, `float:<bits>`) it applies to; without them it applies everywhere. Elsewhere the
    lane is judged as any other, so an entry for a defect of, say, DoubleFloat and Float(128)
    neither fails the other precisions nor goes stale in them (`known_failure_scope` in
    `crates/ninjars-cli/tests/case_check.rs`).
  - **History.** Until oneloopmaster 09b2c74 (main's pin since d53dd08) the list held the 24
    suite and control lanes of findings 3 and 5. With 09b2c74 the exact scalar C0/D0 of the
    suite change only on those five boxes, by 51, 9.4, 4.7, 3.3 and 2.1 times their modulus
    (exactly the old errors against COLLIER), while the other 4680 of 4692 agree with
    `scalars/` to 6.7e-73. All 24 entries were removed:
    - the 15 control lanes agree with COLLIER in all five runs (fast; generic f64, double,
      float:128, float:256);
    - of the 9 suite lanes, 7 agree with COLLIER within 1e-6 in DoubleFloat, Float(128) and
      Float(256) (the 4 without mu^2 to <= 2.5e-14, the mu^2 ones to 1.2e-13..3.6e-7, where
      COLLIER's two evaluations differ by up to 1.3e-5);
    - `rn2-int-real-n7-r9-muall` lanes 0 and 1 (COLLIER condition 1.2e18 and 7.6e17) differ
      from COLLIER by 0.86 and 0.14 in DoubleFloat, Float(128) and Float(256). These agree with
      Float(512) to 1.2e-14, 5.4e-22 and 7.9e-75 of the lane (lane 1: 4.2e-16, 2.8e-22,
      1.1e-72), while COLLIER's own second evaluation (`id-coli-quad`) differs from its first by
      190 and 12. The engines are right and COLLIER is not;
    - in binary64 the n = 7 mu^2 lanes (condition 5e9..1e18) miss COLLIER by their
      conditioning (fast up to 1.1e3 on the `-muall` lane 0).

    These lanes keep their COLLIER references: tier B at n = 6 (now passing everywhere) and
    tier C, report only, at n = 7.
    Rerunning `reduction-exact` with scalars regenerated at 09b2c74 would publish independent
    values for them; that is not done yet.
- **Known inaccuracies.** `known-failures.json` also lists, under `inaccuracies`, lanes on
  which one engine at one fixed precision misses its budget although the reference is right
  (`id`, `engine`, `precision`, `limit`, `reason`; one entry per lane, engine and precision).
  Such a lane is `known-failure` while its difference stays within `limit`, `fail` beyond it,
  and `stale-known-failure` once it is within its tolerance. Other engines and precisions
  judge the lane as usual. Limits are about 3 times the measured difference. `Float`
  evaluates C0/D0 with Symbolica's precision-tracking interpreter of oneloopmaster's exact
  expressions, the fixed-precision carriers with its native code. Float(160) and above are
  limited by the references' uncertainty, so the losses below are invisible in Float(256).
  Today (115fb46: oneloopmaster 8ec256fc, Symbolica 9cdc2949) there is one entry:
  - `rn2-int-mixed-n6-r6` lane 2 (a control) for the generic engine in Float(128): 2.6e-31,
    1.4 times its tolerance (1.8e-31, the binary64 budget scaled to 128 bits), limit 1e-30.
    - **The D0.** All but 1.1e-32 of it is the D0 of propagators 0, 1, 2, 5 (masses² 100 - 0.1i,
      0, 6464.16 - 168.84i, 29929), whose term is 7.8e3 times the lane. oneloopmaster's D0 in
      `Float` at 128 bits is 1.0e4 units of 2^-128 off there on Symbolica's interpreter (the
      route of `SymbolicaMasters`), 7.1e3 on its native `Float` code, in all 24 labellings of
      the box. Binary64 is 7.8 units of 2^-53 off and DoubleFloat 20 units of 2^-106.
    - **Not flagged.** The value keeps 119.7 tracked bits, so the tracked-precision check
      (`MasterFailure::ImpreciseResult`, below max(53, bits/4)) passes it.
    - **The exact value.** Float(256), Float(1024) and `scalars/` agree on the D0 to 45 digits,
      and COLLIER's quad C0/D0 of the lane agree with the exact ones to 1.5e-17
      (`scalar_check`).
    - **Cause of the new failure.** Symbolica 9cdc2949 (symbolica-dev/symbolica#97, `Float`
      products rounded once) moved this D0 from 1.5e3 units and the lane from 3.5e-32, a pass.
      Bisected with release CLIs: 2de63f9 and 72a273c, which differ only in that pin, give
      3.5e-32 and 2.5e-31; oneloopmaster 8ec256fc changes it to 2.6e-31.
    - **Other precisions.** Binary64 (2.6e-11), DoubleFloat (1.2e-26) and Float(256) (6.3e-41)
      pass.

  History of the entries:
  - **5ed3b22.** The generic engine's bubble and tadpole selection removed two: the generic f64
    entry of `rn2-phys-real-n6-r7` lane 2 (4.26e-7, limit 1e-6; now 1.4e-8) and the Float(128)
    entry of `rn2-irml-zero-n5-r6` lane 1 (4.3e-32; now 3.5e-33).
  - **Symbolica 9cdc2949.** The other 14 Float(128) entries went stale and were removed after
    115fb46: 3 suite lanes (`rn2-int-mixed-n4-r6` lane 2, `rn2-int-mixed-n6-r8` lane 0,
    `rn2-int-mixed-n7-r9` lane 1) and 11 control lanes. They were 1.0 to 10 times outside the
    binary64 budget scaled to 128 bits. They now differ by 2.4e-36 to 3.0e-33, at most 0.35 of
    their tolerances; the same bisection makes all 14 stale at 72a273c.
  - **The D0 on those lanes.** On the 139 boxes of the 15 lanes listed until 5ed3b22,
    Float(128)'s D0 (Symbolica's interpreter) was off by a median 471 and at most 1.4e5 units
    in the last place at 128 bits with Symbolica e679e8c1 and oneloopmaster 09b2c74, against
    34 and 4670 in binary64 and 29 and 4120 in DoubleFloat: up to 10 bits more (C0: at most 4
    bits more). With 9cdc2949 and 8ec256fc it is off by a median 80 and at most 2.4e4 units
    (binary64 28 and 4.6e3, DoubleFloat 30 and 3.7e3); C0 by a median 7.3 and at most 383
    units (before: 8.4 and 811).

## How the values are obtained

All preprocessing is exact rational arithmetic (`gmpy2.mpq`):
- mu^2 is removed with `mu^2 = q^2 + 2 q.V_a + V_a^2 - m_a^2 - D_a`, and scaleless pieces
  vanish;
- every piece is shifted to its reference propagator;
- the optional rank reduction is exact.

| evaluation | what it is | used for |
|---|---|---|
| `coli-quad`, `dd-quad` | COLLIER 1.2.9 built with `-freal-8-real-16`, COLI and DD branch, `SetReqAcc_cll(1e-20)`. The tensor components are contracted with the exact coefficients in 90-digit decimals, and the driver reads every input as its **exact** decimal expansion | cross-check (`verified_to`); fallback source |
| `id-dd-quad`, `id-coli-quad` (n >= 5) | exact rank reduction `l.p_j = (D_j - D_0 + f_j)/2` over four momenta down to rank <= N pieces, evaluated with DD (resp. COLI) quad and shift invariants | cross-check (DD cannot evaluate rank n+2 at n = 5, 6) |
| `coli-quad-shifts` | COLI quad with the invariants taken from the exact shifts | input ambiguity |
| `mp` (n <= 2) | `harness/mp_lowpoint.py`, with no COLLIER code. Tadpoles in closed form. Bubbles through a Feynman parameter, with exact polynomial algebra and mpmath quadrature of `int x^b log(Delta(x) - i0)` at 50 and 70 digits | **source** for n <= 2 |
| `reduction-exact` (n >= 3) | the review's from-scratch 60-digit Ninja-style reduction (`harness/review_reduction/red.py`, copied verbatim), assembled with the theory's integrated values, mpmath A0/B_n, and oneloopmaster's exact scalar C0/D0 at 256 bits (`scalars/`). `reduction-exact80` is the same at 80 digits | **source** for n >= 3 |
| `reduction-exact-shifts` | the same reduction with every invariant taken from the shifts (inconsistent lanes only) | input ambiguity |
| `generic-f256` (rank <= n+1) | NinjaRS generic engine (`SymbolicaAmplitudeKernels`) in Float(256) with `SymbolicaMasters` | validates `mp` and `reduction-exact`; input ambiguity |
| `coli-double`, `dd-double` | binary64 COLLIER | diagnostic: what a binary64 code reaches |

**`uncertainty`** is computed as follows:
- for `mp`, its quadrature error estimate (|50 - 70 digits|);
- for `reduction-exact`, the difference between 60 and 80 digits;
- for a COLLIER source, its largest difference to another valid COLLIER evaluation;
- in all cases, a floor of 1e-38 (the precision of the 40-digit decimals);
- where an input ambiguity exists, it is raised to that ambiguity.

COLLIER runs with `errflag < 0` are discarded and listed in `checks_invalid`.

**Input ambiguity.** On a lane whose `s_matrix` is not exactly the invariants of its shifts,
the integral is defined only up to how a code combines the two. Correct codes then differ, and
the difference grows with the cancellations, so with n and with the rank.
`input_ambiguity` is the largest of the following:
- COLI quad with the s_matrix against COLI quad with the shift invariants;
- `reduction-exact` against `reduction-exact-shifts`;
- the generic engine against the published value, where the generic engine runs (rank <= n+1);
- the generic engine on the lane's rank <= n+1 twins (same kinematics);
- that twin deviation extrapolated one rank further, geometrically from the two highest twin
  ranks.

On the controls the generic engine shows by far the largest of these deviations: up to 3e-10 on
`phys` n = 6 rank 7, and 8e-5 on `syn-n6-r6` lane 0, which cancels by 1e12. The extrapolation
makes 5 `phys` n = 6 suite lanes and 2 `syn` lanes tier C.

### Findings that affect the design's V4 numbers

1. **COLLIER in quad precision is not a quad-precision reference.**
   - **Scalars.** Against oneloopmaster's exact values on 3012 suite scalars, COLI's C0/D0 are
     accurate to about 1e-16 (median 8e-17, max 4e-15) and DD's to about 1e-18 (median 1e-18,
     max 4e-17).
   - **Tensor results.** They are off by up to 5e-10 (COLI) and 2e-10 (DD) on the suite lanes,
     by 1.6e-3 on `syn-n6-r8` lane 1, and by 1e-4 on the box Gram scan.
   - **Self-reporting.** COLLIER's own error estimates and flags miss all of this: they give
     1e-22 for 5e-10 errors. DD fails silently on several identity pieces of `phys` hexagons
     (errflag -7, O(1) errors).
   - **Accuracy setting.** `SetReqAcc_cll` below 1e-20 breaks COLI: at 1e-32 its result is off
     by 0.8 relative.
   - **Consequence.** COLLIER serves as the independent cross-check here, not as the published
     value.
2. **The phase-1 harness passed `repr(float)` to the driver.** This is the shortest round-trip
   decimal, which the quad build reads as a different number: up to 1.1e-16 relative per input.
   - The error comes on top of the binary64 preprocessing that the review corrected.
   - It is also present in the review's `hpref` values: `--input-rounding repr` reproduces them
     to 3e-40.
3. **oneloopmaster's exact D0 has the wrong real part for one real-mass box.**
   - This is phase-1 finding 1, seen there in OneLOop and in the native path. The exact
     `SymbolicaMasters` path has it too.
   - The box: `D0(p^2 = 20602, 6775, 113296, -106553; s12 = 80235, s23 = -4769;
     m^2 = 8315.178393760001, 100, 8315.178393760001, 0; mu_R^2 = 8315.178393760001)`.
     oneloopmaster gives `+1.714058e-8 - 8.341413e-9 i`. COLLIER's COLI and DD agree with each
     other to 6e-27 on `-9.718748e-9 - 8.341413e-9 i`.
   - It enters `rn2-int-mixed-n6-r8` lane 1 and the twins `rn2-int-mixed-n6-r{5,6,7}` lane 1,
     at 3 % of the rank n+2 lane and 10-30 % of the twins. On those lanes the published value is
     COLLIER's (tier B).
   - NinjaRS disagrees there by 0.1-0.3, both the generic engine in Float with exact masters
     and the fast engine with native masters, and will until oneloopmaster is fixed.
4. **The review's B_n quadrature shifted `Delta` by `-1e-45 i |scale|`.**
   - At on-shell thresholds (`irsoft` lanes, `Delta = x^2 M^2`) that shift costs 1e-21.
   - `e2e_reduction.py` uses the exact `-i0` prescription instead and now agrees with the
     generic engine to 4e-40.
5. **The oneloopmaster D0 bug of finding 3 is not isolated (S5b, n = 7).** Four more
   real-mass boxes of the n = 7 kinematics give an exact (and native, and OneLOop) D0 whose
   real part differs from that of COLLIER's COLI and DD, which agree with each other to
   8e-20..6e-16; the imaginary parts agree. `|D0 - D0_COLLIER| / |D0_COLLIER|`:

   | lanes | box (p1^2..p4^2; s12, s23; m0^2..m3^2) | error |
   |---|---|---|
   | `rn2-int-real-n7-*` lane 0 | (126627, 117904, 4340, -33109; -38553, 200750; 6464.16, 15625, 100, 29929) | 51 |
   | `rn2-int-real-n7-*` lane 0 | (200750, 247667, 5480, -73039; -61989, 98035; 15625, 29929, 15625, 15625) | 9.4 |
   | `rn2-int-real-n7-*` lane 1 | (127304, 248388, 26377, -115917; -95504, 64791; 29929, 6464.16, 100, 6464.16) | 4.7 |
   | `rn2-shifted-mixed-n7-*` lane 1 | (103359, 84842, 9291, -10106; -31393, -31173; 100, 100, 0, 6464.16) | 3.3 |

   (The finding-3 box: 2.1.) A direct Feynman-parameter integration of two of these boxes
   (S5b verifier) agrees with COLLIER.

   These lanes (all ranks, all mu^2 variants: 20 lanes) fail `scalar_check`, so their source
   is COLLIER's COLI; without DD at n = 7 they have no cross-branch uncertainty and are
   tier C. They were listed in `known-failures.json` until oneloopmaster 09b2c74 fixed this D0
   (and finding 3's); see "Known failures".
6. **The D0 bug depends on the order of D0's arguments (S5b fix-up).** In
   `tests/fast_parity.rs` (V8), lane 3 of `parity_rank_n2_n6` (n = 6, masses alternately zero
   and real) changes by 2.0 and 4.4 relative (with and without mu^2 terms) in both engines
   when its propagators are listed in reverse order. With a rank-0 numerator, the scalar
   hexagon, the two orders differ by 2.7 %: -68.89 - 81.09i in the original order and
   -71.20 - 82.95i reversed, in binary64 and in Float(256) alike, while COLLIER's COLI quad
   gives -71.20 - 82.95i for both. Of its 20 triangles and 15 boxes, one box differs
   from COLLIER: `D0(p^2 = 1.06248, -1.04543, 4.20999, 1.69788; s12 = -0.380152,
   s23 = 5.68250; m^2 = 0, 0.650833, 0.765473, 1.29829)`, whose exact (and native) value has
   the real part 5.3993 instead of COLLIER's -6.9964 (imaginary part 0.70962 in both). With
   its arguments in reverse order, oneloopmaster gives COLLIER's value. So the bug of findings
   3 and 5 is not confined to special kinematics: it appears on a random V8 point and is
   avoided by a different argument order. The V8 tests compare the engines at equal ordering
   and pass; they print `FAST_PARITY_UNSTABLE` for such points.

## Validation

### Independent methods against each other

| comparison | lanes | result |
|---|---|---|
| `reduction-exact` vs NinjaRS generic Float(256), controls, consistent kinematics (`int`, `shifted`, `bub0`, `irml`, `irsoft`; with and without mu^2) | 246 | max 1.7e-38 |
| `mp` vs NinjaRS generic Float(256), controls, consistent kinematics | 126 | max 3.9e-40 |
| the same on inconsistent (snapped) lanes | 132 | max 7.9e-5 (`syn-n6-r6` lane 0); within `uncertainty` on every lane |
| NinjaRS generic Float(256) vs the 400-bit values in `validation/benchmark/ill-conditioned/n6r6.jsonl` (`syn-n6-r6`) | 3 | max 1.1e-40 |
| `reduction-exact` at 60 vs 80 digits, suite | 201 | 171 identical in all 40 digits; max 2.3e-26 |
| suite: published vs closest COLLIER (`verified_to`) | 291 | median 4e-20, max 4.8e-10 |
| analytic 1/eps pole of the rank 2N-4 integrals (`uv_check.py`, IR-finite lanes), suite / controls | 45 / 72 | max 2.6e-31 (the COLLIER-sourced lane), otherwise <= 2e-42 / max 8e-41 |
| `test_mu2.py`: `int mu^(2b) / prod D` against the dimension-shift closed form for k = 1..6 propagators and b = 1..4, plus `mu^2 q^mu / D^3` | 56 | quad COLLIER <= 6.3e-20 (the mu^8 hexagon); mp <= 1.6e-40 |
| identities on the published references (`identity/residuals.json`): V6a / V6b / V7a / V7b | 27 / 12 / 28 / 84 | n <= 6: max 0 / 2.9e-40 / 0 / 0; n = 7: 5.4e-52 / 1.5e-40 / 5.2e-39 / 1.5e-39 (40-digit rounding). COLLIER quad: 2.4e-14 / 1.7e-14 / 1.2e-14 / 1.4e-8 (5.2e-12 at n <= 6; 1.4e-8 on `rn2-exact-mu-n7-r9-perm` lane 7) |
| NinjaRS fast engine on main (binary64, native masters, `case-bench`) vs control references, including the mu^2 twins | 531 | <= 3.2e-9, apart from the finding-3 lanes. S5b correction: `case-check` on every lane of the n <= 6 controls finds up to 2.6e-6 (`rn2-phys-real-n6-r7` lane 1, where original Ninja is off by 3.6e-6); see "Checking the engines" |

### Against the review's `hpref` (`provenance/hpref-comparison.json`)

`compare_hpref.py` covers 36 lanes of 12 cases:
- **Port check.** `--input-rounding repr`, the review's driver input, reproduces `hpref` to at
  most 3.3e-40 on every lane, so the port of `collier_hp.py` is exact.
- **Differences.** The new references differ from `hpref` by at most 1.4e-11 (median 1e-16).
- **Accounting.** Two measured errors explain the difference on every lane, so that
  `new_vs_hpref <= input_rounding + collier_error + uncertainty` holds on all 36:
  - `input_rounding`: hpref's decimal input, up to 3.7e-16;
  - `collier_error`: COLLIER quad's own error against the new reference, up to 1.4e-11.
- **Conclusion.** `hpref` agrees with the new references within its own uncertainty, which is
  now measured. It is not good to quad precision, and neither were the phase-1 stated
  uncertainties: on `rn2-int-complex-n5-r7` lane 0 the stated uncertainty is 8.2e-14 and the
  actual error 1.9e-12.

### Uncertainty table (suite, by n; `harness/tables.py summary/suite.json --by family` for families)

| n | rank - n | lanes | A/B/C | source | max unc. | max verified_to | max input amb. | COLI quad err | DD quad err | COLI binary64 err |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 33 | 33/0/0 | mp33 | 1e-38 | 1e-33 | - | 1e-33 | 1e-33 | 1e-15 |
| 2 | 2 | 54 | 42/12/0 | mp54 | 1e-16 | 1e-31 | 1e-16 | 1e-31 | 3e-18 | 4e-13 |
| 3 | 2 | 42 | 40/2/0 | coli2,red40 | 4e-17 | 3e-16 | - | 2e-14 | 3e-16 | 2e-10 |
| 4 | 2 | 54 | 39/15/0 | red54 | 4e-14 | 1e-10 | 4e-14 | 1e-10 | 2e-10 | 1e-08 |
| 5 | 2 | 54 | 39/15/0 | red54 | 3e-12 | 2e-12 | 3e-12 | 3e-11 | - | 2e-01 |
| 6 | 2 | 54 | 38/11/5 | coli1,red53 | 2e-07 | 5e-10 | 2e-07 | 5e-10 | - | 5e-04 |

The 291 lanes split into tier A 231, B 55 and C 5.
- Tier A has `uncertainty <= 1.3e-36` on every lane, with `verified_to` up to 1e-10.
- Tier B consists of the `phys` lanes at n = 2, 4, 5, 6 (input ambiguity up to 8.4e-10) and three
  COLLIER-sourced lanes: `rn2-int-mixed-n6-r8` lane 1 (finding 3) and `rn2-irml-zero-n3-r5[-mu2]`
  lane 0, where `red.py` has no tadpole basis for the massless on-shell kinematics.
- Tier C consists of `rn2-phys-complex-n6-r8{,-mu2,-muall}` lane 1, `rn2-phys-mixed-n6-r8` lane 0
  and `rn2-phys-real-n6-r8` lane 1 (extrapolated input ambiguity 5e-9 to 1.8e-7).

The synthetic rank n+2 cases have 18 lanes: tier A 6 (n = 1, 3, whose S-matrices are
consistent), B 10 and C 2 (`syn-n6-r8` lanes 0 and 1, ambiguity up to 8e-5).

## Controls (rank <= n+1)

- **Twins.** The 162 phase-1 twins (ranks n-1, n and n+1 on each of the 54 mu-free
  kinematics), reproduced number for number.
- **mu^2 twins.** 18 mu^2 twins at rank n+1, with `mu^2 x rank n-1`, on `int` (real and
  complex) and `shifted`.
- **`controls/ninja`.** Results of original Ninja @05fb0d3 + OneLOop 3.6 for the 162 twins
  (`ninja_bench-o3`, phase-1 run, trimmed to the Laurent coefficients).
- **Synthetic twins.** `controls/reference/syn-n{2..6}-r{n,n+1}.json` give high-precision
  references for existing synthetic benchmark cases, with Ninja's errors from
  `validation/benchmark/ninja`.

| n | rank - n | lanes | A/B/C | source | max unc. | max verified_to | COLI quad err | DD quad err | Ninja err |
|---|---|---|---|---|---|---|---|---|---|
| 1 | -1..1 | 72 | 72/0/0 | mp | 1e-38 | 1e-33 | 1e-33 | 1e-33 | 4e-15 |
| 2 | -1..1 | 108 | 81/27/0 | mp | 5e-17 | 3e-32 | 3e-32 | 2e-18 | 1e-12 |
| 3 | -1..1 | 90 | 87/3/0 | red, coli 3 | 6e-17 | 2e-16 | 1e-14 | 2e-16 | 6e-9 |
| 4 | -1..1 | 90 | 63/27/0 | red | 3e-14 | 6e-12 | 2e-11 | 6e-12 | 2e-6 |
| 5 | -1..1 | 90 | 63/27/0 | red | 1e-12 | 1e-12 | 1e-11 | 3e-13 | 9e-10 |
| 6 | -1..1 | 90 | 60/30/0 | red, coli 3 | 3e-10 | 6e-12 | 3e-11 | 2e-13 | 0.3 (finding 3) |

## Identity and invariance data (`identity/`)

All inputs are small dyadic rationals: integer momenta, dyadic masses and coefficients
`k 2^(-2d)` with |k| < 128. Every derived case is therefore exact in binary64 (the generator
refuses otherwise), and a Float(256) test can demand 1e-60.

| design item | cases | identity (manifest key) |
|---|---|---|
| V2 | `rn2-exact-q-n{n}-r{n+2}` (q only) and `rn2-exact-mu-n{n}-r{n+2}` (every mu^2 power), n = 1..7, 2 lanes each | numerators of full rank for the polynomial identities (Laurent division, on-cut reconstruction, full vs sampling box) |
| V7a | `...-shift` | `V_i -> V_i - P` with `N'(q) = N(q - P)`: equals the base lane (`V7a`; P recorded) |
| V7b | `...-perm` | lane k is base lane 0 with propagator i -> sigma_k(i): all 2n dihedral orders for n <= 4, the identity plus 7 random permutations for n = 5..7 (`V7b`) |
| V6a | `rn2-id-pinch-n{n}` (lane j: `D_j M_j`, v2) and `-sub` (lane j: `M_j` on the other n-1 propagators), n = 2..7 | `I_n[D_j M_j] = I_(n-1)[M_j]` for every j (`V6a`) |
| V6b | `rn2-id-decomp-n{n}` with `N = R + sum_i D_i B_i + sum_(i<j) D_i D_j C_ij` (v2; the double pinches bring mu^4), with pieces `-R` (rank n+1), `-B` (rank n on n-1 propagators) and `-C` (rank n-2 on n-2), n = 2..7, 2 lanes | `I[N] = I[R] + sum I[B_i] + sum I[C_ij]`; every piece runs through the rank <= n+1 engine (`V6b`) |
| V5 | `padding-manifest.json`: 238 rank <= n+1 cases (753 lanes, the n = 7 twins included) with Ninja results | run declared as rank n+2 |

**The integrand mu^2 in V6.** `D_i = (q + V_i)^2 - m_i^2 - mu^2` includes the integrand
mu^2, so the V6 numerators are v2 cases.

**V6b is constructive.** The design's q-only V6b variant, the exact rank reduction of
`identity_check`, is replaced by the constructive decomposition above because:
- it cannot be exact in binary64, since its pieces carry the Gram determinant of four momenta
  in the denominator;
- it leaves rank-6 boxes, which are themselves excess 2.

The reduction itself is run here as `id-*`.

**References.** The 284 lanes of n <= 6 are tier A, with sources `mp` (63) and
`reduction-exact` (221). The n = 7 cases (S5b fix-up: `gen_identity.py` appends them after
every earlier case with their own seeds, so every n <= 6 case and manifest entry is
unchanged; `run_suite.sh identity-n7`, `finalize_n7.sh identity`) add 98 lanes: 95 tier A
(`reduction-exact`) and 3 tier C:
- `rn2-exact-mu-n7-r9` lane 1 and its shifted copy: the review's `red.py` divides by an
  exactly vanishing `2 P1.k` in its tadpole recipe for these integer kinematics, so the
  published value is COLLIER's COLI quad, without a second branch at n = 7;
- `rn2-exact-mu-n7-r9-perm` lane 7 (a `mu^8` numerator): COLLIER reproduces the reduction
  only to 1.4e-8 (60 vs 80 digits: 2e-39).

The V6 and V7 tests compare engine results with each other and need no reference beyond the
lane scale; V6 requires tier A on its left-hand sides, which all n = 7 ones are.

## Conditioning scan (`scan/`, V9, report only)

- **`rn2-scan-bubp2-{real,complex}-n2-r4[-mu2]`.** p = (5a, 0, 0, 3a) with p^2 = m0 2^-k for
  k = 0, 4, ..., 24, then p = 0. That gives 8 lanes, all with exact `mp` references.
  COLI loses up to 2e-13 on this ladder.
- **`rn2-scan-gram-n3-r5`, `rn2-scan-gram-n4-r6`.** Normalised Gram determinants go down to
  9e-11 and 2e-12 (lane order k = 0, 2, ..., 12; values in the `scan` field). COLLIER loses up
  to 1e-4 (COLI, box k = 6) and 5e-6 (DD, box k = 4) here, while the 60/80-digit reduction stays
  within 4e-24. The box k = 12 lane is tier B.

## Synthetic rank n+2 cases

`validation/benchmark/make_synthetic_cases.py` now also writes `syn-n{1..6}-r{n+2}`.
- **Location.** They go into `validation/benchmark/synthetic-rank-n2/`, not `synthetic/`. Every
  tool that loops over `synthetic/*.json` with original Ninja or a rank <= n+1 build is
  therefore unaffected.
- **Existing files.** The 20 existing files are regenerated byte for byte.
- **Inputs.** For n >= 2, the kinematics, masses and S-matrices are those of `syn-n{n}-r{n+1}`
  (same seed and stream), and the tensors come from `random.Random(4000 + n)`. `syn-n1-r3` is a
  complex-top tadpole with V_0 zero, timelike or spacelike.
- **References.** In `reference/syn-*.json`.

## n = 7 (S5b)

The user's practical range is n <= 7, so S5b extends the suite by one multiplicity with the
same generators and pipeline.

- **Kinematics.** `gen_suite.py` appends the specs `int` (real, complex, mixed masses) and
  `shifted` (mixed) at n = 7, after every earlier spec, so every existing case is regenerated
  byte for byte. No `phys` family: its snapped S-matrices tie the uncertainty to the generic
  engine's own twins. `conditioning.cayley_min` now stops at 6-point subsets: in four
  dimensions the modified Cayley matrix of 7 or more points is singular (identical for
  n <= 6). The Gram screening at n = 7 needs up to a few hundred trial points per lane.
- **Cases.** Suite (`cases/`): `rn2-int-{real,complex,mixed}-n7-r9`,
  `rn2-shifted-mixed-n7-r9`, `rn2-int-{real,complex}-n7-r9-mu2`,
  `rn2-shifted-mixed-n7-r9-mu2` (blocks mu^0, mu^2, mu^4) and `rn2-int-real-n7-r9-muall`
  (up to mu^8): 8 cases, 24 lanes. Controls (`controls/cases/`): ranks 6, 7, 8 of the four
  kinematics and three mu^2 twins at rank 8: 15 cases, 45 lanes. Original Ninja's results
  for the 12 mu-free twins: `controls/ninja/` (`harness/ninja_controls.py`).
- **Pipeline.** `run_suite.sh n7-controls` and `run_suite.sh n7` (COLLIER COLI quad and
  `id-coli-quad`; no DD at n = 7), with `GENERIC_RUNNER="ninjars case-check"` for the exact
  scalars and `generic-f256`, then `finalize_n7.sh` (references, `summary/n7*.json`,
  `scalars/n7*.json`; no other file is rewritten).
- **Results.**

  | case | tiers | source | max unc. (A/B) | verified_to |
  |---|---|---|---|---|
  | `rn2-int-complex-n7-r9` | AAA | reduction-exact | 7e-38 | 4e-15..5e-13 |
  | `rn2-int-complex-n7-r9-mu2` | AAA | reduction-exact | 6e-30 | 2e-14..6e-12 |
  | `rn2-int-mixed-n7-r9` | AAA | reduction-exact | 1e-37 | 6e-16..4e-13 |
  | `rn2-int-real-n7-r9` | CCA | coli-quad (lanes 0, 1), reduction-exact | 2e-33 | 8e-11 |
  | `rn2-int-real-n7-r9-mu2` | CCB | coli-quad, reduction-exact | 4e-25 | 3e-11 |
  | `rn2-int-real-n7-r9-muall` | CCB | coli-quad, reduction-exact | 1e-17 | 4e-9 |
  | `rn2-shifted-mixed-n7-r9` | ACA | coli-quad (lane 1), reduction-exact | 2e-38 | 6e-15..3e-14 |
  | `rn2-shifted-mixed-n7-r9-mu2` | ACC | coli-quad, reduction-exact | 2e-30 | 2e-14..5e-7 |

  Suite: 13 A, 2 B, 9 C. Controls: 33 A, 12 C; on the 33 tier-A control lanes the generic
  engine in Float(256) agrees with `reduction-exact` within the lanes' uncertainty
  (`d_generic-f256`): to <= 8.4e-38 on the mu-free lanes and up to 1.2e-33 on the mu^2 twins
  (`rn2-int-real-n7-r8-mu2` lane 2). There the difference equals the uncertainty, the
  difference between the published 60-digit and the 80-digit reduction: the generic engine
  reproduces the 80-digit values. Publishing those (with an uncertainty from a third, higher
  precision) would tighten the n = 7 tier-B tolerances; the references keep the 60-digit
  values for now.
  - 8 of the 9 suite C lanes (and the 12 control C lanes) are finding 5. The ninth,
    `rn2-shifted-mixed-n7-r9-mu2` lane 2, is a correct reduction value (60 vs 80 digits:
    2e-32) that COLLIER reproduces only to 5e-7, so it lacks independent verification.
  - The B lanes of `rn2-int-real-n7-r9-{mu2,muall}` are reductions whose 60- and 80-digit
    runs differ by 4e-25 and 1e-17 (the integrand mu^2 blocks cancel strongly at n = 7, rank 9).
    Their tolerance is 3 times that.

## Checking the engines (S5b)

`ninjars case-check` reduces bench cases and judges every lane against a reference file:

```sh
ninjars case-check validation/rank-n2/cases/*.json --engines fast,generic \
  --precision f64,float:256 --reference-dir validation/rank-n2/reference \
  --known-failures validation/rank-n2/known-failures.json \
  --twins validation/rank-n2/summary/controls.json --twins validation/rank-n2/summary/synthetic-twins.json \
  --twins validation/rank-n2/summary/n7-controls.json \
  [--declare-rank n+2] [-o report.json]
```

- Engines: the fast engine (`TensorNumerator`, `MuTensorNumerator` for v2 cases, native
  masters, lanes batched as `case-bench` does) and the generic engine at `f64` (native
  masters), `double` or `float:<bits>` (oneloopmaster's exact expressions; B1111 from its
  native tensor family). The generic
  engine expands the dense tensor numerically, so a dense rank-9 numerator costs no
  Symbolica preparation: all 205 control cases (615 lanes, n = 1..7) take 6 minutes in Float(256).
- Verdicts per lane: `pass`, `fail`, `report-only`, `known-failure`, `stale-known-failure`
  (a listed lane that agrees: remove the entry), `unchecked` (no reference lane), `error`
  (the engine rejects the case, e.g. rank n+3). The exit status is non-zero if any lane
  fails or is a stale known failure. The report gives per lane the tolerance, the Ninja
  error estimate behind a binary64 budget, the new-term parts and, for known failures, the
  difference to COLLIER's value.
- On rank <= n+1 cases the fast run carries `case_bench_reference_check`, which is
  `case-bench`'s own reference check of the same batch (tested to be equal).
- `--bits B --out-dir DIR` writes `DIR/<case>.json` with 45-digit `lanes[].laurent_decimal`:
  the `GENERIC_RUNNER` of `run_suite.sh` (used for the n = 7 scalars).
- Original-Ninja references (`validation/benchmark/ninja`) are judged at a scaled 1e-8
  (`--ninja-tolerance`). Original Ninja's own binary64 error exceeds that on some synthetic
  cases: case-check against them fails 5 lanes of 4 synthetic cases (e.g. `syn-n6-r6`
  lane 0 at 0.26, where Ninja is the inaccurate one). The CLI test that compares with
  `case-bench` therefore runs with `--no-fail`; use the high-precision references to judge
  NinjaRS.

Tests (`crates/ninjars-symbolica/tests/`, `crates/ninjars-cli/tests/case_check.rs`):

| test | checks | binary64 noise / tolerance |
|---|---|---|
| `rank_n2_reference.rs` | V4 (Float(256)), V4f, V9, the policy's verdict rules, fast vs generic f64 | the policy above |
| `rank_n2_identity.rs` | V6, n = 2..7 | `max(1e-10 max(1, cond), 10 min(e_generic, 1e-6))`, `e_generic` the same identity with the generic engine in f64 |
| `rank_n2_invariance.rs` | V7a, V7b, V7c, n = 1..7, at rank n+2 and on the rank n+1 truncations | `max(1e-11, 10 noise)`, `noise = min(max(generic noise, min(fast noise, 1e-9)), 1e-6)` |
| `rank_n2_padding.rs` | V5 | `max(1e-11, 10 fast noise)` |
| `fast_parity.rs` | V8, n = 1..7 | 1e-9, or `10 noise`, `noise = min(max(generic noise, min(fast noise, 1e-8)), 1e-6)` |

- **Noise.** An engine's noise is the change of its binary64 result when the propagators
  are listed in reverse order (an identical integral). V7 and V8 take it from the generic
  engine, an independent implementation whose algorithm the Float(256) identities validate.
  The fast engine's own noise counts only up to a cap (1e-9 in V7, 1e-8 in V8), and any noise
  only up to 1e-6. Before, V7 took the noise from the fast engine alone (V8 from the larger
  of both engines), so an order-dependent fast-engine error widened its own tolerance.
  Demonstration on a throwaway copy of this branch, with `c[1] *= 1 + 1e-7` injected into
  the fast engine's triangle coefficients (rank <= n+1 and rank n+2 paths):
  - V7b relabelling differences grow to 1.5e-3 (rank n+1) and 5.3e-3 (rank n+2). The old
    rule passes every V7b pair; the new rule fails 55 of 84 pairs at rank n+1, and 41 pairs
    (plus 14 reported, n = 7) at rank n+2.
  - V8 fails under both rules, because the injection leaves the fast engine's noise small
    on some points. With the new rule: `parity_n3..n5` 38 of 90, 59 of 108 and 62 of 126
    checks, `parity_rank_n2_n3..n5` 14 of 18, 14 of 18 and 18 of 18.
  - V5 fails too (`edge-low-rank-n5-r3` lane 1 against original Ninja: 1.3e-7, noise
    1.0e-8). It keeps the fast engine's own noise, as it compares the fast engine with
    original Ninja, the same algorithm.
  - Shifts (V7a), which keep the propagator order, are unaffected.
- **Fast engine accuracy (rank n+2 basis conditioning, 79f306a).** With the bubble
  references and tadpole bases chosen per lane (`crates/ninjars/src/fast/bases_x2.rs`), V4f
  (`suite_fast_binary64_matches_references`) and V8 at n = 6 rank 8 (`parity_rank_n2_n6`)
  pass and are no longer ignored. **Still ignored** (`needs fast accuracy fix`):
  - `suite_fast_within_ten_times_generic_binary64`: on every suite lane with a reference
    (known failures excepted), the fast engine's error must be at most `max(10 e_generic,
    tier tolerance)`, the tier tolerance being the lane's V4f binary64 tolerance for the fast
    engine. 1 of 324 lanes fails (before the conditioning: 12):
    `rn2-shifted-mixed-n7-r9-mu2` lane 0 (condition 7.9e9, report-only in V4f), fast 1.7e-3
    against generic f64 1.6e-4, 1.1 times the limit. The conditioning does not change this
    lane, and its rank n+1 twin `rn2-shifted-mixed-n7-r8-mu2` lane 0, on the unchanged
    rank <= n+1 path, shows the same ratio (fast 3.2e-9, generic f64 3.1e-10);
  - `rank_n2_integrals_equal_their_pieces_binary64` (V6 at rank n+2, 5 of 39 identities):
    `rn2-id-pinch-n7` lanes 0, 2, 3, 6 at 5.8e-9..1.6e-8 (generic f64 <= 1.4e-10), which is
    the error of their pieces below, and `rn2-id-pinch-n4` lane 3 at 5.3e-9 against a
    tolerance of 3.7e-9;
  - `pieces_reproduce_the_rank_n2_references_binary64` (the V6 pieces at n = 7 through the
    fast engine's rank <= n+1 path: `rn2-id-pinch-n7` lanes 0, 2, 3, 6 at 5.8e-9..1.6e-8
    against generic f64 <= 1.5e-12);
  - `relabelling_invariance_rank_n2_binary64` (V7b at n = 7: `rn2-exact-mu-n7-r9-perm` lane 1,
    fast 8.0e-7 against generic noise 5.0e-8).

  The n = 7 identity kinematics have propagator pairs whose shift difference has a zero time
  component (`rn2-id-pinch-n7`: pairs (0,3), (1,5), (2,6); none at n <= 6): the bubbles where
  the fast engine is known to lose accuracy. They are a regression target for that work.
- **Order-dependent master integrals (finding 6).** `fast_parity.rs` prints `FAST_PARITY_UNSTABLE`
  for points whose result changes by more than 1e-6 under reversed propagators in either
  engine; both engines agree there, as they share the master integrals, and V8 passes (such
  points are still compared, within 1e-5, as the noise counts only up to 1e-6). At
  n = 6 rank 8 (lane 3) this is oneloopmaster's D0 of finding 6; n = 7 rank 5 (lane 3, 1.0
  relative) shows the same pattern but was not traced to a master.
- **No run-to-run variation.** The S5b verifier saw the generic engine's f64 result on a
  lane depend on the cases evaluated before it in the same process (up to 2.5e-9 relative),
  through the symbol-registration history that orders Symbolica's evaluators. Main's
  positional evaluators (b11767a, merged in dec3d55) removed it: `rn2-phys-real-n6-r7`
  evaluated alone, after `rn2-int-complex-n1-r2`, and after 40 other control cases gives
  bitwise identical results.

Results at 115fb46 (oneloopmaster 8ec256fc, Symbolica 9cdc2949) with this `known-failures.json`,
as pass / fail / report-only / known-failure; no lane is `stale-known-failure`, `unchecked` or
`error`, every lane's status is `Success` (so the tracked-precision check of the `Float` masters
flags none), and every run exits 0.

| run | controls (rank <= n+1, 615 lanes) | suite (rank n+2, 333 lanes) | largest checked difference (controls; suite) |
|---|---|---|---|
| fast, native masters | 600 / 0 / 15 / 0 | 276 / 0 / 57 / 0 | 2.6e-6 (`rn2-phys-real-n6-r7` lane 1, tier B, original Ninja 3.6e-6); 2.6e-8 (`rn2-int-real-n7-r9` lane 2) |
| fast, `--conditioned-cuts` | 600 / 0 / 15 / 0 | 276 / 0 / 57 / 0 | 2.4e-7 (`rn2-phys-complex-n6-r7` lane 1, tier B); 2.6e-8 (the suite's rank n+2 engine is the same) |
| generic f64, native masters | 600 / 0 / 15 / 0 | 275 / 0 / 58 / 0 | 1.6e-7 (`rn2-phys-complex-n6-r7` lane 1, tier B); 3.3e-8 (`rn2-int-real-n7-r9` lane 2) |
| generic DoubleFloat, exact masters | 600 / 0 / 15 / 0 | 297 / 0 / 36 / 0 | 1.4e-10 (`syn-n6-r7` lane 1, tier B; tier A <= 9.8e-26 at n <= 6, 2.3e-25 at n = 7); 7.5e-12 (`rn2-phys-real-n6-r8` lane 0, tier B; tier A <= 9.6e-25 at n <= 6, 1.0e-24 at n = 7) |
| generic Float(128), exact masters | 599 / 0 / 15 / 1 | 297 / 0 / 36 / 0 | 1.4e-10 (tier B; tier A <= 1.3e-30 at n <= 6, 1.0e-30 at n = 7); 7.5e-12 (tier B; tier A <= 5.3e-29 at n <= 6, 7.3e-30 at n = 7) |
| generic Float(256), exact masters | 601 / 0 / 14 / 0 | 285 / 0 / 48 / 0 | 1.4e-10 (tier B; tier A <= 1.7e-38 at n <= 6, up to 1.2e-33 = the uncertainty at n = 7); 7.1e-12 (`syn-n6-r8` lane 2, tier B; tier A <= 1.3e-36 at n <= 6, 5.9e-30 at n = 7) |

- The known failure is the known inaccuracy of `rn2-int-mixed-n6-r6` lane 2 in Float(128)
  ("Known inaccuracies" above). Its rank twins on the same kinematics pass but moved with the
  same D0: `rn2-int-mixed-n6-r{5,7}` lane 2 4.8e-33 -> 8.5e-33 and 2.9e-31 -> 1.3e-30 (0.25 and
  0.27 of their tolerances), and in the suite `rn2-int-mixed-n6-r8` lane 2 7.1e-30 -> 5.3e-29
  (0.42), which sets the Float(128) tier-A maximum at n <= 6.
- Against the generic engine's selection at f9a8b16 (oneloopmaster 09b2c74, Symbolica e679e8c1)
  and integration pass 1's fast runs (f33ff8d), lane by lane: no verdict changes in the
  fast, conditioned, generic f64, DoubleFloat and Float(256) runs. Values move at rounding level
  with the masters:
  - fast: 121 suite and 221 control lanes, scaled differences 0.95 times before in geometric
    mean, none more than 3.6 times;
  - generic f64: 127 and 231 lanes, 0.94 and 0.92 times;
  - DoubleFloat: 249 and 461 lanes, 0.98 and 0.96 times;
  - Float(256): 24 and 190 lanes, by at most 1.3e-71 and 5.6e-61 of the lane (the latter on
    `syn-n6-r6` lane 0, which cancels by 1e12);
  - Float(128): 295 and 535 lanes, 0.84 and 0.59 times; the 14 known inaccuracies went stale
    (above) and `rn2-int-mixed-n6-r6` lane 2 failed.
- Compared with integration head e116813 (oneloopmaster 09b2c74, the previous version of this
  table): the generic f64 known failure of `rn2-phys-real-n6-r7` lane 2 and the Float(128)
  one of `rn2-irml-zero-n5-r6` lane 1 went with 5ed3b22, the other 14 Float(128) ones with
  Symbolica 9cdc2949. The largest checked differences moved with the fast engine's conditioned
  rank n+2 box geometry (a0e7b50: suite 5.3e-8 -> 2.6e-8) and the generic engine's selection
  (5ed3b22: generic f64 suite 5.2e-7 -> 3.3e-8, controls 3.5e-7 -> 1.6e-7; DoubleFloat and
  Float(128) suite 1.4e-11 -> 7.5e-12; the high-precision controls 3.2e-10 -> 1.4e-10).
- Against 8346699 (oneloopmaster fe550e3): the 15 control and 9 suite known failures of
  findings 3 and 5 are gone. `rn2-int-mixed-n6-r{5..8}` lane 1 (tier B) now pass, in binary64
  with the corrected Ninja budget of `ninja_unreliable`; the n = 7 lanes have tier-C COLLIER
  references and are report-only. DoubleFloat no longer reports the stale known failure of
  `rn2-int-mixed-n6-r8` lane 1 (there the merge run's D0 already agreed with COLLIER, while
  `Float` and native binary64 did not: the bug depended on the carrier).
- Not part of these runs: the identity cases (`identity/`) have no binary64 budget (no Ninja
  results; the V6 and V7 tests compare engines instead) and pass in Float(256) (379 pass,
  3 report-only); the conditioning scan (`scan/`, V9) is report only by design.

## Regenerating

```sh
source <devenv>                      # gfortran 15 (dev shell), SYMBOLICA_LICENSE
H=validation/rank-n2/harness
$H/build.sh                          # COLLIER 1.2.9 -> target/rank-n2-collier (quad, precise, stock)
$H/python.sh $H/gen_suite.py --out validation/rank-n2/cases
$H/python.sh $H/gen_suite.py --out validation/rank-n2/controls/cases --controls
$H/python.sh $H/gen_identity.py --out validation/rank-n2/identity
$H/python.sh $H/gen_scan.py --out validation/rank-n2/scan
$H/python.sh $H/padding_manifest.py --out validation/rank-n2/identity/padding-manifest.json
python3 validation/benchmark/make_synthetic_cases.py
for s in controls synthetic-twins suite synthetic identity scan; do
  GENERIC_RUNNER="ninjars case-check" $H/run_suite.sh $s 32
done
$H/finalize.sh                       # references, summaries, scalars, hpref comparison, identity residuals
# n = 7 (S5b)
$H/python.sh $H/ninja_controls.py --driver <ninja-runs/bin/ninja_bench-o3> \
  validation/rank-n2/controls/cases/*-n7-r*.json --out validation/rank-n2/controls/ninja
for s in n7-controls n7 identity-n7; do GENERIC_RUNNER="ninjars case-check" $H/run_suite.sh $s 48; done
$H/finalize_n7.sh                    # suite and identity parts (needs $WORK/identity from the identity set)
$H/python.sh $H/test_mu2.py --build quad; $H/python.sh $H/test_mu2.py --build mp
```

- **COLLIER 1.2.9.**
  - `build.sh` copies the source tree. The default is
    `/common/dev/MG5_aMC_v3_7_0/HEPTools/collier/COLLIER-1.2.9`; set `COLLIER_SRC` to
    override. The upstream tarball `collier-1.2.9.tar.gz` from collier.hepforge.org has the
    same `src/`.
  - It edits only the Release flags in `CMakeLists.txt`, then builds static libraries with
    cmake (from `nix shell nixpkgs#cmake` when cmake is not on the PATH).
  - Variants: `quad` (no `-ffast-math`, plus `-freal-8-real-16`), `precise` (no `-ffast-math`)
    and `stock` (the shipped flags).
  - The driver `collier_tensor_driver.f90` prints 35 digits in the quad build.
- **Python.** `harness/python.sh` runs python3 with mpmath and gmpy2 from nixpkgs. Any
  python3 >= 3.10 with mpmath also works; gmpy2 only speeds up the rationals.
- **Scalar masters and the generic engine.**
  - `reduction-exact` needs the exact C0/D0 of every 3- and 4-subset; `scalar_cases.py` writes
    them as rank-0 bench cases.
  - They were evaluated with the NinjaRS generic engine at Float(256) with `SymbolicaMasters`
    (main b5b93be, oneloopmaster 264a6ce). A rank-0 numerator makes the result the scalar
    integral itself.
  - That driver was a scratch program, because S5a added no Rust. `GENERIC_RUNNER` can
    be any command `RUNNER --bits 256 --out-dir DIR CASE.json...` that writes `DIR/<case>.json`
    with `lanes[].laurent_decimal`; `ninjars case-check` is one (it produced the n = 7
    scalars and `generic-f256` runs).
  - The values are committed in `scalars/`, so the following reruns the reduction without the
    runner:
    `E2E_SCALARS=validation/rank-n2/scalars harness/review_reduction/run_reduction.py CASE.json... --out-dir DIR`.
    Use `scalars/shift-invariants` for `reduction-exact-shifts`.
  - The same runner provides `generic-f256` on the controls. Its first lane per case at
    n = 6, rank 7 takes about 10 minutes.
- **Run times.** On 32 cores, each set takes about 10 minutes for the COLLIER runs. The
  80-digit reduction of the suite takes about 40 minutes.

## Provenance

- `harness/collier_reference.py` descends from the phase-1 harness
  (`scratchpad/rank-n2/reference/harness/collier_reference.py`, sha256 f11bd42113a93da0...).
  The review's 60-digit preprocessing (`review/collier_hp.py`, 35de49664bcd293c...) is
  replaced by exact rationals. Added: exact driver inputs, v2 cases and the `mp` backend.
- `harness/gen_suite.py` is the phase-1 generator (4a3eda4057276292...) with v2 output and the
  new families; `conditioning.py` is unchanged.
- `harness/review_reduction/red.py` is the review's `red.py` verbatim (a03275f3659320db...).
  `e2e_reduction.py` is the review's `e2e.py` (3abfebd35eca0aef...) with the patches listed in
  its docstring.
- `provenance/review-hpref/` holds the review's COLLIER values with 60-digit preprocessing,
  unchanged.
