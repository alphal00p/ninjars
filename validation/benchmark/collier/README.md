# COLLIER at numerator rank n+2: accuracy reference

Original Ninja stops at numerator rank n+1, so the rank n+2 part of the benchmark report
([`docs/benchmark/benchmark.typ`](../../../docs/benchmark/benchmark.typ), section "Validation
at numerator rank n + 2", data in `docs/benchmark/data/rank_n2.json`) validates NinjaRS against
high-precision references and the tensor-integral library COLLIER 1.2.10 (Denner, Dittmaier,
Hofer, Comput. Phys. Commun. 212 (2017) 220, arXiv:1604.06792; COLI and DD branches: Denner,
Dittmaier, Nucl. Phys. B 658 (2003) 175, B 734 (2006) 62, B 844 (2011) 199).

COLLIER computes tensor integrals and contracts them with the numerator's tensor coefficients.
That is a different method from integrand reduction, so the comparison is an independent check,
not like for like (see "What is compared").

> **Timings not maintained (2026-10-21).** Since the report revision of 2026-10-21 its rank n+2
> section ("Validation at numerator rank n + 2") validates NinjaRS against the high-precision
> references and COLLIER and makes no timing comparison. The timing files in `results/`
> (`collier.json` rounds, `ninjars-fast.json`, `ninjars-generic.json`, `instructions.json`,
> `environment.json`) are those of the run at NinjaRS 0d0d7e7 (2026-10-19, CPU 260) and are kept
> for reference only; they are not maintained, and the timing sections below describe that run.
> `run_bench.py` and `count_instructions.py` are kept, but the report no longer reads their
> output (`collect.py` still copies it into `rank_n2.json`).

## Pipeline

```sh
source <devenv>                                     # gfortran, python3; cmake via nix if absent
validation/benchmark/collier/build.sh               # COLLIER 1.2.10 (precise, stock) + collier_bench
cargo build --profile perf --locked -p ninjars-cli --features numerator-jit   # target/perf/ninjars
python3 validation/benchmark/collier/run_bench.py --rounds 2     # timings (pinned, interleaved per round), ~1.5 h; not maintained
python3 validation/benchmark/collier/run_accuracy.py            # case-check verdicts, ~10 min
python3 validation/benchmark/collier/run_final_review.py        # fresh cases of the final review, ~4 min
python3 validation/benchmark/collier/count_instructions.py      # perf stat instruction counts; not maintained
python3 validation/benchmark/collier/collect.py                 # -> docs/benchmark/data/rank_n2.json
cd docs/benchmark && nix shell nixpkgs#typst -c typst compile benchmark.typ benchmark.pdf
```

An accuracy-only refresh (validating NinjaRS against the references and COLLIER, with no
timing) runs `build.sh` if the binaries are missing, the cargo build, `run_accuracy.py`,
`run_final_review.py` and `collect.py`; `collect.py` then keeps the committed timing files
and marks them with `timings.from_previous_run` (see "Provenance of `results/`").

| file | role |
|---|---|
| `build.sh` | downloads `collier-1.2.10.tar.gz` from hepforge into `target/benchmark/collier-1.2.10/download`, checks its sha256 (`460624ea…62713`), unpacks it unmodified and builds two static libraries with cmake: `precise` (the shipped gfortran Release flags without `-ffast-math`: `-O2 -funroll-loops`) and `stock` (the shipped flags, `-O2 -funroll-loops -ffast-math`); links `collier_bench` and `collier_cache_repro` against each with the same flags |
| `prepare.py` | decomposes every lane into COLLIER pieces with `validation/rank-n2/harness/collier_reference.py` (`lane_sink`, `sink_to_jobs`: exact removal of the integrand mu^2 by pinching, shift of each piece to p_0 = 0, the lane's S-matrix as invariants), then rounds masses, momenta, invariants and coefficients to binary64 once |
| `collier_bench.f90` | the binary64 driver: validation pass (three Delta settings, Laurent coefficients, flags, error estimates), then timed quantities (below) |
| `collier_cache_repro.f90` | standalone reproducer of the COLLIER global-cache defect (see Findings); `build.sh` builds it as `bin/collier_cache_repro-<variant>` |
| `run_bench.py` | runs NinjaRS (`case-bench`, fast engine) and COLLIER, interleaved per round on one CPU; the generic engine once at the end; precise COLI with the global cache for validation only (untimed); `--lanes-only` re-runs only COLLIER's validation pass and replaces the lane values of `results/collier.json` |
| `run_accuracy.py` | `ninjars case-check` on the suite and synthetic cases: fast and generic f64, generic DoubleFloat, Float(128), Float(256); per lane also the scaled difference to COLLIER's values (`collier`: COLI and DD 1.2.10 binary64 from `results/collier.json`, COLI quad 1.2.9 from the references) |
| `run_final_review.py` | NinjaRS (fast, generic f64, the generic engine's `--stability-test` status, and the generic engine in DoubleFloat, Float(128) and Float(256) with its lane status) and COLLIER (COLI, DD) on the final review's 58 fresh cases with 60-100 digit references (`/shared/external/ninja/references/final-review`, outside the repository); per lane also NinjaRS's binary64 values against COLLIER's (`fast_vs_coli`, ...) |
| `count_instructions.py` | retired user-space instructions per point (`perf stat`, difference of two fixed-work runs) |
| `collect.py` | the data file of the report, including `accuracy.ninjars_vs_collier` (see "NinjaRS against COLLIER") |
| `results/` | the raw, consolidated measurements (`collier.json`, `ninjars-fast.json`, `ninjars-generic.json`, `accuracy.json`, `accuracy-final-review.json`, `instructions.json`, `environment.json`) |

## What is compared

- **Inputs.** NinjaRS (`ninjars case-bench`) reads the bench case: shifts V_i, complex masses,
  S-matrix, and the dense tensor `N(q, mu^2) = sum_k (mu^2)^k T_k(q)`. COLLIER needs a loop
  momentum with p_0 = 0 and no integrand mu^2. `prepare.py` removes mu^2 by pinching (up to
  five pieces with fewer propagators per lane) and shifts each piece, in exact arithmetic,
  before any timing. On cases without mu^2 and with V_0 = 0 (56 of the 111) the COLLIER input
  is the bench case's tensor itself, so no preprocessing is left out of COLLIER's time. On the
  others COLLIER's time excludes it.
- **COLLIER's time.** `TNten_cll` (`Aten_cll` for tadpoles) for every piece of the lane plus
  the contraction of its tensor components `TNten(n0,n1,n2,n3)` with the coefficients (plain
  binary64 numerics in the driver). Quantities (ns per point, median of the blocks):
  - `total_finite`: one evaluation, the finite part (eps^0) with the Ninja/OneLOop normalisation
    (DeltaUV = DeltaIR1 = 0, DeltaIR2 = pi^2/6, mu_UV^2 = mu_IR^2 = the lane's mu_R^2);
  - `total_laurent`: the three evaluations that give eps^0, eps^-1, eps^-2 (Delta settings of
    `validation/rank-n2/harness/collier_tensor_driver.f90`), including the parameter changes
    between them; NinjaRS returns all three coefficients (and the cut-constructible/rational
    split) in one evaluation;
  - `calls_finite`, `contraction`: the two parts of `total_finite`;
  - `delta_switch`, `scale_switch`: `SetDeltaUV_cll`/`SetDeltaIR_cll` for the three settings,
    and `SetMuUV2_cll` + `SetMuIR2_cll`. Each call of these runs DD's `DDsetparam`, which
    determines the machine precision in a loop of `exp(log(1 + x))` (about 1 µs in the precise
    build). The timed quantities therefore set the scale outside the timed region, once per
    lane and block, as an application with a fixed scale does.
- **COLLIER's configuration.** Default target accuracy 1e-8 (`SetReqAcc_cll`; the library is
  tuned for it), COLI (`SetMode_cll(1)`) and DD (`2`) where every piece is within DD's limits
  (N <= 6, pentagons up to rank 5, hexagons up to rank 6: n <= 4 at rank n+2), no global cache
  (validated, not timed, with one cache per lane: `precise-coli-cache`; its 7-point rank-9
  results are wrong, see Findings), `Init_cll(Nmax, Rmax)` with the case's largest piece.
- **NinjaRS's time.** `ninjars case-bench` (widths 1 and 32): fast engine total with the
  tensor-numerator expansions, the pure reduction (replayed numerator and masters) and the
  native masters (replayed calls), exactly as in the rank <= n+1 benchmark. The generic engine
  in binary64 (`case-bench --engines generic`) uses `case-check`'s numerically expanded
  dense-tensor numerator, which is not the production generic numerator; its reduction-only
  time (replayed numerator and masters) is the comparable figure.

## Timing method

- Every process is single-threaded and pinned with `taskset -c CPU` to one CPU that was idle
  (`/proc/stat` over 5 s; the host has no SMT siblings). The codes are interleaved per round,
  not per case: a round runs NinjaRS on every case (about 6 minutes), then every timed COLLIER
  variant on every case (about half an hour); `--rounds` rounds; `collect.py` takes the smallest
  of the rounds' block medians. Blocks last at least 20 ms (NinjaRS: 9 blocks; COLLIER: 5;
  generic: 5). `environment.json` records, per run, the time other processes spent on the CPU
  (`foreign_s`).
- The core clock varies between 1.8 and 3.1 GHz on the loaded host, and it can differ between
  the NinjaRS and the COLLIER part of a round. A per-case wall-clock ratio NinjaRS / COLLIER
  is therefore uncertain by about +-20 % (the verification re-timed three cases interleaved
  per case and found the ratios within 3 %, but the `cycles:u` counts imply ratios up to 30 %
  away from the wall-clock ones on single cases). At n <= 3, where the per-n geometric means
  are x0.52 to x1.24 and single cases x0.22 to x1.81, this matters; the orders of magnitude at
  n >= 4 do not depend on it.
  `count_instructions.py` therefore also counts retired instructions per point, which do not
  depend on the clock.

## Provenance of `results/`

The timings (`collier.json` rounds, `ninjars-*.json`, `instructions.json`) are those of the
run recorded in `environment.json` (CPU 260, 2026-10-19, NinjaRS main 0d0d7e7). The COLLIER
lane values of `collier.json` were refreshed afterwards with `run_bench.py --lanes-only` at the
commit recorded under `environment.json` `collier_lanes`, after two driver changes:

- the lane error estimate `err_estimate` used TNerr of the last call (Delta set 3) instead of
  Delta set 1; it now keeps the set-1 copy. This changed the estimate on the 24 COLI and 12
  DD lanes of the IR-divergent `rn2-irml-zero-*` cases (both builds) by factors 0.79 to 1.24;
  the Laurent
  coefficients, flags and `absum` of every uncached lane are bitwise unchanged, and so are
  the counts of the report's table of COLLIER's accuracy information;
- the global-cache variant is now validated in the cache mode the timed loop runs in (15
  untimed warm-up events per lane cache) and is no longer timed. The earlier run timed it at
  x0.53 of the uncached time at n = 7 (x0.92 geometric mean over all cases); that time is
  not a usable speed-up because the cached 7-point results are wrong (Findings), and it is
  no longer recorded.

The accuracy files were refreshed on 2026-10-21 at NinjaRS aa6a630 (oneloopmaster 8ec256fc,
Symbolica 9cdc2949; `accuracy.json` `ninjars_head`, `accuracy-final-review.json`), without
re-timing: `run_accuracy.py`, `run_final_review.py` and `collect.py` only. `environment.json`,
`collier.json`, `ninjars-*.json` and `instructions.json` are those described above, and
`rank_n2.json` records this as `timings.from_previous_run` (timed tree 0d0d7e7, oneloopmaster
09b2c74) next to `accuracy_head`. Before the refresh, COLLIER's uncached validation pass was
re-run with a fresh `build.sh` (same tarball and flags) on every case: all 1122 lanes of
`collier.json` (precise and stock, COLI and DD) were bitwise identical, so COLLIER's lane
values were kept. Two `case-check` runs of the refresh gave bitwise identical lanes in all
five modes.

## NinjaRS against COLLIER

`collect.py` compares NinjaRS's lane values with COLLIER's directly, per `case-check` run
(`accuracy.ninjars_vs_collier`), against three COLLIER evaluations: COLI and DD in binary64
(1.2.10, precise build, no global cache) and COLI in quad (1.2.9, `SetReqAcc_cll(1e-20)`, the
references' `collier_coli_quad_decimal`).

- **Agreement.** A lane agrees when |x_NinjaRS - x_COLLIER| / scale is at most the sum of the
  two codes' own claims: `case-check`'s tolerance of the lane (`docs/rank-n2.md` section 8.3)
  and COLLIER's, max(1e-8 * max(1, absum / scale), estimate / scale) in binary64 (1e-8 is
  COLLIER's default target, absum the sum of the contraction terms' magnitudes, estimate its
  error estimate) and 1e-20 * max(1, condition) in quad. COLLIER's binary64 claim exceeds 1e-6
  on 141 of 333 COLI and 50 of 195 DD lanes (mostly `mu2`/`muall` lanes, whose pinched pieces
  cancel), so the counts at fixed thresholds (`within`) and the binary64 comparison set of the
  accuracy table (`comparison_set`) are given too.
- **Disagreements** are decided by the reference: NinjaRS is right when it is within its
  tolerance, COLLIER when it is within its claim plus the reference's uncertainty.
  `disagreements` lists in full every one where NinjaRS is not right, `collier_misses` the
  others. Lanes without a `case-check` tolerance (8 tier-C n = 7 lanes whose reference is
  COLLIER's quad value) are listed under `untolerated`.
- **Fresh cases** (`accuracy.final_review.ninjars_vs_collier`): the same with NinjaRS's
  tolerance set to the report's 1e-6, decided by the 60-100 digit references.

Results at aa6a630 (suite and synthetic, 333 lanes; the previous run's raw `case-check`
reports at 4dec120 give the old values in brackets):
- **Binary64 COLI**: 323 of 333 lanes agree in every run (fast f64 [323], generic f64 [320],
  DoubleFloat, Float(128), Float(256) [323 each]); the other 10 are 8 untolerated lanes and 2
  disagreements. `rn2-irml-zero-n5-r7-mu2` lane 1: COLI is off by 0.23 with accuracy flag 0
  (its estimate 6.0e-2), NinjaRS by 5.2e-10 (fast) and 5.0e-10 (generic). `rn2-phys-real-n6-r8`
  lane 2: COLI is off by 5.0e-4 with accuracy flag 0 (claim 7.7e-8); in binary64 NinjaRS is
  off too, by 1.1e-5 (fast) and 1.3e-6 (generic [1.1e-3]) against a budget of 5.9e-7 (a
  report-only lane, now flagged by `--stability-test`), and right from DoubleFloat on (3.4e-10).
  On the comparison set (298 lanes) fast and COLI differ by at most 2.0e-6
  (`rn2-phys-complex-n6-r8-mu2` lane 2) and by at most 1e-8 on 269.
- **Binary64 DD**: all 195 lanes agree in every run, by at most 9.7e-9.
- **Quad COLI**: fast f64 323 [320], generic f64 321 [319], DoubleFloat and Float(128) 209
  [209], Float(256) 205 [203]. In binary64 NinjaRS is the one off on 3 lanes, all report-only:
  `rn2-phys-real-n6-r8` lane 2 (above), and with the generic engine `rn2-int-real-n7-r9-mu2`
  lane 2 (2.3e-4 against 7.8e-5) and `rn2-shifted-mixed-n7-r9-mu2` lane 0 (1.6e-4 against
  7.9e-5); the fast engine is within budget on both (2.8e-5 [8.7e-4], 1.3e-5 [1.7e-3]). At high
  precision the 116 to 119 disagreements are COLLIER quad missing its 1e-20 target (finding 1
  of `validation/rank-n2/README.md`; NinjaRS within 1e-12 of it on 299 lanes), except
  `rn2-phys-real-n6-r8` lane 0 in Float(256): 7.5e-12 [1.35e-11] from the reference against a
  tolerance of 2.8e-12, report-only, while COLLIER quad is within 1.3e-13. DoubleFloat,
  Float(128) and Float(512) agree with that Float(256) value to 3e-25, so this is the input
  ambiguity of the snapped S-matrix (`docs/rank-n2.md` section 8.3), not a precision loss.
- **Untolerated**: on `rn2-int-real-n7-r9-muall` lanes 0 and 1 NinjaRS's DoubleFloat,
  Float(128) and Float(256) values differ from COLLIER quad (their reference) by 0.86 and 0.14
  and agree with Float(512) to 1.2e-14, 2.6e-21 and 4e-74 (lane 1: 4.2e-16, 2.3e-22, 5e-75);
  binary64 COLI is off by 2.4e4 and 2.3e3. COLLIER is wrong there, as
  `validation/rank-n2/README.md` ("Known failures") found. On the other 6 untolerated lanes
  NinjaRS at high precision is within 3.6e-7 of COLLIER quad.
- **Statuses**: every lane is `Success` in all five runs: the tracked-precision check of the
  `Float` masters (2a7df0d) fails no lane in Float(128) or Float(256), here and on the 166
  fresh lanes.

## Validation of the driver

Against the high-precision references of `validation/rank-n2/reference`, the binary64 COLI
results of the precise build reproduce the 1.2.9 `coli-double` diagnostics of the references
(precise build, exact decimal inputs, 90-digit contraction): `collect.py` writes the comparison
to `accuracy.collier_vs_1.2.9` of the data file. COLLIER 1.2.10 changes, against 1.2.9, the
imaginary parts of complex-mass C0 roots in `coli_c0.F` and two error-estimate guards for
rank 0 (`reductionC.F90`).

## Findings

- **COLLIER's global cache returns wrong 7-point rank-9 tensor integrals.** With the global
  cache on (`InitCacheSystem_cll`, `InitEvent_cll` before every phase-space point, the calls
  in the same order), every 7-point rank-9 lane of the suite is wrong from the third event of
  its cache on, in COLLIER's optimisation and fully optimised modes alike:
  - `run_bench.py`'s validation (one cache per lane, the three Delta settings as separate
    events, values after 15 warm-up events): all 24 n = 7 lanes with a reference differ from
    the uncached values; on the 9 that the accuracy tables compare, the scaled errors are
    0.068 to 77 against at most 3.9e-8 without the cache, and COLLIER reports accuracy flag
    -2 on 8 of them (22 of the 24). The committed data before this refresh (three events per
    cache) already differed on 77 of 354 lanes.
  - The manual's usage alone reproduces it: `collier_cache_repro.f90` uses one cache, one
    `TNten_cll(N = 7, R = 9)` call per event, the same kinematic point in every event (the
    first lane of `rn2-int-mixed-n7-r9`) and no parameter change. Events 1 and 2 return the
    uncached tensor; from event 3 on (checked to event 20) max |T_cache - T| / max |T| = 3.9
    with accuracy flag -2, in the precise and the stock build. Caching only N <= 4 integrals
    (`InitCacheSystem_cll(1, 4)`) leaves a 8.0e-6 difference with flag 0; caching only
    two-point functions, or rank 8 instead of 9, gives bitwise identical tensors. A scan of
    every case during the investigation (a variant of the reproducer reading the prepared
    inputs: one cache, the same lane in every event) gives the same picture: all 24 lanes at N = 7, R = 9 wrong by 0.1 to 46 of the largest component,
    the three N = 7, R = 8 twin lanes bitwise identical, N <= 6 within 1.4e-7.
  - So it is a COLLIER defect, not the driver's usage. At n <= 6 the cache changes 53 of 309
    lanes at the level of COLLIER's own accuracy (errors change by factors 0.04 to 100, at
    most from 1.9e-10 to 1.9e-8 on a tier A lane; the per-n medians and maxima of the
    compared lanes are unchanged to two digits). COLLIER is not patched here; the benchmark
    runs it without the global cache, COLLIER's default.
  - `collect.py` writes the comparison to `accuracy.collier_cache` of the data file.
- **Accuracy flags.** The rank n+2 reference harness aggregates COLLIER's accuracy flag with
  `max` starting from 0 in two places, the Fortran driver
  (`validation/rank-n2/harness/collier_tensor_driver.f90`, `accmax = max(accmax, accflag)`)
  and the Python aggregation over a lane's pieces
  (`validation/rank-n2/harness/collier_reference.py`, `accflag = max(accflag, ...)`). The flag
  takes the values 0, -1 and -2 (COLLIER propagates it with `min`), so the harness always
  reports 0. `collier_bench` takes the minimum. The references themselves do not use the flag,
  so no reference value changes; both places would need `min` if the flag is ever used.
- **Parameter changes are not free.** Every `Set*_cll` call of a regulator or scale parameter
  runs `DDsetparam`, which recomputes the machine precision (about 1 µs, precise build), even
  in COLI mode. Changing mu_R^2 per point, or Delta between the three Laurent evaluations,
  costs a few µs per point.
- **Where COLLIER's time goes at rank n+2.** `perf` profiles of the precise build on
  `syn-n5-r7` and `syn-n6-r8` attribute about 60-75 % of the time to `addtotensortn`
  (`BuildTensors.F90`; 69 % and 57 % in one profile, 72.5 % and 65.0 % in the verification's),
  the construction of the tensor components from the Lorentz-covariant coefficients, whose
  number grows steeply with the rank.
- **Fresh cases.** On the final review's cases (`run_final_review.py`, aa6a630), the 20 of 166
  lanes on which NinjaRS's binary64 error exceeds 1e-6 (21 at 4dec120:
  `fr-mixed-complex-n5-r7` lane 2 went from 6.2e-7 and 1.2e-6 to 4.1e-7 and 4.0e-7, fast and
  generic) are lanes whose cut contributions cancel by 4e7 to 4e22 relative to the result.
  COLI stays below 1e-6 on 15 of the 20 (median 1.0e-8); on the other 5 its error is 4.2e-6
  to 1.2. The generic engine's
  opt-in `--stability-test` flags 18 of the 20, every one above 1e-5, and no lane below 1e-6;
  the two it misses are `fr-light-mixed-n4-r6-mu2` lane 1 (fast 6.5e-6, generic 4.2e-7) and
  `fr-mixed-complex-n4-r6` lane 1 (6.1e-6, 6.2e-6), on which COLI is within 1.7e-10 and DD
  within 2.5e-9. Against COLLIER directly (NinjaRS tolerance 1e-6), fast and COLI disagree on
  22 lanes: COLLIER is right on 15 (13 flagged `Unstable`), NinjaRS on 3
  (`fr-phys-real-n5-r7` lane 0, `fr-phys-real-n6-r8` lane 2, `fr-light-mixed-n7-r9-mu2`
  lane 1, where COLI is off by 7.2e-4, 7.8e-6 and 2.4e-6 with accuracy flag -1) and neither on
  4 (all flagged); DD's only miss is `fr-phys-real-n4-r6` lane 1 (2.5e-6, accuracy flag 0).
  The generic engine at higher precision returns every lane as `Success` and recovers all of
  them: DoubleFloat to 9.1e-9 at worst (`fr-light-mixed-n6-r8-mu2` lane 2, cancellation
  4e22), Float(128) to 8.2e-15, Float(256) to 6.5e-25 (`fr-mixed-complex-n7-r9` lane 1).
  The last is the references' own floor: their scalar C0/D0 table carries 45 digits, which the
  reduction amplifies by the cancellation (3e21 there), and Float(512) agrees with Float(256)
  to 1.1e-55 on that lane.
