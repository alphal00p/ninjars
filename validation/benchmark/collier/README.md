# COLLIER at numerator rank n+2: timing and accuracy reference

Original Ninja stops at numerator rank n+1, so the rank n+2 part of the benchmark report
([`docs/benchmark/benchmark.typ`](../../../docs/benchmark/benchmark.typ), section "Numerator
rank n + 2", data in `docs/benchmark/data/rank_n2.json`) compares NinjaRS with the
tensor-integral library COLLIER 1.2.10 (Denner, Dittmaier, Hofer, Comput. Phys. Commun. 212
(2017) 220, arXiv:1604.06792; COLI and DD branches: Denner, Dittmaier, Nucl. Phys. B 658 (2003)
175, B 734 (2006) 62, B 844 (2011) 199).

COLLIER computes tensor integrals and contracts them with the numerator's tensor coefficients.
That is a different method from integrand reduction, so the comparison is a timing reference,
not like for like (see "What is compared").

## Pipeline

```sh
source <devenv>                                     # gfortran, python3; cmake via nix if absent
validation/benchmark/collier/build.sh               # COLLIER 1.2.10 (precise, stock) + collier_bench
cargo build --profile perf --locked -p ninjars-cli --features numerator-jit   # target/perf/ninjars
python3 validation/benchmark/collier/run_bench.py --rounds 2     # timings (pinned, interleaved), ~1 h
python3 validation/benchmark/collier/run_accuracy.py            # case-check verdicts, ~10 min
python3 validation/benchmark/collier/count_instructions.py      # perf stat instruction counts
python3 validation/benchmark/collier/collect.py                 # -> docs/benchmark/data/rank_n2.json
cd docs/benchmark && nix shell nixpkgs#typst -c typst compile benchmark.typ benchmark.pdf
```

| file | role |
|---|---|
| `build.sh` | downloads `collier-1.2.10.tar.gz` from hepforge into `target/benchmark/collier-1.2.10/download`, checks its sha256 (`460624ea…62713`), unpacks it unmodified and builds two static libraries with cmake: `precise` (the shipped gfortran Release flags without `-ffast-math`: `-O2 -funroll-loops`) and `stock` (the shipped flags, `-O2 -funroll-loops -ffast-math`); links `collier_bench` against each with the same flags |
| `prepare.py` | decomposes every lane into COLLIER pieces with `validation/rank-n2/harness/collier_reference.py` (`lane_sink`, `sink_to_jobs`: exact removal of the integrand mu^2 by pinching, shift of each piece to p_0 = 0, the lane's S-matrix as invariants), then rounds masses, momenta, invariants and coefficients to binary64 once |
| `collier_bench.f90` | the binary64 driver: validation pass (three Delta settings, Laurent coefficients, flags, error estimates), then timed quantities (below) |
| `run_bench.py` | runs NinjaRS (`case-bench`, fast engine) and COLLIER, interleaved per round on one CPU; the generic engine once at the end |
| `run_accuracy.py` | `ninjars case-check` on the suite and synthetic cases: fast and generic f64, generic DoubleFloat, Float(128), Float(256) |
| `count_instructions.py` | retired user-space instructions per point (`perf stat`, difference of two fixed-work runs) |
| `collect.py` | the data file of the report |
| `results/` | the raw, consolidated measurements (`collier.json`, `ninjars-fast.json`, `ninjars-generic.json`, `accuracy.json`, `instructions.json`, `environment.json`) |

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
  (also measured with one cache per lane: `precise-coli-cache`), `Init_cll(Nmax, Rmax)` with the
  case's largest piece.
- **NinjaRS's time.** `ninjars case-bench` (widths 1 and 32): fast engine total with the
  tensor-numerator expansions, the pure reduction (replayed numerator and masters) and the
  native masters (replayed calls), exactly as in the rank <= n+1 benchmark. The generic engine
  in binary64 (`case-bench --engines generic`) uses `case-check`'s numerically expanded
  dense-tensor numerator, which is not the production generic numerator; its reduction-only
  time (replayed numerator and masters) is the comparable figure.

## Timing method

- Every process is single-threaded and pinned with `taskset -c CPU` to one CPU that was idle
  (`/proc/stat` over 5 s; the host has no SMT siblings). A round runs NinjaRS on every case,
  then every COLLIER variant on every case; `--rounds` rounds; `collect.py` takes the smallest
  of the rounds' block medians. Blocks last at least 20 ms (NinjaRS: 9 blocks; COLLIER: 5;
  generic: 5). `environment.json` records, per run, the time other processes spent on the CPU
  (`foreign_s`).
- The core clock varies between 1.8 and 3.1 GHz on the loaded host. `count_instructions.py`
  therefore also counts retired instructions per point, which do not depend on it.

## Validation of the driver

Against the high-precision references of `validation/rank-n2/reference`, the binary64 COLI
results of the precise build reproduce the 1.2.9 `coli-double` diagnostics of the references
(precise build, exact decimal inputs, 90-digit contraction): `collect.py` writes the comparison
to `accuracy.collier_vs_1.2.9` of the data file. COLLIER 1.2.10 changes, against 1.2.9, the
imaginary parts of complex-mass C0 roots in `coli_c0.F` and two error-estimate guards for
rank 0 (`reductionC.F90`).

## Findings

- **Accuracy flags.** The rank n+2 reference harness aggregated COLLIER's accuracy flag with
  `max(accmax, accflag)` starting from 0 (`validation/rank-n2/harness/collier_tensor_driver.f90`),
  but the flag takes the values 0, -1 and -2, so the harness always reported 0. `collier_bench`
  takes the minimum. The references themselves do not use the flag.
- **Parameter changes are not free.** Every `Set*_cll` call of a regulator or scale parameter
  runs `DDsetparam`, which recomputes the machine precision (about 1 µs, precise build), even
  in COLI mode. Changing mu_R^2 per point, or Delta between the three Laurent evaluations,
  costs a few µs per point.
