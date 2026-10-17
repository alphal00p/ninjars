# MadLoop reference pipeline (Part 3)

MadLoop numerator + original Ninja (`o3-fortranrules`, integral cache off) +
OneLOop-3.6, for the single loop diagrams that NinjaRS evaluates. Results:
`SUMMARY.md` / `summary.json` (tables) and `<graph>.json` (everything measured,
with provenance). Exact Ninja inputs of MadLoop: `cases/madloop-<graph>.json`.

## Graphs and points

The graphs, MadLoop loop-filter ids, helicities, scales and the three-point
phase-space pools are those of the NUMERATOR stream
(`references/claude_numerator/validation/benchmark/points/*.json`; their sha256
is recorded in every result file):

| graph | process | filter id | pool | helicity (MadLoop index) | mu_R² |
| --- | --- | ---: | --- | --- | ---: |
| hgg-top-triangle-0 | `h > g g QED=1 [virt=QCD]` | 3 | hgg.json | 0,+,+ (4 of 4) | 15625 |
| four-photon-top-box-0 | `a a > a a [virt=QED]` (loop_sm, QED expansion on) | 182 | four-photon.json | ++++ (16 of 16) | 10000 |
| gg-hh-top-triangle-0 | `g g > h h QCD=2 QED=2 [virt=QCD]` | 11 | gg-hh.json | ++00 (4 of 4) | 10000 |
| gg-hh-top-box-2 | same | 10 | gg-hh.json | ++00 | 10000 |
| gg-hhh-top-triangle-0 | `g g > h h h QCD=2 QED=3 [virt=QCD]` | 66 | gg-hhh.json | ++000 (4 of 4) | 10000 |
| gg-hhh-top-box-8 | same | 58 | gg-hhh.json | ++000 | 10000 |
| gg-hhh-top-pentagon-26 | same | 55 | gg-hhh.json | ++000 | 10000 |

The four-photon model is a copy of `loop_sm` with the QED coupling order's
`perturbative_expansion` set to 1, exactly as `validation/madloop/four_photon/
prepare_four_photons.py` does (created under `target/benchmark/madloop/models`).

## Commands

```sh
source <devenv>
python3 validation/benchmark/build_ninja.py            # Part 1 (Ninja + OneLOop + instr build)
python3 validation/benchmark/madloop/run_madloop.py --reps 3 --cpu 29
python3 validation/benchmark/run_ninja_bench.py validation/benchmark/madloop/cases/*.json \
    --variants o3-fortranrules o3 --cpu 29             # replay cross-check
python3 validation/benchmark/madloop/summarize.py
```

CutTools (linked by every MadLoop output, unused with `MLReductionLib=6`) is a
copy of MG5's `vendor/CutTools` compiled once with the current gfortran 15.3
(`target/benchmark/madloop/cuttools`, its own makefile: `-O2 -funroll-all-loops`),
because MG5 rejects a CutTools library built by another compiler version.

`run_madloop.py`, per graph:

1. **Generation** — `mg5_generate.py` drives stock MG5_aMC 3.7.0 through its own
   command interpreter: `set group_subprocesses False`, `set loop_optimized_output
   True`, `set gauge Feynman`, `set output_dependencies external`,
   `set ninja <o3-fortranrules prefix>/lib`, `set collier None`, `import model`,
   `generate <process> --loop_filter="<base filter> and id==<id>"`,
   `output standalone`. All `set` use `--no_save` (the MG5 installation is not
   modified). The stock loop-filter predicate is wrapped only to record the
   selected diagram; MG5's serial `make` of `Source/` that `output standalone`
   runs is timed separately and subtracted. "gen" = Python start to end of
   `output` minus that make; "gen wall" adds interpreter start-up/imports.
   Exactly one loop diagram with multiplier 1 and one loop group is asserted.
   Repeated cold: 3 repetitions × 2 exports (fresh process and directory each).
2. **Compilation** — `default`: MadLoop's own standalone flags. Its `make_opts`
   defines no optimisation flag (`GLOBAL_FLAG` unset), so everything except
   `polynomial.f` (`-O3`) is compiled at gfortran's default `-O0`; the nix
   wrapper hardening that would silently add `-O2` is disabled. `O3`:
   `GLOBAL_FLAG=-O3` for `Source/` (rebuilt from scratch) and the P* directory.
   Source: `make` (inside `output` for default); P dir: `make check` (the Fortran
   standalone executable; plain `make` would also build f2py modules). Serial,
   pinned. `SHELL=<nix bash>` is passed because the generated makefiles
   hard-code `/bin/bash`, absent on this NixOS host.
3. **MadLoopParams.dat** — `MLReductionLib=6` (Ninja only), `CTModeRun=1`,
   `CTModeInit=1`, `ImprovePSPoint=-1`, `NRotations_DP=0`, `NRotations_QP=0`,
   `UseQPIntegrandForNinja=.false.`, `UseQPIntegrandForCutTools=.false.`;
   `param_card.dat`: top and Higgs widths 0 (all other values are the defaults,
   which equal the pool model: MT=ymt=173, MH=125, MZ=91.188, aEWM1=132.507,
   Gf=1.16639e-5, aS=0.118).
4. **Timing** (first repetition, both exports, CPU 29) — `bench_madloop.f` calls
   `ML5_0_SLOOPMATRIXHEL(P, HEL, ANS)` for the pool helicity over the three pool
   points (MadLoop leg order; `MU_R` from the pool, couplings not re-run),
   3 warm-up sweeps, calibration, 9 blocks of ≥25 ms, median. Executables:
   `bench` (plain), `bench_skip` (`loop_matrix.f` with `SKIPLOOPEVAL=.TRUE.`:
   external wavefunctions and loop-coefficient construction only) and
   `bench_instr` (instrumented Ninja; same objects otherwise). Single-helicity
   runs use `HelicityFilterLevel=0`: `SLOOPMATRIXHEL` then computes only that
   helicity, and the filter's initialization/double-check — which for an
   isolated, not gauge-invariant diagram can fail (H→gg, four photons) and
   re-initialize inside timed blocks — is avoided. The timed sequence
   (full, skip, instr) is run twice; medians over all 18 blocks. The
   helicity-summed `ML5_0_SLOOPMATRIX` run uses MadLoop's default filter (level 2,
   double check) when its initialization succeeds without warnings, else
   level 0 (recorded as `helicity_filter_level_summed`).
5. **Ninja inputs** — the instrumented library's `NINJA_BENCH_DUMP` writes every
   `ninja_tensor_evaluate` call (n, rank, mu², shifts, complex masses², MadLoop's
   S-matrix, tensor). The steady-state calls (one per evaluation, verified to
   cycle over the points) become `cases/madloop-<graph>.json`
   (`ninjars-bench-case-v1`), which the Part-2 replay driver times.
6. **Checks** — (a) the selected diagram (filter id, loop and attached PDGs,
   every vertex id and leg id/number/state/loop-line flag) equals the diagram of
   the existing raw MadLoop↔FeynKit matching evidence
   (`validation/evidence/{gg-hhh,gg-hh,four-photon}-madloop.json*`,
   `target/reference/madloop-hgg-validated/graphs.json` of the main checkout);
   (b) for every point the routing-invariant data of MadLoop's Ninja call —
   the multiset of the S-matrix entries (V_i−V_j)² and of the internal masses² —
   equals that of the NinjaRS bench case of the same graph and point.

## Split of the per-point time

Instrumented Ninja (`instrument_ninja.py`): RAII time-stamp-counter scopes
(rdtsc, calibrated against CLOCK_MONOTONIC) around the body of
`ninja_tensor_evaluate` (MadLoop's entry point, incl. TensorNumerator and
Amplitude construction), the four TensorNumerator callbacks and the eight
uncached AvHOneLoop master methods, plus an (n, rank) histogram of the calls.

* coef. constr. = `bench_skip` time (MadLoop numerator: tensor coefficients);
* TensorNum = TensorNumerator callbacks inside Ninja (polynomial evaluation at
  Ninja's sampling points — also numerator work in the NinjaRS sense);
* masters = OneLOop; Ninja reduction = Ninja total − TensorNum − masters;
* glue = full(instr) − coef. constr. − Ninja total (MadLoop's `NINJA_LOOP`
  interface: tensor copy, `INVERT_MOMENTA_IN_POLYNOMIAL`, S-matrix, bookkeeping).

The instrumentation overhead is visible as full(instr) − full (≤ a few per cent).
The replay of `cases/madloop-<graph>.json` with `ninja_bench` gives the same
split without any timers (`../ninja/madloop-<graph>.json`).

**Rank received by Ninja:** MadLoop declares the loop rank from the diagram's
propagator count: triangles rank 3, boxes rank 4, the pentagon rank 5 (see
`n_rank` in the JSON), while the NinjaRS numerators have ranks 2/4/4 (the
top-loop numerator degree). The higher rank makes MadLoop's TensorNumerator and
reduction work larger than necessary; this is MadLoop's normal behaviour.

## Results (CPU 29; full tables in SUMMARY.md)

One helicity (pool helicity), µs per phase-space point; generation+compile in seconds
(median gen + Source make + P-dir make). numerator = coefficient construction + TensorNumerator
callbacks; reduction = Ninja reduction + MadLoop glue; masters = OneLOop (no cache).

| graph | rank (ML) | flags | gen+compile | numerator | reduction | masters | total |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| hgg-top-triangle-0 | 3 | default | 4.26 | 17.83 | 4.68 | 3.31 | 25.61 |
| hgg-top-triangle-0 | 3 | O3 | 8.37 | 14.26 | 4.08 | 3.26 | 21.41 |
| four-photon-top-box-0 | 4 | default | 5.19 | 42.90 | 14.34 | 9.24 | 65.80 |
| four-photon-top-box-0 | 4 | O3 | 9.39 | 41.22 | 10.11 | 9.08 | 59.86 |
| gg-hh-top-triangle-0 | 3 | default | 4.57 | 16.86 | 4.73 | 1.97 | 23.23 |
| gg-hh-top-triangle-0 | 3 | O3 | 8.74 | 12.57 | 3.88 | 1.97 | 18.25 |
| gg-hh-top-box-2 | 4 | default | 4.95 | 39.11 | 12.74 | 32.63 | 84.09 |
| gg-hh-top-box-2 | 4 | O3 | 9.20 | 34.75 | 11.54 | 32.12 | 78.21 |
| gg-hhh-top-triangle-0 | 3 | default | 5.66 | 16.60 | 4.66 | 1.27 | 22.26 |
| gg-hhh-top-triangle-0 | 3 | O3 | 9.89 | 12.56 | 3.86 | 1.25 | 17.53 |
| gg-hhh-top-box-8 | 4 | default | 6.04 | 39.17 | 12.85 | 27.61 | 78.70 |
| gg-hhh-top-box-8 | 4 | O3 | 10.55 | 34.72 | 11.71 | 27.23 | 73.35 |
| gg-hhh-top-pentagon-26 | 5 | default | 6.83 | 106.81 | 29.06 | 105.39 | 239.77 |
| gg-hhh-top-pentagon-26 | 5 | O3 | 11.73 | 104.53 | 25.85 | 105.10 | 234.94 |

Generation alone (in-process, excluding the Source make) is 0.9–2.2 s per graph;
with Python start-up and imports 2.3–3.7 s. Compilation: Source 1.2–1.4 s and P dir 2.0–3.6 s
at MadLoop's default flags; 2.6–3.0 s and 4.5–7.0 s with -O3.

### Full process: all 50 top-loop diagrams of g g > h h h

`run_madloop_full.py`: filter `n in [3,4,5] and all(abs(p)==6 for p in loop_pdgs)` (8 triangles,
18 boxes, 24 pentagons; MadLoop forms 24 loop groups), same settings, gg-hhh pool, MadLoop's default
helicity filter for the sum. µs per point:

| flags | gen+compile (s) | helicity-summed | ++ only | coef. constr. | TensorNum | masters | Ninja reduction | Ninja calls/pt | master calls/pt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default | 8.01 | 6300 | 6320 | 517 | 2398 | 2434 | 883 | 50 | 1046 |
| O3 | 13.81 | 6286 | 6333 | 491 | 2378 | 2418 | 859 | 50 | 1046 |

Ninja is called once per diagram (50 calls per point: 8 triangles at rank 3, 18 boxes at
rank 4, 24 pentagons at rank 5), and with the integral cache disabled every call recomputes its
masters (1046 OneLOop calls per point; MG5's stock HEPTools Ninja has the cache enabled, which
would reuse the masters shared between diagrams). MadLoop's helicity filter reduces the helicity
sum to one independent helicity, so summed ≈ one helicity here.
