# Reference timings: original Ninja + OneLOop, and the MadLoop pipeline

> **Re-timed on 2026-10-18.** The committed timing JSONs were re-measured with the same executables
> on CPU 250, the core of the NinjaRS runs (`--cpu 250`); the CPU 29 tables below are the original
> runs. Both agree within about 1 % except OneLOop masters on the n = 5 cases (5–8 % slower on
> CPU 250); see [cpu-crosscheck.md](cpu-crosscheck.md).

This directory holds the *reference side* of the NinjaRS performance comparison:
original C++ Ninja (`references/ninja` @ 05fb0d3) with Fortran OneLOop, driven
either directly (Parts 1–2) or through MadLoop (Part 3, `madloop/README.md`).
Build products live in `target/benchmark/` (gitignored). Every timing was taken
on the shared 384-thread AMD EPYC 9754 host, pinned with `taskset -c 29` (an
L3 complex that was idle when measured; other users' jobs were not stopped).

| Path | Content |
| --- | --- |
| `build_ninja.py`, `ninja-builds.json` | Part 1: Ninja/OneLOop builds, exact commands, flags, hashes |
| `instrument_ninja.py`, `ninja_bench_timers.hh` | timer insertion for the `-instr` build (MadLoop split) |
| `ninja_bench.cc`, `run_ninja_bench.py` | Part 2: replay driver and runner |
| `make_synthetic_cases.py`, `synthetic/` | 20 synthetic `ninjars-bench-case-v1` cases |
| `ninja/<case>.json`, `ninja/<case>.masters.json`, `ninja/SUMMARY.md` | Part 2 results; exact master-integral call lists |
| `masters_native/` | oneloopmaster native masters on the identical call lists |
| `madloop/` | Part 3 (MadLoop generation, compilation, per-point timings) |

## Part 1 — Ninja builds

```sh
source <devenv>; python3 validation/benchmark/build_ninja.py --jobs 24
```

A fresh copy of the unchanged Ninja checkout per variant, `autoreconf -fi`, then
`configure --disable-shared --disable-autoflags --disable-gosam --enable-higher_rank
--disable-avholo_cache --with-avholo=... FCINCLUDE=... CXXFLAGS=... FCFLAGS=-O3`,
`make -C src -j24 libninja.la; make; make install`. Configuration is checked
from `config.h`: OneLOop on, integral cache **off**, X1RANK **on**, quadninja off.
The f90 module (`mninja.mod`) is built for every variant; the install prefix has
the layout MadGraph's `set ninja <prefix>/lib` expects (with a `libavh_olo.a`
symlink and `bin/ninja-config -quadsupport` = false). The nix compiler-wrapper
hardening is disabled (`NIX_HARDENING_ENABLE=''`), so the effective flags are
exactly the listed ones (nixpkgs gcc still defaults to frame pointers and
`-march=x86-64 -mtune=generic`). gcc/gfortran 15.3.0.

| variant | C++ flags | OneLOop | libninja.a sha256 | libavh_olo.a sha256 |
| --- | --- | --- | --- | --- |
| `o3` | `-O3 -DNDEBUG` | 3.6 | `b6c3a01e70c15521…` | `502929c43f0dcb30…` |
| `o3-fortranrules` | `-O3 -DNDEBUG -fcx-fortran-rules` | 3.6 | `858309a8a0759066…` | `502929c43f0dcb30…` |
| `o3-fortranrules-instr` | same + timers (`instrument_ninja.py`) | 3.6 | `77e89dc5f41f0a16…` | `502929c43f0dcb30…` |
| `o3-ol34` | `-O3 -DNDEBUG` | 3.4 | `d2951af29fd60779…` | `28c37c328fadd6f4…` |

OneLOop is compiled with `gfortran -O3 -fPIC`. **Version choice:** Ninja does not
bundle OneLOop. The primary builds use OneLOop-3.6 exactly as MG5_aMC's HEPTools
installs it (`HEPTools/oneloop/OneLOop-3.6/avh_olo.f90`, generated with
`QPKIND=16`), because that is what MadLoop links and its module names do not
clash with the renamed OneLOop copies inside CutTools and IREGI that MadLoop also
links. `o3-ol34` reproduces the earlier `target/reference-uncached-v2` choice
(IREGI's OneLOop-3.4 + `validation/reference/avh_olo_alias.f90`; same
`libavh_olo.a` hash 28c37c32…). Master timings of 3.4 and 3.6 agree within a
few per cent (Part 2 tables). For reference, the stock HEPTools Ninja of this MG5
installation was built with `-O2` + autoflags (`-fcx-fortran-rules`), quadninja,
the integral cache enabled, and OneLOop at `-O`.

## Part 2 — replay driver (`ninja_bench.cc`)

```sh
python3 validation/benchmark/make_synthetic_cases.py
python3 validation/benchmark/run_ninja_bench.py --discover \
    --variants o3 o3-fortranrules o3-ol34 --cpu 29 --rounds 9
```

`--discover` adds `synthetic/*.json` and the ENGINE/NUMERATOR case directories
(`validation/fast/cases`, `validation/benchmark/cases`) of this worktree, the main
checkout and `references/claude_{engine,numerator}`. The driver parses the
`ninjars-bench-case-v1` JSON itself (small built-in parser) and, per lane, uses
`ninja::Amplitude<ninja::ComplexMasses>(n, rank, V, m2)` with
`setRenormalizationScale(mu_squared)` — Ninja's "renormalization scale" is
**mu_R²**: it is passed to `IntegralLibrary::init(muRsq)` and `AvHOneLoop`
hands `sqrt` of it to OneLOop's `rmu` argument — and `setSMatrix` when the case
has one, with a `ninja::TensorNumerator` over the lane's tensor.

Modes (each evaluates one lane; lanes cycled):

* `full` — TensorNumerator + `ninja::avh_olo` (OneLOop, no cache).
* `pure_reduction` — `ReplayNumerator` (a `ninja::Numerator` returning, in call
  order, the outputs recorded from TensorNumerator; one `memcpy` per call) +
  `ReplayLibrary` (an `IntegralLibrary` returning recorded OneLOop results).
* `reduction_plus_tensor_numerator` — TensorNumerator + ReplayLibrary.
* `masters_only` — the recorded `IntegralLibrary` calls re-issued through
  `ninja::avh_olo` (same entry points Ninja used), no reduction; also split by
  family (`masters_boxes/triangles/bubbles/tadpoles`).
* `numerator_only` — the recorded TensorNumerator calls (inputs) re-issued.
* `full_madloop_entry` — `ninja_tensor_evaluate_[smat_]cm_`, i.e. exactly
  MadLoop's call (new TensorNumerator + Amplitude per call).

The recording pass wraps TensorNumerator (outputs captured through a NaN-sentinel
scratch buffer, so the replay writes exactly the entries the numerator wrote) and
AvHOneLoop. Every lane checks that the recorded, pure-replay, tensor+replay and
MadLoop-entry evaluations, and the masters-only replay, are **bitwise identical**
to the direct evaluation, and that the replays are consumed exactly. Protocol:
one calibration and one discarded warm-up block per mode, then 9 interleaved
rounds (one ≥25 ms block of every mode per round); median over rounds. The
numerator part is `reduction_plus_tensor_numerator - pure_reduction`; the check
`(pure + numerator part + masters)/full` lies in 0.967–1.01 for every case except
the 0.8 µs bubble `syn-n2-r1` (1.02–1.05, fixed per-call overheads).

**OneLOop warnings are switched off** (`olo_unit(-1,'warning')`; `--oneloop-warnings`
restores them). OneLOop-3.6 writes one line to stdout for every D0 call whose
diagonal invariant s31 or s42 vanishes ("input does not seem to represent hard
kinematics, trying to permutate") and then permutes the arguments. The NinjaRS
physical box/pentagon cases list their propagators in an order in which this
happens (1 of 1 boxes for the box graphs, 2 of 5 for the pentagon); MadLoop's
cyclic order never triggers it, and IREGI's OneLOop-3.4 (`o3-ol34`) prints
nothing. With the warnings on, the formatted write into the runner's pipe was
timed as master work: ~1.3 µs per warning, i.e. the first commit's `masters`
were 12 % (four-photon box), 5–7 % (gg→HH/HHH boxes) and 3 % (pentagon) too
high. The permutation itself is still performed and timed.

Synthetic cases (`make_synthetic_cases.py`): n = 2..6 with ranks n-1, n, n+1;
gg→HHH-like pentagons (five masses 173², sqrt(s)=450, MH=125, two leg orders,
ranks 4 and 5); random two-to-many kinematics with timelike and spacelike
invariants; real, complex (Γt = 1.5) and massless internal masses; dense random
tensors. Every case except `syn-gghhh-nosmat` carries an explicit S-matrix whose
rounding residues (e.g. -6e-12 for massless legs) are snapped to exact 0 or an
external mass², as MadLoop's S-matrix and OneLOop's on-shell threshold do.

Results: `ninja/SUMMARY.md` (all cases, three builds). Physical cases (NUMERATOR
stream, `validation/benchmark/cases` of the main checkout, merged in e6d7fa0 with
the sha256 recorded per result file), build `o3-fortranrules`, µs per point:

| case | n | rank | full | pure reduction | numerator part | masters | D0 | C0 | B (B11+B0) | A0 | full (`o3`) | pure (`o3`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hgg-top-triangle-0 | 3 | 2 | 10.09 | 2.61 | 0.90 | 6.54 | - | 4.80 | 1.63 | 0.12 | 11.03 | 3.26 |
| four-photon-top-box-0 | 4 | 4 | 42.70 | 9.10 | 22.65 | 10.13 | 3.87 | 4.73 | 1.33 | 0.16 | 52.86 | 11.00 |
| gg-hh-top-triangle-0 | 3 | 2 | 5.38 | 2.55 | 0.88 | 1.88 | - | 0.90 | 0.86 | 0.12 | 6.35 | 3.22 |
| gg-hh-top-box-2 | 4 | 4 | 60.37 | 9.25 | 22.76 | 27.06 | 16.28 | 6.59 | 3.91 | 0.15 | 70.50 | 11.13 |
| gg-hhh-top-triangle-0 | 3 | 2 | 4.78 | 2.54 | 0.90 | 1.26 | - | 0.66 | 0.50 | 0.12 | 5.68 | 3.21 |
| gg-hhh-top-box-8 | 4 | 4 | 54.84 | 9.30 | 22.65 | 22.02 | 13.14 | 5.41 | 3.15 | 0.16 | 65.96 | 11.70 |
| gg-hhh-top-pentagon-26 | 5 | 4 | 134.03 | 18.95 | 22.04 | 91.38 | 59.19 | 24.20 | 7.24 | 0.19 | 147.32 | 23.42 |

These are a review re-run of all 34 cases (CPU 29, OneLOop warnings off). Against the
first commit's values (new/old, all cases and builds): full 0.97–1.01 (median 0.998;
the lowest values are the warning effect), pure reduction 0.98–1.04, masters 0.97–1.05
where no warning was involved, numerator-only replay 0.99–1.04; the derived numerator
part (a difference of two modes) 0.92–1.11 on the sub-µs bubbles, ±1 % elsewhere.

**Propagator order changes the OneLOop cost of the same integrals.** MadLoop's
Ninja inputs of the same diagrams and points (`madloop/cases`, cyclic loop order)
give, for the identical master integrals: H→gg C0 1.47 µs (NinjaRS order
C0(0, mH², 0): 4.80 µs), four-photon masters 8.97 vs 10.13, gg→HH box 32.13 vs
27.06, gg→HHH box 26.92 vs 22.02, pentagon 102.03 vs 91.38 µs. Compare NinjaRS's
masters with the MadLoop pipeline's own (`madloop/`), and consider canonicalising
the master-call argument order in NinjaRS (it also changes the native cost below).

`-fcx-fortran-rules` (MadGraph's Ninja autoflag) makes the reduction ~20 % and
the TensorNumerator ~27 % faster: without it every complex product checks for
NaN/inf (C99 Annex G) — a check NinjaRS (Rust) does not perform. It never
changed a result (bitwise identical to `o3` on all physical cases). The rank
dominates the numerator cost: e.g. the n=5 pentagon at rank 5 (what MadLoop
passes) costs 112 µs of TensorNumerator vs 31 µs at rank 4.

## oneloopmaster native masters on the same calls (`masters_native/`)

```sh
# exact revisions are exported with `git archive` (the shared checkouts are untouched)
python3 validation/benchmark/masters_native/build_native.py --label olm-3023fa6-sym-e679e8c1-tensor \
    --oneloop 3023fa6 --symbolica e679e8c1 --tensor   # the report's timings (CPU 250); the pinned
                                                       # 149ddff returns the same bits on every call
python3 validation/benchmark/masters_native/build_native.py --label olm-a92b148-sym-e679e8c1-tensor \
    --oneloop a92b148 --symbolica e679e8c1 --tensor
python3 validation/benchmark/masters_native/build_native.py --label olm-c9dd82f-sym-e679e8c1 \
    --oneloop c9dd82f --symbolica e679e8c1
L=olm-a92b148-sym-e679e8c1-tensor   # likewise for the other label
taskset -c 29 target/benchmark/masters-native/$L/target/release/masters-native-bench \
    validation/benchmark/masters_native/results/$L.json validation/benchmark/ninja/*.masters.json --label $L
python3 validation/benchmark/masters_native/summarize.py
```

A standalone crate (NinjaRS release profile: thin LTO, one codegen unit) calls
oneloopmaster's direct ahead-of-time Rust backend, `oneloop::NativeEvaluator<f64>`
(`generated-evaluators`; not the interpreted expression or SymJIT routes), on
exactly the master calls Ninja made (`<case>.masters.json`, OneLOop argument
order), and with `--tensor` `NativeTensorEvaluator<f64>` B11/B111 for Ninja's
rank-2/3 bubble calls. It also checks native C0/D0 against their cyclic/reflected
relabellings. Results and per-family tables: `masters_native/SUMMARY.md`.

**The dilogarithm decides everything.** With the NinjaRS workspace's Symbolica pin
(2c06f245) the generated code's f64 `dilog` goes through Symbolica's registered
`polylog(2, z)` callback, which converts to MPFR and sums a series (12 µs – 2 ms per
call): the first commit's measurement (`results/olm-c9dd82f-sym-2c06f245.json`)
had C0/D0 25–500× slower than OneLOop (pentagon 39.6 ms). oneloopmaster's own
documentation (STATUS.md, `patches/README.md`) relies on a binary64 dilogarithm
specialization that the Symbolica-main migration had dropped; it is restored in
Symbolica e679e8c1 (`claude/oneloopmaster-integration-dilog` = 2c06f245 + binary64
Bernoulli-series Li2: 17–110 ns per call), which oneloopmaster a92b148
(`claude/native-masters`) calls directly. Native vs OneLOop (`o3-fortranrules`,
warnings off) on identical call streams, µs per point, a92b148 + e679e8c1:

| case | OneLOop | native | ratio | D0 OL/native | C0 OL/native | bubbles OL/native (B11 tensor) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hgg-top-triangle-0 | 6.54 | 3.58 | 0.55 | – | 4.80 / 3.19 | 1.63 / 0.26 |
| four-photon-top-box-0 | 10.13 | 15.60 | 1.54 | 3.87 / 8.09 | 4.73 / 6.85 | 1.33 / 0.43 |
| gg-hh-top-triangle-0 | 1.88 | 1.90 | 1.01 | – | 0.90 / 1.55 | 0.86 / 0.23 |
| gg-hh-top-box-2 | 27.06 | 34.30 | 1.27 | 16.28 / 24.50 | 6.59 / 8.68 | 3.91 / 0.75 |
| gg-hhh-top-triangle-0 | 1.26 | 1.74 | 1.38 | – | 0.66 / 1.42 | 0.50 / 0.20 |
| gg-hhh-top-box-8 | 22.02 | 33.02 | 1.50 | 13.14 / 23.77 | 5.41 / 8.16 | 3.15 / 0.70 |
| gg-hhh-top-pentagon-26 | 91.38 | 133.68 | 1.46 | 59.19 / 103.51 | 24.20 / 27.50 | 7.24 / 1.39 |
| madloop-gg-hhh-top-pentagon-26 | 102.03 | 134.10 | 1.31 | 61.39 / 100.05 | 32.40 / 31.37 | 7.23 / 1.39 |

oneloopmaster main c9dd82f with the same Symbolica (registered callback now binary64,
rank-2 bubbles as B0 only) is within ~1 % of these totals (pentagon 135.2 µs).
Accuracy vs OneLOop ≤ 6e-14 (scaled) on every physical case, and native C0/D0 are
relabelling-invariant there to ≤ 1e-14. The remaining gap is D0 (pentagon: 1.75×
OneLOop) and, on the triangles-only graphs, C0; bubbles are 3–6× faster than
`olo_b11`, which also computes B00.

**Native C0 defect** (independent of the dilogarithm, reported to the coordinator
by the first pass): C0 with one massless and two massive internal lines above the
massive-pair threshold has a wrong imaginary part, e.g. C0(0, 0, 490000; 173², 0,
173²): native 3.4587e-5 (real; equal to |OneLOop|), OneLOop -5.698e-6 - 3.411e-5 i;
in `syn-n6-r6` C0(0, 70988, 271317; 173², 0, 173²) the native value is not even
invariant under relabelling (deviation 1.7). No physical case contains such a
triangle (all-massive top loops).

## Part 3 — MadLoop

See `madloop/README.md` (commands, definitions, results) and `madloop/SUMMARY.md`.
In short, for the gg→HHH pentagon (filter id 55), one helicity, MadLoop's default
flags: 240 µs per point = 25.6 coefficient construction + 81.2 TensorNumerator
(MadLoop passes rank 5) + 26.1 Ninja reduction + 105.4 OneLOop + 3.0 glue;
generation 2.0 s + compilation 4.8 s. Replaying MadLoop's exact Ninja inputs
with `ninja_bench` reproduces the split (pure 22.5, numerator 81.1, masters
102.0 µs). A review re-timing of all 7 graphs × 2 flag sets on the same
executables (`madloop/retime_madloop.py`, `madloop/retime-review.json`) agrees
within 1 % (pentagon default: 241.1 vs 239.8 µs).

## Caveats

* Shared host: medians over interleaved blocks on one pinned CPU; host load
  varied (load average 25–50). Independent review re-runs (Part 2: all cases and
  builds; Part 3: all graphs) agree within ~1–4 % (see above); no build ran
  during those timings.
* Part 2 constructs the Amplitude and TensorNumerator once per lane outside the
  timed loop (`full_madloop_entry` shows that per-call construction costs ≤1 %).
* OneLOop 3.4 and 3.6 differ by up to 8e-8 (scaled) on one synthetic rank-5
  pentagon lane (≤ 1e-12 elsewhere, ≤ 2e-15 on the physical cases); `o3` and
  `o3-fortranrules` were bitwise identical on every physical case.
