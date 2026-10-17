# Reference timings: original Ninja + OneLOop, and the MadLoop pipeline

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
`(pure + numerator part + masters)/full` lies in 0.97–1.01 for all cases.

Synthetic cases (`make_synthetic_cases.py`): n = 2..6 with ranks n-1, n, n+1;
gg→HHH-like pentagons (five masses 173², sqrt(s)=450, MH=125, two leg orders,
ranks 4 and 5); random two-to-many kinematics with timelike and spacelike
invariants; real, complex (Γt = 1.5) and massless internal masses; dense random
tensors. Every case except `syn-gghhh-nosmat` carries an explicit S-matrix whose
rounding residues (e.g. -6e-12 for massless legs) are snapped to exact 0 or an
external mass², as MadLoop's S-matrix and OneLOop's on-shell threshold do.

Results: `ninja/SUMMARY.md` (all cases, three builds). Physical cases (NUMERATOR
stream, `references/claude_numerator/validation/benchmark/cases`, sha256 recorded
per result file), build `o3-fortranrules`, µs per point:

| case | n | rank | full | pure reduction | numerator part | masters | D0 | C0 | B (B11+B0) | A0 | full (`o3`) | pure (`o3`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hgg-top-triangle-0 | 3 | 2 | 10.02 | 2.54 | 0.93 | 6.51 | - | 4.76 | 1.62 | 0.12 | 11.05 | 3.24 |
| four-photon-top-box-0 | 4 | 4 | 44.17 | 9.17 | 22.67 | 11.41 | 5.07 | 4.73 | 1.33 | 0.15 | 54.56 | 11.14 |
| gg-hh-top-triangle-0 | 3 | 2 | 5.44 | 2.56 | 0.92 | 1.93 | - | 0.92 | 0.91 | 0.12 | 6.46 | 3.26 |
| gg-hh-top-box-2 | 4 | 4 | 61.36 | 9.22 | 22.76 | 28.57 | 17.71 | 6.59 | 3.93 | 0.15 | 71.91 | 11.16 |
| gg-hhh-top-triangle-0 | 3 | 2 | 4.76 | 2.56 | 0.93 | 1.26 | - | 0.66 | 0.50 | 0.12 | 5.76 | 3.26 |
| gg-hhh-top-box-8 | 4 | 4 | 56.50 | 9.32 | 22.62 | 23.60 | 14.67 | 5.48 | 3.18 | 0.16 | 67.26 | 11.33 |
| gg-hhh-top-pentagon-26 | 5 | 4 | 136.76 | 18.57 | 22.17 | 94.61 | 62.06 | 24.22 | 7.23 | 0.19 | 150.36 | 23.61 |

`-fcx-fortran-rules` (MadGraph's Ninja autoflag) makes the reduction ~20 % and
the TensorNumerator ~27 % faster: without it every complex product checks for
NaN/inf (C99 Annex G) — a check NinjaRS (Rust) does not perform. It never
changed a result (bitwise identical to `o3` on all physical cases). The rank
dominates the numerator cost: e.g. the n=5 pentagon at rank 5 (what MadLoop
passes) costs 112 µs of TensorNumerator vs 31 µs at rank 4.

## oneloopmaster native masters on the same calls (`masters_native/`)

```sh
cd validation/benchmark/masters_native && cp ../../../Cargo.lock . && \
  CARGO_TARGET_DIR=../../../target/benchmark/masters-native-target cargo build --release --offline
taskset -c 29 target/benchmark/masters-native-target/release/masters-native-bench \
  validation/benchmark/masters_native/results.json validation/benchmark/ninja/*.masters.json
python3 validation/benchmark/masters_native/summarize.py
```

A standalone crate (NinjaRS release profile: thin LTO, one codegen unit) calls
`oneloop::NativeEvaluator<f64>` — the direct ahead-of-time Rust backend
(`generated-evaluators`), not the interpreted expression or SymJIT routes — on
exactly the master calls Ninja made (`<case>.masters.json`, OneLOop argument
order). Results agree with OneLOop to ≤6e-14 on all physical cases, but C0/D0
are **100–500× slower** (`masters_native/SUMMARY.md`): e.g. the gg→HHH pentagon's
masters take 94 µs in OneLOop and 39.6 ms natively (D0 26 ms, C0 13.6 ms). The
cause is the dilogarithm: the generated code's f64 `dilog` calls Symbolica's
`polylog(2, z)` for `Complex<f64>`, which converts to MPFR and sums a series
(12 µs – 2 ms per call, `DILOG_TEST=1`). B0 and A0 are as fast as or faster
than OneLOop. Two synthetic native C0 values with one massless and two massive
internal lines are also wrong (real part right, imaginary part/sign wrong; e.g.
C0(0,0,490000; 173²,0,173²): native 3.46e-5, OneLOop -5.70e-6-3.41e-5i).

## Part 3 — MadLoop

See `madloop/README.md` (commands, definitions, results) and `madloop/SUMMARY.md`.
In short, for the gg→HHH pentagon (filter id 55), one helicity, MadLoop's default
flags: 240 µs per point = 25.6 coefficient construction + 81.2 TensorNumerator
(MadLoop passes rank 5) + 26.1 Ninja reduction + 105.4 OneLOop + 3.0 glue;
generation 2.0 s + compilation 4.8 s. Replaying MadLoop's exact Ninja inputs
with `ninja_bench` reproduces the split (pure 22.6, numerator 81.1, masters
101.6 µs).

## Caveats

* Shared host: medians over interleaved blocks on one pinned CPU; host load
  varied (load average 25–50). Repeated runs agree within ~1–2 %.
* Part 2 constructs the Amplitude and TensorNumerator once per lane outside the
  timed loop (`full_madloop_entry` shows that per-call construction costs ≤1 %).
* OneLOop 3.4 and 3.6 differ by up to 3e-8 (scaled) on one synthetic rank-5
  pentagon lane; `o3` and `o3-fortranrules` were bitwise identical on every
  physical case.
