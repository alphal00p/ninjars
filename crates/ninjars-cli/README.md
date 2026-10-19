# NinjaRS command-line preparation

`ninjars prepare examples/graphs/bubble.json` imports the FeynKit graph and emits
exact per-sample numerator, propagator order, shifts, squared masses and graph
prefactors. The JSON fixture's numeric strings stay in Symbolica until a
numerical evaluator is explicitly selected. `--output` saves the JSON result.

The optional `reference-codegen` feature enables an independent original Ninja
validation export:

```console
nix develop --command cargo run -p ninjars-cli --features reference-codegen -- prepare examples/graphs/bubble.json --reference-dir target/reference/bubble-export --output target/reference/bubble-prepared.json
nix develop --command python3 validation/reference/build.py --runner-only --numerator-header target/reference/bubble-export/sample-000000.hh --output target/reference/bubble-reference
./target/reference/bubble-reference < target/reference/bubble-export/sample-000000.inputs
```

Run the complete reference build once before using `--runner-only`, as described
in `validation/reference/README.md`. This export contains:

- `prepared.json`: the complete preparation result with exact expressions and
  original fixture inputs.
- `manifest.json`: sample order, rank, numeric reference inputs, and conversion
  metadata.
- `sample-NNNNNN.hh`: Symbolica-generated callbacks for the fully bound sample.
- `sample-NNNNNN.inputs`: a matching request for the original reference runner.

The callback contains the prepared numerator multiplied by its overall factor
exactly once, including graph weights and loop-independent internal propagators.
Its integrated result must **not** be multiplied by the graph factor again.
The ordinary preparation JSON keeps numerator and overall factor separate for
library consumers.

The C++ reference uses binary64. Export retains exact inputs alongside an explicit
record of conversion: Symbolica evaluates shifts, masses and scale at 200 bits,
then rounds to binary64. Nonfinite values, overflow and underflow are rejected.
Complex squared masses are supported; complex external momenta are rejected
because original Ninja's public kinematic input uses real momentum components.
The numerator must be fully bound except for `ninjars::q0` through `q3` and
`ninjars::mu2`. Callback generation and compilation are setup costs; these
reference exports are not performance acceptance evidence.

A reproducible end-to-end check uses the provided graph fixtures:

```console
nix develop --command cargo build -p ninjars-cli --features reference-codegen
nix develop --command python3 validation/reference/test_prepared_runner.py --cli target/debug/ninjars
```

Use Cargo's configured target path if customized. The checks compare the graph's
exported callbacks against original Ninja's independent tensor interface plus
analytic extra-dimensional rational contributions. They currently cover three
bubble samples and one triangle. This verifies preparation/reference agreement;
it does not claim parity for the unfinished Rust reduction engine.

## R2 rational terms (`--r2`)

Every command that builds a physical numerator (`prepare`, `reduce`, `bench`,
`compare`, `export-bench`, `numerator-bench`, `pipeline-bench`) accepts
`--r2 off|fdh`:

- `off` (default, unchanged results): FeynKit's Feynman rules and Idenso's
  `d_s = 4` tensor algebra give a polynomial in scalar products of the loop
  momentum `K`; every product is bound with the four-dimensional metric,
  `K.K -> q0^2-q1^2-q2^2-q3^2`. The numerator has no `mu^2` terms, so the
  reduction's d-dimensional denominators supply R1 but no R2.
- `fdh`: four-dimensional-helicity scheme. The internal algebra stays
  `d_s = 4` while the loop momentum is d-dimensional: `K.K -> q.q - mu^2`
  (`K.P`, `K.eps` unchanged). The explicit `mu^2` terms then give all of R2
  in FDH; `mu^2` has rank weight two, so the rank does not change. Closed
  fermion, scalar or ghost loops with only external bosons (H→gg, gg→HH(H),
  γγ→γγ) have scheme-independent R2 and match MadLoop's (HV) R2 vertices;
  with internal gluons FDH differs from HV/CDR by the universal
  `lambda_HV` terms, whose `(d_s-4)`-dimensional algebra (`N0 + eps N1 +
  eps^2 N2`) is not implemented.

The FDH binding is exact only if no four-dimensional-only identity reached the
loop momentum before binding. `--r2 fdh` therefore rejects numerators with
`gamma5`/chiral projectors or Levi-Civita tensors (FDH needs a `gamma5`
prescription; Idenso uses four-dimensional axial-trace identities), external
spinors (open fermion lines would need the loop momentum in components, which
loses `mu^2`; hand-written spin-trace projectors work) and Minkowski metrics in the external projector (four-dimensional
polarization sums); see `ninjars_graph::R2Scheme` and
[`docs/physical-preparation.md`](../../docs/physical-preparation.md#r2-rational-terms-in-fdh).
The prepared document records the scheme under `r2`.

```console
nix develop --command cargo run --release -p ninjars-cli -- reduce \
  examples/graphs/physical/hgg/top-triangle-0.json --r2 fdh --engine fast
```

## Hand-written DOT graphs

`prepare`, `reduce`, `bench`, `export-bench` and `compare` accept any connected
one-loop graph in FeynKit's compact DOT dialect or its stable export. Every
choice that changes the integrand is recorded in the preparation JSON. Test
fixtures for each case are in `tests/graphs`.

**Models.** `--model` selects `phi3`, `phi4`, `sm`, or one of the Standard Model
sectors `qcd`, `qed`, `qcd_qed`, `yang_mills` and `electroweak`. Any other value
is a FeynKit model JSON path, relative to the working directory (or, in a
fixture's `model`, to the fixture). An unknown name reports the path it tried.

**Numerators.** Without `--feynman-rules`, local `num` attributes are the
complete integrand. A compact DOT without any `num` attribute is rejected rather
than silently given the numerator one; write `num="1"` for a scalar integral.
With `--feynman-rules`, FeynKit's model rules and default external states
(`u`, `ubar`, `v`, `vbar`, `epsilon`, `epsilonbar`) are used unless the DOT
supplies a `projector`.

**Graph factors.** FeynKit's generator folds `1/AutG` and a `-1` per closed
fermion or ghost loop into a generated diagram's `overall_factor`. For a compact
DOT without `overall_factor`, every command preparing it with `--feynman-rules`
(including `numerator-bench` and `pipeline-bench`) derives and applies the same
factors. A stated `overall_factor`, or the stable export, is used as written.
`graph_factors` reports the automorphism order, closed fermion/ghost loops,
open fermion lines, the factor, and whether it was applied; explicit mode notes
a factor different from one. The generator also derives a sign per diagram
from the order of the external fermions on its open lines. Only a process's
global sign is a convention: relative signs between diagrams with two or more
open fermion lines are physical (e.g. the s- and t-channel diagrams of Bhabha
scattering). That sign is not derived, so preparation warns for such graphs:
state it with `overall_factor`.

**Color.** When the contracted numerator leaves exactly one joinable pair of
open color indices (two gluons, or a quark and an antiquark), the coefficient
of their invariant is extracted with `delta/N` (`delta_ab/8`, `delta_ij/3`), as
in the shipped physical examples; `color_projector` records it. Other open
indices are an error naming each slot and its external leg. A custom DOT
`projector` refers to external states by edge and to open indices by half-edge:
`external_legs` lists both for every leg. `--color-dimension-invariants` (or the
fixture's `color_dimension_invariants`) evaluates invariants such as
`cas(2,coad(8))` and `idx(2,cof(3))`.

**External legs and samples.** `external_momenta` keys are momentum-basis
positions `i` of `P(i)`; `external_polarizations` and `external_spinors` keys are
physical leg indices. FeynKit lists external edges in leg order, so both
numberings coincide; `external_legs` reports `leg` and `momentum_key` for each
leg, and preparation warns if a graph ever differs. Each sample may bind:

- `external_polarizations`: contravariant components of `epsilon` or
  `epsilonbar`, for vector legs;
- `external_spinors`: four components of the spin-1/2 leg's carrier as it enters
  the amplitude, columns for `u` and `v`, rows for `ubar` and `vbar` (no
  conjugation is applied). Open fermion lines, with any number of Dirac
  matrices and Lorentz indices shared with other lines or vectors, are
  expanded in the chiral (HELAS) basis: `gamma^0 = [[0,1],[1,0]]`,
  `gamma^k = [[0,sigma^k],[-sigma^k,0]]`, `gamma^5 = diag(-1,-1,1,1)`. Unit
  spinors give the amputated Dirac matrix of the amplitude element by element.
  Explicit spinors need `--r2 off` (see above); alternatively, close each
  fermion line with a spin-trace `projector`.

`spinor_basis` and `polarization_basis` list the bound carriers. Anything the
numerator still needs (spinors, polarizations, color invariants) is reported by
`prepare`: an error when samples are given, a warning and the `unbound` list
otherwise.

**Levi-Civita tensors.** Idenso evaluates `gamma5` traces, from chiral
couplings such as the `Z`'s, with `epsilon` tensors normalized by
`Tr[gamma5 gamma^mu gamma^nu gamma^rho gamma^sigma] = 4 epsilon^{mu nu rho sigma}`.
With `gamma5 = i gamma^0 gamma^1 gamma^2 gamma^3`, the
`diag(-1,-1,1,1)` above, this is `epsilon^{0123} = -i`
(`idenso::epsilon::contract_epsilon4`), so a fully contracted tensor becomes
`epsilon(a,b,c,d) = i det[a; b; c; d]` in contravariant components once every
external polarization is bound. The loop momentum enters with its four
components only. A tensor whose index joins an open fermion line is summed
over components with the line. UFO `Epsilon` vertices, the real tensor with
`Epsilon^{0123} = -1` of ALOHA, are lowered by FeynKit as `-i epsilon`.
Rational terms of axial numerators depend on the `gamma5` scheme and are not
included (`--r2 fdh` rejects them). `tests/graphs` checks the convention
against a hand-written trace of explicit matrices (`pentagon-axial-trace.*`)
and reduces `H -> Z a` through a top loop (`hza-top.*`) in both engines.

**Widths.** Propagator widths are ignored by default, with a warning naming
every nonzero width. `--complex-mass-propagators` uses `m^2 - i*m*width` in every
propagator denominator, loop and tree alike; numerators and couplings keep
their real masses, so this is not the complete complex-mass scheme.
`propagator_widths` records the mode and the widths.

**Rejected inputs.** `reduce` requires at least one sample, and disconnected
graphs are rejected. A sample whose propagators coincide numerically (for
example a vanishing external momentum) fails only its lane, with status
`repeated_propagator` in both engines.
Symbolica's warnings are written to standard error, so standard output stays
valid JSON.

## Reduction engines

`reduce` and `bench` accept `--engine generic|fast`. The default `generic`
engine carries every precision (`f64`, `double-float`, `float`), both box
mu-extraction modes and the optional native reduction kernels. Both engines
reduce numerator rank up to n+2, one beyond original Ninja; see
[rank n+2](../../docs/rank-n2.md).

Both engines, at every rank up to n+2, apply Ninja's stability checks
precision-aware: a Gram determinant, uncut denominator or Laurent leading
coefficient whose magnitude is at most `2^10 ε` (Ninja's `REAL_EPS`) times the
magnitudes of the terms it is computed from is rounding noise of an exact zero,
and the lane reports `cut_failed` in every precision (`SingularParameter` for
the generic engine), where an exact-zero test accepted or rejected it depending
on the carrier. `reduce --stability-test` and `case-check --stability-test`
(generic engine) also report `unstable` lanes whose cut contributions cancel so
much that fewer than half of the carrier's bits survive (the summed
contribution magnitudes exceed `2^(p/2)` times the result). It is opt-in like
Ninja's accuracy tests: an exactly vanishing amplitude (a Ward identity, a zero
helicity amplitude) is rounding noise of its cut contributions and fails it in
every precision.

Whether or not the flag is set, every successful generic `reduce` lane reports
`cancellation` (binary64 numbers):
- `contribution_moduli`: the moduli `|Re| + |Im|` of the cut contributions,
  summed per Laurent coefficient (finite, single pole, double pole);
- `ratio`: their largest over the result's largest modulus (null when the
  result vanishes), the factor by which cancellation amplifies rounding;
- `rounding_estimate`: `2^-p` times the moduli (`p` mantissa bits), the
  rounding error of summing the contributions alone; errors inside each
  contribution come on top. It underflows to 0 when it leaves the binary64
  range (e.g. `--bits` above 1074);
- `precision_bits`: `p`, so the estimate can always be recomputed.

The fast engine does not report it.

`--engine fast` runs `ninjars::fast`, a batched binary64 port of original
Ninja's `Amplitude::evaluate` (complex masses, box mu-expansion, Ninja's skip
rules and stability checks), with oneloopmaster's native binary64 masters
(`OneLoopNativeMasters`; no expression is interpreted at evaluation time):

```console
nix develop --command cargo run --release -p ninjars-cli -- reduce \
  examples/graphs/physical/gg-hh/top-box-2-pool.json --engine fast
nix develop --command cargo run --release -p ninjars-cli --features numerator-jit -- bench \
  examples/graphs/physical/gg-hhh/top-pentagon-26-pool.json --engine fast --numerator-evaluator symjit-o2
```

- Inputs are bound as for the generic engine, including a fixture's exact
  `s_matrix`; `mu^2` is the fixture's `scale_squared`. External kinematics
  must be real (complex shifts or S-matrix entries are rejected); squared
  masses may be complex.
- A numerator without explicit `mu^2` terms becomes original Ninja's dense
  tensor numerator, filled per batch through `TensorCoefficientPlan` from
  coefficient heads evaluated by `--numerator-evaluator` (Symbolica's
  interpreter, `symjit-o2` or `symjit-o2-factored`). A numerator with explicit
  `mu^2` terms (`--r2 fdh`, or a fixture's own `ninjars::mu2`) uses the same
  heads in `ninjars::fast::MuTensorNumerator`, `N = sum_k mu^(2k) T_k(q)`
  with `T_k` a rank-`(R-2k)` tensor, so it stays on the tensor path.
  `numerator_provider.interface` records `tensor` or `mu-tensor` (with
  `tensor.mu_powers`).
- `--cut-stop` is honoured. `--precision` other than `f64`, `--bits`,
  `--mu-extraction sampling`, `--full-box-coefficients`,
  `--mu-sampling-scale`, `--reduction-kernels` other than `symbolica` and
  `--native-block-rows` are rejected before preparation.
- Output keeps the generic JSON schema and records `"engine": "fast"`. Failed
  lanes report `cut_failed` (Ninja's stability checks; `cut_index` is null),
  `integration_failed` (a master failed) or `nonfinite_result`. With
  `--diagnostics`, `cut_counts` is null.

Rank n+2, which original Ninja does not support, runs in a separate fast
engine (`ninjars::fast` `engine_x2.rs`, chosen once at construction). It
leaves the rank <= n+1 path bitwise unchanged. Per cut and lane it picks the
bubble reference vector and tadpole basis that best condition the uncut
lines, so its cut-constructible/rational split need not match original
Ninja's.

On the synthetic and physical fixtures of `tests/fast_engine.rs` both engines
agree on the Laurent coefficients within the scaled 1e-9 criterion. The
cut-constructible/rational split depends on the cut basis: the fast engine
reproduces original Ninja's split (`validation/benchmark/ninja`), the generic
engine's may differ. Like original Ninja, the fast engine needs real cut
bases: points with purely spacelike cut planes, such as
`examples/graphs/triangle.json` and `pentagon-rank4.json`, fail their lanes
there and remain covered by the generic engine.

## Tensor-numerator bench cases and per-point preparation

`export-bench` writes the shared `ninjars-bench-case-v1` file consumed by the
original Ninja/MadLoop timing harnesses: per lane the propagator shifts `V_i`,
squared masses, `mu^2` (the fixture's `scale_squared`), the exact S matrix
`s_ij = (V_i - V_j)^2` (or `null` with `--s-matrix none`), and the
`binomial(rank+4,4)` coefficients of `N(q) = sum_k tensor[k] monomial_k(q)` in
original Ninja's `TensorNumerator`/MadLoop ordering
(`ninjars::fast::tensor_layout`). The complete numerator (graph factor included
once) is laid out by `ninjars_graph::TensorCoefficientPlan`; values are
evaluated from the exact fixture inputs at `--bits` (default 256) and rounded
once. A numerator with explicit `mu^2` dependence (`--r2 fdh`) has no
original-Ninja layout: it is written as `ninjars-bench-case-v2`, with
`mu_powers` `K`, `r2`, and per lane the blocks `T_0, ..., T_K` of
`N = sum_k mu^(2k) T_k(q)` concatenated in `tensor` (`T_0` is exactly the
four-dimensional v1 tensor). The C++ harnesses accept only v1; `case-bench`
and `numerator-bench --check-case` read both. Every lane is
checked against the existing `PolynomialNumerator` at random complex loop
momenta (and random `mu^2` for v2 cases; report on standard output or
`--report`):

```console
nix develop --command cargo run --profile perf -p ninjars-cli --features numerator-jit -- \
  export-bench examples/graphs/physical/gg-hhh/top-pentagon-26-pool.json \
  --name gg-hhh-top-pentagon-26 --output validation/benchmark/cases/gg-hhh-top-pentagon-26.json
```

`--select 2,0` exports a subset of fixture samples in the given order and
`--rank R` pads to a larger tensor rank. Exported cases use the prepared FeynKit
loop momentum and propagator order; `validation/benchmark/validate_cases.py`
reproduces `ninjars reduce` with original C++ Ninja on every exported lane.

`numerator-bench` builds a reusable `ninjars_graph::PhysicalTensorNumerator`
from a fixture's graph and a shared phase-space pool
(`validation/benchmark/points/*.json`, `ninjars-phase-space-pool-v1`), reports
every setup stage (graph import, Feynman rules, propagators, contraction, color,
polynomial, coefficient planning/factoring, tensor plan, evaluator construction,
SymJIT O2 compilation), optionally checks the prepared tensors and propagator
data against a bench case (`--check-case`), and times warm per-point
preparation from raw binary64 momenta, helicities and model rows at
`--widths` (median of `--blocks` blocks of at least `--min-block-ms`):

```console
taskset -c 201 ./target/perf/ninjars numerator-bench \
  examples/graphs/physical/gg-hhh/top-pentagon-26-pool.json \
  --points validation/benchmark/points/gg-hhh.json --feynman-rules \
  --check-case validation/benchmark/cases/gg-hhh-top-pentagon-26.json --widths 1,8,32
```

The pools, pool fixtures (`examples/graphs/physical/*/*-pool.json`) and cases
are regenerated by `cargo run --profile perf -p ninjars-cli --example
benchmark_pools -- .` followed by the `bench_case_command` recorded in each
pool. `validation/benchmark/numerator_prep_ninjars.py` and
`generation_ninjars.py` record the per-point and generation timings.

## End-to-end and bench-case timings

`pipeline-bench` times the binary64 pipeline of one physical graph over a
shared phase-space pool: `PhysicalTensorNumerator` preparation (wavefunctions,
couplings, SymJIT O2 coefficient heads, Ninja tensor and propagator data), the
pure `ninjars::fast` reduction (numerator outputs and master values replayed
at O(1) cost), the tensor-numerator expansions, oneloopmaster's native masters
(recorded calls replayed) and the uninstrumented total, at each `--widths`
batch width (build with `--features numerator-jit`). Each width also records
`lane_results` (Laurent, cut-constructible and rational parts) of the distinct
pool points. With `--r2 fdh` the tensor carries `mu^2` blocks and the
reduction runs `MuTensorNumerator` on the same fast path; `--r2 off` is
Ninja's `TensorNumerator` pipeline, bit for bit. `--reference` checks the
integrated results against original-Ninja results for the same pool points
(which have no R2):

```console
taskset -c 201 ./target/perf/ninjars pipeline-bench \
  examples/graphs/physical/gg-hhh/top-pentagon-26-pool.json \
  --points validation/benchmark/points/gg-hhh.json --feynman-rules --factor \
  --widths 1,32 --reference validation/benchmark/ninja/gg-hhh-top-pentagon-26.json
```

`case-bench` performs the same reduction, tensor-numerator and master
breakdown on `ninjars-bench-case-v1` files, i.e. exactly the propagators and
tensor coefficients given to original Ninja (and on `-v2` cases with `mu^2`
blocks through `MuTensorNumerator`); `--reference-dir
validation/benchmark/ninja` checks each case's results:

```console
taskset -c 201 ./target/perf/ninjars case-bench validation/benchmark/cases/*.json \
  --widths 1,32 --reference-dir validation/benchmark/ninja
```

Both report medians of `--blocks` blocks of at least `--min-block-ms`;
`validation/benchmark/collect.py` gathers them for the
[benchmark report](../../docs/benchmark/benchmark.typ). `case-bench` also
times rank n+2 cases (`validation/benchmark/synthetic-rank-n2/`,
`validation/rank-n2/cases/`), for which original Ninja has no reference.
Two options serve the rank n+2 comparison with COLLIER
([`validation/benchmark/collier`](../../validation/benchmark/collier/README.md)):

- `--engines generic` (or `fast,generic`; the default is `fast`) adds the same
  breakdown for the generic engine in binary64 (`SymbolicaAmplitudeKernels`,
  native masters), reported per width as `generic_per_point_timing`:
  `reduction` (recorded numerator outputs and master values replayed),
  `reduction_and_numerator` (live numerator, replayed masters), `masters`
  (recorded master requests replayed) and `total`. Its numerator is
  `case-check`'s numerically expanded dense tensor, not the production
  generic numerator that a graph prepares symbolically (`ninjars bench`).
- `--total-calls K` makes the fast engine run fixed work for `perf stat`
  instead of timing: per width, one recording pass (with the reference check)
  and then exactly `K` live totals. Two runs with different `K` give the
  instructions per call by difference
  (`validation/benchmark/collier/count_instructions.py`). The generic engine
  is still timed as usual.

```console
taskset -c 201 ./target/perf/ninjars case-bench validation/rank-n2/cases/rn2-int-complex-n5-r7.json \
  --engines fast,generic --widths 1,32
perf stat -e instructions:u taskset -c 201 ./target/perf/ninjars case-bench \
  validation/rank-n2/cases/rn2-int-complex-n5-r7.json --widths 32 --total-calls 2000
```

## Checking cases against references (`case-check`)

`case-check` reduces `ninjars-bench-case-v1` and `-v2` files and judges every
lane against a reference file. It is the checker of the rank n+2 validation
suite ([rank n+2](../../docs/rank-n2.md),
[`validation/rank-n2`](../../validation/rank-n2/README.md)).

```console
R=validation/rank-n2
nix develop --command cargo run --release -p ninjars-cli -- case-check \
  $R/cases/*.json --engines fast,generic --precision f64,float:256 \
  --reference-dir $R/reference --known-failures $R/known-failures.json \
  --twins $R/summary/controls.json --twins $R/summary/synthetic-twins.json \
  --twins $R/summary/n7-controls.json -o report.json
```

**Engines.**
- `--engines fast` runs the fast engine with `TensorNumerator`
  (`MuTensorNumerator` for v2 cases) and native masters. Lanes are batched as
  `case-bench` batches them (`--width`), and repeated lanes must be bitwise
  identical. On rank <= n+1 cases the report includes `case-bench`'s own
  reference check of the same batch.
- `--engines generic` runs the generic engine at each `--precision`:
  - `f64` with native masters;
  - `double` (DoubleFloat) and `float:<bits>` with oneloopmaster's exact
    expressions, and B1111 from its native family.

  The dense tensor is expanded numerically, so a dense rank-9 numerator needs
  no Symbolica preparation.
- `--declare-rank R` (or `n+K`) runs every numerator declared at a higher rank
  with zero top coefficients.
- `--stability-test` marks generic-engine lanes whose cut contributions
  cancel below half the carrier's bits as `Unstable`.

**References and verdicts.** `--reference-dir` holds `<case>.json`: either
the high-precision `ninjars-reference-collier-v1` files, with tier, uncertainty
and condition, or original-Ninja results, judged at `--ninja-tolerance`
(default 1e-8). Each lane gets one verdict:
- `pass` or `fail`;
- `report-only`, for one of these reasons:
  - the lane is tier C;
  - at high precision (`float:<bits>` with at least 160 bits), the lane is
    tier B and its uncertainty comes from the generic engine's own rank twins;
  - at lower precision (`f64`, `double`, `float` below 160 bits), the
    condition or the cancellation of the new rank n+2 terms is above 1e6;
  - the tolerance is too wide to resolve the lane's new rank n+2 terms;
- `known-failure` and `stale-known-failure`, from `--known-failures` (a
  `ninjars-known-failures-v1` file):
  - `lanes`: upstream bugs, judged against COLLIER's value. An entry may name
    the `engines` and `precisions` it applies to; elsewhere the lane is judged
    as any other, so a failure of one precision does not fail the others;
  - `inaccuracies`: one engine at one fixed precision, within a `limit`;
  - `ninja_unreliable` (no verdict of its own): kinematics whose
    original-Ninja errors are no binary64 budget, with corrected `errors` per
    rank excess or none;
- `unchecked`: no reference lane;
- `error`: the engine rejects the case.

The exit status is non-zero if any lane fails or is a stale known failure,
unless `--no-fail` is given.

**Binary64 budget.** The binary64 tolerance builds on original Ninja's error
on the lane. For lanes without their own Ninja result, `--twins` supplies it
from rank twins on the same kinematics. `--fast-ninja-factor` (30),
`--generic-ninja-factor` (100) and `--binary64-factor` (1e-14) set the budget;
the full rule is in [rank n+2](../../docs/rank-n2.md#83-ninjars-case-check-on-the-whole-suite).

**Runner mode.** `--bits B --out-dir DIR` writes `DIR/<case>.json` with
45-digit `lanes[].laurent_decimal`. This is the `GENERIC_RUNNER` interface of
`validation/rank-n2/harness/run_suite.sh`.
