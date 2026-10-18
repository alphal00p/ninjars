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

## Reduction engines

`reduce` and `bench` accept `--engine generic|fast`. The default `generic`
engine carries every precision (`f64`, `double-float`, `float`), both box
mu-extraction modes and the optional native reduction kernels.

Both engines apply Ninja's stability checks precision-aware: a Gram
determinant, uncut denominator or Laurent leading coefficient whose magnitude
is at most `2^10 ε` (Ninja's `REAL_EPS`) times the magnitudes of the terms it is
computed from is rounding noise of an exact zero, and the lane reports
`cut_failed` in every precision (`SingularParameter` for the generic engine),
where an exact-zero test accepted or rejected it depending on the carrier.
`reduce --stability-test` (generic engine) also reports `unstable` lanes whose
cut contributions cancel so much that fewer than half of the carrier's bits
survive (the summed contribution magnitudes exceed `2^(p/2)` times the result).
It is opt-in like Ninja's accuracy tests: an exactly vanishing amplitude (a
Ward identity, a zero helicity amplitude) is rounding noise of its cut
contributions and fails it in every precision.

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
[benchmark report](../../docs/benchmark/benchmark.typ).
