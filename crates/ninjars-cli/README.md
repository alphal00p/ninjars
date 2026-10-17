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
once. Numerators with explicit `mu^2` dependence are rejected. Every lane is
checked against the existing `PolynomialNumerator` at random complex loop
momenta (report on standard output or `--report`):

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
