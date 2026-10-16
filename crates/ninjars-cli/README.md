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
