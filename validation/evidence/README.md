# Recorded validation evidence

`generated-callbacks.json` records the five integrated original-Ninja controls
run on 2026-10-16. Reproduce them using `validation/reference/test_generated_runner.py`
after building the reference library and the `generate_reference` Rust example.
Source dependency revisions and tested upstream patches are pinned in
`dependencies/manifest.json`.

These checks validate the Symbolica-generated callbacks through original Ninja
and OneLOop. They do not establish integrated Rust-reducer parity. Higher-rank
linearity checks use separate generated and tensor terms, not an independent
closed-form higher-rank integral. Recorded timings are diagnostic and are not
performance acceptance evidence.

`prepared-graphs.json` records four DOT-to-reference comparisons: three bubble
samples and one triangle. Reproduce them with
`validation/reference/test_prepared_runner.py --cli target/debug/ninjars` after
building the CLI with `reference-codegen`. The independent tensor interface
checks the mu-free polynomial, and analytic rational controls check the explicit
mu terms. Exact prepared expressions are retained in the export bundles; the
reference's binary64 conversion is explicit. This is preparation/reference
validation and does not add an integrated Rust-reducer claim.

## Laurent-stage evidence

`laurent-stage.json` records the 66-test workspace milestone, independent
subtraction comparisons, upstream dependency regressions and reproduction
commands. It validates triple-cut residues and constituent kernels; it is not
integrated amplitude or performance acceptance evidence. The corresponding
scope and remaining work are recorded in `references/PLAN.md`.

## Integrated and high-cut evidence

`integrated-stage.json` records the tadpole/bubble/triangle milestone and its
103-test workspace validation. `box-masters.json`, `box-full-reconstruction.json`
and `box-cli.json` record independent D0 controls, two original Ninja defects,
and 36 prepared-box comparisons. Reproduction commands and reference-domain
restrictions are in [the box audit](../../docs/box-reference-audit.md).

`high-reduction.json` contains 18 original Ninja controls covering four through
six propagators, ranks 0, 2 and n+1, and real/complex masses. Regenerate it with
`python3 validation/reference/test_high_reduction.py` after building the
`generate_reference` example. Original nonfinite responses at two initial
kinematic configurations are retained separately as `reference_exceptions`.

`high-stage.json` records 36 full Rust amplitude comparisons against those
controls in mu-expansion and finite-mu sampling modes, using sampling scales
comparable to the kinematic invariants. Run the `amplitude_high` Rust test with
`--nocapture --test-threads=1` and `RUST_MIN_STACK=134217728`; its
`NINJARS_HIGH_REDUCTION` line contains the machine-readable report. The maximum
scaled difference is 2.58e-11 at a fixed 1e-9 gate. An observed small-scale
cancellation failure is retained explicitly; automatic scale selection and
precision rescue remain outstanding. These records establish neither physical
validation nor performance acceptance.

`controls-stage.json` records the subsequent 155-test milestone: automatic
sampling scales, cut stopping, supplied S-matrices, the comparison CLI and the
first physical-preparation controls. Eight complete CLI comparisons cover
bubbles/triangles/boxes and DoubleFloat with supplied invariants/cut stopping.
Reproduce the full workspace command stored in the JSON with `--nocapture`,
then run `python3 validation/record_controls.py TEST_LOG`. The automatic-scale
regression retains the previous scale-three failure and its corrected scale-247
result. This record does not establish physical integrated or performance
acceptance. The comparison test removes temporary build artifacts after success;
standalone `ninjars compare` retains its artifacts for inspection.

## Physical numerators and generated Rust kernels

`physical-hgg-madloop.json` records both individually exported top-quark H→gg
orientations, three polarization states, and fixed double/quadruple CutTools
modes without R2 or UV. Ninety complex loop-momentum checks match independently
prepared FeynKit numerators. See [physical preparation](../../docs/physical-preparation.md)
for routing, color/sign, normalization and reproduction commands. Quadruple
execution promotes binary64 physical inputs; it is not an independent
high-precision model-input oracle.

`physical-hgg-cli.json` records the full 36-row integration matrix. Its four
f64 longitudinal pole failures remain explicit at the unchanged 1e-9 gate.
Physical helicities and all DoubleFloat/Float256 rows pass. Binary, source and
fixture hashes accompany the exact run commands. This is partial physical
validation and does not establish full project acceptance.

`rust-kernel-codegen.json` records 36 compiled-versus-interpreted checks for
three existing exact plans, in scalar and higher/mixed Dual forms, across f64,
DoubleFloat and Float256. Every component is checked; generated-source hashes,
plan parameters and toolchain details are retained. Reproduce it with
`python3 validation/rust_codegen/run.py --evidence validation/evidence/rust-kernel-codegen.json`.
Execution is a scalar row loop used as a control, not production batching or
performance evidence.

`validation/record_checkpoint.py TEST_LOG --output REPORT --evidence RELATED_JSON`
records a successful all-feature workspace run alongside source/dependency
hashes and toolchain/host metadata. Invoke it after sources have been frozen and
tests completed. Related evidence can explicitly report a failed numerical
gate; passing regression tests do not override that result.

`physical-stage.json` records the 165-test workspace checkpoint with all reference
controls enabled. It links the physical and generated-kernel records by hash,
including the explicitly failed longitudinal matrix. Strict Clippy and
formatting results and pending upstream work are recorded in the plan.
