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
