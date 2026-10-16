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
