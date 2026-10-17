# Supported native batch policy validation

This records correctness and compatibility of the explicit upstream execution
policy and NinjaRS amplitude-factory opt-in. It is separate from the earlier
frozen microbenchmark. No supported-path timing benchmark was executed.

Source commits:

- oneloopmaster: `c9dd82fd1a6e3fc638b01df95bd0cf09ceda3460`, based on `53106429b849e5bcf843f111429be76227b4e0ce`.
- NinjaRS: `9013b787e57c70d3d9622af83333d599cbce0dfb`, based on `dbaaadff45712f4df608ec197a9c5be2e138fee1`.

The manifest retains changed-source hashes, all 31 unchanged generated-source
hashes, dependency revisions/cleanliness, four exact test-executable hashes, and
20 original build/test/check logs. Canonical and frozen dependency checkouts and
aliases were not changed. The isolated owner commit awaits review before direct
publication to upstream main.

## Results

All 58 distinct focused tests pass. The Dual factory test was additionally run
alone and then included in the native-rank suite, so it is counted once below.

| Suite | Passed | Evidence |
| --- | ---: | --- |
| Owner batch workspace/policy | 6 | `owner-batch.log` |
| Owner emitter unit regressions | 14 | `owner-codegen.log` |
| Owner native primitives | 12 | `upstream-primitives.log` |
| Consumer probe and supported execution controls | 14 | `consumer-controls.log` |
| Consumer native rank specialization, including executed Dual factory | 11 | `native-rank.log` |
| Owner emitted batch integration | 1 | `upstream-batch.log` |

The consumer timing test stays ignored. Supported execution compares 421,872
complex components in each of f64, DoubleFloat, and 192-bit Float, independently
of the retained old test-wrapper checks. It also covers mixed input precision,
signed zero and exceptional classification, empty/single/natural tails, lazy
shared bodies, full Dual rows, grow/tighten/shrink/bound changes, all-buffer
validation before mutation, reducer statuses/residues/retry, and builder order.
The executed n5/r4 Dual factory test observes actual generic and guarded alternate
workspace strides while comparing asymmetric higher/mixed components exactly.
Owner callback controls verify Whole calls once, block order, exact error
propagation, and writes in the failing block without rollback.

Strict Clippy (`-D warnings`), formatting, whitespace checks, and a build with
default/native/codegen features disabled pass. Existing dependency warnings are
retained. The first external-package-only integration build was rejected by Cargo
feature selection before compilation; selecting owner and consumer together
succeeded. Both attempts are archived. No numerical test failed.

The emitted-batch integration compiles tiny generated fixtures and checks ordinary
and six-component Dual arithmetic, input precision, lazy shared helpers, primitive
visitation order, and allocation contracts. Its existing emission/rustc elapsed
diagnostics are incidental correctness-run metadata, not a timing comparison.

## Reproduction

Run from the isolated NinjaRS consumer with its oneloopmaster alias pointing to
the owner source commit above and Symbolica/FeynKit revisions in `manifest.json`.
The local environment files supply the toolchain, native tool paths, and license;
they are not part of the evidence or committed source.

```sh
source /tmp/ninjars-build-env-paths
export CC CXX PKG_CONFIG CARGO_BUILD_JOBS=4
source /tmp/ninjars-symbolica-env
export RUST_MIN_STACK=134217728
ulimit -n 65536

cargo test -p oneloop -p ninjars-symbolica \
  --features ninjars-symbolica/native-kernels,oneloop/rust-codegen \
  --lib --no-run --message-format=json
cargo test -p oneloop -p ninjars-symbolica \
  --features ninjars-symbolica/native-kernels,oneloop/rust-codegen \
  --test rust_batch_codegen --test native_primitives --no-run --message-format=json

cargo clippy -p oneloop -p ninjars-symbolica \
  --features ninjars-symbolica/native-kernels,oneloop/rust-codegen \
  --lib --tests -- -D warnings
cargo check -p oneloop -p ninjars-symbolica --lib --no-default-features
cargo fmt -p ninjars-symbolica --check
```

Use the executable paths returned by Cargo; this run's exact paths and hashes are
in the manifest. Invoke each executable separately under the numerical lock,
releasing it between commands:

```sh
flock /tmp/ninjars-symbolica-tests.lock "$owner_lib" native::batch:: --test-threads=1 --nocapture
flock /tmp/ninjars-symbolica-tests.lock "$owner_lib" rust_codegen:: --test-threads=1 --nocapture
flock /tmp/ninjars-symbolica-tests.lock "$native_primitives" --test-threads=1 --nocapture
flock /tmp/ninjars-symbolica-tests.lock "$consumer_lib" kernel::blocking_probe --test-threads=1 --nocapture
flock /tmp/ninjars-symbolica-tests.lock "$consumer_lib" explicit_batch_policy_reaches_dual_factory --test-threads=1 --nocapture
flock /tmp/ninjars-symbolica-tests.lock "$consumer_lib" rank_specialization::tests::native --test-threads=1 --nocapture
flock /tmp/ninjars-symbolica-tests.lock "$rust_batch_codegen" --test-threads=1 --nocapture
```

Owner formatting used explicit changed files to avoid editing shared dependencies:

```sh
rustfmt --edition 2024 --config skip_children=true --check src/native/batch.rs src/rust_codegen.rs
git diff --check
```

`Whole` remains the default direct generated call. `FixedRows(32)` remains an
explicit experiment choice. No complete-reduction speedup or acceptance claim is
made by these checks.
