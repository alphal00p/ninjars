# Combined native execution policy validation

Main source `a6739cab16754750b72605c90be86ee7dfd19717` with published
oneloopmaster `c9dd82fd1a6e3fc638b01df95bd0cf09ceda3460` passes 49 tests,
with four explicit probes intentionally ignored. Strict Clippy, feature-off
compilation, formatting and diff checks also pass.

`manifest.json` records commands, source/dependency identities, executable
hashes, retained logs and independent verification of the two integrated
candidates. Numerical tests ran serially under the Symbolica test lock; builds
ran outside it. Executable paths identify the tested artifacts in the local
build cache and are not distributed binaries.

These are correctness and compatibility checks. They contain no new timing
samples and do not establish performance acceptance. Whole remains the
default native execution policy; the fixed-row option is currently a library
API. Rank selectors explicitly support n5/r4.
