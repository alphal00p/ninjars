# Ordinary Laurent catalog inventory

The existing `LaurentDivision` and `RustKernelExport` APIs can produce a bounded
catalog for up to six propagators, rank at most n+1, and cuts of size one to
three. The inventory example uses only these exact Symbolica plans and
oneloopmaster's emitter. It installs no backend and introduces no algebra.

```sh
cargo run --locked -p ninjars-symbolica --features rust-codegen --example inventory_laurent_catalog -- target/laurent-catalog-inventory.json
```

The recorded run produces 58 nonempty plans and skips 31 empty quotients.
Their ordinary instruction-major Rust bodies total 864,498 bytes. Comparing
fresh per-plan constant tables and complete generated source gives 45 distinct
groups (814,779 bytes including their tables). This byte equality is only an
inventory observation; runtime selection must still validate cut identity and
the ordered numerator, denominator and quotient layouts.

The largest plan, n6/rank7/cut2, has 712 instructions and 135,243 source bytes.
The current n5/rank4 physical case uses only 5, 32 and 45 instructions for its
single, double and triple-cut quotients respectively. The older intrusive
profile attributes less time to its two main division plans than to the
tadpole geometry probe; generated size alone does not establish priority or
speedup.

The [machine-readable inventory](../validation/evidence/laurent-catalog-inventory.json)
retains all layouts, sizes, exact duplicate groups, dependency revisions and
generator identity. Code generation, strict Clippy and formatting pass. This
does not compile or execute the generated catalog, cover Dual exports, or
establish numerical/performance acceptance. Native selection, fallbacks,
coverage reporting and full precision/batch controls remain future work.
