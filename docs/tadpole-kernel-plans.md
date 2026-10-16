# Tadpole projection and reference-probe plans

This is an extraction and code-generation stage. Production still uses the
Symbolica interpreter for these plans. The existing native catalog selection
and CLI are unchanged; no speedup or performance acceptance is claimed.

`TadpoleSourceProjection::new(3)` exposes the existing 24→6 triangle-parent
projection. `new(2)` exposes the 28→9 bubble-parent projection. Both retain the
parent's own shift and ordered basis axes, independently of the tadpole origin.
`TadpoleGeometryProbe::new()` exposes the existing 8→28 complementary-shift
probe for all six exact Weyl reference bases. Its output does not contain the
finished tadpole geometry. The separate 5→30 finish, candidate scoring and
failure policy are unchanged.

These expressions were moved from their owning `tadpole_kernels` and `cuts`
constructors without a formula change. FeynKit's `FourMomentum<Atom>` and
public Weyl methods remain the source of contractions and basis construction.
Symbolica's `Atom::evaluator_multiple`, `Dualizer` and batch evaluator prepare
and execute the expressions; oneloopmaster's existing `RustEmitter` consumes
the resulting instruction graph. No polynomial, differentiation, numerical
contraction, or instruction representation is implemented here.

The generated files use a separate `tadpole_constants.rs` pool initialized at
working precision. The ordinary generic bodies cover f64, DoubleFloat and
Float; separately generated bodies flatten the ordered Taylor shape
`[[0,0],[1,0],[0,1],[2,0],[1,1]]`. This fixed generated shape is not an automatic
runtime selector. The interpreted adapters continue accepting supported static
and dynamic shapes through their existing complete-shape checks.

Regenerate all six modules and their constant table with:

```sh
cargo run --locked -p ninjars-symbolica --features rust-codegen \
  --example generate_native_tadpoles
```

The example accepts an optional output directory for independent regeneration.
`tadpole_metadata.json` records dimensions, instruction counts and source sizes.
The modules are compiled by focused tests; they are not installed in production
executors at this stage.

Validation compares projection values to shared momentum arithmetic at complex
parent origins, checks all six reference bases against the public Weyl API,
and checks every higher/mixed component. Generated scalar and batch execution
are compared to the interpreter for f64, DoubleFloat and Float, including
empty/single/tail/permuted batches and a shared workspace across all layouts.
Malformed dimensions leave outputs unchanged; the interpreted Dual wrapper
also rejects malformed or reordered shapes. Float controls retain per-input
tracked precision. Regeneration checks every emitted byte and constant index.

The existing tadpole and cut-geometry suites cover downstream parent
subtraction, lane isolation, on-shell identities and shared basis conventions.
