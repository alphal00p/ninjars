# Original Ninja reference runner

Build with `nix develop --command python3 validation/reference/build.py`.
Sources are copied into `target/reference`; reference checkouts stay unchanged.
The runner links the original Ninja against the unmodified OneLOop 3.4 source
distributed with MadGraph. A module-name alias is the only compatibility shim.

On a fresh checkout, bootstrap Ninja with `python3 tools/bootstrap.py` and place
MadGraph 3.7.0 at `references/MG5_aMC_v3_7_0`. Its official Git tag is `v3.7.0`
(commit `adf3d62d096abc1f1512c2101b40218f7c529d4f`) in
`https://github.com/mg5amcnlo/mg5amcnlo.git`. The harness records the SHA-256 of
the actual bundled OneLOop source and compiler identity in `revisions.json`.
Run `python3 validation/reference/test_runner.py` after the build.

The runner accepts successive whitespace-delimited requests on standard input:

1. `propagator_count rank repetitions renormalization_scale_squared`.
2. One row per propagator: `V0 V1 V2 V3 Re(mass_squared) Im(mass_squared)`.
3. `binomial(rank+4,4)` rows: `Re(coefficient) Im(coefficient)`.

Tensor coefficients follow original Ninja ordering. Homogeneous ranks are
concatenated; within a rank enumerate powers of q3, then q2, then q1 in ascending
order, with the remaining power assigned to q0. There are no multinomial or
metric factors: coefficients multiply component monomials literally.
The current tensor protocol does not represent explicit mu-squared monomials.
Such inputs require the Symbolica-generated callback reference adapter.

Each response is a JSON object containing status, epsilon coefficients in
finite/single/double-pole order, rational and cut-constructible finite parts,
and steady-state evaluation timing after one warm-up. Setup and IPC are excluded.
The initial OneLOop license banner can precede the first JSON response; consumers
must select JSON lines. Timing uses repeated kinematics and the original cache.
This alone is not a changing-phase-space benchmark or a performance acceptance
claim. Nonfinite numerical values are emitted as JSON null and must be rejected.

To use a Symbolica-generated callback header (including explicit mu-squared
terms), build with `--numerator-header /path/to/numerator.hh`. The header exports
`ninjars_reference::parameter_count`, `rank`, `propagator_count`, and
`GeneratedNumerator(const ninja::Complex*)`. In this mode, replace the tensor
coefficient rows with `parameter_count` complex parameter rows in exporter order.
The topology and rank must match the compiled header. The resulting executable
is `target/reference/ninja-generated-reference`; `--output` overrides this path.

After a complete reference build, `--runner-only` recompiles the runner against
those same libraries without rebuilding them. Per-executable JSON metadata
records the inherited reference revisions, linked library hashes, and generated
header hash. It does not relabel old libraries with newer source revisions.

Build the Symbolica exporter example with
`cargo build -p ninjars-symbolica --features reference-codegen --example generate_reference`,
then run `python3 validation/reference/test_generated_runner.py --generator target/debug/examples/generate_reference`
inside the development environment. Set the generator path to Cargo's configured
target directory if customized. Run Symbolica processes serially when using its
single-thread evaluation license.

The generated-callback suite checks triangle mu-squared and box mu-to-the-fourth
rational controls, agreement with the independent original tensor interface,
changing complex parameters in a streamed process, and rank-n-plus-one linearity.
The higher-rank linearity check is not an independent analytic master comparison.
Machine-readable evidence is written to
`target/reference/generated-callback-validation.json`. Generated callbacks currently
compute the maximum Laurent depth before returning the requested prefix; their
timings are correctness-harness diagnostics, excluded from performance acceptance.

A known exceptional reference point is preserved in
`exceptional-spacelike-triangle.inputs`: original Ninja at the recorded revision
returns status zero with nonfinite rank-two results for orthogonal purely spatial
triangle shifts. Its real-valued basis square root has a negative discriminant;
a rank-zero constant numerator can hide the invalid spurious basis. The runner
emits these nonfinite values as JSON null. Consumers must reject null values even
when the reported status is zero. Integrated callback checks therefore use
separate stable timelike points. This does not establish support for these
exceptional points in NinjaRS; that requires complex bases or explicit instability
reporting from the reduction engine.

The CLI can export the same prepared DOT numerator with its graph factor included
once: build `ninjars-cli` with `--features reference-codegen`, then use
`prepare examples/graphs/bubble.json --reference-dir target/reference/bubble-export`.
See `crates/ninjars-cli/README.md` for the bundle contract and commands.
`test_prepared_runner.py --cli target/debug/ninjars` checks three bubble samples
and one timelike triangle through both original callback interfaces. The
mu-dependent analytic controls are `(3*(m0²+m1²)-p²)/6` for the bubble's mu² term,
and `-(V0[0]+V1[0]+V2[0])/6` for the triangle's mu²*q0 term. Fixture masses are one.
Results, tensor controls and reference artifact hashes are written to
`target/reference/prepared-graph-validation.json`; a recorded passing run is
committed in `validation/evidence/prepared-graphs.json`.
