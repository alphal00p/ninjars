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
`target/reference/generated-callback-validation.json`. Generated callbacks now
dispatch to a separate Symbolica evaluator for each requested Laurent depth.
The legacy runner's timings remain correctness-harness diagnostics, excluded
from performance acceptance because it uses the default cached library build.

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

Box controls and two confirmed original-reference limitations are described in
[`docs/box-reference-audit.md`](../../docs/box-reference-audit.md). Run
`python3 validation/reference/test_box_masters.py` for six independent D0
controls, an unsupported complex-mass reference point, and a cache-key probe.
Run `python3 validation/reference/test_box_full.py` after building the generator
example to reproduce the higher-rank full-box c3 discrepancy. The original
reference cache must be cleared between changing phase-space points: its D0 key
omits s14. These probes do not modify the original production implementation.
The streaming tensor/generated-callback runner clears the original cache before
each new request. Its warm-up and timed repetitions reuse identical kinematics.

`python3 validation/reference/test_box_cli.py --cli target/debug/ninjars` rebuilds
the prepared box callback bundle and compares three graph samples with original
Ninja, an independent tensor-plus-analytic control, and 12 Rust CLI precision,
extraction-mode, and batch-size configurations. Build the CLI with
`--features reference-codegen` and serialize this script with other Symbolica
runtime processes. Its committed evidence is `validation/evidence/box-cli.json`.

## Persistent uncached benchmark control

The separate `benchmark_runner.cc` consumes one common parameterized numerator
and a preloaded pool of numerical inputs. It is validation tooling; no C++ code
is linked into NinjaRS. First export the shared template and the exact binary64
buffers used by the Rust runtime:

```sh
ninjars prepare examples/graphs/bubble.json \
  --reference-bench-dir target/reference-bench/pools/bubble
python3 validation/reference/build.py --benchmark --disable-avholo-cache \
  --numerator-header target/reference-bench/pools/bubble/numerator.hh
python3 validation/reference/test_benchmark_runner.py \
  --bundle target/reference-bench/pools/bubble
```

These commands require the repository development environment; the CLI needs
the `reference-codegen` feature. The dedicated default build directory is
`target/reference-uncached-v2`. `--disable-avholo-cache` passes the literal upstream
configure option `--disable-avholo_cache`; `--benchmark` also requires this
policy. The build checks `config.h` rather than trusting the requested flag.
Reusing libraries with `--runner-only --benchmark` requires matching recorded
configuration and library hashes. A cached or altered library build is rejected.
Copied Ninja sources and headers, the copied OneLOop sources, and generated
headers under `ninja-build` are recorded during library compilation and checked
before and after runner compilation. The exact input-file set is checked, so
added, removed and modified headers are detected. Older records without these
hashes are rejected for benchmark reuse; they must be rebuilt in a fresh
directory rather than retrospectively relabeled.
The runner additionally refuses to compile with the cache macro defined. Ninja
and OneLOop sources remain unchanged. Compiler identities, exact build commands,
configuration/source/library/generated-header/binary hashes accompany each
executable; original higher-rank support must be enabled. No fast-math option is
added.

The full whitespace-delimited protocol is:

1. `NINJARS_BENCH_V1`.
2. `propagator_count rank parameter_count pool_size`, matching the compiled header.
3. For each pool row, `renormalization_scale_squared has_s_matrix`, then one
   `V0 V1 V2 V3 Re(mass_squared) Im(mass_squared)` row per propagator, followed by
   `parameter_count` complex pairs. When `has_s_matrix` is one, append the
   `n*n` real matrix entries in row-major order. Matrices must be symmetric with
   zero diagonal. All numerical values must be finite, and scales positive.
4. `job_count`, followed by that many rows of
   `job_id workload points warmup rounds iterations`. IDs must be distinct;
   workload zero repeats source row zero, and one cycles through source rows in
   fixture order. Every count except the workload and ID must be positive.
   Changing workloads require at least two source rows and two points.

The exported `pool.inputs` ends after step three. An orchestrator appends the job
section before sending the complete input to the process. For a short protocol
control, append `2`, `0 0 5 1 2 1`, and `1 1 7 1 2 1`, each on its own line.
The runner rejects trailing data, negative/overflowing counts, unsupported
options, and malformed or mismatched pools before evaluation. Input is parsed
completely before any timing.

`--cut-stop 0..6` and `--mu-sampling` match original Ninja's public controls.
Coefficient mode is integrated; the original full-box mode cannot be enabled
independently of extra reconstruction or printing through the public API.
There is no invented manual sampling-scale switch. Original Ninja selects a
scale independently for each box from real mass-squared parts and absolute
S-matrix entries, with its `INFRARED_EPS` fallback. NinjaRS's automatic scale
policy differs; comparison reports must retain both policies.

The runner retains one amplitude and one generated numerator. Every timed point
rebinds the borrowed parameter row and kinematics/masses/S-matrix pointers, sets
the scale, resets the amplitude, and runs original reduction plus uncached
OneLOop masters. Callback buffers are allocated once; parameter copies occur in
the selected callbacks. Sequence packing, modulo selection, parsing, setup,
warmup, output serialization, IPC, and checksum calculation are outside timing.
An optimizer barrier consumes results/statuses without adding numerical checksum
work to the primary loop. Every round's nanosecond count and point count are
reported. This is a **serial scalar baseline**, with no batch API claim.

All source rows are evaluated before and after the timed jobs. The report keeps
their three Laurent coefficients, both finite-part components, statuses, and
finite-value flags. A failed or nonfinite initial row suppresses timing. The
final warmup pass is checked, and a failure stops subsequent timed work. Pre/post
result fingerprints use FNV-1a over explicitly ordered binary64 bits; fingerprint
work is untimed, and signed zeros survive JSON formatting. This is a result
fingerprint, not a cryptographic source hash. Exact pre/post agreement is
required. Timed iterations are not individually revalidated; the report states
that scope explicitly.

The protocol test compares retained results against fresh constructions in the
legacy runner using the same uncached libraries and generated header. It covers
sample/parameter rebinding, permutations, singleton/repeated/changing controls,
S-matrix pointer replacement, malformed inputs, and cached-build rejection.
For an input with one source row it adds a second renormalization scale solely
as a protocol control; this is explicitly recorded and is not a new phase-space
point. The short test timings are never performance evidence.

With `--ninjars /path/to/copied/ninjars --fixture INPUT.json`, the protocol test
also checks full-reduction Laurent coefficients against the Rust runtime at the
fixed scaled `1e-9` criterion. Partial `cut-stop=2` results are retained as
diagnostics: the parameterized bubble and triangle controls currently differ
between implementations although complete evaluation agrees. The origin of this
difference is unresolved; it must not be described as an expected redistribution
without checking complementary contributions. A subsequent rotation control for
the parameterized bubble did verify that its retained bubble contribution changes
with the chosen numerical basis while the complementary tadpole contributions
restore the same complete result. Original Ninja's `setCutStop` explicitly assumes
omitted lower cuts contribute zero; that assumption fails for this control.
This explains the limitation of this partial comparison, without resolving every
partial-cut discrepancy. Initial cross-implementation
timings therefore require full reduction (`cut-stop=1`). Flag propagation is
independently checked against fresh original-Ninja evaluations.

This harness alone does not satisfy the performance gates. Actual comparisons
still require verified matching inputs and coefficients, optimized Rust builds,
quiet-host runs with matching CPU affinity and thread settings, reproducible
command/source/dependency provenance, and timing uncertainty. Complete evaluation
uses different production master providers; reduction with a shared master
provider remains a separate unimplemented measurement. All runner reports keep
`performance_acceptance: false`.
