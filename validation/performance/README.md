# Complete-evaluation timing comparison

`compare.py` validates common binary64 inputs before comparing the warmed Rust
batch path with a persistent scalar original-Ninja/OneLOop process. It uses the
uncached build and protocol described in
[`../reference/README.md`](../reference/README.md). Production code remains Rust;
Python and C++ are validation tools.

Run only after builds have finished, using an optimized CLI and its matching
build record:

```sh
python3 validation/performance/build.py --output-dir target/performance/build
python3 validation/performance/compare.py examples/graphs/bubble-parameterized.json \
  --ninjars target/release/ninjars \
  --rust-build-record target/performance/build/rust-build.json \
  --output-dir target/performance/bubble-parameterized-first
```

Run these commands inside the pinned development shell with the Symbolica
license available. The build recorder invokes Cargo with the release profile,
all CLI features, the locked dependency graph and eight build jobs. It checks
production sources and dependency state before and after compilation, and
records the completed command and log hash. Its output directory must be new
or empty. Configuration files are hashed; only numerical/compiler flags from an
explicit environment allowlist are recorded.

The artifact directory must be new or empty. The runner selects the first CPU in
the process's allowed affinity set unless `--cpu N` selects another allowed CPU.
Every subprocess is pinned to that CPU. OpenMP, OpenBLAS, MKL and Rayon thread
counts are explicitly one, and both reduction implementations use one worker.
The script records CPU model, allowed and selected affinity, frequency/governor
information where available, load averages, and compiler-process snapshots before
and after timing. Those snapshots detect active builds; they are not proof of
exclusive CPU ownership or constant frequency. Retained timing runs refuse an
active build window.

The Rust build record must identify the supplied binary by `binary_sha256` and
include its optimized build `command`, compiler identity, environment allowlist,
production source/dependency hashes, and build-log hash. The script requires a
`--release` command and rejects declared fast-math flags. Before timing it checks
the recorded production file set and hashes against the actual checkout, and
checks each vendored repository's HEAD, tracked status and binary diff hash.
Cargo configuration files in the checkout, ancestor directories and Cargo home
must match the recorded configuration, including additions and removals. These
checks share the definitions in `build.py`; legacy broader source snapshots are
filtered to that same production scope. It preserves the record and its hash
rather than inferring an old executable's flags from the current environment.
The original build separately records and verifies configuration,
compiler commands, source/header/library/binary hashes, higher-rank support, and
the disabled numerical master cache.

Before primary timing the script:

1. Exports one common numerator template and the exact `NumericalPool<f64>` rows.
2. Builds the reference runner against verified uncached libraries.
3. Runs Rust `reduce` and a minimal reference preflight, comparing every total
   Laurent coefficient and finite-part consistency at the fixed criterion
   `abs(a-b)/max(1,abs(b)) <= 1e-9`.

The preflight's tiny reference loop is a correctness control; its timing is not
included in comparisons. Complete evaluation is required (`cut-stop=1`), with
integrated box coefficients. Partial-cut comparisons are not yet suitable
controls. `--mu-extraction expansion|sampling` is propagated to both programs.
Automatic sampling-scale policies differ and remain explicit in the report.
The reference's per-box scale and NinjaRS's topology-wide scale are not relabeled
as identical controls.

Default cases use repeated and changing workloads, batch sizes `1,8,32,128,1024`,
1025 points per iteration, one warmup pass, five rounds, and one iteration per
round. Repeated work uses row zero; changing work cycles through fixture rows in
their original order. A changing workload requires at least two source rows and
two points. Equivalent or repeated rows do not become distinct phase-space
points. The reference performs one serial scalar loop per workload; its same raw
round samples are compared with each Rust batch width. Larger Rust widths have
the same incomplete tails as the ordinary `bench` command.

Both programs retain their numerical sessions internally. Setup, parsing, input
packing, compilation, process startup, JSON and IPC are excluded from the primary
measurements. Rust includes parameter rebinding and its automatic scale selection;
original Ninja includes parameter/pointer rebinding and its own scale selection.
Results and statuses are consumed through optimizer barriers. Full reference
pre/post rows are compared to the validated Rust results after timing, and Rust's
representative outputs are checked against its untimed reduction. Warmup and
diagnostic failure counts must remain zero. This does not assert that every timed
iteration was independently revalidated.

Reports preserve all raw timings, per-point medians, and Rust/reference latency
and throughput ratios. They are ratios of observed medians, not confidence
intervals or paired-sample estimates. `--order reference-first` reverses process
order for independent drift checks; the selected order is recorded. Keep both
orders when making a stronger comparison. Source, dependency, binary and input
hashes must remain unchanged throughout a run, including the checkout revision.
Commands, logs, exported inputs, preflight results and both timing reports remain
in the artifact directory. Actual vendored dependency state and Cargo
configuration are checked again after timing. Failure leaves logs and
`failure.json` for diagnosis. Compiler snapshots are saved before applying the
quiet-window guard and copied into the failure record, so short-lived interfering
processes remain identifiable even after they exit. Only process name, PID,
parent, state and working directory are collected; command lines and environment
contents are not collected.

The inexpensive provenance/failure-path controls run with
`python3 validation/performance/test_compare.py`. They cover modified, added and
removed production inputs, dependency revisions and tracked changes, Cargo
configuration changes, invalid binaries/flags, surviving rejection snapshots,
occupied-lock ownership, and all post-evaluation Laurent coefficients.

For a quick end-to-end harness test only:

```sh
python3 validation/performance/compare.py examples/graphs/bubble-parameterized.json \
  --ninjars target/debug/ninjars --smoke --batch-sizes 1,2 \
  --points 3 --rounds 2 --output-dir target/performance/bubble-smoke
```

Smoke mode explicitly permits a debug binary, a missing build record, and an
active build window. Its timings are diagnostic and must never be presented as
optimized performance evidence.

Every report sets `performance_acceptance: false`. This measures complete
evaluation with each implementation's production master backend. It does not
measure reduction with a shared master provider, complete the fixed acceptance
suite or uncertainty gates, or establish higher-precision and Dual reference
performance. A favorable ratio in one case does not satisfy the overall goal.
