# Reduction kernel attribution

This is a validation-only profiler. It does not implement arithmetic, change
production sources, measure a parity ratio, or establish a quiet timing window.
The first retained report is
[`reduction-kernel-profile.json.gz`](../../validation/evidence/reduction-kernel-profile.json.gz).

`run.py` compiles a small C++ driver against the immutable shared-provider Rust
library and unchanged original Ninja. The driver captures both reducers' master
request union, freezes it, checks total Laurent coefficients, warms the retained
Rust coordinator, and then enables own-process `perf` only around repeated Rust
reductions and a checksum. Setup, coefficient preparation, symbolic compilation,
master evaluation and the independent checks are outside sampling. Frozen-master
lookups, numerator callbacks, FFI and ordinary reduction work remain inside.
The n5 control checks the independent archived MadLoop target as well.

The physical provider selects **SymJIT O2 coefficient-head preparation**. Its
actual direct/T3/T2 `PolynomialNumerator` callbacks remain interpreted. The
synthetic bubble providers also use interpreted callbacks. Both original and
Rust implementations use the same provider, but these profiles do not establish
the intended fully SymJIT tensor-numerator execution path. The separate matched
measurement harness owns original/Rust numerator timings.

The immutable library predates native denominator dispatch, and the diagnostic
rebuild leaves `native-kernels` disabled. All recorded reduction arithmetic uses
the Symbolica interpreter. Actual old library source/dependency identities and
the diagnostic rebuild identities are retained separately in the report.

## Reproduce

From the repository root, with its Rust/C++/Fortran environment and Symbolica
license configured:

```sh
python3 validation/reduction_profile/run.py \
  --repository "$PWD" --output-dir target/reduction-profile/n5-new \
  --perf /path/to/perf --workload n5 --width 1 --rounds 30000
```

The immutable inputs currently live in
`target/performance/shared-workloads-diagnostic-newpins-release`; they are checked
against the archived shared-workload artifact hashes. The unchanged original
libraries live in `target/reference-uncached-v2` with master caching disabled.
The validation binary never calls production masters while profiling: all such
calls are exact-key frozen replay.

`--workload bubble2` and `bubble3` select the previously tested shared-provider
polynomials `a*q0^2+b*mu2+5` and `a*q0^3+b*mu2*q0+5`. Both use
`a=2+i`, `b=3-i`, squared masses `[2,3]`, shifts `[(1,0,0,0),(1,1,0,0)]`,
and scale squared `7`. These are diagnostic counterparts, not replacements for
the historical parameterized-bubble benchmark gate.

The optimized old Rust library does not unwind reliably beyond its interpreter
frames. `instrument.py` therefore makes a disposable **copy** of the existing
adapter and bridge sources, adds semantic phase guards and caller-location
kernel guards, and rebuilds only those two crates against explicitly hashed
cached rlibs from a Cargo JSON artifact manifest:

```sh
python3 validation/reduction_profile/instrument.py \
  --repository "$PWD" \
  --cargo-artifacts target/performance/shared-workloads-diagnostic-newpins-release/cargo-build.stdout \
  --output-dir target/reduction-profile/instrumented-new
python3 validation/reduction_profile/run.py \
  --repository "$PWD" --output-dir target/reduction-profile/spans-new \
  --diagnostic-build target/reduction-profile/instrumented-new/build.json \
  --workload n5 --width 8 --rounds 1000
python3 -m unittest discover -s validation/reduction_profile -p 'test_*.py' -v
```

The build record includes original and injected source hashes, cached dependency
rlib hashes, compiler identity and complete rustc commands. This mixed cached
build is a diagnostic artifact; it is not the matched optimized production
build. `track_caller` propagates through both generic executor implementations
and the ordinary kernel method. Labels never select arithmetic. Existing exact
Symbolica expressions and their parameter order remain unchanged.

Phase guards count direct geometry, division, correction, projection and
integration calls. Kernel guards count actual evaluator rows and shapes. Nested
elapsed time is subtracted to report phase self spans, with correctness tests
for nesting and owner propagation. Timers/map bookkeeping perturb the measured
program, including empty calls. In particular, the existing Symbolica evaluator
returns immediately for zero rows: an empty correction's instrumented duration
is not evidence of its uninstrumented cost.

Perf acknowledges commands with the bytes `ack\n\0`, including the trailing
NUL: see Linux's [`evlist__ctlfd_ack`](https://github.com/torvalds/linux/blob/master/tools/perf/util/evlist.c).
The first attempt used line-based acknowledgement reads and failed at disable;
that attempt remains under `target/reduction-profile/n5-width1`. Its numerical
and sampling output was not promoted as a successful retained profile. The
corrected protocol has positive, corrupt and truncated-response regressions.

`summarize.py` validates complete
fixed-workload counters, geometry recomputation, semantic caller locations and
source/numerical guards; it also audits unchanged copied reference sources and
generated headers against the reference build record. That last audit is
explicitly post-run, not backdated into the original run protocol. Archive with
`--profile-dir` for each external sample and `--span-dir` for each diagnostic
run. Gzip uses deterministic `mtime=0`.
