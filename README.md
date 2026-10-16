# NinjaRS

NinjaRS is a Rust port of Ninja one-loop integrand reduction. It uses FeynKit's
physics types, Symbolica's algebra and evaluators, and oneloopmaster's scalar
integrals. The intended interface carries batches and generic numerical domains
through numerator callbacks, reduction, and master integrals.

This is a development version. The library provides reduction stages through
six propagators and rank n+1, with batched polynomial/Dual callbacks, both box
mu-extraction modes, lower-cut subtraction, master integrals and lane-local accumulation.
Analytic and original Ninja controls validate an initial synthetic suite.
Independent FeynKit preparation also matches raw MadLoop numerators for both
top-quark H→gg orientations. Their physical helicity amplitudes agree in all
three scalar precisions; four f64 longitudinal diagnostic rows still miss the
fixed pole tolerance. Six top-quark four-photon boxes also pass independent raw
numerator comparisons for three helicities and generic complex polarizations;
their integrated comparison is pending. Full reconstruction controls, broader physical validation,
derivative continuation and performance acceptance remain in development.
The complete specification and evidence ledger are in [the plan](references/PLAN.md).

## Development

The pinned Nix shell provides Rust 1.98.1, native numerical dependencies, and the
C++/Fortran reference compilers on x86_64 Linux. Rust 1.96 or newer is required.
Bootstrap the source dependencies before invoking Cargo:

```sh
nix develop path:. --command python3 tools/bootstrap.py
nix develop path:. --command cargo check --workspace --locked
RUST_MIN_STACK=134217728 nix develop path:. --command cargo test --workspace --locked -- --test-threads=1
nix develop path:. --command cargo run -p ninjars-cli -- prepare examples/graphs/bubble.json
nix develop path:. --command cargo run -p ninjars-cli -- reduce examples/graphs/bubble.json --batch-size 2
nix develop path:. --command cargo run -p ninjars-cli -- reduce examples/graphs/triangle-reference.json --precision float --bits 256
nix develop path:. --command cargo run -p ninjars-cli -- reduce examples/graphs/box.json --mu-extraction sampling --mu-sampling-scale 64 --full-box-coefficients
nix develop path:. --command cargo run -p ninjars-cli --features reference-codegen -- compare examples/graphs/bubble.json --batch-size 2
nix develop path:. --command cargo run --release -p ninjars-cli -- bench examples/graphs/bubble.json --output bubble-bench.json
nix develop path:. --command cargo run -p ninjars-cli -- reduce examples/graphs/physical/hgg/top-triangle-0.json --precision double-float --batch-size 2
nix develop path:. --command cargo run -p ninjars-symbolica --example bubble
```

Dependencies are pinned in `dependencies/manifest.json`. Small upstream patches
are stored in `dependencies/patches` while their pull requests are pending;
`tools/bootstrap.py` applies them to isolated dependency checkouts. It does not
reset or discard changes in existing checkouts. The large source checkouts and
build products are excluded from this repository.

Symbolica retains its own licensing and runtime requirements. Run its tests
serially when using a license that limits concurrent instances. Production
library code and graph preparation are Rust; Python and C++/Fortran here are
development and reference-validation tools.

## Packages

| Package | Responsibility |
| --- | --- |
| `ninjars` | Topology, batched reduction callback contracts, kernels and results |
| `ninjars-symbolica` | Symbolica polynomial/Dual callbacks, arithmetic kernels and masters |
| `ninjars-graph` | Independent FeynKit graph, model and numerator preparation |
| `ninjars-cli` | `prepare`, `reduce`, `bench` and optional `compare` |

Numerical momenta are FeynKit `FourMomentum<T>`, with component order `(E,x,y,z)`
and the `+---` metric. Laurent results use `[finite, single pole, double pole]`.
Squared masses and the extra-dimensional variable enter denominators as
`(q + V_i)^2 - m_i^2 - mu_squared`.

`reduce` writes JSON with exact prepared inputs, per-lane statuses, and complex
results as decimal strings. It accepts `--precision f64`, `double-float` or
`float --bits N`. Working precision and the actual precision of each result
component are recorded separately. Failed numerical lanes have no reported
value and cause a nonzero exit status; successful neighboring lanes are retained.
Box sampling automatically selects the largest squared-mass or invariant
magnitude in each lane, retaining working precision. JSON records each selected
scale and its source. `--mu-sampling-scale EXPR` supplies an exact override,
independent of the renormalization scale. The default mu-expansion mode obtains
the integrated rational coefficient directly from the numerator expansion.
Conditioning diagnostics and precision rescue remain under development.

`--cut-stop N` omits cuts smaller than N; values 0/1 retain full reduction.
As in original Ninja, using a higher stop assumes that omitted cuts contribute
zero. A retained partial contribution can depend on the cut basis: the
[rotation control](validation/evidence/partial-cut-basis.json) demonstrates this
for a bubble with nonzero tadpoles. Compare complete reductions unless that
assumption has been independently established.
Fixtures may supply an exact `s_matrix` per sample, in prepared propagator order,
to preserve invariants that rounded momentum components would lose. These
invariants feed both master integrals and rational terms.

With `--features reference-codegen`, `compare` prepares the same numerator and
propagators for NinjaRS and original Ninja/OneLOop, then checks all Laurent
coefficients with the fixed scaled 1e-9 criterion. It accepts the reduction
precision and cut controls. Original Ninja uses binary64, so these comparisons
alone do not establish higher-precision accuracy. Generated callbacks, input
files, build logs and a JSON report are retained in a unique directory under
`target/compare` or the chosen `--reference-dir`. Reference libraries are built if absent; Python and
C++/Fortran are needed for this validation command. Recorded reference timings
are diagnostics; performance acceptance requires the separate fixed-suite
measurements with matching cache policies.

`bench` reuses the reduction session across repeated or changing inputs at
batch sizes 1, 8, 32, 128 and 1024, including tails. Preparation and warmup are
outside primary timings; callbacks and masters are measured in a separate
diagnostic pass. Reports retain individual timings and failed-lane counts.
These are Rust-only measurements, with performance acceptance explicitly
marked false. See [benchmark controls and timing boundaries](docs/benchmarks.md).
The [matched reference timing harness](validation/performance/README.md) uses a
common parameterized numerator, an uncached original-Ninja/OneLOop build, and
persistent numerical sessions. It records full Laurent checks, compiler/build
provenance and raw timing rounds; the performance gates remain unfulfilled.

The optional `ninjars-symbolica/rust-codegen` feature exports existing exact
kernel graphs through oneloopmaster's shared emitter. The
[compiled scalar/Dual probe](validation/rust_codegen/README.md) records correctness
across all three carriers; native batch execution and production dispatch remain
under development.

See [synthetic graph fixtures](examples/graphs/README.md) and the
[original Ninja reference harness](validation/reference/README.md).
The [physical preparation guide](docs/physical-preparation.md) gives exact
polarization/model bindings, isolated MadLoop graph comparisons and the retained
physical accuracy results, including the unresolved longitudinal f64 poles.
The [adapter guide](crates/ninjars-symbolica/README.md) describes precision,
Dual coefficient conventions and current master-domain limitations. Constructing
large oneloopmaster expressions requires a generous thread stack; the test command
above follows its 128 MiB regression-test setting.
The reference harness additionally requires the supplied MadGraph 3.7.0 source
tree under `references/MG5_aMC_v3_7_0`.

## References

- [Original Ninja](https://github.com/peraro/ninja)
- [Laurent reduction](https://arxiv.org/abs/1203.0291)
- [Ninja](https://arxiv.org/abs/1403.1229)
- [Tensor interface](https://arxiv.org/abs/1604.01363)
- [FeynKit/gammaloop](https://github.com/alphal00p/gammaloop)
- [Symbolica](https://github.com/symbolica-dev/symbolica)
- [oneloopmaster](https://github.com/alphal00p/oneloopmaster)

NinjaRS is licensed under GPL-3.0-or-later. Its dependencies retain their own
licenses; see their source distributions.
