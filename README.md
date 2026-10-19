# NinjaRS

NinjaRS is a Rust port of Ninja one-loop integrand reduction. It uses FeynKit's
physics types, Symbolica's algebra and evaluators, and oneloopmaster's scalar
integrals. The intended interface carries batches and generic numerical domains
through numerator callbacks, reduction, and master integrals.

This is a development version. The library provides reduction stages for any
number of propagators and numerator rank up to n+2 in every precision, with
batched polynomial/Dual callbacks, both box mu-extraction modes, lower-cut
subtraction, master integrals and lane-local accumulation. Rank n+2 goes one
beyond original Ninja, which stops at n+1; see [rank n+2](docs/rank-n2.md).
Analytic and original Ninja controls validate an initial synthetic suite.
Independent FeynKit preparation also matches raw MadLoop numerators for both
top-quark H→gg orientations, six four-photon boxes and eight gg→HH graphs.
Their physical integrated comparisons pass in all three scalar precisions.
Some generic-polarization and Ward diagnostics remain outside the fixed gates,
as recorded in the evidence ledger. All 50 top-quark gg→HHH graphs pass raw
numerator matching and the 2,400-row integrated matrix, including independent
analytic pole controls and exact cancellation of the summed poles.
Full reconstruction controls, derivative continuation and performance acceptance
remain in development.
The complete specification and evidence ledger are in [the plan](references/PLAN.md).

## Development

The pinned Nix shell provides Rust 1.98.1, native numerical dependencies, and the
C++/Fortran reference compilers on x86_64 Linux. Rust 1.96 or newer is required.

Every Rust dependency is a pinned git revision (`Cargo.toml`, `Cargo.lock`), so a
plain clone builds without any local checkout:

```sh
nix develop path:. --command cargo check --workspace --locked
RUST_MIN_STACK=134217728 nix develop path:. --command cargo test --workspace --locked -- --test-threads=1
nix develop path:. --command cargo run -p ninjars-cli -- prepare examples/graphs/bubble.json
nix develop path:. --command cargo run -p ninjars-cli -- reduce examples/graphs/bubble.json --batch-size 2
nix develop path:. --command cargo run -p ninjars-cli -- reduce examples/graphs/triangle-reference.json --precision float --bits 256
nix develop path:. --command cargo run -p ninjars-cli -- reduce examples/graphs/box.json --mu-extraction sampling --mu-sampling-scale 64 --full-box-coefficients
nix develop path:. --command cargo run -p ninjars-cli --features reference-codegen -- compare examples/graphs/bubble.json --batch-size 2
nix develop path:. --command cargo run --release -p ninjars-cli -- bench examples/graphs/bubble.json --output bubble-bench.json
nix develop path:. --command cargo run --release -p ninjars-cli -- reduce examples/graphs/physical/gg-hh/top-box-2-pool.json --engine fast
nix develop path:. --command cargo run --release -p ninjars-cli -- bench examples/graphs/physical/hgg/top-triangle-0-pool.json --engine fast
nix develop path:. --command cargo run --profile perf -p ninjars-cli --features numerator-jit -- pipeline-bench examples/graphs/physical/gg-hh/top-box-2-pool.json --points validation/benchmark/points/gg-hh.json --feynman-rules
nix develop path:. --command cargo run --profile perf -p ninjars-cli -- case-bench validation/benchmark/cases/gg-hh-top-box-2.json --reference-dir validation/benchmark/ninja
nix develop path:. --command cargo run --release -p ninjars-cli -- case-check validation/rank-n2/cases/rn2-int-complex-n4-r6-mu2.json --engines fast,generic --precision f64,float:256 --reference-dir validation/rank-n2/reference --known-failures validation/rank-n2/known-failures.json --twins validation/rank-n2/summary/controls.json --twins validation/rank-n2/summary/synthetic-twins.json --twins validation/rank-n2/summary/n7-controls.json
nix develop path:. --command cargo run -p ninjars-cli -- reduce examples/graphs/physical/hgg/top-triangle-0.json --precision double-float --batch-size 2
nix develop path:. --command cargo run -p ninjars-symbolica --example bubble
```

Dependencies are pinned in `Cargo.toml` and recorded with their provenance in
`dependencies/manifest.json`:

| Dependency | Repository | Revision |
| --- | --- | --- |
| FeynKit, Spenso, Idenso (gammaloop) | `https://github.com/alphal00p/gammaloop` | `d1904bf2` (branch `feynkit`, with PRs #142-#144 and #149-#152 merged) |
| oneloopmaster (`oneloop`) | `https://github.com/alphal00p/oneloopmaster` | `09b2c749` |
| Symbolica, Numerica, Graphica | `https://github.com/alphal00p/symbolica` (branch `ninjars-pin`) | `e679e8c1` |

Symbolica `e679e8c1` is upstream `c78cffc2` plus the patches in
`dependencies/patches` and the binary64 dilogarithm of symbolica-dev/symbolica#89,
whose pull requests are pending. NinjaRS replaces the crates.io `symbolica`,
`numerica` and `graphica` through `[patch.crates-io]`; Cargo applies `[patch]`
only in the root workspace, so a project that depends on NinjaRS crates through
git must repeat that block in its own root `Cargo.toml`:

```toml
[dependencies]
ninjars-symbolica = { git = "https://github.com/alphal00p/ninjars", rev = "<revision>" }

[patch.crates-io]
symbolica = { git = "https://github.com/alphal00p/symbolica.git", rev = "e679e8c129ce15ec50c46ffb4ff4b59aeb7813b8" }
numerica = { git = "https://github.com/alphal00p/symbolica.git", rev = "e679e8c129ce15ec50c46ffb4ff4b59aeb7813b8" }
graphica = { git = "https://github.com/alphal00p/symbolica.git", rev = "e679e8c129ce15ec50c46ffb4ff4b59aeb7813b8" }
symbolica-typst-plugin = { git = "https://github.com/symbolica-dev/symbolica-typst-plugin", rev = "57b7455b37c30f4f87353c0073e49f5225e937ab" }
```

The CLI installs directly: `cargo install --locked --git https://github.com/alphal00p/ninjars ninjars-cli`.

To develop NinjaRS together with local checkouts of these dependencies, place
them under `references/` (`python3 tools/bootstrap.py` clones the pinned
revisions there and applies the patches) and pass
`--config dependencies/local-references.toml` to Cargo. Nothing under
`references/` is needed otherwise; the C++/Fortran reference validation
(`--features reference-codegen`, `validation/`) also uses `references/ninja`.

Symbolica retains its own licensing and runtime requirements. Run its tests
serially when using a license that limits concurrent instances. Production
library code and graph preparation are Rust; Python and C++/Fortran here are
development and reference-validation tools.

## Packages

| Package | Responsibility |
| --- | --- |
| `ninjars` | Topology, batched reduction callback contracts, kernels and results |
| `ninjars-positional` | Symbolica evaluators whose results do not depend on the symbols a process registered before |
| `ninjars-symbolica` | Symbolica polynomial/Dual callbacks, arithmetic kernels and masters |
| `ninjars-graph` | Independent FeynKit graph, model and numerator preparation |
| `ninjars-cli` | `prepare`, `reduce`, `bench`, optional `compare`, benchmark-case tools (`export-bench`, `numerator-bench`, `pipeline-bench`, `case-bench`) and `case-check` (bench cases against high-precision references) |

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

`reduce --engine fast` (also `bench`) selects `ninjars::fast`, a batched
binary64 port of original Ninja's reduction with oneloopmaster's native
masters. It requires `--precision f64` and the default mu-expansion, rejects
the generic engine's box-sampling and kernel options, and needs real external
kinematics. Numerators without explicit mu-squared terms use Ninja's dense
tensor numerator, filled from coefficient heads of the selected
`--numerator-evaluator`; numerators with mu-squared terms use the same heads
in `MuTensorNumerator` (`sum_k mu^(2k) T_k(q)`). The JSON schema is unchanged apart from `"engine": "fast"`. On the
tested fixtures, Laurent coefficients agree with the generic engine within the
scaled 1e-9 criterion; the cut-constructible/rational split follows original
Ninja's cut basis. As in original Ninja, points with purely spacelike cut
planes fail their lanes.
See the [CLI guide](crates/ninjars-cli/README.md#reduction-engines).

Rank n+2 (beyond original Ninja) is reduced by both engines. The generic
engine builds every exact plan from a `RankClass`, and its plans for rank
<= n+1 are byte-identical to before. The fast engine reduces rank n+2 in a
separate engine selected at construction (`fast/engine_x2.rs`), which leaves
the rank <= n+1 path bitwise unchanged.

The new pieces are:
- a three-term box mu-expansion on the exact d-dimensional cut,
  `mu² = t² v² - beta`, which numerators with explicit mu² need;
- triangle and bubble residues with 21 and 35 coefficients;
- the rational terms of box mu⁶, bubble mu²(q·e2)² and bubble mu⁴;
- the rank-four bubble master B1111, from oneloopmaster's native family with
  OneLOop's mass-swap sign corrected.

`ninjars case-check` judges both engines against the high-precision suite in
`validation/rank-n2` (333 lanes, n = 1..7). No lane fails in the fast engine
or in the generic engine in binary64, DoubleFloat, Float(128) or Float(256).
With oneloopmaster 09b2c74 no upstream failure is left; three Float(128) lanes
are listed as known inaccuracies of the exact masters in `Float`. See
[rank n+2](docs/rank-n2.md) for the formulas, the validation and the known
limitations.

`--r2 off|fdh` (on `prepare`, `reduce`, `bench`, `compare`, `export-bench`,
`numerator-bench` and `pipeline-bench`) selects whether the physical numerator
carries R2. The default `off` binds the loop momentum four-dimensionally
(R1 only, unchanged results). `fdh` binds `K.K -> q.q - mu^2` with the
`d_s = 4` internal algebra, the four-dimensional-helicity numerator whose
`mu^2` terms give all of R2 in FDH (HV/CDR's `(d_s-4)`-dimensional algebra is
not implemented). The H→gg top triangles then reproduce MadGraph5 `loop_sm`'s
`R2_GGHt` vertex, and the gluon self-energy and the summed four-photon boxes
satisfy their Ward identities; see [R2 in FDH](docs/physical-preparation.md#r2-rational-terms-in-fdh).

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
`bench --engine fast` measures the fast engine with the same report.
`pipeline-bench` times the binary64 pipeline of one physical graph (numerator
preparation, pure fast reduction, tensor numerator, native masters and the
uninstrumented total) over a shared phase-space pool; `case-bench` gives the
same breakdown for the `ninjars-bench-case-v1` files shared with original
Ninja. Both can check results against original Ninja; see the
[CLI guide](crates/ninjars-cli/README.md#end-to-end-and-bench-case-timings)
and the [benchmark report](docs/benchmark/benchmark.typ). `case-check` reduces
the same files with the fast engine and the generic engine at `f64`, `double`
or `float:<bits>`, optionally declared at a higher rank, and judges every lane
against high-precision or original-Ninja references; see the
[CLI guide](crates/ninjars-cli/README.md#checking-cases-against-references-case-check).
The [matched reference timing harness](validation/performance/README.md) uses a
common parameterized numerator, an uncached original-Ninja/OneLOop build, and
persistent numerical sessions. It records full Laurent checks, compiler/build
provenance and raw timing rounds; the performance gates remain unfulfilled.
The [shared physical-provider control](validation/performance/shared_provider/PHYSICAL.md)
also validates a gg→HHH pentagon from raw external momenta through wavefunctions,
model parameters, SymJIT coefficient preparation, reduction and live masters.
It covers three changing phase-space points and batches through 1024; it is a
correctness control, with optimized end-to-end timing still outstanding.

The optional `ninjars-symbolica/rust-codegen` feature exports existing exact
kernel graphs through oneloopmaster's shared emitter. The `native-kernels`
feature enables explicit native batch dispatch for selected denominator,
bubble-correction and tadpole-projection plans. Coverage is partial: geometry
and other reduction stages still use Symbolica evaluators. The
[saved denominator-reuse timings](validation/performance/shared_provider/REUSE-TIMING-553f406.md)
record the remaining performance gap and keep actual master computation separate.

[Same-call denominator reuse](docs/denominator-reuse.md) is integrated. The
optional [guarded rank specialization](docs/guarded-rank-specialization.md)
uses Symbolica to remove structural zeros for five propagators and rank four,
with checks of every Dual component and a generic fallback. Separate explicit
opt-ins select interpreted or generated native alternate kernels. The complete
[native rank/block comparison](validation/performance/shared_provider/RANK-BLOCK-TIMING-8b86822.md)
finds modest improvements while the reduction gap remains substantial. The
[algorithm audit](docs/audits/reduction-algorithm-audit.md) describes further
experiments. Its [bounded triangle study](docs/audits/transpose-probe.md) proves
the forward/transposed symbolic identities but finds lower instruction counts
from forward fusion at one changing point. These counts exclude some required
intermediate diagnostics and do not establish the final performance gates.
The [retained-output follow-up](docs/audits/forward-diagnostic-fusion.md) finds
that whole-network fusion increases the instruction count when those outputs
are retained. A [staged master-weight inventory](docs/audits/integrated-weight-fusion.md)
instead preserves callback parent boundaries and separate lazy masters: its
n3/r3 graph uses 1,071 pre-master instructions versus 1,884, with 90 exact
identities passing. Numerical behavior and runtime remain unvalidated.
A follow-up [real-domain arithmetic audit](docs/audits/real-domain-arithmetic.md)
identifies existing Symbolica annotations that generated Rust could use while
preserving complex masses and all derivative components; that experiment is
not yet implemented or measured.

Optional [swapped-bubble arithmetic pruning](docs/swapped-bubble-arithmetic.md)
uses Symbolica graphs containing only heads needed by residue projection.
Full numerator callbacks and raw coefficient APIs are preserved. Ordinary and
higher/mixed Dual correctness controls pass; this option has no performance
measurement yet and remains separate from the native catalog.

An explicit [native batch execution policy](docs/native-batch-policy.md) can
bound generated-kernel scratch by processing whole rows in fixed blocks. `Whole`
remains the default. The policy preserves complete Dual rows and applies to
native generic kernels and guarded alternates; callbacks and masters retain
their separate providers. The [CLI and harness option](docs/native-block-execution.md)
and optimized physical correctness controls pass. The complete 96-process
rank/block timing comparison retains all samples with actual master execution
excluded; the performance gate remains unmet. Separate instrumented preparation
and live-master accounting is in progress.

See [synthetic graph fixtures](examples/graphs/README.md) and the
[original Ninja reference harness](validation/reference/README.md).
The [physical preparation guide](docs/physical-preparation.md) gives exact
polarization/model bindings, isolated MadLoop graph comparisons and the retained
physical accuracy results and diagnostic limitations.
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
