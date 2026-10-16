# NinjaRS

NinjaRS is a Rust port of Ninja one-loop integrand reduction. It uses FeynKit's
physics types, Symbolica's algebra and evaluators, and oneloopmaster's scalar
integrals. The intended interface carries batches and generic numerical domains
through numerator callbacks, reduction, and master integrals.

This is a development version. The library integrates one-, two- and
three-propagator amplitudes through rank n+1, with batched polynomial/Dual
callbacks, lower-cut subtraction, master integrals and lane-local accumulation.
Analytic and original Ninja controls validate an initial synthetic suite.
It also includes independent DOT preparation and an original Ninja reference
harness. Full topology coverage, physical validation, derivative continuation
and performance acceptance remain in development.
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
| `ninjars-symbolica` | Symbolica polynomial/tensor callbacks and master adapters |
| `ninjars-graph` | Independent FeynKit graph, model and numerator preparation |
| `ninjars-cli` | `prepare` and `reduce`; planned `compare` and `bench` commands |

Numerical momenta are FeynKit `FourMomentum<T>`, with component order `(E,x,y,z)`
and the `+---` metric. Laurent results use `[finite, single pole, double pole]`.
Squared masses and the extra-dimensional variable enter denominators as
`(q + V_i)^2 - m_i^2 - mu_squared`.

`reduce` writes JSON with exact prepared inputs, per-lane statuses, and complex
results as decimal strings. It accepts `--precision f64`, `double-float` or
`float --bits N`. Working precision and the actual precision of each result
component are recorded separately. Failed numerical lanes have no reported
value and cause a nonzero exit status; successful neighboring lanes are retained.

See [synthetic graph fixtures](examples/graphs/README.md) and the
[original Ninja reference harness](validation/reference/README.md).
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
