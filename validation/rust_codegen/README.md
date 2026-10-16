# Existing-kernel Rust emission probe

The optional `ninjars-symbolica/rust-codegen` feature exports the exact existing
Symbolica evaluator through oneloopmaster's public `RustEmitter`. It introduces
no additional algebra or instruction representation. `RustKernelExport`
supports `LaurentDivision`, `TriangleBubbleSubtraction`, and
`DenominatorExpansion`; its Dual method shares `ExactKernel::dual_evaluator`
with the interpreted implementation.

Run from the repository development environment:

```sh
python3 validation/rust_codegen/run.py \
  --evidence validation/evidence/rust-kernel-codegen.json
```

The generator emits instruction-major batch and scalar versions of these exact
plans, both ordinary and after Symbolica's flattened Dual transformation:

- `LaurentDivision::new(4, 5, 3)`;
- `TriangleBubbleSubtraction::new(true)`;
- `DenominatorExpansion::new(2)`.

The shared derivative shape is `[[0,0],[1,0],[0,1],[2,0],[1,1]]`. Input and
output components follow the existing kernel and Symbolica shape order. The
probe compares every component with the ordinary interpreted kernels and emitted
scalar controls for f64, DoubleFloat, and Float256. Batch widths follow
`0, 1, 8, 32, 128, 1024, 3, 31, 127, 0, 8, 1`, reusing one workspace across
widths, kernels and derivative layouts. Reversed row permutations are checked
against reversed interpreter results. Malformed input/output buffers, dimension
overflow, and mismatched interpreter Dual shapes leave outputs unchanged.
Rejected native calls must also leave workspace dimensions unchanged.
Exact rational inputs and constants are constructed
directly at working precision. Tolerances are `1e-12`, `1e-25`, and `1e-65`,
respectively.

Thread-local allocation counting verifies zero warmed f64 allocations for each
kernel and batch layout. A fresh single-row workspace must remain empty; f64
also requires zero allocations on this path. Arbitrary-precision arithmetic
itself may allocate and is not subject to that allocation gate.

Generated modules and the upstream exact constant table live under
`target/rust-kernel-probe`. The runner creates an exclusive temporary example
wrapper in the existing workspace and removes it after execution, retaining
the root dependency graph, patches, lockfile and cache. All numerical
primitives come from `oneloop::rust_codegen::runtime`. The runner records the
compiled executable, generated sources, implementation sources, lockfile and
dependency manifest hashes, and exact commands. Source mutation during the run
invalidates the report. `--jobs` defaults to eight build workers.

These are **compiled instruction-major kernel batches**, using the upstream
emitter and reusable `BatchWorkspace`. The interpreter remains the production
backend; this probe does not establish production dispatch or benchmark acceptance.
`--release` selects the optimized profile for follow-up experiments; this
correctness probe does not report timings or claim a speedup.
