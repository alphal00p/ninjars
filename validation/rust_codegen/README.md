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

The generator emits scalar and flattened Dual versions of these exact plans:

- `LaurentDivision::new(4, 5, 3)`;
- `TriangleBubbleSubtraction::new(true)`;
- `DenominatorExpansion::new(2)`.

The shared derivative shape is `[[0,0],[1,0],[0,1],[2,0],[1,1]]`. Input and
output components follow the existing kernel and Symbolica shape order. The
probe compares every component with the ordinary interpreted kernels for f64,
DoubleFloat, and Float256, with empty, single, three-row, and eight-row inputs.
It tests malformed input/output buffers and mismatched Dual shapes without
modifying output buffers. Exact rational inputs and constants are constructed
directly at working precision. Tolerances are `1e-12`, `1e-25`, and `1e-65`,
respectively.

Generated modules and the upstream exact constant table live under
`target/rust-kernel-probe`. The runner creates an exclusive temporary example
wrapper in the existing workspace and removes it after execution, retaining
the root dependency graph, patches, lockfile and cache. All numerical
primitives come from `oneloop::rust_codegen::runtime`.

This is a **compiled scalar row-loop control**. It does not establish
instruction-major batching, benchmark acceptance, or production dispatch.
`--release` selects the optimized profile for follow-up experiments; this
correctness probe does not report timings or claim a speedup.
