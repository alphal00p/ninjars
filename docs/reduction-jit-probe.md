# Existing Symbolica compiler as a reduction diagnostic

`kernel_jit_probe.rs` is test-only research. It compiles the exact existing
reduction plans through Symbolica's public complex SymJIT API. It installs no
amplitude backend and changes no production selection or generic carrier.
The generated Rust catalog remains the production optimization path.

The algebra/API checks were performed before writing the probe:

1. Symbolica's public `ExpressionEvaluator<Complex<Rational>>::jit_compile`
   accepts `Complex<f64>`; `JITCompiledEvaluator::evaluate` and
   `batch_evaluate` accept the existing row-major complex buffers. No expression
   rewriting or new algebra is needed.
2. `symbolica/src/evaluate/backend.rs` implements these calls and resolves
   exact constants at the selected precision. Its `BatchEvaluator` convenience
   wrapper divides by the row count, so the probe checks complete dimensions
   and returns on empty batches before using the direct API. Existing tests in
   `symbolica/src/evaluate.rs` cover complex compilation and nested bodies.
3. FeynKit/gammaloop's `integrands/process/evaluators.rs` already uses this
   compiler and documents conditional/helper limitations. NinjaRS's graph
   coefficient adapter also uses its explicit O2 settings. This probe accepts
   only Add/Mul/integer-Pow/Assign instructions with no shared helper bodies or
   registered constants; those limitations do not need a new implementation.

The probe disables fast-math, lane SIMD and threads, and rejects external
`SYMJIT_TOML`/local `symjit.toml` configuration. `fast_complex=true` uses packed
real/imaginary arithmetic within one complex number. With these settings,
SymJIT's matrix entry point loops over complete scalar JIT evaluations. This is
therefore a **scalar-JIT-per-row diagnostic**, not an instruction-major batched
replacement. A production specialization would require separate validation of
its batching, numerical domain, selection and reporting.

The default control compares seventeen existing plans against their generic
interpreter at widths 0/1/3/8/31/32/128, then reuses each compiled program with
reversed lanes. It includes ordinary/higher-rank corrections, Laurent division
through six propagators, and an empty quotient. Invalid dimensions must leave
outputs unchanged. Inputs are finite moderate synthetic complex values; this
does not validate exceptional values, branch cuts, Dual transformations or
full physical amplitudes for JIT execution.
In particular, the packed complex division uses a direct norm-squared
denominator; these moderate inputs do not establish extreme-range robustness.

Run correctness with the licensed development environment:

```sh
cargo test -p ninjars-symbolica --all-features --lib complex_reduction_jit_matches_interpreter -- --nocapture
```

The separately ignored `reduction_jit_microbenchmark` requires `--release`.
It records exact evaluator construction/export/JIT setup separately and six
alternating contiguous pairs per plan and width, reusing buffers. Rows vary
within each matrix, but every iteration repeats that same matrix. It includes
neither numerator callbacks nor master
integrals. Raw block durations and parity errors are retained in its JSON line;
they are a development diagnostic, never an original-Ninja comparison or a
performance-acceptance result. Blocks below 20 ms are explicitly flagged.
Short blocks and shared-host interference must be considered before
interpreting any ratios.
