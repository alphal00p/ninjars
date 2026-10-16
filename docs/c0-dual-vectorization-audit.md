# C0 Dual vectorization audit

This audit isolated and fixed an evaluator control-flow defect. It does **not** enable NinjaRS C0/D0 derivatives: coincident dilogarithm differences and other removable limits still need analytic neighborhoods and domain validation.

## Input and revision matrix

The control uses `C0([-1,-5,-4], [2,3,5], scale_squared=1)`. The invariants are Euclidean and the Källén discriminant is −16. The five-component shape is `[[0,0],[1,0],[0,1],[2,0],[1,1]]`; components are Taylor coefficients. The seeded case sets the first mass's first-direction coefficient to `i` and the second mass's second-direction coefficient to `1`, with every other derivative input zero.

With Symbolica `b6edbce2d9b805b481b85a8699bb061e93cd83fb`, the following oneloopmaster revisions produced identical failures at f64 and 256-bit precision:

| oneloopmaster state | Constructor tested |
| --- | --- |
| `606974150011332cfea677f70940f9228e24c9e3` ([PR4](https://github.com/alphal00p/oneloopmaster/pull/4)) | Scalar evaluator followed by `vectorize` |
| `fe2a8d4ac363998ea557d596be57aa10b5bbd013` ([PR5](https://github.com/alphal00p/oneloopmaster/pull/5)) | Both legacy `vectorize` and shape-derived `dual_evaluator` |
| Uncommitted centered sheet-quotient prototype based on `fe2a8d4` | Both constructors; prototype was not pinned to NinjaRS |

The scalar finite coefficient is approximately `-0.1256250231336009578892739940172599963396291754344754061529406730171729509864`, with an imaginary cancellation residual of order `1e-77` at 256 bits. A one-component shape preserves it. The five-component shape instead gave primal `0.25806925911904335+0.07445977235530410i`, even with every derivative seed zero; its first derivative was then `-0.9330816350700663+0.4162743868108943i` instead of zero. With the complex seeds above, the wrong primal persisted and the first derivative became `-1.1304207538454724+0.2456341144510951i`.

The failure therefore predates the PR5 constructor and is not a floating-point tolerance issue.

## Reduction and fix

The minimal case uses the shared function `f(z)=if(c,if(d,z+1,z+2)+z^2,z^2+3)`, called as `f(x)+f(x+1)`. With a two-component Dual, `c=0`, `x=0.3+0.7i`, and zero seeds, it returned `9` then `21` on repeated calls instead of `6.8+2.24i`. No master integrals or transcendental functions are needed.

Both Symbolica common-instruction and common-pair elimination left one branch scope for each `Join` instruction. Vectorization emits several joins per conditional, so nested scopes ended too early. The optimizers could then reuse values from branches that had not run. The fix restores the parent scope at the forward jump's target label, exactly once per conditional.

[Symbolica PR83](https://github.com/symbolica-dev/symbolica/pull/83), upstream-main commit `e1ce502a`, contains the generic fix and direct regression. The tested NinjaRS integration commit is `52ef4b8e3a10cf3a68385230d55ebe8c3d153948`: [PR82](https://github.com/symbolica-dev/symbolica/pull/82) base `b6edbce2`, optimizer fix `a9fde33b`, and dependent shared-Dual regression `52ef4b8e`. The extra regression stays on the integration branch because it needs PR82's shared-body vectorizer.

## Evidence and reproducibility

- Upstream-main direct CSE tests: **2 passed**. The regression `nested_multi_result_branches_keep_common_operations_in_scope` exercises both optimizer passes and alternating/repeated branches.
- Integration evaluator suite: **62 passed**, including all five batch tests and the recursive, higher and mixed Dual regressions. `nested_shared_branches_preserve_all_vector_components` failed before the fix at dimension two and passes afterward for dimensions one, two and five.
- Baseline oneloopmaster `6069741`, plus both PR5 constructors, preserve the C0 scalar primal for shapes one/five at f64/256-bit precision after the fix. Every zero-seed derivative component is exactly zero. The constructors agree component-wise for the complex-seeded case.
- At 256 bits the first imaginary mass seed gives `i*0.0137950660845947717980711131293930864979566529279067836200951905824276631528`, up to a real residual of order `1e-77`. Independent real scalar mass differences converge from `0.0137950667608855` at step `1e-3` to `0.0137950660954178` at step `1e-4`. This is a first-derivative control at one point, not higher/mixed master-domain validation.
- PR5 follow-up `33a2e89b76252fc22556c0da50d1689cfb2101ec` adds the permanent oneloopmaster test `removable_log_tests::euclidean_c0_preserves_primal_and_zero_seeds`, which checks all three Laurent coefficients and every shape-five zero-seed component at f64/256-bit precision. That test passes; the preceding **56 unit tests pass, with one existing ignored test**.
- Changed Rust files pass formatting and diff checks. Normal Symbolica Clippy completes with existing warnings; strict warnings-as-errors remains blocked by unrelated baseline lints. No lint points to a changed line. Oneloopmaster strict package Clippy passes with the pre-existing documented lint allowances.

For Symbolica, run the `evaluate::` library-test filter at the integration revision, with `integer-gmp,float-mpfr,native_code_generation,bincode` enabled. For oneloopmaster, run the named permanent regression at `33a2e89`, patching Symbolica, Numerica and Graphica to the integration tree. These tests use `RUST_MIN_STACK=134217728` and a configured Symbolica license. The original standalone diagnostic is retained locally at `/tmp/ninjars-dual-primal-probe.rs`; detailed runtime logs are `/tmp/oneloop-c0-branch-fixed.log` and `/tmp/oneloop-pr5-branch-fixed.log`.

No D0 derivative result was established by this audit, and no NinjaRS master-domain gate was relaxed.
