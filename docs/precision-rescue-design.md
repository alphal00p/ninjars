# Proposed precision rescue after the Hgg cancellation probe

This is a design proposal, not implemented behavior or acceptance evidence.

## Evidence and numerical boundary

The Hgg longitudinal diagnostic contains tadpole single-pole contributions near
+1.0789e7, -7.1927e6 and -3.5964e6. Small imaginary errors in extracted residue
weights are magnified by the A0 pole m²=29929. Bubble contraction also subtracts
terms near 9.3878e5. The f64 failure therefore starts before the final sum.
Reevaluating only accumulation in a wider carrier cannot recover the lost data.
Complete DoubleFloat evaluation from the exact prepared expressions passes the
fixed 1e-9 comparison; reference original-Ninja binary64 also misses the
longitudinal zero-pole control. These observations support full-lane precision
rescue; they do not establish a general error bound.

A controlled rescaling of the bubble references from
`(1, ±2/3, ±2/3, ±1/3)` to the same directions `(3, ±2, ±2, ±1)` removed
fractional reference constants but worsened the largest Hgg scaled error from
`1.0669202114e-9` to `1.1779333906e-9`. Both variants passed 32 of 36 matrix
rows; the integer variant also passed all 16 focused analytic, Dual and existing
reduction controls. The integer change was reverted. This rules out that simple
constant rescaling as a sufficient remedy; it does not establish that every
alternative basis choice would perform worse.

## Reuse existing code and types

- Keep core `Amplitude<T,K>` and `Numerator<T>` generic and fixed to one carrier.
  Preserve its current per-lane transactional accumulation and failure statuses.
- Add conditioning/reconstruction diagnostics separately from geometric failure.
  If contribution magnitudes or residuals are added, construct their arithmetic
  with the existing Symbolica `ExactKernel`/batch evaluator and FeynKit types.
  Keep primal chart decisions and every Dual component's validity checks explicit.
- Implement an explicit CLI policy using existing `PreparedDocument`,
  `NumericalPool<T>` and `EvaluationSession<T>` from `runtime.rs`. Lazily create
  one session per selected precision and compact only requested rescue lanes.
  The exact numerator template, parameter bindings and input Atoms already exist.
- Build the DoubleFloat or Float numerical pool directly from those Atoms. Do
  not widen f64 values or reuse f64 residue/master results. Run the complete
  batched pipeline, including callbacks, geometry, division, subtraction,
  contraction and accumulation, then restore original lane order.
- A library caller with an external numerical numerator callback must supply its
  own callback/input source at the rescue precision. A fixed f64 callback cannot
  promise additional accuracy. No new momentum, polynomial or tensor type is
  needed for this protocol.

## Explicit policy and reporting

An initial implementation can expose requested rescue lanes or an opt-in retry
for existing unstable statuses; both have defined triggers. A future automatic
conditioning policy needs separately validated thresholds and reconstruction
checks. Neither near-cancellation nor a small final value alone proves failure.
Do not silently rescue after comparing against a reference value, and do not
claim `ErrorPropagatingFloat` estimates are certified bounds.

Each lane should retain initial status/result, trigger, attempted precisions,
actual output carrier/significant bits, rescue outcome and cost. Fixed-precision
mode remains reproducible. A failed rescue preserves the caller's accumulated
result. For CLI one-shot evaluation, rescue sessions start from zero; reusable
library accumulation must explicitly synchronize the prior result or commit
only the newly evaluated contribution through the owning accumulator.

JSON may contain different actual precisions by lane; report them rather than
rounding every rescued result back to f64 and concealing the rescue. Benchmarks
must distinguish native evaluations from rescue work, use the same policy for
both implementations, and include the rescue cost/rate in complete timings.

## Existing numerical mechanisms and limits

Numerica `DoubleFloat` supplies compensated arithmetic, including
`from_compensated_sum`, and already works throughout the complete pipeline.
Symbolica `Float` preserves working precision and tracked significant bits.
Numerica `ErrorPropagatingFloat<T>` supplies `new`, `new_with_accuracy`,
`get_absolute_error` and `get_num`, and implements real/numerical traits;
Symbolica evaluation tests show coefficient mapping to it. However its error
storage is f64 and equality does not compare the error estimate. It is not a
validated arbitrary-precision stability guarantee or cache key. Add independent
controls before using it to choose rescue lanes.

## Required tests before enabling an automatic policy

1. Reproduce the fixed Hgg longitudinal cancellation gate, preserving the native
   failure while demonstrating higher-precision recovery from exact inputs.
2. Mixed good/failed/canceling lanes, empty and tail batches, lane permutations,
   and unchanged unaffected rows.
3. Inputs with information below binary64 precision to detect widening mistakes.
4. Failed second attempts and transactional accumulation/reset across retries.
5. Higher/mixed Dual carriers, explicit unsupported master derivative domains,
   and full-shape validation without primal-only equality.
6. Repeated/changing inputs, shared cache policies, rescue counts and total cost.
7. Independently chosen conditioning thresholds tested away from Hgg; no
   tuning against reference answers or tolerance relaxation.
