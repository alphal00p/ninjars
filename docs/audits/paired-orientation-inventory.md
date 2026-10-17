# Paired bubble orientation inventory

Status: source-only experiment prepared from NinjaRS `c13cc2f`. It has not been
built, symbolically executed, numerically validated, or timed. The ignored test
must wait for explicit authorization after the active timing window. Dependencies
are the existing clean Symbolica `2c06f245`, FeynKit/gammaloop `474478a7`, and
oneloopmaster `c9dd82fd` checkouts; only this new worktree has dependency aliases.

## Algebra reuse checks, recorded before implementation

1. **Symbolica public API:** `ExpressionEvaluator::merge` in
   `references/symbolica/src/evaluate/optimize.rs:233` accepts another evaluator
   with the same parameters and concatenates all outputs. `Some(0)` disables
   common-pair rounds but still eliminates identical instructions; `Some(2)`
   bounds common-pair elimination. `AtomCore::replace_multiple`, documented at
   `references/symbolica/src/atom/core.rs:2173`, performs simultaneous replacement.
   `export_instructions` and `count_operations` expose the required inventory.
2. **Implementation/tests/examples:** merge checks parameter counts, not symbolic
   parameter identities, then combines constants/instructions/result indices and
   optimizes stack storage (`optimize.rs:239,475-495`). Both plans must therefore
   explicitly use the identical ordered canonical parameter vector. Existing
   `references/symbolica/tests/evaluation.rs:238-367` compares separate, joint and
   merged complex evaluators, including 256-bit values and both zero/unbounded
   common-pair rounds. `evaluate/evaluator.rs:1415` covers merged shared functions.
   The public replacement examples cover simultaneous swaps. The batch evaluator
   documentation describes row-major inputs/results, retained storage, lazy
   branches, and Dual flattening (`evaluate/batch.rs:3-12`).
3. **FeynKit/gammaloop uses:** the tested shared checkout already calls
   `tree.merge(eval, settings.cpe_iterations)` before numerical mapping in
   `crates/gammalooprs/src/integrands/process/evaluators.rs:2935`; amplitude and
   cross-section loading use the same API. Spenso's reference `EvalTensorSet`
   calls `merge(..., None)` in
   `references/gammaloop_feynkit/crates/spenso/src/tensors/parametric.rs:2974`.
   NinjaRS's existing bounded algorithm probe already uses simultaneous literal
   replacement to bind exact plans. No new optimizer, contraction, polynomial
   manipulation, numerical evaluator, or physical formula is justified here.

## Fixed graph boundary

The test rebinds existing `DenominatorExpansion::new(2)` and
`BubbleSourceProjection::new()` expressions. Their physics remains owned by
those exact plans and shared FeynKit momentum arithmetic. Each family is kept
independent; there is no denominator-to-projection fusion.

| Graph | Ordered canonical scalar inputs | Retained output order |
| --- | --- | --- |
| Denominator pair, 37 inputs | `a0[0..4], a1[4..8], growing0[8..12], shrinking0[12..16], growing1[16..20], shrinking1[20..24], beta[24..27], Vanchor[27..31], manchor2[31], Vline[32..36], mline2[36]` | first seven denominator powers, then the same seven swapped powers |
| Source-projection pair, 36 inputs | `a0[0..4], a1[4..8], growing0[8..12], shrinking0[12..16], growing1[16..20], shrinking1[20..24], Vtriangle[24..28], E3[28..32], E4[32..36]` | first eight projections, then the same eight swapped projections |

Every range is half-open and every four-vector retains `(E,x,y,z)` order. The
four transverse vectors are independent inputs. No swap/normalization helper is
composed, and no null-basis, mass, cut, transverse, or parent-origin identity is
assumed. The parent triangle origin remains independent of the denominator
anchor. All 30 original heads are retained, including duplicate expressions.

For each family, record two separately optimized evaluators on the same canonical
parameter vector, their summed instruction/arithmetic counts, and two merged
variants with `Some(0)` and `Some(2)`. Use the existing `ExactKernel::evaluator`
settings. Record individual input/output, instruction, addition, multiplication,
inversion, function-call, temporary, constant, and expression-text sizes. Separate
temporary sums are not peak storage. Merged expression text is unchanged because
merge operates on instructions; it is not a reconstructed optimized expression.
Arithmetic counts are Symbolica scalar-carrier operations, not real flop counts
or SIMD instructions. An additional original-parameter-plan control records any
effect of rebinding both orientations to the larger canonical parameter vector.

Exact checks rebind every mapped head back to its original parameter symbols
and require Symbolica's expanded difference to be zero. Ordered parameter maps,
original/rebound expressions, full head identities, and all input labels are
retained in the report. The check validates wiring, not independently derived
physics. It does not reconstruct or numerically execute merged instructions.

The cumulative retained expression text is capped at 8 MiB, with a required
external timeout of 180 seconds. The test writes partial reports before each
family and merge stage. Neither bound may be silently increased on failure. No
Rust arithmetic generation, numerical kernels, master provider, physical graph,
callback, quotient/correction composition, or production dispatch is included.

## Why inspect this after the negative full-network fusion result?

These two small polynomial graphs retain every output and have no master,
division, or branch graph inside them. Common `a0/a1` heads are visible without
eliminating diagnostics. Original Ninja calls its denominator expansion anew for
both orientations (`ninja_implem.cxx:1023,1080`) and computes longitudinal source
projections in both correction branches (`coefficient_level_subs.cc:52-56,298-302`).
Sharing this work is a new implementation opportunity; the separate swapped-head
pruning candidate restores pruning already present in Ninja.

For all-active n5/r4, each family has 60 orientation rows or 30 pair rows. All
420 denominator and 480 projection output carriers remain. Packing these proposed
layouts would use 1,110 instead of 1,740 denominator inputs and 1,080 instead of
1,680 projection inputs. These are layout counts, not measured performance.
The previous full-network retained-output loss neither proves nor disproves a
benefit here; this inventory alone also cannot establish one.

## Production issues deliberately left unresolved

Adjacent public stage rows are not proof that `a0/a1` or other inputs are shared.
The core currently clones equal vectors into both rows; arbitrary callers need
not do so. A production opt-in needs explicit paired provenance, or complete
component/precision-aware validation plus a batched unmatched fallback. Dual
primal equality is insufficient. No such API or dispatch is introduced here.

Keep denominator and source projection at their current separate validation
stages. Pair only eligible corresponding rows, preserve original row identities
after compaction, and retain unmatched valid orientations. Preserve all full
input/Dual checks, denominator finite/pivot checks, source provenance/containment,
source coefficient/projection checks, same-call reuse, and lane transactionality.
All outputs being retained does not prove unchanged overflow, signed zero,
rounding, or failure order after common-pair optimization.

Later numerical validation would map exact constants at explicit working
precision, then use the existing ordinary/complex and Symbolica Dual/batch
machinery. It must compare every component and all statuses, including mixed and
higher derivatives, heterogeneous magnitudes, partial pairs, zero/single/tail
batches, and grow/shrink/retry. A future native row bound would count complete
pair-kernel rows and must be reported accordingly. No present metadata claims
that any production pair kernel ran.

## Intended validation command, not yet executed

Compile the library test outside the numerical lock with jobs at most four.
After explicit authorization, invoke only
`paired_orientation_probe::inventory_paired_orientations` with `--exact --ignored
--nocapture --test-threads=1`, under `flock /tmp/ninjars-symbolica-tests.lock`
followed by `timeout 180s`. Set `NINJARS_PAIRED_ORIENTATION_OUTPUT` to a retained
JSON path. Archive the raw log, partial/final report, source/dependency identities,
executable and hash. No numerical or performance acceptance follows from this
symbolic instruction inventory.
