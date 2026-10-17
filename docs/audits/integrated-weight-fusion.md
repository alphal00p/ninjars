# Integrated-only master-weight fusion inventory

Status: source-only proposal, based on `ea9a9da`. No build, symbolic execution,
numerical control, or timing has run. The coordinated timing hold remains in
force. This is an ignored n3/r3 symbolic experiment, not a production fast path.

## Algorithm boundary and algebra reuse audit

The proposed output is **20 ordered per-cut master-weight/rational heads**:
triangle `[C0, rational]`, each of bubbles 01/02/12
`[B0, B1, B11, rational]`, and each of tadpoles 0/1/2 `[A0, rational]`.
The existing `MasterContraction` determines every formula and master order.
These are 13 master weights and seven rational heads, not 20 master calls.
Master values remain absent from weight dependencies. Their values are supplied
only to the unchanged symbolic contractions in the exact recovery control.

Before implementing the inventory, the following three checks were made:

1. Symbolica's public `AtomCore::replace_multiple`, `Atom::evaluator_multiple`,
   and `ExpressionEvaluator::count_operations` provide substitution, joint
   optimization, and arithmetic inventory. `ExpressionEvaluator::merge` joins
   equal-parameter evaluator outputs; it does not substitute dependent blocks.
2. Their implementations in `src/atom/core.rs`, `src/evaluate/optimize.rs`, and
   `src/evaluate/evaluator.rs`, plus `tests/evaluation.rs`'s merged/joint complex
   evaluator controls and the public simultaneous-replacement example, confirm
   these operations already exist. N-ary arithmetic counts use their existing
   operation semantics; no local operation estimator is introduced.
3. FeynKit/gammaloop uses `tree.merge(eval, settings.cpe_iterations)` in process
   evaluators and Spenso's evaluator-set merging. NinjaRS already provides the
   `Network`, `forward_network`, literal substitution, `ExactKernel`, exact
   `MasterContraction`, and shared oneloopmaster Rust emitter used here.

The new child module only selects graph blocks and boundaries, binds the
existing expressions, and asks Symbolica to substitute, expand, optimize, and
generate its instruction graph. It implements no algebra optimizer, polynomial
arithmetic, reduction formula, or hand-written native kernel. The original
transpose and retained-diagnostic probes remain unchanged except for registering
this sibling child module.

## Selected comparisons and exact controls

The separate baseline records the existing forward blocks once, retaining both
the historical full total and a **pre-master total** that excludes the seven
five-output master contractions and the final five-output sum. Its 243 summed
heads include current intermediate diagnostics; this is a comparison boundary,
not a requirement that an integrated-only algorithm retain all of them.

The two preselected candidates are:

- One evaluator with 20 master-weight/rational outputs, with no master inputs.
- Three distinct evaluators: triangle **10** residue heads, bubble **30**
  residue heads, and final **20** weight/rational heads. The bubble evaluator
  takes the triangle ten as an explicit parameter prefix. The final evaluator
  takes the triangle ten followed by the bubble thirty as its explicit prefix.
  Later stages do not re-evaluate earlier blocks. External suffixes retain the
  original canonical order and include inputs used by the selected original
  blocks, including slots eliminated after fusion. All 60 callback inputs remain
  present across the three stages; callback APIs are not shortened by this probe.

The report lists the exact prefix/suffix symbols, callback subsets, expression-
live inputs, and retained unused ABI inputs. Counts
for the staged candidate sum **three actual evaluator inventories**, not one
globally fused 60-output graph. Forty parent-boundary values may be kept in
reusable buffers. The maximum individual temporary count and their sum are
reported separately; neither is a measurement of packing or peak memory.

An independent full-network composition provides reference expressions, but is
not counted as a candidate evaluator. Every one of the 40 parent boundaries and
20 final weights is checked after substituting the staged boundary parameters.
All 20 weights-only outputs are also checked. Ordered master/cut identities and
the unchanged local contraction bindings are asserted. Both candidates recover
the five original integrated expressions through those unchanged contractions
and the unchanged amplitude sum. Every exact difference must expand to zero.
Unresolved intermediate symbols and any master dependency in a pre-master plan
are rejected. These are graph-wiring controls, not an independent derivation of
the shared reduction formulas.

## Runtime contracts beyond this experiment

True runtime constraints include valid input shapes and all-component finite
inputs; topology/provenance; supported rank and denominators; valid geometry,
pivots and branch decisions; and nonanalytic master domains. They stay outside
this abstract geometry network. Weight checks must inspect **all Dual
components**: a zero primal with nonzero derivatives must request its master.
Lazy master execution remains after weight calculation and cannot be replaced
by eagerly evaluating every master mentioned in the topology.

Today the reducer additionally rejects nonfinite quotient, correction,
subtraction, and residue heads. Those checkpoints describe the current
algorithm, rather than mathematical necessity for a different opt-in algorithm.
An avoided dead intermediate need not recreate its overflow, provided an
independent numerical and stability validation establishes the alternative's
accuracy. This inventory neither promises nor tests such validation.

The staged triangle/bubble boundaries are justified by downstream source
residues and callback eligibility. They offer places to validate parents before
requesting lower callbacks, while avoiding every old transient coefficient.
The one-shot 20-output option instead needs an explicitly eligible pure
polynomial adapter, or an explicitly changed callback/failure contract: arbitrary
callbacks currently see staged active lanes after parent success. The staged
inventory still does not encode validation order within a cut size, callback
errors, lane compaction, transactionality, numerical zero masks, or retry.
Default execution remains unchanged. Cut-stop and reconstruction/complete-residue
diagnostics require explicit fallback; an integrated-only implementation must
never present stale residue arrays as computed results.

## Bounded later execution

Each candidate-stage composition, reference composition, and integrated
recovery has its **own** cumulative 8 MiB expression-text cap, matching the
original `Network::compose` per-call policy. This is not one global memory bound.
The report records each candidate's composition bytes, the three-stage sum, and
the sum across all four candidate compositions. These aggregate counts exclude
the separately capped reference/recovery calls and are not resident memory.
The identity groups instead share one cumulative 8 MiB exact-rebinding budget.
Partial JSON is written before each major stage, immediately after each
completed stage layout, and after each recorded block. Single Symbolica
operations are not interrupted by a byte check, so the
one ignored test must later run under the existing shared numerical lock with
an **external 180-second timeout**, one test thread, and no production numerical
execution. A limit failure remains an incomplete result; do not increase bounds
or add topologies automatically.

After root releases the timing hold, compile outside the numerical lock in an
existing warm cache with at most four build jobs, then run only
`algorithm_transpose_probe::integrated_weights::inventory_integrated_weights_n3_r3`
with `NINJARS_INTEGRATED_WEIGHTS_OUTPUT` set to a retained JSON path. Preserve the
command, source/dependency revisions, executable identity, raw log, partial or
complete JSON, and exit status. No runtime or performance claim follows from
instruction counts, emitted source bytes, or the symbolic check's elapsed time.
