# Bounded n3/r3 transpose experiment

Status: source prepared; **not built or executed**. The active measurement
window forbids builds, symbolic generation, numerical runs and timing. There
are no operation counts, exact identity results or speedup claims yet.

The ignored crate-internal test `inventory_transpose_n3_r3` starts with all
Laurent callback heads for one triangle, three bubbles and three tadpoles.
It composes the existing `LaurentDivision`, `ResidueProjection`,
`TriangleBubbleSubtraction`, `TadpoleSubtraction`, and `MasterContraction`
plans. The ordinary tadpole projection is the existing selection of the
constant head into c[0]; the four unused slots are zero. The experiment does
not duplicate a higher-rank tadpole formula. It includes both bubble and
triangle orientations, all ordinary correction heads, explicit rational terms,
and five outputs: finite, single pole, double pole, cut-constructible finite,
and explicit rational.

Each local plan is rebound into a unique cut/edge/orientation namespace.
Numerator callback heads are independent inputs. Source projections retain
source-cut identity, including its origin. Denominator rows are shared only
for an identical target, orientation and extra propagator, matching the
same-call reuse contract. The integration masses and pair invariants are
shared according to physical propagator indices. Otherwise geometry remains
abstract: no relation between independent projected coordinates or cut bases
is assumed. This establishes an algebraic comparison of the data flow, not
physical cross-cut cancellations.

The probe prepares four controls with the same `ExactKernel::evaluator`
settings and the existing oneloopmaster Rust emitter:

1. Separate forward blocks, preserving currently calculated unused heads.
2. Fused forward expressions for the five integrated outputs.
3. A factored reverse traversal, followed by application to callback heads.
4. Fused reverse expressions for the same five outputs.

Every local linear map is extracted with Symbolica's polynomial coefficient
API. The probe rejects affine terms, nonlinear terms and hidden dependence on
the data variables, then checks its linear reconstruction with an exact
Symbolica expansion. Reverse traversal propagates all five adjoints together;
it does not explicitly form a dense product such as A12*A23. The eventual
forward/reverse comparison requires every difference to expand to literal zero.
Statistical zero testing is disabled. Master Laurent components remain
independent symbolic inputs, so this identity also distinguishes the
coefficient of each master and the rational contribution.

The report records optimized arithmetic/instruction counts, temporary slots,
expression bytes and generated Rust bytes for every block. It separates
reverse-map preparation from application. Summed temporary slots are not peak
memory, and summed per-call source sizes are not the current library's binary
size: existing runtime kernels are reused across cuts. Counts include master
weight/contraction arithmetic but exclude actual master evaluation, numerical
geometry construction and preparation of numerator callback heads. A cumulative
8 MiB expression-text cap stops composition. The eventual executable must also
run under a **180-second wall-time limit inside the numerical lock**, because
the byte cap cannot interrupt one expensive expansion, coefficient extraction
or optimization. Initial and forward JSON checkpoints plus per-stage log lines
identify incomplete work. Retain the partial checkpoint, log and failure/timeout
exit status as a negative result if either bound is reached.

For one numerator at changing kinematics, reverse-map preparation belongs
inside the reduction cost. For several numerators at fixed denominator
kinematics, future measurements must report preparation and application
separately. A reduction in the fused forward graph already captures ordinary
cross-kernel common-expression savings; it is not evidence for transposition.
If reverse arithmetic or expression size grows, record that result and stop
before extending to n3/r4 or inserting physical geometry. A negative result
with independent abstract geometry does not rule out cancellations after
actual cut geometry is composed; that is a distinct, currently deferred
experiment.

Production constraints remain unresolved by this exact experiment. A reverse
map seeded with evaluated masters could evaluate an integral whose coefficient
would have been zero; it must preserve lazy master selection, derivative-only
weights and nonanalytic/failure behavior. Required-cut stability, per-lane
transactionality, batching, all Dual components, overflow/conditioning,
cut-stop, reconstruction and coefficient diagnostics need independent controls
or the original forward fallback. Symbolic master values here do not authorize
eager production master calls.

## Algebra reuse check

Before implementing the graph orchestration, the following sources were
checked. No new polynomial, series, differentiation or expression optimizer is
implemented by this probe.

- Public API: Symbolica `AtomCore::replace_multiple`, `expand`, and
  `to_polynomial_in_vars_with_field` (`src/atom/core.rs`), plus
  `Atom::evaluator_multiple` and evaluator instruction export/counting.
- Implementation and examples: Symbolica's polynomial conversion/coefficient
  implementation, `examples/coefficient_ring.rs`, evaluator `dual.rs`, and
  evaluator optimization code. `Dualizer` transforms evaluator bodies for
  ordered Dual layouts; it is not a general reverse linear-map interface.
  The inspected evaluator APIs expose no reusable arbitrary kernel-network
  transpose facility.
- Existing consumers: NinjaRS `laurent.rs`, `subtraction.rs` and
  `algorithm_rank_probe.rs`; FeynKit `symbolica-utils::PatternReplacement`,
  generator grouping substitutions and the existing CFF differentiation code.

After the measurement window is released, compile without running tests first.
Run only the produced ignored-test executable under
`/tmp/ninjars-symbolica-tests.lock`, with one test thread and
`NINJARS_TRANSPOSE_INVENTORY_OUTPUT` pointing to a retained artifact. Archive
the exact source/dependency revisions, executable hash, log, output and any
failure before considering numerical or performance follow-up.

The command order must be `flock LOCK timeout --signal=TERM --kill-after=10s
180s TEST_EXECUTABLE --ignored --exact
algorithm_transpose_probe::inventory_transpose_n3_r3 --test-threads=1
--nocapture`. The timeout starts only after acquiring the lock. Capture the
exit status directly; do not hide it behind `tee` without `pipefail`. Exit 124
or forced termination is an incomplete experiment, never an identity pass.
