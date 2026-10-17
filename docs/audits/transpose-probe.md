# Bounded n3/r3 transpose experiment

Status: the bounded n3/r3 symbolic check has executed successfully. All five
exact output identities passed, with 60 independent callback heads and 39
symbolic master Laurent components. No numerical evaluator, timing or n3/r4
experiment has run. The initial source `c10279a` result and logs are retained in
`validation/evidence/transpose-probe/`; the subsequent source change only fixes
lint/style issues.

## Result

| Graph | Instructions | Additions | Multiplications | Inversions | Local temporary slots | Generated source bytes |
|---|---:|---:|---:|---:|---:|---:|
| Separate forward blocks | 1,954 | 636 | 1,500 | 81 | largest 30; sum 520 | 517,266 summed call sites |
| Fused forward | 1,075 | 429 | 991 | 36 | 132 | 249,531 |
| Factored transpose preparation | 3,054 | 828 | 1,969 | 75 | largest 47; sum 956 | 873,883 summed call sites |
| Factored transpose application | 305 | 295 | 300 | 0 | 65 | 118,218 |
| Factored transpose total | 3,359 | 1,123 | 2,269 | 75 | largest 65; sum 1,021 | 992,101 summed call sites |
| Fused transpose | 1,925 | 797 | 1,541 | 36 | 292 | 400,640 |

This is a negative result for transposition at one numerator per changing
phase-space point. Ordinary **forward fusion** reduces the count by 879
instructions (45%) relative to separate forward blocks. The fused transpose
still costs 850 more instructions than fused forward, and the factored
transpose costs 1,405 more than separate forward. These are symbolic graph
counts, not a performance claim. Instruction counts differ from summed scalar
arithmetic because evaluator instructions may have several operands.

The five-output fused graphs omit intermediate heads that the current
cut-failure, coefficient-diagnostic and reconstruction contracts inspect. A
production fused path therefore needs additional diagnostic outputs and checks,
or a deliberate fast-path eligibility contract with a full forward fallback.
The 45% graph-count reduction is not a directly deployable reduction speedup.
Fused forward also requires 132 local temporaries versus a maximum of 30 for
one separate forward block; cross-block storage remains unmeasured.

The [diagnostic-output follow-up](forward-diagnostic-fusion.md) retains all 283
existing forward block outputs and passes 288 exact identities. This conservative
fusion uses 3,177 instructions, exceeding the separate forward blocks by 1,223.
It reuses the same n3/r3 graph and adds no transpose or physical-geometry experiment.

For H numerators with genuinely fixed geometry and master values, an
**instruction-count-only** model gives `3054 + 305*H` for the factored
transpose. It falls below `1954*H` at H=2 and below `1075*H` at H=4
(strict thresholds H>1.852 and H>3.966). These thresholds are not timing
predictions: they omit master-selection differences, memory/packing, setup
and differing instruction costs. At changing kinematics the preparation must
be charged again for every point, giving 3,359 instructions at H=1. A negative
abstract-geometry result does not exclude cancellation after physical cut
geometry is composed, but that further experiment is deferred.

## Scope and construction

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
