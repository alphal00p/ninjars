# NinjaRS using shared FeynKit and Symbolica types

## 1. Goal and startup

**Goal:** Implement a Rust library and CLI providing Ninja-compatible one-loop reduction, with batching throughout phase-space evaluation, cut sampling, numerator callbacks, reduction, and master integrals. Reuse FeynKit’s physics types and Symbolica’s algebra. Support f64, DoubleFloat, arbitrary precision, and Symbolica Dual types with higher and mixed derivatives. Validate against original Ninja and MadLoop, meeting the agreed correctness and performance gates.

At implementation start:

1. Write this complete plan to `/shared/external/ninja/references/PLAN.md`.
2. Register the goal above using `create_goal`, without a token budget.
3. Record dependency revisions, upstream patches/PRs, validation evidence, and outstanding work in the plan.
4. Establish a reproducible Rust ≥1.96 development environment and build the reference tools.

Use the [official Ninja code](https://github.com/peraro/ninja), [Laurent-reduction paper](https://arxiv.org/abs/1203.0291), [Ninja paper](https://arxiv.org/abs/1403.1229), and [tensor-interface paper](https://arxiv.org/abs/1604.01363) as references.

Production implementation and numerator preparation remain Rust. Symbolica’s native numerical dependencies are allowed; reference C++/Fortran/Python programs remain validation tools.

## 2. Shared types and package boundaries

The inspected FeynKit types will become the canonical representations used by Ninja:

| Concept | Representation to reuse |
|---|---|
| Numerical momenta | `feynkit_kinematics::{FourMomentum<T>, ThreeMomentum<T>, Energy<T>}` |
| Momentum routing | `Signature`, `MomentumSignature`, and FeynKit’s `LoopMomentumBasis` |
| Symbolic scalar products and assumptions | `feynkit_kinematics::Kinematics` |
| Ordered symbolic denominators | `feynkit_graph::IntegralFamily` |
| Quadratic denominator decomposition | `IntegralFamily::quadratic_denominator` and `QuadraticMomentum` |
| Verified routing transformations | `IntegralMapping`, `PropagatorMapping` |
| Symbolic numerators | Symbolica `Atom` and `MultivariatePolynomial`, as used by FeynKit |
| Tensor numerators | Idenso `SymbolicTensor` and Spenso `ParamTensor`/`DataTensor` |
| Tensor evaluation | Spenso `EvalTensor` and Symbolica evaluators |
| Models and physical states | FeynKit model, parameter-binding, projector, and wavefunction APIs |

Use these types directly or re-export them. Do not introduce Ninja-owned momentum, symbolic denominator, polynomial, or tensor representations.

FeynKit’s `FourMomentum<T>` already has the correct component order and (+---) metric. Its basic arithmetic supports complex and Dual carriers through ordinary arithmetic bounds. Keep real-only `KinematicScalar` bounds out of complex cut kernels.

Use existing momentum types for both batch layouts:

- Public sample input: `&[FourMomentum<T>]`.
- Internal component-contiguous storage: `FourMomentum<Vec<T>>`.
- Borrowed component views: `FourMomentum<&[T]>`.

Validate component lengths at the batch boundary. Avoid a new batch physics type unless an actual missing operation requires one.

Maintain four packages:

- **`ninja`:** reduction interfaces, topology plans, workspaces, kernels, master-provider interface, and results; uses shared FeynKit types.
- **`ninja-symbolica`:** thin polynomial/tensor adapters implementing Ninja’s reduction callbacks over existing representations.
- **`ninja-graph`:** independent DOT/model/routing/numerator preparation, composed from FeynKit APIs.
- **`ninja-cli`:** orchestration, reference adapters, comparisons, and benchmarks.

Ninja’s `Numerator` remains a reduction-specific callback contract. Its implementation consumes shared numerator data. Graph loading, Feynman rules, and physical numerator preparation stay outside the reduction engine.

## 3. Reduction, algebra, and upstream extensions

Preserve Ninja’s semantic API: amplitude accumulation/reset, tensor and callback numerator interfaces, S-matrix and scale controls, cut stopping, rational separation, reconstruction diagnostics, and per-lane stability status.

Support:

- Ranks R ≤ n+1, massless/real/complex squared masses, and explicit μ² numerators.
- Batched `evaluate`, `muExpansion`, `t3Expansion`, and `t2Expansion`, retaining cut partitions and expansion depths.
- Results ordered `[finite, single pole, double pole]`.
- Denominators (q+V_i)²−m_i²−μ², with μ² carrying rank weight two.
- Explicit rejection of unsupported ranks, raised/repeated loop propagators, and nonquadratic loop denominators.
- Unchanged accumulated results for lanes whose evaluation fails as unstable.

Cache topology-only partitions, complements, containment relationships, and coefficient layouts. Keep numerical bases, branch choices, and stability checks lane-local. Identify callback requests by phase-space lane, cut, expansion, and sample; compact active lanes and restore their order.

Arithmetic must process batches throughout the pipeline. Arbitrary precision uses batched scalar operations; SIMD is an independently validated specialization. Use reusable workspaces and a warmed batch-size-one path without packing allocations.

**Algebra policy:** before adding any algebraic operation, document checks of Symbolica’s public API, its implementation/tests/examples, and FeynKit/gammaloop usages. Use their polynomial, series, contraction, differentiation, and evaluator facilities. Generate Rust arithmetic mechanically from Symbolica instruction graphs using the existing oneloopmaster approach.

**Focused FeynKit/gammaloop PRs:**

- Expose the existing ordered propagator extraction as `(EdgeId, Atom)` entries, and make `propagator_family` delegate to it. This preserves provenance without duplicating denominator-selection rules.
- Expose Spenso’s existing `EvalTensor::evaluate_components` functionality through a supported buffer API; extend it for batched output where necessary, preserving sparse-index reconstruction.
- Add precision-preserving model parameter/evaluation bindings. Current `ComplexValue`, parameter cards, and evaluator values use f64; high-precision and Dual inputs must not pass through them.
- Relax momentum `zero` constructors to the numerical traits they actually require, so complex and Dual values retain precision and derivative shape.

Keep changes in their existing owning crates. Any further shared capability belongs upstream rather than in a parallel Ninja abstraction. Follow FeynKit’s feature-workspace workflow, open focused PRs, attach them to this chat, and pin the port to the tested changes while review proceeds.

**Symbolica and oneloopmaster:**

- Unify Symbolica, Numerica, and Graphica on compatible pinned revisions.
- Reuse existing oneloopmaster scalar and tensor masters, including `b1`, `b11`, `b111`, and `an`.
- Extend master execution to carry batches through arithmetic and helper calls, preserving lazy branches.
- Support Dual transformations through shared evaluator bodies recursively.
- Audit complex derivative primitives, continuation helpers, and higher/mixed components.
- Generate neighborhood expansions before specializing removable limits such as equal masses and zero invariants. Report genuinely nonanalytic derivative points explicitly.
- Make necessary Symbolica patches minimal, independently tested, and submitted as upstream PRs.

Construct constants directly at working precision. Check all Dual components for validation and cache identity: Symbolica Dual equality compares primal values and is insufficient for these purposes.

## 4. CLI preparation and reference validation

Provide `prepare`, `reduce`, `compare`, and `bench`, accepting DOT graphs, models, parameter bindings, phase-space samples, precision, batch size, and diagnostics. Produce readable summaries and reproducible machine-readable results.

Preparation will:

- Import synthetic numerators from DOT `num` attributes.
- Apply FeynKit’s Feynman rules, projectors, wavefunctions, routing, and Symbolica/spenso/idenso contraction for physical graphs.
- Use `propagator_family`, which preserves physical propagators. Do not append the auxiliary denominators introduced by `integral_family`.
- Use `QuadraticMomentum` and `Kinematics::linear_terms` for exact decomposition.
- Preserve denominator normalization factors. Move loop-independent internal propagators into the overall prefactor using Symbolica.
- Apply graph weights, numerator prefactors, and projectors exactly once.
- Pass shared numerator and propagator data to both Rust Ninja and the reference adapters.

Validation progression:

1. Explicit bubble and triangle polynomials.
2. Tadpole, box, pentagon, and six-propagator cases, including R=n+1, complex masses, and explicit μ²/μ⁴.
3. Top-quark H→gg triangle graphs.
4. Top-quark γγ→γγ box graphs.
5. Top-quark gg→HH triangle/box graphs.
6. Top-quark gg→HHH pentagon graphs.

Compare synthetic inputs with original Ninja using identical prepared polynomials and propagators. Check callbacks/coefficient extraction separately from complete original Ninja–OneLOop evaluation.

For physical graphs, compare independently prepared FeynKit and MadLoop numerators before integrated results. Match graph identity, orientation, helicity, color projection, routing, parameters, scale, and normalization. Disable grouping and hidden backend switching; extract graph contributions before squaring or Born interference.

The physical convention remains **four-dimensional numerators with R1 retained, excluding explicit R2 and UV counterterms**. Start with zero widths; cover complex masses in synthetic tests.

## 5. Milestones and acceptance

Implement in validated stages:

1. Shared-type integration, upstream extensions, dependency unification, reference harness, and generic/batched/Dual master probes.
2. Bubble/triangle CLI preparation and reduction parity.
3. Full Ninja coverage through R=n+1.
4. Complete batching and derivative support.
5. Physical validation through pentagons.
6. Performance optimization, independent audits, documentation, and final acceptance.

Delegate parallel audits of shared APIs, numerical correctness, reference matching, and performance.

**Correctness gates**

- Validate every Laurent coefficient, rational decomposition where available, and supported reconstruction checks.
- Explicitly document original Ninja’s incomplete higher-rank global reconstruction coverage; independently validate higher-rank callbacks and integrated results.
- Test empty/single/tail batches, permutations, mixed branches, failed-lane isolation, and scalar/batch equivalence.
- Cover scaleless integrals, equal masses, zero invariants, thresholds, near-Gram degeneracy, complex masses, accumulation/reset, and both μ² extraction modes.
- Verify first, higher, and mixed derivatives using symbolic differentiation, analytic controls, and converged high-precision differences.
- Add upstream regressions for complex/Dual momentum arithmetic, exact parameter bindings, propagator order/provenance, and evaluator buffer behavior.
- Fix precision-dependent tolerances before optimization: ordinary f64 comparisons use a scaled 10^-9 relative criterion; higher precisions must demonstrate additional accuracy. Diagnose exceptional points separately.
- Run relevant tests, formatting, Clippy, and dependency regression suites.

**Performance gates**

- Compare optimized builds on the same host and thread count, without fast-math, using matching accuracy and cache policies.
- Require f64 batch-size-one latency within 5% of original Ninja and equal-or-better batched throughput on the fixed suite.
- Measure preparation, callbacks, reduction, masters, and complete evaluation separately; exclude setup and IPC from steady-state comparisons.
- Include batch sizes 1, 8, 32, 128, and 1024, incomplete tails, changing phase-space points, and repeated-point workloads.
- Report per-case timing uncertainty and precision-rescue rates.
- Compare reduction with a shared master provider and complete evaluation with production backends.
- Benchmark DoubleFloat, arbitrary precision, and Dual separately, recording accuracy and derivative order.

**Additional performance scope (user clarification, 2026-10-16)**

- Retain reduction-only NinjaRS versus original Ninja measurements with the same
  prepared numerator provider and matching master policy. Also measure complete
  end-to-end evaluation of a fixed large physical graph, including numerator
  coefficient evaluation, reduction and masters. Report cold graph/model
  preparation and compilation separately from warmed repeated/changing
  phase-space evaluation; neither component may disappear from the report.
- Add a CLI-selectable numerator preparation path using one multi-output-head
  SymJIT evaluator at optimization level O2 for the polynomial coefficients,
  reusing the existing Symbolica/FeynKit evaluation APIs. Evaluate those
  coefficients once per phase-space sample before cut callbacks. Keep this
  capability in the CLI's numerator-preparation package, with ordinary shared
  polynomial data and NinjaRS callbacks at the boundary. The reduction engine
  acquires no graph/model/JIT responsibilities.
- Validate the compiled numerator against the exact/preexisting provider before
  timing, and explicitly retain precision-preserving paths for carriers not
  supported by the selected JIT backend. Record compilation/cache policy,
  coefficient preparation, callbacks, reduction, masters and total evaluation
  for both providers. End-to-end speed and accuracy are independent acceptance
  evidence from reduction-only speed.

Completion requires reproducible correctness/performance evidence, usable library and CLI examples, recorded Symbolica and FeynKit/gammaloop PRs, and an updated plan with all remaining limitations stated.

## Implementation record

Started 2026-10-16. Goal registered and active. This document contains the accepted plan; completed milestones below require recorded evidence.

User naming update: the product is **NinjaRS**, the library and CLI are `ninjars`, and supporting crates are `ninjars-symbolica`, `ninjars-graph`, and `ninjars-cli`. This supersedes the package spellings in the originally accepted plan above.

Toolchain established with pinned `flake.nix`/`flake.lock`; `nix develop path:. --command cargo --version` reports Cargo 1.98.1. FeynKit changes live in the task-owned `references/gammaloop_ninja_shared` worktree. Exact `Model::scalar_bindings` overrides already preserve symbolic parameter values; reuse and regression-test them instead of introducing another parameter schema.

### Reference revisions

| Reference | Initial revision |
|---|---|
| Ninja | `05fb0d3a0f47f216eceef411a09ac19e1b9a19f0` |
| Symbolica | `c78cffc20f318df1fce50539a587cbe8ffc717c0` |
| oneloopmaster | `45cf79e6c8d1e052f655ca4e26ff299b661986a6` |
| FeynKit/gammaloop | `d3b91d31461e50d91dce4702d68c45294faca350` |
| MadGraph | supplied `MG5_aMC_v3_7_0` tree |

### Progress

- [x] Save accepted plan and register active goal.
- [x] Establish pinned Rust toolchain and core dependency build (full workspace validation follows).
- [x] Initial shared FeynKit API changes and upstream PR.
- [x] Initial Symbolica evaluator/Dual changes and upstream PR.
- [ ] Generic batched reduction and numerator callbacks.
- [ ] Graph preparation and reference CLI.
- [ ] Numerical, derivative, physics, and performance acceptance gates.

### Algebra reuse audit

Before implementing symbolic operations: use Symbolica `AtomCore` polynomial conversion and substitution, `MultivariatePolynomial` division/coefficient access, `Series` expansion, evaluator Horner/CSE and `Dualizer`; verify implementations/tests under Symbolica `src/poly` and `src/evaluate`; reuse FeynKit `Kinematics::linear_terms`, `IntegralFamily::quadratic_denominator`, Symbolica-utils polynomial emission, and Spenso/Idenso tensor contraction. No replacement algebra engine is authorized.

### Initial publication milestone — 2026-10-16

User authorized pushing to `git@github.com:alphal00p/ninjars.git` and committing/pushing subsequent validated milestones. Commit author: `ValentinHirschi <valentin.hirschi@gmail.com>`. The root remote is configured. GitHub CLI authentication is available; pushes use its HTTPS credential helper because no SSH identity is loaded in this session.

- `cargo check -p ninjars` passes with Rust/Cargo 1.98.1. Core topology partitions and explicitly batched numerator callback contracts use shared FeynKit momenta.
- The core six-propagator partition test, five DOT-preparation tests, and five CLI tests pass, checking cut containment, physical propagator order, single prefactor application, exact component routing, bridge factors, JSON framing, sub-f64 parameter values, and invalid input rejection. `ninjars prepare` runs on both bubble and triangle fixtures.
- FeynKit patch commit `1cdae7cad7ad6384b789a68ca969020256191558`: 14 focused tests pass serially (complex/static/dynamic Dual momenta, graph integral provenance, exact bindings, and sparse tensor batch output). Original reference checkout remains unchanged.
- Symbolica patch commit `336db77cbdd30802603427e649902ae053c95530`: 12 focused Dual tests pass serially, including complex absolute-value derivatives, nested shared bodies, higher/mixed components, lazy branches, evaluator merging and serialization.
- Symbolica batch evaluator commit `2d1255653f6a83580ce5a9f9449e3838f6053f11`: five focused batch tests pass, covering shape/overflow checks, empty/single/tail batches, changing batch sizes, nested lazy helpers, instruction-major callback order and complex f64/DoubleFloat/192-bit evaluations with second derivatives.
- Original Ninja plus bundled OneLOop 3.4 builds successfully. Three reference-protocol regression tests pass. Constant massive bubble and spacelike triangle smoke results are recorded by the harness; these are reference checks, not Rust reduction parity.
- `dependencies/manifest.json`, checked-in upstream patches and `tools/bootstrap.py` make the source dependencies recoverable independently of the supplied checkouts.
- FeynKit changes submitted as draft [gammaloop PR #142](https://github.com/alphal00p/gammaloop/pull/142) and attached to this chat. Focused Clippy passed with existing dependency/macro warnings.
- Symbolica changes submitted as [Symbolica PR #82](https://github.com/symbolica-dev/symbolica/pull/82) and attached to this chat. Follow-up `62e546cb` corrects direct static/dynamic Dual complex norms and normalizes modulus composition for extreme finite values. All 13 evaluator Dual, 7 Numerica Dual and 5 batch tests pass. Clippy passes after allowing an existing `never_loop` violation in `src/poly/factor.rs:11853`; baseline warnings remain.
- Bootstrap succeeds against the pinned dependencies. Applying each checked-in patch to its original base in an isolated Git index reproduces the corresponding patched commit tree exactly.
- Reduction kernels, complete CLI, generic/Dual master support, physical comparisons, and all performance acceptance gates remain outstanding. This initial publication is not a completed implementation.

Initial snapshot published on `main` as `2ecdef3` at [alphal00p/ninjars](https://github.com/alphal00p/ninjars). Subsequent commits are pushed after their milestone checks pass.

### Callback and master milestone — validated foundation

- Symbolica-backed polynomial callbacks use its signed `MultivariatePolynomial` support for finite Laurent series. Rank validation uses mu-squared weight two; callbacks preserve lane identity and caller-requested raw powers. Five tests pass, including exact coefficient controls, reconstruction, invalid input isolation and DoubleFloat/256-bit coefficient retention beyond f64.
- Static and dynamic Dual carriers are supported through a minimal shared Numerica `DualNumber<T>` component-view trait. Upstream follow-up `c0a4aaf2` is pushed to Symbolica PR #82; its component-layout regression passes. Symbolica stores Taylor coefficients: repeated derivatives require the corresponding factorial conversion when compared with analytic derivatives. No primal-only equality is used to validate derivative results or shape.
- Batched oneloopmaster adapters cover scalar and available tensor master expressions through Symbolica's batched interpreter. Ten probes pass, including independent original-Ninja bubble/triangle values, changing batch sizes/branches, tensor poles, higher precision, static/dynamic Dual parity, and higher/mixed Dual coefficients. Equal-mass bubble transverse first, repeated second, and mixed components agree with converged Float192 differences at steps 1e-8 and 1e-12; successive estimates agree within 1e-16.
- Master derivative support currently excludes triangle/box derivatives, zero-invariant removable limits, non-real seeds and other unvalidated continuation regions. Invalid master requests preserve the whole output buffer; lane-specific rescue remains for the reduction engine.
- Optional `reference-codegen` uses Symbolica's own C++ exporter on the same exact callback expressions. Two reference callback tests pass, compiling against original Ninja headers and checking all expansion-depth prefixes, both mu modes, complex parameters, explicit mu-squared/mu-fourth terms, and output-buffer bounds. Generated callbacks currently compute full coefficient lists, so they are not a final performance baseline.
- An original Ninja exceptional point is retained for diagnosis: rank-two triangle with equal unit masses and shifts `(0,0,0,0)`, `(0,1,0,0)`, `(0,0,1,0)` can return status zero with nonfinite coefficients. This occurs for both original tensor `q0^2` and generated mu-squared callbacks, while scalar rank zero is finite. Reference comparisons must reject nonfinite results independently of the status code. Stable timelike-shift points are used for initial integrated callback controls.
- The first physical H-to-gg preparation probe imports the SM model and applies Feynman rules, but reports open external color indices. Explicit color/helicity projection and independent MadLoop graph matching remain unfinished; this is not a physical validation success.

Final checks for this foundation:

- `RUST_MIN_STACK=134217728 cargo test --workspace --all-features --locked -- --include-ignored --test-threads=1`: **30 tests pass** (1 core, 5 CLI, 5 graph, 2 Dual callback, 10 master, 5 polynomial, 2 generated callback).
- `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings`: passes for the workspace. Existing upstream Symbolica/Numerica warnings remain. Known arithmetic lints generated by Symbolica's Dual macro are scoped to the affected test modules.
- `cargo fmt --all --check` and `git diff --check`: pass.
- Original reference protocol: **4 tests pass**. Integrated generated callback controls: **5 checks pass**, including triangle mu-squared = +1/2, box mu-fourth = -1/6, ordinary polynomial/TensorNumerator parity, changing complex parameters, and higher-rank integrated linearity. The latter is explicitly not an independent closed-form higher-rank result.
- Full reduction kernels, accumulation/stability handling, derivative continuation and removable limits, physical matching, and performance acceptance remain unfinished. Dual shape inspection currently allocates through the upstream shape API; a fully allocation-free warmed scalar path is still an optimization requirement.

### Prepared graph reference milestone

- `ninjars prepare --reference-dir DIR` (opt-in `reference-codegen` feature) exports exact preparation metadata and per-sample original Ninja callbacks, runner inputs and conversion metadata. The callback includes graph weights and loop-independent propagators exactly once.
- Eleven CLI tests pass, including six reference-export regressions. Precision is preserved until the explicit reference binary64 boundary. Unbound parameters, nonfinite or overflowing/underflowing values, and unsupported complex external shifts are rejected before writing the bundle; complex squared masses remain supported.
- Four integrated prepared-graph comparisons pass: three changing bubble samples and one timelike triangle. Original Ninja's independent tensor interface plus analytic mu-dependent rational terms provide the controls. Reproduction script: `validation/reference/test_prepared_runner.py`; evidence: `validation/evidence/prepared-graphs.json`.
- CLI default-feature checking, strict feature Clippy, formatting and whitespace checks pass. Timings in this evidence are protocol smoke measurements, not performance acceptance. Actual Rust reduction remains in development.

### Laurent kernels and triple-cut reduction milestone

- Added actual batched triple-cut coordination in the core library. `TriangleReducer` compacts geometry-active cut/lane requests, batches both numerator orientations, divides by every uncut denominator, projects original-index residues and restores cut/lane order. This is a reduction stage, not a complete integrated amplitude engine.
- Four triangle-stage tests pass: explicit mu-squared/mu-fourth coefficients with lane permutations and failures, four- and six-line numerator cancellation controls, higher/mixed Dual parameter and kinematic seeds, and independent rank-four residue-polynomial reconstruction. API failures preserve caller output buffers.
- Four geometry/sampling tests pass for triangle, bubble and tadpole cuts, including spatial/timelike/null and complex kinematics, masses, singular sampling parameters, failed lanes, DoubleFloat/256-bit values and higher/mixed Dual components. Complex Gram square roots and primal chart selection retain derivative components without clipping mass-shell variations.
- Laurent division is generated by Symbolica series at inverse t, with signed-polynomial coefficient extraction. Four tests check all cut types through six lines and rank n+1, scalar/batch equivalence, empty/low-rank layouts, 192-bit accuracy and symbolic mixed derivative controls. Two projection tests independently recover every ordinary/higher-rank residue coefficient and check basis derivatives.
- Two denominator tests reconstruct uncut propagators directly on all cut parameterizations, including changing complex and higher/mixed Dual inputs. Six subtraction tests include 14 comparisons to unchanged original Ninja coefficient-level correction routines, shifted cut origins, ordinary/higher ranks, and derivative controls.
- Six master-contraction tests check ordinary/higher-rank indexed formulas, rational separation, independent original Ninja tensor-moment values, mu-dependent triangle controls, precision and higher/mixed derivatives. Rational decomposition is representation-dependent; integrated comparisons must match the residue basis.
- Independent audits found an uncut-only malformed Dual shape could abort good rows. The adapter now validates and compacts all row inputs before denominator arithmetic. Direct public division also rejects repeated/overlapping partitions. Focused regressions cover both cases.
- Shared exact complex-null Weyl helpers are pinned at FeynKit `66f59dd98c67a0b905145cad5ace1585df41fd72`, submitted as focused stacked [PR #143](https://github.com/alphal00p/gammaloop/pull/143), and attached to this chat. Three upstream symbolic tests and focused Clippy pass; PR #142 remains unchanged.
- An independent denominator oracle exposed static and dynamic Numerica Dual `pow(0)` returning the input. Minimal fix `b6edbce2d9b805b481b85a8699bb061e93cd83fb` is pushed to Symbolica PR #82. The new regression failed before the fix, and all nine Numerica Dual tests pass afterward, including higher/mixed complex components, zero primals, working precision and shape retention.
- Algebra API checks are documented in `docs/algebra-audit.md`. The current implementation uses exact Symbolica instruction graphs; generated native Rust specialization is still required for the performance phase. Current coordinator request/geometry allocations, scaled small-denominator conditioning and rescue are explicitly unfinished.
- Remaining required work: bubble/tadpole coordination and integrated CLI parity; amplitude accumulation/reset, S-matrix/cut controls and reconstruction diagnostics; boxes/pentagons/full rank coverage; derivative continuation/removable limits; physical FeynKit–MadLoop matching; all performance acceptance gates. The goal remains active.
- Final milestone validation: all **66 workspace tests** pass with all features and ignored reference tests enabled, serial execution and `RUST_MIN_STACK=134217728`. Strict all-target/all-feature Clippy passes (existing upstream warnings remain). Dependency patches reproduce both pinned commit trees exactly in isolated Git indexes. The standalone triangle-stage example returns `c0=c7=1`.

### Integrated tadpole, bubble and triangle milestone

The Laurent-stage snapshot was published as `6357f3e`. This next milestone
connects the existing kernels into integrated amplitudes and a standalone
`reduce` command; the full NinjaRS goal remains active.

- Core `BubbleReducer` subtracts all containing triangle residues in both
  orientations and retains complete original 10/20 coefficient arrays. Six
  tests cover two through four propagators, multiple parents and shifted
  origins, independent cut reconstruction, permutations/tails, higher/mixed
  Dual inputs, malformed uncut values and DoubleFloat/Float precision.
- Core `TadpoleReducer` subtracts triangle and corrected-bubble parents. Its
  result explicitly marks only c0 (ordinary), or c0/c14/c15 (rank n+1), as
  computed. Six tests cover analytic tensor coefficients, denominator
  cancellation, independently supplied nonzero parents, shifted origins,
  parent failure/provenance, higher/mixed derivatives and 256-bit accuracy.
- `MasterProvider::evaluate_batch` reports per-row `MasterStatus` without
  degrading to scalar execution. Both Symbolica providers compact valid rows
  before one family evaluation. Seven tests cover mixed domains and lazy
  branches, output overflow, all Dual components and shapes, empty/single/tail
  batches, strict API transactionality and precision propagation.
- `ResidueIntegrator` batches master families across active cuts and phase-space
  lanes. Generated Symbolica kernels construct invariants, weights, contraction
  and accumulated sums. All-component zero weights skip unnecessary masters.
  Four tests check original Ninja tensor-moment values, explicit rational
  integrals, compact batches, failed-row preservation and analytic higher/mixed
  derivative products.
- Core `Amplitude` now integrates one, two and three propagators through rank
  n+1. Successful lanes accumulate; failed lanes retain all previous numerical
  fields. Reset, fixed batch identity, separate rational/cut-constructible
  components and Laurent accessors are available. Four tests cover shifted
  tadpoles, higher-rank bubbles, triangle mu-squared/mu-fourth controls,
  scaleless tadpoles/null bubbles and failure/reset semantics. An independent
  five-test audit adds late-error transactionality, repeated-propagator rejection,
  derivative-only weights, higher/mixed Dual history and higher precision.
- Independent finite B0 controls use the closed form
  `2-sqrt(5)*log((sqrt(5)+1)/(sqrt(5)-1))`, evaluated with Symbolica at 384 bits.
  Errors are below 1e-14 for f64, 1e-29 for DoubleFloat, 1e-52 for Float192 and
  1e-70 for Float256; each precision increase improves the error by at least
  twelve decimal digits. Independent A0 controls reach below 1e-45 at 160 bits
  and 1e-90 at 320 bits against a 512-bit reference.
- The CLI `reduce` reuses exact DOT/model preparation and one parameterized
  polynomial adapter across phase-space samples. It supports f64, DoubleFloat
  and arbitrary precision, batch sizes/tails, diagnostics and JSON output with
  decimal complex components. All sixteen CLI tests pass, including the original
  Ninja bubble/triangle reference rows in all three carriers, exact corrections
  below f64 resolution, mixed failed lanes, empty inputs and rank rejection.
- Full CLI testing exposed an incorrect NinjaRS assumption that every component
  must retain the nominal working precision. Symbolica Float intentionally
  tracks significant bits under cancellation, including tiny imaginary
  residuals. Integration and masters now preserve those values and bit counts;
  constants still originate at the configured working precision. No values are
  padded, clipped or converted through f64. The CLI records component precision
  separately. Upstream documented behavior and its zero-does-not-restore-precision
  regression establish that no Symbolica patch is appropriate for this issue.
- All **103 workspace tests** pass with every feature and ignored reference
  test enabled; none are skipped. Strict all-target/all-feature Clippy passes.
  The complete bubble example runs successfully. Reproducible reference
  comparisons and analytic accuracy gates are recorded in
  `validation/evidence/integrated-stage.json`.
- Remaining work includes boxes/pentagons and full six-line integration;
  S-matrix/cut stopping and reconstruction controls; triangle/box Dual masters,
  continuation and removable limits; tensor adapters and physical FeynKit/MadLoop
  matching; `compare`/`bench` and CLI derivative controls; numerical rescue,
  allocation removal/native Rust specialization and all performance gates.
  `docs/box-pentagon-design.md` records next-stage research, including an
  explicitly unconfirmed original higher-rank box reconstruction hypothesis.

### Four/five-cut and six-line integration milestone (2026-10-16)

- Added shared-momentum `HighCutProvider`, `BoxCut` and `PentagonCut`, with
  scalar and static/dynamic Dual Symbolica adapters. Four box coordinate charts
  and the five-cut solution use Symbolica's exact matrix solver; branch/chart
  selection remains lane-local. Four test groups check every chart, on-shell
  identities, complex roots/masses, singularities, permutations/tails, higher
  and mixed derivatives, and DoubleFloat/Float256 residuals below 1e-25/1e-65.
- Added `PentagonReducer` and generated division kernels. Its residue is
  `coefficient*mu_squared`, with no direct E0 master. Three test groups cover
  five/six propagators, dense cut/lane identity, mixed derivatives, failed
  lanes, zero mu, zero uncut denominators and API transactionality.
- Added `BoxReducer`, full/integrated coefficient masks and both mu-expansion
  and finite-mu sampling. Pentagon parents are constructed/subtracted only
  when the selected mode needs them. Seven test groups cover all four-line
  ranks through n+1, five/six-line nonzero parent subtraction, independent
  reconstruction, higher/mixed Dual, malformed uncut values/provenance,
  complex masses and coefficients below binary64 resolution.
- Extended `MasterContraction`, `ResidueIntegrator` and `Amplitude` through
  six propagators. Box integration uses `c0*D0-c4/6`, invariants ordered
  `[s21,s32,s43,s14,s31,s42]`, and no artificial basis. Existing low-cut clients
  retain `evaluate_batch`; higher topologies use `evaluate_batch_with_high_cuts`
  with explicit sampling scales. Integration excludes every failed lane before
  master requests. Three added boundary groups verify argument order/scales,
  failure/history/reset semantics, late API errors, and rational-only Dual boxes.
- Independent Symbolica-constructed box/pentagon numerator ansätze, multiplied
  by actual complementary propagators, produce zero lower-cut coefficients in
  ten source/rank cases and two lanes. These controls include c5*mu4*u, shifted
  origins and permuted source partitions and use no box/pentagon reducer to
  construct the expected result. No extra higher-parent subtraction is needed.
- Six independent D0 controls validate all Laurent coefficients and mixed
  lazy branches. DoubleFloat versus Float256 agrees below 1e-25; Float160
  versus Float256 agrees below 1e-42, improving on f64. Decimal test inputs
  were replaced by exact fractions to respect Symbolica's literal precision.
  Dual C0/D0 domains have deliberately not been extended by these tests.
- The CLI now accepts four-to-six-line reduction, both box extraction modes,
  full box diagnostics and an exact mu sampling scale. The new box DOT fixture
  has three phase-space samples. All 36 original-Ninja comparisons across
  f64/DoubleFloat/Float256, both modes and batch sizes 1/2 pass, with maximum
  scaled difference 1.34e-14. Independent tensor plus analytic mu4 controls
  verify the prepared reference polynomial. Evidence: `box-cli.json`.
- Eighteen independent original-Ninja controls cover n=4,5,6, ranks 0,2,n+1,
  and real/complex masses. The 36 Rust comparisons across both extraction
  modes pass the fixed 1e-9 criterion, with maximum scaled difference
  2.58e-11. `high-reduction.json` records source controls and two nonfinite
  reference configurations separately; `high-stage.json` records Rust results.
- Numerical limitation: mu-sampling scale 3 at the rank-seven six-line real
  case produced a scaled error about 2.6e-9 through cancellation. A scale
  comparable to the largest invariant passes without changing the tolerance.
  The observation is retained in `high-stage.json`; callers currently supply
  sampling scales. Automatic scale choice, reconstruction certification and
  precision rescue remain required. Finite success is not an accuracy certificate.
- Confirmed two original Ninja defects using unchanged production libraries:
  higher-rank full-box finite-mu extraction misses a scale factor in c3 (the
  integrated c0/c4 are unaffected), and the D0 cache omits s14. The streaming
  validation runner now clears caches between distinct requests. Also recorded
  OneLOop 3.4's zero-return restriction at a complex-mass spacelike box.
  `docs/box-reference-audit.md` gives reproductions and comparison policy.
- No new upstream patches were needed; pinned FeynKit/Symbolica PR revisions
  remain unchanged. All **128 workspace tests** pass with all features and
  ignored reference tests enabled, with no failures or skips. Strict all-target
  workspace Clippy, formatting and diff checks pass. All five streaming
  reference protocol tests pass, including an exactly dyadic changing-s14
  regression for the new between-request cache clearing. Evidence is retained
  in `high-stage.json`.
- The goal remains active. Remaining work includes S-matrix/cut controls,
  full reconstruction and stability/rescue; automatic sampling scales;
  C0/D0 Dual continuation and removable neighborhoods; tensor and physical
  FeynKit/MadLoop preparation/validation through pentagons; CLI compare/bench
  and derivative controls; generated native Rust execution, workspace
  allocation removal and every performance acceptance gate.

### Sampling scales, API controls and comparison CLI milestone (2026-10-16)

- Added scalar and static/dynamic Dual automatic sampling-scale providers.
  Batched invariant/magnitude arithmetic uses shared FeynKit momenta and
  Symbolica evaluators; selection preserves working precision. Only all-zero
  kinematics use a unit fallback. Auxiliary scales are constant in the original
  Dual shape. Four library and three CLI test groups pass, including mixed
  invalid lanes, tails, exact overrides and DoubleFloat/Float mixed derivatives.
  The rank-seven six-line control improves from scaled error 2.59e-9 with scale
  3 to at most 2.66e-11 with automatic scale 247 across all four box modes.
- Added `AmplitudeInput` with an optional lane-major S-matrix of `(Vi-Vj)^2`,
  used for both master arguments and rational weights. Numerical matrix
  validation includes every Dual component. Added Ninja-compatible cut stopping
  without changing existing accumulated results. Six library regression groups
  pass, checking exact overrides, mixed derivatives, API transactionality,
  omitted callbacks, accumulation, noncontiguous cut partitions and failed
  pentagon lanes when stopping at five cuts. Five CLI control groups pass.
- Added an optional Rust `compare` command, under `reference-codegen`, which
  prepares common inputs, exports Symbolica-generated reference callbacks,
  builds original Ninja runners and records all Laurent comparisons against
  the fixed 1e-9 gate. Reference runners accept the same S-matrix, cut-stop and
  mu-extraction controls. Generated inputs, source/library hashes and logs are
  retained. Reference processes are fresh per sample; timing output is diagnostic
  and not performance acceptance evidence. All eight complete comparisons in
  four configurations pass, including supplied invariants and cut stopping in
  DoubleFloat. Large finite-magnitude error controls and artifact/build-directory
  concurrency tests prevent false successes or mixed generated inputs.
- Published and attached [oneloopmaster PR #4](https://github.com/alphal00p/oneloopmaster/pull/4),
  commit `606974150011332cfea677f70940f9228e24c9e3`. Minimal analytic-argument
  fixes preserve imaginary and higher/mixed Dual seeds in real-axis log/sqrt
  helpers. Five new regression instances fail before the fix and pass after;
  all 52 upstream unit tests pass, with one existing diagnostic ignored.
  Formatting/diff checks pass; upstream Clippy completes with existing unrelated
  warnings. The main dependency checkout is fast-forwarded to that commit,
  with a consolidated reproducible patch in the dependency manifest. The
  pre-existing formatting-only `epsilon.rs` change remains untouched.
- C0/D0 Dual domains remain gated: scalar-specialized removable limits in
  log ratios, divided differences and dilogarithm helpers still need complete
  generated neighborhoods. The continuation patch does not establish those
  derivative domains. FeynKit H→gg preparation and a separate original MadLoop
  graph harness are in progress; neither is physical acceptance evidence yet.
- Published and attached [FeynKit/gammaloop PR #144](https://github.com/alphal00p/gammaloop/pull/144),
  stacked on PR #143, at `474478a7a80bc9f9f21bead0e1cde6acbcffbe93`.
  Explicit one-thread graph generation now keeps coloring, graph construction
  and grouping on the caller. Parallel generation remains available. All 104
  upstream generator tests pass, including caller-thread callbacks, determinism, cancellation,
  filtering and two/four-worker execution; strict upstream Clippy passes.
  The tested integration checkout and consolidated dependency patch are updated.
  Two Rust physical-preparation regression groups pass: both top-loop
  orientations, resolved graph sign, three MT² propagators, four-dimensional
  SU(3)-projected component polynomial, chirality zero and invalid-model/dimension
  handling. The graph weight uses FeynKit's existing resolver; explicit physical
  color normalization uses Idenso. `prepare_hgg` exports graph/model sources and
  prepared symbolic numerators. Both MadLoop top orientations are independently
  exported with multiplier one and a fixed backend. Raw numerical numerator and
  integrated physical comparisons remain pending; `docs/physical-preparation.md`
  records the exact convention and remaining work.
- The user supplied a Symbolica runtime license for development and validation.
  It is sourced from a local file outside the repository and is not embedded in
  code, dependency patches, logs or published artifacts.
- Published and attached [oneloopmaster PR #5](https://github.com/alphal00p/oneloopmaster/pull/5),
  stacked on PR #4 at `fe2a8d4ac363998ea557d596be57aa10b5bbd013`. A shape-derived
  Dual evaluator generates the removable sheet-log-ratio neighborhood using
  Symbolica series before specialization/vectorization. Degree-eight,
  higher/mixed, near-unit high-precision and wound-sheet-pole controls pass;
  the upstream suite reports 56 passed plus one existing ignored diagnostic,
  and eight function-definition integration tests pass. This is prepared
  upstream work for the next milestone; NinjaRS remains pinned to PR #4 here.
  Other removable dilogarithm/divided-difference neighborhoods and C0/D0 domain
  validation are still required.
- All **155 workspace tests pass**, with all features and ignored reference
  controls enabled, no failures and no skips. Strict all-target workspace
  Clippy, formatting and diff checks pass. All five streaming reference
  protocol tests pass after rebuilding the runner. The physical `prepare_hgg`
  example runs successfully, and both updated dependency bundles pass bootstrap
  verification. `validation/evidence/controls-stage.json` records dependency
  revisions, comparison coefficients/errors and the sampling-scale diagnostic;
  `validation/record_controls.py` extracts it from the complete workspace log.
- The goal remains active. Next work includes physical CLI bindings and raw
  FeynKit/MadLoop numerator/amplitude matching, remaining C0/D0 Dual neighborhoods,
  reconstruction/stability/rescue, reusable timed CLI sessions and matching
  reference cache policies, native Rust arithmetic generation and allocation
  removal. Physical validation through pentagons and every performance gate
  remain unfulfilled.

### Physical numerator, benchmark and upstream evaluator checkpoint (2026-10-16)

- Added reusable CLI numerical sessions and `bench`, sharing the numerical
  execution path with `reduce`. The default workloads cover widths
  1/8/32/128/1024 and incomplete tails. Setup, exact binding, packing and warmup
  are separate from primary timings; callback/master attribution runs in a
  separate diagnostic pass. Reports retain individual rounds, numerical lane
  failures and disabled-rescue status. Four benchmark regression groups pass,
  including f64/DoubleFloat/Float parity with reduction. This is Rust-only
  instrumentation; matching original-Ninja cache policy, shared-master timing,
  Dual benchmarking and every performance acceptance gate remain outstanding.
- Added exact CLI polarization bindings and embedded-SM physical fixtures for
  both top-quark H→gg orientations, with ++, +− and double-longitudinal states.
  The MadLoop harness calls raw graph numerators before squaring or Born
  interference and explicitly excludes R2/UV. It records fixed CutTools modes
  1 and 4 with no adaptive switching. Quadruple execution promotes the same
  binary64 model/wavefunction inputs; it is not a high-precision physical-input
  oracle. Fifteen independent complex points determine each rank-two polynomial.
  All 90 independently prepared FeynKit/MadLoop numerator checks pass, maximum
  scaled error 8.66e-16. Source hashes, routing, color/sign projection and the
  1/(16π²) convention are retained in `physical-hgg-madloop.json` and the physical
  preparation documentation.
- Physical reduction exposed an axis-aligned bubble reference whose uncut
  denominator had an identically zero leading Laurent coefficient. Generic
  exact rational null references now come from FeynKit Weyl operations and
  Symbolica evaluation. The regression reduces an explicit complementary
  denominator to B0(0;m²,m²), checks f64/DoubleFloat/Float accuracy, higher/mixed
  Dual translation invariance and failed-lane isolation. Two new and fourteen
  existing geometry/bubble/amplitude test groups pass. This fixes the observed
  lightlike channel; complement-aware chart retry and precision rescue remain
  outstanding.
- `physical-hgg-cli.json` retains the first complete 36-row physical comparison:
  32 pass the unchanged 1e-9 Laurent gate. Both physical helicities pass in all
  three precisions. Four f64 longitudinal diagnostic rows fail at a zero single
  pole by 1.066920211e-9; DoubleFloat and Float256 pass those rows. The artifact
  explicitly reports failure. Cancellation/conditioning improvement remains
  required; no exceptional-point exemption or tolerance change was applied.
- Published and attached [Symbolica PR #83](https://github.com/symbolica-dev/symbolica/pull/83).
  Nested vectorized conditionals produce several joins; two optimizer passes
  incorrectly ended a branch for every join. They now restore its parent at the
  matching end label. A small shared polynomial reproducer exposed history-
  dependent primals, independent of master formulas. The pinned integration is
  `52ef4b8e3a10cf3a68385230d55ebe8c3d153948`, combining PR #82 with this fix and
  its shared-Dual regression. All 62 evaluator tests pass, including batching
  and higher/mixed components. The C0 zero-seed/primal discrepancy is fixed at
  f64/Float256; a first complex mass derivative matches independent differences.
  `docs/c0-dual-vectorization-audit.md` records the revision matrix and limits.
  C0/D0 Dual master domains remain gated. PR #5's permanent follow-up regression
  is published at `33a2e89b76252fc22556c0da50d1689cfb2101ec`, still pending NinjaRS
  integration and the remaining removable-neighborhood work.
- Published and attached [oneloopmaster PR #6](https://github.com/alphal00p/oneloopmaster/pull/6),
  pinned at `de0114add9937a2225a3e5d89af1d497ccd31d07`. It extracts the existing
  Rust emitter, supports arbitrary entry output counts and transactional exact
  constant tables, and reuses its numerical runtime. All six master source
  files are byte-identical before/after extraction; 64 library, 11 primitive,
  four native-contract and one compact-sector tests pass, with one pre-existing
  diagnostic ignored. Changed-file formatting and diff checks pass; upstream
  Clippy completes with existing unrelated warnings. Both refreshed dependency
  bundles pass bootstrap verification. NinjaRS kernel export/probes and a final
  combined workspace validation are in progress; no native speedup is claimed.
- NinjaRS now exposes a thin `RustKernelExport` adapter for the existing exact
  Laurent-division, triangle-to-bubble-subtraction and denominator-expansion
  evaluators, including the shared Symbolica Dual transformation. The compiled
  control passes 36 cases, checking 1,044 scalar outputs and 5,220 flattened
  higher/mixed components. Maximum scalar/Dual errors are 8.13e-16/1.83e-15 in
  f64, 6.96e-32/3.10e-31 in DoubleFloat and 9.86e-77/2.56e-76 in Float256.
  `validation/evidence/rust-kernel-codegen.json` retains exact constructor
  parameters and generated-source hashes. This is scalar execution used as a
  control; native instruction-major batching, production dispatch, allocation
  targets and performance acceptance are not established.
- Fixed the reference adapter's initialization of registered constants:
  exporting an unmapped exact evaluator emitted π's zero placeholder.
  Symbolica's existing `map_coeff_with_prec` resolves coefficients and registered
  constants at 200 bits before C++ literal export. Both compiled callback tests
  pass, including the physical electroweak factor and π^(3/2). H→gg physical
  helicities now agree with original Ninja. Its binary64 longitudinal pole also
  misses the fixed gate; this is a cancellation diagnostic, not an established
  original-Ninja cut-geometry bug.
- Per-cut diagnostics locate longitudinal cancellation before final accumulation:
  tadpole poles are about +1.0789e7, -7.1927e6 and -3.5964e6, and bubble tensor
  contraction subtracts terms near 9.3878e5. Rescaling the exact off-axis null
  references from fractional to integer components worsened the matrix maximum
  error from 1.06692e-9 to 1.17793e-9 and was reverted. No tolerance changed.
  The restored 32/36 baseline was reproduced with binary/source/fixture hashes
  and all run commands. `docs/precision-rescue-design.md` records a full-lane
  retry design using existing exact inputs and shared numerical types; automatic
  conditioning triggers and rescue remain unimplemented.
- Published and attached [oneloopmaster PR #7](https://github.com/alphal00p/oneloopmaster/pull/7),
  head `aa6491450b22b4341abfffc6ce0d576c232811b8`, stacked on PR #5. Symbolica-
  generated centered quotient expressions preserve coincident exact primals and
  independently seeded higher/mixed derivatives. All 59 library tests pass with
  one existing ignored diagnostic; strict package Clippy and formatting pass.
  This remains isolated pending broader master-domain validation.
- Published and attached [Symbolica PR #84](https://github.com/symbolica-dev/symbolica/pull/84),
  standalone head `eab147eaa5ceb69169bacb35281c3680d474bd84`. The public series API
  previously returned indeterminate for Li2(1-x) about x=1. A focused native
  positive-integer polylog series handler reuses Symbolica Series arithmetic;
  shifted/composed, Puiseux and removable-quotient regressions pass, as do all
  64 transcendental tests. Formatting passes and Clippy reports baseline
  warnings only. NinjaRS pins and C0/D0 Dual gates remain unchanged while the
  dependent divided-dilog neighborhood is investigated.
- Final combined validation: **165 workspace tests passed, zero failed and zero
  ignored**, with all features and reference controls enabled. Strict all-target
  Clippy, workspace formatting and diff checks pass. Refreshed Symbolica and
  oneloopmaster dependency bundles pass bootstrap verification.
  `validation/evidence/physical-stage.json` records the tested source hashes,
  dependency heads, toolchain and related evidence. Passing regressions do not
  override the explicitly failed four longitudinal f64 comparison rows.
- Independent allocation/performance audits identify avoidable deep cloning of
  staged residue vectors and temporary orchestration buffers. The Symbolica
  batch evaluator already has a warmed size-one scalar path and retained batch
  workspace. A fair original-Ninja baseline still needs an explicitly uncached
  master build, a common parameterized numerator with depth-specific callbacks,
  and a persistent numerical loop. Original and NinjaRS finite-mu sampling-scale
  policies differ and must remain reported. The goal is active; no performance,
  full reconstruction, broad master-derivative or higher-multiplicity physical
  acceptance is claimed by this milestone.

### Allocation, shared-code and matched timing work (2026-10-16, in progress)

- Removed deep copies when committing staged cut residues, retained accepted-row
  scratch in high-cut geometry, and borrowed the integration master list. Existing
  output transactionality is preserved. The warmed f64 allocator probe covers
  n=2/4/5 at rank n+1, widths 1/8, full box coefficients and finite-mu sampling.
  Allocator calls decrease from 50/120 to 43/78 for bubbles, 129/423 to 103/257
  for boxes, and 189/854 to 142/527 for pentagons. Five repetitions give identical
  counts; this is allocation evidence, not a runtime speedup measurement.
  `validation/evidence/allocation-workspace.json` retains commands, binary/source
  hashes and reallocations. Forty-six existing focused tests pass.
- Added `prepare --reference-bench-dir` to export one common exact numerator
  template, its ordered parameter rows, and the production binary64 momentum,
  mass, scale and invariant bindings. Original-Ninja callbacks now compute only
  their requested Laurent depth and rebind parameters without allocation. Two
  compiled callback tests and thirteen CLI tests pass, including a first sample
  whose numerator specializes to lower rank and a non-real shift too small to
  survive conversion to binary64. The common header keeps the template's rank.
- Added a persistent original-Ninja timing runner and an explicitly uncached
  OneLOop build. Protocol controls cover parameter rebinding, ordering, malformed
  inputs, supplied matrices, nonfinite results, pre/post fingerprints, and full
  Laurent agreement in both mu extraction modes. Box and parameterized-bubble
  protocol suites each pass eight tests with one triangle-specific skip;
  triangle passes all nine. These final runs use a fresh uncached build which
  hashes copied sources and generated headers at compilation and refuses header
  mutations or incomplete older build records. The three complete reports are
  retained as `validation/evidence/reference-benchmark-*.json`.
  The optimized comparison harness fixes affinity and
  thread counts, records source/compiler/cache policy and individual rounds,
  and checks every source-pool Laurent coefficient before timing.
- A partial-cut diagnostic differs between implementations. The rotation control
  independently shows that this bubble's retained contribution changes under a
  spatial rotation while the full amplitude agrees. Nonzero omitted tadpoles
  violate original Ninja's documented `setCutStop` assumption. The diagnostic is
  preserved in `validation/evidence/partial-cut-basis.json`; performance controls
  require full reductions. This does not excuse other partial-cut discrepancies.
- The first completed optimized parameterized-bubble run passes all numerical
  comparisons but fails the performance target: observed Rust/original median
  latency ratios are 10.13/15.26 at width one (repeated/changing), and 6.36–11.98
  at widths 8/32/128/1024. The reference is uncached and uses the same prepared
  template and numerical rows. Reverse-order and box runs were interrupted
  before timing by background compiler activity and remain pending. No confidence
  interval or performance acceptance is claimed; retain raw completed runs and
  rerun in a quiet window before drawing stronger conclusions.
  `performance-preliminary.json` preserves that initial run and its incomplete
  build provenance: the draft Rust record omitted Cargo's home configuration,
  and the draft reference record did not hash copied/generated headers.
  The new reproducible build recorder checks production/dependency state before
  and after compilation; the comparison verifies these inputs, actual Cargo
  configuration and executable hashes. Nine fast provenance/failure-path tests
  and an end-to-end diagnostic smoke pass. Fresh controlled runs use the stricter
  records and a newly built uncached reference.
- Published and attached [Symbolica PR #86](https://github.com/symbolica-dev/symbolica/pull/86),
  standalone head `b764ae3749d5e82ba3e6020d5b5d88a40c1d23c0`. A defaulted
  object-safe borrowed shape accessor avoids allocating during repeated static
  and dynamic Dual checks. Four new and five existing tests pass, including a
  zero-allocation repeated access control and malformed/reordered/higher/mixed
  structures. Full validation must check all rows and the end sentinel rather
  than trusting the mutable value-vector length. Main pins are unchanged.
- Published and attached [oneloopmaster PR #8](https://github.com/alphal00p/oneloopmaster/pull/8),
  head `05c87c3c4dabf6a26b8f9396468406778af457e2`, stacked on PR #7 and tested
  against an isolated Symbolica integration including PR #84. Symbolica-generated
  coincident divided-dilog neighborhoods preserve independently seeded arguments
  and higher/mixed components, including x=y=1. Sixty-three unit and eight
  function-definition tests pass, with one existing diagnostic ignored; formatting
  and Clippy pass with baseline allowances. Nonanalytic sheet cases remain
  nonfinite. Debug C0 graph construction reached about 9 GB RSS, so graph sharing
  and construction cost need work before broad integration. Plain-expression
  helpers and full C0/D0 derivative-domain validation remain incomplete. Production
  dependency pins and Dual master-domain gates remain unchanged.
- The shared native Rust emitter still executes a row at a time. An independent
  audit identifies an instruction-major extension in oneloopmaster using the
  existing exported slots, temporary counts, primitives and exact constant table.
  Initial implementation must reject unsupported control flow recursively, retain
  input precision, support empty/tail/constant-only graphs, and validate actual
  instruction-major execution. Lazy branches and active shared-helper batches
  require separate controls. No native batching speedup is claimed yet.
- Combined validation reports **167 workspace tests passed, zero failed and zero
  ignored**, with all features and ignored reference controls enabled. Strict
  all-target workspace Clippy, formatting and diff checks pass for the same Rust
  sources. The reproducibly recorded optimized binary has SHA-256
  `b89e5df7853e6162a29eb0fd0e92781ac7a055c69c4a0c7d9efd2a3fd4acde3c`.
  Physical accuracy limitations and the active goal remain unchanged.

### Matched timing checkpoint and upstream policy (2026-10-16)

- Two fresh parameterized-bubble runs, with reversed implementation order, pass
  all Laurent comparisons (maximum scaled error about 1.07e-14). Both fail the
  performance gate: width-one Rust/original median ratios are 10.36–10.45 for
  repeated points and 15.77–15.83 for changing points. Larger batches still trail
  the reference. `validation/evidence/performance-checkpoint.json` indexes both
  complete reports, source snapshots and the reproducible optimized build record.
  Compiler snapshots are observations, not proof of an exclusive host.
- The box run passes all pre/post numerical controls (maximum scaled error
  1.50e-14), but unrelated compilation was observed during timing. Its raw rounds
  and explicit contention record are retained as diagnostic evidence only; no
  reverse-order repeat or performance acceptance is claimed. Native batching,
  master costs and remaining orchestration costs require further work.
- The user explicitly changed the oneloopmaster contribution policy: consolidate
  and merge every existing PR, then push tested changes directly to its `main`.
  No new oneloopmaster PRs will be opened. Symbolica and FeynKit/gammaloop continue
  to use focused upstream PRs. The consolidation must include PRs #7 and #8 even
  though their intermediate base branches were merged before these later changes.

### Consolidated oneloopmaster main (2026-10-16)

- Pushed the complete oneloopmaster PR ancestry to upstream `main` at
  `317ce6b98bf310e88435fc8aac330bb5f70fd1f7`. GitHub marked PRs #3, #4 and #6
  merged at 18:41:37 UTC and reports zero open PRs. The merged history also
  retains the later PR #7/#8 changes, which were absent from PR #4's old head.
  Subsequent changes are pushed directly to oneloopmaster `main`; Symbolica and
  FeynKit/gammaloop continue through focused PRs.
- The consolidated native emitter adds branch-free instruction-major execution
  with shared helper bodies and reusable storage. Its 27 focused checks pass,
  including f64/DoubleFloat/Float, flattened higher/mixed Dual components,
  instruction-order tracing and zero warmed f64 allocations. Lazy active-lane
  control flow and production reduction/master dispatch remain outstanding.
- Standalone Rust and both Python build roots now select the identical Symbolica,
  Numerica and Graphica integration `8e23925ed7b5cbb91607c8633dda114cb43be8ce`,
  combining the already submitted Symbolica PRs #82/#83/#84. The integration is
  available on the contributor fork, and both locked Python compilation checks
  pass. NinjaRS uses the same Symbolica source via its updated verified bundle.
  Its oneloopmaster dependency now pins upstream main without a local patch.
- The full initial upstream run completed 35 targets: 169 passed, six failed,
  six ignored. Every failure came from the independently reproduced D0 JIT
  argument-capacity limit (70 required versus 64 configured). A focused repair
  through the existing backend configuration API is being validated separately;
  the initial run is retained rather than reported as passing.
- Added independent FeynKit preparation and exact input fixtures for six oriented
  top-quark four-photon boxes. Two graph controls pass, including topology,
  four-dimensional rank, exact masses and the closed-color/Dirac trace. MadLoop's
  six corresponding standalone exports pass their internal raw-driver checks.
  Verified IntegralFamily mappings now establish all six loop orientations and
  denominator permutations. Independent FeynKit/MadLoop comparison passes 1,680
  checks over three helicities and generic complex polarizations, with maximum
  scaled error 1.169774266e-14. The independent exact trace/color control fixes
  the raw factor to -3; the audited CT normalization then gives -48*pi^2 for the
  future integrated comparison. The separate internal MadLoop raw/wrapper checks
  total 3,360 and are not counted as independent FeynKit checks. Reproducers and
  source/library hashes are archived under `validation/madloop/four_photon` and
  `validation/evidence/four-photon-*.json`. Integrated comparisons remain pending.
- Pushed the JIT argument-capacity repair and completed validation records
  directly to oneloopmaster main at `69761cdf8ffc0c02dbc2e36ce41ee6fa365268ae`.
  The repair sizes nested complex calls using existing exported instructions and
  backend configuration, checks overflow/capacity, and recomputes requirements
  when loading serialized evaluators. All six previously failing targets pass
  focused reruns. Controls include all 293 scalar fixtures, cache reloads,
  multiple batch layouts, fresh-process entry routes and registered callbacks.
  Both Python locked checks pass with SymJIT 2.27.3. This is combined regression
  evidence, not a claim that the complete upstream suite was rerun afterward.
  GitHub still reports zero open oneloopmaster PRs.
- NinjaRS pins the final upstream commit and removes the obsolete oneloopmaster
  patch bundle. All **169 workspace tests pass, zero failed and zero ignored**,
  with all features and ignored reference controls enabled. Strict all-target
  workspace Clippy, formatting and dependency bootstrap checks pass. Runtime
  tests compiled oneloopmaster's identical Rust sources at `1bdcb28`; the final
  commit adds only validation documentation and Python lock updates. Existing
  H→gg longitudinal f64 failures, derivative-domain gates, performance gaps and
  remaining reconstruction/physical validation work are unchanged. The overall
  goal remains active.

### Native batch integration and remaining setup costs (2026-10-16)

- The reduction adapter now uses oneloopmaster's existing `RustEmitter::emit_batch`
  for ordinary and Symbolica-flattened Dual graphs. The API audit checked the
  public emitter, `rust_codegen/batch.rs`, `native/batch.rs`, and upstream batch
  regressions; the adapter introduces no algebra or alternate instruction
  representation. A compiled probe covers the existing Laurent division,
  triangle/bubble subtraction and denominator expansion plans. Its first
  monolithic compilation was stopped after 1,364 seconds and approximately
  5.6 GB resident memory without reaching numerical validation. This incomplete
  attempt is recorded in `validation/evidence/rust-kernel-batch-unoutlined-build.json`.
  Bounded extraction of generated batch functions is now tested and pushed to
  oneloopmaster main at `5091653086b32428c6e024ee89404b4b21214e88`, the new
  NinjaRS dependency pin. The isolated full probe passes 108 groups; combined
  upstream validation passes 14 emitter tests and the compiled numerical probe.
  The former giant subtraction/Dual batch body is split at existing instruction
  boundaries into 69 functions, preserving scalar source and arithmetic order.
  Its dev-profile build completed in 57.10 seconds with 2,362,496 KiB peak RSS;
  this demonstrates build feasibility, not a controlled performance ratio.
  Evidence is in `validation/evidence/native-batch-outlining.json`. Root probe
  verification on the final pins also passes all 108 groups, covering 118,581
  scalar outputs and 592,905 higher/mixed Dual components. This includes empty,
  single, large and tail batches, permutations, malformed/overflow dimensions,
  retained workspace reuse and zero warmed/fresh-single-row f64 allocations.
  Exact source, executable and dependency hashes are retained in
  `validation/evidence/rust-kernel-batch-codegen.json`. Production dispatch and
  performance acceptance remain outstanding.
- An isolated Symbolica audit found that importing every scalar in an already
  flattened Dual constant pool padded it with another full derivative group at
  each helper level. A defaulted vectorizer hook can reuse the existing exact
  literal interner without adding equality bounds to generic vectorizers.
  Registered constants are excluded from interning before literal import, so
  equal placeholder zeros retain distinct identities. The focused helper-depth,
  registered-constant and carrier-compatibility controls pass. On the same C0
  primal/zero-seed regression, diagnostic debug runs changed from 132.74 s /
  15,038,696 KiB peak RSS to 5.19 s / 406,324 KiB. These shared-host measurements
  are not a performance acceptance result. The fix is submitted in the
  existing Symbolica PR #82, now updated at `6af8de565eb3908eacbb93390355f2bdb98bfa9e`.
  The tested integration is `58b72d438e7bcfce158d132612db7d35fdefa4ba`, available
  on the public contributor fork. Its 17 Dual and five batch tests pass; the
  PR-only tree passes 16 Dual and five batch tests. All 65 active oneloopmaster
  unit tests pass against the integration, with one existing ignored control.
  The exact C0 stored scalar count drops from 46,106,630 to 110. Reproduction
  commands, measurements and logs are archived in
  `validation/evidence/symbolica-dual-constant-pool.{md,json}`. NinjaRS now pins
  this integration through its reproducible patch bundle. oneloopmaster also
  pins all three compatible packages at this revision in its Rust and Python
  build roots, pushed directly to main at
  `aedc2ec145d26701922f66794e96e89522301be3`. Combined validation passes 82 unit
  tests (one existing ignored diagnostic), the compiled native batch probe,
  and locked checks of both Python build roots. Evidence is archived in
  `validation/evidence/oneloop-dual-pool-integration.json`.
- Pushed tested lazy batch emission directly to oneloopmaster main at
  `df99fb48c2f87a6fec9199cccefc83587b381737`; GitHub still reports zero open PRs.
  It uses exported forward control flow, retained active/resume frames and
  parent-stride lane subsets through shared helpers. Higher/mixed Dual tests
  also exposed and fixed scalar-emitter liveness for consecutive branch joins.
  The focused validation passes 38 tests with one existing ignored control,
  including the existing production native integration suites. Primitive traces
  prove inactive singular arms are skipped and arithmetic runs across active
  lanes. Warmed f64 and fresh single-row calls allocate zero times. Scoped strict
  Clippy passes with the three recorded baseline lint allowances; formatting
  passes. Evidence is in upstream `patches/lazy-native-batch-validation.md` at
  that commit. Production native master dispatch still uses scalar row loops;
  the reusable emitter is ready for a separate integration. These changes are
  included in the combined main revision `5091653` pinned above.
- The integrated four-photon preflight exposed expensive expansion of external
  coefficients during exact callback preparation. The repair uses existing
  Symbolica `map_coeff`, `flatten(false)`, selective `expand_in`, and evaluator
  `add_aliases` to preserve those coefficients while extracting cut powers.
  The same aliases serve every lane and are resolved before Dualization.
  Independent full-expansion controls check 798 exact coefficients through
  rank five, including explicit mu powers and negative Laurent powers; a
  second regression checks higher/mixed Dual values on reordered and repeated
  lanes. No polynomial implementation or algebraic representation was added.
  `replace_all` requires positive exponent types and therefore
  cannot be used directly with the existing signed-exponent polynomial API.
  The complete six-graph physical matrix passes with an immutable compiled
  CLI and recorded source/dependency hashes at Symbolica `58b72d4` and
  oneloopmaster `aedc2ec`. It covers 108 physical rows and 36 generic-polarization
  diagnostics, three precisions, both mu-extraction modes and incomplete tails.
  All physical Laurent coefficients, finite/rational decomposition, mu-mode
  controls and summed graphs pass against fixed MadLoop mode 4 at `1e-9`.
  Independent Symbolica double-Laplacian controls establish each graph's pole
  and exact cancellation across all six graphs. Physical pole errors improve
  from `2.23e-15` (f64) to `1.87e-31` (DoubleFloat) and `1.92e-76` (Float256),
  passing the predeclared `1e-9`, `1e-25`, `1e-65` targets. Finite-part MadLoop
  inputs remain binary64 and are not a high-precision input oracle. Generic
  polarization mode-1 diagnostics fail, while every mode-4 generic diagnostic
  passes; these failures are recorded without changing gates. The aggregation
  checks coverage, process success, source identity and all physical controls.
  Reports are `validation/evidence/four-photon-{integrated,integrated-matrix,uv,cli-build}.json`.
- The H→gg comparison rerun with the coefficient-alias CLI passes all 36 rows
  at the unchanged `1e-9` criterion. The four previously failing f64 longitudinal
  pole rows now have error `5.06e-10`, compared with historical `1.07e-9`.
  The earlier report remains intact; the new report
  `validation/evidence/physical-hgg-cli-coefficient-aliases.json` records the
  actual binary build and distinguishes dependency snapshots. This is not a
  general precision-rescue implementation or proof of isolated causality.
- The combined NinjaRS workspace passes 171 tests across 49 suites, with no
  failures or ignored tests under `--include-ignored`. Strict workspace Clippy,
  formatting, the dependency bootstrap and diff checks pass. Commands, source
  hashes and logs are retained in
  `validation/evidence/batch-dependency-workspace-checks.json`.
- C0 derivative controls and their domain audit are pushed directly to
  oneloopmaster main at `7ed9f6e49a56e1018a5dbac8c859187be2a3e7df`; no PR was
  created. Both new tests pass on the combined main tree. They cover four
  massive spacelike real centers with both Källén signs, equal/unequal masses,
  higher/mixed complex seeds, degree-five scaling and mixed batches. Independent
  Float384 differences converge, with largest final discrepancy `5.34e-33`.
  Evidence is in `validation/evidence/c0-derivative-controls.json`. An apparent
  complex-invariant scalar discontinuity was traced to bypassing the documented
  real-invariant contract through unchecked exact expressions. The tested local
  complex derivatives do not require arbitrary complex scalar invariants.
  No NinjaRS master domain gate is broadened by these fixed-point controls.

### CLI coefficient provider and physical end-to-end scope (2026-10-16)

- Added `--numerator-evaluator symbolica|symjit-o2` to reduction and benchmark
  commands. The optional `numerator-jit` feature compiles all distinct physical
  polynomial coefficients into one Symbolica/SymJIT O2 evaluator, using the
  existing polynomial and evaluator APIs. The plan lives in `ninjars-graph`;
  typed numerical binding and dispatch live in `ninjars-cli`. No reduction-core
  API, generic bound or algebra representation changes.
- Each complete evaluation computes coefficient heads for all current samples
  before binding the existing polynomial numerator. The benchmark's timed loop
  includes this work; diagnostic coefficient timing is reported separately and
  removed from the residual attribution. Exact preparation and compilation are
  recorded as setup. Raw external parameters remain in the numerical pool;
  coefficient values are not precomputed outside end-to-end timing.
- SymJIT mode supports f64 only and rejects higher-precision requests before
  graph preparation. The existing Symbolica mode remains the default. The new
  generic coefficient interpreter preserves DoubleFloat/Float precision for
  controls. A nonfinite coefficient rejects the whole preparation binding
  transaction before reduction; compacting valid preparation rows remains
  outstanding and is documented in `docs/numerator-coefficients.md`.
- Focused controls pass for exact reconstruction, graph factor application,
  complex/lazy coefficient heads, registered constants, zero/constant plans,
  permutations/tails, transactional errors, and direct/Mu/T3/T2 callback parity.
  CLI integration checks also pass for synthetic and physical box reductions,
  both mu modes, timing attribution and precision rejection. Full workspace
  verification passes **185 tests across 51 suites**, with zero failures or
  ignored tests; strict Clippy, formatting, and the feature-disabled rejection
  control also pass. Commands, logs and source hashes are retained in
  `validation/evidence/numerator-symjit-workspace-checks.json`.
- The compiled provider also passes the complete f64 six-box four-photon
  matrix: **36 physical rows** and **12 generic diagnostic rows**, both mu
  modes, fixed MadLoop mode four, the unchanged scaled `1e-9` criterion,
  finite decomposition, physical graph sums and independent exact pole
  controls. Prepared input hashes match the earlier Symbolica-provider
  oracle exactly. `validation/evidence/four-photon-symjit-o2.json` records
  the gates and links the immutable matrix/build records. Aggregator mutation
  controls reject incomplete or corrupted evidence. These are correctness
  controls; no new performance acceptance is claimed.
- The matched reference harness still exports the original physical numerator
  and evaluates its external coefficients inside C++ callbacks. Supplying
  precomputed coefficients only to Rust would change the measurement boundary.
  The next harness work must share the coefficient basis/provider for isolated
  reduction and include coefficient execution for both complete physical
  evaluations. A shared master-provider reference adapter is also outstanding.
- A complete physical numerical boundary starts from momenta, helicities and
  model parameters. Timing only prebound polarization/component fixtures omits
  wavefunction and dependent-parameter evaluation. Both boundaries must be
  labeled explicitly; the full benchmark must include the latter operations
  before coefficient evaluation. The validation-only bridge and detailed
  measurement design are being developed separately from the reduction core.

### Shared helper storage and gg→HH validation (2026-10-16)

- [Symbolica PR #87](https://github.com/symbolica-dev/symbolica/pull/87)
  applies the existing stack allocator to freshly lowered shared-function
  bodies, reserving explicit and captured inputs. It adds no allocator or
  algebra. The focused upstream head is `417414bdebd9a8395a409d8cd216594cc7e47d50`;
  the tested integration is `fd7bced5307e94e994d5deba63c60b8380aa7d02`.
  Source and generation-only layout evidence are archived in
  `validation/evidence/helper-stack-generation.json`. D0 arena storage falls
  from 210,393 to 2,405 complex cells per lane; its generated batch source is
  still about 59.4 MB and has not been compiled or enabled in production.
- oneloopmaster main `e9d14f7b5fa546af33fced2a421c616220178643` pins all three
  build roots to that public Symbolica integration. The combined selected
  upstream run passes 123 tests with five existing diagnostic/child-helper
  ignores, including all 293 scalar master fixtures, JIT/cache controls,
  native batch controls and C0 higher/mixed derivative controls. Both Python
  roots compile with their actual public git dependencies and locked files.
  `validation/evidence/helper-stack-upstream-checks.json` retains commands,
  logs and dependency identities. Zero open oneloopmaster PRs were verified
  after this direct-main push. NinjaRS dependency pins and reproducible patch
  bundle now include these changes. Combined validation again passes **185
  tests across 51 suites**, with no failures or ignored tests, strict Clippy,
  formatting and dependency bootstrap verification. Full logs and source
  hashes are retained in `validation/evidence/helper-stack-workspace-checks.json`.
- Independently generated **gg→HH** fixtures contain two Higgs-channel
  triangles and six top-quark boxes. FeynKit's existing preparation correctly
  moves the Higgs bridge into the overall factor. Exact shared-family mappings
  and color controls fix the raw factor `−1/2` and integrated factor `−8π²`.
  All 4,480 independent raw numerator checks pass; 8,960 separate MadLoop
  raw/wrapper checks also pass. The retained DOT graphs keep symbolic momenta,
  wavefunctions and model couplings.
- The integrated matrix passes **192 physical rows** at four helicities,
  f64/DoubleFloat/Float256, both mu modes, with per-graph Laurent/decomposition,
  physical sums and mu-mode agreement. All 192 generic/Ward per-graph rows
  pass fixed reference mode four. Independent Symbolica differentiation gives
  triangle poles `Delta_q N / 8` and box poles `Delta_q² N / 192`; the eight
  graph poles cancel exactly for all eight states. Physical pole errors are
  at most `4.20e-14`, `2.70e-30`, and `2.01e-75`, below the fixed respective
  `1e-9`, `1e-25`, and `1e-65` gates. Evidence and reproduction instructions
  live under `validation/evidence/gg-hh-*` and `validation/madloop/gg_hh/`.
- Diagnostic limitations are retained: 36 of 144 individual Ward rows fail
  reference mode one; the summed both-Ward pole misses the fixed target for
  both f64 mu modes (`1.015e-9`, `1.053e-9`) and both DoubleFloat mu modes
  (`2.244e-25`, `2.157e-25`). Float256 passes. These are not physical-helicity
  gates, and are not reported as Ward acceptance. MadLoop finite-part inputs
  remain binary64 even in mode four. Fifteen aggregation mutation regressions
  pass. No performance acceptance is implied.

### Matched-provider bridge and physical numerical inputs (2026-10-16)

- Added an independent validation workspace under
  `validation/performance/shared_provider`. Both reducers can call the same
  non-inlined Rust numerator and master entry points through dimensioned,
  component-copied arrays. The retained Rust session constructs its topology
  and geometry once. Production crates have no C++ dependency or FFI change.
- Frozen evidence in `shared-retained-provider-probe.json` passes five Rust
  regressions, 291 callback components, 48 bubble Laurent comparisons, 80
  retained-session components, eight explicit mu-squared split comparisons and
  12 C0/D0 adapter-order components. The union contains 12 exact master keys;
  frozen replay covers 106 rows without additional backend execution and rejects
  an intentional missing key. Strict Clippy and formatting pass. Earlier
  numerator-only and master-union checkpoints are retained separately.
- These controls are not timing acceptance. Larger integrated batches, physical
  graph integration, descriptor/master staging allocations and measured provider
  costs remain outstanding. A `q0^2` bubble has matching total Laurent values
  but different cut/rational splits between residue bases; the discrepancy is
  preserved and rational-separation acceptance remains incomplete.
- The bridge exposed a nonsingular Euclidean C0 failure at invariants
  `(-1,-2,-3)`, squared masses `(2,3,5)` and squared scale `7`:
  `MasterFailure::NonFiniteResult`. A positive-result regression is explicitly
  run separately and currently fails. It is recorded as an acceptance blocker,
  not counted among passing controls; isolated upstream diagnosis is underway.
- `validation/performance/physical_inputs` prepares a top-quark four-photon box
  from numerical momenta, helicities, top mass and electromagnetic coupling.
  It uses FeynKit's shared wavefunctions, model bindings and routing, plus
  Symbolica's propagator and multi-output coefficient evaluators. Three distinct
  momentum points and three helicities cover empty/single/tail/permuted batches
  through 1024. Wavefunction phases agree with HELAS to `1.12e-16`; coefficient
  provider parity and full polynomial reconstruction pass the fixed `1e-9` gate.
- A stronger Float256 coefficient oracle finds a vanishing head evaluated as
  `1.72346e-9` by both unfactored f64 providers. Accordingly the archived report
  has `passed:false`; agreement between providers is not sufficient evidence.
  An isolated exact Symbolica `factor()` experiment reduces maximum errors to
  `7.45e-12` (interpreter) and `5.04e-12` (SymJIT O2), passing the same gate.
  This is a candidate preparation policy, not a general stability guarantee.
  Public API, factor implementation/examples, and FeynKit's installed FeynCalc
  usages were checked before composition; no factoring algorithm was added.
- Full physical timing must include per-point wavefunctions, dependent model
  parameters, routed propagators and coefficient evaluation for both paths.
  The existing prebound-component fixture benchmark remains explicitly narrower.
  Large pentagon integration and controlled end-to-end measurements are pending.

### gg→HHH independent graph preparation and raw matching (2026-10-16)

- Independent FeynKit and unmodified MadLoop generators agree on 50 top-quark
  graphs: eight triangles, 18 boxes and 24 pentagons. The triangles distinguish
  two quartic-Higgs trees from six nested cubic-Higgs trees; the exact matching
  preserves tree identity, fermion orientation and ordered propagator routing.
  Shared `IntegralFamily` mappings verify every transformed denominator.
- Symbolic DOT fixtures retain momenta, polarizations and model couplings.
  Loop-independent Higgs bridges are included exactly once in the overall
  factor. Exact controls derive the raw factor `-1/2` and integrated `-8*pi^2`
  normalization independently. Bound loop rank is at most four (at most two
  for triangles), so the 70-point complex degree-four grid is sufficient.
- All **28,000 independent raw numerator checks** pass the fixed scaled `1e-9`
  criterion, with maximum `6.78444e-13`. The separate **56,000 MadLoop internal
  wrapper checks** also pass, using fixed CutTools modes one and four, grouping
  disabled, four-dimensional numerators and R1 without R2/UV.
- The 36,180,542-byte reference JSON is retained losslessly as a deterministic
  2,351,000-byte gzip archive. Both hashes and byte counts are recorded; ten
  archive-integrity and missing/duplicate/nonfinite mutation tests pass.
  Evidence is under `validation/evidence/gg-hhh-{generation,raw,reference-archive}*`;
  reproduction instructions are in `validation/madloop/gg_hhh/README.md`.
- The 2,400-row integrated comparison is running against a separately copied,
  immutable development CLI on pins `fd7bced5/e9d14f7`. Raw matching is complete;
  integrated acceptance, pole controls and performance remain separate gates.

### Shared physical pentagon preparation and reduction (2026-10-16)

- A frozen numerical pool supplies three distinct on-shell gg→HHH points at
  center-of-mass energies 450, 525 and 600, fixed helicity, zero widths and
  squared scale 10000. Shared FeynKit momenta and Symbolica exact construction
  verify conservation and mass shells before rounding each component once.
  The MadLoop harness reads the same stored binary64 values unchanged; 420
  internal raw/wrapper checks and six fixed-mode integrations pass. Five
  boundary and coverage mutation tests pass.
- The validation-only physical provider evaluates five wavefunctions, dependent
  model parameters, 57 exact-factored polynomial coefficient heads, routed
  propagators and the S matrix on every preparation call. Both reducers consume
  the same Rust numerator callback and numerical input preparation. SymJIT O2
  remains entirely in the preparation adapter, outside the reduction engine.
- All 210 independent raw numerator checks pass (maximum scaled error
  `3.78e-15`). Float256 coefficient controls exactly promote the binary64
  component inputs and pass at `2.36e-15`; they do not establish higher-precision
  physical wavefunctions. The independently verified graph26→MadLoop55 mapping
  retains the `-1/2` raw and `-8*pi^2` integrated factors.
- Integrated widths 0, 1, 3, 8, 31, 32, 128 and 1024 pass **3,681 Laurent
  component comparisons**, with reducer agreement `2.14e-16` and fixed MadLoop
  mode-four agreement `1.15e-16`. Preparation covers 3,705 rows and 18,525
  wavefunction evaluations. Four failing-input controls preserve output and
  prior provider bindings; changed model parameters also pass the coefficient
  oracle. Six Rust regressions, strict Clippy and formatting pass.
- Frozen master replay uses the actual union of 85 requested integrals.
  Separate live mode bypasses every result-table lookup and insertion: original
  Ninja executes 150 master rows, while NinjaRS executes 120 rows in five
  batches. Repeating NinjaRS adds another 120 rows and five backend calls,
  verifying that repeated points still execute the master expressions.
- Evidence is `validation/evidence/shared-physical-provider-probe.json` and
  `gg-hhh-changing-*.json`, with before/after source guards, artifact hashes and
  commands. No optimized timing, separate production-backend performance, or
  cross-basis rational-separation acceptance is claimed. The Euclidean C0
  exceptional point remains a separately documented failing regression at these
  pins; its isolated fix is still under validation.

### Upstream conditional reuse and regular C0 limit (2026-10-16)

- Symbolica [PR 88](https://github.com/symbolica-dev/symbolica/pull/88), focused
  head `f629ebc4`, reuses completed conditional expressions in their parent
  scope while preserving branch-local evaluation and arithmetic order. The
  downstream integration is `2c06f245`. It passes 51 upstream and 69 integration
  evaluator tests, all 293 original master fixtures across three precisions,
  existing batch/JIT controls and C0 derivative tests. The new shape tests fail
  on the old revision. Supplementary sources and reproduction instructions are
  in `validation/symbolica/conditional_cache`; raw logs and identities are in
  `validation/evidence/conditional-cache-upstream-checks.json`.
- The D0 instruction graph shrinks from 209,051 to 78,040 instructions and its
  generated batch source from 57.59 MB to 14.35 MB. These master batches have
  not been compiled or enabled in production; diagnostic construction timings
  are not a runtime performance result.
- The regular Euclidean C0 failure is fixed on oneloopmaster `main` at
  `c838e8ed`, following separately recorded mechanical regeneration `06e24194`.
  An inactive, locally constant sheet correction now skips its singular
  logarithm through the existing lazy helper. Native and interpreted values
  previously returned NaN at `(-1,-2,-3; 2,3,5; scale_squared=7)`. Independent
  60/90-digit parameter quadrature supplies the finite control; 39 nearby
  point/routing controls and central higher/mixed derivatives pass.
- The selected upstream suite passes 82 tests with three explicitly ignored
  diagnostics. The nearby f64 third derivative still misses its unchanged gate
  identically before and after the patch (`1.09690e-8` error versus `9.32021e-10`
  bound). Both asserting failures are retained. Evidence, source hashes and
  commands are in `validation/evidence/c0-projective-validation.json`.
- The upstream push was a verified fast-forward, without a new oneloopmaster
  PR. Primary dependency checkouts remain on `fd7bced5/e9d14f7` until the frozen
  physical matrix completes; adopting these tested changes is the next step.

### Optional factored coefficient provider (2026-10-16)

- Added `PolynomialCoefficientPlan::factor_coefficients` using only Symbolica's
  exact `factor`, `map_coeff` and `flatten(false)` APIs. It rebuilds the template
  and deduplicates equivalent heads. The CLI exposes `symjit-o2-factored` for
  reduce/compare/bench; the existing defaults and reduction-core API are unchanged.
  Exact factoring is setup work, with a separately reported duration included
  in total coefficient preparation. Per-point evaluation remains inside each
  complete-evaluation batch. Both JIT policies reject non-f64 carriers explicitly.
- The public-provider physical probe reproduces the successful factored error
  bounds (`7.45e-12` interpreter, `5.04e-12` JIT) against Float256 while retaining
  the unfactored `1.72e-9` failure. Its source and linked artifacts remained
  unchanged during the run. Evidence is
  `physical-numerical-inputs-factored-provider.json`; factoring is an optional
  evaluation-order choice, not a general conditioning guarantee.
- Exact reconstruction/deduplication, weighted rank, cancellation, inactive
  singular branches, both mu modes, physical box reductions, timing attribution
  and feature/precision rejection pass. Combined validation passes **187 tests
  across 51 suites**, including the five separately executed original-Ninja
  reference tests. Strict all-feature/all-target Clippy and formatting pass.
  `factored-coefficient-workspace-checks.json` retains commands, source hashes,
  logs and earlier rejected attempts. One overlapping feature-build attempt
  rewrote a CLI artifact during subprocess tests; final validation serialized
  build/test invocations in this worktree. No tolerance was changed.
- Implementation and validation were isolated in the coefficient-factoring
  worktree while the fixed gg→HHH matrix guarded primary runtime sources.
  These checks establish correctness of the optional provider, not performance
  acceptance or complete large-graph end-to-end timing.

### gg→HHH integrated physical milestone (2026-10-16)

- The frozen `fd7bced5/e9d14f7` CLI passes all **2,400 integrated rows**:
  50 graphs, eight states, three precisions and both mu-extraction modes.
  All 1,200 physical rows and 1,200 generic/Ward diagnostic rows pass fixed
  reference modes one and four. Finite decomposition, mu-mode agreement and
  summed-graph controls pass. Maximum physical mode-four error is `1.10e-15`;
  all runtime source and process guards pass.
- Independent Symbolica differentiation computes 400 per-graph poles. Rank
  bounds are checked for every topology; harmonic higher-degree negative
  controls guard the Laplacian formula's domain. All eight exact graph sums
  vanish. Physical individual pole errors are at most `1.76e-16`, `4.58e-32`
  and `5.54e-77` for f64, DoubleFloat and Float256. The summed numerical poles
  also meet the unchanged `1e-9`, `1e-25` and `1e-65` targets. Generic/Ward
  diagnostic gates pass for this process.
- The complete 9,480,296-byte matrix is archived losslessly as a deterministic
  603,164-byte gzip, with both identities in `gg-hhh-integrated-archive.json`.
  Plain `gg-hhh-integrated-aggregate.json` and `gg-hhh-uv-poles.json` retain
  independent checks. All 23 aggregation/archive mutation regressions and 15
  reference/pool controls pass; source, CLI and linked-library guards pass.
  Reproduction is documented in `validation/madloop/gg_hhh/INTEGRATED.md`.
- This establishes the fixed physical correctness suite through pentagons.
  Finite-part MadLoop inputs remain binary64, performance acceptance is still
  pending, and previously recorded four-photon/gg→HH diagnostics are unchanged.

### Adopted conditional reuse and C0 dependencies (2026-10-16)

- Primary NinjaRS now pins Symbolica integration `2c06f245` and oneloopmaster
  `bf7632a9`. The consolidated Symbolica patch reproduces the new tree from its
  upstream base; dependency bootstrap verification passes. The oneloopmaster
  pin was pushed directly to `main`, without a new PR, after combined validation
  with the lazy C0 correction and conditional reuse enabled together.
- The combined dependency suite passes 118 tests with three explicitly retained
  ignored diagnostics. Both Python build roots pass locked checks with the
  actual public git dependencies, and all three manifests resolve one shared
  Symbolica/Numerica/Graphica revision. Evidence is
  `conditional-c0-upstream-checks.json`; it includes the unchanged nearby f64
  third-derivative limitation by reference to the C0 diagnostic evidence.
- NinjaRS passes **187 tests across 51 suites**, including all five separately
  executed original-Ninja reference controls, with strict all-target/all-feature
  Clippy and formatting. Build and test commands ran serially. Logs, source
  identities, compiler details and bootstrap output are archived in
  `validation/evidence/conditional-c0-workspace-checks.json`.
- The completed physical 50-graph matrix retains its original immutable binary
  and older pins. Updated-pin optimized shared-physical controls and preliminary
  timing diagnostics are being run separately. No performance acceptance or
  global derivative-accuracy completion is implied by this pin update.

### Matched large-graph preparation and complete timings (2026-10-16)

- The validation harness now separates physical preparation, reduction with a
  common numerator and frozen master union, complete evaluation with common live
  masters, and complete evaluation using original Ninja/OneLOop versus
  NinjaRS/SymbolicaMasters. Every complete call includes raw-momentum staging,
  FeynKit wavefunctions, dependent model values, all factored SymJIT O2 coefficient
  heads, propagators, reduction and integration. Both reducers receive the same
  batched preparation; original Ninja subsequently reduces individual lanes.
  Setup and driver I/O are excluded, ordinary allocations and common callback
  FFI/staging are included. This is not unmodified CLI latency.
- At current pins `2c06f245/bf7632a9`, the optimized physical oracle passes
  210 independent raw numerator checks, a Float256 coefficient oracle and 3,681
  Laurent components over widths 0, 1, 3, 8, 31, 32, 128 and 1024. All 16
  scope/workload/width-one-or-eight correctness records pass, including the two
  production master backends. Seven standalone Rust tests, eight paired-statistics
  tests, strict all-feature Clippy and formatting pass. Source guards remain
  unchanged. See `shared-physical-release-probe.json` and the three
  `shared-workload-{release-correctness,allocations,diagnostic-timing}.json`
  indexes with lossless deterministic gzip archives. Root independently verified
  every archive hash/size and current source identity.
- The first changing-point diagnostic finds a reduction gap independently of
  master execution: paired Rust/original median ratios are 3.44 at width one and
  3.15 at width eight. Complete common-live ratios are 1.027 and 0.966; complete
  production-backend ratios are 161.8 and 200.7. The current interpreted
  production masters dominate complete Rust evaluation. These results identify
  optimization work; performance gates are **not met or accepted**.
- Measurements used CPU35 with other host workloads present, four balanced
  pairs, and one call per block. The standalone release profile has no explicit
  LTO/codegen-unit override, unlike the main CLI. For multi-round records the
  current driver sums individual call intervals with counters between calls.
  Final gates require aligned profiles, contiguous longer timed blocks, all
  requested widths, repeated runs and reviewed host conditions. Full raw paired
  samples, uncertainty, counters and CPU/process snapshots remain archived.
- Separate Rust-allocator diagnostics observe 162 allocations at width one and
  559 at width eight for changing complete production evaluation. Preparation
  accounts for 6/48, validation numerator transport for 25/30, and residual
  reduction/transport for 131/481; direct warmed masters allocate no Rust heap
  objects in this control. C++ and native GMP/MPFR allocations are unobserved.
  A source audit matches the residual counts to repeated temporary containers,
  residues and box-projection copies. Persistent workspace reuse and compiled
  batched master execution are the next independent optimization targets.
- The shared raw physical-input library and explicit generated denominator
  dispatch are being validated in isolated branches. Main production sources,
  dependency pins and earlier evidence were held fixed throughout this baseline.
  Rational-separation parity, broader Dual domains and final performance
  acceptance remain outstanding.

### Current optimization priority (user clarification, 2026-10-16)

- Prioritize NinjaRS reduction latency and throughput until they are on par with
  or faster than original Ninja. Hold the numerator and precomputed master
  providers fixed for these comparisons. Advance workspace reuse and generated
  reduction kernels first; further numerator-provider and production-master
  integration is deferred behind this reduction target.
- Report time spent computing one-loop masters separately from reduction time
  excluding masters, and retain total evaluation time. Use separately instrumented
  provider calls and explicit scopes; do not infer master time by subtracting
  timings from different runs. Keep uninstrumented aggregate timing alongside
  the stage diagnostics and the matched precomputed-master reduction control.
- The intended fast numerator representation is a symmetric tensor polynomial
  in loop momentum, compiled as such with SymJIT O2 in the numerator-preparation
  package. Preserve the shared Symbolica/Spenso/Idenso representations and keep
  this provider separate from the core reducer. Resume this work after the
  reduction performance gap has been addressed.

### Optional generated denominator dispatch (2026-10-16)

- Integrated the explicit `native-kernels` feature. The library's
  `with_native_denominators()` and CLI `--reduction-kernels native-denominators`
  select mechanically generated Rust for T2/T3 denominator expansion. Other
  reduction kernels, integration and accumulation remain interpreted and are
  reported separately. The default remains unchanged. The CLI option is
  independent of `--numerator-evaluator`, including factored SymJIT O2.
- Ordinary f64, DoubleFloat and arbitrary-precision carriers are supported.
  Static/runtime Dual shapes `[00,10,01,20,11]` use generated arithmetic;
  other shapes retain the interpreter. Full ordered-shape validation remains
  active. A newly exposed malformed dynamic-Dual prototype now fails at setup
  instead of panicking later. Working and tracked input precision are retained.
- The catalog reproduces byte-for-byte from Symbolica instruction graphs and
  oneloopmaster's existing emitter. Forty-one library/native regression tests
  and sixteen CLI tests pass, including original-Ninja controls at all three
  ordinary precisions, tails, SymJIT composition, and feature-disabled errors.
  Strict Clippy, formatting and default-feature checks pass. Evidence is
  `native-denominator-dispatch.json` and `native-denominator-cli.json`.
- Direct warmed f64 denominator calls allocate zero; existing Dual shape-check
  allocations remain unchanged pending the separately published PR86. No
  end-to-end allocation or speedup claim follows from this limited catalog.
  The next measurements keep the same numerator and master providers and must
  attribute reduction and master time separately.

### Reusable reduction coordinator storage (2026-10-16)

- Integrated `c33d3c1`: all five cut coordinators and amplitude integration retain
  owned scratch between calls. Box projection uses flat momentum storage and
  borrowed quotient chunks. Borrowed request arrays remain local and are sized
  once per call; no unsafe lifetime conversion or arithmetic changes were used.
- Published results/statuses remain transactional on callback and master API
  errors. New tests cover every callback failure position, both box mu modes,
  widths 1/8/1/0/3/8, failed-lane recovery, accumulation and reset. A separate
  Float/Dual control compares all six higher/mixed components and precision
  metadata against fresh amplitudes across changing widths and 53/192/256-bit
  inputs. Forty-eight focused tests, strict Clippy and formatting pass.
- The unchanged optimized allocation harness observes complete production-path
  Rust allocations fall from 162 to 94 at width one and 559 to 421 at width eight.
  Shared frozen-master reduction falls from 225 to 157 and 860 to 722. Warmed
  reallocations fall from 46/128 to zero. Numerator, master, preparation and
  original-path allocation counts remain unchanged; maximum scaled numerical
  discrepancy is 2.24e-16. This measures Rust allocator activity only.
- Evidence is `validation/evidence/reduction-workspace-reuse.json` and its
  lossless allocation archive. Root independently checked the seven changed
  source identities and both archive hashes. No elapsed-time improvement or
  performance acceptance is inferred from allocation counts. Per-row residue
  coefficient/mask allocations remain; kernel attribution and separately timed
  master execution are being developed in isolated branches.

### Reduction and numerator cost attribution (2026-10-16)

- Added a validation-only own-process profiler and disposable instrumented
  adapter build. Two external CPU profiles and six kernel-span controls cover
  the physical n5/rank4 graph and ordinary/higher-rank bubbles at widths one and
  eight. Pre/post Laurent checks, source guards, geometry recomputation counts
  and zero frozen-master misses pass; eleven harness regressions pass. Root
  independently reran those regressions and checked both archive hashes/sizes.
- The shared physical provider uses SymJIT O2 for external coefficient heads;
  its direct/Mu/T3/T2 cut callbacks remain interpreted PolynomialNumerator on
  both sides. The observed interpreter CPU share therefore includes callbacks
  and cannot be attributed wholly to reduction kernels. Separate numerator
  spans are required in the matched timing report.
- The profiled exact kernels suggest source-axis projections and selected
  triangle-to-bubble corrections as the next generated catalog increment after
  the existing denominator option. Bubble costs differ materially from the
  physical graph, so both remain required controls. Intrusive spans include
  instrumentation overhead, especially for empty calls; they are not parity
  timings or evidence of an optimization gain.
- See `docs/reduction-kernel-profile.md` and
  `validation/evidence/reduction-kernel-profile.json` with its lossless archive.
  The immutable historical provider/core and disposable diagnostic rebuild are
  identified separately. Native kernels were disabled in these profiles, and
  their measurements are not performance acceptance evidence for current main.

### Residue buffer reuse and static numerical checks (2026-10-16)

- Integrated `ff2189a`: private pools retain discarded residue coefficient and
  mask buffers across reductions, resets and shrinking batches. Published
  buffers remain unchanged on callback errors. Tests exercise consecutive late
  failures, successful retries, empty/tail batches and actual allocation reuse
  across 53/192/256-bit Float values and six higher/mixed Dual components.
- The same optimized allocation harness observes production-path Rust
  allocations fall from 94 to 54 at width one and 421 to 101 at width eight,
  with zero warmed reallocations. Frozen-master reduction falls from 157 to
  117 and 722 to 402. Provider/preparation counts remain unchanged and maximum
  scaled numerical discrepancy remains 2.24e-16. Fifty focused tests, strict
  Clippy and formatting pass; evidence is `reduction-residue-pools.json` and
  its allocation archive. These are allocation measurements, not runtime gains.
- Integrated `f273489`: ordinary and Dual validity checks now use static marker
  dispatch in triangle/bubble/tadpole adapters. Ordinary carriers avoid indirect
  calls and redundant shape checks. Dual checks still inspect all components
  for finiteness, use the primal value for nonzero pivots, and compare the full
  ordered shape and value count. No arithmetic or public interface changed.
- Forty-nine focused static-check tests pass. After combining these changes
  with residue pools and optional native denominators, fifteen additional
  tests cover original-Ninja controls, higher cuts, failure isolation, reset,
  accumulation and precision reuse. Strict all-target/all-feature Clippy and
  formatting pass. Evidence is `static-kernel-checks.json` and
  `static-checks-residue-pools.json`; source snapshots identify both runs.
- Fresh optimized reduction measurements remain pending. Actual one-loop
  master computation, cached lookup/transport, numerator callbacks and full
  preparation are being separated in the validation harness before timing.

### Independent master execution accounting (2026-10-16)

- Integrated `036067f`: a separate execution-diagnostic build reports actual
  master calls, evaluated rows and time by family and backend. Original
  OneLOop's combined tensor-bubble calls remain grouped; initialization has its
  own record. Shared master lookup/transport excludes its nested backend time.
- Preparation, numerator callbacks, actual masters, shared transport and the
  reduction/staging remainder form an exact same-call additive breakdown.
  The report also provides total time excluding masters and full preparation.
  No stage is inferred from differences between independently timed runs.
  Primary aggregate builds contain none of these diagnostic timers.
- Aggregate sampling now uses one clock pair per contiguous block, counters
  outside the interval, complete three-point cycles, and one retained reducer
  per implementation. Both paths switch immutable prepared coefficient/input
  states for reduction-only timing; complete scopes include preparation.
  The standalone release profile matches workspace thin LTO/codegen settings.
- All sixteen development-profile correctness cases and sixteen execution
  cases pass, with maximum scaled error 2.24e-16. Frozen reduction has exactly
  zero actual backend calls/rows/time and zero table misses. Nine Rust and
  sixteen parser/statistics tests, strict Clippy and formatting pass. Root
  independently checked archive hashes, all additive/source-identity records,
  and reran the sixteen Python tests. Evidence is
  `shared-workload-methodology-validation.json` and its two lossless archives.
- These development runs validate accounting only and identify their earlier
  core revision explicitly. Optimized measurements on the combined workspace
  and static-check checkpoint remain pending; performance acceptance is false.

### Expanded generated reduction catalog and combined checks (2026-10-16)

- Integrated `fe28c15`: `with_native_kernels()` and CLI
  `--reduction-kernels native` select the currently generated catalog: T2/T3
  denominator expansions, bubble source-axis projection and selected
  triangle-to-bubble corrections. The existing denominator-only option keeps
  its narrow scope. Other reduction kernels, geometry, integration, numerator
  callbacks and masters remain independently selected; this is partial coverage.
- The new ordinary kernels support f64, DoubleFloat and Float. Dual correction
  and projection kernels retain the interpreter, with explicit coverage reports.
  Exact plans use the existing Symbolica expressions and FeynKit four-momenta;
  Rust arithmetic and precision-preserving constants regenerate byte-for-byte
  through oneloopmaster's emitter. Semantic quotient powers select each plan.
- Forty-one library/CLI/feature-disabled tests pass, including source origins,
  rank/degree variants, all ordinary precisions, tracked Float precision,
  empty/single/tail/permuted batches, full Dual components, failure isolation,
  reset and original-Ninja Laurent controls. Two existing C++ tests in the
  focused library suite remain explicitly ignored. Strict Clippy, formatting
  and feature-disabled checks pass. Evidence is `native-bubble-catalog.json`;
  root verified all 138 committed source hashes and seven validation logs.
- Integrated `cff029d`: box/pentagon/integration validity checks use the same
  static dispatch as the lower cuts. Integration still compares every Dual
  component for symmetry and zero weights, and checks the primal scale.
  Thirty focused tests and strict Clippy/formatting pass. Integrated `2c8da57`
  removes one redundant internal tadpole reference-vector allocation; ten
  focused controls pass. Neither change modifies arithmetic or callback APIs.
- Fifteen combined tests pass after integrating the catalog, all static checks,
  residue pools and tadpole view removal. Strict core/adapter Clippy and
  formatting pass; `native-static-combined.json` records exact source identity.
  Independent reviews found no blockers. No runtime improvement is inferred
  from these tests or direct warmed allocation controls.
- Release correctness and timing preparation retain immutable `0c77597` as
  the first contiguous-block baseline. The new catalog will be measured using
  an explicit validation-harness selector with identical numerator and master
  providers; it must not be silently substituted into baseline results.

### CLI accounting, further exact plans and release controls (2026-10-16)

- Integrated `962fc1c`: CLI `bench` now records integer same-pass stage
  durations, explicitly separates actual direct master execution and inclusive
  reduction excluding masters/coefficient preparation, and checks additivity.
  An inconsistent remainder is an error rather than a zero clamp. Primary
  timing is unchanged. Six CLI controls cover three precisions, workload/tail
  behavior and exactly zero master work when cuts stop before integration; a
  separate regression rejects overlapping stage intervals. Strict Clippy and
  formatting pass; evidence is `cli-stage-accounting.json`.
- Integrated `57506f9` as `1699401`: extracted existing tadpole source-axis
  projection plans (24→6 and 28→9) and the complementary-shift reference probe
  (8→28). Shared FeynKit Weyl bases and Symbolica expressions are unchanged.
  Ordinary/five-component Dual Rust bodies regenerate exactly. Fifteen focused
  tests pass; native runtime selection for these exports remains deferred.
  Corrected the earlier profiler's 8→28 label to probe; the separate geometry
  finish is 5→30. Immutable raw profile evidence remains unchanged.
- Root verified all nineteen committed extraction source hashes. Eighteen
  combined adapter/CLI tests pass after merging the exact plans and CLI
  accounting with existing native bubble kernels and static checks. Strict
  adapter/CLI Clippy and formatting pass. Evidence is
  `tadpole-plan-extraction.json` and `tadpole-cli-combined.json`.
- Immutable `0c77597` passes the optimized physical controls: 210 raw numerator
  samples, Float256 coefficient arithmetic, and 3,681 Laurent components at
  widths 0/1/3/8/31/32/128/1024. Maximum raw error is 3.77e-15, coefficient
  error 2.36e-15, shared integrated error 2.15e-16 and MadLoop integrated error
  1.17e-16. The coefficient oracle starts after f64 wavefunction evaluation;
  it does not validate high-precision wavefunctions.
- Twelve optimized frozen-master correctness cases and sixteen independently
  instrumented execution cases pass. Every frozen case has zero actual master
  calls/rows/time and zero misses. Root verified all three lossless archives,
  identical runtime manifests and exact same-pass/source-specific accounting.
  Evidence is `shared-reduction-methodology-release-0c77597.json`. These
  controls contain no primary timing samples; paired contiguous measurements
  follow under a coordinated pause of our builds. Host/SMT isolation remains
  unverified, so performance acceptance stays false.

### Paired interpreted reduction baseline (2026-10-16)

- Integrated `84cabc2`: immutable `0c77597` was measured with matched
  numerator callbacks and frozen master requests at widths 1/8/32/128/1024/31,
  for repeated and changing physical inputs. Each record has eight balanced
  paired blocks; all 192 individual blocks exceed 20 ms. Team builds paused,
  but unrelated host jobs and the unreserved SMT sibling prevent acceptance.
- Changing-input median NinjaRS/original time ratios are
  3.004/2.638/2.864/2.745/3.303/2.768. Width-one medians are 309.73/103.17
  microseconds per point. Reduction performance parity remains unmet. These
  measurements establish a new baseline, not a gain versus older methodology.
- Actual master backend calls/rows/time and coefficient preparation remain
  zero in this frozen scope; lookup and contraction are included in reduction.
  Both implementations have identical numerator sample counts. Live/production
  master timing is retained in the separate same-pass execution report.
- Root verified all six lossless archive hashes, recomputed twelve statistical
  summaries and 192 block normalizations, and checked source guards and counter
  equivalence. Evidence is `shared-reduction-timing-0c77597.json`; detailed
  boundaries and confidence intervals are in `BASELINE-0c77597.md`.
- Native-catalog comparisons remain next. A separate test-only probe is being
  prepared for Symbolica's existing complex SymJIT compiler on unchanged exact
  reduction plans; no production backend or generic carrier behavior changes.

### Explicit catalog controls and compiler probe (2026-10-16)

- Integrated `24b01c8`: the shared-provider harness now selects interpreted,
  native denominators, or the current native catalog at session construction.
  Compatibility defaults stay interpreted. Reports verify actual family
  coverage independently of numerator and master selection; invalid choices
  preserve the caller's handle.
- All three selections pass same-artifact frozen-master physical controls at
  widths 0/1/3/8. The native catalog's maximum scaled difference is 3.50e-18;
  its 85-entry union has no misses or backend execution during replay. The
  existing 210 raw MadLoop and integrated controls also pass. Constructor
  feature-on/off tests, nineteen Python tests, four C++ configurations, strict
  Clippy and formatting pass. Root verified the archived report/log hashes and
  all 1,933 runtime/dependency hashes against the measured worktree. Evidence:
  `shared-provider-kernel-selection.json`. This seven-family checkpoint still
  predates the pending tadpole runtime/coverage extension and has no timings.
- Integrated `071e291`: a test-only probe compiles seventeen unchanged exact
  reduction plans with Symbolica's complex SymJIT O2 API. 48,720 moderate finite
  complex component comparisons pass, including empty/tail/permuted batches;
  maximum scaled error is 4.65e-15. Strict adapter Clippy and formatting pass.
  Evidence: `reduction-jit-probe.json`; independent review found no blocker.
- This compiler setting executes scalar JIT programs per row. It is explicitly
  not a production batched backend. Extreme-range/near-singular division, Dual
  JIT and full physical JIT remain unvalidated. Its ignored microbenchmark has
  not been timed; native reduction measurement retains priority.

### Native tadpole source dispatch and atomic harness update (2026-10-16)

- Integrated `8727132` as `97dbfd9` together with `decc970` as `e48051a`:
  the native catalog includes triangle/bubble parent-axis tadpole projections
  and reports eight families through the validation bridge's ten-cell ABI.
  The denominator-only selection remains narrow; geometry stays interpreted.
- f64, DoubleFloat and Float ordinary kernels use the existing emitted Rust
  instruction batches. The supported ordered five-component Dual shape uses
  the existing Dualizer export; unsupported/reordered shapes keep the full
  interpreter. Full shape validation and all-component checks remain intact.
- Eighteen adapter checks, eleven CLI checks and one feature-disabled check
  passed on the dispatch branch. Warmed f64 projection evaluation allocates
  zero times. Dual shape checks retain their existing allocations. The named
  n3 Dual test preserves the existing C0 unsupported-derivative status; it is
  not a claim of integrated triangle Dual support.
- The eight-family harness passes feature-on/off constructors, nineteen
  Python tests, four C++ configurations, strict Clippy and formatting. All
  three selections agree on the physical frozen-master control through width
  eight (maximum scaled difference 3.50e-18, no misses/backend execution).
  Root verified all fourteen dispatch source hashes and 1,946 measured
  runtime/dependency hashes plus archive/log hashes.
- Twenty-five combined adapter/CLI tests pass on the integrated checkpoint,
  including all six exact-stage-accounting benchmark controls. Adapter/CLI
  all-target/all-feature strict Clippy and formatting pass. Evidence:
  `tadpole-native-dispatch.json`, `shared-provider-tadpole-selection.json`,
  and `tadpole-native-combined.json`. Optimized same-artifact measurements of
  all three selections are next; no speedup or acceptance is inferred yet.

### Extreme-domain audit blocks native timing (2026-10-16)

- The optimized `ee1f152` oracle passed all three selections through width
  1024 (6,135 complex components per selection against one frozen union).
  Twelve interpreted-mode timing-binary checks also passed. These bounded
  regular-input controls are explicitly superseded; no timing was collected.
- An independent audit found a real native numerical defect in the degree-zero
  triangle-to-bubble correction. With finite inputs c3=d0=d1=1e110 and the
  indicated source projection equal to one, the exact-plan Float256 result is
  approximately one, native f64 returns zero, and interpreted f64 is nonfinite.
  The defect reproduces in scalar and batched execution near the d0-cube
  overflow boundary. The native reciprocal's axis shortcut can mask a
  nonfinite intermediate as finite zero.
- The owning oneloopmaster fix and upstream regressions are being prepared in
  isolation. Timing is held until that fix is reviewed, pushed directly to
  oneloopmaster main, pinned here, and the combined controls pass again. The
  unfinished diagnostic build was cancelled without claiming its output.
  Evidence: `shared-reduction-catalog-ee1f152-superseded.json`; root verified
  both lossless archives and equal unchanged runtime-source manifests.
- Integrated `cc70c34`: an ordinary Laurent export inventory uses existing
  Symbolica plans and RustEmitter only. The n<=6/r<=n+1 suite has 58 nonempty
  and 31 empty plans; ordinary emitted bodies total 864,498 bytes, with 45
  identical-source/table groups. Generation, strict example Clippy and
  formatting pass. This installs no production kernels and claims no speedup;
  `laurent-catalog-inventory.json` records layouts and exact source identity.

### Independent algorithm audit requested (2026-10-16)

- Launched a separate audit of substantive changes to Ninja's reduction
  algorithm, beyond compiled arithmetic and storage reuse. It will examine
  cut/subtraction organization, coefficient extraction, shared work across cuts,
  tensor structure and alternatives in the primary literature.
- Candidates must preserve the established algebra policy, Ninja semantics,
  numerical stability, batching and generic/Dual support. The audit must
  distinguish arithmetic savings from execution overhead and propose bounded,
  falsifiable experiments before any production algorithm change.
- Integrated the report as `d1f932d`, in
  `docs/audits/reduction-algorithm-audit.md`. It identifies unused swapped
  bubble heads (already avoided by original Ninja), within-evaluation
  denominator reuse, rank-structural zeros, joint-orientation expressions,
  and an experimental transposed lower-cut projection. No speedup is inferred.
- Separate experiments now target denominator reuse with explicit same-call
  provenance and Symbolica operation counts after rank specialization. For
  stable n5/r4 cuts, denominator rows should fall from 120 to 60 for bubbles
  and from 100 to 20 for tadpoles. Tests must verify every residue coefficient,
  complete Dual components, failures/retries and changing phase-space inputs.

### Native reciprocal correction integrated (2026-10-16)

- Reviewed and pushed `53106429b849e5bcf843f111429be76227b4e0ce` directly to
  oneloopmaster main, following the user's direct-push instruction. NinjaRS's
  dependency manifest now pins it. The production patch is a five-line guard
  delegating nonfinite reciprocal operands to Symbolica's existing operation.
- Integrated permanent regression/evidence `d132d767` as `0451d17`. The exact
  degree-zero correction's archived 48 case/width configurations change from
  12 incorrect finite outputs to zero. f64/DoubleFloat overflow is exposed as
  nonfinite; Float256 controls remain accurate. This does not implement range
  extension or precision rescue. Root verified all 84 recorded artifact/source
  hashes. See `docs/native-domain-audit.md` and `native-domain/summary.json`.
- Twenty-four upstream tests pass. Strict upstream Clippy retains four existing
  warnings in untouched files; the focused run allows only the two documented
  baseline lints. Thirty-five integrated NinjaRS tests pass, as do strict
  adapter/CLI all-target/all-feature Clippy, focused new-test Clippy and format
  checks. Evidence: `native-inverse-combined.json` and its retained logs.
- Fresh optimized same-artifact oracle and execution controls will precede
  native timing. Superseded `ee1f152` controls remain archived. The library-only
  native geometry patch `91add1c` remains isolated; its 22 tests and 29 verified
  file hashes do not change this measurement checkpoint. Algorithm experiments
  likewise remain separate until independently validated.

### Algorithm audit: validated symbolic specialization probe (2026-10-16)

- Audit branch `codex/reduction-algorithm-audit` retains `a908e70` and `cc93e52`:
  an ignored Symbolica inventory and reproducible evidence, with no production
  dispatch changes. One inventory test, strict adapter Clippy and formatting
  pass; root verified five source hashes and both validation-log hashes.
- For n5/r4, structural source zeros reduce triangle-to-bubble instructions
  from 102 to 34. Restricting its swapped output additionally gives six
  instructions. Selected rank-specialized triangle/bubble-to-tadpole constants
  require eight/five instructions, compared with current full plans' 30/50.
  Weighted across the correction rows, these plans use 1,540 instructions
  instead of 8,020. Geometry, division, callbacks and masters are excluded;
  this is not measured runtime or numerical acceptance.
- An intermediate variant retaining both full bubble orientations uses 2,380
  correction instructions. It avoids changing callback batching/layouts and
  is a useful first numerical experiment after denominator reuse. Original
  Ninja also computes generic cubic/quadratic terms removable by these zeros.
- Public amplitude factories alone cannot guarantee source provenance.
  Specialization needs a private trusted boundary or validation of every zero
  component with generic fallback. Existing `IntegrationChecks::is_zero`
  supplies the complete-Dual predicate. The larger transpose experiment must
  keep master values symbolic to preserve current lazy zero-weight selection.

### Same-call denominator reuse reviewed (2026-10-16)

- Pushed isolated `codex/denominator-reuse` at `dc0100f`, pending integration
  after the frozen `de0f190` timing run. Additive combined stage callbacks
  retain third-party kernel compatibility; Symbolica reuses denominator rows
  only within the same call and with verified live input provenance.
- Seven new test groups and 38 existing regressions pass, with strict Clippy,
  feature-off and formatting checks. Two independent reviews found no blocker;
  root verified 19 source and retained-log hashes in `denominator-reuse.json`.
  Complete residue coefficients, precision and every higher/mixed Dual
  component agree with the separate-call control.
- Instrumented execution confirms n5/r4 bubble denominator rows fall from
  120 to 60 and tadpole rows from 100 to 20 per stable lane. These are operation
  counts, not measured runtime gains. Default third-party implementations
  retain temporary compact-buffer allocations.
- Also pushed isolated native geometry commits `91add1c`/`a12b805` on
  `codex/reduction-workspace-reuse`. The fixed-reciprocal dependency control
  passes all six new geometry tests. This opt-in geometry dispatch remains
  outside the current measured source, as does rank specialization.

### Reviewed reuse integrated after baseline sampling (2026-10-16)

- The frozen `de0f190` baseline completed all 18 cached timing processes
  (three backend selections, six batch widths). All source/artifact/counter
  guards passed, with no timing block below 20 ms. The shared-host results
  remain diagnostic and do not meet the original Ninja performance target.
  Actual master computation was zero in this reduction-only scope; separate
  complete-evaluation controls retain actual master execution accounting.
- Integrated denominator reuse as `ef6ab23`, and the symbolic rank inventory
  as `a7189ef`/`88a0d1f`. Production sources match the independently reviewed
  reuse commit exactly. The combined seven reuse test groups and one symbolic
  inventory pass, along with strict core/adapter library-and-tests Clippy,
  formatting and whitespace checks. Evidence: `denominator-reuse-integrated.json`.
- An isolated implementation now tests guarded n5/r4 rank specialization with
  complete-component zero checks and a generic fallback. Another isolated
  probe examines bounded native instruction batches: the pre-reuse native
  baseline can retain about 240 MiB of temporary arenas at phase-space width
  1024 for the n5/r4 case.
  These experiments have no accepted timing result or production integration.
  The more ambitious transposed reduction-map experiment remains research.

### Frozen native-catalog timing evidence retained (2026-10-16)

- Integrated `eb14882` as `774be29`: 18 lossless raw reports, 36 records,
  288 paired blocks and 576 individual blocks. The minimum block is
  39.138507 ms. Root independently verified all archive hashes, 35,100 source
  hash entries, 108 artifact entries and zero master calls/rows/misses.
- Changing-point native-catalog/original-Ninja median ratios are
  2.915/2.559/2.325/2.478/2.982/2.306 at widths 1/8/32/128/1024/31.
  Full interpreted/denominator-only controls, both workloads and confidence
  intervals appear in `validation/performance/shared_provider/TIMING-de0f190.md`.
  These ratios compare each selection with original Ninja; modes were not
  directly paired with each other. The uncontrolled host prevents acceptance.
- This source predates denominator reuse. Native coverage remains partial and
  numerator callback expansion depths differ. Actual masters remain separately
  accounted for. No source-only operation or memory saving is promoted to a
  measured improvement; the next experiments must retain those boundaries.

### Bounded native execution probe validated (2026-10-16)

- Integrated test-only source `7e68928` and evidence `b3c69ae` as
  `e9a016e`/`2e0c741`. Existing generated reduction arithmetic and production
  dispatch remain unchanged. The experiment compares whole rows with natural
  blocks of 8, 16, 32 and 64, including scalar tails and fresh workspaces.
- Eight controls pass. Each of f64, DoubleFloat and Float192 compares 421,872
  complex components exactly, including supported native Dual components.
  Complete reducer residues/statuses, retries, mixed significant bits,
  signed zeros and lazy shared-helper invocation counts are covered. Strict
  Clippy includes the generator example; feature-off and format checks pass.
  Root verified 11 source hashes, nine logs and the test executable hash.
- The ignored benchmark has not run. It requires a coordinated quiet window
  and retains raw JSONL blocks and provenance. No blocking execution policy
  has been added to production; a supported general policy belongs in
  oneloopmaster after measurement and validation.
- Guarded rank-specialization candidate `7f1591e` passed 56 tests but remains
  isolated: independent review found that its Dual alternate could derive
  working precision from the shape prototype rather than the explicit scalar
  prototype. A focused correction and mixed-prototype regression are required
  before integration. See `docs/native-blocking-probe.md` for the completed
  execution probe; rank specialization remains a separate experiment.

### Optimized denominator-reuse controls (2026-10-16)

- Integrated `dd5ea4d` as `d7d886e`. The frozen `553f406` source has the
  reviewed reuse runtime and unchanged dependency revisions. All three
  selections pass the physical oracle through width 1024, including empty
  and tail batches, Float256 coefficient controls and independent MadLoop
  comparisons. Maximum errors are unchanged from the prior checkpoint.
- Three six-width repeated/changing checks cover 66,096 Laurent components
  in 36 records. Backend calls, backend rows and frozen-union misses are zero;
  every source/artifact guard passes. Root verified all four lossless archives,
  7,808 source entries and 24 artifact entries. The three checks use one saved
  ordinary library and runner; prior benchmark artifacts remain intact.
- Evidence: `denominator-reuse-release-553f406.json`. These are optimized
  correctness controls, not timings. Cached artifacts are ready for a later
  coordinated sampling window; the blocked-execution probe is separately
  preparing optimized correctness controls before its microbenchmark.

### Guarded rank specialization and precision correction integrated (2026-10-16)

- Integrated reviewed `7f1591e` and focused fix `5c50941` as `73c3bb1` and
  `33e9cee`. The opt-in n5/r4 alternate uses Symbolica-derived structural zeros
  and full-component guards; arbitrary public inputs retain the generic batch
  fallback. Both bubble orientations and complete input layouts remain intact.
- The fix preserves the explicit scalar working precision when the Dual
  carrier uses a different precision. The retained Float256/Dual128 regression
  fails before the fix and passes afterward for direct stages and amplitude
  factories. Independent review reconstructed the failing source and verified
  its hashes. Root verified 18 current source hashes, four executable hashes,
  ten logs and both reproducer artifacts.
- Combined main validation passes 30 tests, with three intentional probes
  ignored, plus strict adapter library/tests/generator Clippy and formatting.
  The only integration conflict combined two independent test declarations.
  Evidence: `rank-specialization-integrated.json` and the preserved candidate
  and focused precision records. No performance claim follows from these tests.
- Native rank-specialized generation is being prepared separately using the
  shared exact plans and oneloopmaster emitter. The auditor is also preparing
  a bounded symbolic n3 lower-cut transpose experiment. Neither is production
  dispatch or a measured algorithmic speedup. Both remain source-only while
  the blocking and denominator-reuse timing window is coordinated.

### Native guarded corrections and bounded algorithm comparison (2026-10-16)

- Integrated native candidate `466a66d` as `b618106`. The explicit
  `with_native_rank_specialized_corrections` builder selects three ordinary
  and three ordered-Dual bodies mechanically generated from the same guarded
  Symbolica plans. Unsupported Dual shapes retain the interpreted alternate;
  generic fallback and the separate scalar working precision are preserved.
  No dependencies, core coordinator, numerator callbacks or masters changed.
- The candidate passes 21 focused controls and byte-identical regeneration.
  Independent source review prompted asymmetric complex derivative seeds and
  a distinct-first-derivative check, preventing component swaps from hiding
  behind equal expected values. Root verified 27 source hashes, three executable
  hashes and 14 retained logs. The ordinary bodies have 34/8/5 instructions;
  these are generated counts, not measured speedups.
- Integrated the bounded n3/r3 transpose experiment and evidence as
  `4ccaf54`/`b24dbde`/`e716113`. Both runs prove all five exact symbolic output
  identities and reproduce the same inventory. Separated forward, fused forward,
  factored transpose and fused transpose use 1954/1075/3359/1925 instructions.
  Transpose preparation costs 3054 instructions and application costs 305.
  Instruction-only reuse thresholds are H>=2 against separated forward and
  H>=4 against fused forward, with preparation charged at every changing point.
- This negative H=1 transpose result favors investigating forward fusion first.
  Abstract geometry does not test physical cross-cut cancellations. Five-output
  fusion omits intermediate failure/reconstruction heads; lazy master selection
  and transactional lane semantics remain requirements. None of these counts
  predicts runtime. Root verified 14 source hashes, eight logs, two inventories
  and the symbolic executable hash. See `docs/audits/transpose-probe.md`.
- Combined main validation passes 41 tests, with four intentional probes ignored,
  plus strict library/tests/examples Clippy and formatting. Evidence is in
  `native-rank-integrated.json`. The opt-in is not yet exposed in the CLI timing
  harness, and optimized physical timing remains outstanding. The frozen blocking
  probe is still compiling in thin LTO; its build cost is preparation, not timed
  reduction. Fresh timing authorization follows completion of all validation jobs.

### Bounded execution evidence and real-domain audit (2026-10-17)

- Integrated blocking diagnostic evidence `aae2f6a` as `6fa83dd`. Frozen source
  `b3c69ae` passed eight optimized controls, including 421,872 exact complex
  components per numerical domain. Its 24m43s build is preparation, not runtime.
  All 2,100 raw timing blocks are retained; the minimum is 21.467 ms and none
  fell below 20 ms. Root verified 30 source hashes, six raw-file hashes and the
  preserved executable, and recomputed all 42 groups' paired statistics.
- Block32/whole median ratios span 0.975–1.019 at phase width one and
  0.581–0.824 at width 1024; intermediate widths include regressions, reaching
  1.139 at width eight. These repeated synthetic kernel measurements include
  packing/arithmetic/unpacking and zero actual master computation. They exclude
  complete reduction, callbacks and changing physical inputs. CPU75's SMT sibling
  remained unreserved. No default policy or performance acceptance follows.
- An explicit owner-level policy is drafted in isolated oneloopmaster and NinjaRS
  worktrees. It retains Whole by default, complete-buffer validation, natural
  whole-row tails, all flattened Dual components and the explicit scalar
  prototype; tightened bounds release oversized arenas. Correctness validation
  is underway. Canonical dependencies remain unchanged. After review, any
  oneloopmaster publication goes directly to main as requested, without a PR.
- The independent algorithm auditor checked the actual reference harness and
  found complex masses on both sides, but real external shifts and longitudinal
  geometry in original Ninja. Symbolica already exports real-operand metadata;
  the current Rust emitters do not consume it for Add/Mul. The documented
  bounded candidate reuses Numerica mixed operations and preserves generic
  complex fallback, every Dual component and precision. Operand reordering,
  signed zeros and intermediate overflow require separate controls. No emitter
  change or measured gain is claimed. See `docs/audits/real-domain-arithmetic.md`.

### Matched physical reduction timing checkpoint (2026-10-17)

- Integrated `a41df4e` as `b36ccf2`. Frozen runtime `553f406` completed all
  18 cached processes: three selections, six widths, repeated and changing
  phase-space workloads, eight balanced paired blocks per record. All 576
  individual blocks exceeded 20 ms; the minimum was 38.015065 ms. No builds or
  links ran during sampling. The coordinated hold is released.
- Root independently reran verification of all 18 lossless archives, 35,136
  source entries (1,952 distinct files), 108 artifact entries and every paired
  statistic/bootstrap interval. Source and artifact guards pass. Actual master
  backend calls/rows and frozen-union misses are exactly zero in this scope.
  Coefficient preparation is excluded; geometry, ordinary callbacks, reduction,
  master lookup/transport and contraction are included. Separate older actual
  master-computation controls retain their original source identity.
- Changing-point native-catalog NinjaRS/original ratios are 2.726, 2.103, 2.217,
  2.302, 2.638 and 2.289 at widths 1, 8, 32, 128, 1024 and 31. The performance
  target remains unmet. CPU75 was pinned but its SMT sibling was unreserved.
  Each mode is paired with original Ninja, not directly with the other modes;
  earlier pre-reuse schedules are not paired before/after speedup evidence.
- The same numerator callback provider and equal sampled rows are used, but
  original Ninja requests fewer swapped-orientation expansion heads. Native
  geometry, guarded rank alternates and blocking are excluded from this frozen
  runtime. The latter two candidates' new CLI/API validation is now underway.
  See `validation/performance/shared_provider/REUSE-TIMING-553f406.md` and its
  evidence index for intervals, repeated-point results, counters and limitations.

### Rank selectors and supported native execution integrated (2026-10-17)

- Integrated rank-harness `5f0f7a9` as `d0fc61f`. CLI and shared-provider modes
  `interpreted-rank` and `native-rank` support n5/r4 explicitly. Existing family
  metadata remains generic/fallback coverage; conditional alternates have their
  own metadata and an additive accessor. Existing selectors and ten-cell ABI
  remain compatible, including historical three-mode reports.
- The candidate passes 22 test executions, including two strengthened tail
  reruns, strict CLI/ABI Clippy, C++ syntax and formatting. All five modes pass
  physical widths 0/1/3/8 (60 complex outputs each): maximum catalog difference
  4.00e-18, original integrated difference 2.15e-16. These use the dev profile
  and do not establish optimized performance or additional CLI precision.
  Root verified 27 source hashes, eight preserved executable artifacts, six
  physical artifacts, 47 log members and 1,970 physical source entries.
- Published reviewed oneloopmaster `c9dd82f` directly to upstream main, without
  a PR, and pinned it in `dependencies/manifest.json`. Integrated consumer
  `9013b78` and evidence `c9e51c2` as `d951046`/`13b3775`. Whole retains direct
  generated dispatch; fixed row bounds remain an explicit library opt-in.
- Independent review found no source blocker. Its suggestions are covered:
  callback errors explicitly permit writes in the failing block, and an
  executed asymmetric mixed-Dual n5/r4 factory control observes both generic
  and rank-alternate workspace bounds in both builder orders. All 58 distinct
  focused tests pass, including emitted lazy-helper/six-component-Dual tests;
  strict Clippy, formatting and feature-off checks pass. Root verified two
  owner and 15 consumer source hashes, 31 unchanged generated files, four test
  executables and 20 logs. The ignored timing probe remains unrun here.
- Before updating the canonical owner checkout, ten historical NinjaRS worktree
  aliases were pinned directly to the immutable old5310642 checkout. Their
  source bytes and archived benchmark artifacts remain intact. Root reverified
  all 1,952 distinct source files of the frozen physical timing runtime after
  updating these aliases. No new rank or supported-blocking timing is claimed.
- The next harness extension will use an independent positive
  `--native-block-rows` option, an additive constructor/query, and separate
  execution-policy metadata. Existing calls/reports retain Whole behavior;
  explicit blocking requires native coverage. Physical controls must precede
  same-artifact timings, with actual masters kept separate from reduction.

### Combined native policy checkpoint (2026-10-17)

- Main source `a6739ca` passes 49 tests: all 44 adapter unit tests, two CLI
  reduction rank tests, one CLI benchmark rank test, and two existing native
  selection compatibility tests. Four explicit probes remain intentionally
  ignored. Strict adapter/CLI library/tests/examples Clippy, feature-off
  compilation, workspace formatting and diff checks pass.
- Retained logs, commands, executable hashes, 170 source hashes, dependency
  revisions and independent candidate-verification summaries are recorded in
  `validation/evidence/native-policy-integrated/manifest.json`. Upstream
  oneloopmaster main was independently verified at `c9dd82f` after publication.
- The independent row-bound CLI/harness extension is in source review. The
  next algorithm audit examines swapped bubble expansion head liveness and
  the smallest compatible batched change. Neither effort has new timing
  evidence; the last physical reduction measurements still miss acceptance.

### Swapped bubble audit follow-up (2026-10-17)

- The requested independent algorithm audit has a concrete, bounded follow-up
  in `docs/audits/swapped-bubble-heads.md` (`a048832`). Original Ninja already
  requests reduced depth in the second bubble orientation. The proposed first
  control keeps callbacks unchanged and selects positive-power division and
  correction heads through Symbolica. For n5/r4 this reduces computed division
  heads from 60 to 40 and correction heads from 180 to 120 per lane; these
  counts are not runtime predictions.
- Existing combined and standalone kernel APIs expose full raw outputs. A
  defaulted projection-specific method is needed to preserve them. The opt-in
  must retain original rank/higher-rank layouts, full Dual checks, denominator
  reuse and complete residues. A failure confined to discarded arithmetic can
  disappear; the default and old raw APIs retain their original behavior.
- Callback trimming is a separate control because it doubles the shared
  wrapper's n5/r4 callback groups from ten batches of 2L rows to twenty of L
  rows. No implementation, numerical or performance claim follows from the
  audit. Current rank/block validation and performance remain the next checkpoint.

### Forward fusion with retained outputs (2026-10-17)

- The bounded n3/r3 follow-up (`bf6bde1`, evidence `39e2ba7`) keeps every existing
  forward block output, with the five integrated outputs first. Its single
  1.30-second symbolic run passes all 283 local identities and five integrated
  identities. The unchanged inputs include 60 callback heads and 39 symbolic
  master components; no provider or numerical master executes.
- The conservative retained-output graph has 3,177 instructions and 470 local
  temporaries, compared with 1,954 separate instructions (largest local plan:
  30 temporaries) and 1,075 five-output fused instructions (132 temporaries).
  This loses the earlier count advantage. It does not rule out smaller fusion
  regions or establish runtime behavior. Denominator/source-projection
  construction, validation ordering, compaction and lazy master control remain
  outside this experiment.
- Root verified 38 source hashes, the retained executable, 15 raw artifacts,
  complete/nonoverlapping output ranges, category totals and all reported
  counts. Strict Clippy with native kernels and formatting pass. The separate
  feature-off strict lint attempt hits the existing test-only `probe_workspace`
  dead-code configuration issue; that failed log is retained, with no unrelated
  source edit. See `validation/evidence/forward-diagnostic-fusion`.
- Whole-network fusion is not promoted. Arithmetic-only bubble pruning is now
  being prepared in an isolated source candidate, with independent sampled-cut
  reconstruction tests; its execution awaits review. The native row-bound CLI
  has passed both feature variants and is validating its standalone ABI and
  physical controls before the next optimized reduction checkpoint.

### Test-only native probe configuration (2026-10-17)

- Restricted workspace probe hooks to tests with native kernels enabled. Strict
  adapter library/tests Clippy now passes with both rust-codegen alone and
  rust-codegen/native-kernels; workspace formatting and diff checks pass.
  Production code is unchanged. Logs are in
  `validation/evidence/native-probe-cfg-lint`. The earlier failed feature-off
  fusion audit log remains intact as historical evidence.

### Native block CLI and physical controls (2026-10-17)

- Integrated candidate `8b86822` as `f17f64e`. CLI `reduce`, `compare` and
  `bench` accept an independent positive `--native-block-rows` for native
  selections; omission remains Whole. The additive validation ABI preserves
  existing constructors and queries and reports configured execution policy.
- All 33 test executions pass (21 Rust, 12 Python), including both feature
  variants, malformed input/sentinel controls and CLI serialized parity for
  f64, DoubleFloat and Float. Serialized comparisons use f64 and do not claim
  extra precision. Strict CLI/ABI Clippy, formatting and C++ syntax checks pass.
- On one development-profile physical executable/library/input, all three
  native modes give exact Whole/fixed-7 agreement for 375 complex components
  per policy across widths 0/1/3/8/31/32. All five Whole modes also pass the
  shared catalog control. Original Ninja agreement is within 2.2e-16 scaled
  error. These controls do not establish complete production-backend or
  rational-decomposition parity.
- Root verified 26 source hashes, eight preserved feature-specific executables,
  six physical artifacts, 62 archived logs, 1,971 runtime source entries and
  4,125 parity components across four reports. Evidence is in
  `validation/evidence/native-block-harness.json` and
  `validation/evidence/native-block-root-verification.json`.
- Release correctness now uses the same frozen source, a separate output
  directory, integrated widths through 1024 and the previously selected bound
  32. Timing awaits these controls. Master execution, preparation and reduction
  remain separately accounted; the existing performance gate remains unmet.

### Projection-only swapped bubble arithmetic correctness (2026-10-17)

- Source draft `97e095e` adds explicit ordinary/Dual factory and stage opt-ins,
  with a defaulted projection-specific combined method. Existing raw APIs and
  numerator callbacks remain full. Selected positive-head Symbolica graphs keep
  the complete original parameter/source ABI and remain interpreted, independent
  of native coverage or row policy. Numerical bases and the all-component rank
  guard retain their existing validation rules.
- Root's `bd01a8e` adds actual-master bubble amplitude agreement, accumulation,
  failed-lane isolation, recovery, reset and empty-input coverage. The native
  test-probe cfg fix is main `1d0106b` (candidate cherry-pick `5978461`).
- Bounded validation passed 87 tests: 31 with rust-codegen and 56 with native
  kernels also enabled, zero failed/ignored. This includes exact liveness,
  independent sampled reconstruction through rank n+1, raw APIs, scratch/retry,
  callback identity, f64/DoubleFloat/Float, static/dynamic higher/mixed Duals,
  whole-active-kind rank fallback, denominator reuse and workspace regressions.
- Strict core/adapter library/test Clippy passes both variants; workspace format
  and diff checks pass. The retained first lint failure was an unused test
  import, removed in `6da695e`; no numerical implementation fix was needed. The
  initial empty rank-filter attempt is retained and excluded from totals.
- Exact numerical source, dependency revisions, executable/argv identities and
  logs are recorded in `validation/evidence/swapped-bubble-arithmetic`; behavior
  and intentional discarded-overflow status differences are documented in
  `docs/swapped-bubble-arithmetic.md`. This is an opt-in correctness candidate,
  with no performance measurement or speedup claim. Future measurements must
  keep actual masters separate from reduction and the shared-master control.

### Integrated pruning and optimized rank/block controls (2026-10-17)

- Integrated pruning source as `048fa59`, root amplitude test as `020bc3d`, and
  the import-only cleanup as `c13cc2f`; evidence is `c1b6587`. Root independently
  verified all eight source snapshots, eight tested binaries and the 87 test
  executions. The complete native-enabled CLI compiles with the integrated
  core and adapter. All eight test binaries now have read-only preserved copies.
- Optimized physical controls from frozen `8b86822` pass across widths
  0/1/3/8/31/32/128/1024. Three native Whole/fixed32 pairs agree exactly;
  five Whole modes agree within 4.0e-18. Together these check 67,485 complex
  parity components; integrated agreement with original Ninja is within
  2.2e-16. Evidence `a113221` is integrated as `dffbd82`.
- Root verified all 41 archived release logs and ran the independent strict
  preflight against reviewed manifest
  `d12273632f58e85f3ce88336c3d8f42d29501ddd53cc6f074022b1f4015b1f47`.
  It checks all 1,971 source hashes, dependencies, release artifacts, actual
  runtime loader target, compiler configuration, modes/bound and physical widths.
- The archived protocol has twelve pure Python corruption/schema regressions.
  Its preselected 2x2 comparison isolates rank specialization and fixed32
  execution on one executable/library, across four cyclic process-order sweeps.
  Raw paired blocks and uncertainty remain separate from cross-setting process
  ratios. Actual master calls/rows and misses must be zero in reduction timing.
  No speedup or acceptance claim follows from these correctness controls;
  actual-master/preparation instrumentation and complete timings remain separate.
- A source audit identifies paired bubble denominator and source-projection
  evaluators as a smaller possible improvement beyond original Ninja's separate
  orientation loops. It would retain every output and share common longitudinal
  dot products through Symbolica merge. Only a bounded source inventory is being
  prepared; no production dispatch or numerical/performance claim is made.

### Timing validator counter correction (2026-10-17)

- The first ordinary timing process completed correctly, with zero actual master
  calls/rows and frozen misses, but strict postprocessing stopped: its expected
  lookup-row count was 50 rather than 40 per point. Source inspection and archived
  width1/8 measurements independently confirm D0:5+C0:10+B0:10+B1:10+A0:5;
  the absent degree-one x² coefficient gives zero B11 weight and no B11 call.
- Corrected only the external validator and its fixture, adding the retained
  actual counters as an independent regression. All thirteen Python tests pass.
  The first attempt's complete logs and raw samples remain archived as a failed
  validator pilot in `validation/evidence/native-rank-block-protocol-v2`.
  The entire same-artifact comparison restarts with the identical schedule,
  rounds, trials, modes and duration floor; no timing magnitudes motivated the
  correction and no partial favorable results are promoted.

### Continued algorithm audit and measurement preparation (2026-10-17)

- Paired-orientation inventory source `fda9047` rebinds the existing T2
  denominator and bubble source-projection graphs, retains all 30 outputs, and
  compares separate evaluators with two bounded Symbolica merge settings.
  Root and an independent reviewer checked the parameter maps against the
  core's actual orientation construction. It is not built or executed yet.
- The separate design note `ce2f781` recommends a private core-created request
  descriptor if the paired inventory is favorable. It must encapsulate its
  request slices and original-row pairing; arbitrary public requests retain the
  existing default path. Value-only equality is insufficient for precision,
  signed zero and higher/mixed Dual identity. No paired dispatch is implemented.
- A further source audit examines fusion to master weights and rational terms
  before master execution. This would keep lazy zero-weight request selection
  separate from the reduction graph. It must distinguish current intermediate
  failure checks from input/pivot constraints, and explicitly address optional
  diagnostics, callback eligibility and any changed stability contract before
  a production experiment. No performance benefit is established.
- The independent pruning harness source is `8a33a59`, based on `38f8569`. It
  adds an independent disabled-by-default ABI option, explicit interpreted
  configuration metadata, full/pruned five-head physical controls, and timing
  preflight checks for the same sources, selection, policy and covered widths.
  It passes independent source review, formatting and syntax checks. Its tests
  are unexecuted pending the current timing window.
- A separate instrumented-release driver is prepared for all four native/rank
  and Whole/fixed32 settings, widths 1/8, and preparation/shared-reduction/live
  shared/production-complete scopes. Source review checks exact accounting,
  reference/build/loader provenance and separation from primary timing artifacts.
  The instrumented build and execution are still pending.
- The full 96-process rank/block comparison continues under the preselected
  schedule. Source-only work proceeds while builds and numerical/symbolic tests
  are held. No partial timing results are selected or promoted.

### Master-weight fusion audit outcome (2026-10-17)

- The independent n3/r3 source audit identifies 20 pre-master outputs: thirteen
  separate master weights and seven per-cut rational terms. Keeping this
  boundary permits existing all-component Dual zero tests and lazy master
  requests; no abstract master value belongs in the reduction inputs.
- A global weight graph would need every callback head up front. A staged
  alternative retains the ten triangle residue slots and thirty bubble slots
  at their callback boundaries, then emits the twenty weights/rational heads.
  Three separate evaluators must be inventoried; a single sixty-output graph
  would not preserve stage-dependent callback eligibility.
- Geometry/input/pivot/scale constraints remain outside this experiment.
  Avoiding an overflowing intermediate can change a lane's acceptance relative
  to today's route; any production opt-in must explicitly validate that change.
  Reconstruction, residue diagnostics and cut stopping require the existing
  fallback. Weight zero-mask agreement and master-domain failures need separate
  numerical controls, even after exact algebraic identities pass.
- Source preparation now targets these two alternatives and the same separate
  pre-master blocks, excluding contractions/final summation from the comparison.
  It retains the existing 8 MiB/180-second limits and checks all weight/parent
  identities plus recovery of the five integrated expressions. No symbolic or
  numerical execution, production dispatch or timing claim has occurred.

### Broader reduction algorithm follow-up (2026-10-17)

- The independent source audit `3fc262d`, integrated as `0818970`, examines
  alternatives beyond the prepared paired-orientation and master-weight probes.
  It finds no current evidence for replacing Laurent reduction wholesale. Its
  ranked candidates and stopping criteria are in
  `docs/audits/reduction-algorithm-followup.md`.
- The strongest additional higher-rank candidate preserves a factored sequence
  of existing one-line Laurent plans instead of expanding their complete
  denominator product. Exact power relabeling retains the full weighted degree
  and mu-squared layout. The proposed first comparison covers n6/r7 and n5/r4
  bubbles, keeping callbacks and all final quotient heads unchanged. No new
  divider, symbolic manipulation, runtime dispatch, or experiment was added.
- Another candidate shares leading reciprocals among parents of one tadpole.
  For an all-active, successfully guarded n5/r4 evaluation, emitted ordinary
  correction bodies predict 80 reciprocal evaluations referring to 20 distinct
  denominator values. These are source counts, not observed dispatch or timing.
  The conservative candidate charges a separate 20-value preparation pass;
  existing numerator-division scratch is not a supported reusable table.
- Source-only staged master-weight inventory is frozen at `ee0c8b6`, including
  formatting, and paired-orientation inventory remains `fda9047`. Both await
  bounded symbolic execution after the active 96-process timing window. The
  existing release comparison keeps actual master computation excluded; the
  separate instrumented pass will account for preparation and live masters.

### Complete rank/block timing diagnostic (2026-10-17)

- The full preselected comparison on frozen `8b86822` finished successfully:
  96 processes, 192 records, 1,536 paired blocks, 3,072 individual blocks. The
  minimum block duration is 39.016648 ms, exceeding the fixed 20 ms floor.
  All samples are retained; no adaptive sampling or selective reruns occurred.
- Root independently checked all 299 archived files, command status/order,
  complete width/workload/setting coverage, source/dependency/artifact/loader
  identity before and after the run, and zero actual-master calls/rows/misses.
  It reproduced every deterministic paired-bootstrap summary and the four
  rotation contrasts. Evidence is in `validation/evidence/native-rank-block-timing-v2`.
- With changing inputs, native-rank fixed32 Rust/original ratios are
  2.5556/1.9643/1.9869/1.9627/2.0270/2.4028 for widths
  1/8/31/32/128/1024, respectively (medians of four process paired-ratio
  medians). All four settings and both workloads, per-process intervals and
  descriptive cross-process MADs are retained. Rank/block effects are modest
  and width-dependent; no default changes or causal historical comparison.
- Maximum scaled Laurent error is 2.09e-16. The shared host is not isolated,
  and the performance gate remains unmet. Scope1 includes callbacks,
  geometry/reduction, frozen lookup/contraction and staging, with actual master
  computation and physical preparation excluded. Original/Rust T2 head counts
  remain 40/60 by source inspection, not by an observed head counter.
- The coordinated timing hold is released. The paired and staged-weight
  symbolic inventories and physical pruning harness validation are running in
  separate caches. The instrumented execution wrapper passes seven pure Python
  regressions and its separate optimized build is running; it will report
  preparation, callback, actual-master and remaining execution spans separately.

### Validated staged master-weight instruction inventory (2026-10-17)

- The bounded symbolic test at `ee0c8b6` passed once, with all 90 exact
  identities. Separate pre-master blocks use 1,884 instructions; a single
  20-weight graph uses 1,097; three actual triangle/bubble/final stages use
  19+753+299=1,071. The full historical 1,954-instruction baseline is reproduced.
  Inversions fall from 81 to 36 in this abstract n3/r3 graph. These counts
  exclude geometry construction and are not performance measurements.
- The three stages retain all 60 callbacks (12/42/6), explicit prefixes
  0/10/40, and all ordered master weights/rational heads, without master values
  among their inputs. Largest local temporary count is 55, with 40 separately
  retained parent values; neither their sum nor the aggregate temporary count
  is a peak-memory measurement. Composition/identity limits were respected.
- Source and evidence are integrated as `9417a17`, `24b5663`, and `452187d`.
  Root verified 181 frozen source hashes, all retained artifacts including the
  executable, 15 archived files, the four successful build/symbolic/format/lint
  statuses, and equality of all 173 code/manifest files after integration.
  Evidence is `validation/evidence/integrated-weight-inventory`.
- Strict package Clippy and formatting pass. No production dispatch or ordinary
  numerical test ran: callback eligibility, finite/status behavior, lane
  compaction, precision/Dual accuracy, zero-weight master requests, and runtime
  still require a separate implementation and validation.
- The instrumented release build passed independent provenance review at
  manifest `0760a4215aa46f83431e78e24585522e3f93e86f582c313e1fee3a373ec5539b`.
  Its preparation/shared/live/production execution checks have started, using
  a separate copied instrumented artifact and preserving the primary timings.

### Performance gates and fast-path strategy (2026-10-17, Claude takeover)

User priorities, in order: (a) pure reduction time in NinjaRS on par with or
better than original Ninja; (b) one-loop masters through oneloopmaster's
*native* direct-Rust backend (not interpreted Symbolica expressions) at most
1.3× OneLOop's time; (c) end-to-end time (SymJIT O2 numerator coefficients +
reduction + masters) for a complicated physical diagram (gg→HHH top pentagon,
graph 26) on par with or better than MadLoop (MadLoop numerator + Ninja +
OneLOop) in double precision. Deliverable: `docs/benchmark/benchmark.typ` and
the compiled PDF with a coloured summary table (generation time and
per-point numerator/reduction/master breakdown with multipliers versus
MadLoop/Ninja/OneLOop) and sections on NinjaRS's innovations.

Survey findings that set the strategy:

- Instrumented single-call evidence shows the reduction remainder (excluding
  numerator callbacks and master lookup) is ~4–5× slower than original Ninja
  for n5/r4 (≈240 µs vs ≈55 µs at width one); the headline 2–2.5× ratio was
  compressed by a large, common, interpreted numerator. A scratch benchmark of
  the reference `libninja.a` with a null numerator and frozen masters gives
  ≈23 µs/point (≈18 µs with `-fcx-fortran-rules`, MadLoop's flag).
- Completing generated-kernel coverage would not close the gap: per-kernel
  packing, validation and dispatch overheads alone exceed Ninja's arithmetic.
- NinjaRS never enabled oneloopmaster's `generated-evaluators`; its masters
  ran through Symbolica's interpreter (~50 ms/point). The native backend's f64
  dilogarithm itself routed through Symbolica's MPFR polylog (32–250 µs per
  Li2), making native C0/D0 150–600× slower than Fortran at oneloopmaster
  c9dd82f. With a plain f64 Li2, C0 ≈1.0× and D0 ≈1.4× Fortran. Native
  B1/B11/B111/An did not exist.
- MadLoop declares rank 5 for the pentagon (Ninja's higher-rank path) and
  uses Ninja's compiled `TensorNumerator`.

Strategy (parallel worktree streams, merged by the coordinator):

1. `ninjars::fast`: a monomorphic, batched binary64 port of Ninja's
   reduction (all cut levels, rank ≤ n+1) with precomputed topology,
   allocation-free workspaces, one batched numerator call per cut/orientation
   and one batched master call per family; plus a batched port of Ninja's
   tensor numerator, generic adapters and replay providers for pure-reduction
   timing. The generic Symbolica engine remains the path for DoubleFloat,
   arbitrary precision and Dual numbers.
2. Numerator: map SymJIT O2 coefficient heads onto Ninja's tensor layout
   (`fast::tensor_layout`), shared phase-space pools and bench-case export.
3. oneloopmaster: fast f64 dilogarithm (Symbolica PR), native tensor bubbles
   and An, then C0/D0 generation/hand optimization to ≤1.3× Fortran.
4. Reference harness: original Ninja builds (`-O3`, `-O3 -fcx-fortran-rules`,
   cache disabled), a replay-based C++ driver measuring full/pure
   reduction/numerator/masters on identical bench cases, and optimized
   single-diagram MadLoop outputs with generation/compilation/per-point
   timings.

Commits so far: `7b57642` (fast-path complex type and provider traits),
`df776c3` (`perf` cargo profile), `29b8ff8` (tensor layout).

### Fast engine, native masters and end-to-end pipeline (2026-10-17)

- `ninjars::fast` (merged from `claude/engine-alt` via `claude/integration`):
  batched binary64 port of Ninja's reduction (all cut levels, any n, rank ≤
  n+1, Ninja defaults, skip rules, stability checks, per-lane failures),
  truncated-jet tensor numerator, mechanically translated and
  rank-specialized coefficient-level subtractions, generic adapters and O(1)
  replay providers. Adversarial review: no engine bugs; physical cases agree
  with original Ninja (identical masters) to ≤7e-13, 30 edge cases to
  ≤3.9e-12, bitwise identical across widths/lane orders, zero allocations
  in timed blocks. Gate (a): pure reduction 1.49–1.86× faster than Ninja
  `-O3 -fcx-fortran-rules` at width 1 and 1.65–2.12× batched (n5/r4 graph26:
  9.0 vs 18.6 µs); tensor numerator 2–2.6× faster than Ninja's.
- Numerator (`claude/numerator`): `TensorCoefficientPlan` scatters SymJIT O2
  heads into Ninja's tensor layout; `PhysicalTensorNumerator` prepares a
  batch allocation-free (graph26 ≈3 µs/point); shared 3-point pools for seven
  representative physical graphs; `ninjars export-bench`; all exported cases
  agree with original C++ Ninja to ~1e-12.
- Reference harness (`claude/reference`): Ninja builds o3 / o3-fortranrules
  (fair baseline; `-fcx-fortran-rules` is MadLoop's flag), replay-based C++
  driver, optimized single-diagram MadLoop outputs with generation, compile
  and per-point breakdown (graph26: MadLoop 235–240 µs/point, numerator 105,
  Ninja 26, OneLOop 105; MadLoop declares rank 5 for the pentagon).
- Masters: NinjaRS's generic `SymbolicaMasters` uses oneloopmaster's native
  evaluators for fixed-precision carriers and applies OneLOop's on-shell
  threshold as Ninja configures it (fixes a 1.8e-7 four-photon error from
  rounded light-like invariants). `OneLoopNativeMasters` (fast path) uses
  native A0/C0/D0 and oneloopmaster's native tensor families. Symbolica PR
  #89 (fast binary64 dilogarithm, 12–79 ns instead of an MPFR detour of
  32–240 µs) is pinned via integration commit e679e8c1; oneloopmaster main
  a7fc9c3 adds native B11/B111/An (0.18–0.5× OneLOop). Gate (b) status at
  a7fc9c3: graph26 1.15–1.20×, gg→HH box ≈1.0×, H→gg 0.6×, γγ→γγ box
  1.6–1.7× (D0 with massless legs 2.1×, C0(0,0,s) 1.8×) — not yet met;
  C0/D0 optimization in progress on oneloopmaster `claude/native-c0d0-perf`.
- Correctness defects found in oneloopmaster expressions (fix in progress on
  `claude/expression-fixes`): B0(p², m², 0) NaN at ~1% of points (inexact
  unit root), C0 with one massless internal line (NaN / wrong imaginary
  part), D0 four-mass real boxes with s12·s23 < 0 (wrong real part; OneLOop
  3.6 and COLLIER agree), symbolic B1/B11/B111 cut side at p²=0 with masses
  of opposite sign. The 2024 OneLOop snapshot and OneLOop 3.4 each have one
  wrong D0 point. Physical top-loop benchmark diagrams are unaffected.
- `ninjars pipeline-bench` (physical, end to end) and `ninjars case-bench`
  (bench cases) time numerator preparation, tensor numerator, pure reduction
  (replay), replayed native masters and the uninstrumented total. Gate (c)
  status: NinjaRS total 0.23–0.67× MadLoop on all seven physical graphs
  (graph26: 158 vs 235 µs per point). Report: `docs/benchmark/benchmark.typ`
  (data via `validation/benchmark/collect.py`).
- Coordination lesson: messaging agents that run inside a Workflow resumes a
  duplicate; the duplicates were stopped and their streams consolidated.

### Gates met on the final revisions; correctness sweep (2026-10-18)

- Pins: oneloopmaster main 264a6ce (continuation and degenerate-kinematics
  fixes 26aad03, compact shared definitions 89631b3, Float trif2 zero
  caad3c0, contour-residue r0 windings 264a6ce), Symbolica e679e8c1 (PR #89
  binary64 dilogarithm). Final measurements: `validation/benchmark/ninjars/`
  (pipeline-bench, case-bench), `validation/benchmark/masters_native/`
  (olm-264a6ce), `validation/benchmark/arbitrary-precision/` (summarize.py),
  report data `docs/benchmark/data/*.json`.
- Gates on the seven physical diagrams (identical inputs, width 32; every
  timing, references included, pinned to CPU 250): (a) pure reduction
  0.48–0.58× Ninja; (b) native masters 0.26–0.57× OneLOop; (c) end to end
  0.17–0.51× MadLoop (graph26: 78 vs 241 µs per point). Over all 27 bench
  cases: reduction 0.48–0.60×, masters 0.26–0.87× (width 1: at most 1.21×,
  syn-n2-r2), every lane succeeds. References re-timed on CPU 250 agree with
  the CPU 29 runs within ~1 % except OneLOop on n = 5 (5–8 % slower on CPU
  250); `validation/benchmark/cpu-crosscheck.md`.
- Report review: a claim-by-claim check of the report against data and code
  (34 agents, adversarially confirmed) found 24 issues, all fixed; notably
  MadLoop's Ninja-interface glue is now counted with its numerator (as
  NinjaRS's propagator/S-matrix preparation is), and Table 2's Δ column
  uses the identical-input comparison. Call by call on the 1,998
  master calls Ninja makes, native masters take 0.24–0.85× OneLOop's time and
  agree to ≤8e-14 (physical) / ≤1.4e-12 (synthetic; one rank-3 tensor bubble
  2.5e-9).
- NinjaRS fixes: Float S-matrices are derived exactly from the prepared
  momenta (662d906; rounded light-like invariants made the four-mass D0
  wrong at 104/116/128/... bits with a claimed 8 bits); tied bubble
  reference vectors are broken deterministically (606d94a; round-off moved
  ±1375 between cut-constructible and rational parts at 23 of 51
  precisions). Symbolica `Float`'s tracked precision overstates accuracy by
  up to ~13 bits (rounding is not tracked), so accuracy is always measured
  against a higher precision.
- Ill-conditioned synthetic lanes (Table 2 †): syn-gghhh-ggHHH-n5-r5 lane 1
  (near-singular box Gram 8e-8; Ninja 3e-4 from the 256-bit result) and
  syn-n6-r6 lane 0 (1e12–1e13 cancellation; Ninja 2.3 %). NinjaRS's
  reduction is as accurate as Ninja's with identical masters; the remaining
  excess on syn-n6-r6 is one C0 with nearly coincident roots (native 2e-13
  vs OneLOop's 3e-15 from its dedicated near-coincident dilogarithm
  difference) — open in oneloopmaster, as is the Float D0 with a tiny
  nonzero invariant (zero-external chart conditioning).
- Arbitrary precision: graph26 at 700 bits agrees with 1000 bits to 208
  digits in ~7.5 s for three points (including preparation);
  double-double 30 digits, binary64 fast engine 14 digits.

### Numerator rank n+2 (2026-10-19)

User requirement: support numerator rank n+2, one beyond original Ninja.
Keep n general in every precision; the practical range is n <= ~7.
Stability comes first. Write-up: `docs/rank-n2.md`, covering formulas,
architecture, validation, limitations and errata.

- **Branch.** `claude/rank-n2-integration`, head 8346699, not pushed. It holds:
  - S0 interfaces and baselines (96b3653, f7e8bad);
  - generic engine S2 (c066178);
  - fast engine S3 (1d5cb5f, ebdbc00);
  - native B1111 providers S4 (07395b2);
  - reference suite S5a (498432f);
  - case-check and V-tests S5b (3c6c88c);
  - main dec3d55 (1524dbd);
  - fixes: validation policy (dda8f8e), Float B1111 (bd2022b, 91deffa), fast
    accuracy (79f306a, 61079d4).

  Docs are on `claude/rank-n2-docs`.
- **Pins.** oneloopmaster fe550e3 adds `TensorIntegral::B1111`, OneLOop's
  `bub1111` with the mass-swap sign corrected: OneLOop 3.4 and 3.6 negate it
  in every swap branch. B0..B111 and the B11/B111 kernels are bitwise
  unchanged. gammaloop 474478a7 and Symbolica e679e8c1 are unchanged.
- **Recipe.** Both engines reduce rank n+2, checked exactly in F_p[i] for
  n = 1..6 by an independent reduction.
  - The box mu-expansion runs on the exact d-dimensional cut,
    `mu² = t² v² - beta`: `c6 = n0/(v²)³`, `c4 = n2/(v²)² + 3 beta c6`. Ninja's
    `mu² = t² v²` gives a wrong c4 for mu²-dependent numerators.
  - Triangles have 21 coefficients (mindeg 5, A/B average at t⁰), bubbles 35
    (mindeg 4, 22 slots), and tadpoles keep the 16-entry layout.
  - New integrated terms: box mu⁶, bubble mu²(q·e2)² and mu⁴, and B1111.
- **Generic engine.** Plans are built from `RankClass`; the 1,239 rank
  <= n+1 plan fingerprints (V0p) are unchanged.
- **Fast engine.** A separate `engine_x2.rs` is selected at construction, and
  rank <= n+1 stays bitwise unchanged (V0). It uses series-division
  subtractions and per-lane conditioned bubble references and tadpole bases
  (`bases_x2.rs`). Rank n+2 costs 2.1-2.9 times its rank n+1 twin in the
  replayed reduction (ebdbc00), and the basis conditioning added 3.0-5.8%
  instructions (79f306a).
- **Validation at 8346699.**
  - `cargo test --workspace`, release: 389 passed, 0 failed, 12 ignored with
    default features; 498, 0, 21 with all features. V0 and V0p pass in both.
  - `ninjars case-check` on the 333-lane suite (n = 1..7; pass / report-only /
    known failure, 0 fail):
    - fast: 275 / 49 / 9;
    - generic f64: 274 / 50 / 9;
    - Float(256): 284 / 40 / 9, tier A <= 1.3e-36 at n <= 6;
    - DoubleFloat: 296 / 28 / 8 plus 1 stale known failure.
- **Known limitations and open items.**
  - oneloopmaster's real-mass D0 has the wrong real part on several boxes
    (24 known-failure lanes, findings 3 and 5; also argument-order
    dependent). It is not fixed.
  - The finding-3 lane is right in DoubleFloat. A per-precision scope for
    known-failure entries needs a decision.
  - Four binary64 fast-engine tests stay ignored (`docs/rank-n2.md` 8.4), on
    these lanes:
    - the n = 7 pinch identities, whose p⁰ = 0 bubbles lose 5.8e-9..1.6e-8 on
      the rank <= n+1 path;
    - `rn2-id-pinch-n4` lane 3, at 1.4 times its tolerance;
    - `rn2-exact-mu-n7-r9-perm` lane 1 under relabelling;
    - `rn2-shifted-mixed-n7-r9-mu2` lane 0, at 1.1 times 10 times the generic
      f64 error.
  - Generic f64 misses `rn2-phys-real-n6-r7` lane 2 (4.3e-7, not traced).
  - `Float` keeps the exact B0..B111, which lose 1-4 digits per decade of
    `m²/|p²|`.
  - There is no native degree-4 bubble correction.
  - Still to do: the design's §8 A/B timing on the integrated head (last
    passed at S3 and S4) and a rank n+2 section in `docs/benchmark/benchmark.typ`.

### Rank n+2 integrated with main d53dd08 (2026-10-19)

Integration of the rank n+2 work with current main, after the final review of
8346699 (which found that oneloopmaster fe550e3 gives wrong real-mass D0 in
Float(256..1024), fixed at 09b2c74).

- **Branch.** `claude/rank-n2-integration`, not pushed: 8346699, then the
  docs (3be5183, merged ba14aea), the performance guard (21a3701, merged
  b67f5e2), main d53dd08 (merged e53c298), case-check scopes and known
  failures (abaa0c0), doc pointers (e116813) and this documentation.
- **Merge of main.** 11 conflicts: `oneloop_native.rs` keeps the branch's
  `B1111 => &mut self.b1111`; `masters.rs` module doc merged; pins take
  main's oneloopmaster 09b2c74 (`TensorIntegral::B1111` unchanged since
  fe550e3). No manifest references `references/`.
- **V0.** 09b2c74 changes rank <= n+1 bits, so the fixture was regenerated on a
  clean clone of main d53dd08 (`validation/regression/r-le-n1-d53dd08.json`,
  identical to the final review's): 69 of 237 fast width-1 lanes and 71 of 237
  generic binary64 lanes moved, median 2e-14, at most 8.8e-5 on the cancelling
  `syn-gghhh-ggHHH-n5-r5` lane 1; Float(256) by <= 7.8e-72; no status. The
  merge passes V0 bitwise and V0p unchanged.
- **Known failures.** All 24 lanes of findings 3 and 5 are fixed and removed.
  With 09b2c74 exactly those five boxes change among the suite's 4692 exact
  C0/D0. `case-check` entries can now be limited to `engines` and
  `precisions`; `ninja_unreliable` replaces original Ninja's errors where
  OneLOop's D0 was wrong (corrected `rn2-int-mixed-n6` lane 1: 1.3e-12,
  4.9e-12, 1.5e-11 instead of 0.10, 0.30, 0.24). Known inaccuracies:
  `rn2-phys-real-n6-r7` lane 2 (generic f64) and 15 generic Float(128) lanes
  (exact C0/D0 in Symbolica's precision-tracking interpreter: up to 10 bits
  more than the native code).
- **case-check, suite (333 lanes), pass / report-only / known; 0 fail, 0
  stale, exit 0 in all five runs:** fast 276/57/0, generic f64 275/58/0,
  DoubleFloat 297/36/0, Float(128) 294/36/3, Float(256) 285/48/0. Controls
  (615 lanes): 600/15/0, 599/15/1, 600/15/0, 588/15/12, 601/14/0.
- **Final review finding 1** is gone: `repro-d0-real.json` agrees with
  COLLIER in f64..Float(1024), and the full n = 7 and n = 8 lanes agree with
  the review's references to <= 6.5e-39 in Float(256) and Float(512).
- **Tests at e116813.** Release workspace suite, default features: 392 passed, 0 failed, 12 ignored; with `--all-features`: 501 passed, 0 failed, 21 ignored. V0 (`regression_bitwise`) and V0p pass; clippy (workspace, all targets, all features) and rustfmt clean.
- **Open.**
  - Two fixes in preparation, documented as limitations in `docs/rank-n2.md`
    section 9: DoubleFloat/Float(128) bubbles at small |p²| with unequal
    masses (exact B1..B111), and `--stability-test` missing cancellation
    inside one cut contribution.
  - The n = 7 lanes of the former known failures keep COLLIER references
    (tier C); `reduction-exact` with scalars regenerated at 09b2c74 would
    publish independent values.
  - Still ignored with "needs fast accuracy fix" (fast binary64 at n = 7, all within 10-100x of generic f64 on ill-conditioned lanes): `pieces_reproduce_the_rank_n2_references_binary64`, `rank_n2_integrals_equal_their_pieces_binary64` (rn2-id-pinch-n7), `relabelling_invariance_rank_n2_binary64` (rn2-exact-mu-n7-r9-perm lane 1) and `suite_fast_within_ten_times_generic_binary64` (rn2-shifted-mixed-n7-r9-mu2 lane 0, cond 7.9e9). The other ignored tests need the C++/Fortran reference builds, are child processes, generators, inventories or microbenchmarks.
  - The performance guard (21a3701) ran against dec3d55 on 8346699 and was not
    rerun after this merge; `syn-n2-r2` width 1 reduction stays at 1.061
    (code placement). No rank n+2 section in `docs/benchmark/benchmark.typ`.
### Levi-Civita tensors from gamma5 traces (2026-10-18)

- `ninjars prepare/reduce` now evaluate numerators that keep `spenso::epsilon`
  (axial traces, e.g. `H -> Z a` through a top loop) instead of only
  diagnosing them. Idenso's rules (`Tr[g5 g^mu g^nu g^rho g^sigma] = 4 eps`,
  the three-gamma expansion, `eps eps = +det g`) fix `eps^{0123} = -i` for
  `g5 = i g^0 g^1 g^2 g^3` = `diag(-1,-1,1,1)` (chiral basis, as
  spenso-hep-lib's Idenso validation already binds). The generic part is
  upstream in draft
  [FeynKit/gammaloop PR #149](https://github.com/alphal00p/gammaloop/pull/149):
  branch `idenso-epsilon-components` at
  `3147bcad041640015472d702af3d2748a84c2b88` (based on `feynkit`
  `d3b91d314`, independent of PRs #142–#144) documents the normalization and adds
  `idenso::epsilon::contract_epsilon4`, with a basis-free test
  (`Tr[g5 a b c d]` against `-i Tr[e0 e1 e2 e3 a b c d]`).
- UFO's Lorentz `Epsilon` is real: ALOHA (MG5_aMC 3.7.0 `aloha_object.py`,
  "matching FR choices") has `eps^{0123} = -1`, and with its own Weyl-basis
  matrices `Tr[g5 g^a g^b g^c g^d] = 4i Epsilon^{abcd}` holds exactly for all
  256 index choices. FeynKit lowered it to Idenso's epsilon unchanged, a factor
  `i` off on every CP-odd UFO vertex (none in the SM). Draft
  [FeynKit/gammaloop PR #150](https://github.com/alphal00p/gammaloop/pull/150)
  (`ufo-epsilon-normalization` at `1b612ac6d8958d8ee36cb2625c1782fccabe98a7`,
  based on `feynkit`) lowers it as `-i spenso::epsilon` with exactly four
  indices; its test reproduces ALOHA's trace identity through Idenso and fails
  without the factor.
- Both commits are cherry-picked onto the integration branch
  `ninjars-shared-integration-epsilon`
  (`eb314a274faa8e4ed4e21e6cf78b42c68ec77cd7`, on PR #144's head), published
  on alphal00p/gammaloop. `Cargo.toml` pins that revision, and the
  consolidated `feynkit-shared-types.patch` is regenerated from `d3b91d314`;
  a `references/gammaloop_ninja_shared` checkout used with
  `dependencies/local-references.toml` must move to it.
- NinjaRS keeps the one-loop glue: `ninjars_graph::expand_components` sums
  open Lorentz indices over components jointly for spinor chains, Levi-Civita
  tensors and vectors whose contraction left an explicit index (contraction
  leaves `eps(.., mink(4,i)) V(mink(4,i))` and `V(mink(4,i))^2` explicit when a
  closed trace meets a spin-trace projector through a tree propagator).
  `reduce` on every `examples/graphs` fixture is bitwise unchanged.
- Validation (`crates/ninjars-cli/tests/robustness.rs`): a hand-written
  `Tr[g5 p0 p2 p3 p4]` pentagon equals the explicit-matrix trace times the
  scalar pentagon; `H -> Z a` reduces in both engines, its axial part flips
  sign with the loop orientation (C parity) while the vector part does not;
  `e+ e- -> Z a` through a photon and a top loop agrees between 16
  unit-spinor lanes and a spin-trace projector. Rational terms of axial
  numerators are gamma5-scheme dependent (Shao, Zhang, Chao,
  arXiv:1106.5030) and are not computed; no reading-point prescription is
  needed for these four-dimensional numerators.

### Pruning validation harness source draft (2026-10-17)

- From `38f8569`, added a validation-only, default-off pruning flag independent
  of kernel selection and native row policy. Existing constructors and query
  widths remain supported; the additive constructor/query expose configured
  interpreted selected heads without changing raw callbacks or production CLI.
- The physical control prepares each width once for full/pruned sessions and
  checks all five result heads with a common frozen master union. It compares
  callback call/row counts separately; those counters are not complete identity
  traces. Existing original Ninja/MadLoop convention limitations remain stated.
- Source tests cover ABI output sentinels, defaults, all quotient degrees,
  inactive bubbles, rank/native policy independence, exact metadata, malformed
  parity and enabled-only timing preflight. The latter requires matching current
  runtime sources and full/pruned physical controls covering each requested
  timing width before building.
- Only source review, Python syntax parsing, targeted Rust formatting and diff
  whitespace checking were performed during the independent timing hold. All
  builds, test execution, numerical controls and pruning timings remain pending;
  the frozen measured worktree and its evidence are unchanged.

### Pruning validation harness correctness checkpoint (2026-10-17)

- Frozen source `8a33a59` passes all nine ABI tests with native kernels disabled
  and all nine enabled, plus 18 Python metadata/preflight tests, formatting and
  strict Clippy in both configurations. No source fixes were required. The prior
  87 core pruning tests remain separate and were not repeated.
- A single preserved optimized native executable/library passes full/pruned
  `native-rank` Whole and fixed32 controls at widths
  0/1/3/8/31/32/128/1024, all five selectors with pruning, and the default-off
  positional invocation. The controls compare 79,755 complex parity components:
  full/pruned differs by at most 1.54e-33, Whole/fixed32 agrees exactly, and
  all-selector differences are at most 4.00e-18. Call/row deltas remain equal at
  each width. Original Ninja and MadLoop Laurent checks also pass.
- `validation/evidence/pruned-swapped-harness.json` indexes all commands, 73
  retained record/log files, exact binary identities and all 1,975 runtime source
  hashes. Read-only copies retain the test and physical binaries, compile-only
  timing launcher, reference libraries, input protocol and full runtime-source
  archive. Source maps match before/after, and all dependency trees are clean.
- The timing launcher compiles with strict warnings and the enabled preflight
  validates the actual reports and every requested width. It was not executed:
  no timing, pruning speedup or performance acceptance is claimed. The fixed
  physical control is binary64 graph26; broader physical, precision and original
  rational-component gates remain in the overall plan.

### Symbolica pin 9cdc2949: Float products rounded once (2026-10-20)

- **Pin.** `alphal00p/symbolica` `ninjars-pin` `9cdc2949cdb153478c1eca0d40690dc0abfc616e`
  = `e679e8c1` plus [Symbolica PR #97](https://github.com/symbolica-dev/symbolica/pull/97)
  ("Round Float products once and keep factors of ±1 exact"; open). numerica's
  `Float * &Float` rounded twice (so `a*b != b*a`) and `a - b` re-rounded `b`
  through a working-precision `-1`; one-bit residues flipped exact sign tests
  in oneloopmaster's C0/D0 at 128..1024 bits. Binary64 and DoubleFloat are
  unchanged.
- **Manifests.** The root, `validation/performance/shared_provider`,
  `validation/benchmark/ill-conditioned` and `validation/benchmark/masters_native`
  `[patch.crates-io]` blocks and lockfiles (only the three `source` lines move;
  `cargo metadata --locked` passes in each root). `dependencies/patches/symbolica-batch-dual.patch`
  is regenerated as `git diff --abbrev=8 c78cffc2 9cdc2949` in a Symbolica
  clone (sha256 `c8f55f66…b348`; `--abbrev=8` fixes the width of the `index`
  lines, which otherwise depends on the clone's object count, so the bytes
  and the hash reproduce);
  `patched_revisions` now lists `9cdc2949` instead of `e679e8c1`, because
  bootstrap accepts exactly the checkouts that already contain the
  consolidated patch, and `e679e8c1` no longer does. `tools/bootstrap.py`
  applies the patch to a clean `c78cffc2` checkout and yields the tree of
  `9cdc2949` exactly, recognizes the applied patch on a second run and accepts
  a `9cdc2949` checkout unchanged.
- **Effect on V0/V0p** (fixture `r-le-n1-d53dd08.json` not re-blessed; it
  already differs on this branch, HANDOFF.md §2 "Integration pass 1"). Before
  the re-pin 43 entries differ, all in `generic_float256`; after it 87, all in
  `generic_float256`. `fast_tensor`, `generic_f64` and `mu_numerators` are
  bitwise identical before and after the re-pin. The re-pin moves 44 of the 85
  Float(256) complex values by at most 8.7e-71 of their lane's largest
  coefficient; no status changes. V0p passes before and after.

### Pruning harness on the integration pins (2026-10-20)

`codex/pruned-validation-harness` (6db5602) merged into
`claude/todo-next-integration`, checked on the integration branch with Symbolica
`9cdc2949`, oneloopmaster `09b2c74` and FeynKit `d1904bf2` (scratch evidence,
not retained in the repository).

- **Build.** `validation/performance/shared_provider` builds `--locked --release`
  against these pins with no warning and no source change.
- **Physical control** (`run_physical.py --release --maximum-integrated-batch 32`,
  interpreted selection, graph26, integrated widths 0/1/3/8/31/32): passes
  without and with `--prune-swapped-arithmetic`, with identical raw (3.8e-15),
  coefficient-oracle (2.4e-15), integrated (1.3e-16) and MadLoop (6.2e-17)
  scaled errors. The pruned run's same-artifact full/pruned parity checks 375
  complex components (75 rows over the integrated widths 0/1/3/8/31/32, 5
  heads each): largest `|pruned - full| / max(1, |full|)` is
  7.7e-34, numerator call/row deltas equal. Between the two runs, the primary
  session's 15 Laurent/CC/R1 components are bitwise identical except one
  analytically vanishing single pole (6.4e-18) that moves by one ulp
  (1.2e-16 relative, 1.5e-31 of the lane's largest coefficient).
- **Cost** (in-process `instructions:u` around the timed blocks of
  `physical_timing.cc`, a scratch copy with counters, scope shared-reduction
  with frozen masters, 4 trials x 30 rounds, two repetitions; spread <= 1.6e-4):
  the Rust reduction (with its interpreted numerator callbacks and master
  lookups) costs 4.54M/4.03M/3.97M instructions per row at widths 1/8/32
  without pruning and 2.2 %/2.5 %/2.6 % fewer with it (about 0.10M per row);
  original Ninja's blocks are unchanged (1.26M per row). No wall-clock timing
  was taken.

### oneloopmaster pin 8ec256fc, Float rescue check and V0 re-bless (2026-10-21)

- **Pin** (8a8bc55). oneloopmaster `09b2c74` → `8ec256fc26cb361b20e10f59c19beb35717ef32c`
  (main; the code of the verified candidate 3a322d0): contour box near
  vanishing Gram determinants, Denner–Dittmaier sheets for one- and two-mass
  boxes with complex masses, two-mass complements, the soft-box relabelling at
  the numerical D0 entry points; it takes Symbolica from `alphal00p/symbolica`
  `9cdc2949`, the same source as NinjaRS's `[patch.crates-io]`. Root,
  `validation/benchmark/ill-conditioned`, `validation/benchmark/masters_native`
  and `validation/performance/shared_provider` manifests/lockfiles,
  `dependencies/manifest.json`, the README pin table; `cargo metadata --locked`
  passes in all four roots and each resolves one symbolica/numerica/graphica
  (9cdc2949) and one oneloop. No API change.
- **Soft-box relabelling on the `Float` route** (8e583f2): `SymbolicaMasters`
  applies `oneloop::regular_d0_labelling` to every D0 row after the on-shell
  threshold. `tests/soft_box_labelling.rs`: pyAmpliCol's `e+ e- -> t t~ h`
  box `(m, m, 0, m)` in all 24 labellings through every provider and carrier,
  within 5.2e-16 (binary64), 1.8e-31 (DoubleFloat), 2.0e-38 (`Float(128)`,
  `Float(256)`) of the 38-digit reference and 5.2e-16 of COLLIER; without the
  relabelling `Float(128)` puts labelling 0132 73 times its largest coefficient
  away from 0123.
- **Tracked-precision check** (2a7df0d, user decision 2026-10-21: points must
  be rescued consistently when the precision increases). A `Float` master row
  whose finite part or pole keeps fewer than `max(53, bits/4)` tracked bits
  (complex: `-log2(sqrt((|re| 2^-p_re)² + (|im| 2^-p_im)²)/|z|)`) fails with
  `MasterFailure::ImpreciseResult`. Rule from oneloopmaster's adversarial
  sweeps (session-2026-10-20b `oneloop-compare`, `verify-olm-sweep2`):
  false negatives only on the 6 rows whose tracked precision is unreliable,
  false positives only at 128 bits (1059 right values with 35–52 bits, all
  right with >= 64 bits at 256); `bits/4` alone would pass 392 values between
  1e-10 and 1e-6 at 128 bits. NinjaRS's 8,487 recorded master calls all pass at
  128 and 256 bits. `SymbolicaDualMasters` checks the value components only
  (derivatives of constant coefficients are rounding residues). Cost 77 ns per
  checked value (1 % of a native `Float(128)` B11 row) and nothing when both
  parts keep the required bits.
- **Effect.** V0 and `case-check --engines generic --precision float:128|float:256`
  on 16 suite cases (48 lanes) are unchanged by the relabelling and the check
  (statuses, verdicts and decimal values). At `float:128` those 16 cases show 3
  stale known inaccuracies (`rn2-int-mixed-n4-r6` lane 2, `-n6-r8` lane 0,
  `-n7-r9` lane 1) with or without them: the re-pins fixed those lanes;
  `known-failures.json` is re-derived separately.
- **V0 re-bless** (`validation/regression/README.md`, History): once, in place,
  for (a) native B0, (b) the generic selection, (c) oneloopmaster, (d) Symbolica;
  the fast sections equal, bitwise, those of a fixture blessed on main
  0d0d7e7 with only the two re-pins.
