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
