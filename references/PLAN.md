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
