`PhysicalInputPlan` prepares one common scalar/vector graph from numerical physical momenta, helicities and free model parameters. It belongs to `ninjars-graph` and has no reduction-engine dependency. The caller supplies a `PreparedIntegrand` with its intended projector, five loop-variable symbols, an ordered list of free model symbols and exact fixed bindings. The plan includes the prepared overall factor once and preserves propagator edge order.

The plan uses FeynKit's physical external-leg metadata to map positive-energy momenta in physical index order to the graph's independent momentum carriers. It identifies the actual `epsilon` or `epsilonbar` carrier already present in the contracted numerator. Scalar and vector states use FeynKit `FourMomentum::wavefunction`; external fermions are explicitly unsupported in this first stage. No local wavefunction, routing or polynomial formulas are introduced.

Free and fixed model keys must name a parameter or coupling present in the model.
Unlike the broader `Model::scalar_bindings` API, this physical-input boundary
rejects additional scalar keys, catching unused input-name typos. Free names must
also be distinct and disjoint from fixed overrides.

`Model::scalar_bindings` builds exact expressions for dependent parameters/couplings. Free inputs replace their model symbols by different evaluator symbols to preserve dependency resolution. Inputs pass directly to generic evaluators: they do not pass through `ComplexValue`, parameter-card f64 storage or a binary64 conversion. Parameters not overridden retain the precision supplied by the model. Optional coefficient factoring delegates to `PolynomialCoefficientPlan::factor_coefficients` and is a setup policy, not a guarantee of better conditioning.

`compile(prototype, relative_tolerance)` supports f64, DoubleFloat and Symbolica Float. The caller chooses the tolerance for represented physical-input conservation and mass shells. `evaluate_batch` accepts shared `FourMomentum<T>` values, shared `Helicity` values, complex model inputs and real positive squared scales. It evaluates one batch of propagator, mass, S-matrix and external-mass heads, constructs wavefunctions at working precision, and evaluates one batch of numerator coefficient heads. Shape/domain errors leave the previously committed output unchanged. The returned buffers use the existing shared momentum and complex types; their ordering is documented by the plan's metadata.

Dual preparation is deliberately not accepted by the current ordinary-scalar capability bound. Symbolica's `SingleFloat::is_finite` and equality on Dual values inspect the primal, so accepting them through blanket bounds would miss derivative failures. A subsequent Dual adapter must use shared evaluator Dualization, validate the full derivative structure and every component, and test branch/chart derivatives before advertising support.

The remaining phase-space arithmetic batching gap is explicit: FeynKit's public `FourMomentum<T>::wavefunction` is scalar, so this prototype calls it once per external leg and lane. The coefficient and propagator evaluators are batched. This stage makes no SIMD or performance-acceptance claim. A focused owning-crate extension should group active wavefunction branches and expose reusable batch buffers; the scalar implementation remains its correctness control.

The algebra/API audit covered `feynkit-kinematics/src/wavefunction.rs` and its tests (shared helicity phases, negative-z and longitudinal branches), `feynkit-model/src/scalar_bindings.rs` and exact-override regressions, the existing `PreparedIntegrand::{component_kinematics,shifts,squared_masses}` composition and FeynKit `QuadraticMomentum`/routing APIs, plus Symbolica polynomial/evaluator APIs already audited in `docs/algebra-audit.md` and `docs/numerator-coefficients.md`. The prototype performs only composition, numerical validation and buffer ordering outside those shared operations.

CLI integration follows as a separate stage. Schema-version-1 fixtures remain unchanged. A distinct raw-physical input branch must retain exact momentum/model strings and helicities until numerical precision dispatch; it cannot eagerly calculate binary64 polarizations during `prepare`. Common graph contraction and evaluator construction happen once, while each numerical batch performs wavefunctions, dependent coefficients and propagators. The first graph-only stage does not modify CLI behavior.

The proposed separate CLI schema is version 2 with `input_kind: "physical"`:

```json
{
  "schema_version": 2,
  "input_kind": "physical",
  "graph": "top-triangle-0.dot",
  "model": "sm",
  "color_dimension_invariants": true,
  "fixed_parameters": {"UFO::WT": "0", "UFO::WH": "0"},
  "free_parameters": ["UFO::MT", "UFO::ymt", "UFO::MH"],
  "samples": [{
    "momenta": {
      "0": ["125", "0", "0", "0"],
      "1": ["125/2", "0", "0", "125/2"],
      "2": ["125/2", "0", "0", "-125/2"]
    },
    "helicities": {"0": 0, "1": 1, "2": 1},
    "model_parameters": {"UFO::MT": "173", "UFO::ymt": "173", "UFO::MH": "125"},
    "scale_squared": "10000"
  }]
}
```

This is a proposal, not an implemented parser. All momenta in this branch are the
physical positive-energy momenta; incoming/outgoing signs and dense leg indices
come from the graph. `free_parameters` fixes evaluator ordering, and every sample
must bind exactly those names. Fixed and free names are disjoint. Every scalar
string is parsed by Symbolica and evaluated only after precision dispatch. The
example leaves other model parameters at their documented defaults; precision
studies must override each relevant input with the intended exact value.

The preparation result needs an explicit physical variant containing the shared
plan plus exact raw rows. The existing version-1 prepared result and legacy
runtime path remain intact. `prepare` reports physical leg metadata, propagator
order, coefficient-template metadata and exact inputs without numerically
constructing polarizations. `reduce` compiles the physical plan and the common
coefficient-template numerator once at the requested precision, prepares each
batch and forwards the resulting shared momenta, masses, coefficients and scales
to the existing reducer. `bench` must identify physical preparation as part of its
measured scope instead of reporting a prebound-coefficient result as complete
physical evaluation. Reference export performs the same preparation at its
explicit binary64 boundary and records that conversion. The first parser should
reject explicit polarization and S-matrix overrides in the physical branch until
their interaction with physical validation is designed and tested.

The raw plan also exposes `compile_symjit_o2(relative_tolerance)` under the
optional `numerator-jit` feature. This f64-only constructor reuses the existing
`SymJitCoefficientEvaluator` behind a private coefficient-executor variant. The
ordinary generic `compile` continues to use `CoefficientEvaluator<T>`. Packing
physical inputs, model propagation, wavefunction calls, validations and
transactional commit share one implementation; only coefficient execution
changes. The executor allocates once at setup and dispatches once per batch. No
numerical carrier is converted. Both executors pass the changing physical-row
controls, independent pentagon raw-numerator grid and unchanged-output failure
checks; this is correctness evidence, not a throughput claim. CLI integration is
still a separate stage.
