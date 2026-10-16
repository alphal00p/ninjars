This Rust probe makes the numerical input boundary for a future complete physical
benchmark explicit. It prepares one top-quark four-photon box through FeynKit,
then computes its wavefunctions, routed propagators and polynomial coefficients
from numerical momenta, helicities, top mass and inverse electromagnetic coupling.
It does not perform reduction or report timings.

Use one locked build for every linked dependency:

```sh
cargo build --locked -p ninjars-graph --features numerator-jit \
  --message-format=json > target/physical-input-artifacts.jsonl
python3 validation/performance/physical_inputs/run.py \
  --cargo-artifacts target/physical-input-artifacts.jsonl \
  --output-dir target/performance/physical-input-probe
```

The output directory must be empty. Compiler identity, artifact/source hashes,
dependency pins and logs accompany the numerical result. The Python runner only
builds and records the validation program; preparation and evaluation are Rust.

The graph supplies the physical-leg mapping, momentum routing and epsilon versus
epsilonbar carrier choice. The probe calls the shared `FourMomentum::wavefunction`
API and checks its phases against all three independently retained MadLoop HELAS
fixture helicities. It uses existing `Model::scalar_bindings` with free symbols
for the two changing model inputs; analytic coupling expressions remain inside
the coefficient evaluator. No per-point coupling or polarization values are
cached. Symbolica generates the routed shift and squared-mass evaluator, while
`PolynomialCoefficientPlan` supplies the same multi-output SymJIT O2 mode used
by the CLI. No new algebra, physics type or routing solver is introduced.

Three distinct on-shell momentum configurations and model input rows exercise
changing phase space. The batches include empty, single, standard widths through
1024 and incomplete tails, with reordered samples and three helicities. Every
compiled coefficient is compared with Symbolica's interpreter; selected complete
polynomials are reconstructed and compared with the original prepared expression.
All numerical inputs and compiled coefficients in this probe are binary64.

The probe also promotes those same inputs exactly to Float256 and evaluates all
nine distinct momentum/helicity rows. This additional oracle currently makes
the baseline report **fail**: a coefficient that should vanish evaluates to
`1.7234597589860246e-9` in both unfactored f64 providers, above the unchanged
scaled `1e-9` gate. Provider-to-provider parity and full polynomial reconstruction
pass, so that agreement alone would have missed the coefficient cancellation.
The runner writes `result.json` and then exits with status one when any required
gate fails; the failure is retained in `validation/evidence/physical-numerical-inputs.json`.

A separate preparation experiment applies Symbolica's existing `factor()` to
each coefficient and verifies every exact identity. It reconstructs the shared
polynomial through `PolynomialCoefficientPlan::factor_coefficients` (the
audited `map_coeff`/`flatten(false)` composition) and uses the existing evaluators.
The worst Float256 discrepancy falls to `7.45e-12` for the interpreter and
`5.04e-12` for SymJIT O2. This candidate passes, but the report deliberately keeps
the unfactored failure. Factorization changes floating-point evaluation order;
these controls do not establish a general conditioning improvement. The initial
provider-only parity record is retained separately as historical evidence.
The public optional-provider rerun is recorded in
`validation/evidence/physical-numerical-inputs-factored-provider.json`; it preserves
the same baseline failure and passing factored candidate.

This establishes correctness of preparation components, not a full physical
performance result. Integration with both reducers, a large pentagon workload,
retained allocation-controlled buffers, independent reference checks at the new
momentum points and controlled timing remain outstanding. The existing CLI
fixture benchmark still begins with prebound polarization components; this probe
does not silently change that command's measurement boundary.
