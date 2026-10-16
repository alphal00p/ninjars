# Physical H → gg preparation

`ninjars_graph::prepare_top_higgs_gluons` generates the two oriented top-quark triangles from FeynKit's embedded Standard Model. Vertex selection uses the PDG particle content (ttH and ttg), one loop, exactly three vertices, and no bridges or numerator grouping. Model masses and couplings remain symbolic. The numerator starts with four-dimensional Lorentz and Dirac representations; no dimensionally integrated expression is specialized afterwards.

FeynKit supplies the graph rules, external wavefunctions, momentum routing, ordered physical propagators and exact quadratic decomposition. `Amplitude::legs` supplies the external color-port labels. The gluon projector is `ColorSum::averaged(true)`, inserted once as δab/8: the returned numerator is the coefficient of δab, without a helicity sum or spin average. Idenso's `ColorSimplifySettings::with_cof_dimension_invariants` supplies the fundamental SU(3) normalization, including Tr(Ta Tb)=δab/2. The general graph preparation API keeps arbitrary color invariants symbolic.

The graph's annotated overall weight is evaluated using FeynKit's `expressions::evaluate_overall_factor`, just as FeynKit's amplitude constructor does. The raw diagram retains its provenance annotations. Both oriented top triangles have overall factor −1, which remains separate in the preparation result and must enter the integrand exactly once. The CLI includes this factor in the numerator callback.

These operations were checked against the public APIs, their implementations and tests (`feynkit-amplitude/src/color.rs`, `feynkit-graph/src/expressions.rs`, `idenso/src/color/mod.rs`, and `idenso/src/tensor/simplification.rs`), and the existing FeynKit `installed_feyncalc_higgs_gluons.py` fixture. No tensor, polynomial or color manipulation is implemented in NinjaRS.

The first Rust tests validate two orientations, three equal MT² denominators, an entirely scalar component polynomial after physical momentum/polarization binding, and vanishing of the chirality trace as MT→0 with the Yukawa held fixed. They reject incompatible dimensions and models. They do not yet establish a physical integrated result or a Ward identity. The existing FeynKit integrated fixture includes dimension-dependent rational terms and is not an oracle for the requested four-dimensional, no-R2 convention.

Generate reproducible graph/model sources and prepared symbolic numerators with:

```sh
cargo run -p ninjars-graph --example prepare_hgg -- target/physical/hgg
```

The DOT files preserve the generated graph rules and projector; the accompanying numerator text records the contracted, SU(3)-normalized preparation result. The committed `examples/graphs/physical/hgg/top-triangle-{0,1}.json` fixtures supply exact Standard Model inputs, `MT=ymt=173`, `MH=125`, and `WT=WH=0`. Each contains (++), (+−), and double-longitudinal polarization samples at rest-frame decay kinematics. `external_polarizations` binds exact `[E,x,y,z]` expressions by physical external-leg index. Its keys are translated to the actual FeynKit projector carriers using Symbolica pattern matching; kinematic component substitution uses the existing shared `FourMomentum<Atom>` and `Kinematics` APIs. The evaluator's common parameter list includes these components, so varying polarization values remain batched.

```sh
ninjars prepare examples/graphs/physical/hgg/top-triangle-0.json
```

`model: "sm"` selects FeynKit's embedded Standard Model. The explicit fixture option `color_dimension_invariants` applies Idenso's existing color simplification policy, and the public `PreparedIntegrand::with_color_dimension_invariants` method supplies the same behavior to Rust callers. Model parameter overrides stay exact `Atom` expressions through `Model::scalar_bindings`; the physical fixtures independently derive the strong and Yukawa couplings from model inputs instead of copying the MadLoop coupling values.

# MadLoop reference generation

`validation/madloop/prepare_hgg.py` uses the supplied MadGraph 3.7.0 code. Its discovery hook only observes the stock filter, before automatic graph identification. The stock per-graph filter then exports each orientation separately and asserts a single graph with multiplier one. The current model/process gives pre-grouping IDs 3 and 4.

```sh
python3 validation/madloop/prepare_hgg.py --output target/reference/madloop-hgg
```

Use the repository Nix environment for Python/six and the Fortran compiler. The script refuses to overwrite existing graph exports. It selects `loop_optimized_output=False`, disables subprocess grouping, uses internal copied CutTools dependencies, sets `MLReductionLib=1`, `CTModeRun=CTModeInit=1`, `CTLoopLibrary=2`, disables phase-space improvement and QP integrand switching, and sets the top/Higgs widths to zero. Generated wavefunction calls exclude counterterms; loop calls and linear color-amplitude coefficients are recorded separately in `graphs.json`.

`validation/madloop/probe_hgg.py` builds `raw_hgg.f` against unchanged generated HELAS numerators and `CT_interface.f`. The only helper extracted from the matrix-element source is the stock helicity-selection function; the matrix-element driver, color square, Born interference, and R2 amplitude are never invoked. The harness validates the recorded loop signature, couplings, unit graph multiplier, and linear color coefficient before compiling. It calls `ML5_0_LOOPNUMHEL(-q,H)`, `ML5_0_LOOPNUM(q)`, and the unsquared `ML5_0_LOOP_3_3` routine directly. Raw results agree exactly between the two numerator entry points.

```sh
python3 validation/madloop/probe_hgg.py \
  --exports target/reference/madloop-hgg \
  --output target/reference/madloop-hgg-probe
cargo test -p ninjars-cli --all-features --test physical -- --test-threads=1 --nocapture
```

The reference samples form a fifteen-point unisolvent complex affine grid for total degree two in four loop components. The Rust parity test independently checks the prepared polynomial's rank is two, then compares all ninety points across two orientations and three polarization states. It passes with maximum scaled error `8.651927310692645e-16`. The reference evidence is `validation/evidence/physical-hgg-madloop.json`, including input parameters, external polarization components, source hashes, graph metadata, raw numerators, stable flags, and complex Laurent coefficients. The test also verifies complete binding to a scalar polynomial, exact masses, denominator provenance, graph sign, and consistent polarization-leg sets across a batch.

The matched conventions are:

| Quantity | FeynKit/NinjaRS relative to MadLoop |
|---|---|
| Orientation 0 | `q_FK = -q_CT` |
| Orientation 1 | `q_FK = q_CT` |
| Ordered propagator slots | CutTools `[2,0,1]` |
| Linear color and graph sign | `-1/2`, from loop amplitude coefficient `-1` and `Tr(Ta Tb)=delta_ab/2` |
| Generated integrated normalization | MadLoop multiplies by `1/(16*pi^2)` |
| Raw Laurent conversion to NinjaRS convention | Multiply by `-8*pi^2` |

The probe runs fixed CutTools mode 1 (double) and fixed mode 4 (quadruple), always with the OneLOop backend and without adaptive precision switching. Mode 4 promotes the same binary64 external wavefunctions and couplings exactly; it is a control of reduction arithmetic, **not** a high-precision physical-input control. Mode 1 reports stability despite roughly `1e-8` absolute noise in the helicity-zero finite result and up to `1e-7` noise in longitudinal poles after normalization. Mode 4 reduces those residuals to approximately `1e-27` and `1e-25`; it is the explicit reference for the unchanged scaled `1e-9` integration gate.

The on-shell triangle exposed a bubble-cut Weyl-chart singularity for a lightlike external separation with equal nonzero internal masses. The shared-basis fix now evaluates all samples successfully. Reproduce the integrated matrix with `python3 validation/madloop/compare_hgg.py`; `validation/evidence/physical-hgg-cli.json` records both orientations, f64/DoubleFloat/Float256, and batch sizes one/two, including incomplete tails.

Of the 36 result rows, 32 pass the unchanged scaled `1e-9` gate. All physical-helicity rows and all higher-precision longitudinal rows pass. Four f64 double-longitudinal rows miss only the expected-zero single-pole coefficient: `9.313225746154785e-10 + 5.205353009980522e-10 i`, with magnitude `1.066920211437358e-9`. Their finite coefficients agree to `2.1e-16` scaled error. The baseline remains explicitly failed while cancellation accuracy is investigated; the tolerance is unchanged. The double-longitudinal samples are independent numerator-contraction diagnostics; the no-R2 integrated contribution is nonzero, and no complete Ward-identity claim is made. Physical performance acceptance and automatic generic helicity preparation also remain outstanding.

FeynKit PR [#144](https://github.com/alphal00p/gammaloop/pull/144) makes explicit one-thread generation stay on the calling thread, including numerator/color preparation. It is stacked on shared-type PR #143; the full generator unit suite passes (104 tests), including the caller-thread and two/four-worker regressions, with strict crate Clippy and formatting checks.
