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

The historical baseline passes 32 of 36 rows at the scaled `1e-9` gate; four f64 double-longitudinal rows miss only the expected-zero single pole, at `1.066920211437358e-9`. That failed record is retained. The later combined coefficient-alias repair and dependency update passes **all 36 rows**, with maximum error `5.062690174549114e-10`, recorded in `validation/evidence/physical-hgg-cli-coefficient-aliases.json`. The tolerance is unchanged; the combined rerun does not isolate which change reduced cancellation. The double-longitudinal samples are independent numerator-contraction diagnostics; the no-R2 integrated contribution is nonzero. With `--r2 fdh` it vanishes (see [R2 rational terms in FDH](#r2-rational-terms-in-fdh)). Physical performance acceptance and automatic generic helicity preparation remain outstanding.

FeynKit PR [#144](https://github.com/alphal00p/gammaloop/pull/144) makes explicit one-thread generation stay on the calling thread, including numerator/color preparation. It is stacked on shared-type PR #143; the full generator unit suite passes (104 tests), including the caller-thread and two/four-worker regressions, with strict crate Clippy and formatting checks.

# R2 rational terms in FDH

`--r2 off|fdh` (all CLI commands that build a physical numerator) and
`ninjars_graph::R2Scheme` select how the contracted numerator's scalar
products of the loop momentum `K` are bound to components
(`PreparedIntegrand::dimensional_kinematics`):

| `--r2` | `K.K` | `K.P`, `K.eps` | numerator | reduction result |
|---|---|---|---|---|
| `off` (default) | `q0^2-q1^2-q2^2-q3^2` | four-dimensional | `N(q)`, no `mu^2` | cut-constructible + R1 |
| `fdh` | `q0^2-q1^2-q2^2-q3^2-mu2` | four-dimensional | `N(q, mu^2)` | cut-constructible + R1 + R2 (FDH) |

In the four-dimensional-helicity scheme the internal Lorentz/Dirac algebra is
`d_s = 4` dimensional while the loop momentum `Kbar` is d-dimensional,
`Kbar^2 = q^2 - mu^2` and `Kbar.p = q.p` for four-dimensional external momenta
and polarizations. FeynKit's rules and Idenso's `AlgebraSettings::hep()`
contract all indices with `d_s = 4` identities on abstract vectors and return
a polynomial in scalar products, so binding `K.K -> q.q - mu^2` yields exactly
the FDH numerator, whose explicit `mu^2` terms give all of R2 (the
d-dimensional denominators give R1). `mu^2` has rank weight two, so the
rank is unchanged. `--r2 off` results are bit-for-bit those of earlier releases.

**Assumption check.** Idenso evaluates closed traces without `gamma5` by the
dimension-generic trace recursion (`TRACE_GAMMA_RECURSION`) with `Tr 1 = 4`,
and adjacent contractions by the Clifford relation with `g^mu_mu = 4`; neither
expands `K` into components. Its four-dimensional-only rules are the
`gamma5` trace recursion with Levi-Civita tensors (and the epsilon-pair
determinant pass), the optional three-gamma Chisholm expansion (off in
`hep()`), and the "Chisholm" contraction of a repeated Lorentz index around a
Dirac chain, whose odd/even reversal formulas are four-dimensional for five or
more interior gammas. At one loop the latter needs an internal vector attached
twice to an open fermion line. A metric between external indices in the
projector (a polarization sum) is a four-dimensional regular metric that the
scalar-product binding cannot distinguish from internal ones. `--r2 fdh`
therefore rejects (`PreparedIntegrand::check_fdh`, `PreparationError::UnsupportedFdh`)
numerators containing `gamma5` or chiral projectors, Levi-Civita tensors,
external spinors, or Minkowski metrics in the projector. The H→gg,
gg→HH(H), four-photon and gluon self-energy graphs contain none of these:
their Dirac structures are closed top-quark traces whose Lorentz indices are
contracted with external polarization vectors.

**HV/CDR are not implemented.** With internal gluons, HV/CDR also contract
`(d_s - 4)`-dimensional internal indices (`g^mu_mu = 4 - 2 eps`,
`gamma^mu a-slash gamma_mu = (2 - d) a-slash`), adding `N0 + eps N1 + eps^2 N2`
pieces that multiply UV poles and are not encoded in `mu^2`; these are the
`lambda_HV` parts of the R2 vertices (MadGraph's `loop_sm` sets `lhv = 1`).
FDH results convert to HV/CDR with the universal scheme shifts after
renormalization, or the `eps N1` pieces can be reduced separately. Closed
fermion, scalar or ghost loops with only external bosons have no such terms:
their R2 is scheme independent (`R2_GGHt`, `R2_GGHHt`, `R2_GGt`, γγ→γγ).

**Fast path.** `ninjars::fast::MuTensorNumerator` stores
`N = sum_k mu^(2k) T_k(q)` with `T_k` a dense rank-`(R-2k)` tensor in Ninja's
layout and reuses the truncated-jet Horner sweep block by block: `t^-R
mu^(2k) T_k(q) = mu^(2k) u^(2k) J_k(u)` shifts block `k` by `2k` Laurent
orders and `k` powers of `mu^2` in Ninja's `t3`/`t2` layouts, and contributes
`(v0^2)^k` times its own leading coefficients in the box `mu`-expansion.
`TensorCoefficientPlan::with_mu_squared` lays out the coefficient heads (it
is identical to `with_rank` without `mu^2` terms), `reduce --engine fast`,
`pipeline-bench` and `case-bench` dispatch to it only for numerators with
`mu^2` terms, and the `mu^2`-free path is unchanged.

**Validation** (`crates/ninjars-cli/tests/r2.rs`, `crates/ninjars-graph/tests/r2.rs`):

- The FDH binding equals the four-dimensional binding after `K.K -> K.K - mu^2`
  for both H→gg orientations, all six four-photon boxes and the four gluon
  self-energy graphs, with unchanged rank.
- H→gg: `--r2 fdh` changes only the rational part. The sum of both
  orientations equals MadGraph5 `loop_sm`'s `V_GGH` R2 vertex,
  `R2_GGHt = 4 (-i yt/sqrt 2)(1/2) G^2/(8 pi^2) MT` with color `Tr(Ta Tb)`
  and Lorentz `g^{mu nu}`, to `8e-16` relative: in NinjaRS normalization
  `16 pi^2 M = 16 pi^2 (-i) R2_GGHt (1/2) eps1bar.eps2bar` (MadLoop amplitudes
  are `NinjaRS/(16 pi^2)` by the matched conversion above; ALOHA amplitudes
  carry `-i`), i.e. `+360.490...` for (++) at the committed parameters. The
  double-longitudinal sample, `1.41e6` without R2, vanishes (`7e-10`) with it.
  `pipeline-bench` reproduces the same R2 per orientation from raw momenta and
  FeynKit wavefunctions.
- SU(3) gluon self-energy (`examples/graphs/physical/gluon-self-energy`:
  gluon loop with symmetry factor 1/2, ghost loop with sign -1, massless down
  and massive top loops): with one or both polarizations replaced by `p`,
  `p_mu Pi^{mu nu}` vanishes to `1e-12` relative with `--r2 fdh`, separately
  for the gauge part and each quark, while `--r2 off` leaves up to `2e6`.
- Four photons through a top loop: replacing one polarization by its momentum,
  the sum of the six boxes vanishes (`1e-10` against boxes of `5e4`) with
  `--r2 fdh` and not without (`37`).
- Fast and generic engines agree to the 1e-9 criterion; generic f64,
  DoubleFloat and Float(200) agree to `1e-9` and `1e-20` of the largest value.

**Cost on the fast path** (`pipeline-bench`, one core, widths 1/32/128,
[`validation/benchmark/ninjars/r2-fdh-pipeline.json`](../validation/benchmark/ninjars/r2-fdh-pipeline.json)).
`--r2 off` reproduces the main-branch pipeline bit for bit (identical
original-Ninja comparisons on all seven graphs; timings within noise). With
`--r2 fdh` the triangles gain one `mu^2` coefficient (16 instead of 15) and the
boxes and the pentagon two blocks (86 instead of 70); every graph stays on the
tensor path:

| graph | total fdh/off (w=1, 32, 128) | tensor numerator fdh/off (w=1, 32, 128) |
|---|---|---|
| H→gg triangle | 1.08, 1.02, 1.03 | 1.53, 1.21, 1.20 |
| γγ→γγ box | 1.07, 1.04, 1.06 | 1.18, 1.08, 1.08 |
| gg→HH triangle | 1.11, 1.02, 1.02 | 1.61, 1.18, 1.18 |
| gg→HH box | 1.07, 1.04, 1.06 | 1.17, 1.11, 1.09 |
| gg→HHH triangle | 1.10, 1.04, 1.03 | 1.60, 1.20, 1.18 |
| gg→HHH box | 1.07, 1.04, 1.05 | 1.19, 1.12, 1.15 |
| gg→HHH pentagon | 1.03, 1.01, 1.00 | 1.22, 1.16, 1.17 |

The pentagon's R2 vanishes (rational parts unchanged to `1e-18`), the gg→HH
box's is the same `0.976` at all three points.

# Larger physical graphs

The same convention and independent matching workflow now cover six top-quark
four-photon boxes, eight gg→HH graphs, and all 50 gg→HHH graphs. Reproduction
guides and retained diagnostic limitations are in
[`validation/madloop/four_photon`](../validation/madloop/four_photon/README.md),
[`validation/madloop/gg_hh`](../validation/madloop/gg_hh/README.md), and
[`validation/madloop/gg_hhh`](../validation/madloop/gg_hhh/INTEGRATED.md).

The gg→HHH suite has eight triangles, 18 boxes and 24 pentagons. All 28,000
independent raw numerator checks and 2,400 integrated rows pass. The integrated
matrix covers four helicities plus generic/Ward diagnostics, all three scalar
precisions, and both mu-extraction modes. Independent Symbolica differentiation
checks each pole; the 50-graph sum cancels exactly in all eight states. Analytic
pole controls demonstrate additional accuracy in DoubleFloat and Float256.
MadLoop finite-part inputs remain binary64 even in its fixed quadruple mode.

The [shared physical-provider control](../validation/performance/shared_provider/PHYSICAL.md)
starts from raw momenta, helicities and model inputs at three distinct physical
points. It includes FeynKit wavefunctions, dependent parameters, coefficient
evaluation and routed propagators in each complete evaluation. Both reducers
then use the same numerator provider. This validation adapter is separate from
the CLI's prebound-polarization fixtures; reusable numerical preparation and
controlled performance measurements remain in development.
