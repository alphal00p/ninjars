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

The DOT files preserve the generated graph rules and projector; the accompanying numerator text records the contracted, SU(3)-normalized preparation result. External polarizations and exact model bindings still need to be supplied before reduction. Top pole mass `MT` and Yukawa mass `ymt` are independent parameters; physical validation will bind them consistently and set `WT=WH=0`.

# MadLoop reference generation

`validation/madloop/prepare_hgg.py` uses the supplied MadGraph 3.7.0 code. Its discovery hook only observes the stock filter, before automatic graph identification. The stock per-graph filter then exports each orientation separately and asserts a single graph with multiplier one. The current model/process gives pre-grouping IDs 3 and 4.

```sh
python3 validation/madloop/prepare_hgg.py --output target/reference/madloop-hgg
```

Use the repository Nix environment for Python/six and the Fortran compiler. The script refuses to overwrite existing graph exports. It selects `loop_optimized_output=False`, disables subprocess grouping, uses internal copied CutTools dependencies, sets `MLReductionLib=1`, `CTModeRun=CTModeInit=1`, `CTLoopLibrary=2`, disables phase-space improvement and QP integrand switching, and sets the top/Higgs widths to zero. Generated wavefunction calls exclude counterterms; loop calls and linear color-amplitude coefficients are recorded separately in `graphs.json`.

Both isolated exports have been generated successfully. A raw complex numerator/Laurent-amplitude harness, linear color-basis projection, matching of routing/helicity/normalization, and independent Ward checks remain outstanding. Standard generated MadLoop drivers still include R2 terms and must not be used as the requested no-R2/UV comparison. No physical correctness or performance acceptance claim follows from generation alone.

FeynKit PR [#144](https://github.com/alphal00p/gammaloop/pull/144) makes explicit one-thread generation stay on the calling thread, including numerator/color preparation. It is stacked on shared-type PR #143; the full generator unit suite passes (104 tests), including the caller-thread and two/four-worker regressions, with strict crate Clippy and formatting checks.
