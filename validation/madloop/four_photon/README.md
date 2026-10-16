These are validation tools for the six oriented top-quark boxes in `gamma gamma -> gamma gamma`. Numerator generation in NinjaRS remains Rust/FeynKit/Idenso. The reference scripts call isolated stock MadLoop graph routines before squaring or interference, with multiplier one, no diagram grouping, no R2/UV, zero widths, and fixed CutTools modes 1 and 4. The copied loop model only enables the existing QED vertices in loop generation by changing its QED perturbative-expansion metadata. The original model is hashed before and after and must remain unchanged.

The reference uses one subthreshold phase-space point and three helicities (`pppp`, `ppmm`, `mpmp`). A fourth case binds four generic complex polarization vectors directly in HELAS; its integrated result is diagnostic. Each bound numerator has 70 independent affine complex loop-momentum samples, unisolvent for total degree at most four in four components. This does not establish polynomial identity for arbitrary polarizations. The reference wavefunctions and couplings are binary64 inputs, including when promoted to quadruple arithmetic.

To regenerate the reference with the pinned development environment:

```sh
python3 validation/madloop/four_photon/prepare_four_photons.py \
  --output-dir target/reference/madloop-four-photon-new
for graph in 181 182 183 184 185 186; do
  python3 validation/madloop/four_photon/probe_four_photons.py \
    --graph "$graph" --output-dir target/reference/madloop-four-photon-new
done
python3 validation/madloop/four_photon/collect.py \
  --exports-dir target/reference/madloop-four-photon-new \
  --probes-dir target/reference/madloop-four-photon-new \
  --output target/reference/madloop-four-photon-new/combined.json
```

Graph IDs are checked against the generated manifest; if the pinned MadLoop generation changes, inspect that manifest before substituting IDs. To reuse existing immutable exports while keeping new probe outputs separate, pass `--exports-dir` and a fresh `--output-dir` to the probe. Its assertions check generated signatures, color metadata, graph multiplicity, absent counterterms, lazy backend selection, raw/wrapper equality, finite values, and all 70 samples. Mode 4 is an explicitly selected backend precision, not a rescue switch.

`match.rs` independently regenerates the six FeynKit graphs. It identifies each directed physical-leg cycle, constructs MadLoop's denominator family, and requests a verified `IntegralFamily::find_mapping`. It verifies every mapped denominator exactly. The returned momentum image is resolved with `Kinematics::linear_terms` and shared `FourMomentum<Atom>` operations before `component_numerator` binds momenta and polarization carriers. No custom routing solver or numerator algebra is used. The matcher also checks the bound polynomial degree with Symbolica and compares every complex output at the fixed scaled `1e-9` criterion.

Compile the matcher using `match.py` and explicit rlibs from one locked workspace build. This keeps the validation probe out of the production Cargo packages. For example, after `cargo test -p ninjars-graph --no-run --locked`, identify its actual Cargo artifact directory and supply the matching `ninjars_graph`, `symbolica`, `feynkit_graph`, and `serde_json` rlibs:

```sh
python3 validation/madloop/four_photon/match.py \
  --reference validation/evidence/four-photon-madloop.json \
  --output-dir target/physical/four-photon/raw-repeat \
  --graph-rlib /path/to/deps/libninjars_graph-HASH.rlib \
  --symbolica-rlib /path/to/deps/libsymbolica-HASH.rlib \
  --feynkit-graph-rlib /path/to/deps/libfeynkit_graph-HASH.rlib \
  --serde-json-rlib /path/to/deps/libserde_json-HASH.rlib
```

Rust rejects mismatched crate identities. The retained result includes the exact successful compiler/run commands, compiler version, binary and library hashes, reference/source/fixture hashes, and relevant dependency heads. Repeat using `four-photon-generic-madloop.json` for the fourth polarization assignment. Compilation is a reproducibility step; these runs make no timing claim.

The normalization is fixed independently of numerical fitting. The FeynKit regression gives `-12*GC_2^4*MT^4` at zero loop/external momenta with temporal polarization vectors: Dirac trace four, internal color trace three, and one fermion-loop minus sign. MadLoop's raw routine omits its separately recorded `(-1)*Nc = -3` linear color-amplitude factor. Thus the raw comparison uses **−3**. Its stock `CT_interface` multiplies integrated coefficients by `1/(16*pi^2)`, giving **−48π²** when comparing returned linear Laurent rows in NinjaRS normalization. This raw harness does not perform that integrated comparison.

`compare.py` separately evaluates all six unsquared graphs using the prepared
fixtures, both μ-extraction modes, and f64/DoubleFloat/Float256. Its default batch
width three leaves a one-lane tail. Each of 108 physical helicity rows is compared
coefficient by coefficient with fixed MadLoop modes one and four; mode four is the
primary gate. The 36 generic-polarization rows remain explicitly diagnostic.
The criterion is `abs(actual-reference)/max(1,abs(reference)) <= 1e-9`, separately
for finite, single-pole and double-pole coefficients. The finite split must also
sum to the total. Raw matching, integrated matching and arithmetic-precision
convergence remain separate evidence.

Use an immutable copy of a CLI built from the recorded dependency pins:

```sh
python3 validation/madloop/four_photon/compare.py \
  --ninjars target/physical/four-photon/final-binary/ninjars \
  --output-dir target/physical/four-photon/integrated-final
```

`uv_poles.rs` provides an independent analytic pole control. Symbolica computes
the four-dimensional Minkowski Laplacian twice on each prepared numerator,
including its overall factor exactly once. The pole is `Delta_q^2 N / 192`.
This follows from oneloopmaster's rank-four tadpole pole `(m^2)^3/24`: three
mass-squared derivatives divided by `3!` give `1/24` for each box metric pairing,
and the quartic numerator contracts the three pairings to `Delta_q^2 N / 8`.
All six graph poles cancel exactly for each of the four polarization states.
This checks the ultraviolet divergence; it does not add a UV counterterm.

Supply rlibs from the same locked build to reproduce the pole control:

```sh
python3 validation/madloop/four_photon/uv_poles.py \
  --ninjars target/physical/four-photon/final-binary/ninjars \
  --symbolica-rlib /path/to/deps/libsymbolica-HASH.rlib \
  --serde-json-rlib /path/to/deps/libserde_json-HASH.rlib \
  --integrated target/physical/four-photon/integrated-final/result.json \
  --output-dir target/physical/four-photon/uv-integrated
```

The optional integrated comparison fixes pole-accuracy targets before evaluation:
scaled `1e-9`, `1e-25`, and `1e-65` for f64, DoubleFloat, and Float256. The pole
oracle is differentiated symbolically and evaluated from exact fixture inputs
at 256 bits. This provides an independent precision control for the pole only;
MadLoop's binary64 wavefunction/coupling inputs still limit the independent
finite-part comparison. Neither harness is a performance-acceptance benchmark.

The final gate is a separate aggregation of those immutable numerical reports:

```sh
python3 validation/madloop/four_photon/aggregate.py \
  --matrix validation/evidence/four-photon-integrated-matrix.json \
  --poles validation/evidence/four-photon-uv.json \
  --build-record validation/evidence/four-photon-cli-build.json \
  --output validation/evidence/four-photon-integrated.json
python3 validation/madloop/four_photon/test_aggregate.py \
  --matrix validation/evidence/four-photon-integrated-matrix.json \
  --poles validation/evidence/four-photon-uv.json \
  --output validation/evidence/four-photon-aggregation-checks.json
```

The aggregate requires complete sample coverage, successful processes, the fixed
working precisions, unchanged runtime sources, per-graph Laurent/decomposition
checks, μ-mode equivalence, summed-graph checks, exact UV cancellation, and the
independent pole-accuracy targets. Thirteen mutation regressions cover missing or
duplicate rows, unsuccessful processes, mode/sum mismatches, weakened precision
targets, and the separation of diagnostic comparisons.

The archived matrix passes all 108 physical rows at fixed mode four, with maximum
scaled errors `6.08e-16` for finite parts and `2.23e-15` for single poles. Mode one
also passes the physical rows, with maximum `1.32e-10`. All 36 generic mode-four
diagnostics pass, while **all 36 generic mode-one comparisons fail** the unchanged
`1e-9` criterion: maximum finite error `8.22e-7` and pole error `2.25e-9`. Both
reference modes and their stable flags remain in the archived matrix.

Against the independent exact pole control, maximum physical errors are
`2.23e-15`, `1.87e-31`, and `1.92e-76` for f64, DoubleFloat, and Float256. Including
generic diagnostics, the maxima are `2.59e-12`, `6.07e-29`, and `3.95e-73`.
`four-photon-callback-setup.json` separately records the setup bottleneck and
Symbolica-alias experiment; it establishes no steady-state performance gate.

The exact graph mapping is:

| FeynKit index | Directed physical-leg cycle | MadLoop ID | Loop image |
|---|---|---|---|
| 0 | 1,4,3,2 | 182 | −q |
| 1 | 1,2,4,3 | 183 | q |
| 2 | 1,3,4,2 | 181 | −q |
| 3 | 1,4,2,3 | 184 | −q |
| 4 | 1,2,3,4 | 185 | q |
| 5 | 1,3,2,4 | 186 | q |

All source-to-CutTools denominator permutations are `[3,0,2,1]`. The three-helicity archive contains **1,260 independent FeynKit/MadLoop checks**. MadLoop's separate raw/wrapper comparisons at both fixed modes total **2,520**. With the generic case these counts are **1,680** and **3,360**, respectively. Maximum independent scaled error is `1.1697742660401556e-14` for both archives. Integrated comparisons and graph sums pass as described above; physical Ward controls for this process remain pending.

The optional compiled coefficient provider has a separate f64 matrix. Build the
CLI with `--features numerator-jit`, retain an immutable binary, then run:

```sh
python3 validation/madloop/four_photon/compare.py \
  --ninjars target/physical/four-photon/symjit-binary/ninjars \
  --precisions f64 --numerator-evaluator symjit-o2 \
  --output-dir target/physical/four-photon/symjit-matrix
python3 validation/madloop/four_photon/check_provider.py \
  --matrix validation/evidence/four-photon-symjit-o2-matrix.json \
  --poles validation/evidence/four-photon-uv.json \
  --output validation/evidence/four-photon-symjit-o2.json
python3 validation/madloop/four_photon/test_provider.py
```

All **36 physical rows** and **12 generic mode-four diagnostic rows** pass the
same `1e-9` criterion. The gate also checks successful processes, both mu modes,
decomposition, physical graph sums, exact poles, and byte-identical prepared
input hashes against the independent pole oracle. SymJIT remains f64-only;
this evidence makes no performance or higher-precision coefficient claim.
