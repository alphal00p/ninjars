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

The exact graph mapping is:

| FeynKit index | Directed physical-leg cycle | MadLoop ID | Loop image |
|---|---|---|---|
| 0 | 1,4,3,2 | 182 | −q |
| 1 | 1,2,4,3 | 183 | q |
| 2 | 1,3,4,2 | 181 | −q |
| 3 | 1,4,2,3 | 184 | −q |
| 4 | 1,2,3,4 | 185 | q |
| 5 | 1,3,2,4 | 186 | q |

All source-to-CutTools denominator permutations are `[3,0,2,1]`. The three-helicity archive contains **1,260 independent FeynKit/MadLoop checks**. MadLoop's separate raw/wrapper comparisons at both fixed modes total **2,520**. With the generic case these counts are **1,680** and **3,360**, respectively. Maximum independent scaled error is `1.1697742660401556e-14` for both archives. Integrated NinjaRS comparisons and physical Ward/summed-amplitude checks remain pending.
