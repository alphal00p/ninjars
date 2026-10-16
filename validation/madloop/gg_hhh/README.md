This validation harness prepares and matches each unsquared top-quark graph in `g g → H H H`. Python only orchestrates reference programs and files; FeynKit preparation, exact routing, numerator contractions and symbolic checks remain Rust using shared FeynKit/Symbolica APIs.

Both independent generators find50 graphs:8 triangles,18 boxes and24 pentagons. The triangle subset contains two quartic-Higgs trees and six nested cubic-Higgs trees. Topological matching preserves the rooted scalar-tree structure, not just its total momentum. Each directed fermion cycle is matched to the generated MadLoop `LCUT_F` or `LCUT_AF` chain, then FeynKit `IntegralFamily::find_mapping` supplies the routing transformation. Every transformed denominator is checked exactly.

The original `loop_sm` model is unchanged. MadLoop is configured with grouping disabled and an isolated pre-grouping graph filter; every export must have one loop with multiplier1. The probe links only the selected loop numerator and CutTools interface. R2/UV drivers are excluded, an unexpected multiprecision wavefunction callback aborts, and both fixed CutTools modes1/4 are evaluated explicitly. Automatic backend switching is disabled. Mode4 promotes the same binary64 momenta, wavefunctions and couplings; it does not provide an independently high-precision physical-input oracle.

The raw factor is `−1/2`: one closed-fermion sign times `Tr(Ta Tb)=δ_ab/2`, with FeynKit's `δ_ab/8` projection applied once. Undoing the reference interface's `1/(16π²)` normalization gives the integrated factor `−8π²`. Exact zero-momentum temporal-polarization controls check the complete FeynKit numerator and every Higgs bridge:

| Topology | Exact control |
| --- | --- |
| Quartic triangle | `2 GC11² GC94 GC33 MT³/MH²` |
| Nested cubic triangle | `−2i GC11² GC94 GC69² MT³/MH⁴` |
| Box with Higgs bridge | `2i GC11² GC94² GC69 MT⁴/MH²` |
| Pentagon | `−2i GC11² GC94³ MT⁵` |

The coupling correspondence is `GC11=MG GC5=iG`, `GC94=MG GC37=−i yt/√2`, `GC69=MG GC30=−6i λv`, and `GC33=MG GC_HHHH=−6i λ`. The bridge propagators are already included in the FeynKit overall factor and MadLoop scalar wavefunctions.

The70-point affine complex loop grid is unisolvent through total degree4. Symbolica verifies that every bound numerator has degree≤4 (triangles≤2) and no explicit μ². Each fixed polarization is checked separately; loop-grid unisolvence is not a claim about arbitrary polarization dependence. Four helicities, generic complex polarization vectors, and three Ward controls give28,000 independent FeynKit/MadLoop raw comparisons. The separate56,000 checks internal to MadLoop include both fixed modes. Longitudinal states remain diagnostic in this R2-excluded convention; finite Ward-zero identities are not assumed. The largest independent scaled error is `6.784434851757174e−13` under the unchanged `1e−9` gate.

Run from the repository root with the pinned development environment and Symbolica license available:

```sh
cargo build --locked -p ninjars-graph --features numerator-jit \
  --message-format=json > /tmp/gg-hhh-cargo-artifacts.jsonl
python3 validation/madloop/gg_hhh/prepare_feynkit.py \
  --cargo-artifacts /tmp/gg-hhh-cargo-artifacts.jsonl --output-dir target/physical/gg-hhh/generated-new
python3 validation/madloop/gg_hhh/prepare_madloop.py \
  --cuttools-dir target/reference/madloop-gg-hh/graph-9/Source/CutTools \
  --output-dir target/reference/madloop-gg-hhh-new
```

The explicit CutTools directory must contain a library built with the active compiler. MadLoop's existing external-dependency mechanism is used, with source/library hashes and link destinations checked; compiler mismatches are rejected. The archived run reused the previously verified gcc15.3 build, without modifying MadLoop's vendor files. Then run `probe.py --graph ID --exports-dir … --output-dir …` for every ID in `graphs.json`, followed by:

```sh
python3 validation/madloop/gg_hhh/collect.py --exports-dir target/reference/madloop-gg-hhh-new \
  --probes-dir target/reference/madloop-gg-hhh-probes-new --output target/physical/gg-hhh/reference-new.json
python3 validation/madloop/gg_hhh/match.py \
  --cargo-artifacts /tmp/gg-hhh-cargo-artifacts.jsonl \
  --reference validation/evidence/gg-hhh-madloop.json.gz \
  --fixtures examples/graphs/physical/gg-hhh --output-dir target/physical/gg-hhh/matching-new
python3 -m unittest discover -s validation/madloop/gg_hhh -p test_reference.py -v
```

The large raw reference is stored losslessly as deterministic gzip (`mtime=0`, no filename metadata). `gg-hhh-reference-archive.json` records both hashes and byte counts. The shared reader accepts JSON or JSON.gz, and tests check exact byte round-trip, deterministic compression, corruption refusal and missing/duplicate/nonfinite raw controls.

`build_cli.py` records a fresh locked all-feature development build and copies an immutable executable. `compare.py` uses this record for the separate integrated matrix; its operational timings are not performance measurements. Raw matching alone does not establish integrated correctness or any performance gate.
