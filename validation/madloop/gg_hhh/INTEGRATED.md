The integrated control evaluates all50 independently matched unsquared graphs, each in f64, DoubleFloat and Symbolica Float256, using both μ-expansion and μ-sampling. Width3 processes eight fixed states with an incomplete tail. The primary gate uses the fixed MadLoop mode4 result; mode1 is retained separately. No grouping, backend rescue, explicit R2 or UV counterterm contributions are included.

The immutable correctness executable and its fresh locked build record are created with:

```sh
python3 validation/madloop/gg_hhh/build_cli.py --output-dir target/physical/gg-hhh/new-binary
python3 validation/madloop/gg_hhh/compare.py \
  --ninjars target/physical/gg-hhh/new-binary/ninjars \
  --build-record target/physical/gg-hhh/new-binary/build.json \
  --output-dir target/physical/gg-hhh/new-integrated
```

The expected matrix has2400 graph/state rows:1200 physical helicity rows and1200 generic/longitudinal diagnostic rows. Every row records all three Laurent coefficients, the cut-constructible/rational sum, both fixed reference modes, the process status and working precision. Raw graph matching fixes the overall integrated conversion `−8π²` before this comparison. The criterion remains `|actual−reference|/max(1,|reference|) ≤ 1e−9` for each complex coefficient. μ-mode equivalence and summed physical graphs are separate gates.

Exact poles are obtained without the reduction engine: Symbolica differentiation gives `Δ_q N/8` for rank≤2 triangles and `Δ_q² N/192` for rank≤4 boxes, in the `+---` metric. Massive rank≤4 pentagons have superficial UV degree at most−2 and no internal soft/collinear poles. The probe checks the degree bound for every topology and all squared masses, and depends on the shared preparation's normalized unit-power propagator contract. Harmonic polynomials above the allowed degree are explicit rejection controls. It applies the graph prefactor exactly once. At the fixed fixture point, the400 graph/state poles are computed at256 bits and all eight fifty-graph sums cancel exactly.

Use the `symbolica` and `serde_json` rlibs recorded in the immutable build's `cargo-artifacts.jsonl`:

```sh
python3 validation/madloop/gg_hhh/uv_poles.py \
  --ninjars target/physical/gg-hhh/new-binary/ninjars \
  --symbolica-rlib /path/from/cargo-artifacts/libsymbolica.rlib \
  --serde-json-rlib /path/from/cargo-artifacts/libserde_json.rlib \
  --integrated target/physical/gg-hhh/new-integrated/result.json \
  --output-dir target/physical/gg-hhh/new-uv
python3 validation/madloop/gg_hhh/aggregate.py \
  --matrix target/physical/gg-hhh/new-integrated/result.json \
  --poles target/physical/gg-hhh/new-uv/result.json \
  --build-record target/physical/gg-hhh/new-binary/build.json \
  --output target/physical/gg-hhh/new-aggregate.json
```

The independent per-graph and summed pole accuracy targets remain `1e−9`, `1e−25` and `1e−65` for the three precisions. Generic and longitudinal substitutions are explicitly diagnostic; no finite Ward-zero identity is assumed in the R2-excluded convention. The finite reference inputs are binary64 MadLoop wavefunctions and couplings even in fixed mode4, so finite-part convergence toward Float256 does not establish an independent high-precision physical-input oracle. All elapsed times are operational records, not performance acceptance evidence.

The large matrix can be archived losslessly with `archive_integrated.py --input RESULT --archive MATRIX.json.gz --manifest MANIFEST.json`. The existing shared reader supports both plain JSON and gzip in the pole and aggregation commands. Compression fixes the timestamp to zero and omits filename metadata. The plain manifest retains compressed and uncompressed hashes/byte counts, complete graph/state coverage, the source hashes recorded before execution and the source-change report checked afterward. The exact-pole probe additionally compares its source, CLI and dependency artifact hashes before and after execution. `test_aggregate.py` checks corruption/truncation, missing/duplicate rows, altered numeric values with stale flags, fixed accuracy targets and diagnostic separation.

The retained matrix passes all 1,200 physical rows and all 1,200 diagnostic rows in both fixed MadLoop modes. All 350 CLI processes, μ-mode equivalence checks, summed-graph checks and runtime source guards pass. The largest mode-4 Laurent discrepancy is `1.09254e−15` for physical helicities and `7.28073e−12` including the diagnostic substitutions. No criterion was relaxed.

The independently computed single-pole errors demonstrate the additional numerical accuracy:

| Arithmetic | Maximum physical per-graph pole error | Maximum physical 50-graph pole sum |
| --- | ---: | ---: |
| f64 | `1.75634e−16` | `3.62695e−16` |
| DoubleFloat | `4.57064e−32` | `1.00875e−31` |
| Float256 | `5.53969e−77` | `1.11865e−76` |

The generic/longitudinal pole controls also pass their unchanged precision targets. The compressed matrix is `validation/evidence/gg-hhh-integrated.json.gz`; the adjacent `gg-hhh-integrated-archive.json` records both byte identities. The plain `gg-hhh-uv-poles.json` and `gg-hhh-integrated-aggregate.json` retain the exact controls, recomputed per-row errors, summed-pole residuals, provenance and gates. The matrix shrinks losslessly from 9,480,296 bytes to 603,164 bytes. These results establish this correctness matrix, with the binary64-reference limitation above; they do not establish a performance gate.
