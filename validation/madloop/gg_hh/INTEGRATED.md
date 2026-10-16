The completed integrated gate passes all **192 physical rows**: eight unsquared
graphs, four helicities, f64/DoubleFloat/Float256, and both μ-extraction modes.
The default Symbolica numerator provider is used throughout. Batch width three
leaves a two-lane tail in each eight-state fixture. Every process succeeds and
the recorded runtime sources remain unchanged.

The same run includes 192 diagnostic rows: generic complex polarizations and
three Ward substitutions. Every per-graph diagnostic row passes against fixed
MadLoop mode four. The fixed `1e-9` criterion is unchanged. Mode one is retained
independently: all physical and generic-complex rows pass, while **36 of 144 Ward
rows fail** (triangle graphs 0/1 with either single Ward substitution and box
graphs 5/7 with the first Ward substitution). The largest mode-one Ward finite
error is `2.44e-9`.

The physical mode-four maximum scaled errors are `7.03e-13` for finite parts and
`4.21e-14` for single poles; double poles vanish. Physical sums, the finite split
`finite=cc+rational`, and μ-mode equivalence pass. No physical finite-part
accuracy claim beyond the binary64 MadLoop input oracle is made: its mode-four
arithmetic promotes the same binary64 momenta, wavefunctions and couplings.

An independent exact pole control uses Symbolica differentiation:

- Massive rank-two triangles: `Delta_q N / 8`.
- Massive rank-four boxes: `Delta_q^2 N / 192`.

Here `Delta_q` is the four-dimensional `+---` Laplacian and the prepared overall
factor is included once. The shared master normalization gives the triangle
coefficient from two mass-squared derivatives of `A00_pole=(m²)²/4`, divided by
`2!`; the box coefficient follows from three derivatives of
`A0000_pole=(m²)³/24`, divided by `3!`. Analytic controls check `q²`, `q0²`, `q1²`
for triangles and three quartic polynomials for boxes. The prepared triangle
numerators have zero Laplacian. All eight graph poles sum to **exactly zero**
for each of the eight bound states.

The independent pole oracle is evaluated from exact fixture inputs at 256 bits.
The fixed per-precision targets and observed maximum physical errors are:

| Arithmetic | Target | Maximum physical pole error |
|---|---:|---:|
| f64 | `1e-9` | `4.21e-14` |
| DoubleFloat | `1e-25` | `2.71e-30` |
| Float256 | `1e-65` | `2.02e-75` |

Large diagnostic cancellations remain visible. For the summed double-Ward pole,
f64 misses the `1e-9` target in expansion/sampling modes with absolute errors
`1.015e-9` and `1.053e-9`. DoubleFloat misses its fixed `1e-25` target with
`2.244e-25` and `2.157e-25`. These are sums of the printed components using
120-digit decimal arithmetic; no additional binary64 summation is introduced.
Float256 passes. The aggregate explicitly reports these diagnostic failures,
while all physical summed-pole controls pass. No tolerance is relaxed.

Reproduce with an immutable all-feature CLI and its matching build record:

```sh
python3 validation/madloop/gg_hh/compare.py \
  --ninjars target/physical/four-photon/symjit-binary/ninjars \
  --build-record target/physical/four-photon/symjit-binary/build.json \
  --output-dir target/physical/gg-hh/integrated-repeat
python3 validation/madloop/gg_hh/uv_poles.py \
  --ninjars target/physical/four-photon/symjit-binary/ninjars \
  --symbolica-rlib /path/to/deps/libsymbolica-HASH.rlib \
  --serde-json-rlib /path/to/deps/libserde_json-HASH.rlib \
  --integrated target/physical/gg-hh/integrated-repeat/result.json \
  --output-dir target/physical/gg-hh/poles-repeat
python3 validation/madloop/gg_hh/aggregate.py \
  --matrix validation/evidence/gg-hh-integrated-matrix.json \
  --poles validation/evidence/gg-hh-uv.json \
  --build-record validation/evidence/gg-hh-cli-build.json \
  --output validation/evidence/gg-hh-integrated.json
python3 validation/madloop/gg_hh/test_aggregate.py \
  --matrix validation/evidence/gg-hh-integrated-matrix.json \
  --poles validation/evidence/gg-hh-uv.json \
  --output validation/evidence/gg-hh-aggregation-checks.json
```

The aggregate requires full matrix/control coverage, correct working precision,
process success, unchanged source inputs, physical Laurent/split agreement,
physical μ-mode equivalence, physical summed results, exact UV cancellation,
and independent physical pole accuracy. Fifteen mutation regressions verify
these gates and preserve the separation of diagnostic failures. This is
correctness evidence, not a performance-acceptance measurement.
