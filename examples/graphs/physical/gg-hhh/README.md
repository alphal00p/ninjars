These 50 graphs describe the top-quark contribution to `g g → H H H` with four-dimensional numerators, zero widths, R1 retained, and no explicit R2 or UV counterterms. They were generated independently with FeynKit's Standard Model, local Feynman rules, Idenso contractions and `ColorSum::averaged(true)` projector. The color projection extracts the coefficient of `δ_ab` through `δ_ab/8` exactly once; it is not a color-summed squared matrix element.

The graph classes are:

| Files | Number | Higgs trees |
| --- | ---: | --- |
| `top-triangle-0..1` | 2 | One quartic Higgs vertex |
| `top-triangle-2..7` | 6 | Two cubic Higgs vertices |
| `top-box-8..25` | 18 | One cubic Higgs vertex |
| `top-pentagon-26..49` | 24 | None |

Every DOT numerator retains symbolic external momenta, gluon polarizations, top mass and model couplings. The accompanying JSON binds those inputs independently, so a change of phase-space point or polarization does not require a new symbolic numerator. Internal Higgs propagators belong to the overall prefactor; only the three, four or five top propagators enter the reduction family.

The fixed example has `sqrt(s)=450`, `MH=125`, `MT=ymt=173`, and renormalization scale squared `10000`. The incoming gluons have energy225 along opposite z directions; three outgoing Higgs bosons each have energy150 and form an equilateral triangle in the xy plane. Momenta and couplings use exact rational/radical strings. The eight sample rows are `pp`, `pm`, `mp`, `mm`, generic complex polarizations, and three longitudinal Ward controls. Longitudinal substitutions remain diagnostic in the requested R2-excluded convention; no finite Ward-zero identity is assumed.

For example, from the repository root:

```sh
ninjars prepare examples/graphs/physical/gg-hhh/top-pentagon-26.json
ninjars reduce examples/graphs/physical/gg-hhh/top-pentagon-26.json \
  --precision double-float --batch-size 3 --mu-extraction expansion --diagnostics
```

Raw numerator matching against all50 isolated MadLoop graphs passes the fixed scaled `1e-9` criterion on70 complex loop points per sample. The exact routing and normalization evidence is in `validation/evidence/gg-hhh-raw.json`; the reproducible harness is under `validation/madloop/gg_hhh`. Integrated validation is recorded separately and must not be inferred from raw agreement.
