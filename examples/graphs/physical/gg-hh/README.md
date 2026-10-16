These eight graphs are generated from FeynKit's Standard Model rules using only
its top-antitop-gluon, top-antitop-Higgs and triple-Higgs vertices. Graphs 0 and 1
are the two Higgs-channel triangles; graphs 2 through 7 are the six oriented
boxes. They retain four-dimensional numerators, the closed-fermion sign, and
one `delta_ab/8` projector. The resulting amplitude is the coefficient of
`delta_ab`, without a spin average or a squared-amplitude color average.

The two triangle fixtures retain the tree-level Higgs bridge. Shared graph
preparation moves its denominator into the overall factor; it is not a fourth
loop denominator. At this phase-space point the complete overall factor is
`-1/74375` for triangles and `-1` for boxes.

Inputs use exact strings: `sqrt(s)=300`, `MH=125`, `MT=ymt=173`, zero top/Higgs
widths, scale squared `10000`, and outgoing Higgs momenta
`(150, ±15*sqrt(11), 0, ±20*sqrt(11))`. The eight samples are ordered
`pp`, `pm`, `mp`, `mm`, generic complex gluon polarizations, first-gluon Ward
substitution, second-gluon Ward substitution, and both Ward substitutions.
Generic and Ward inputs are diagnostics. Their unsquared graph numerators are
independently checked; no individual graph Ward-vanishing condition is imposed.
No explicit R2 or UV counterterm is included.

The generated DOT files can be prepared or reduced with the existing CLI:

```sh
ninjars prepare examples/graphs/physical/gg-hh/top-triangle-0.json
ninjars reduce examples/graphs/physical/gg-hh/top-box-3.json \
  --precision double-float --batch-size 3
```

Regeneration, exact routing/color controls, reference limitations and the
independent MadLoop comparison are described in
[the validation harness](../../../../validation/madloop/gg_hh/README.md).
