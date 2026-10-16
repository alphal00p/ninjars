# Top-quark H → gg triangles

The two DOT files were produced by `ninjars-graph`'s `prepare_hgg` example
from FeynKit's Standard Model rules. They preserve the two fermion-loop
orientations, local numerators, graph sign, routing, and color projector.
Their numerator is four-dimensional from the start, with no R2 or UV graphs.
The projector extracts the coefficient of δab using δab/8 once; no color
or helicity sum is performed.

Each JSON file supplies exact model parameters, zero top/Higgs widths,
and three samples at H-rest-frame kinematics:

1. Gluon helicities (++), using MadLoop's external-wavefunction phases.
2. Gluon helicities (+−).
3. Both polarization vectors replaced by their outgoing momenta, as a
   longitudinal contraction diagnostic. This is not a claim that the
   integrated no-R2 contribution satisfies a complete Ward identity.

`external_momenta` keys refer to FeynKit's external momentum basis;
`external_polarizations` keys refer to the stable physical external-leg
indices. Both use `[E,x,y,z]` components parsed by Symbolica at exact
precision. All samples bind the same polarization legs. The optional
`color_dimension_invariants` policy invokes Idenso's existing fundamental
color invariant formulas.

```sh
ninjars prepare examples/graphs/physical/hgg/top-triangle-0.json
```

The independent reference harness and normalization are described in
`docs/physical-preparation.md`. Physical integrated validation remains
under development. Physical helicity coefficients pass the fixed comparison
gate; the large double-longitudinal diagnostic retains a small f64 pole
cancellation failure recorded explicitly in the evidence.
