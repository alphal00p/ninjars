# Robustness fixtures

Hand-written compact DOT graphs and sample fixtures used by `tests/robustness.rs`.
Most come from a survey of the CLI on arbitrary one-loop graphs.

| Fixture | Exercises |
| --- | --- |
| `gluon-bubble.dot`, `ghost-bubble.dot` | Built-in sectors; derived `1/AutG` and ghost-loop sign; color projector |
| `hgg-top.dot`, `hgg-top.json` | Automatic `delta_ab/8` projector and derived fermion-loop sign, against `hgg-top-projector.*` and the generated `examples/graphs/physical/hgg/top-triangle-0.json` |
| `hgg-top-width.json` | Width warning and `--complex-mass-propagators` (WT = 3/2) |
| `haa-wloop.dot`, `haa-wloop-one-polarization.json` | Explicit mode without `num`; a missing polarization |
| `gg-gluon-loop.*`, `gg-color-only-projector.dot` | Color invariants; open Lorentz indices named by leg |
| `eea-vertex.dot`, `eea-vertex-spinors.json` | Explicit unit spinors for the QED vertex with a massive electron |
| `eea-vertex-trace.*` | The same vertex closed by a hand-written spin-trace projector |
| `phi3-box.dot`, `phi3-box-zero-momentum.json` | A vanishing momentum making two propagators coincide |
| `phi3-leg-bubble.*` | An on-shell tree propagator: Symbolica's warning on stderr |
| `disconnected.dot` | A disconnected graph |

The spinor test uses that the reduced amplitude is bilinear in its external
spinors: with unit spinors `vbar = e_a`, `u = e_b` it returns the amputated
matrix `Gamma_ab`, and `sum_ab Gamma_ab (p0-slash p1-slash)_ba` must equal the
basis-independent trace projector result.
