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
| `emu-vertex.*`, `emu-vp.*` | Single-vertex fermion lines and loop vectors with open Lorentz indices (e- mu- vertex correction and vacuum polarization), against slashed-momentum trace projectors |
| `pentagon-axial-trace.*`, `pentagon-scalar.*` | A hand-written `Tr[gamma5 p0 p2 p3 p4]` numerator against explicit chiral-basis matrices times the scalar pentagon |
| `hza-top.*`, `hza-top-reversed.*` | `H -> Z a` through a top loop: Levi-Civita tensors in both engines; the axial part flips sign with the loop orientation |
| `eeza-top.dot`, `eeza-top-spinors.json`, `eeza-top-trace.*` | `e+ e- -> Z a` through a photon and a top loop: Levi-Civita indices summed with an open spinor chain, against a spin-trace projector |
| `dangling-color-indices.dot` | Dangling color indices in a hand-written numerator are not joined |
| `phi3-box.dot`, `phi3-box-zero-momentum.json` | A vanishing momentum making two propagators coincide |
| `phi3-leg-bubble.*` | An on-shell tree propagator: Symbolica's warning on stderr |
| `disconnected.dot` | A disconnected graph |

The spinor test uses that the reduced amplitude is bilinear in its external
spinors: with unit spinors `vbar = e_a`, `u = e_b` it returns the amputated
matrix `Gamma_ab`, and `sum_ab Gamma_ab P_ba` must equal the basis-independent
trace projector result for the same `P`: `p0-slash p1-slash` for the QED
vertex, `p0-slash p1-slash p2-slash` for `eeza-top`, whose tree vertex carries
one gamma matrix.
