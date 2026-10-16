# Graph fixtures

`bubble.dot`, `triangle.dot` and `box.dot` use FeynKit's compact DOT dialect with the
built-in phi3 model. Local vertex and edge factors are one. The graph-level
`num` contains the entire synthetic numerator. `ninjars::q0` through
`ninjars::q3` are contravariant loop-momentum components; `ninjars::mu2` is
the extra-dimensional squared-momentum parameter with rank weight two.

Companion JSON files use schema version 1. Numeric inputs are strings: integers
and fractions are exact, while decimal literals use Symbolica's arbitrary
precision parsing derived from the literal text. Use fractions for exact decimal
values or Symbolica's explicit precision notation when needed; inputs never
pass through binary64. `graph` resolves relative to the JSON file. `scale_squared` is the
renormalization scale squared. `parameter_overrides` maps fully qualified
Symbolica parameter names to exact expressions. `samples` is the phase-space
batch, and each `external_momenta` maps FeynKit's external momentum index to
four components ordered `[E,x,y,z]`. Indices and signs follow FeynKit's canonical
edge routing, exposed in the preparation output's `momentum_basis`; do not
infer their order from DOT statements. Only independent momenta are required.
Supplied dependent momenta are checked against that routing (for this triangle,
`P(2)=P(0)-P(1)`).

The examples are synthetic off-shell configurations for deterministic reduction
checks. They are not physical scattering events. Bubble rank is two, triangle
rank is three, and each contains an explicit mu-squared term. The timelike box
has rank four and an explicit mu-to-the-fourth term, contributing -1/6 to the
finite integral. Its three samples exercise changing kinematics and batch tails.

`triangle-reference.json` supplies a separate timelike configuration for checking
original Ninja's generated and tensor callbacks. The Euclidean triangle remains
useful for preparation and future complex-basis tests, but original Ninja's real
basis can produce nonfinite higher-rank results there. See the documented
reproducer under `validation/reference`.

`bubble-parameterized.json` uses the shared FeynKit/Spenso scalar product in
`1 + (1 + P(0)^2)*(q0^2 + mu2)`. Its first row specializes to the constant one;
the other rows retain rank two. This checks parameter rebinding and a fixed
common-template rank when benchmarking changing kinematics.

## Reduction controls and supplied invariants

`reduce --cut-stop 2` retains bubbles and higher cuts, assuming tadpoles contribute
zero. Values `0` and `1` perform the full reduction; values above the available
cut sizes produce zero contributions. Output reports both `requested_cut_stop`
and the effective `cut_stop`. Use `--diagnostics` to inspect the retained cut
counts. A partial reduction changes the computed amplitude unless the omitted
contributions vanish.

Each sample may optionally supply `s_matrix` as an `n` by `n` nested array of
exact-expression strings, in the `propagator_edges` order shown by `prepare`:

```json
"s_matrix": [["0", "-1-1/10^40"], ["-1-1/10^40", "0"]]
```

Entries represent `(V_i-V_j)^2`. The matrix must be symmetric with a zero diagonal.
Expressions can use the fixture's model parameter bindings. Supplied values
control master-integral invariants and rational weights; the cut geometry still
uses the supplied momenta. This permits exact invariants to accompany rounded
momentum components. Omitted matrices are derived from momenta, including in
batches mixing supplied and omitted matrices. Per-lane `s_matrix_source` reports
`supplied` or `momenta`. An invalid numerical matrix fails its lane without
discarding valid neighboring results; malformed dimensions reject the fixture.

The automatic auxiliary mu sampling scale continues to use masses and momentum
invariants, independently of the optional matrix and squared renormalization
scale. `--mu-sampling-scale` accepts an exact manual override. Sampling-scale
failures affect reduction only when box cuts are active.
