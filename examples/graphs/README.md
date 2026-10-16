# Graph fixtures

`bubble.dot` and `triangle.dot` use FeynKit's compact DOT dialect with the
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

The examples are off-shell Euclidean configurations intended for deterministic
reduction checks. They are not physical scattering events. Bubble rank is two,
triangle rank is three, and each contains an explicit mu-squared term.

`triangle-reference.json` supplies a separate timelike configuration for checking
original Ninja's generated and tensor callbacks. The Euclidean triangle remains
useful for preparation and future complex-basis tests, but original Ninja's real
basis can produce nonfinite higher-rank results there. See the documented
reproducer under `validation/reference`.
