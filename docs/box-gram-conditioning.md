# Conditioning bound of the box Gram determinant

The generic engine in double-double and in MPFR from 106 bits (not in
binary64 or shorter MPFR, section 4) refuses a box cut, typed, when its cut solutions lie so
far outside its shifts that the reduction is predicted to lose more digits
than the carrier keeps: `CutOutcome::Unstable(CutFailure::DegenerateGram)`
(lane status `CutFailed { cut_index, failure: DegenerateGram }`). The bound
is in `ninjars::conditioning` (`crates/ninjars/src/conditioning.rs`); the
generic box reducer asks for it after the box geometry is solved
(`BoxReductionKernels::gram_conditioned_batch`, `crates/ninjars/src/boxes.rs`),
and the Symbolica kernels evaluate it in the carrier
(`crates/ninjars-symbolica/src/box_kernels.rs`, `gram_conditioned`). This
note derives it and records its calibration. It answers NinjaRS issue #15.

## 1. Why a bound

A box cut solves `q̄.k_i = h_i` (`k_i = V_i - V_0`, `i = 1, 2, 3`,
`h_i = (m_i² - m_0² - k_i²) / 2`, `q̄ = q + V_0`) and `q̄² = m_0²`. Its
solutions are `q = a ± t v` with `a` the center (Minkowski-orthogonal to the
normal `v`), `t² = β / v.v` and `β = m_0² - q̄_center²`. The Gram determinant
`G = det(k_i.k_j)` of the box is the Minkowski square of the unnormalised
normal `ε(k_1, k_2, k_3)` (Cauchy-Binet). It vanishes in two ways:

- the `k_i` are nearly linearly dependent: the unnormalised normal vanishes
  as a whole, the chart-normalised normal stays of order one, and the center
  `a ~ h / σ_min` runs away;
- the `k_i` span a nearly light-like hyperplane: the normal is nearly null,
  `v.v -> 0`, and `t` runs away.

Either way the cut solutions are large compared with the box's shifts. At
numerator rank `r` the integrand is evaluated there with terms of relative
size `λ^r`, and the box coefficients (and, through the subtraction of the
box, the triangle and bubble coefficients) come out of their cancellation.

Before this bound the box geometry was refused only when its scores fell
below `2^-p` (`ninjars-symbolica/src/high_cuts.rs`: the chart's normalised
3x3 minor squared, and `|v.v|² / |v|⁴`), i.e. at the carrier's rounding level,
and the box mu-expansion bound (`docs/box-mu-conditioning.md`) models only
the uncut lines. NinjaRS issue #15: at two points of HC05 `g g > x0 g g`
whose final state is collinear to binary64 rounding (the box with legs
`{p1 + p2, p3, p4, p5}` has `|G| / s³ = 1.4e-35` and `3.1e-36`), the engine
in MPFR served values `4.7e34 / 6.7e30` (40 digits), `3.4e18 / 9.4e13`
(57 digits) and `8.2e-5 / 4.4e-11` (80 digits) off with status success; it
lost about 96 / 91 decimal digits at every precision. The degeneracy there is
the first kind: `p1 + p2`, `p3` and `p4` span only two dimensions, so the
quantity `|v|² / |v.v|` of the normal (proposed in the issue) stays exactly 1
in the centre-of-mass frame and does not see it.

## 2. The loss model

The bound measures the size of the cut solutions against the box's own
scale:

```text
λ² = (|a|² + |β / v.v| |v|²) / S²,    S² = max_i max(|V_i|², |m_i²|),
```

with `|.|` the Euclidean norm of the four components (squared moduli for
complex components) and the maximum over the four cut lines. `λ²` is
invariant under the scaling of the normal, and of order one at a regular box
(the calibration's regular boxes have `log2 λ² = -1 .. 3`, planar 2 -> 3
points up to `7.2`).

The predicted loss in bits at numerator rank `r` is

```text
L = (r + 1) / 2 · log2 λ²  =  (r + 1) log2 λ,
```

one power of `λ` per power of the loop momentum and one more for the
division of the numerator by the box's own structure. That counting is a
heuristic; the evidence is the measured slope (section 3), which does not
depend on the number of propagators of the lane.

`L` is an upper envelope, not a predictor of the loss. Generic numerators
(random tensors) come within 11.3 bits above it; real lanes stay far below:
HC05 lanes lose 9 to 83 bits less than `L`, and on the issue #15 geometry
with random tensors the measured loss is 154 bits against `L = 348`. Only
the excess side matters for safety, and that is what the margin calibrates;
the overshoot shows up as over-refusal (section 3). A cut is refused when

```text
L > p - M,    M = BOX_GRAM_MARGIN_BITS = 28,
```

`p` the carrier's precision (106 for `DoubleFloat`, the bits of an MPFR
`Float`); carriers below `BOX_GRAM_MIN_PRECISION = 106` bits are left out
(section 4). The kernels compare `(λ²)^(r + 1)` with `2^(2 (p - M))`
in the carrier itself (no conversion to binary64: `λ²` can exceed its
range), NaN and infinity (`v.v = 0`) refused. The quantity uses only the
solved geometry, the shifts and the masses: it costs a few dozen operations
per box cut and does not depend on the numerator. The test only refuses:
every served value is unchanged bit for bit.

**Invariance** (measured by the review of #16 with an independent mpmath
implementation, which reproduces every fixture's `log2 λ²` to 0.01 bit):
`log2 λ²` is exact under an overall rescaling of the kinematics, under
rotations and under the choice of the cut's first line (`a` is the absolute
centre); it changes by at most 2.2 bits under a boost of rapidity 3 and by
at most 1 bit under re-routing the loop momentum (`V_i -> V_i - V_j`) on the
degenerate box. The floating-point loss is not boost or routing invariant
either, and the margin absorbs it.

Two other quantities were tried and rejected (section 3):

- `|v|² / |v.v|` of the normal: blind to linearly dependent shift
  differences (issue #15).
- The normalised Gram determinant `Π_i |k_i|² / |G|`: it sees both kinds of
  degeneracy, and the loss grows with the same slope `(r + 1) / 2` per bit,
  but from a geometry-dependent baseline (11 to 14 bits on the calibration
  geometries): `|G|` of a regular box is small against the Euclidean product
  whenever its momenta are light-like or massive. A bound on it either
  refuses regular boxes or needs an offset that is not generic. The cut
  solutions' size has no such baseline: on the same geometries
  `log2 λ² = log2(Π |k_i|² / |G|) - 13.6` at the degenerate box, and the
  regular boxes stay at `-1 .. 3`.

## 3. Calibration

**Synthetic lanes** (`validation/conditioning/box-gram` keeps a subset). A
2 -> 3 point at `sqrt(s) = 1 TeV` with a massive `x0` (125 GeV) and two
massless partons on one axis `n` (`p3`, `p4` along `n`, `p5` balancing),
`p4` tilted out of it by `θ = 1e-1 .. 1e-40` rad, all momenta at 420 digits;
the pentagon lane with the legs in order (`n = 5`, ranks 5, 6, 7) and the
degenerate box itself (`n = 4`: legs `{p1 + p2, p3, p4, p5}`, ranks 4, 5,
6), massless internal lines, three random tensor numerators per lane (every
monomial of degree `d` scaled by `sqrt(s)^(r - d)`, generic: no symmetry
suppresses the leading terms). Two sets: massless incoming and outgoing
partons, and massive external legs (30, 40, 125, 50, 60 GeV). Only the MPFR
runs of the massless set are used. Its generator wrote the massless external
invariant `p5²` as a 420-digit residue (`1.9e-416`) instead of an exact 0:
binary64 and double-double round it to 0, MPFR keeps it, and the harness
evaluates the masters without an on-shell threshold, so MPFR reduced a
slightly off-shell leg and the carriers' finite parts differed by order one.
With the entry set to an exact 0, Float(128) agrees with double-double to
about `1e-29`. Every MPFR precision and the references saw the same residue,
so the MPFR losses are measured on one consistent lane. The committed
fixtures have exact invariants, and the generator now writes a single leg's
mass exactly. Planar
points (the final state in a plane containing the beam, tilted out of it by
`0 .. 1e-2`) as controls. Each lane was reduced by this branch with the
bound switched off (a local switch, not committed) in binary64,
double-double and MPFR at 128, 256 and 512 bits, against MPFR at 1800 and
2000 bits (agreeing to better than `1e-187`); the measured loss is
`p + log2(error)` relative to the lane's value: 240 fixtures, 720 lanes,
3120 carrier-lanes with a reference.

Least-squares fits of the measured loss against `log2 λ²` over the lanes with
a measurable loss (`6 < loss < p - 6`) and a degenerate box
(`log2 λ² >= 20`):

| n | rank | lanes | slope | (r + 1) / 2 | offset | loss - L, max | log2 λ² |
|---|---|---|---|---|---|---|---|
| 4 | 4 | 117 | 2.503 | 2.5 | +5.8 | 11.3 | 26 - 192 |
| 4 | 5 | 99 | 3.005 | 3.0 | +5.6 | 10.2 | 26 - 159 |
| 4 | 6 | 87 | 3.521 | 3.5 | +4.0 | 10.2 | 26 - 126 |
| 5 | 5 | 102 | 3.004 | 3.0 | +0.2 | 5.2 | 26 - 159 |
| 5 | 6 | 88 | 3.526 | 3.5 | -1.9 | 2.9 | 26 - 126 |
| 5 | 7 | 81 | 4.019 | 4.0 | -1.5 | 2.6 | 26 - 126 |

The slope is `(r + 1) / 2` to better than one percent at every rank, for the
pentagon and the box lane alike, in double-double and in MPFR (whose
significance tracking does not change it). Against the normalised Gram
determinant the same lanes give the same slopes with offsets of `-28.6 ..
-55.1` bits.

**What the calibration covers.** One geometry family (a 2 -> 3 point tilted
out of a final state on one axis), lanes with `n = 4` and `n = 5`, ranks 4
to 7, and the first kind of degeneracy (linearly dependent shift
differences). Not measured: the second kind (a nearly light-like hyperplane,
`v.v -> 0`), lanes with `n >= 6` containing a degenerate box (2 -> 4
processes), and other degenerate geometries. The bound applies to them with
the same slope and margin; its safety there rests on `L` being an envelope.

**pyAmpliCol lanes.** pyAmpliCol `ebc209a8` (path-patched to this branch,
with local debug prints of `λ²` per box cut and of every lane's Laurent
coefficients, never pushed) evaluated, with the bound switched off,
HC05 `g g > x0 g g` (Born-interfered, HEFT; pentagon lanes of rank 6 and
box lanes of rank 4 and 5) and Born-interfered `g g > g g g` at the same kind
of points: the final state on one axis tilted by `1e-1 .. 1e-30` rad, and
planar points tilted by `0 .. 1e-2`, in double-double and in MPFR at 18, 40,
57, 60, 80, 100 and 134 digits (pyAmpliCol adds about 67 bits: 127 .. 512
bits), against 360 digits; loop-induced `g g > z z g` at tilts `1e-1 ..
1e-16` in double-double and at 18 and 57 digits against 134 (its MPFR runs
take about an hour per precision). Errors are measured on the family's scale
(the largest truth modulus over the lanes of one family at one point).

| set | carrier-lanes with boxes | `log2 λ² > 20` | loss - L, max | served > 1e-5 off, not refused at M = 28 | refused within 1e-5 at M = 28 |
|---|---|---|---|---|---|
| HC05 collinear, MPFR | 3954 | 286 | 4.2 | 0 | 31 |
| HC05 collinear, double-double | 496 | 16 | -7.4 | 1 (*) | 11 |
| HC05 planar | 1548 | 0 | | 0 | 0 |
| `g g > g g g` planar | 3168 | 0 | | 0 | 0 |

(*) a box lane with `log2 λ² = -10`, `1e-3` off on the family scale at a
point that double-double serves `1.5e-22` from the truth: not a box-Gram
loss, unaffected by any margin. `g g > g g g` collinear and `g g > z z g`
are left out of the lane-level table: there, lanes of the same family could
not be matched reliably across runs (loop-induced families repeat with
identical shapes) or are dominated by the triangle source of section 4; they
enter the point-level comparison below.

**Point level, before and after** (the same tilt scans, this branch against
the bound switched off; wrong = more than `1e-5` from the 360-digit truth,
over-refused = newly refused but within `1e-5` before; tilts in rad):

| process, carrier | wrong before | wrong after | newly refused | over-refused |
|---|---|---|---|---|
| HC05 collinear, double-double | 2 (1e-6, 1e-8) | 0 | 4 | 2 (1e-4, 1e-5) |
| HC05 collinear, 18 digits | 2 | 0 | 4 | 2 (1e-5, 1e-6) |
| HC05 collinear, 40 digits | 4 | 0 | 5 | 1 (1e-8) |
| HC05 collinear, 57 / 60 digits | 3 / 3 | 0 / 0 | 5 / 5 | 2 / 2 (1e-10, 1e-12) |
| HC05 collinear, 80 digits | 4 | 0 | 5 | 1 (1e-14) |
| HC05 collinear, 100 digits | 3 | 0 | 4 | 1 (1e-16) |
| HC05 collinear, 134 digits | 2 | 0 | 2 | 0 |
| HC05 planar, `g g > g g g` collinear and planar, every carrier | 0 | 0 | 0 | 0 |
| `g g > z z g` on one axis (7 tilts, truth 134 digits), double-double | 3 (1e-6, 1e-8, 1e-12) | 0 | 4 | 1 (1e-4) |
| `g g > z z g` on one axis, 18 digits | 2 (1e-6, 1e-8) | 0 | 2 | 0 |
| `g g > z z g` on one axis, 57 digits | 2 (1e-12, 1e-16) | 0 | 2 | 0 |

Every value served after the change equals the one served before, digit for
digit. The over-refusal is one or two decades of tilt per carrier, the price
of a margin set by generic numerators.

**Ordinary kinematics** (the review of #16: `λ²` over all 30 boxes of
`g g > g g g` at `sqrt(s) = 1 TeV`). Over 100 000 flat points the largest
`log2 λ²` is 18.2, so nothing is refused in double-double (edge 26 at rank
5), at 57 digits (76) or at 500 digits (544); with generation cuts
(`ΔR > 0.4`, `pT > 20 GeV`) it stays at or below 10. A soft gluon leaves
`log2 λ²` at 1.8. Two final-state gluons collinear, a gluon collinear to the
beam, or the whole final state on one axis grow 6.6 bits per decade of
angle: double-double refuses below about `1e-4` rad, 57 digits below about
`1e-11` rad. At 500 digits (1661 bits) a rank-6 box is refused only above
`log2 λ² = 467` (rank 7: 408), a tilt of about `1e-70`; points given in
binary64 sit at `log2 λ² <= 110` unless exactly degenerate, so there the
bound refuses nothing physical (the #15 points, 108.7 and 108.5, are served
right at 500 digits).

Real lanes lose less than the synthetic ones (HC05 lanes with
`log2 λ² > 20`: median `loss - L` from `-9` bits at `log2 λ² = 20 .. 30` to
`-83` at `160 .. 170`; the Born interference suppresses the leading terms),
so the synthetic lanes set the margin.

**The margin.** A served lane must keep the lax target `1e-5`, `log2(1e5) =
16.6` bits. Over every calibration lane with a degenerate box
(`log2 λ² > 20`) the measured loss exceeds `L` by at most `11.3` bits
(synthetic `n = 4`, rank 4, at 512 bits; pyAmpliCol lanes at most 4.2), so

```text
M = ceil(16.6 + 11.3) = 28.
```

On the synthetic lanes (massive external legs for binary64 and double-double)
the operational criterion (no lane served more than `1e-5` off stays served)
holds in MPFR from `M = 14`, in binary64 from `M = 22` and in double-double
from `M = 26`; `M = 28` refuses 8 (double-double), 3 (MPFR at 128 and 256
bits) and 0 (binary64, MPFR at 512 bits) lanes that were within `1e-5`.

## 4. What the bound does not cover, and binary64

- **Carriers below 106 bits are not tested** (`conditioning::box_gram_applies`,
  `BOX_GRAM_MIN_PRECISION = 106`): the bound acts on double-double, on MPFR
  from 106 bits, and on their Dual shapes. The limit is the larger of two
  derived numbers. The budget `p - 28` must clear the envelope `L` of
  regular boxes: the largest `log2 λ²` of 100 000 flat `g g > g g g` points
  is 18.2, which at rank n+2 of a pentagon predicts `(7 + 1) / 2 · 18.2 =
  72.8` bits and needs `p >= 101`. And the calibration's smallest carrier is
  double-double, 106 bits. In binary64 the budget is `53 - 28 = 25` bits, far
  inside the regular regime: regular boxes reach `log2 λ² = 5 .. 8` (a
  planar 2 -> 3 point: 7.2), so at rank 6 .. 9 `L` reaches 25 .. 40 bits
  while such lanes lose 10 .. 20. With the bound on binary64, the generic
  engine refused one lane each of the rank-n2 controls `rn2-int-mixed-n6-r7`,
  `rn2-int-real-n7-r8-mu2` and `rn2-int-complex-n7-r9` (`DegenerateGram`)
  that it serves within tolerance (`9e-12` scaled), and the planar synthetic
  lane above (served `2e-11` off). MPFR between 54 and 105 bits is
  uncalibrated and left out for the same reason. These carriers keep the
  noise tests of the box geometry and the box mu-expansion bound; on the
  synthetic collinear lanes the generic binary64 carrier serves degenerate
  boxes wrongly from `log2 λ² = 12.7` (tilt `1e-3`: up to `0.4` off; tilt
  `1e-6`: up to `1e20`), a known limit that pyAmpliCol does not hit (its
  binary64 path is the fast engine, below).
- **The fast engine** (`fast/engine.rs`, `fast/box_x2.rs`, binary64) is
  unchanged, for the reason above: a screen with this budget would refuse
  regular boxes at high rank that it serves correctly today, and the
  pyAmpliCol evidence requires its values bit for bit. Its own tests (the
  noise tests of the normal and the box mu-expansion bound) refuse the
  degenerate HC05 points from tilt `1e-4` on and `g g > g g g` from `1e-3`;
  at tilt `1e-3` (largest `log2 λ² = 21.0`) it serves HC05 `0.13` off (and
  `1.3e-6` at `1e-2`). A binary64 screen needs its own calibration (an a
  posteriori decision as for the box mu-expansion,
  `docs/box-mu-conditioning.md`).
- **Point 141 of the HC05 sample of issue #15 is a different defect.** The
  fast engine serves it `9.6e-2` off with status success, before and after
  this change, but the point is regular: every box has `log2 λ² <= 2.1`,
  every final-state pT is above 170 GeV and the smallest `|s_ij| / s` is
  0.074. It is not a degenerate box Gram, and no box-Gram bound can see it.
  pyAmpliCol's default rescue ladder escalates it (double-double serves it
  `3.1e-17` off).
- **A degenerate triangle Gram.** `g g > g g g` with two massless final
  partons collinear: `s34 -> 0` makes the triangle (and bubble) cuts with
  `p3`, `p4` as adjacent legs degenerate. Lanes with regular boxes lose up to
  the whole carrier there; no box test can see it. pyAmpliCol refuses these
  points (every served point was within `1e-26` of the truth).
- **Uncalibrated degeneracies** (section 3): the nearly light-like
  hyperplane (`v.v -> 0`), degenerate boxes in lanes with `n >= 6`, and
  geometries other than the tilted 2 -> 3 family.
- **Diagnostics.** The refusal reuses `CutFailure::DegenerateGram`, which
  the geometry also returns for an exactly singular box; a caller cannot
  tell the two apart from the status.
- **Massive loops in MPFR.** Synthetic lanes with internal masses (10 and
  173 GeV) at tilts `<= 1e-4` (10 GeV) and `<= 1e-2` (173 GeV) fail in MPFR
  with `IntegrationFailed` (master failures): typed refusals, not wrong
  values.

## 5. Fixtures and tests

`validation/conditioning/box-gram` (11 fixtures, one lane geometry and two
or three random tensors each, references from the generic engine at 1800 and
2000 bits): the geometry of the issue #15 points A and B (the HC05 momenta of the issue as
binary64, shifts as exact sums, on-shell external invariants, pentagon lane
with the legs in order, rank 6: `log2 λ² = 99.4` and `103.0`), the collinear
tilt scan with massive external legs at `1e-1, 1e-4, 1e-8, 1e-16` (`n = 5`,
rank 6), `1e-6, 1e-12` (`n = 4`, rank 5) and `1e-10` (`n = 5`, rank 7), and
planar controls at `1e-2, 1e-6`. Every fixture records `box_gram_log2_lambda2`
per box (120 digits). `crates/ninjars-symbolica/tests/box_gram_conditioning.rs`
reduces every lane with the generic engine in binary64, double-double and
MPFR at 128, 200, 256, 266, 400 and 512 bits and requires:

- a typed refusal or agreement within `1e-5` of the fixture's scale, in
  double-double and MPFR (binary64, outside the bound, serves the tilt-`1e-6`
  box lane `1e15` off before and after). This is the wrong-value regression
  (`every_carrier_refuses_or_agrees_with_the_reference`): without the bound
  the engine serves the `collinear-ext-*` lanes up to `1e44` off with status
  success, and the test fails;
- refusal wherever the fixture's predicted loss exceeds the budget by more
  than a bit, and no `DegenerateGram` where it is more than a bit below;
- the issue #15 lanes refused (`DegenerateGram`) at 200, 256 and 266 bits and
  served from 400 bits up. This tests that the predicate fires on the issue's
  geometry; it does not reproduce the wrong values of #15. With random
  tensors on that geometry the engine without the bound serves the lanes
  right (`1.8e-14` at 200 bits, `2e-31` at 256 bits: a loss of about 154
  bits against `L = 348 / 360`); the wrong values of #15 need pyAmpliCol's
  numerators and inputs, and the `collinear-ext-*` fixtures carry the
  wrong-value regression;
- no `DegenerateGram` for the controls in any carrier (passes with or
  without the bound, by design).

Three of the four tests fail with the bound switched off.

`crates/ninjars/src/conditioning.rs` tests the loss model itself.

To re-derive the calibration: generate tilt-scan fixtures (any 2 -> 3 point
with a collinear final state, tilted by `θ`; random tensors of the rank),
reduce them with `examples/box_mu_conditioning` (`--carriers
dd,float:128,float:256,float:512,float:1800,float:2000`) on a build whose
`gram_conditioned` accepts every row, compute `log2 λ²` of each box from the
exact geometry, and fit `p + log2(error)` against it per rank; the margin is
`ceil(16.6 + max(loss - L))` over the lanes with `log2 λ² > 20`.
