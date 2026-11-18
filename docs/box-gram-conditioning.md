# Conditioning bound of the box Gram determinant

The generic engine in double-double and MPFR (not binary64, section 4)
refuses a box cut, typed, when its cut solutions lie so
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
division of the numerator by the box's own structure (section 3 measures the
slope; it does not depend on the number of propagators of the lane). A cut
is refused when

```text
L > p - M,    M = BOX_GRAM_MARGIN_BITS = 28,
```

`p` the carrier's precision (106 for `DoubleFloat`, the bits of an MPFR
`Float`; binary64 is left out, section 4). The kernels compare `(λ²)^(r + 1)` with `2^(2 (p - M))`
in the carrier itself (no conversion to binary64: `λ²` can exceed its
range), NaN and infinity (`v.v = 0`) refused. The quantity uses only the
solved geometry, the shifts and the masses: it costs a few dozen operations
per box cut and does not depend on the numerator. The test only refuses:
every served value is unchanged bit for bit.

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
partons, and massive external legs (30, 40, 125, 50, 60 GeV) for
binary64 and double-double (with massless external legs their IR-divergent
finite parts follow a different master normalisation than MPFR's, see
section 4, so they cannot be compared with the MPFR references). Planar
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

- **Binary64 is not tested** (`conditioning::box_gram_applies`): the bound
  acts on carriers above 53 bits (double-double, MPFR, and their Dual
  shapes). In binary64 its budget `53 - 28 = 25` bits lies inside the regular
  regime, where `L` does not predict the loss: regular boxes reach
  `log2 λ² = 5 .. 8` (a planar 2 -> 3 point: 7.2), so at rank 6 .. 9 `L`
  reaches 25 .. 40 bits while such lanes lose 10 .. 20. With the bound on
  binary64, the generic engine refused one lane each of the rank-n2 controls
  `rn2-int-mixed-n6-r7`, `rn2-int-real-n7-r8-mu2` and `rn2-int-complex-n7-r9`
  (`DegenerateGram`) that it serves within tolerance (`9e-12` scaled), and
  the planar synthetic lane above (served `2e-11` off). Binary64 keeps the
  noise tests of the box geometry and the box mu-expansion bound; on the
  synthetic collinear lanes it serves degenerate boxes wrongly from
  `log2 λ² = 12.7` (tilt `1e-3`: up to `0.4` off; tilt `1e-6`: up to `1e20`),
  a known limit of the generic binary64 carrier that pyAmpliCol does not use
  for these lanes (its binary64 path is the fast engine, below).
- **The fast engine** (`fast/engine.rs`, `fast/box_x2.rs`, binary64) is
  unchanged, for the reason above: a screen with this budget would refuse
  regular boxes at high rank that it serves correctly today, and the
  pyAmpliCol evidence requires its values bit for bit. Its own tests (the
  noise tests of the normal and the box mu-expansion bound) refuse the
  degenerate HC05 points from tilt `1e-4` on and `g g > g g g` from `1e-3`;
  at tilt `1e-3` it serves HC05 `0.13` off (and `1.3e-6` at `1e-2`). On the
  HC05 sample of issue #15 it serves one point `> 1e-5` off before and after
  this change (point 141), which pyAmpliCol's default rescue ladder
  escalates. A binary64 screen needs its own calibration (an a posteriori
  decision as for the box mu-expansion, `docs/box-mu-conditioning.md`).
- **A degenerate triangle Gram.** `g g > g g g` with two massless final
  partons collinear: `s34 -> 0` makes the triangle (and bubble) cuts with
  `p3`, `p4` as adjacent legs degenerate. Lanes with regular boxes lose up to
  the whole carrier there; no box test can see it. pyAmpliCol refuses these
  points (every served point was within `1e-26` of the truth).
- **Massless IR-divergent lanes in binary64 / double-double against MPFR.**
  On the synthetic lanes with massless internal and external lines the
  finite parts of binary64 and double-double agree with each other and
  differ from MPFR's (which agree among 128 .. 2000 bits) by order one, at
  every tilt, while the rational parts agree to every digit: a different
  normalisation of the IR-divergent masters between the carriers, not a
  conditioning effect.
- **Massive loops in MPFR.** Synthetic lanes with internal masses (10 and
  173 GeV) at tilts `<= 1e-4` (10 GeV) and `<= 1e-2` (173 GeV) fail in MPFR
  with `IntegrationFailed` (master failures): typed refusals, not wrong
  values.

## 5. Fixtures and tests

`validation/conditioning/box-gram` (11 fixtures, one lane geometry and two
or three random tensors each, references from the generic engine at 1800 and
2000 bits): the issue #15 points A and B (the HC05 momenta of the issue as
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
  box lane `1e15` off before and after);
- refusal wherever the fixture's predicted loss exceeds the budget by more
  than a bit, and no `DegenerateGram` where it is more than a bit below;
- the issue #15 lanes refused (`DegenerateGram`) at 200, 256 and 266 bits and
  served from 400 bits up;
- no `DegenerateGram` for the controls in any carrier.

`crates/ninjars/src/conditioning.rs` tests the loss model itself.

To re-derive the calibration: generate tilt-scan fixtures (any 2 -> 3 point
with a collinear final state, tilted by `θ`; random tensors of the rank),
reduce them with `examples/box_mu_conditioning` (`--carriers
dd,float:128,float:256,float:512,float:1800,float:2000`) on a build whose
`gram_conditioned` accepts every row, compute `log2 λ²` of each box from the
exact geometry, and fit `p + log2(error)` against it per rank; the margin is
`ceil(16.6 + max(loss - L))` over the lanes with `log2 λ² > 20`.
