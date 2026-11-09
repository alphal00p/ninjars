# Conditioning bound of the box mu-expansion

Both engines refuse a lane, typed, when the box mu-expansion of one of its
cuts is predicted to lose more digits than the carrier keeps: the fast engine
with `FastStatus::Unstable`, the generic engine with
`CutOutcome::Unstable(CutFailure::SingularParameter)` (lane status
`CutFailed { cut_index, failure: SingularParameter }`). The bound is
`ninjars::conditioning` (`crates/ninjars/src/conditioning.rs`); this note
derives it and records its calibration. The noise tests of the same
quantities (the fast engine's `FastOptions::stability_threshold` and
`relative_tolerance`, Ninja's `REAL_MIN` and `REAL_EPS`; the generic engine's
`Negligible`) are unchanged and independent of it: the bound stays active with
both thresholds at zero.

## 1. Why a bound

At rank `r = n + k` (`k = 0, 1, 2`) the rational coefficients of a box cut
(`c4`, and `c6` at rank n+2) come from the leading terms of the integrand along
`q = t v + a` on the cut, `v` the box normal. Every uncut line is linear there,
`D_j = d0_j t + d1_j` with `d0_j = 2 v.(V_j - V_first)`, and the expansion
divides the numerator's leading coefficient `n0 = N_r(v)` by `d0_j`:

- rank n: `c4 = n0 / (Π_j d0_j (v.v)²)`;
- rank n+1: `c4` gets `n0 d1 / d0²`;
- rank n+2: `c6 = n0 / (Π_j d0_j (v.v)³)` and `c4` gets `n0 d1² / d0³`.

`d0_j` measures how far line `j` lies from the box's hyperplane. It vanishes
for every box of an `n`-point lane whose shifts span only three dimensions:
at a 2 -> 3 point whose event plane contains the beam, the pentagon Gram
determinant vanishes and so does `d0` of all five boxes, in proportion to the
tilt of the event plane out of the beam axis. The rational part of the lane
stays finite, because the large `c4` of the five boxes cancel (the scalar
pentagon reduces to its boxes), but their rounding errors do not.

Before this bound, `d0_j` and `v.v` were tested for noise only (relative to
`2^10 ε` of their scale): binary64 refused a lane only once `d0_j` reached
rounding level, double-double and MPFR almost never. Loop-induced
`g g > z z z` (and `z z g`, `z g g`, `a a g`, `a g g`, `W+ W- g`) lanes then
came back as successes with relative errors up to `1e17` in binary64 (tilts
`1e-11 .. 5e-7` rad) and `1.3e7` in double-double (tilt `<= 1e-14`).

## 2. The loss model

First-order rounding of the division. `d0_j` is computed from the box normal
`v`, whose components carry a relative error of order `u`, and two shifts;
its absolute error is at most of order `u S_j` with the Euclidean scale

```text
S_j = 2 |v| |V_j - V_first|
```

(`|.|` the Euclidean norm of the four components; the dot product's own
rounding is bounded by the same expression, Cauchy-Schwarz). It is the scale
of the large terms that cancel in `d0_j`: a box normal orthogonal to the beam
and a shift along it give a tiny componentwise product
(`difference_dot_scale`, the noise test's scale) although the normal's own
rounding makes `d0_j` uncertain at `u S_j`.

With `R_j = S_j / |d0_j|` per uncut line (of order 1 at regular kinematics):

- the leading term's relative error is `u Σ_j R_j`;
- its size relative to a regular box's (`|d0_j| ~ S_j`) is
  `ν Π_j R_j ρ^k`, where `ν = |n0| / N̄ <= 1` compares the numerator's
  leading coefficient along the normal with its scale `N̄`, and
  `ρ = Σ_j |d1_j| / (|d0_j| t)` compares the term `n0 d1^k / d0^(k+1)` with
  the regular `n_k / d0` (`n_k ~ n0 t^k`, `t` the distance of the box's
  four-dimensional cut solutions from its center along `v`,
  `t² = |beta / v.v|`); `ρ` grows like `R` at a degenerate lane, is of order 1
  at regular ones, and is taken at least 1;
- the rounding of `n0` and the subleading terms (one division less) add
  `u Π_j R_j ρ^k` relative.

The predicted loss in bits, relative to the regular box, is

```text
L = log2(Π_j R_j) + k log2(max(ρ, 1)) + log2(ν Σ_j R_j + 1),
```

for one uncut line at a degenerate lane (`ρ ~ R`) `L = (k + 1) log2 R +
log2(ν R + 1)`: about `(k + 2) log2 R` for a generic numerator and
`(k + 1) log2 R` when the leading coefficient vanishes together with `d0`
(`ν ~ 1/R`).

**`ν` and parity.** A helicity-summed Born interference of a parity-even
theory is even under the reflection through the event plane at a planar
point, so its leading coefficient along the plane's normal vanishes there
and is `O(tilt)` near it: `ν ~ 1/R`, one division's loss (measured below:
`g g > g g g` at tilt `1e-8` has `ν ~ 8e-8`, tilt `1e-12` has `ν ~ 8e-12`).
A single helicity amplitude (loop-induced `|A|²` reduces per amplitude) has no
such symmetry: `ν ~ 1`, two divisions' loss. This is why Born-interfered
`g g > g g g` was unaffected while loop-induced lanes at the same points were
off by `1e12 .. 1e17`.

`N̄` is the largest modulus of the leading coefficient along three directions
of the box's own hyperplane: the pair sums of the shift differences from the
first cut line (light-like only by accident), each scaled to the normal's
taxicab norm (`N_r` is homogeneous of degree `r`). `ν` is capped at 1, is 0
when the leading coefficient vanishes exactly (no term of the box's top
degree, e.g. a numerator declared above its true rank) and 1 when only the
reference vanishes.

**Refusal.** A lane is refused when any box cut has `L > p - M`, `p` the
carrier's precision (53 for binary64, 106 for `DoubleFloat`, the bits of an
MPFR `Float`) and `M = BOX_MU_MARGIN_BITS`. At rank n the engines test a cheap
upper bound first, `R_j <= 3 S' / |d0_j|₁` with the taxicab scale
`S' = 4 max_{i,μ} |V_i,μ| Σ_μ |v_μ|₁` the fast engine already formed for its
noise test (`S_j <= 2 S'`, `|d0| >= |d0|₁ / sqrt(2)`): regular rows pass it with
one division and two products per uncut line. Only a row over it forms the
Euclidean `R_j` (and above rank n, where `d1` is at hand anyway, `R_j` and
`ρ` are formed directly), and only a row still over the budget evaluates
`N̄` (the fast engine's cold stage; the generic engine's `BoxReducer`
retries the rows its kernels refused with the reference values).

## 3. Calibration

**Calibration set.** Real lanes, not synthetic numerators: pyAmpliCol
(`c4fb3274`) evaluated loop-induced `g g > z z z`, `z z g`, `z g g`, `a a g`,
`a g g`, `W+ W- g` and Born-interfered `g g > g g g` at the stream's planar
points (event plane containing the beam), at the constructed planar points of
the investigation (final energies 400 / 500 / 600 GeV at `sqrt(s) = 1500`)
tilted out of the plane by `0 .. 1e-2` rad, and at regular controls, through a
local probe build of NinjaRS that wrote every binary64 lane it received
(geometry and tensor numerator; never pushed). Each lane was then rebuilt with
its exact geometry (shifts as exact sums of the binary64 external momenta,
on-shell external invariants) and its binary64 tensor, and reduced by NinjaRS
`98e223da` (before the bound) in binary64 (fast engine, also with zero noise
thresholds), double-double and MPFR at 107 and 128 bits, against the generic
engine at 200 / 260 (binary64 runs) or 260 / 340 bits (extended carriers)
agreeing to better than `1e-30`: 105,744 lanes in 1,260 families at 52 points
for binary64, 10,632 lanes in 468 families at 19 points for the extended
carriers. Errors are measured relative to the family's scale (the largest
reference modulus of the family at the point: a suppressed helicity lane is
measured on the scale of its contribution to `|A|²`); the measured loss is
`p + log2(error)`.

**Excluded.** Lens A's planar stream point 44 at tilts `>= 1e-6` (section 4):
there a triangle cut with a small Gram determinant loses about two decades per
numerator rank, independently of the tilt, and dominates the error; the bound
is not tuned to it.

**Predicted against measured** (rank-n lanes with `L >= 30` bits, the
region where the bound acts; slack = `L - measured`; `ν < 1e-3` is the
parity-even regime):

| carrier | rank | regime | lanes | L range (bits) | slack min | 1% | median |
|---|---|---|---|---|---|---|---|
| dd | n | nu<1e-3 | 4083 | 37-78 | 4.0 | 6.6 | 16.6 |
| dd | n | nu>=1e-3 | 5832 | 68-121 | 0.9 | 3.7 | 10.6 |
| fast | n | nu<1e-3 | 13160 | 30-52 | -2.6 | 3.9 | 15.9 |
| fast | n | nu>=1e-3 | 34208 | 30-89 | 1.4 | 4.3 | 10.0 |
| fast-unchecked | n | nu<1e-3 | 19923 | 30-81 | -2.6 | 5.3 | 15.9 |
| fast-unchecked | n | nu>=1e-3 | 36028 | 30-117 | 1.4 | 4.3 | 10.0 |
| float:107 | n | nu<1e-3 | 4152 | 37-78 | -13.4 | 6.2 | 15.6 |
| float:107 | n | nu>=1e-3 | 5832 | 68-121 | 1.1 | 3.7 | 10.8 |
| float:128 | n | nu<1e-3 | 4152 | 37-78 | 3.7 | 6.1 | 15.8 |
| float:128 | n | nu>=1e-3 | 5832 | 68-121 | 2.3 | 3.8 | 10.6 |

`L` overestimates the loss by a slack that depends on how the lane's value
compares with the box's regular `c4` scale: Born-interfered lanes, whose
cut-constructible part dominates, sit 15-20 bits below `L`, loop-induced
amplitudes 4-10 bits. Negative slacks at small `L` are lanes whose error is
not the box mu-expansion's (section 4). Rank n+1 and n+2 have no real lanes in
the set (loop-induced fermion loops have rank n); the synthetic fixtures of
section 5 cover them (refused or within the target).

**The margin.** The operational criterion: no lane that the old engine served
more than `1e-5` (family scale) off stays served. Per carrier, the number of
such lanes served at margin `M` (refusal when `L > p - M`), with the worst
error still served, and the lanes served within `1e-5` before and refused at
the shipped margin:

| carrier | lanes | M = 10 | M = 11 | M = 12 | M = 13 | M = 14 (shipped) | false refusals at 14 |
|---|---|---|---|---|---|---|---|
| binary64 fast | 70,632 | 37 (2.6e-5) | 0 (9.7e-6) | 0 (5.0e-6) | 0 (3.9e-6) | 0 (3.9e-6) | 9,089 |
| binary64 fast, zero noise thresholds | 79,224 | 37 (2.6e-5) | 0 | 0 | 0 | 0 (3.9e-6) | 9,987 |
| double-double | 10,632 | 1 (1.1e-5) | 1 | 1 (1.1e-5) | 1 (1.1e-5) | 0 (4.1e-6) | 441 |
| MPFR 107 bits | 10,632 | 2 (1.5e-5) | 2 | 2 (1.5e-5) | 2 (1.5e-5) | 0 (6.3e-6) | 308 |
| MPFR 128 bits | 10,632 | 16 (1.8e-5) | 7 | 1 (1.2e-5) | 0 (6.2e-6) | 0 (6.2e-6) | 1,448 |

The smallest slack at the budget's edge is 3.4 bits (double-double,
`g g > z z z` point 44 at tilt `1e-13`: `L = 92.9`, error `1.1e-5`); a lane
must keep `log2(1e5) = 16.6` bits; `M = ceil(16.6 - 3.4) = 14`, one margin for
every carrier (the model does not depend on the carrier). With `M = 14` and
one uncut line at rank n a generic numerator is refused below
`|d0| / S = 1.4e-6` in binary64 and `1.4e-14` in double-double
(`R (R + 1) > 2^(p - 14)`), against `1.0e-4` and `1.1e-8` for the ruling's
starting point `|d0| <= 2^(-p/4) S`.

The first version of the bound used the taxicab scale `S'` throughout, with
`R_j` measured against `|d0_j| ~ S'` and `(Σ_j R_j)^k` in place of `ρ^k`.
It calibrated to `M = 12` but refused regular lanes with three uncut lines at
rank n+1 and n+2 (NinjaRS's own random-kinematics tests), whose taxicab `R_j`
is of order 10 to 100 per line; the Euclidean `R_j` and `ρ` are of order 1
there.

**False refusals** (lanes the old engine served within `1e-5`, now refused)
are the bound's cost; the calibration set is concentrated on near-planar
points by construction. Born-interfered `g g > g g g` at tilt `1e-8` stays
served (`L` at most 38.4 bits in binary64); at tilt `1e-12` it is refused in
binary64 (served before with lane errors up to `1.2e-5`); double-double
serves both unchanged.

**Regular points** are unchanged: the bound only adds refusals, never moves a
served value, and at regular points `R` is of order 1 (the regular controls of
the fixtures and the bitwise regression fixtures keep their bits).

## 4. What the bound does not cover

- **A small triangle Gram determinant.** At lens A's point 44 the triangle
  cut `(0, 2, 3)` of pentagon family 12 (shifts `p(2), p(18), p(19), p(27),
  p(31)`) has a Gram determinant of `1.07e-2` of its scale. Its reduction
  loses about two decades per numerator rank, independently of the tilt
  (binary64, tilt `1e-2`: ranks 0 .. 6 off by `3e-15, 4e-15, 3e-12, 5e-10,
  8e-9, 4e-6, 8e-5`; double-double the same seven decades at rank 4), and is
  absent at the stream's regular points. It is the second error source of the
  investigation's tilt scan (`1e-6` .. `1e-2`); the fixtures
  `synthetic-tilt-1e-2-f12-r6` and `-r7` record it (`outside_bound`: binary64
  `8e-5` and `7e-3` off, served). Not the box mu-expansion: below rank n it
  involves none, and its predicted loss is small there.
- **A degenerate box Gram determinant.** `g g > h h h` stream points 29 and 62
  (rank-4 pentagons, no mu-expansion) have a box (lines `1, 2, 3, 4`) whose
  three momenta are linearly dependent up to the inputs' rounding (3x3 Gram /
  scale `7e-34`). Ninja's noise tests of the box normal's factors `p1.e3`,
  `p1.e4` refuse it; with both noise thresholds at zero the lanes come back
  `1e30` off. The conditioning bound does not cover this test.
- **Input inconsistency.** Binary64 external momenta are off-shell (`1e-15`)
  and do not conserve momentum (`1e-13`); with on-shell external invariants
  the shifts and invariants disagree at that level and planar points amplify
  it (converged backends agree to `~5e-11`); consistent inputs agree to
  `1e-60` (the fixture `synthetic-tilt-1e-8-f12-r5-formed`, invariants formed
  from the shifts, NinjaRS at 340 / 420 bits against TIRust at 80 digits:
  `2e-62`).

## 5. Fixtures and tests

`validation/conditioning/box-mu` (55 fixtures, one family and one or two
lanes each, references from the generic engine at 340 and 420 bits agreeing to
`1e-56 .. 1e-101`): `g g > z z z` point 44 tilted by `0, 1e-14, 1e-13,
1e-12, 1e-11, 1e-9, 1e-7, 1e-5, 1e-2`; the constructed planar points of
`z z z, z z g, z g g, a g g, a a g, W+ W- g` at tilts 0 and `1e-12`; `a a g`
at `1e-13 .. 1e-4`; stream points 44, 30, 79, 57, 109 (57 and 109 were `1e30`
off with zero noise thresholds); regular controls (stream points 0 and 1);
`g g > g g g` at tilts `0, 1e-12, 1e-8, 0.3`; synthetic rank n+1 and n+2
numerators at `1e-12 .. 1e-2`; the consistent-input fixture above.
`crates/ninjars-symbolica/tests/box_mu_conditioning.rs` reduces every fixture
lane with the fast engine (default, conditioned cuts, zero noise thresholds;
rank n+2 fixtures take the rank n+2 stages), the generic engine in binary64,
double-double and MPFR at 107, 128 and 200 bits, and requires a typed refusal
or agreement within `1e-5` of the fixture's scale; that the fast engine with
zero noise thresholds refuses the marked fixtures; and that the controls keep
the statuses and binary64 bits of `98e223da`. `crates/ninjars/src/
conditioning.rs` tests the loss model itself. The example
`box_mu_conditioning` reduces any fixture with chosen carriers. Two unit tests
with random elongated kinematics (`fast::tests::conditioned_cuts_are_per_lane_
and_ignored_at_rank_n2`, `fast::tests_x2::rank_n2_batches_and_replay_are_
bitwise`) now accept the bound's typed refusal of a lane (single box lines at
`R ~ 1e4`; served before at `1e-8 .. 2e-5` off) and still compare the batch
and the single-lane evaluations bitwise, statuses included.
