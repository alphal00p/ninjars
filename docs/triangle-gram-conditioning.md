# Cancelling lanes: whole-lane double-double re-reduction

The fast engine re-reduces in double-double every lane whose binary64
reduction it measures to miss a target relative accuracy
(`FastOptions::cancellation_rescue`, default OFF: on for callers that keep a
test of their own behind it, as pyAmpliCol's checked mode does;
`FastOptions::rescue_target`, default `DEFAULT_RESCUE_TARGET = 2^-30`). The flagged lane is reduced again
WHOLE (box, triangle, bubble and tadpole stages, every master) at one
kinematic point; only the numerator's coefficients stay binary64. Lanes it
does not flag keep their binary64 values bit for bit; with the option off
nothing of this runs. Code: `crates/ninjars/src/fast/engine.rs`
(`rescue_after`, `rescue_cancelling_lanes`, `rescue_with`), `dd_driver.rs`,
`dd.rs`, the generated `dd_stages.rs`. This note records why, what the
re-reduction has to get right besides precision, how a lane is judged, and the
domain.

## 1. The defect

pyAmpliCol's profiling campaign (`g g > g g g`, leading colour, massless
loops, `sqrt(s) = 1 TeV`, 10,000 points, pyAmpliCol v1.0.3) served 122 points
below 5 true digits and 491 below 8 in binary64, the worst at -4.9 (TIRust:
7.3 at worst). Every bad point has a triangle with one massless leg and two
massive legs `P`, `Q` whose Gram determinant `∝ (P² - Q²)²` nearly vanishes
inside phase space. Original Ninja (MadLoop 3.7.0) loses the same digits.

On point 308 (digits -2.3), lane 33 (value `25.44i`): the triangle's
coefficients reach `2e13`, its spurious coefficients enter its bubbles'
subtraction terms, and the bubble coefficients reach `1e15 .. 5e21`. Inside
each bubble the tensor terms and the rational term cancel from `1.5e20` to
`1e13`; between the cuts the bubbles (`±9.7e12`) and the triangle (`1.7e9`)
cancel again, to 25. The binary64 error of every bad lane is 5 to 50 times
`2^-53` times its largest coefficient. No parametrisation removes it (all
points of the triple cut lie at distance `~1/δ`, `δ = (P² - Q²)/P²`), so the
cure computes the coefficients, and their contraction, beyond binary64.

The same mechanism, cancellation between the terms of a lane that binary64
cannot resolve, occurs at nearly degenerate boxes (section 6).

## 2. What the re-reduction has to get right besides precision

The large terms cancel only for a geometry in which every identity Ninja's
formulas assume holds to the double-double rounding (measured on point 308,
lane 33, generic engine at 200 bits):

1. **Light-like legs.** pyAmpliCol gives the shifts as binary64 sums of
   on-shell momenta and the invariants of on-shell legs exactly
   (`s_matrix = 0` for a massless leg). The shifts' own `p² = 1.5e-11` is taken
   as massless by the cut basis but not by the cut solution. Even at 200 bits
   the lane then comes out at `-19958i`, `-50450i`, `-5452i` for three
   perturbations of the shifts by `2e-16`; projected exactly light-like (and
   invariants from the shifts) it is `25.4386i` for all of them. Without a
   caller refinement (section 8) the re-reduction projects the shifts in
   double-double onto exact light-likeness of every pair with a zero invariant
   (`dd_driver::project_light_like`: minimal-norm Gauss-Newton, all pairs
   simultaneously, two steps).
2. **The bubble reference.** Ninja's `Basis(k)` reference is light-like only
   to binary64 rounding; in double-double its energy is the norm of its
   spatial part (`generate_dd_stages.py`).
3. **Invariants.** The rational terms multiply invariants by coefficients of
   order `1e15`; every invariant of the re-reduction (rational terms, masters)
   is that of its own point (the projected shifts', or the caller's refined
   ones), except that an invariant the binary64 pass takes as exactly zero, or
   as exactly a real squared mass of the lane, is exactly that.
4. **One point.** Every master is evaluated at that point, in double-double,
   by the provider (`FastMasters::evaluate_rescue`, section 4). Mixing a
   binary64 master at the input point with double-double coefficients at the
   projected point leaves the cross-cut floor `2^-53 Σ|cut|` (measured `2e-3`
   on lane 33); with all masters at the point the dumped lanes agree with the
   200-bit references to `3e-12 .. 3e-11` relative.

A caller-given invariant strictly between zero and `infrared_eps` is zero by
the engine's convention (cut bases, scaleless bubbles, on-shell masters) but
not by the caller's kinematics: such a lane is not re-reduced (typed
`Failed`, section 5). Invariants NinjaRS computes from the shifts (no matrix
given) are projected instead.

## 3. Trigger

After `integrate`, per successful lane (and per lane the box mu-expansion
screen refused, `BoxMuConditioning`), a binary64 estimate of its relative
error, the largest of:

- `2^-53 Σ|terms| / |finite|`, the moduli of the finite parts of every
  integrated term (each coefficient times its master, each bubble tensor term
  separately, each rational term; `term_moduli`);
- the box-Gram model `2^((r+1)/2 log2 λ² + 11.3 - 53)` of each box (λ² the
  box solutions' spread over the box's scale, `box_lambda_bits`,
  `docs/box-gram-conditioning.md`), the loss inside the box coefficients that
  the contraction does not show;
- `2^-53 mu_error / |finite|`, the box mu-expansion screen's first-order
  rounding estimate (`docs/box-mu-conditioning.md`), also invisible in the
  contraction. Before this term the screen refused a lane only above `2^-17`;
  between the target and `2^-17` binary64 served it unflagged (`g g > t t~ g`
  point 1186, section 7).

A lane whose estimate exceeds `rescue_target` (a lane refused by the screen
always) is re-reduced, if its numerator exposes its tensor
(`FastNumerator::tensor_coefficients`; others keep today's behaviour). The
default `2^-30` keeps about seven digits per lane; a caller with an accuracy
target of its own passes it, divided by its safety factor for a lane being one
of several terms of its point (pyAmpliCol: `reduction_target / 1000`).

## 4. Re-reduction

The flagged lanes of one call are re-reduced together: shifts promoted (and
projected, section 2) or taken from the caller's refinement; box, triangle,
bubble and tadpole stages in double-double (`dd_stages.rs`, generated from
the binary64 sources by `tools/fast/generate_dd_stages.py`: geometry,
coefficient-level subtractions, the tensor numerator's jets, the four stages;
binary64 scalars replaced, the stability checks reduced to finiteness and the
box conditioning screens dropped, the box-Gram measure kept); the tensor
coefficients promoted exactly (`FastNumerator::tensor_coefficients`,
implemented by `TensorNumerator`). A test fails while the generated file is
stale (`dd_driver::tests::dd_stages_are_generated_from_the_binary64_sources`).

Masters: `FastMasters::evaluate_rescue` (default: not provided). 
`ninjars_symbolica::OneLoopNativeMasters` serves, in `DoubleFloat`
(oneloopmaster's generated native evaluators, every sector, real, massive and
complex masses): `A0`, `BubbleRank2` (`B0, B1, B11`), `BubbleRank3`
(`.., B111`), `C0`, `D0`; not `BubbleRank4` (rank n+2 bubbles, outside the
rescue's domain). A master the provider does not serve at the point (not
provided, or refused) is taken from the binary64 pass at the input point and
counted at `2^-53` in the estimate; at a caller-refined point it refuses the
lane instead (two points in one lane). A wrapper provider must forward
`evaluate_rescue` (pyAmpliCol's `MasterCache` does).

## 5. Estimate and decision

The re-reduced lane's own absolute error estimate is the largest of

- `2^-53 M64 + 2^-100 Mdd + 2^-53 |V'|`: `M64` the moduli of its terms with a
  binary64 factor (a binary64 master), `Mdd` the others, each double-double
  `C0` term weighted by `1/δ` (δ the triangle's normalised Källén function
  `sqrt|λ(p1², p2², p3²)| / max|pi²|`): the provider's double-double `C0`
  loses `log2(1/δ)` bits at a nearly degenerate triangle (measured against
  200-bit MPFR by the second audit: `8.6e-27 = 2^-104/δ` at `δ = 5e-6`, for
  massless, massive and complex lines; A0, B0..B111 and D0 agree to `6e-31`).
  `2^-100` is double-double's `2^-104` with four bits of margin;
  `|V'| 2^-53` the rounding of the served binary64 value. Measured: on the
  first audit's synthetic lanes (168) and on the second audit's independent
  set (293; new seed, `δ` to `1e-9`, n = 3 to 6, massless, top, complex Z,
  mixed, W and on-shell massive lines) no served lane exceeds the target;
  the largest error over estimate is 1.01 (first set) and 182 at the `2^-53`
  floor (second set: `2e-14` against `1.1e-16`);
- the box-Gram model at 106 bits times `M' = M64 + Mdd` (relative to `|V'|`
  in the relative estimate, as calibrated);
- `2^-104 mu_error`.

Its relative estimate is the same over `|V'|`. Outcomes
(`Rescue::outcome: RescueOutcome`, the names of the generic engine's
cancellation ladder):

- `Accepted`: the relative estimate meets the target: served;
- `Missed`: it does not: refused. A lane below binary64's resolution of its
  own terms (`|V'| <= 2^-53 M'`) is NOT served on that ground
  (`BelowResolution` is not produced by the fast engine): its value is an
  ordinary part of the point the caller sums, and no absolute scale of the
  point is known to NinjaRS (the second audit: such a rule served lanes
  `1e0 .. 1.5e7` off);
- `Failed`: no re-reduction (a degenerate or non-finite quantity, an
  unsolvable projection, a caller-given invariant below `infrared_eps`).

An `Accepted` lane must also be clearly better
than the binary64 lane: its absolute estimate at most a sixteenth of the
binary64 lane's (the binary64 error ranges over about 0.03 to 1 times its
estimate on the dumped lanes; four bits keep a served value from being worse).
Served: status `Success`, the re-reduced value. Refused (`Missed`, `Failed`):
status `Unstable` with `Instability { test: CancellationRescue, value, bound:
target }`, never a silent success; the result slot holds the BEST AVAILABLE
value, the re-reduced one when it is at one point and its absolute estimate is
below the binary64 lane's, else the binary64 one; `Rescue::held_bits` (53 or
106) and `held_estimate` (= the instability's `value`) say which and how good,
for a caller serving it under a best-effort status. `take_rescues()` lists the
call's re-reduced lanes (precision 106 bits, estimates, outcome, held value);
`take_rescued_lanes()` counts served lanes per thread.

## 6. Boxes

The whole-lane re-reduction covers the box stage itself (double-double box
solutions, coefficients and mu-expansion; double-double `D0`):

- **Nearly dependent momenta / nearly light-like box hyperplane (box Gram).**
  The trigger's box-Gram term flags them; the re-reduced lane's estimate
  carries the same model at 106 bits. Fixtures
  `validation/conditioning/box-gram` (11): lanes within double-double's reach
  are served; the collinear final state tilted by `1e-6` (rank n+1, binary64
  `1e15` off as a success) is beyond it and refused typed
  (`lanes_beyond_double_double_are_refused_typed`).
- **Box mu-expansion (lanes the screen refuses).** Re-reduced and judged with
  the screen's estimate at the double-double unit: `g g > a a g` planar point
  at tilt `1e-9` and the synthetic tilts `1e-2 .. 1e-7` (rank 6) are served
  within `2e-15 .. 6e-13` of exactly consistent references; the planar points
  of `z z z, z z g, z g g, a g g, a a g, W+ W- g` at tilt `1e-12` (double-double
  `1e-8 .. 6e-7` off) are refused typed (estimates `2e-9 .. 3e-6`).
- **Massive box lanes the screen let through** (`g g > t t~ g` point 1186:
  29 lanes `1e-13 .. 2.3e-5` off in binary64, unflagged before the mu term):
  re-reduced to `<= 1.2e-10`, the point from 6.7 to 12.5 digits.
- Mass configurations: every massless, real massive and complex-mass box the
  provider's `D0` serves (all sectors of oneloopmaster). A configuration it
  refuses falls back as section 4 (binary64 master counted at `2^-53`, so the
  lane is refused unless that term is small; at a refined point refused).

## 7. Results

See `REPORT.md` of the work folder for the tables (pyAmpliCol v1.0.3 + this
branch, campaign points against the campaign's double-double references).

## 8. Rescue-precision kinematics from the caller

`FastAmplitude::evaluate_refined(input, refinement, numerator, masters,
results, statuses)` takes an optional `FastRefinement { shifts, s_matrix,
masses_squared }`: the low parts of the rescue-precision kinematics, in the
layouts of `FastInput`, so that each value is the `FastInput` value plus its
low part (an exact double-double two-sum; low parts need not be normalised).
`evaluate(..)` is `evaluate_refined(.., None, ..)`. The lazy form
`evaluate_refined_with(input, Option<&dyn FastRefinementSource>, ..)` asks the
source only for the lanes the re-reduction takes up
(`FastRefinementSource::lane(lane, shifts_lo, s_matrix_lo, masses_lo) ->
bool`, buffers zeroed; `false` = the internal projection for that lane; a NaN
invariant low part = that invariant from the refined shifts).

- The binary64 pass reads `FastInput` only, bit for bit as without a
  refinement.
- With a refinement the flagged lane is re-reduced whole at the refined point
  (geometry, Gram matrices, rational terms and every master), provided the
  refinement is consistent to double-double: every pair the binary64 pass
  takes as exactly zero, or as a real squared mass of the lane, and every
  given invariant, within `2^-100` of the lane's scale `4 max|V|²` of its
  refined shifts; otherwise the lane takes the internal projection (an
  inconsistent refinement, e.g. zero low parts for binary64 momenta, is never
  trusted). Because the whole lane is recomputed there, the refined point may lie
  any distance from the binary64 input that the caller's cure produces; what it
  must be is exactly consistent to double-double: on-shell legs (exactly zero
  invariants stay exactly zero, `lo = 0`), invariants those of the refined
  shifts. The numerator stays at the binary64 point: a lane whose value is
  sensitive to its kinematics (measured on `g g > a a g` lanes: `1e-10 ..
  1e-8` under consistent `2e-16` perturbations) moves with the distance, as it
  would with any binary64 input; the internal projection moves the point by
  about one ulp (minimal norm), the refinement by the caller's cure.
- Lanes that the rank n+2 engine reduces at a lower rank are renumbered and
  use the projection.

## 9. Domain and limits

- Default off. Rank <= n+1 lanes of tensor numerators (`TensorNumerator`, pyAmpliCol's).
  `MuTensorNumerator`, other numerators and rank n+2 lanes keep today's
  behaviour (no trigger, no refusal by the rescue).
- Losses inside a stage that neither the term moduli, the box-Gram model nor
  the mu screen sees are not flagged (time-like synthetic pentagons, `4e-4`;
  the first audit's F3). The trigger divides by the computed value: a lane
  whose binary64 value is dominated by such a stage-internal error can come
  out unflagged (second audit B4: an `8e19` error unflagged at target `1e-8`
  on a time-like rank n+1 synthetic lane with top lines). The rescue is
  therefore not a replacement for a caller's own lane test: pyAmpliCol keeps
  its lane test on in the checked mode.
- "Never worse" is guarded, not proven: the served value's absolute estimate
  must be at most 1/16 of the binary64 lane's ESTIMATE; on top-loop and
  `u u~ > e- e+ g g g` lanes the binary64 value is often far better than its
  estimate, and the second audit measured 1 to 5 % of points losing more than
  0.3 digit against v1.0.3 (none below 7.7 digits; before the `C0` weight).
- The estimate is measured, not proven: margins from the synthetic,
  box-mu, box-gram and dumped lanes (sections 5, 6); `g g > a a g` pentagon
  lanes with a vanishing shift come out `1e-14 .. 9e-14` off with relative
  estimates `1.1e-16` (far within any target).
- Masses' low parts: exact binary64 masses unless the caller refines them.

## 10. Tests, fixtures, regeneration

- `crates/ninjars-symbolica/tests/triangle_gram_rescue.rs` (feature
  `native-masters`): the dumped degenerate lanes served within `1e-8`; the
  regular control unchanged bit for bit; double-double masters against
  OneLOop; box-mu-screened lanes cured; refined kinematics drive the rescue,
  an inconsistent refinement is not trusted, the projection serves callers
  without an invariant matrix; the target decides and misses are typed; lanes
  beyond double-double, and lanes whose estimate misses the target, refused
  typed with the better value in their slot.
- `crates/ninjars/src/fast/dd.rs`, `dd_driver.rs` unit tests: arithmetic, the
  light-like projection, the generated stages' currency.
- `python3 tools/fast/generate_dd_stages.py` regenerates `dd_stages.rs`
  (`--check`: exit 1 while stale).
