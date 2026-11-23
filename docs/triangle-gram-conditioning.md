# Degenerate triangle Gram determinants: double-double re-reduction

The fast engine re-reduces in double-double, from the triple cuts down, every
lane whose binary64 reduction it measures to cancel beyond a target relative
accuracy (`FastOptions::cancellation_rescue`, default on;
`FastOptions::rescue_target`, default `DEFAULT_RESCUE_TARGET = 2^-30`; `crates/ninjars/src/fast/engine.rs`, `dd_driver.rs`, `dd.rs`, and
the generated `dd_stages.rs`). Lanes it does not flag keep their binary64
values bit for bit. This note records why, what the re-reduction has to
get right besides precision, and its calibration. It covers the case that
`docs/box-gram-conditioning.md` (section 4) lists as not covered: a degenerate
triangle Gram determinant.

## 1. The defect

pyAmpliCol's profiling campaign (`g g > g g g`, leading colour, massless
loops, `sqrt(s) = 1 TeV`, `pT > 25 GeV`, `ΔR > 0.7`, 10,000 points, pyAmpliCol
v1.0.3 with NinjaRS a2546170) served 122 points below 5 true digits and 491
below 8 in binary64, the worst at -4.9. TIRust on the same points: 7.3 digits
at worst. Every bad point has a partition `{i}{jk}{lm}` of the five gluons with
`x = |s_jk - s_lm| / s < 1.1e-3`: a triangle with one massless leg `p` and two
massive legs `P`, `Q` whose Gram determinant `-(p.K)² ∝ (P² - Q²)²` nearly
vanishes inside phase space. Original Ninja (MadLoop 3.7.0) loses the same
digits.

Per lane (point 308, digits -2.3; the lanes dumped from pyAmpliCol, reduced
here by the fast engine and by the generic engine at 200 bits):

- The triangle's coefficients are large (`c0 ~ 2e13` for a lane whose value
  is 25) and its spurious coefficients enter the subtraction terms of its
  bubbles through its `e3`, `e4`, which grow like `δ^(-1/2)`
  (`δ = (P² - Q²)/P²`). The bubble coefficients reach `1e15 .. 5e21`.
- Inside each bubble, the tensor-bubble terms `c0 B0 + c1 (k.e2) B1 +
  c2 (k.e2)² B11` and the rational term `-(k/6) c9` cancel from `1.5e20` to
  `1e13`: the bubble residue is large on the cut and its integral small.
  Between the cuts, the two bubbles (`±9.7e12`) and the triangle (`1.7e9`)
  cancel again, to 25: the scalar basis's spurious poles in `P² - Q²`.
- The binary64 error of every bad lane is 5 to 50 times `2^-53` times its
  largest bubble coefficient: the intra-bubble cancellation dominates; the
  cross-cut floor `2^-53 Σ|cut|` is 5 to 6 orders lower.

No parametrisation removes this: all points of the triple cut lie at a
Euclidean distance `~ 1/δ` along `p` (`q.p = 0`, `q.K = -K²/2` force
`q = α p + q⊥` with `α ~ 1/(p.K)`), and Ninja's own conditioned bubble
references (`FastOptions::conditioned_cuts`) do not help (measured). The
coefficients themselves are the issue, so the cure computes them, and their
contraction, beyond binary64.

## 2. What the re-reduction has to get right besides precision

Double-double arithmetic alone is not enough. The large terms cancel only for
a geometry in which every identity Ninja's formulas assume holds to the
double-double rounding; three such identities hold only to binary64 rounding
in the binary64 engine, and each one, left as it is, costs as many digits as
the rounding (measured on point 308, lane 33, value `25.4386i`; generic engine
at 200 bits):

1. **Light-like legs.** pyAmpliCol gives the shifts as binary64 sums of
   on-shell momenta and the invariants of on-shell legs exactly
   (`s_matrix = 0` for a massless leg). The shifts' own `p² = 1.5e-11`
   (rounding of the components) is taken as massless by the cut basis
   (`Basis::from_pair`, `|k²| < infrared_eps`) but not by the cut solution, so
   the geometry is inconsistent at `ε`. Even at 200 bits the lane then comes
   out at `-19958i`, `-50450i`, `-5452i` for three perturbations of the shifts
   by `2e-16`; with the legs projected exactly light-like and the invariants
   recomputed from the shifts it is `25.4386i` for all of them (to `1e-15`).
   The re-reduction projects the shifts in double-double onto exact
   light-likeness of every pair with an exactly zero invariant
   (`dd_driver::project_light_like`: minimal-norm Gauss-Newton, two steps).
2. **The bubble reference.** Ninja's `Basis(k)` reference
   `(±1, ∓1/√3, ∓1/√3, ∓1/√3)` is light-like only to binary64 rounding, and the
   basis takes it as light-like. In double-double its energy is the norm of
   its spatial part (`generate_dd_stages.py`, `from_single`).
3. **The invariants of the contraction.** The bubbles' rational terms
   (`-(k/6) c9`, ...) multiply an invariant by a coefficient of order `1e15`;
   with `k` from the input matrix rather than from the shifts the cancellation
   of section 1 breaks at `2^-53 k c9 ≈ 2e4`. The re-reduction takes every
   invariant of its rational terms and of the bubble tensors' reduction from
   the projected shifts.

After these, the re-reduction of lane 33 (double-double coefficients,
binary64 masters) gives `25.4405i` (error `2e-3`, from `2e5`); the remaining
error is the cross-cut floor of binary64 masters, `2^-53 Σ|cut|`
(`2^-53 1.9e13`). Two more pieces remove it where it can be done exactly:

4. **Bubble tensors.** `B1`, `B11`, `B111` are reduced to `B0` and `A0` by
   OneLOop's own Passarino-Veltman relations in double-double
   (`dd_driver::bubble_tensors`, every Laurent order; matches OneLOop to
   `1e-15`), so the intra-bubble cancellation is exact and only `B0`'s
   rounding is left. Applied where the reduction is well conditioned,
   `|p²| >= |m0²| + |m1²|`; otherwise OneLOop's binary64 tensors.
5. **Massless masters.** For massless internal lines, `B0(p²)` and the one-
   and two-mass `C0` are closed forms (QCDLoop triangles 2 and 3), evaluated in
   double-double at the projected geometry's invariants
   (`dd_driver::massless_b0`, `massless_c0`). Their binary64 values at the
   input invariants would leave the floor twice: by their own rounding and by
   the `ε` difference between the input invariants and the geometry's.

With all five the dumped lanes agree with the 200-bit references to
`1e-10 .. 4e-9` (relative `3e-12 .. 3e-11`): lane 33 of point 308 `7e-11`
(binary64: `2.2e5`), lane 47 `1e-10` (`4.5e4`), lane 28 of point 9117 `4e-9`
(`1.9e8`).

## 3. Trigger and re-reduction

After `integrate`, a separate pass sums per lane the moduli of the finite
parts of every integrated term (each coefficient times its master, each
bubble tensor term separately, each rational term; `term_moduli`). A
successful lane whose estimate `2^-53 Σ|terms| / |finite|` exceeds the target
`FastOptions::rescue_target` is re-reduced (default `2^-30`, about four
sevenths of binary64's bits: the measured binary64 error of the degenerate
lanes is 5 to 50 times `2^-53` times their largest term, so a lane served
within the target keeps about seven digits; a caller with an accuracy target
of its own passes it, divided by its safety factor for a lane being one of
several terms of its amplitude). The re-reduced lane is judged the same way:
its own estimate counts `2^-53` for every term that keeps a binary64 factor
(box terms, unselected cuts, binary64 masters) and `2^-104` for the others;
above the target the lane is refused, typed (`FastStatus::Unstable`,
`InstabilityTest::CancellationRescue`, with the estimate and the target),
never served as a silent success. `take_rescues()` lists the evaluation's
re-reduced lanes with their precision (106 bits), estimate and whether they
were served; `take_rescued_lanes()` counts them per thread.

The masters of the selected cuts come in double-double from the provider when
it offers them (`FastMasters::evaluate_rescue`, default not provided;
`OneLoopNativeMasters` evaluates oneloopmaster's generated native masters in
`DoubleFloat`, every sector, massive and complex masses included) at the
geometry's invariants, input invariants given as exactly zero or exactly a
real squared mass of the lane kept exactly; else the massless closed forms of
section 2; else the binary64 masters (counted at `2^-53` in the estimate).
The lane is re-reduced: the default stages and their machine code are unchanged, and
unflagged lanes are not touched. The sum is a measure of the actual
cancellation of the lane, not an a-priori bound: on the campaign's 10,000
points it flags 1.3 % of the lanes, against the a-priori triangle screen of
the earlier diagnosis that refused lanes at 40 % of the points.

Inside a flagged lane only the cuts that carry the cancellation are
recomputed (`select_cuts`): a bubble or triangle whose own term moduli exceed
the lane's allowance `target 2^53 |finite|` divided by twice its number of
cuts,
every parent triangle of a selected bubble whose spurious terms
`|c_k| (E |e3|)^p` (`E` the lane's largest shift and mass scale) exceed the
same bound, and the tadpoles below a selected parent. Every other cut keeps
its binary64 coefficients (promoted) and its binary64 contraction; their
rounding then stays within half the target by construction. On a `g g > g g g`
pentagon lane this recomputes one triangle and three bubbles of ten each
(820 µs -> about 170 µs per re-reduced lane).

The flagged lanes of one call are re-reduced together: their shifts promoted
and projected (section 2), the triangle, bubble and tadpole stages in
double-double (`dd_stages.rs`, generated from the binary64 sources by
`tools/fast/generate_dd_stages.py`: geometry, coefficient-level subtractions,
the tensor numerator's jets, the three stages; binary64 scalars replaced, the
stability checks reduced to finiteness, which the binary64 pass already
tested), with the tensor coefficients promoted exactly
(`FastNumerator::tensor_coefficients`, implemented by `TensorNumerator`).
The contraction (`rescue_with`) takes the binary64 box coefficients and the
binary64 masters of the evaluation's own provider calls (no master is
evaluated again), except for sections 2.4 and 2.5. A lane whose
re-reduction is not finite keeps its binary64 value.

## 4. Results (pyAmpliCol v1.0.3 + this branch, plain binary64 `evaluate`)

Against the campaign's double-double references:

| | points | below 5 digits | below 8 | worst | median | lanes re-reduced |
|---|---|---|---|---|---|---|
| `g g > g g g` LC, v1.0.3 | 10,000 | 122 | 491 | -4.9 | 13.4 | - |
| `g g > g g g` LC, this branch | 10,000 | 0 | 0 | 8.01 | 13.47 | 10,070 (1.3 %) |
| `g g > g g g g` LC, v1.0.3 | 300 | 13 | 48 | -2.34 | 11.04 | - |
| `g g > g g g g` LC, this branch | 300 | 0 | 0 | 8.53 | 11.60 | 4,077 |

Regression (first 500 campaign points each): `g g > g g` LC, `u u~ > e- e+ g`
(complex-mass scheme) and `g g > h g` (top loop) flag no lane and are
unchanged; `g g > t t~ g` re-reduces 54 lanes, its worst point goes from 8.46
to 9.76 digits and two points move from 14.5 / 13.6 to 13.5 / 12.6 digits.

Time: see `REPORT.md` of the work folder (section "Cost").

## 5. Domain and limits

- The option: `FastOptions::cancellation_rescue`, default `true` for
  standalone users (a lane that loses all its digits otherwise comes back
  with status success). With `false` neither the trigger's bookkeeping nor
  the re-reduction runs and every value is the binary64 reduction's, bit for
  bit. pyAmpliCol sets it from its reduction-check switch: on in the checked
  mode, off in the UNCHECKED mode (`fast_options(unchecked)`).

- Rank <= n+1 lanes of numerators that expose their tensor
  (`FastNumerator::tensor_coefficients`): `TensorNumerator`, pyAmpliCol's.
  Others, `MuTensorNumerator` and the rank n+2 engine keep binary64.
- Massive or complex internal masses: the coefficients and the bubble
  tensors' reduction are double-double, the masters binary64; the cross-cut
  floor `2^-53 Σ|cut|` remains there (`g g > t t~ g` above). Three-mass
  massless triangles (from six external legs) also keep the binary64 `C0`.
- Only exactly zero invariants are projected. A massive external leg keeps
  its binary64 shell residue; its masters are smooth there, so the residue
  costs at the cross-cut floor, not more (no threshold branch of the basis
  depends on it).
- The trigger measures the final contraction, so it also flags lanes that
  cancel for other reasons (e.g. a degenerate box); their re-reduction uses
  the binary64 box coefficients and leaves the box part as it was.
- Boxes stay binary64 (their coefficients are moderate on these lanes).

## 6. Tests, fixtures, regeneration

- `crates/ninjars-symbolica/tests/triangle_gram_rescue.rs` (feature
  `native-masters`): the three dumped lanes of
  `validation/conditioning/triangle-gram` are served within `1e-8` of their
  references with exactly one re-reduction each, and the binary64 reduction
  is more than `1e-2` off; the regular control lane is unchanged bit for bit;
  the double-double `B0`, `C0` and bubble tensors agree with OneLOop.
- `crates/ninjars/src/fast/dd.rs` and `dd_driver.rs` unit tests: the
  double-double arithmetic, `exp`, `ln`, and the light-like projection.
- `python3 tools/fast/generate_dd_stages.py` regenerates `dd_stages.rs`;
  `--check` fails while it is stale (after changes to the binary64 stages,
  geometry, subtractions or jets).

## 7. Rescue-precision kinematics from the caller

`FastAmplitude::evaluate_refined(input, refinement, numerator, masters,
results, statuses)` takes an optional `FastRefinement { shifts, s_matrix,
masses_squared }`: the low parts of the rescue-precision kinematics, in the
layouts of `FastInput` (`shifts[lane * n + i]` four components;
`s_matrix[lane * n * n + i * n + j]`, only with an `s_matrix` in `FastInput`;
`masses_squared[lane * n + i]`), so that each rescue-precision value is the
`FastInput` value plus its low part (double-double for this binary64 engine).
`evaluate(..)` is `evaluate_refined(.., None, ..)`. A lazy form,
`evaluate_refined_with(input, Option<&dyn FastRefinementSource>, ..)`, asks
the source only for the lanes the re-reduction takes up
(`FastRefinementSource::lane(lane, shifts_lo, s_matrix_lo, masses_lo) ->
bool`, buffers zeroed on entry; `false` = repairs for that lane; a NaN
invariant low part = that invariant from the refined shifts); `FastRefinement`
implements it. Low parts need not be normalised against the `FastInput`
values: the rescue forms each value by an exact double-double two-sum.

- The binary64 pass reads `FastInput` only, bit for bit as without a
  refinement (`refined_kinematics_drive_the_rescue`).
- With a refinement, the re-reduction of a flagged lane builds its geometry,
  Gram matrices, rational terms and double-double masters from it (invariants
  from the refined `s_matrix`, else from the refined shifts) and applies no
  repair of its own (section 2, items 1 and 3). Without one, those repairs are
  the fallback.
- The caller supplies the rescue-precision cure of the very binary64 point:
  on-shell legs (a zero invariant keeps a zero low part), momentum
  conservation, shifts that are exact sums of the refined momenta, invariants
  consistent with them. An inconsistent refinement degrades the rescue to the
  binary64 level (the test's all-zero refinement: the lanes stay `1e3`
  off), it never touches the binary64 pass.
- The rescue runs after each entry point has published its lanes
  (`rescue_after`), with the refinement passed explicitly; it is read only for
  flagged lanes. Lanes that the rank n+2 engine reduces at a lower rank are
  renumbered and use the repairs.
