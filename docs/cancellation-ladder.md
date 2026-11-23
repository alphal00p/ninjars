# Cancellation ladder of the generic engine

`ninjars_symbolica::cancellation_ladder` re-reduces, in a higher precision
chosen from the measured loss and a target accuracy, every lane of the
precision-generic engine (binary64, double-double, MPFR) whose reduction
measurably cancels beyond that target. It is the generic engine's
counterpart of the fast engine's double-double re-reduction
(`FastOptions::cancellation_rescue`, `docs/triangle-gram-conditioning.md`),
with the same trigger written relative to the carrier, so that a caller who
asks for double-double or MPFR accuracy is rescued one carrier up.

## 1. The defect one carrier up

The degenerate two-mass triangles of `docs/triangle-gram-conditioning.md`
cost the same bits in every carrier: the cancellation is a property of the
lane (bubble coefficients of order `1e20 .. 1e23` whose tensor-bubble terms
cancel inside each bubble and then across cuts, to a lane of order
`1 .. 1e4`), not of the arithmetic. Measured on the dumped lanes of
pyAmpliCol's `g g > g g g` point 4150 (generic engine in double-double, the
lanes made consistent at 512 bits and rounded to double-double): nine of 77
lanes lose 64 to 75 of their 106 bits, their largest term moduli are those of
the bubbles, and their double-double values are `1e-12 .. 7e-10` off. On the
campaign's 10,000 points the double-double result of pyAmpliCol has 12.3 true
digits at worst (18 of the 24 weak points below 16 digits) where a regular
point has about 29. The generic engine's own stability test
(`SymbolicaAmplitudeKernels::with_stability_test`, `EvaluationStatus::Unstable`
beyond `p/2` bits lost) refuses exactly these lanes, but is opt-in.

## 2. The rule

One number drives it: the target relative accuracy `τ` of a lane
(`LadderRule::target`; the caller's, or by default `default_target(p) =
2^-(p/2)`, the generic engine's own `Unstable` level: `1.1e-8` in binary64,
`1.4e-16` in double-double).

- **Measurement** (`lane_loss`): with `M` the summed moduli of the terms
  contracted into the lane (every master times its weight, every bubble
  tensor master and rational term separately; `Amplitude::term_moduli`) and
  `V` their sum, the lane lost `L = log2(M / |V|)` bits (finite part; also the
  measure over the Laurent orders, which sees cancelling poles). Its error
  estimate in a carrier of `p` bits is `ε = 2^-p M / |V| = 2^(L - p)`.
- **Trigger** (`LadderRule::flags`): `ε > τ`. The target, not the base
  engine's own stability test, decides: a lane the base pass refuses
  `Unstable` (more than `p/2` bits lost) whose `ε` meets `τ` is served at its
  base value (`WithinTarget`).
- **Rescue precision** (`LadderRule::rung`): the smallest available carrier
  above the base with `p' >= L + log2(1/τ) + m`, an MPFR rung `L` bits more
  (`mpfr_tracking`: numerica's `Float` consumes about `0.65 L` further bits of
  tracked precision in the reduction; without it nearly every 128-bit rung
  climbs to 192). Available carriers (`Rung`):
  double-double (106 bits) above a smaller base, MPFR at a multiple of 64
  bits from 128 (MPFR's limb: padding to it is free). Hence, at the default
  targets, binary64 -> double-double up to `L = 71`, else MPFR
  `ceil64(2 L + 35)`; double-double -> MPFR `ceil64(2 L + 61)` (192 bits up to
  `L = 65`, 256 up to 97); MPFR `p` -> MPFR `ceil64(2 L + p/2 + 8)`.
- **Margin** `m = 8` bits: the rounding of a cancelling lane is `γ 2^-p M`
  with `γ = 5 .. 50` measured on the degenerate binary64 lanes
  (`ceil(log2 50) = 6`), plus one bit for measuring `L` from the base value
  itself, plus one bit of slack.
- **Re-measurement** (`LadderOutcome`): the rung's own estimate
  `2^(L' - p')` is compared with `τ`, where for an MPFR rung `L'` is at least
  `p' - t`, `t` the tracked precision of the result (`tracked_bits`):
  numerica's `Float` tracks the significant bits of every intermediate and
  rounds to them, so a reduction keeps fewer bits than `p' - L'` (measured: a
  128-bit rung of a lane that loses 64 bits kept about 22 bits, not 64; its
  value was `2.5e-7` off while `2^(L'-p')` claimed `7e-20`; the tracked
  estimate sends it to the 192-bit rung, which serves it within `1e-26`). The
  outcome is `Accepted` (served); `BelowResolution`
  when the rung's value lies below the base carrier's resolution of the
  lane's terms, `|V'| <= 2^-p M'`, with `p' >= p + log2(1/τ)` (a lane that
  vanishes up to rounding: no relative accuracy exists, its absolute error
  `2^-p' M'` is below `τ` times the base pass's; served and reported);
  otherwise the next rung `L' + log2(1/τ) + m` (at most `max_rungs = 3`,
  `p' <= max(4p, 512)`), after which the lane is `Missed`: no value, the
  caller refuses it (`Unstable`). A rung that fails (a cut, a master, a
  non-finite value) is `Failed`, likewise refused. Every rescued lane's
  record (`LaneRescue`) states its base loss and estimate, the target, every
  rung's precision and the outcome; `PolynomialLadder::take_counts` sums
  them per precision.

## 3. What the rescue needs besides precision

- **Consistent kinematics.** Lifting the base inputs exactly keeps their
  inconsistency at `2^-p` (a massless leg's `p² = O(2^-p E²)` against an
  exactly zero invariant), which the lane amplifies like its rounding (the
  fast engine's section 2.1). The ladder projects the rung's shifts, in MPFR
  at the rung's bits, onto `(V_i - V_j)² = s_ij` for every pair whose
  invariant is exactly zero or exactly a real squared mass of the lane
  (`project_shells`: minimal-norm Gauss-Newton), and takes every other
  invariant from the projected shifts. The point moves by `O(2^-p)` relative,
  below the base carrier's resolution of it. A caller with higher-precision
  kinematics passes them as low parts in the base carrier
  (`LadderRefinement`, lazy: asked only for rescued lanes; `SliceRefinement`
  for lane-major slices); the ladder sums `hi + lo` exactly in the rung's
  precision and projects them too (a move below their own precision when
  they are consistent).
- **Numerator.** The rung builds the same `PolynomialNumerator` template in
  its carrier and binds the base coefficients lifted exactly: the numerator's
  own accuracy (`2^-p` of its scale) is kept, the catastrophic amplification
  is removed.
- **Masters.** `SymbolicaMasters` in the rung's carrier (oneloopmaster's
  native double-double and MPFR evaluations), with the base provider's pole
  selection; no on-shell threshold.

## 4. Use

```rust,ignore
let mut amplitude = SymbolicaAmplitudeKernels::new(n, rank, prototype.clone())?
    .with_stability_test() // the measurement; nothing is measured without it
    .into_amplitude(0)?;
let mut ladder = PolynomialLadder::new(n, &numerator)
    .with_rule(LadderRule { target: Some(caller_target), ..LadderRule::default() });
amplitude.reset_batch(lanes);
amplitude.evaluate_batch_with_inputs(input, &mut numerator, &mut cuts, &mut high, &mut masters)?;
for record in ladder.rescue(&amplitude, &input, &parameters, None)? {
    // record.value replaces lane record.lane's value (None: refuse it).
}
```

The stability test does not change any value (it splits the rational terms
for their moduli); without it and without the ladder nothing is measured or
computed (a caller's "checks off").

## 5. Domain and limits

- Every topology and rank the generic engine reduces (massless, massive,
  complex masses): the rung is the same generic engine. Projection only for
  real shifts; complex shifts are lifted as given (their inconsistency then
  limits the rescue to the base level).
- Shell pairs are those with an exactly zero invariant or one exactly equal
  to a real squared mass of the lane; other on-shell legs (an external mass
  that is no internal mass) take the invariant of the projected shifts, a
  move of `O(2^-p)` at a smooth point of their masters.
- Lanes refused before any value exists (box Gram `DegenerateGram`, box
  mu-expansion `SingularParameter`, `docs/box-gram-conditioning.md`,
  `docs/box-mu-conditioning.md`) are not touched: their bounds predict a loss
  that the same rule could take as `L`, once they expose it per lane.
- A lane is judged relative to itself: a lane that is small in its
  amplitude is judged too strictly (time only). A caller who knows the
  amplitude's scale can pass a looser `τ`.

- A measured trigger sees only the loss the base pass can measure. When the
  base inputs are inconsistent far beyond the base rounding (a binary64 lane
  rounded from a consistent point: `p² = O(2^-53 E²)` amplified by `2^70`),
  the base value can be noise as large as the terms themselves and the lane is
  not flagged (observed on the test lanes in binary64). pyAmpliCol's binary64
  lanes take the fast engine (CORE's rescue); only its rare routed lanes reach
  the generic binary64 engine. In double-double, with inputs consistent to
  their rounding (pyAmpliCol's cure), every degenerate lane of the benchmark
  is flagged.
- MPFR: numerica's `Float` rounds every intermediate to its tracked bits; the
  estimate counts them (section 2), at the price of a climb from 128 to 192
  bits for lanes losing about 64-67 bits from a double-double base.
- A refused lane (`Missed`, `Failed`) keeps `LaneRescue::best`: the value with
  the smallest estimate among the base pass and the rungs, with that estimate
  and its carrier's bits, for the caller to hand on wrapped as refused. An
  MPFR-served value carries the ladder's estimate as its tracked precision
  (`LadderCarrier::limit_precision`).

## 6. Tests

`crates/ninjars-symbolica/tests/cancellation_ladder.rs` (feature
`native-masters`): the degenerate lanes of
`validation/conditioning/triangle-gram`, made consistent at 512 bits, in
double-double (MPFR rungs, within `1e-20` of the 512-bit reduction),
every rung width the rule produces (first rungs 128, 192, 256 bits: every
served value within `2^8` times its own estimate), binary64 (served lanes
within target) and MPFR 200 bits (flagged only at a caller's `1e-50`, first
rung 256 bits); a loose target serves the refused lane `WithinTarget`, and the
regular control lane is left alone. Unit tests of the rule and the projection in the module.
`examples/ladder_diag.rs`: the per-lane look of section 1 on any dumped lanes.
