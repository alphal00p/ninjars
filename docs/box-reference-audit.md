# Box contraction and reference audit

The reference is original Ninja commit
`05fb0d3a0f47f216eceef411a09ac19e1b9a19f0`, linked against the unchanged
OneLOop 3.4 source bundled with MadGraph 3.7.0. The probes below call those
compiled implementations. No reference production source was modified.

## Master conventions and validation

Original `ninja.cc:318` and `Xninja.cc:213` use one scalar D0 master with
weight `c[0]` and add `-c[4]/6` to the finite rational contribution.
The ordinary box has five residue coefficients; rank n+1 has six.
`MasterContraction::new(4, higher_rank)` preserves these indices. Its input
row contains the coefficient array, six invariants
`[s21,s32,s43,s14,s31,s42]`, and four squared masses in cut-partition order.
There is no basis scalar-product input. The two weight outputs are
`[c0,-c4/6]`; the five contracted outputs retain the existing finite,
single-pole, double-pole, cut-constructible finite, rational order.

The expressions use the existing Symbolica Atom and evaluator facilities.
Both scalar and higher/mixed Dual contraction tests cover the same generated
graphs; this does not extend the supported domains of Dual D0 masters.

`validation/reference/test_box_masters.py` produces
`validation/evidence/box-masters.json`. Six independent cases cover unequal
real masses, unequal complex masses, all-massless off-shell kinematics,
two massive lines, timelike kinematics, and a massless on-shell case with
nonzero single and double poles. Five cases also compare a constant numerator
through original Ninja reduction with its direct OneLOop wrapper. Independent
FeynKit momentum dot products in the Rust test verify the invariant order.
The reference cache is cleared between requests.

`tests/box_masters.rs` checks every Laurent component at scaled relative
tolerance 1e-9 in empty, single, mixed six-row, and permuted 31-row batches,
including a failed mass-domain row between valid branches. Precision tests
compare f64 and DoubleFloat with Float at 160 and 256 bits on real and complex
mass cases. They require DoubleFloat agreement below 1e-25 and Float160/256
agreement below 1e-42, and explicitly verify improvement over binary64.
Complex squared masses use exact fraction strings, avoiding Symbolica's
default finite precision for decimal literals. None of these checks establish
the latency or throughput acceptance gates.

## Prepared box CLI comparison

`validation/reference/test_box_cli.py` regenerates the prepared box reference
bundle from `examples/graphs/box.json`. All three samples use the same prepared
polynomial `1+q0+2*q1*q2+mu2^2`, with the graph factor applied once. Each
sample is evaluated in a fresh original-Ninja process to avoid the changing-point
cache issue below. An independent original TensorNumerator evaluation of
`1+q0+2*q1*q2`, supplemented by the analytic finite contribution `-1/6` from
`mu2^2`, agrees with the generated callback result.

The Rust CLI agrees on all three Laurent coefficients for all three samples in
12 configurations: f64, DoubleFloat, and Float256; expansion and direct sampling;
and batch sizes one and two, including the final one-row tail. Each uses full
box coefficients and sampling scale two. The finite cut-constructible and
rational parts also sum to the finite result. Results and artifact hashes are
recorded in `validation/evidence/box-cli.json`, consumed by the CLI regression
test `box_reduce`. The largest recorded component error divided by
`1+abs(reference)` is `1.34e-14`; binary64 conversion of the recorded outputs
shows identical batch-size-one and batch-size-two results. The higher-precision
CLI comparison is against a binary64 reference, so it does not itself establish
additional accuracy. This checks 36 Rust lane results and makes no performance
acceptance claim.

## Confirmed original full-box reconstruction discrepancy

`validation/reference/test_box_full.py` generates the numerator
`(q dot Vort)*mu2^2` through Symbolica's existing C++ exporter. Here `mu2`
denotes mu squared, `Vfirst=0`, and `Vort` is taken from the original cut.
This is a pure `c[5]=1` residue in Ninja's own convention. The probe calls
the unchanged linked higher-rank `evaluateFullBox` method; a header macro
changes only C++ access checking so the existing coefficient array can be read.

At shifts `[0,(2,1,0,0),(3,0,1,0),(4,0,0,1)]` and unit squared masses, the
original sampling scale is 15. With mu expansion disabled, the original
method returns `c5≈1` and `c3≈14`, although the input has `c3=0`.
At an independent complex q and mu sample, its own residue evaluator gives
approximately `-18.740086-16.721334i`; the generated numerator gives
`-0.679061-1.041982i`. With mu expansion enabled, `c3≈0` and reconstruction
agrees. Full arrays, the sample values, and hashes are recorded in
`validation/evidence/box-full-reconstruction.json`.

This confirms the scale-factor hypothesis in `ninja_implem.cxx:724`:
the positive sample contains `c3 + c5*scale_choice`, but the last update
subtracts only c5. The residual is `c5*(scale_choice-1)`. This concerns a
spurious coefficient in full reconstruction. Both probes retain the correct
c0 and c4 within floating-point error, so it is not evidence of a direct
integrated box discrepancy.

## Reference restrictions affecting comparison policy

OneLOop 3.4's complex box implementation rejects configurations without a
positive candidate Källén discriminant (`avh_olo_foriregi.f90:5689`) by
returning zero. Ninja suppresses the underlying Fortran diagnostic. The
purely spatial unequal-complex-mass case in `exceptional_reference_points`
records this behavior; those zeros are explicitly excluded from correctness
controls. The valid complex-mass control uses timelike shifts.

The original cached D0 keys also omit the fourth invariant s14, called p4
in `avholo_interface.cc:105,131,155`. The six-element cache-key arrays are
initialized with only `[p1,p2,p3,p12,p23]`. The read-only cache probe holds
the masses and other invariants fixed while changing s14 from -9 to -8.
Fresh evaluation changes the finite value from `0.010010797997189838` to
`0.010231290366029707`; retaining the cache incorrectly repeats the first
value. Both results are recorded in `box-masters.json::cache_key_probe`.
Changing-point validation and performance comparisons must clear the original
cache between different kinematic inputs and state that policy explicitly.

## Reproduction

Build the original reference libraries with
`python3 validation/reference/build.py` in the Nix development environment.
Run `python3 validation/reference/test_box_masters.py` for the D0 controls
and cache observation. Build the `generate_reference` example with the
`reference-codegen` feature, then run
`python3 validation/reference/test_box_full.py` for the full-box observation.
The latter invokes Symbolica and should run under the same runtime lock as
Rust tests when using the single-thread community license. Rust tests are
`box_contraction` and `box_masters`, with `RUST_MIN_STACK=134217728` and
`--test-threads=1` as in the other master suites.

For the prepared graph comparison, build `ninjars-cli` with
`--features reference-codegen`, then run
`python3 validation/reference/test_box_cli.py --cli target/debug/ninjars`
under the Symbolica runtime lock. Its output records both reference and Rust
evaluations, rather than timing the setup and reference compilation as reduction
work.
