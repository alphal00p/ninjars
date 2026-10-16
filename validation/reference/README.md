# Original Ninja reference runner

Build with `nix develop --command python3 validation/reference/build.py`.
Sources are copied into `target/reference`; reference checkouts stay unchanged.
The runner links the original Ninja against the unmodified OneLOop 3.4 source
distributed with MadGraph. A module-name alias is the only compatibility shim.

On a fresh checkout, bootstrap Ninja with `python3 tools/bootstrap.py` and place
MadGraph 3.7.0 at `references/MG5_aMC_v3_7_0`. Its official Git tag is `v3.7.0`
(commit `adf3d62d096abc1f1512c2101b40218f7c529d4f`) in
`https://github.com/mg5amcnlo/mg5amcnlo.git`. The harness records the SHA-256 of
the actual bundled OneLOop source and compiler identity in `revisions.json`.
Run `python3 validation/reference/test_runner.py` after the build.

The runner accepts successive whitespace-delimited requests on standard input:

1. `propagator_count rank repetitions renormalization_scale_squared`.
2. One row per propagator: `V0 V1 V2 V3 Re(mass_squared) Im(mass_squared)`.
3. `binomial(rank+4,4)` rows: `Re(coefficient) Im(coefficient)`.

Tensor coefficients follow original Ninja ordering. Homogeneous ranks are
concatenated; within a rank enumerate powers of q3, then q2, then q1 in ascending
order, with the remaining power assigned to q0. There are no multinomial or
metric factors: coefficients multiply component monomials literally.
The current tensor protocol does not represent explicit mu-squared monomials.
Such inputs require the Symbolica-generated callback reference adapter.

Each response is a JSON object containing status, epsilon coefficients in
finite/single/double-pole order, rational and cut-constructible finite parts,
and steady-state evaluation timing after one warm-up. Setup and IPC are excluded.
The initial OneLOop license banner can precede the first JSON response; consumers
must select JSON lines. Timing uses repeated kinematics and the original cache.
This alone is not a changing-phase-space benchmark or a performance acceptance
claim. Nonfinite numerical values are emitted as JSON null and must be rejected.
