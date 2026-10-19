# Rank n+2 test data (`tests/rank_n2_integration.rs`, `tests/masters.rs`, `tests/rank_n2_float_bubbles.rs`)

Inputs and references for the V4 smoke test of the generic engine at numerator
rank n+2: the complex-mass `int` lanes of the phase-1 COLLIER reference suite
for n = 1 (with explicit mu² tensors) and n = 2, 3, 4 (with and without).

| file | content |
|---|---|
| `rn2-int-complex-*.json` | The case inputs (`ninjars-bench-case-v1`, optional `mu2_tensors`), copied unchanged from the phase-1 suite (`scratchpad/rank-n2/reference/suite/cases`, generator `gen_suite.py`, seed 20261028). |
| `hpref-rn2-int-complex-n{2-r4,3-r5,4-r6}.json` | COLLIER references for the mu²-free cases, made exactly like `tools/rank_n2/review/hpref` (`tools/rank_n2/review/collier_hp.py`: COLLIER quad, COLI branch, reqacc 1e-20, 60-digit preprocessing). Rerunning the same procedure reproduces the three `-mu2` files of `hpref` byte for byte. |
| `scalar-masters.json` | Exact C0 and D0 (oneloopmaster expressions in 256-bit `SymbolicaMasters`) of every three- and four-line subset of every lane; written by the ignored test `rank_n2_dump_scalar_masters`. |
| `red-rn2-int-complex-*.json` | Independent 60-digit references: the review's reduction (`tools/rank_n2/review/red.py`, sharing no code with NinjaRS; every residue checked against its independent fit, worst recipe-vs-fit 4e-26) assembled as in `tools/rank_n2/review/e2e.py` (A0 closed form, B0..B1111 by Feynman-parameter quadrature) with the exact C0/D0 of `scalar-masters.json`. Produced by `red_reference.py`. |
| `b_family_feynman.txt` | Feynman-parameter finite parts of B0, B1, B11, B111 and B1111 at 200 digits (160 kept) on 220 binary64 rows: p² = 0, \|p²\|/m² down to 1e-14, thresholds, complex masses, the scaleless bubble and the rows of `b1111_rows()`. Read by `tests/masters.rs` (B1111 in binary64, DoubleFloat, `Float(256)` and `Float(512)`; B0..B111 in binary64, DoubleFloat and `Float(128/256/512)`); written by `tools/rank_n2/masters/b_family_reference.py`. |
| `repro-smallp2-bubble-r{4,2}.json` | The rank n+2 final review's reproducers of finding 3 (`references/final-review/repro`, copied unchanged): one bubble lane with p² = 1.86e-9, m² = 25 and 2025, mu_R² = 30, numerator ranks 4 and 2. Read by `tests/rank_n2_float_bubbles.rs` (DoubleFloat and `Float(128)`). |
| `ref-repro-smallp2-bubble-r{4,2}.json` | Their references in `ninjars-reference-collier-v1` form (so `case-check --reference-dir` and the harness read them): the review's 50-digit values (`final-review/repro/ref-scan-bub.json` lane `scan-tiny-uneq`, `ref-scan-bub-r2.json` lane `scan-tiny-uneq-twin-twin`; reduction `tools/rank_n2/review/red.py` at two precisions (`ref.py` default: 60 and 80 digits), A0 closed form, B0..B1111 by quadrature), with the review's uncertainty, its reduction's sum of absolute contributions over the scale as `absum_over_scale` (1.5e7 and 3.2e3) and its new-term parts. |
| `red_reference.py` | `PYTHONPATH=tools/rank_n2/review tools/rank_n2/review/py.sh red_reference.py CASE.json scalar-masters.json OUT.json` |

## Why two references

COLLIER's quad-precision tensor integrals limit the `hpref` values to about
1e-17..1e-14 relative on several lanes (its tadpole coefficients in
particular, review §6.4; a second COLLIER branch, DD, moves the same lanes by
similar amounts). The engine agrees with them exactly as well as the
independent reduction does (n2-r4-mu2: 4.4e-33, 4.3e-24, 7.1e-17 for both),
so the test requires

- <= 1e-25 (scaled by the lane's largest Laurent coefficient) against the
  `red-*` references, and
- a COLLIER difference no larger than the independent reduction's own, plus
  1e-25.
