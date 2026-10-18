# Rank n+2: theory, review and master-integral artifacts

This directory keeps the derivations and independent checks behind numerator
rank `r = n + 2` (n propagators) in NinjaRS. They are the phase-1 evidence the
implementation follows; nothing here is built or run by `cargo`. The engines'
conventions (coefficient layouts, Laurent slot layouts, `RankClass`) are fixed
in `crates/ninjars/src/topology.rs` and `crates/ninjars/src/fast/numerator.rs`.

Where the theory and the review disagree, the review wins (see "Corrections"
below).

## Contents

### `theory/` (derivation)

| file | content |
|---|---|
| `formulas_rank_n2.json` | Canonical tables: residue monomials per cut with spurious/non-spurious flags, Laurent slot layouts, the level-4 `T2` division index maps, extraction rules, new subtraction terms and integrated values. **The coefficient order of this file is canonical** (bubble c20..c34 included). |
| `subtractions_rank_n2.json` | Closed forms of every new coefficient-level subtraction term, per (parent coefficient, child slot). |
| `subtractions_rank_n2_cse.txt` | The same terms as common-subexpression-eliminated code, one block per kind (`tri_to_bub`, `tri_to_tad`, `bub_to_tad`), meant to be ported mechanically. |
| `subtraction_formulas.py` | The sympy derivation of the subtraction terms (large-`t` series of a parent residue divided by the extra denominators); also validates the rank <= n+1 terms against original Ninja's FORM routines through `ninja_harness/`. |
| `laurent_prototype.py`, `laurent_lib.py` | A 40-digit numerical prototype of the whole Laurent reduction at any rank. |
| `common.py`, `decomposition_check.py` | Exact rational decomposition: completeness and uniqueness of the residue spaces for n = 1..6. (The `common.py` docstring's "r-2n" bound for the 0-point term is stale; see Corrections.) |
| `integrals_check.py` | End-to-end integral check: Feynman-parameter integral of `N/prod D` against the reduced coefficients times masters. |
| `check_formulas_vs_series.py`, `check_cse.py` | The closed forms and the CSE code against the series definition. |
| `build_formulas_json.py` | Rebuilds `formulas_rank_n2.json` and the CSE file. |
| `ninja_harness/harness.cc` | Driver calling original Ninja's FORM-generated `x1cuts` subtractions on stdin inputs; build it against `references/ninja` (headers in `references/ninja/src`, `coefficient_level_subs.cc` compiled in) into `ninja_harness/harness` before running `subtraction_formulas.py`'s Ninja comparison. |
| `logs/` | Decomposition and integral-check logs (the multi-hundred-kB Laurent prototype logs are not kept; rerun `laurent_prototype.py`). |
| `py.sh` | Python with sympy/mpmath/numpy/gmpy2 through `nix shell`. |

### `review/` (independent referee)

| file | content |
|---|---|
| `red.py` | An independent Ninja-style reduction at any rank, exact in F_p[i] (p = 2^61 - 1) or 60-digit mpmath, sharing no code with `theory/`. Includes an independent residue fit used as the oracle. |
| `check_decomp.py`, `check_controls.py` | Completeness and the negative controls (rank n+1 spaces at n+2, missing subtractions, triangle from one orientation). |
| `check_cse.py`, `check_cse_neg.py` | The CSE code of `theory/` against the series definition (0 mismatches over 24 combinations; the perturbed control gives 360). Usage: `check_cse.py ../theory/subtractions_rank_n2_cse.txt`. |
| `slots_fed.py` | Which child slots each new parent coefficient reaches. |
| `box_convention.py` | Shows that the box mu-expansion needs the `beta` offset (`mu² = t² v² - beta`) at rank n+2 for mu²-dependent numerators. |
| `e2e.py`, `collier_hp.py` | End-to-end comparison with the COLLIER reference, and the reference harness rerun with 60-digit preprocessing. Both need the COLLIER reference harness: set `NINJARS_COLLIER_HARNESS` to `collier_reference.py` (default `validation/rank-n2/harness/collier_reference.py`). |
| `hpref/` | Reference values recomputed with 60-digit preprocessing (`collier_hp.py`) for the rank n+2 suite cases; the high-precision targets for the generic engine. |
| `logs/` | Logs of the checks above. |
| `py.sh` | Python with sympy/mpmath/gmpy2 through `nix shell`. |

### `masters/` (integrated values and B1111)

| file | content |
|---|---|
| `verify_masters.py`, `.out` | The new rational terms (box mu⁶, bubble mu²(q·e2)², bubble mu⁴) and B1111 checked three ways: closed form, Feynman parameters at eps = 1e-25, COLLIER UV poles (worst 4.7e-15). |
| `native_rank4_prototype.py`, `.out` | The native rank-4 bubble recipe (OneLOop root representation with the corrected mass-swap sign) against quadrature, <= 1e-31 over 10 sectors. |
| `bnlog_series.py` | Truncation lengths of the bnlog series at rank 4 (17 coefficients for radius 0.2, 9 for 0.01 at 53 bits). |
| `bubble_refs.py` | 50-digit Feynman-parameter bubble references. |
| `exact_b1111_precision.out` | Precision of oneloopmaster's exact (Passarino-Veltman) B111/B1111 in f64, DoubleFloat and Float(256) against those references: unusable in fixed precision at small p². |
| `rationals.py` | sympy derivation of every rational term from the dimension shift. |

## Conventions (summary)

- `D_i = (q + V_i)² - m_i² - mu²`, metric (+,-,-,-). Rank counts q with weight 1 and mu² with weight 2.
- Residue degree on a k-cut is `r - n + k`; at r = n+2 the coefficient counts are pentagon 1, box 7, triangle 21, bubble 35; the integrated tadpole layout keeps 16 entries (`ninjars::residue_coefficient_count`).
- New non-spurious terms: box `c6 mu⁶`; bubble `c20 (q·e2)⁴` (master B1111), `c21 mu² (q·e2)²` and `c22 mu⁴`. Everything else new is spurious. The only new master is B1111.
- Integrated values (finite part, Ninja/OneLOop normalisation):
  - box: `c6 ((m1+m2+m3+m4)/12 - (s12+s13+s14+s23+s24+s34)/60)`;
  - bubble: `c20 ke2⁴ B1111 + c21 ke2² (5 m1 + 15 m2 - 3 s)/60 - c22 (s² - 5 s (m1+m2) + 10 (m1² + m1 m2 + m2²))/60`.
- `B1111 = 1/(5 eps) - int_0^1 x⁴ log(Delta/mu_R²) dx`, `Delta = x m1 + (1-x) m0 - x(1-x) p²`.
- Box mu-expansion on the exact d-dimensional cut: `q = t v0 + v1`, `mu² = t² v0² - beta` with `beta = m_{i1} - q̄0²` (`ExpansionKind::MuShifted`, `FastNumerator::mu_expansion_x2`); `c6 = n0/(v²)³`, `c4 = n2/(v²)² + 3 beta c6`.
- Bubble Laurent level i (power `t^(r-i)`) holds `x^a mu^(2j)` with `a + 2j <= i`, ordered by a then j: level sizes 1, 2, 4, 6, 9 (`fast::t2_len`).

## Corrections (the review takes precedence)

1. **OneLOop B1111 sign.** OneLOop 3.6 and 3.4 negate the mass-swap reconstruction in every branch (p² = 0 and general p²). The correct identity is `B1111 = +(B0' + 4 B1' + 6 B11' + 4 B111' + B1111')` (primes: masses swapped). Never use OneLOop or Ninja's avholo as a B1111 source.
2. **Tadpole basis.** The 0-point remainder depends on the tadpole basis: with Ninja/NinjaRS per-tadpole bases it is non-zero already at r = n+1 (n >= 2). Integrals are unaffected; any global `N = sum Delta prod D` test must use one common tadpole basis.
3. **`SizeHelper<2>`** (`1 + i + i/2 + i/3`) is exact only for levels i <= 3; level 4 has 9 entries, not 8.
4. **Reference precision.** The phase-1 COLLIER suite preprocesses in binary64 and is only good to 1e-16..1e-13 on `-mu2` and shifted lanes; use `hpref/` (60-digit preprocessing) for high-precision checks.
5. **Ninja's mu-expansion convention** (`mu² = t² v²`, no offset) gives a wrong c4 at r = n+2 for mu²-dependent numerators (0.24-1.9 relative, integral off by up to 176x); the `beta` offset is mandatory.
