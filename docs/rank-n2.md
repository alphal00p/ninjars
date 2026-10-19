# Numerator rank n+2

NinjaRS reduces one-loop integrands with n >= 1 propagators and numerator rank
up to **n+2**, one rank beyond original Ninja, which stops at n+1
(`references/ninja/src/ninja.cc:121-135`). Both engines support it:

- the generic engine (`ninjars-symbolica`) in binary64, DoubleFloat,
  arbitrary-precision `Float` and Dual carriers;
- the binary64 fast engine (`ninjars::fast`).

n is general in every precision: no part of the algorithm depends on a
maximal n, and the validation suite covers n = 1..7.

This note collects the mathematics, the implementation, the validation and the
known limitations. The derivations and their independent checks are in
[`tools/rank_n2/`](../tools/rank_n2/README.md). The high-precision reference
suite and the tolerance policy are in
[`validation/rank-n2/`](../validation/rank-n2/README.md).

Every number below names its source: a test, a file in the repository, a
commit message, or one of the implementation and verification reports listed
under [Sources](#sources).

## 1. Using rank n+2

| entry point | rank n+2 |
|---|---|
| `SymbolicaAmplitudeKernels::new(n, n + 2, prototype)` (and the Dual variant) | generic engine, every carrier |
| `ninjars::fast::FastAmplitude::new(n, n + 2, options)` | fast engine; `TensorNumerator`, `MuTensorNumerator` and `GenericNumerator` implement the rank n+2 expansions |
| `ninjars reduce`, `bench` (`--engine generic` or `fast`) | as for other ranks |
| `ninjars case-check` | bench cases against high-precision references, any engine and precision (section 8.3) |
| `ninjars case-bench` | reads rank n+2 bench cases; there is no original-Ninja reference for them |
| `ninjars export-bench --rank R` | zero-pads a numerator to a higher declared rank |

`Error::Rank` is raised only above n+2 and says that n+2 is the highest
supported rank (c926841). `export_ninja_cpp` and the CLI's reference export
still reject rank n+2, because original Ninja cannot run it, and they say so.

`ninjars::RankClass { UpToN, NPlus1, NPlus2 }` (excess 0, 1, 2) classifies a
topology:
- `RankClass::of(n, rank)` returns `None` above n+2;
- `Topology::rank_class()` gives the class of a topology;
- `residue_coefficient_count(cut_size, class)` gives the coefficient counts:

| cut | rank <= n | rank n+1 | rank n+2 |
|---|---|---|---|
| pentagon | 1 | 1 | 1 |
| box | 5 | 6 | **7** |
| triangle | 10 | 15 | **21** |
| bubble | 10 | 20 | **35** |
| tadpole (integrated layout) | 5 | 16 | **16** |

## 2. Residues at rank n+2

**Conventions.**
- Denominators are `D_i = (q + V_i)² - m_i² - mu²`, with metric (+,-,-,-).
- Rank counts q with weight 1 and mu² with weight 2.
- On a cut whose first line is i1, `q̄ = q + V_i1`.
- Bases follow original Ninja (`basis.cc`). `e3`, `e4` are null and transverse with
  `e3.e4 = mp34 = -mp12`. On a bubble, `e2` is the reference vector.
- `qe_i = q̄.e_i`. On a box, `x = q̄.v_orth`.

**Degrees.** The k-cut residue has degree `R_k = r - n + k`, which is `k + 2`
at r = n+2.
- There are no residues with k >= 6: five cut conditions fix all of q and mu².
- The remainder after all cuts, `Δ₀ ∏ D_i`, vanishes on every cut and
  integrates to zero, so it is never computed. Δ₀ is a constant for n >= 2 and
  has degree <= 1 at n = 1, with one common tadpole basis.
- These facts were checked by exact decomposition for n = 1..6
  (`tools/rank_n2/theory/decomposition_check.py`, `logs/`; review
  `check_decomp.py`).

The monomials follow Ninja's pattern at every degree. The coefficient order of
`tools/rank_n2/theory/formulas_rank_n2.json` is canonical in both engines.
Existing indices keep their meaning; the new ones are:

| cut | new coefficients | non-spurious |
|---|---|---|
| box | `c6 mu⁶` | **yes**: rational term |
| triangle | `c15 qe3⁵`, `c16 qe4⁵`, `c17 mu² qe3³`, `c18 mu² qe4³`, `c19 mu⁴ qe3`, `c20 mu⁴ qe4` | no |
| bubble | `c20 qe2⁴` | **yes**: master B1111 |
| | `c21 mu² qe2²`, `c22 mu⁴` | **yes**: rational terms |
| | `c23 qe3⁴`, `c24 qe4⁴`, `c25 qe2 qe3³`, `c26 qe2 qe4³`, `c27 qe2² qe3²`, `c28 qe2² qe4²`, `c29 qe2³ qe3`, `c30 qe2³ qe4`, `c31 mu² qe3²`, `c32 mu² qe4²`, `c33 mu² qe2 qe3`, `c34 mu² qe2 qe4` | no |
| tadpole | degree-3 terms, all odd in q̄ | no; not extracted |

The box residue is `c0 + c2 mu² + c4 mu⁴ + c6 mu⁶ + (c1 + c3 mu² + c5 mu⁴) x`.

The tadpole keeps its 16-entry integrated layout: c0, c14 = mu² and
c15 = qe3 qe4, with slot 9 unused. The full rank n+2 tadpole residue has 35
coefficients (theory report), but its new terms integrate to zero.

The only new master integral is **B1111**. The only new rational terms are box
`mu⁶`, bubble `mu² qe2²` and bubble `mu⁴` (section 5).

## 3. Laurent extraction per cut

Each cut is parametrised so that every uncut denominator is a Laurent
polynomial in a large parameter t. The numerator is expanded at large t,
divided by the uncut denominators, and the parent residues are subtracted.
The residue coefficients are then read off the non-negative powers.

A slot is (t power, x power, mu² power). Slots are ordered by decreasing t.

### 3.1 Box: the exact d-dimensional mu-expansion

**The exact cut.** On a box cut the differences `D_i - D_i1` of the four cut
lines are linear in q̄. The cut is therefore the line `q̄ = q̄0 + t v`, where:
- `v = v_orth` is orthogonal to the three cut momenta;
- `q̄0 = (l₊ + l₋)/2` is the midpoint of the two four-dimensional solutions,
  so `q̄0.v = 0`.

The remaining condition `D_i1 = 0` then fixes mu²:

```text
mu² = q̄² - m_i1² = t² v² - beta,    beta = m_i1² - q̄0²     (BoxCut::beta)
```

On this parametrisation `D_i1` is exactly zero. Every uncut line l is linear in
t, with `kk = V_l - V_i1`:

```text
D_l = d0 t + d1,   d0 = 2 v.kk,   d1 = 2 q̄0.kk + kk² + m_i1² - m_l²
```

**Expansion and division.**
1. Expand `N(q(t), mu²(t))` and keep its three leading coefficients n0, n1,
   n2, those of `t^r`, `t^(r-1)` and `t^(r-2)`. The kinds are
   `ExpansionKind::MuShifted { powers: [r, r-1, r-2] }` (vectors `[v, q̄0 - V_i1]`,
   parameter `[beta]`) and `FastNumerator::mu_expansion_x2`.
2. Divide by each uncut line in turn:
   `n0 <- n0/d0`, `n1 <- (n1 - d1 n0)/d0`, `n2 <- (n2 - d1 n1)/d0`.
3. The box residue on the cut, with `x = t v²` and `mu² = t² v² - beta`, has
   `t⁶` coefficient `c6 (v²)³`, `t⁵` coefficient `c5 (v²)³` and `t⁴`
   coefficient `(c4 - 3 beta c6)(v²)²`, because
   `mu⁶ = (t² v² - beta)³ = t⁶ v⁶ - 3 beta t⁴ v⁴ + ...`. Hence:

```text
c6 = n0/(v²)³,    c5 = n1/(v²)³ (spurious),    c4 = n2/(v²)² + 3 beta c6
```

c0 (and c1) come unchanged from the two four-dimensional solutions. Pentagons
need no subtraction here, because their terms reach only up to `t¹`
(review §3).

**Why the offset beta is mandatory.** Original Ninja expands its box at
`mu² = t² v²` (`ExpansionKind::Mu`). That parametrisation is off the cut by
beta. The difference first enters the numerator expansion at relative order
`t^(-2)`, as `-beta [t^(r-2)] ∂N/∂mu²`.
- **Rank <= n+1.** Only n0 and n1 are needed, so Ninja's convention is exact.
- **Rank n+2.** c4 is read from n2. beta enters there twice:
  - in the read-off, as the `3 beta c6` term above;
  - in the numerator's argument, whenever N depends explicitly on mu². FDH
    numerators (`--r2 fdh`) and `MuTensorNumerator` inputs do.

The evidence:
- `tools/rank_n2/review/logs/controls.log`: with the offset dropped from the
  numerator's argument, every box fails the exact fit for mu²-dependent
  numerators (1 of 1, 5 of 5 and 15 of 15 boxes at n = 4, 5, 6). None fails for
  q-only numerators. Only c4 changes.
- Review §3 (`box_convention.py`): on the suite lanes
  `int-{complex,real}-n4-r6-mu2`, c4 is wrong by 0.24-1.9 relative, which moves
  the integral by 0.38-176 times the lane value.
- `rank_n2_box_expansion_needs_the_cut_offset` (`tests/rank_n2_boxes.rs`):
  dropping the offset leaves c4 within 1e-60 for q-only numerators and changes
  it by more than 1e-3 with mu² terms, at n = 4, 5. The S2 report measured 2-3
  relative.

`Mu` keeps its meaning, so every rank <= n+1 plan stays byte-identical.

**Other box modes** (generic engine, `BoxReductionOptions`; `box_kernels.rs`
`interpolation_n2`). Write E and O for the even and odd root combinations at
the pentagon-subtracted mu² sample s.
- **Full coefficients in expansion mode** (tests):
  - `c2 = (E(s) - c0)/s - c4 s - c6 s²`;
  - `c3 = (O(s) - c1)/s - c5 s`.
- **Sampling mode** uses the four levels mu² ∈ {0, s, -s, 2s}, against three
  at rank <= n+1 (`BoxReductionOptions::sample_count`):
  - `c4 = (E(s) + E(-s) - 2c0)/(2s²)`;
  - `c6 = (E(2s) - c0 - (E(s) - E(-s)) - 4 c4 s²)/(6 s³)`;
  - `c2 = (E(s) - E(-s))/(2s) - c6 s²`;
  - `c5 = (O(s) + O(-s) - 2c1)/(2s²)`;
  - `c3 = (O(s) - O(-s))/(2s)`.

### 3.2 Triangle

**Two expansions** share one numerator callback, each at mindeg 5 (12 slots):
- **A:** `q = a + t e3 + (beta_t + mu²)/t · e4/(2 mp34)`;
- **B:** the same with e3 and e4 exchanged.

Here a and beta_t come from the cut conditions as in Ninja. On the cut, an
uncut line is `d0 t + d1 + (d2₀ + d2₁ mu²)/t`. The division needs no parent
subtraction.
- A box residue reaches the triangle only at `t⁰`, through x.
- Its contributions in A and B are proportional to `(e3.v)/(2 e3.kk)` and
  `(e4.v)/(2 e4.kk)` times the same function of mu². Their sum is
  proportional to `v.kk = 0`.
- So the non-spurious c0, c7 and c14 are the **A/B averages** at `t⁰ mu⁰`,
  `t⁰ mu²` and `t⁰ mu⁴` (theory report §5).
- Taking A alone fails on 10 of 10 triangles at n = 5
  (`tools/rank_n2/review/logs/controls.log`).

Slots of either orientation:

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (t, mu²) | (5,0) | (4,0) | (3,0) | (3,1) | (2,0) | (2,1) | (1,0) | (1,1) | (1,2) | (0,0) | (0,1) | (0,2) |

Read-off, with `ι = 1/mp34`:

| from | coefficients |
|---|---|
| A/B average | `c0 = (A[9]+B[9])/2`, `c7 = (A[10]+B[10])/2`, `c14 = (A[11]+B[11])/2` |
| A (qe4 powers) | `c4 = A[6]ι`, `c5 = A[4]ι²`, `c6 = A[2]ι³`, `c13 = A[1]ι⁴`, `c16 = A[0]ι⁵`; `c9 = A[7]ι`, `c11 = A[5]ι²`, `c18 = A[3]ι³`; `c20 = A[8]ι` |
| B (qe3 powers) | `c1 = B[6]ι`, `c2 = B[4]ι²`, `c3 = B[2]ι³`, `c12 = B[1]ι⁴`, `c15 = B[0]ι⁵`; `c8 = B[7]ι`, `c10 = B[5]ι²`, `c17 = B[3]ι³`; `c19 = B[8]ι` |

### 3.3 Bubble

**Orientation A** (mindeg 4, 22 slots):

```text
q = e1eff + x e2eff + t e3 + (p0 + p1 x + p2 x² + mu²)/t · e4/(2 mp34)
```

`e1eff`, `e2eff` and `p0..p2` are as in Ninja's bubble.
- An uncut line is `d0 t + (d1₀ + d1₁ x) + (d2₀ + d2₁ mu² + d2₂ x + d2₃ x²)/t`.
- Level i holds `t^(4-i) x^a mu^(2j)` with `a + 2j <= i`, ordered by a, then j.
- The level sizes are 1, 2, 4, 6, 9. Each level is updated by
  `level_i -= d1·level_(i-1) + d2·level_(i-2)`. The index maps are in
  `formulas_rank_n2.json` (`division_T2_index_maps`); the fast engine uses
  `divide_t2_5`.

| index | slot | index | slot | index | slot |
|---|---|---|---|---|---|
| 0 | (4,0,0) | 8 | (1,0,1) | 16 | (0,1,0) |
| 1 | (3,0,0) | 9 | (1,1,0) | 17 | (0,1,1) |
| 2 | (3,1,0) | 10 | (1,1,1) | 18 | (0,2,0) |
| 3 | (2,0,0) | 11 | (1,2,0) | 19 | (0,2,1) |
| 4 | (2,0,1) | 12 | (1,3,0) | 20 | (0,3,0) |
| 5 | (2,1,0) | 13 | (0,0,0) | 21 | (0,4,0) |
| 6 | (2,2,0) | 14 | (0,0,1) | | |
| 7 | (1,0,0) | 15 | (0,0,2) | | |

After the division the parent triangles are subtracted on all 22 slots
(section 4). Boxes and pentagons need no subtraction: their terms do not reach
`t >= 0` here (review §3, tested at n = 4..6).

**Orientation B** exchanges e3 and e4: e4 grows, and e3 takes the effective
shrinking role.
- It uses mindeg 3 and the 13 slots with t >= 1, i.e. indices 0..12 of the
  table above.
- It is computed only when spurious bubble coefficients are needed (massive
  tadpole lines), as in Ninja's `spurious_terms_needed`.
- Its triangle subtraction uses the same slot formulas with the vectors
  swapped, restricted to t >= 1.

**Read-off.** A coefficient is `Q[t^b x^a mu^(2j)] / (mp12^a mp34^b)`, where
b is the qe4 power from A or the qe3 power from B, with the sign conventions
of `fast/engine.rs` (`formulas_rank_n2.json`, `extraction.bubble`).
- A gives 22 coefficients, among them every non-spurious one:
  - c0 from (0,0,0), c1 from (0,1,0), c2 from (0,2,0), c13 from (0,3,0);
  - **c20** from (0,4,0);
  - c9 from (0,0,1), c10 from (0,1,1);
  - **c21** from (0,2,1), **c22** from (0,0,2).
- B gives the 13 coefficients with a qe3 power.

### 3.4 Tadpole

Two orientations share one callback, each at mindeg 3 (6 slots):
- **A:** `q = -V0 + t e1 + (m² + mu²)/t · e2/(2 mp12)`;
- **B:** `q = -V0 + t e3 + (m² + mu²)/t · e4/(2 mp34)`.

The slots are (3,0) (2,0) (1,0) (1,1) (0,0) (0,1) in (t, mu²).

After the division by the n-1 uncut denominators, the parent triangles and
bubbles are subtracted on slots 4 = (0,0) and 5 = (0,1) only. Then:

```text
c0 = A[4],   c14 = A[5],   c15 = 2 (B[5] - c14)/mp34
```

This is the rank n+1 formula with the index shifted. The odd new monomials
never reach `t⁰`, so they do not contaminate these slots. The review found
c0, c14 and c15 equal to the fitted full tadpole, exactly in F_p[i] for
n = 1..6 (review §3).

## 4. Coefficient-level subtractions

**Definition.** A subtraction term is computed in three steps:
1. write a parent residue on the child cut's Laurent parametrisation;
2. divide it by the parent's extra denominators;
3. keep the polynomial part (`t >= 0`) of the large-t expansion.

These terms are subtracted from the child's quotient.

**Which parents reach which cuts** (theory report §5):

| parent | on the child cut | reaches |
|---|---|---|
| pentagon | `t⁰` | nothing |
| box | `t¹`, only through x | the triangle's `t⁰`, which cancels in the A/B average |
| triangle | `t^R3` | every bubble and tadpole level |
| bubble | `t^R2` | every tadpole level |

**Reuse.** Write X for the coefficient of the shrinking vector, e.g.
`X = p0 + p1 x + p2 x² + mu²` on a bubble, and give t, x and X/t weight 1. A
parent monomial of degree d, divided by m uncut denominators, reaches only
slots of weight <= d - m.
- The rank n+1 triangle and bubble monomials therefore never reach the new
  bubble and tadpole slots.
- So the x2 subtraction equals Ninja's x1 routine with its output re-indexed,
  plus the terms of the new parent coefficients (review §3).

**Slots fed by new coefficients** (`tools/rank_n2/review/slots_fed.py`):
- triangle c15 and c16 feed all 22 bubble slots;
- triangle c17 and c18 feed 7 bubble slots;
- triangle c19 and c20 feed only bubble `(0,0,2)`;
- bubble c20 and c23-c30 feed 6 tadpole slots; c21 and c31-c34 feed 2;
- bubble c22 feeds none.

The closed forms are in `tools/rank_n2/theory/subtractions_rank_n2.json` and
the CSE code in `subtractions_rank_n2_cse.txt`. The review checked them against
the definition with 0 mismatches over 24 combinations; its perturbed control
gives 360.

**Generic engine.** `TriangleBubbleSubtraction::new_class(class)` and
`TadpoleSubtraction::new_class(cut, class)` build the class-aware parent ansatz
(`triangle_residue_class`, `bubble_residue_class`). Symbolica's series then
runs to degree `2 + excess` and `1 + excess` respectively. Old and new parent
terms come out of one series, so no reuse is needed. The old `bool`
constructors remain and produce the same plans as before.

**Fast engine.** `fast/subtractions_series_x2.rs` evaluates the definition
directly:
1. each parent variable `q.e/t` is a truncated jet in `u = 1/t` in the
   generated `tensor_jets` algebra;
2. the parent residue is accumulated by one Horner sweep per variable and
   converted once to Ninja's `(x, mu²)` layout;
3. the sum is divided top-down by the uncut denominators.

The generated closed forms (`fast/subtractions_x2.rs`, from
`tools/fast/generate_subtractions_x2.py`) are Ninja's routines re-indexed plus
the CSE terms. They are compiled for tests only, as the reference. Both
implementations are checked against `validation/rank-n2/subtraction_fixtures.json`,
which is computed from the series definition: 24 cases, worst scaled error
1.24e-15 (series) and 1.26e-15 (closed form). Each of 444 perturbed-coefficient
controls is detected (`fast/tests_x2.rs`, S3 report). The series form brought
the rank n+2 reduction from 2.8-5.8 times its rank n+1 twin down to 2.1-2.9
times (ebdbc00).

**Conditioning.** The new terms divide by up to `(e3.k)⁵` and `(e.k)⁴`
(`pow(K3,-5)` in the CSE code). When an uncut line is nearly orthogonal to the
growing expansion vector, these powers amplify rounding (section 7.2,
`bases_x2`).

## 5. Integrated values

The new terms come from dimension shifting, in Ninja/OneLOop normalisation,
ordered [finite, 1/eps, 1/eps²]. For light-like v and `b >= k - 2`:

```text
I_k[mu^(2b) (q̄.v)^a] = (-1)^(b+1) (b-1)!/(b+2-k)! ∫_simplex (-X.v)^a Δ^(b+2-k)
```

The integral is O(eps) otherwise. In the table, `s` and `m_i` are the cut's
invariants and squared masses in partition order:
- bubbles: `ke2 = (V_i2 - V_i1).e2`, `s = (V_i2 - V_i1)²`, and m1 is the
  origin line;
- boxes: the sums run over the 4 masses and the 6 invariants.

| cut | integrated value (rank n+2 additions in bold) |
|---|---|
| box | `c0 D0 - c4/6` **`+ c6 ((m1+m2+m3+m4)/12 - (s12+s13+s14+s23+s24+s34)/60)`** |
| triangle | `c0 C0 + c7/2 + c14 ((s12+s23+s13)/24 - (m1+m2+m3)/6)` (unchanged) |
| bubble | `c0 B0 + c1 ke2 B1 + c2 ke2² B11 + c13 ke2³ B111` **`+ c20 ke2⁴ B1111`** `+ c9 (3(m1+m2) - s)/6 + c10 ke2 (s - 2m1 - 4m2)/12` **`+ c21 ke2² (5m1 + 15m2 - 3s)/60 - c22 (s² - 5s(m1+m2) + 10(m1² + m1m2 + m2²))/60`** |
| tadpole | `(c0 + c15 mp34 m/4) A0(m) + m²/2 (c14 + c15 mp34/4)` (unchanged) |

```text
B1111 = 1/(5 eps) - ∫₀¹ x⁴ log(Δ/mu_R²) dx,   Δ = x m1 + (1-x) m0 - x(1-x) p²
```

This is OneLOop's coefficient of `p^μ1..p^μ4` in
`∫ q^μ1..q^μ4 / ((q² - m0)((q+p)² - m1))`, with `B_n = (-1)^n ∫ x^n [1/eps - log Δ]`.

The same formula reproduces all of Ninja's existing constants, which fixes the
signs. The new values were checked three ways: closed form, Feynman
parameters at eps = 1e-25, and COLLIER UV poles. The worst relative difference
is 4.73e-15 (`tools/rank_n2/masters/verify_masters.out`).

The existing skip rules stay valid: scaleless bubbles have every new rational
term equal to 0, and massless tadpoles have `A0 = m² = 0`.

The generic engine adds these terms in `MasterContraction::new_class`
(`contraction.rs`), the fast engine in `engine_x2.rs` `integrate`. The fast
engine requests the rank-four bubble family (`FastMaster::BubbleRank4`) only
when `c20 ke2⁴` is not exactly zero.

## 6. B1111

**Native family.** oneloopmaster fe550e3 adds `TensorIntegral::B1111` (commit
fe550e3):
- input `[p², m0², m1², mu²]`; output `[B0, B1, B11, B111, B1111]`
  (5 coefficients, 15 outputs);
- available on every `NativeFloat`, with a binary64 specialisation;
- it is OneLOop's `bub1111`, with the mass-swap reconstruction corrected
  (`B1111 = B0' + 4B1' + 6B11' + 4B111' + B1111'`, primes for swapped masses);
- B0..B111 inside the family are bitwise those of the B111 family.

NinjaRS pins oneloopmaster at 09b2c74, main's pin since d53dd08 (`Cargo.toml`,
`dependencies/manifest.json`). It contains fe550e3; the tensor family is
unchanged since, and its later commits fix C0/D0 (among them the real-mass D0 of
section 9's former known failures).

**OneLOop's sign bug.** OneLOop 3.6 and 3.4 negate the mass-swap sum in all
three swap branches (p² = 0, m0 = 0 and general p²), not only at p² = 0:
- `MG5_aMC_v3_7_0/vendor/CutTools/src/avh/avh_olo.f90:3400, 3424, 3439`;
- `vendor/IREGI/src/oneloop/src/avh_olo_bub.h90:89, 115, 132` (review §6.1).

Their B1111 finite part is wrong whenever the swap branch is taken. Never use
OneLOop or Ninja's avholo as a B1111 source.

The regression tests:
- the swap identity `B_n(p; m0, m1) = (-1)^n Σ_k C(n,k) B_k(p; m1, m0)` holds to
  3.8e-16 on all 617 bubble calls of the recorded Ninja streams
  (`tests/fast_native_masters.rs`, 07395b2); OneLOop's sign would break it by
  `2|B1111|`;
- `B1111(0; 2, 5) = -0.29984750299708588` (OneLOop-like sign: +0.2998...;
  `tools/rank_n2/masters/native_rank4_prototype.out`).

**Binary64 truncation.** At 53 bits rank four needs 9 and 17 series
coefficients at the 0.01 and 0.2 radii of the existing 8- and 16-coefficient
tables. Rank four therefore keeps the tables with its own radii, 0.0085 and
0.17. The B11 and B111 kernels compile to the same instructions as before, and
their outputs are bitwise unchanged (fe550e3).

**Routes per carrier** (`ninjars-symbolica/src/masters.rs`):

| carrier | B1111 from | why |
|---|---|---|
| binary64, DoubleFloat (`SymbolicaMasters`, `OneLoopNativeMasters`) | native family | the exact Passarino-Veltman B1111 divides by `(p²)⁴`: in binary64 the relative error is 5.5e-6 at `p² = m²/256` and 1.89 at `m²/4096` (design §3.1; `tools/rank_n2/masters/exact_b1111_precision.out`) |
| `Float` | native family, each row at the precision its inputs determine relative to the row's scale (bd2022b) | the exact form loses about 5-6 digits per decade of `m²/\|p²\|`, e.g. 1.2e-4 in Float(256) at `\|p²\|/m² = 4e-13`, while its tracked precision overstates the result |
| Dual (any) | exact `oneloop::b1111` | derivatives; small-p² loss as above (section 9) |

**Accuracy.**
- **binary64:** 1.4e-15 of `max_k |B_k|` on the 119-point tiny-momentum grid
  (`fast_native_masters.rs`).
- **binary64 and DoubleFloat:** 9.7e-16 and 3.5e-32 on 31 rows covering every
  sector and the `p² = 2^-4..2^-24` ladder (`tests/masters.rs`, 07395b2).
- **Float(256) and Float(512):** 2.6e-72 and 2.3e-149 against 200-digit
  Feynman-parameter values on 220 rows (bd2022b;
  `tests/rank_n2_data/b_family_feynman.txt` from
  `tools/rank_n2/masters/b_family_reference.py`).
- **Independent check (S4 verifier).** On 2104 rows with both mass orders, all
  interfaces agree bitwise. The binary64 error is at most 2.2e-14, except:
  - within 1e-9 of a threshold, up to 8.7e-10;
  - for equal masses at `p² = -2^-18..-2^-22`, up to 3e-13.

  On those rows the existing native B0..B111 are as wrong or worse, so the
  error comes from the shared bubble kernel.

## 7. Engine architecture

### 7.1 Generic engine: one code path, parameterised by `RankClass`

Every exact plan is built from the `RankClass`, and for `UpToN` and `NPlus1`
the plans are byte-identical to the plans before rank n+2. The V0p test
fingerprints 1,239 plans for n <= 6 and rank <= n+1; all are unchanged.

| component | rank n+2 |
|---|---|
| `Topology`, `LaurentDivision` | guard `rank > n + 2`; `rank_class()`, `rank_excess()` |
| `PolynomialNumerator`, `DualPolynomialNumerator` | `ExpansionKind::MuShifted`, in the truncated integer-polynomial `CutSubstitution` (variables `[t, a.a, a, v, beta]`) |
| boxes (`box_kernels.rs`) | `mu_division_n2` (three-term series, beta in the input row), `interpolation_n2`, 7 coefficients, sampling at 0, s, -s, 2s |
| triangles, bubbles (`projection.rs`) | read-off of c15..c20 and c20..c34 in canonical order |
| subtractions (`subtraction.rs`) | `new_class` constructors (section 4) |
| tadpoles (`tadpole_kernels.rs`) | parent counts by class, two orientations, integrated layout 16 |
| contraction (`contraction.rs`) | `MasterContraction::new_class`: B1111 and the c21, c22, c6 rational terms |
| native kernels (`native_kernels/bubbles.rs`) | no native degree-4 triangle-to-bubble correction: `Ok(None)` keeps the interpreted plan, and `kernel_backends()` reports it |
| pruned swapped arithmetic | tables extended to degree 4 (`bubble_kernels.rs`) |

PR #3's precision-aware singularity checks (`value_checks::Negligible`) guard
every rank n+2 division as well: the rank n+2 box plan shares the
mu-denominator and projection kernels. `--stability-test` and
`with_stability_test()` work at every rank (1524dbd).

### 7.2 Fast engine: a separate rank n+2 engine

`FastAmplitude` holds `x2: Option<Box<RankN2Engine>>`, built in `new` iff
`rank == n + 2`. `evaluate` tests it once and forwards rank n+2 to the cold
`evaluate_x2`. The unchanged rank <= n+1 body is `#[inline(never)]
evaluate_up_to_n1`. The rank n+2 engine is in `fast/engine_x2.rs`:
- It is compiled once against `dyn FastNumerator` / `dyn FastMasters` with its
  own staging types, so downstream instantiations of the rank <= n+1 engine
  compile exactly as before.
- Stage order, skip rules (`cut_stop`, scaleless bubbles, massless tadpoles),
  stability checks, lane compaction, publish-on-success and statuses are those
  of `engine.rs`. Since the merge of main dec3d55 this includes the
  relative-tolerance checks of PR #3 (1524dbd).
- The new numerator methods `t3_expansion_x2` and `t2_expansion_x2` (mindeg 5
  and 4) and `mu_expansion_x2` have defaults, so existing implementors compile
  unchanged.
  - `TensorNumerator` and `MuTensorNumerator` implement them. Their
    `t3_expansion`/`t2_expansion` keep Ninja's depth limits, so that machine
    code is unchanged.
  - `GenericNumerator` serves them through its general expansions and the
    `MuShifted` kind.
  - The recorder, the replayer and the CLI's timed wrapper forward them.
- The `T3K5`, `T2K4` and `Mu3` jets were appended by
  `tools/fast/generate_jets.py`; the existing jets in `tensor_jets.rs` are
  byte-identical.
- The masters are `FastMaster::BubbleRank4` (arity 4, outputs B0..B1111), next
  to D0, C0, BubbleRank2, BubbleRank3 and A0.
- `t2_len` gives the correct rank n+2 level sizes. The rank <= n+1 call sites
  use `t2_len_le3`, Ninja's formula, so their machine code is unchanged
  (61079d4).

**Conditioned bases (`fast/bases_x2.rs`, 79f306a).** Original Ninja fixes the
bubble reference vector and takes the first acceptable tadpole basis. An
uncut line nearly orthogonal to the growing expansion vector (bubble e3,
tadpole e1 or e3) enters through `1/(e.v)^k`, with k <= 5 at rank n+2. On the
headline lanes Ninja's choices scored 0.003-0.03 (`|e.v|/(|e||v|)`), where
the best candidate scores 0.3-0.7.

The rank n+2 engine scores candidates per cut and lane and takes the best one.
Ninja's choice wins ties and is kept when there are no uncut lines.
- **Bubbles:** Ninja's reference plus the 24 signed permutations of
  `(3/4, 1/2, 1/2, 1/4)`, scored by `|k.r|/(|k||r⁰|) · min_v |e3.v|/(|e3||v|)`
  from the Gram determinant.
- **Tadpoles:** Ninja's three bases plus six exactly null rational pairs,
  scored by `min_v min(|e1.v|/|e1|, |e3.v|/|e3|)/|v|`.

Selection is per lane, so a batch is bitwise identical to its lanes run
singly. The effect is measured against Float(256) reductions on the suite and
1260 random lanes, n = 1..7. Consider the lanes whose generic binary64 error
is <= 1e-8. Those outside `max(10 × generic binary64 error, 1e-13)` drop from
486 to 52 of 1399, and the worst ratio from 1.1e13 to 23. About half of the
remaining misses are in the box and triangle stages, which the rank n+1 engine
loses too; the other half are residual bubble and tadpole cancellations.

The bubble reference moves terms between the cut-constructible and rational
parts; only their sum is reference independent. A rank <= n+1 numerator
declared at rank n+2 therefore reproduces the rank <= n+1 engine's Laurent
coefficients, but not its split.

**Zero overhead for rank <= n+1.** Each check names its source:
- **Bitwise results.** V0 (`regression_bitwise`, `validation/regression/r-le-n1-d53dd08.json`,
  generated on a clean checkout of main d53dd08) passes at the integration head.
- **Independent bitwise dump (S3 verifier).** 7484 lane results over 204 cases
  (n = 1..7, ranks 0..n+1, three widths) are byte-identical in hex bits and
  status.
- **Machine code at S3** (normalised disassembly against main e4c16a7, S3
  verifier). Of 85 rank <= n+1 fast-engine and native-masters functions, 58 are
  identical, including `boxes`, `triangles`, `tadpoles`, every subtraction
  routine and `t3_expansion`/`mu_expansion`. The others:
  - the `evaluate` bodies, now `evaluate_up_to_n1`, which lost 1-2 register
    moves;
  - `FastAmplitude::new`;
  - `OneLoopNativeMasters::evaluate`, for the new `BubbleRank4` arm;
  - `integrate`, with two stores swapped and the same instruction count;
  - the `bubbles` and `t2_expansion` bodies changed by `t2_len`, since restored
    to main dec3d55's code by 61079d4.
- **Machine code after 79f306a.** 26281 of 26286 functions are identical to
  the previous commit. The other five are `FastAmplitude::new`, which
  constructs the rank n+2 engine, and rank n+2 code.
- **Instruction counts (S3 verifier).** User-space instructions per point,
  branch against main, on 27 cases at widths 1 and 32: -0.32% to +0.25% in the
  replayed reduction and -0.22% to -0.005% on the live path. The gate was
  0.5%.
- **Wall clock (S3 verifier).** An interleaved A/B over 192 keys, 10 runs per
  side, A/A median spread 0.21%. The geometric mean of all reduction and total
  ratios is 0.9957, and the largest single ratio 1.021 (gates: 1.01 and 1.03).
  The absolute gates against Ninja, OneLOop and MadLoop were unchanged.
- **Masters timing (07395b2).** Native-masters case-bench ratios are 0.997
  geometric mean (max 1.025) for masters and 0.996 (max 1.016) for total.

**Cost of rank n+2** (replayed reduction, width 32, `rn2-int-complex` against
its rank n+1 twin; S3 report, ebdbc00):

| n | rank n+2 (ns/point) | rank n+1 twin (ns/point) | ratio |
|---|---|---|---|
| 2 | 1629 | 783 | 2.08 |
| 3 | 11898 | 4098 | 2.90 |
| 4 | 34317 | 13068 | 2.63 |
| 5 | 75573 | 30022 | 2.52 |
| 6 | 143005 | 59493 | 2.40 |

The S3 verifier measured cycle ratios of 2.18-2.81. The basis conditioning
later added 3.0% (n = 7) to 5.8% (n = 3) instructions to the replayed
reduction and 0.3-1.4% on the live path (79f306a).

## 8. Validation

### 8.1 Reference data

`validation/rank-n2/` (README there) holds:
- **Suite:** 105 cases, 315 lanes, n = 1..7, in the families `int`, `phys`,
  `shifted`, `bub0`, `irml` and `irsoft`, with real, complex, mixed and zero
  masses and numerators with mu² powers up to mu⁸.
- **Synthetic cases:** `syn-n{1..6}-r{n+2}` (18 lanes;
  `validation/benchmark/synthetic-rank-n2/`).
- **Controls:** 205 rank <= n+1 cases (615 lanes): 195 twins on the suite
  kinematics and 10 synthetic benchmark cases. 174 twins have original-Ninja
  results.
- **Identity cases:** 75 exact identity and invariance cases (V2, V6, V7), the
  V5 padding manifest and a conditioning scan (V9).

**How the references were made.** All preprocessing is exact rational
arithmetic. The published value comes from:
- mpmath for n <= 2;
- the review's independent 60-digit reduction for n >= 3 (`red.py`), with
  mpmath A0/B_n and oneloopmaster's exact C0/D0 at 256 bits;
- on a few lanes, COLLIER's COLI.

Every lane is cross-checked against COLLIER 1.2.9 in quad precision: COLI,
and DD for n <= 6. The suite and synthetic lanes split into tiers: 250 A, 67 B and
16 C, where tier C is report only.

COLLIER in quad precision is not a quad-precision reference: its tensor
results are off by up to 5e-10 on suite lanes while it reports 1e-22
(finding 1). It therefore serves as the independent cross-check, not as the
published value.

### 8.2 Tests

V-numbers follow the design. The result column states where each number was
measured.

| id | test (`crates/ninjars-symbolica/tests/` unless noted) | result |
|---|---|---|
| V0 | `regression_bitwise::rank_up_to_n_plus_one_is_bitwise_unchanged` | bitwise against main d53dd08 (`r-le-n1-d53dd08.json`); passes at e116813 |
| V0p | `src/plan_fingerprints_tests.rs::exact_plans_for_rank_up_to_n_plus_one_are_unchanged` | 1,239 fingerprints unchanged; passes at e116813 |
| V1 | `cargo test --workspace` | 8346699, release: default features 389 passed, 0 failed, 12 ignored (47.6 min); `--all-features` 498, 0, 21 (75.4 min) |
| V2a | `rank_n2_laurent` | division recovers known quotients, n = 1..6: 8.7e-77 (Float(256)), 1.8e-15 (binary64); new subtraction terms against the sympy closed forms <= 4e-77, swapped-axis control 1.2 (S2) |
| V2b | `rank_n2_reconstruct` | on-cut reconstruction of every cut size, n = 1..6, with and without mu²: <= 1.1e-73 (Float), <= 5e-13 (binary64); pruned swapped arithmetic identical (S2) |
| V2c | `rank_n2_boxes` | box modes agree, n = 4..6: <= 1.5e-70 (Float); binary64 <= 3e-14 (coefficients), 5e-10 (integrated, n = 6); beta-offset control (S2) |
| V2 (fast) | `crates/ninjars/src/fast/tests_x2.rs` | jets against direct expansion, division multiply-back at NT = 6 and 5, on-cut residue reconstruction, box finite-mu fit, rank-3 tadpole integral, pinch identity, batching and replay bitwise (1d5cb5f) |
| V3 | `tests_x2.rs` (fixtures) | 24 cases, worst 1.24e-15 (series) and 1.26e-15 (closed form); 444 perturbed controls each detected (S3) |
| V4 | `rank_n2_reference::suite_generic_float_matches_references` | Float(256), exact masters: 285 pass, 48 report-only, 0 fail; tier A <= 1.3e-36 at n <= 6, <= 5.9e-30 at n = 7 (integration run, S3 verifier) |
| V4f | `suite_fast_binary64_matches_references`, `suite_generic_binary64_matches_references` | fast 276 pass, 57 report-only; generic f64 275, 58; 0 fail (integration run) |
| V5 | `rank_n2_padding` (`fast_`/`generic_padding_to_rank_n_plus_two`), `fast_rank_n2::fast_rank_n2_padding_reproduces_original_ninja` | rank <= n+1 cases declared at n+2 (`padding-manifest.json`, 238 cases): generic Float within 4.9e-55 on 745 lanes (S5b verifier); fast Laurent coefficients against original Ninja, or within 10 times Ninja's own error of the exact reduction where that error exceeds the tolerance (79f306a) |
| V6 | `rank_n2_identity` | pinch `I_n[D_j M] = I_(n-1)[M]` and constructive decompositions, n = 2..7; Float: rank n+2 sides 5.3e-70 (S5b verifier); binary64 partly ignored (8.4) |
| V7 | `rank_n2_invariance` | shift (V7a) and relabelling (V7b) at rank n+2 in Float(256): 1.1e-69 (S5b verifier); batch width and lane order bitwise (V7c); V7b binary64 at n = 7 ignored (8.4) |
| V8 | `fast_parity::parity_rank_n2_n1..n7` | fast against generic f64 with identical native masters, 1e-9 scaled with a noise model; n = 6 enabled at 8346699 (merge run) |
| V9 | `rank_n2_reference::conditioning_scan_report` | report only (bubble p² ladder, triangle and box Gram ladders) |
| V10 | `rank_n2_dual` | HyperDual against finite differences, n = 2..4 (stand-in masters) and n = 2 (exact masters incl. B1111): <= 5.5e-48 (S2); the S2 verifier found <= 1.6e-44 against central differences at Float(512) |
| V11 | `fast_native_masters`, `masters`, `master_status`, `rank_n2_interfaces`, `rank_n2_float_bubbles` | B1111 accuracy and interface identity (section 6) |
| CLI | `crates/ninjars-cli/tests/case_check.rs` | case-check equals case-bench's reference check; controls with both engines; failures, stale known failures and the engine and precision scope of known-failure entries are reported; the GENERIC_RUNNER interface; a rank n+2 sample; `--stability-test` flags exactly the binary64 cancellations of `rn2-phys-{real,complex}-n6-r8` |

**Independent verification.** Verifier agents re-checked each stream.
- **Generic engine (S2 verifier).** Compared with the review's `red.py` at 100
  digits on 29 problems, n = 1..7, all box modes, the totals agree to
  <= 2e-71. The exceptions are lanes with a `p² = 0` bubble and a massive
  line, at 1e-41..1e-61. That loss is in the exact bubble masters: the same
  lanes agree to <= 7e-91 at Float(512), one lane to 5e-87. In F_p[i] the
  recipe and an independent fit agree exactly on 48 of 58 lanes. The other 10,
  complex-mass lanes with boxes, cannot run there, because `red.py`'s square
  root accepts only F_p elements.
- **Rank <= n+1 in the generic engine (S2 verifier).** 1,428 CLI outputs and
  538 harness jobs are bitwise identical to e4c16a7.

**What V6 cannot see.** For pinch and decomposition numerators, every new rank
n+2 coefficient vanishes identically. In an injected-error experiment, V6 and
V5 passed with a wrong box mu⁶ rational term, while V4 failed 45 of 63 lanes
(S5b verifier). V4 and V8 are the checks of the new integrated terms.

### 8.3 `ninjars case-check` on the whole suite

The command used for these results:

```sh
R=validation/rank-n2
ninjars case-check --engines fast --precision f64 \
  --reference-dir $R/reference --known-failures $R/known-failures.json \
  --twins $R/summary/controls.json --twins $R/summary/synthetic-twins.json \
  --twins $R/summary/n7-controls.json -o report.json \
  $R/cases/*.json validation/benchmark/synthetic-rank-n2/*.json
# and --engines generic --precision f64 | double | float:128 | float:256
```

The results at e116813 (oneloopmaster 09b2c74): 105 suite cases plus 6
synthetic ones, 333 lanes, counted as pass / fail / report-only / known failure
(integration run). No case errors, no unchecked lanes and no stale known
failures occurred, and every one of the five runs exits 0.

| n | lanes | fast (f64) | generic f64 | generic DoubleFloat | generic Float(128) | generic Float(256) |
|---|---|---|---|---|---|---|
| 1 | 36 | 36/0/0/0 | 36/0/0/0 | 36/0/0/0 | 36/0/0/0 | 36/0/0/0 |
| 2 | 57 | 55/0/2/0 | 55/0/2/0 | 57/0/0/0 | 57/0/0/0 | 49/0/8/0 |
| 3 | 45 | 44/0/1/0 | 44/0/1/0 | 45/0/0/0 | 45/0/0/0 | 45/0/0/0 |
| 4 | 57 | 53/0/4/0 | 52/0/5/0 | 57/0/0/0 | 56/0/0/1 | 49/0/8/0 |
| 5 | 57 | 47/0/10/0 | 47/0/10/0 | 52/0/5/0 | 52/0/5/0 | 50/0/7/0 |
| 6 | 57 | 32/0/25/0 | 32/0/25/0 | 41/0/16/0 | 40/0/16/1 | 41/0/16/0 |
| 7 | 24 | 9/0/15/0 | 9/0/15/0 | 9/0/15/0 | 8/0/15/1 | 15/0/9/0 |
| **all** | 333 | **276/0/57/0** | **275/0/58/0** | **297/0/36/0** | **294/0/36/3** | **285/0/48/0** |
| largest checked difference | | 5.3e-8 | 5.2e-7 | 1.4e-11 | 1.4e-11 | 7.0e-12 |
| tier A, n <= 6 / n = 7 | | 9.1e-9 / 5.3e-8 | 6.0e-9 / 3.5e-8 | 1.0e-24 / 1.3e-24 | 7.1e-30 / 1.2e-29 | 1.3e-36 / 5.9e-30 |
| exit status | | 0 | 0 | 0 | 0 | 0 |

The controls (615 rank <= n+1 lanes, same command with `$R/controls/reference`)
give 600/0/15/0 (fast), 599/0/15/1 (generic f64), 600/0/15/0 (DoubleFloat),
588/0/15/12 (Float(128)) and 601/0/14/0 (Float(256)), and also exit 0.

How to read the table:
- **The largest checked differences** sit on tier-B lanes or at n = 7. On
  tier A, Float(256) agrees to <= 1.3e-36 at n <= 6, and its 7.0e-12 is
  `syn-n6-r8` lane 2, whose uncertainty is the input ambiguity of a snapped
  S-matrix.
- **Report-only lanes** have one of four reasons:
  - tier C, among them the 8 n = 7 lanes of the former known failures, whose
    published values are COLLIER's (section 9);
  - in Float(256), a tier-B uncertainty that comes from the generic engine's
    own rank twins (32 `phys` and `syn` lanes);
  - in binary64, DoubleFloat and Float(128), a condition (COLLIER's, or the
    new rank n+2 terms' cancellation `reduction.new_terms_over_scale`) above
    1e6;
  - a tolerance too wide to resolve the lane's smallest new rank n+2 term.
    This is 23 generic and 22 fast binary64 lanes, mostly B1111 parts of 1e-14
    to 3e-6 of the lane (`validation/rank-n2/README.md`).
- **Known failures.** There are no upstream failures since oneloopmaster
  09b2c74. The 3 Float(128) entries are known inaccuracies of the exact masters
  in `Float` (section 9).
- **Against the merge run at 8346699** (oneloopmaster fe550e3): the 9 known
  failures of findings 3 and 5 are gone. `rn2-int-mixed-n6-r8` lane 1 (tier B)
  now passes in every run; the 8 n = 7 lanes are tier C (COLLIER's values) and
  report-only. DoubleFloat no longer reports a stale known failure, and
  Float(128) was not run then.
- **Before the basis conditioning** the fast engine had 266 pass and 9 fail,
  with the worst at 6.2e-4 on `syn-n6-r8` lane 2. It now has 0 fail.

**Binary64 tolerance.** `3 unc + max(1e-14 max(1, cond'), min(A e_ninja, 1e-4))`,
where:
- `cond' = max(condition, new_terms_over_scale)`;
- A = 30 (fast) or 100 (generic f64);
- `e_ninja` is original Ninja's error on the lane. For rank n+2 lanes it is
  taken from the rank twins and extrapolated geometrically by one rank. Where
  OneLOop's own D0 was wrong (findings 3 and 5), `known-failures.json`
  (`ninja_unreliable`) replaces Ninja's errors with its errors against the
  reduction with the same wrong D0, or drops them.

COLLIER's condition alone does not predict the reduction's binary64 loss.
Original Ninja's error does: on 531 control lanes the fast engine stays within
12.6 times it (`validation/rank-n2/README.md`).

### 8.4 Ignored tests at 8346699

| test | failing lanes |
|---|---|
| `rank_n2_reference::suite_fast_within_ten_times_generic_binary64` | 1 of 324: `rn2-shifted-mixed-n7-r9-mu2` lane 0 (condition 7.9e9, report-only in V4f). Fast 1.73e-3 against generic f64 1.57e-4, 1.1 times the 10x limit. Its rank n+1 twin on the unchanged rank <= n+1 path shows the same ratio (fast 3.2e-9, generic 3.1e-10). |
| `rank_n2_identity::rank_n2_integrals_equal_their_pieces_binary64` | 5 of 39: `rn2-id-pinch-n7` lanes 0, 2, 3, 6 at 5.8e-9..1.6e-8 (generic f64 <= 1.4e-10), and `rn2-id-pinch-n4` lane 3 at 5.3e-9 against a tolerance of 3.7e-9 |
| `rank_n2_identity::pieces_reproduce_the_rank_n2_references_binary64` | the same `rn2-id-pinch-n7` lanes through the fast engine's rank <= n+1 path, against generic f64 <= 1.5e-12 |
| `rank_n2_invariance::relabelling_invariance_rank_n2_binary64` | 1 of 84: `rn2-exact-mu-n7-r9-perm` lane 1, fast 8.0e-7 against generic noise 5.0e-8 |

The `rn2-id-pinch-n7` kinematics have propagator pairs whose shift difference
has a zero time component. Those are the bubbles on which the rank <= n+1
fast engine, like original Ninja, loses accuracy.

### 8.5 Reproducing

```sh
cargo test --release --workspace --all-features
cargo test --release -p ninjars-symbolica --features native-masters --test regression_bitwise
cargo test --release -p ninjars-symbolica --features native-masters --test rank_n2_reference
cargo test --release -p ninjars-symbolica --features native-masters --test fast_parity
ninjars case-check validation/rank-n2/cases/rn2-int-complex-n4-r6-mu2.json --engines fast,generic \
  --precision f64,float:256 --reference-dir validation/rank-n2/reference \
  --known-failures validation/rank-n2/known-failures.json \
  --twins validation/rank-n2/summary/controls.json \
  --twins validation/rank-n2/summary/synthetic-twins.json \
  --twins validation/rank-n2/summary/n7-controls.json
```

Source `devenv` (Symbolica license, `RUST_MIN_STACK=134217728`) as for the
other tests. Regenerating the references is described in
`validation/rank-n2/README.md`.

## 9. Known limitations

The two limitations first below are found by the final review of the rank n+2
work and have fixes in preparation; until those land they are current
behaviour.

**DoubleFloat and Float(128) bubbles at small |p²| with unequal masses.** They
return silently wrong results: the status is `Success`, also with
`--stability-test`. The review's case has p² = 1.86e-9 with m² = 25 and 2025
(scaled differences against the exact value; main with 09b2c74 behaves the
same):

| rank | f64 | DoubleFloat | Float(128) | Float(256) |
|---|---|---|---|---|
| 2 (`repro-smallp2-bubble-r2.json`) | 3e-13 | 0.63 | 9.9e-8 | correct |
| 4 (`repro-smallp2-bubble-r4.json`) | 4e-8 | 1.3e13 | 2.0e6 | correct |

- The V9 scan cases `rn2-scan-bubp2-*` show it: the review counted 9 of 16
  lanes failing in DoubleFloat and 11 of 16 in Float(128). The V9 test runs
  them only in Float(256) and binary64, and the scan is not part of the suite
  of section 8.3.
- Cause: only B1111 goes through oneloopmaster's native tensor family
  (section 6). B1, B11 and B111 still use the exact Passarino-Veltman
  formulas, which divide by powers of p². Escalating from f64 to DoubleFloat
  makes the result worse, because binary64 (`OneLoopNativeMasters`, as in the
  fast engine and `case-check`) uses native families for every bubble
  coefficient.
- The bug is older than rank n+2 (rank 2 is affected), but rank n+2 makes it
  worse (rank 4 above).

**`--stability-test` misses cancellation inside one cut contribution.**
`cancellation_stable` compares the result with the sum of whole-cut
magnitudes, so a cut whose cut-constructible and rational parts cancel each
other passes.
- Example (`repro-stability-test-miss-n6-r8.json`, `fr-light-zero-n6-r8`
  lane 2): generic f64 is 24 % wrong and reported `Success`. Its
  cut-constructible part is -0.086, its rational part +0.086, the total 1e-12.
- On the review's fresh lanes, 4 of the 20 binary64 lanes with an error above
  1e-6 go unflagged, and in DoubleFloat 8 of the 13 lanes with an error above
  1e-20.
- The rank n+1 twin of that lane (numerator truncated) is 6e-5 wrong and
  unflagged as well, so the gap predates rank n+2; at rank n+2 the B1111 terms
  reach 6e15 times the lane value.

**Exact masters in Float(128).** `Float` evaluates C0/D0 with Symbolica's
precision-tracking interpreter of oneloopmaster's exact expressions, while
binary64 and DoubleFloat use its native code. At 128 bits that costs up to 10
bits on boxes: on the 139 boxes of the lanes below, D0 is off by a median 471
and at most 1.4e5 units in the last place, against 34 and 4670 in binary64
(C0: at most 4 bits more).
- 3 suite lanes (`rn2-int-mixed-n4-r6` lane 2, `rn2-int-mixed-n6-r8` lane 0,
  `rn2-int-mixed-n7-r9` lane 1) and 12 control lanes miss the binary64 budget
  scaled to 128 bits by 1.0 to 10 times (worst 1.7e-31). They are listed as
  generic Float(128) known inaccuracies in `known-failures.json`.
- From 160 bits on, the references' uncertainty hides this loss.

**Former limitation: oneloopmaster's real-mass D0.** Until oneloopmaster
09b2c74 (main's pin since d53dd08), the exact and native D0 (and OneLOop's)
had the wrong real part on some real-mass boxes, while COLLIER's COLI and DD
agreed with each other:
- finding 3: one box of `rn2-int-mixed-n6-r{5..8}` lane 1, off by 2.1 times
  COLLIER's modulus;
- finding 5: four more boxes in the n = 7 kinematics, off by 51, 9.4, 4.7 and
  3.3 times;
- the final review's finding 1: further real-mass four-mass boxes of fresh
  n = 7 and n = 8 lanes, wrong only in Float(256), Float(512) or Float(1024),
  by up to 51 times (generic Float(256) missed a full lane by 1.14).

With 09b2c74:
- of the 4692 exact scalar C0/D0 of the suite, exactly these five boxes
  change, by their old errors, and the others agree with `scalars/` to
  6.7e-73;
- the 24 known-failure lanes are gone from `known-failures.json`
  (`validation/rank-n2/README.md`, "Known failures"). Their references stay
  COLLIER's (tier B at n = 6, C at n = 7). On `rn2-int-real-n7-r9-muall`
  lanes 0 and 1, COLLIER (condition 1e18) is off by 0.86 and 0.14, and its
  two evaluations disagree with each other by 190 and 12, while Float(256)
  and Float(512) agree to 1e-72;
- the review's reproducer `repro-d0-real.json` agrees with COLLIER in every
  precision, f64 to Float(1024) (to 5.1e-16 and 5.2e-25, COLLIER's own
  spread). The full lanes `fr-phys-real-n7-r9` and `fr8-phys-real-n8-r10`
  agree with the review's references to <= 6.5e-39 in Float(256) and
  Float(512).

**Fast-engine conditioning.** The conditioned bases (section 7.2) apply only
to the rank n+2 engine. The rank <= n+1 path keeps Ninja's choices bitwise.
- The rank n+2 pieces of the n = 7 pinch identities therefore lose accuracy as
  original Ninja does, 5.8e-9..1.6e-8 against <= 1.5e-12 for generic f64.
- One rank n+2 lane, `rn2-shifted-mixed-n7-r9-mu2` lane 0 (condition 7.9e9),
  stays 1.1 times outside 10 times the generic f64 error, like its rank n+1
  twin. See section 8.4.
- Both binary64 engines lose digits on ill-conditioned lanes. On
  `rn2-shifted-mixed-n7-r9-mu2` lane 0 (tier A, condition 7.9e9), the fast
  engine is off by 1.7e-3 and generic f64 by 1.6e-4, against 5.9e-21 in
  DoubleFloat and 1.7e-30 in Float(256) (integration run). For accuracy-critical
  points, use the generic engine in DoubleFloat or `Float`, or the stability
  test.

**Generic binary64.** `rn2-phys-real-n6-r7` lane 2 is off by 4.3e-7, 640 times
original Ninja's error (4.26e-7 with oneloopmaster 09b2c74). It is listed as a
known inaccuracy with limit 1e-6.
- It is not the snapped S-matrix, and `--stability-test` passes it. It is not
  traced.
- The new rank n+2 terms can cancel far beyond COLLIER's condition: 5e7 on
  `rn2-phys-real-n6-r8` lane 2 and 1.4e6 on `rn2-phys-complex-n6-r8` lane 2.
  Generic f64 is off by 1.1e-3 and 5.4e-4 there, Float(256) by about 5e-9.
  `case-check --stability-test` marks both lanes `Unstable` in binary64.

**Master integrals at small `p²`.**
- `Float` and DoubleFloat keep oneloopmaster's exact B1, B11 and B111 (and
  `Float` its exact B0). They lose about 1, 2, 3 and 4 digits per decade of
  `m²/|p²|`; B111 dominates the rank n+2 bubble lanes (91deffa,
  `rank_n2_float_bubbles`). At 128 bits and in DoubleFloat this is the first
  limitation above.
- Dual carriers use the exact B1111, with the small-p² loss of section 6.
- `SymbolicaDualMasters` keeps its existing derivative domains.

**Coverage.**
- In binary64 every n = 7 mu² suite lane is report-only (condition 1e9 to
  1e18), so binary64 checks 9 n = 7 suite lanes.
- The `phys` family at n >= 4 is mostly report-only in Float(256), because its
  uncertainty is the input ambiguity of snapped S-matrices.
- Against original-Ninja references at the default 1e-8, case-check fails 5
  lanes of 4 synthetic cases where Ninja is the inaccurate code (e.g.
  `syn-n6-r6` lane 0 at 0.26). Judge NinjaRS with the high-precision
  references.

**Native kernels.** The `native-kernels` feature has no native degree-4
triangle-to-bubble correction. Rank n+2 bubbles keep the interpreted plan, and
`kernel_backends()` reports it.

**Original-Ninja export.** `export_ninja_cpp` and the CLI's reference export
reject rank n+2, because original Ninja cannot reduce it.

**Performance guard.** The final A/B of design §8 against main dec3d55
(`validation/performance/rank-n2-ab/`, 21a3701, on 8346699 with oneloopmaster
fe550e3) passes every check but one: the geomean of the reduction and total
ratios is 0.9975, the generic `bench` at most 1.019, and the absolute gates
against Ninja hold. `syn-n2-r2` at width 1 (replayed reduction only) is 1.061
slower with the same instruction count per call, a code-placement effect of
the rank n+2 modules in the same codegen unit; its total and width-32 ratios
are 1.003 and 1.007. It was not rerun after the merge of main d53dd08, whose
oneloopmaster pin changes the masters for both sides. A rank n+2 subsection of
`docs/benchmark/benchmark.typ` is not written yet.

## 10. Errata and departures from the design

The design's §10 errata, and how the implementation settled them:

| # | erratum | status |
|---|---|---|
| 1 | OneLOop B1111 sign wrong in every swap branch | corrected in oneloopmaster's native family (fe550e3); swap-identity regression tests in oneloopmaster and NinjaRS (section 6) |
| 2 | Δ₀ depends on the tadpole basis: with per-tadpole bases it is non-zero already at r = n+1 (n >= 2) | integrals unaffected; reconstruction tests check per cut. The `tools/rank_n2/theory/common.py` docstring now states `decomposition_check.zero_point_bound` (e116813) |
| 3 | Ninja's `SizeHelper<2>` (`1 + i + i/2 + i/3`) is exact only for i <= 3; level 4 has 9 entries, not 8 | `t2_len = (i/2+1)(i+1-i/2)` (f7e8bad); `t2_len_le3` keeps Ninja's form at the rank <= n+1 call sites (61079d4) |
| 4 | the phase-1 references preprocess in binary64 (good to 1e-16..1e-13 on `-mu2` and shifted lanes) | regenerated with exact rational preprocessing and exact decimal driver inputs (498432f); the review's `hpref` differs by <= 1.4e-11, explained by its input rounding and COLLIER's error |
| 5a | native B1111 truncation tables | own rank-four radii 0.0085 and 0.17 (fe550e3; the design proposed 0.009 and 0.17) |
| 5b | small-p² loss of the exact B1111 | native family for binary64 and DoubleFloat (07395b2) and for `Float` at the row precision (bd2022b); Dual keeps the exact form |
| 5c | stability of the new divisions | PR #3's precision-aware checks in both engines (1524dbd); V9 scan; fast-engine basis conditioning (79f306a) |
| 6 | the generic-map prototype ordered bubble c20..c34 non-canonically and lacked box, contraction, integration and masters | canonical order (`projection.rs`), the missing parts added (c066178) |
| 7 | rank guards to lift | lifted in `topology.rs`, `laurent.rs`, `fast/engine.rs` and `fast/adapters.rs`. `TensorNumerator` keeps Ninja's mindeg limits and serves rank n+2 through the `*_x2` methods (1d5cb5f). `reference.rs` keeps n+1 and says why (c926841) |

Departures from the design, each recorded in the commit named:
- **Fast engine** (1d5cb5f, ebdbc00):
  - compiled against `dyn` numerator and masters, with its own staging types,
    instead of monomorphic code;
  - new defaulted `*_x2` numerator methods instead of raising the existing
    mindeg limits;
  - subtractions by series division, with the generated closed forms kept as
    the test reference.
- **Basis conditioning** (79f306a) is new. The design kept original Ninja's
  bubble reference and tadpole bases.
- **`Float` B1111** is native at the row precision, not the exact expression
  (bd2022b).
- **V0 baseline.** The fixtures were taken at e4c16a7, not 5f35b07 (96b3653).
  Above rank 5 the generic numerators keep a fixed sixteenth of the monomials.
  The fixture was re-blessed twice, each time for main's own changes and from
  a clean checkout of main (`validation/regression/README.md`):
  - main dec3d55 (PR #3, merged in 1524dbd): the fast engine's
    `TensorNumerator` and `MuTensorNumerator` bits unchanged; generic binary64
    rounding moved by <= 2.1e-12 of the lane maximum, and by <= 1.0e-11 with
    explicit mu²;
  - main d53dd08 (oneloopmaster 264a6ce → 09b2c74, merged in e53c298): C0/D0
    changes move 69 of 237 fast width-1 lanes and 71 of 237 generic binary64
    lanes, median 2e-14, at most 8.8e-5 on the cancelling
    `syn-gghhh-ggHHH-n5-r5` lane 1 (a D0 next to a Gram determinant of 8.2e-8
    moved by 4e-11); no status changed.
- **V4f tolerance.** It is based on original Ninja's error with
  `cond' = max(condition, new_terms_over_scale)`, instead of
  `1e-14 max(1, cond) + 3 unc` (3c6c88c, dda8f8e).
- **V6b** uses a constructive decomposition `N = R + Σ D_i B_i + Σ D_i D_j C_ij`
  instead of the exact rank reduction, whose pieces carry Gram determinants
  (`validation/rank-n2/README.md`).
- **V7/V8 noise** is taken from the generic engine; the fast engine's own
  reversal noise counts only up to a cap (dda8f8e).

## 11. Where things are

| what | where |
|---|---|
| rank class, counts | `crates/ninjars/src/topology.rs` |
| box API (`[T; 3]` mu coefficients, sample counts) | `crates/ninjars/src/boxes.rs` |
| generic plans | `crates/ninjars-symbolica/src/{laurent,projection,subtraction,box_kernels,tadpole_kernels,contraction,integration}.rs` |
| `MuShifted` | `crates/ninjars/src/numerator.rs`, `crates/ninjars-symbolica/src/{polynomial,dual_polynomial}.rs` |
| fast engine | `crates/ninjars/src/fast/{engine_x2,bases_x2,subtractions_series_x2,subtractions_x2}.rs` |
| fast numerators | `crates/ninjars/src/fast/{numerator,tensor,tensor_mu,tensor_jets,adapters,replay}.rs` |
| masters | `crates/ninjars-symbolica/src/{masters,oneloop_native}.rs`; oneloopmaster `src/native/tensor.rs` |
| CLI | `crates/ninjars-cli/src/case_check.rs`, `case_check/harness.rs` |
| derivations, review, master checks | `tools/rank_n2/` |
| fast generators | `tools/fast/{generate_jets,generate_subtractions_x2,make_subtraction_fixtures_x2}.py` |
| references, controls, identities, policy | `validation/rank-n2/` |
| rank <= n+1 regression fixtures | `validation/regression/` |

## Sources

Numbers cite the test or file that produces them, or the commit whose message
records them. Some come only from the implementation and verification reports
of the rank n+2 work (2026-10-18/19), which are not in the repository:

| report | content |
|---|---|
| design | the implementation design (decisions D1-D9, §1-§10) |
| theory, review | phase-1 derivation and independent referee; their artifacts are in `tools/rank_n2/` |
| S0, S2, S3, S4, S5b | stream reports; S2 → c066178, S3 → 1d5cb5f and ebdbc00, S4 → 07395b2, S5b → 3c6c88c |
| S2, S3, S4, S5b verifiers | independent re-checks of each stream |
| merge run | the integration head 8346699: workspace tests and `case-check` on the whole suite in four configurations |
| final review | the review of 8346699 that found the arbitrary-precision D0 bug and the limitations of section 9 (reproducers named there) |
| integration run | e116813, after merging main d53dd08 (oneloopmaster 09b2c74): V0 regenerated, known failures re-derived, `case-check` on the suite and the controls in five configurations, workspace tests |
