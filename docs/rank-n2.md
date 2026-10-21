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
| `ninjars::fast::FastAmplitude::new(n, n + 2, options)` | fast engine; `TensorNumerator`, `MuTensorNumerator` and `GenericNumerator` implement the rank n+2 expansions. A lane whose numerator has a lower rank `r` (`FastNumerator::lane_rank`) is reduced at rank `r`, bitwise as declared at `r` (`fast/declared_rank.rs`); `AtDeclaredRank(&mut numerator)` keeps the rank n+2 engine (section 7.2) |
| `FastOptions::conditioned_cuts`, CLI `--conditioned-cuts` | not needed: the rank n+2 engine always conditions its cut geometry. For rank <= n+1 it is an opt-in, off by default (section 7.2) |
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

## 6. Master integrals: B1111, the pins and the `Float` route

**Native family.** oneloopmaster fe550e3 adds `TensorIntegral::B1111` (commit
fe550e3):
- input `[p², m0², m1², mu²]`; output `[B0, B1, B11, B111, B1111]`
  (5 coefficients, 15 outputs);
- available on every `NativeFloat`, with a binary64 specialisation;
- it is OneLOop's `bub1111`, with the mass-swap reconstruction corrected
  (`B1111 = B0' + 4B1' + 6B11' + 4B111' + B1111'`, primes for swapped masses);
- B0..B111 inside the family are bitwise those of the B111 family.

**The pins.** NinjaRS pins oneloopmaster at 8ec256f, its main (8a8bc55;
`Cargo.toml`, `dependencies/manifest.json`), and Symbolica at
`alphal00p/symbolica` 9cdc2949 (72a273c). 9cdc2949 is e679e8c1 plus
symbolica-dev/symbolica#97, which rounds `Float` products once;
oneloopmaster 8ec256f pins the same revision. 8ec256f contains fe550e3, and
the tensor family is unchanged since. Its code is that of the verified
candidate 3a322d0; a8a3b8c and 8ec256f add documentation only. Against
09b2c74, main's pin since d53dd08, it changes C0 and D0 (8a8bc55):
- the contour box next to vanishing Gram determinants;
- Denner and Dittmaier's sheets for one- and two-mass boxes with complex
  masses;
- the two-mass mass-pair complements, formed from the channels;
- the relabelling of soft boxes at its numerical D0 entry points
  (`oneloop::regular_d0_labelling`).

Its verifiers compared it with 09b2c74 on three adversarial sweeps of D0 and
C0 rows: fs (36,840 rows), and ns and ns2 (42,560 and 21,280 rows, generated
after the fixes). The sweeps are weighted to vanishing Gram determinants and
to collinear, threshold, corner and near-shell boxes. COLLIER 1.2.9 in quad
precision gives references on 99,119 rows per route (oneloopmaster
`STATUS.md`, "Adversarial sweeps and the remaining tail"):
- **Fewer wrong values.** Summed over the four binary64 routes (hand-written
  and generated native code, the interpreted expressions, SymJIT), finite
  values wrong by more than 1e-6 go from 18,377 to 2,798, 6.6 times fewer
  (D0 alone: 18,073 to 2,494). Non-finite values, over all rows, go from
  31,076 to 3,162, 9.8 times fewer. DoubleFloat goes from 3,321 finite and
  wrong values to 21.
- **Most of the gain is the soft boxes** (oneloopmaster d880be8): none of the
  sweeps' 14,400 soft-box relabellings is wrong on any route now. On the
  other 84,719 referenced rows per route, the binary64 counts go from 6,441
  to 2,798 finite and wrong (2.3 times fewer) and from 8,056 to 2,534
  non-finite (3.2 times fewer).
- **A wrong D0 is not guaranteed to be NaN.** 21 binary64 boxes (48 route
  entries) are finite and wrong by more than 1e-6, by up to 0.36, where
  09b2c74 was right or NaN. These are the boxes outside the
  input-conditioning classes J and J* of oneloopmaster's `STATUS.md`; with
  them, 28 boxes and 63 entries.
  - 10 of the 21 (15 of the 28) were NaN at 09b2c74. NinjaRS's non-finite
    check caught them then and cannot now.
  - DoubleFloat is within 3e-14 on all 21.
- **`Float`.** At 128 and 256 bits, 195 rows are finite and wrong, mostly
  exactly vanishing Gram determinants that were NaN at 09b2c74.
  - They usually track 1 to 4 bits. One (ns 9210) tracks 253 of 256 bits and
    is wrong by 4.46.
  - The Symbolica pin makes four oracle-sweep boxes wrong at 128 or 256 bits.
  - `SymbolicaMasters` now reports `Float` results with too few tracked bits
    as failed (below).
- **NinjaRS's own calls.** On NinjaRS's recorded master calls, no D0 is more
  than ten times worse than at 09b2c74.
  - On the 309 distinct D0 calls of the recorded master streams, the worst
    binary64 error against COLLIER in quad precision goes from 3.60e-11 /
    3.59e-11 / 8.89e-12 / 3.41e-11 to 1.43e-12 / 1.42e-12 / 1.29e-12 /
    5.17e-12 (hand-written, generated, interpreted, SymJIT).
  - On the 3,510 D0 calls of the rank n+2 suite it is 7.53e-11 / 7.48e-11 /
    7.51e-11 / 2.58e-11 (SymJIT 1.66e-11 at 09b2c74).

oneloopmaster main landed with the rest as open work, as decided on
2026-10-21 (section 9). Its `STATUS.md` lists the 21 silent binary64 boxes
by row number and the remaining oracle-sweep boxes with their inputs. The
195 silent `Float` rows and the regressions that are not silent (more than
ten times worse but within 1e-6, or NaN where 09b2c74 was right) appear there
as counts with examples.

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

**The `Float` route.** binary64 and DoubleFloat take every master from
oneloopmaster's native code: `OneLoopNativeMasters`, or `SymbolicaMasters`
through oneloopmaster's `NativeEvaluator`. With `Float` carriers,
`SymbolicaMasters` works differently:
- it takes B0..B1111 from the native families at the row precision (bd2022b,
  6487309, c3a6aba; section 9);
- it evaluates oneloopmaster's exact C0 and D0 expressions
  (`OneLoopExpressions`) in its own batch interpreter.

Two checks were added to this route with the oneloopmaster re-pin.

**Soft boxes are relabelled on the `Float` route (8e583f2).**
oneloopmaster's D0 expressions, like OneLOop's `d0`, find a soft line's pole
only on the four external invariants. A labelling of the same box that puts
the soft line's on-shell pair, with a massive partner, on a channel (s12 or
s23) misses the pole.
- oneloopmaster 8ec256f relabels such points at its numerical entry points.
  `SymbolicaMasters` now applies the same `oneloop::regular_d0_labelling` to
  each `Float` D0 row after the on-shell threshold.
- `tests/soft_box_labelling.rs` takes pyAmpliCol's e+ e- → t t̄ h soft box
  (lines (m, m, 0, m), m² = 29929, mu² = 1e6, exact rationals) in all 24
  labellings. It runs every provider and carrier, batched and row by row.
- Over the 24 labellings, relative to the largest coefficient, the worst
  result is within 5.2e-16 (binary64) and 2.0e-16 (DoubleFloat, Float(128),
  Float(256)) of COLLIER. Against oneloopmaster 09b2c74's 38-digit value of
  the cyclic labellings it is within 5.2e-16, 1.8e-31, 2.0e-38 and 1.9e-38.
- Without the relabelling, Float(128)'s labelling 0132 is 73 times its
  largest coefficient away from labelling 0123.

**`Float` results with too few tracked bits fail (2a7df0d).** numerica's
`Float` tracks the precision of every value. `SymbolicaMasters` used to check
only finiteness, so a 128-bit D0 that tracked 2 bits and was wrong by 2e2
counted as a success, and a rescue loop never escalated its lane.
- **The check.** Every `Float` result is checked: the finite part and both
  poles of every family (A0, B0..B1111, DB0, C0, D0, An). Its tracked
  precision as a complex value (`ninjars_symbolica::tracked_bits`) must reach
  `required_tracked_bits(working) = max(53, working/4)`. Below that, the row
  fails with `MasterFailure::ImpreciseResult` through the same path as a
  non-finite result, so the lane is not a success.
- **Scope and cost.** binary64 and DoubleFloat track no precision and are
  unchanged. A value whose parts both keep the required bits passes on their
  precisions alone. Otherwise `tracked_bits` takes 77 ns, 1 % of a native
  Float(128) B11 row.
- **`SymbolicaDualMasters`** checks the value of every coefficient, not the
  derivative components. The derivative of an analytically constant
  coefficient is a rounding residue that no precision makes exact.

The rule was chosen on oneloopmaster 8ec256f's C0 and D0 on the sweeps fs, ns
and ns2 (100,680 rows, native and interpreted `Float`). Against COLLIER in
quad precision, a value is wrong above 1e-6 and right below 1e-10. The table
gives false negatives / false positives over both routes:

| rule | 128 bits | 256 bits | 512 bits | 1024 bits |
|---|---|---|---|---|
| 53 bits | 4 / 1059 | 2 / 0 | 2 / 0 | 2 / 0 |
| `bits/2` | 4 / 3505 | 2 / 72 | 1 / 52 | 1 / 0 |
| `bits/4` | 4 / 0 | 2 / 0 | 2 / 0 | 2 / 0 |
| `max(53, bits/4)` | 4 / 1059 | 2 / 0 | 2 / 0 | 2 / 0 |

- **False negatives.** Every one is among the 6 rows that report 81 to 1023
  tracked bits while wrong (ns 9210, 10378, 10965, 11620, ns2 5278, 11326).
- **Why the binary64 floor.** `bits/4` alone lets 392 of the 581 values
  between 1e-10 and 1e-6 pass at 128 bits (up to 1e-7, with 32-43 tracked
  bits). The floor flags them, and with them 1059 right values with 35-52
  bits (0.5 %). At 256 bits all but 2 of these 1640 values are right with at
  least 64 bits.
- **Other sets.** On all 14 verifier sets (333,000 rows, 128 and 256 bits)
  the rule flags 0.26 % of the right values at 128 bits and none at 256, and
  passes no marginal value.
- **NinjaRS's own calls.** The recorded master calls (843 and 7,644 rows of
  C0 and D0) all succeed through `SymbolicaMasters<Float>` at 128 and 256
  bits. Their finite parts keep at least 101 and 229 tracked bits, their
  poles at least 123 and 251, or are exactly zero.
- **Tests** (`tests/master_status.rs`):
  - ns 15427 and ns2 7068 (vanishing Gram determinants) keep 2-3 tracked bits
    and are off by more than 1 at 128 bits on the interpreted route.
    `SymbolicaMasters<Float(128)>` reports them `ImpreciseResult`. Float(256)
    evaluates them with 108 and 117 tracked bits, 3.3e-31 and 2.1e-31 from
    COLLIER.
  - B1111 rows whose inputs determine too few bits for the working precision
    now fail too: at 256 bits, a row with a 53-bit invariant or a 64-bit mu²
    (`float_b1111_rows_take_the_precision_their_inputs_determine`).

## 7. Engine architecture

### 7.1 Generic engine: one code path, parameterised by `RankClass`

Every exact plan is built from the `RankClass`, and for `UpToN` and `NPlus1`
the plans are byte-identical to the plans before rank n+2. The V0p test
fingerprints 1,239 plans for n <= 6 and rank <= n+1; all are unchanged.

The cut provider's bubble-reference and tadpole-basis selection does not
depend on the rank class (section 7.2, 5ed3b22). It moved rank <= n+1
results at rounding level (V0 re-blessed, `validation/regression/README.md`).

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

`FastAmplitude` holds `replacement: Option<Box<Replacement>>`, built in `new`
iff `rank == n + 2` or `FastOptions::conditioned_cuts` is set:
- `Replacement::RankN2 { engine, conditioned_cuts, lower }` holds the
  `RankN2Engine`, the option as given, and the rank <= n+1 engines of
  `declared_rank.rs`, which are built on first use;
- `Replacement::Conditioned` holds the `ConditionedEngine` (below).

`evaluate` tests the pointer once and calls one of two cold, never-inlined
entries:
- `evaluate_declared_n2`: lanes of a lower rank go through their rank-`r`
  engines (`FastAmplitude::evaluate_lowered`, never back through
  `evaluate`); the rest go to `RankN2Engine::evaluate`;
- `evaluate_conditioned`: generic in the providers, which it type-erases, so
  that the conditioned stages are instantiated next to the default ones.

The unchanged rank <= n+1 body is `#[inline(never)] evaluate_up_to_n1`.
`FastAmplitude`'s layout is main's: the lower-rank state sits in the rank
n+2 variant. The rank n+2 engine is in `fast/engine_x2.rs`:
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
parts; only their sum is reference independent. Run through the rank n+2
engine (`AtDeclaredRank`), a rank <= n+1 numerator declared at rank n+2
therefore reproduces the rank <= n+1 engine's Laurent coefficients, but not
its split.

**The generic engine** uses the same scores in its cut provider, at every rank
class (`cuts.rs::bubble_reference`, `tadpole_line_scores`; 86b378b for rank
n+2, 5ed3b22 for every rank class, f9a8b16 for the cost).
- **Bubbles:** the 24 signed permutations of `(1, 2/3, 2/3, 1/3)`.
  - These are the fast engine's 24 up to the factor 4/3, and the score is
    scale invariant.
  - The provider's first eight come first. A further kernel gives `k.r` of
    the other 16 for every bubble row, and one uncut-line probe gives the
    invariants `v.v`, `k.v` and `r.v` of all 24 (f9a8b16).
- **Tadpoles:** the six exact tadpole bases. Ninja's three are irrational.
- **Ties.** Scores within the conditioning tolerance 2^-(p/2) of the best go to
  the lowest index.
  - Exact ties (symmetric rational kinematics) therefore resolve identically
    in every precision, so the cut-constructible/rational split does not
    follow round-off.
  - Near ties, with gaps inside binary64's tolerance but not inside the higher
    precisions', can resolve differently in binary64 than in DoubleFloat and
    `Float`.
  - A batch equals its rows run singly, bit for bit.
- **Requests carry no rank class.** Padded and unpadded lanes choose the same
  references, so V5 holds at 1e-50 with snapped S-matrices.
- **Rank <= n+1 results** changed at rounding level and in their split. V0 was
  re-blessed with a symmetric statistic (`validation/regression/README.md`);
  V0p is unchanged.
- **Cost** (perf `instructions:u`, `ninjars bench --engine generic`, binary64,
  per point, against main 0d0d7e7, widths 1 / 32):
  - +1.4 % / +1.5 % (phi3 `box`) and +1.0 % / +0.8 % (`pentagon-rank4`);
  - −0.04 % (`heptagon-rank8`) and +0.1 % (four-photon `top-box-0`);
  - 0.0–0.1 % on the review's rank n+2 DOT fixtures, with the setup within
    +1.3 %;
  - with the guard's harness (`validation/performance/rank-n2-ab`,
    `generic_icount.sh` specs), at most +1.54 %; DoubleFloat and Float(128)
    within +0.74 %.

  The determinants come from the probes' invariants. For a null `r`,
  `G(k, r, v) = (r.v)(2(k.r)(k.v) − k²(r.v)) − (k.r)²v²`, combined in the
  conditioning check, in real arithmetic when the invariants are real.
  Evaluating the 24 determinants per uncut line in Symbolica's interpreter
  (5ed3b22) chose the same references, at +15 % (box), +17.5 % (pentagon),
  +6.4 % (heptagon) and +0.5–0.7 % (physical and rank n+2 fixtures). f9a8b16
  leaves every result identical in every printed digit.

**Conditioned box geometry (`fast/box_x2.rs`, 611f6dc).** The box rational
terms come from the large-`t` expansion on the d-dimensional cut
`q = q̄0 - V0 + t v`, `mu² = t² v² - beta` (section 3.1). That expansion
assumes that every cut denominator vanishes identically in `t`.
- Original Ninja's normal `(p1.e4) e3 - (p1.e3) e4` is built from the spinors
  of the cut basis. Its residual `|n.K|/(|n||K|)` reaches 3.2e-15, with a
  median of 9.1e-17. This was measured in Float(256) on 573 boxes of eight
  rank n+2 cases with n = 4..7, when 611f6dc was verified.
- The rank n+2 expansion is one order deeper and amplifies that residual. On
  `rn2-exact-mu-n7-r9-perm` lane 1, the box `c4` came out 5 to 100 times
  further from Float(256) than the generic engine's. There the box sum
  cancels 3e5 times against the triangles.
- An exact check of one box (`|c4| = 1.2e5`): one-ulp perturbations of the
  normal move `c4` by up to 5.5e-10. Rounding the center, `beta` or the
  series division moves it by less than 7e-12.

The rank n+2 engine therefore takes:
- as the normal, the Levi-Civita vector of the three cut momenta, from 3x3
  cofactors with Kahan's difference of products, scaled by a power of two.
  Its residual is at most 8.2e-17;
- Ninja's center (the mean of its two solutions), projected onto the normal's
  complement and refined once against the cut's linear equations with
  compensated residuals, with `beta` compensated as well;
- the two solutions `q̄0 ± sqrt(beta/v²) v` for `c0`.

Ninja's stability checks are kept and run first. The geometry adds one
failure condition: `box_geometry_x2` returns `None`, and the lane is
`Unstable`, when the normal, `v²` or the Gram determinant of the cut momenta
is exactly zero or not finite. This never triggers on the suite, the controls
or the identity cases (all 1345 lane statuses are those of main, with and
without `conditioned_cuts`), nor on the V9 Gram scan (verification of
611f6dc).

Effect on the rank n+2 set, against main 0d0d7e7's spinor geometry. The set
is the 426 tier-A lanes of the rank n+2 cases: 250 in
`validation/rank-n2/cases` and `validation/benchmark/synthetic-rank-n2`, 176
in `validation/rank-n2/identity/cases`. Errors are `case-check`'s scaled
differences, floored at 1e-16 for ratios; "outside" means
`e > max(10 e_generic, 1e-13)` with `e_generic` the generic engine's binary64
error.
- The geometric mean of the error ratio is 0.70.
- 77 lanes are more than 2 times better and 11 more than 2 times worse. The
  worst of these is 5.4e-11 on `rn2-exact-q-n7-r9` lane 1 (was 1.9e-11,
  generic f64 2.1e-10), and none is above 6.9 times the generic engine's
  binary64 error.
- Lanes outside fall from 28 to 13.
- `rn2-shifted-mixed-n7-r9-mu2` lane 0: 1.7e-3 → 1.3e-5 (generic f64 1.6e-4);
  `rn2-exact-mu-n7-r9-perm` lane 1: 6.1e-7 → 1.8e-8; `rn2-id-pinch-n4`
  lane 3: 5.3e-9 → 1.7e-9.

On the 250 suite and synthetic lanes alone the figures are 0.76, 34 better,
9 worse, and 17 → 8 outside.

**Lanes of a lower rank (`fast/declared_rank.rs`, dc8c7e7, baaa240).** At
declared rank n+2, `FastAmplitude::evaluate` asks the numerator for the rank
of every lane (`FastNumerator::lane_rank`, `None` by default).
- `TensorNumerator` and `MuTensorNumerator` read the rank from their
  coefficients. A lane of full rank costs one test of the top block of `T_0`.
- A lane of rank `r < n + 2` is reduced by the rank <= n+1 engine built at
  rank `r` on first use. That engine has the caller's `FastOptions`, and its
  `mu²` powers are capped at `r / 2`. The numerator is presented at rank `r`:
  `lower_rank` copies the selected lanes' leading coefficients to a spare
  buffer, and `restore_rank` swaps them back exactly.
- **What the result equals.** A lane declared at rank n+2 is reduced bitwise
  as declared at its own rank, split included.
  - That equals the undeclared run (and the CLI, which takes the case's rank)
    when the case's rank is the lane's rank.
  - Only declared rank n+2 is lowered. Declaring rank n+1 for a lane of rank
    <= n still changes its result, as in original Ninja.
  - Lowering such lanes only to rank n+1, as first planned, failed 4 of the
    615 suite control lanes declared at n+2 (`rn2-int-mixed-n2-r2` lane 1:
    1.5e-13 against a tolerance of 3.8e-14; also `rn2-irsoft-real-n5-r5`,
    `rn2-phys-real-n6-r6`, `syn-n6-r6`).
- **Mixed batches.** Lanes of different ranks in one batch are reduced as one
  sub-batch per rank and scattered back, so a lane's result still depends on
  that lane alone. Failed lanes keep their previous result.
- **Opting out.** `GenericNumerator` and other numerators answer `None` and
  keep the rank n+2 engine. `AtDeclaredRank(&mut numerator)` keeps it for any
  numerator; V5 uses it.
- **With `conditioned_cuts`.** The lowered engines inherit the option and run
  through their `ConditionedEngine`. Lanes declared at rank n+2 with the
  option are bitwise their own-rank conditioned results:
  - `rank_n2_padding.rs::fast_padding_to_rank_n_plus_two_with_conditioned_cuts`:
    238 cases, 753 lanes, 650 of them different from the default geometry;
  - the conditioned mixed-rank test: 268 lane checks, 22 lowered lanes
    different from the default;
  - `case-check --declare-rank n+2 --conditioned-cuts` on the controls:
    630/630 lanes (integration pass 1).
- **Padded numerators.** On 17 rank <= n+1 cases declared at n+2, the
  replayed reduction takes 4.5 times fewer instructions (geometric mean, 1.8
  to 8.9 times), and the live path 4.8 times fewer.
- **Accuracy.** The rank n+2 engine and the rank-`r` declaration round
  differently, mostly through the basis choices of `bases_x2.rs`. Neither is
  uniformly better (section 9). `conditioned_cuts` gives the lowered lanes
  the conditioned bases without over-declaring.

**Opt-in conditioned geometry for rank <= n+1 (`FastOptions::conditioned_cuts`,
d0819f0, ec2df0f).** With the option, the rank <= n+1 stages take the rank
n+2 engine's choices per cut and lane: the bubble references and tadpole bases
of `bases_x2.rs`, and the box geometry of `box_x2.rs`. `case-check
--conditioned-cuts` and `case-bench --conditioned-cuts` run it from the CLI.

How it is built (ec2df0f):
- The option is a separate engine, `ConditionedEngine`, chosen once in
  `FastAmplitude::new` like the rank n+2 engine.
- Its stages (`evaluate_up_to_n1`, `boxes`, `bubbles`, `tadpoles`) live in
  `fast/engine/conditioned.rs`. That file is generated from `engine.rs` by
  `fast/tests_conditioned.rs` with a short list of edits.
  `NINJARS_BLESS=1 cargo test -p ninjars --lib conditioned_stages`
  regenerates it, and the test fails while the file is stale or an edit's
  anchor is gone.
- The stages run with type-erased providers and are instantiated next to the
  default instances, through a dispatcher that is generic in name only.
  Compiled in the `ninjars` crate instead, they made thin LTO share
  `RawVec::grow_one` with the default instances and turned those instances'
  calls into GOT calls.
- The stages read `StageOptions`, which is `FastOptions` without the flag. A
  larger options field would move `FastAmplitude`'s other fields.
- Without the option, the rank <= n+1 stages keep original Ninja's bubble
  reference, tadpole bases and box geometry, and their machine code is
  main's.

Effect of the option, with the statistics of the box geometry above:
- **Controls** (the 465 tier-A lanes of `validation/rank-n2/controls/cases`
  and `validation/benchmark/synthetic/syn-n{2..6}-r*`, all of rank <= n+1):
  the geometric mean of the error ratio is 0.46. 167 lanes are more than 2
  times better and 20 more than 2 times worse (21 without the 1e-16 floor).
  The worst is 1.1e-10 on `rn2-irml-zero-n5-r6` lane 1: default 1.3e-11,
  generic f64 5.9e-11, so 11 times generic. Lanes outside fall from 43 to 6.
- **Identity pieces** (the 203 tier-A rank <= n+1 lanes of
  `validation/rank-n2/identity/cases`): the geometric mean is 0.135, with 120
  lanes better and 2 worse (at most 1.7e-12). Lanes outside fall from 55 to 3.
- **The n = 7 pinch pieces** (`rn2-id-pinch-n7-sub`, zero-time-component
  bubbles) go from 3.0e-11..1.6e-8 (lanes 0, 2, 3, 6 at 5.8e-9..1.6e-8) to
  6.8e-14..7.6e-12.
- **V4f policy on the controls:** 600/0/15/0, as by default. The largest
  checked difference is 2.4e-7 (default 2.6e-6).
- **Cost,** as instructions per point (`ab_icount.rs`, pinned) against main
  over 54 benchmark keys at widths 1 and 32: replayed reduction +29.8 % in
  geomean (largest +47.6 %), live path +6.5 % (largest +21.9 %). Against
  d0819f0's in-place opt-in, the separate engine costs +0.50 % replayed in
  geomean (largest +3.6 % on `syn-n2-r1` at width 1) and +0.04 % live, from
  the type-erased providers and the dispatch.

**Why it is not the default** (decided 2026-10-20). It is the accuracy option
for binary64 runs that do not need comparability with original Ninja:
1. **Cost.** Almost all of the +29.8 % is the per-lane scoring of 25 bubble
   references and 9 tadpole bases. The performance guard allows at most 1.03
   per key against main, and the reduction-versus-Ninja gate would lose
   margin.
2. **Comparability.** It changes the default results: the V0 fast fixture;
   the comparisons with original Ninja's results (`fast_ninja_reference`,
   which asserts the Laurent coefficients and the cut-constructible/rational
   split to 1e-11; V5's Ninja comparisons; `case-bench`'s Ninja reference
   checks); and the split itself, which moves with the bubble reference, so
   that splits would stop matching Ninja's and MadLoop's.
3. **Not uniformly better.** About 20 of the 465 control lanes get more than
   2 times worse, at worst 1.1e-10 (11 times generic f64).
4. **The production model is escalation.** binary64 behaves as Ninja does;
   ill-conditioned points are flagged (generic `--stability-test`) and
   rescued in DoubleFloat. The option improves binary64 statistically but
   does not replace the rescue.

A cheaper scoring is a follow-up, after which the default can be
reconsidered (`TODO_NEXT_PLAN.md`).

**Zero overhead for rank <= n+1.** Each check names its source:
- **Bitwise results.** V0 (`regression_bitwise`,
  `validation/regression/r-le-n1-d53dd08.json`) holds main's rank <= n+1
  bits. Its fast sections are bitwise those blessed on main 0d0d7e7 with only
  the oneloopmaster 8ec256f and Symbolica 9cdc2949 re-pins applied (c24f976).
  Its generic sections moved only for the generic engine's own accuracy
  changes and the re-pins (section 10).
  <!-- PASS2: state that V0 (with its generic Float(256) section compared within the precision bound, section 10) and V0p pass at the pass-2 head, and name that head. -->
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
- **After the conditioned geometry and the lower-rank lanes** (integration
  pass 1: the release CLI at f33ff8d against main 0d0d7e7, `dis_compare.py`
  without the rename alias):
  - every `boxes`, `bubbles`, `tadpoles`, `triangles` and
    `evaluate_up_to_n1` instance (28) is identical, as are 2104 of the 2129
    matched functions;
  - with struct offsets kept, 18 of the 28 differ only in displacements, from
    the over-declared numerator layouts (`TensorNumerator`,
    `MuTensorNumerator`, `ReplayNumerator` and `RecordingNumerator` fields).
    Each `evaluate_up_to_n1` instance differs by one instruction, the
    numerator's `rank` field (`mov 0x30(%rcx)` → `mov 0x70(%rcx)`). Against
    the integration base cdb5c07 all 28 are identical with offsets kept;
  - the other differences are `FastAmplitude::new`, `TensorNumerator::new`
    and `MuTensorNumerator::new`, drop glue, and the benchmark drivers that
    inline the dispatcher (`time_case` closures 89 → 81 instructions,
    `time_widths` closure 110 → 105, `FastSession::evaluate` 235 → 242);
  - instructions per point on rank <= n+1 against main (54 keys, widths 1
    and 32): replayed +0.001 % (−0.012 % to +0.017 %), live −0.000 %
    (−0.014 % to +0.017 %);
  - rank <= n+1 lanes are bitwise those of main on 630 control lanes and 203
    identity pieces (611f6dc..a0e7b50).
  <!-- PASS2: replace or confirm the f33ff8d machine-code and instruction-count figures with the pass-2 head's: dis_compare.py --no-alias against main 0d0d7e7 (identical stage instances out of how many, matched functions), offsets_cmp.py, and ab_icount.rs rank <= n+1 replayed/live geomean and range (54 keys, widths 1 and 32). -->
- **The Symbolica re-pin** (72a273c, verify-ninjars-prep): all 193
  `ninjars::fast` functions are identical; only 44 `oneloop::native`
  instances on MPFR `Float` differ, as symbolica-dev/symbolica#97 implies.
- **Wall clock, rerun** (73f53d9: main d53dd08 against main 0d0d7e7,
  oneloopmaster 09b2c74 on both sides). Interleaved A/B over 192 keys, 10+10
  runs on each of two cores: geomean 0.9996, max 1.038 (1.012 rerun alone),
  A/A max 1.022.

**Cost of rank n+2.** Replayed reduction at width 32, `rn2-int-complex`
against its rank n+1 twin, from the benchmark report's rank n+2 measurement
(`docs/benchmark/data/rank_n2.json`, `twins`). It was taken on main 0d0d7e7
with oneloopmaster 09b2c74, on 2026-10-19 on CPU 260: after the basis
conditioning, before the box geometry and the lower-rank dispatch.

| n | rank n+2 (ns/point) | rank n+1 twin (ns/point) | ratio | total ratio | COLLIER COLI ratio |
|---|---|---|---|---|---|
| 1 | 145 | 172 | 0.85 | 1.93 | 1.53 |
| 2 | 2689 | 1201 | 2.24 | 2.73 | 1.48 |
| 3 | 15556 | 6108 | 2.55 | 2.54 | 1.65 |
| 4 | 46318 | 18598 | 2.49 | 2.05 | 2.27 |
| 5 | 103092 | 42085 | 2.45 | 1.80 | 5.27 |
| 6 | 198223 | 83199 | 2.38 | 1.65 | 4.45 |
| 7 | 344292 | 152436 | 2.26 | 1.62 | 4.21 |

<!-- PASS2: if the benchmark refresh of pass 2 regenerates docs/benchmark/data/rank_n2.json, refill this table from its `twins` (reduction_w32 rank n+2 / twin / ratio, total_w32 ratio, collier ratio) and the date, head, oneloopmaster pin and CPU of the sentence above; likewise the median and range of the numerator share and the per-n COLLIER factors below (by_n). -->

- The total ratio includes the numerator and the masters. The COLLIER column
  is COLLIER 1.2.10's (COLI) own rank n+2 time over its rank n+1 time on the
  same pair.
- Absolute times are not comparable with the S3 report's (another core and
  clock); the ratios are. The S3 report measured reduction ratios of 2.08 to
  2.90 for n = 2..6 (ebdbc00), the S3 verifier cycle ratios of 2.18-2.81.
- The dense tensor-numerator expansions (mostly the
  `t3_expansion_x2`/`t2_expansion_x2` jets) take a median 64–76 % of the
  fast engine's time per n at rank n+2 (single cases 56–93 %). They are the
  main rank n+2 speed target.
- Against COLLIER 1.2.10 (COLI, finite part, without its global cache, which
  returns wrong 7-point rank-9 results: `COLLIER_CACHE_BUG.md`), NinjaRS takes
  ×0.52, ×1.24, ×1.15, ×0.27, ×0.050, ×0.026, ×0.011 of the time for
  n = 1..7 (benchmark report, section 5). Per-case wall-clock ratios are
  uncertain by about ±20 % at n <= 3.

Instructions per point added to the rank n+2 path:
- the basis conditioning: 3.0 % (n = 7) to 5.8 % (n = 3) on the replayed
  reduction and 0.3-1.4 % on the live path (79f306a; included in the table);
- the box geometry, over 111 rank n+2 cases at widths 1 and 32 against main
  0d0d7e7: +1.27 % replayed in geomean (n = 7: +2.6 %; largest +5.8 %) and
  +0.15 % live (largest +3.3 %) (611f6dc). The dispatch of cdc1930 adds
  +0.03 % replayed;
- the lower-rank dispatch: about 56 instructions per call plus 11 to 16 per
  lane replayed, about 95 plus 33 live. That is about +0.4 % for n >= 2 and
  +2.5 % on the n = 1 tadpole cases, at 2264 instructions per point (dc8c7e7,
  baaa240). `lane_rank` is `#[inline(always)]` (f33ff8d): out of line it cost
  about 35 instructions per lane on the live path;
- together, at f33ff8d against main 0d0d7e7: +1.49 % replayed in geomean,
  +2.6 % at n = 7. At n = 1 and width 1 it is up to +7.4 % replayed and
  +12.5 % live (`rn2-int-mixed-n1-r3`, +92 instructions per point replayed).

<!-- PASS2: rank n+2 instructions per point at the pass-2 head against main 0d0d7e7 (ab_icount: replayed and live geomean, n = 7, largest key), replacing or confirming the f33ff8d line above. -->

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

In binary64, COLLIER 1.2.10 (COLI) is the timing and accuracy reference of
the benchmark report's rank n+2 section (`validation/benchmark/collier`;
measured at 4dec120 on main 0d0d7e7, so before the box geometry, the generic
conditioning and the per-term stability test):
- On the 298 tier A/B suite and synthetic lanes with condition <= 1e6
  (case-check judges 276 of them, 22 are report-only), the fast engine is at
  least as accurate as COLI on 212.
- On the final review's 166 fresh lanes, the 21 lanes where NinjaRS's
  binary64 error exceeds 1e-6 all have cut contributions cancelling by 3e6 or
  more. COLI stays below 1e-6 on 16 of these 21 (median 4.8e-9). The generic
  engine's `--stability-test` flags 18 of them, including every lane above
  1e-5.
- COLLIER's optional global cache returns wrong 7-point rank-9 results (a
  COLLIER defect, `COLLIER_CACHE_BUG.md` and
  `validation/benchmark/collier/collier_cache_repro.f90`), so the benchmark
  runs COLLIER without it.

### 8.2 Tests

V-numbers follow the design. The result column states where each number was
measured.

| id | test (`crates/ninjars-symbolica/tests/` unless noted) | result |
|---|---|---|
| V0 | `regression_bitwise::rank_up_to_n_plus_one_is_bitwise_unchanged` | against `r-le-n1-d53dd08.json` (file name kept): main d53dd08's bits, re-blessed for main's own changes and in place for the generic selection (5ed3b22), the native B0 (c3a6aba) and the oneloopmaster 8ec256f and Symbolica 9cdc2949 re-pins (c24f976; `validation/regression/README.md`, section 10). Fast and binary64 sections bitwise; blessed at c24f976 <!-- PASS2: the pass-2 head at which V0 passes, and how the generic Float(256) section is compared (the precision bound of section 10) --> |
| V0p | `src/plan_fingerprints_tests.rs::exact_plans_for_rank_up_to_n_plus_one_are_unchanged` | 1,239 fingerprints unchanged; passes at c24f976 <!-- PASS2: confirm at the pass-2 head --> |
| V1 | `cargo test --workspace` | 8346699, release: default features 389 passed, 0 failed, 12 ignored (47.6 min); `--all-features` 498, 0, 21 (75.4 min) <!-- PASS2: the pass-2 head's counts (passed / failed / ignored, minutes) for each test command that was run: -p ninjars, -p ninjars-symbolica --features native-masters, -p ninjars-graph, -p ninjars-cli, and --workspace --all-features if run --> |
| V2a | `rank_n2_laurent` | division recovers known quotients, n = 1..6: 8.7e-77 (Float(256)), 1.8e-15 (binary64); new subtraction terms against the sympy closed forms <= 4e-77, swapped-axis control 1.2 (S2) |
| V2b | `rank_n2_reconstruct` | on-cut reconstruction of every cut size, n = 1..6, with and without mu²: <= 1.1e-73 (Float), <= 5e-13 (binary64); pruned swapped arithmetic identical (S2) |
| V2c | `rank_n2_boxes` | box modes agree, n = 4..6: <= 1.5e-70 (Float); binary64 <= 3e-14 (coefficients), 5e-10 (integrated, n = 6); beta-offset control (S2) |
| V2 (fast) | `crates/ninjars/src/fast/tests_x2.rs` | jets against direct expansion, division multiply-back at NT = 6 and 5, on-cut residue reconstruction, box finite-mu fit, rank-3 tadpole integral, pinch identity, batching and replay bitwise (1d5cb5f) |
| V3 | `tests_x2.rs` (fixtures) | 24 cases, worst 1.24e-15 (series) and 1.26e-15 (closed form); 444 perturbed controls each detected (S3) |
| V4 | `rank_n2_reference::suite_generic_float_matches_references` | Float(256), exact masters: 285 pass, 48 report-only, 0 fail; tier A <= 1.3e-36 at n <= 6, <= 5.9e-30 at n = 7 (integration run, S3 verifier) <!-- PASS2: the pass-2 head's Float(256) counts and tier-A maxima from the section 8.3 run --> |
| V4f | `suite_fast_binary64_matches_references`, `suite_generic_binary64_matches_references` | fast 276 pass, 57 report-only; generic f64 275, 58; 0 fail (integration run) <!-- PASS2: the pass-2 head's fast and generic f64 counts from the section 8.3 run --> |
| V5 | `rank_n2_padding` (`fast_`/`generic_padding_to_rank_n_plus_two`, `fast_padding_to_rank_n_plus_two_with_conditioned_cuts`, `fast_mixed_rank_batches_*`), `fast_rank_n2::fast_rank_n2_padding_reproduces_original_ninja` | rank <= n+1 cases declared at n+2 (`padding-manifest.json`, 238 cases): generic Float(256) within 7.3e-61 on 745 lanes with the lanes' own S-matrices (5ed3b22; 4.9e-55 at the S5b verifier); fast engine through `AtDeclaredRank` (the rank n+2 engine): Laurent coefficients against original Ninja, or within 10 times Ninja's own error of the exact reduction where that error exceeds the tolerance (79f306a); fast engine by default: every lane (753) bitwise equal to the case declared at the lane's own rank, also with `conditioned_cuts` (650 of the 753 differ from the default geometry); mixed-rank batches with native masters (ranks n-1 to n+2, plain and with mu² to mu⁶ terms, 5 `Unstable` lanes, with and without S-matrices, alone and in natural, reversed and 37-lane cyclic batches): all 268 lane checks bitwise equal to the lane's own-rank single-lane run, failed lanes keep their previous result (baaa240; with `conditioned_cuts`, 22 lowered lanes differ from the default) |
| V6 | `rank_n2_identity` | pinch `I_n[D_j M] = I_(n-1)[M]` and constructive decompositions, n = 2..7; Float: rank n+2 sides 5.3e-70 (S5b verifier); binary64: pieces with Ninja's geometry for n <= 6, and with `conditioned_cuts` for every n (largest 1.5e-11); rank n+2 sides against the pieces, largest 3.8e-9, generic f64 2.3e-8 (d0819f0). Ninja's geometry at n = 7 stays ignored (`*_default_geometry`, 8.4) |
| V7 | `rank_n2_invariance` | shift (V7a) and relabelling (V7b) at rank n+2 in Float(256): 1.1e-69 (S5b verifier); batch width and lane order bitwise (V7c); V7b binary64 for every n, fast at most 4.9e-7, within 10 times the noise (611f6dc) |
| V8 | `fast_parity::parity_rank_n2_n1..n7` | fast against generic f64 with identical native masters, 1e-9 scaled with a noise model; n = 6 enabled at 8346699 (merge run) |
| V9 | `rank_n2_reference::conditioning_scan_report` | report only (bubble p² ladder, triangle and box Gram ladders) |
| V10 | `rank_n2_dual` | HyperDual against finite differences, n = 2..4 (stand-in masters) and n = 2 (exact masters incl. B1111): <= 5.5e-48 (S2); the S2 verifier found <= 1.6e-44 against central differences at Float(512) |
| V11 | `fast_native_masters`, `masters`, `master_status`, `rank_n2_interfaces`, `rank_n2_float_bubbles`, `soft_box_labelling` | B1111 accuracy and interface identity; B0..B111 from the native families for every carrier; soft D0 boxes in all 24 labellings for every carrier; `Float` results with too few tracked bits fail and are rescued at higher precision (section 6) |
| CLI | `crates/ninjars-cli/tests/case_check.rs` | case-check equals case-bench's reference check; controls with both engines; failures, stale known failures and the engine and precision scope of known-failure entries are reported; the GENERIC_RUNNER interface; a rank n+2 sample; `--stability-test` flags exactly lane 1 of `rn2-phys-{real,complex}-n6-r8` in binary64 (real lane 2 is a false negative, section 9), and the cancellation inside cut contributions of the review's reproducer (`tests/data/repro-stability-test-miss-n6-r8.json`) and of `rn2-phys-complex-n6-r7` lane 1 |
| stability | `stability.rs`; `integration.rs::term_moduli_count_every_master_and_rational_term` | the any-n cancelling lane is flagged in binary64 and in DoubleFloat (off by 6.9e-16 > 2^-53 against Float(256)) and passes in Float(256); the term moduli count every master and rational term (9ff3d0a) |

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

<!-- PASS2: replace the table and the sentence above it (head, oneloopmaster pin) with the five runs at the pass-2 head (oneloopmaster 8ec256f, Symbolica 9cdc2949): per n and in total pass / fail / report-only / known failure, the largest checked difference, tier A n <= 6 / n = 7, case errors, unchecked lanes, stale known failures and exit status of each run. -->

The controls (615 rank <= n+1 lanes, same command with `$R/controls/reference`)
gave 600/0/15/0 (fast), 599/0/15/1 (generic f64), 600/0/15/0 (DoubleFloat),
588/0/15/12 (Float(128)) and 601/0/14/0 (Float(256)) at e116813, and also
exit 0. At 5ed3b22 (the generic selection at every rank class) they give
600/0/15/0 (fast, generic f64 and DoubleFloat), 589/0/15/11 (Float(128)) and
601/0/14/0 (Float(256)): the two entries of generic f64 and Float(128) in
`known-failures.json` went stale and were removed.
<!-- PASS2: the controls' five counts (pass / fail / report-only / known failure) and exit status at the pass-2 head, and any known-failures.json entry added or removed when it was re-derived there. -->

**Changes since e116813**, each measured on its own commit:
- **Generic selection (86b378b, 5ed3b22).** The suite verdicts are unchanged
  in every run, and 5ed3b22 leaves every suite lane bitwise as at 86b378b.
  - The largest checked difference falls from 5.2e-7 to 3.5e-8 for generic
    f64, and from 1.4e-11 to 7.5e-12 for DoubleFloat and Float(128). The fast
    engine is unchanged bit for bit.
  - Float(256) verdicts are unchanged. Its values move only on
    snapped-S-matrix lanes, by up to 5.1e-8 (`rn2-phys-complex-n6-r8`
    lane 1). This is input ambiguity: with the shift invariants, before and
    after agree to 1.6e-37. Its largest checked difference is 7.06e-12 (was
    7.02e-12).
  - On 6 report-only `phys`/`syn` lanes the move exceeds the recorded
    `input_ambiguity`, by up to 15.5 times (`rn2-phys-real-n6-r8` lane 2:
    4.8e-9 against 3.1e-10). That uncertainty comes from the generic engine's
    own rank twins and is not a bound (`validation/rank-n2/README.md`).
- **Box geometry (611f6dc).** The fast column stays 276/0/57/0 on the suite
  and 600/0/15/0 on the controls. Its largest checked difference on the suite
  is 2.6e-8 (was 5.3e-8), and the controls are bitwise unchanged. With
  `--conditioned-cuts` the controls are also 600/0/15/0, with a largest
  checked difference of 2.4e-7.
- **Integration pass 1 (f33ff8d)**, before the generic merge and the re-pins,
  gives the table's counts in all five runs.

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
  in `Float` (section 9). <!-- PASS2: the number of Float(128) known inaccuracies on the suite at the pass-2 head after known-failures.json is re-derived. -->
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

### 8.4 Formerly ignored binary64 tests

Four tests were ignored as "needs fast accuracy fix" at 8346699. Two pass
with the default engine since the box geometry (611f6dc). The two V6 tests
pass with the opt-in `conditioned_cuts` since d0819f0, and their
default-geometry variants are ignored.

| test | at 8346699 | now |
|---|---|---|
| `rank_n2_reference::suite_fast_within_ten_times_generic_binary64` | 1 of 324 outside: `rn2-shifted-mixed-n7-r9-mu2` lane 0 (condition 7.9e9, report-only in V4f), fast 1.7e-3 against generic f64 1.6e-4. Its rank n+1 twin on the rank <= n+1 path showed the same ratio (fast 3.2e-9, generic 3.1e-10) | 0 of 333 outside; that lane at 1.3e-5; largest `e_fast / limit` 0.84 (box geometry) |
| `rank_n2_invariance::relabelling_invariance_rank_n2_binary64` | `rn2-exact-mu-n7-r9-perm` lane 1: 8.0e-7 against 10 times the noise, 5.0e-8 | 84 of 84 (box geometry) |
| `rank_n2_identity::pieces_reproduce_the_rank_n2_references_binary64` | `rn2-id-pinch-n7` lanes 0, 2, 3, 6 at 5.8e-9..1.6e-8 | passes with the pieces through `FastOptions::conditioned_cuts`; largest 1.5e-11 |
| `rank_n2_identity::rank_n2_integrals_equal_their_pieces_binary64` | the same pinch lanes, and `rn2-id-pinch-n4` lane 3 at 5.3e-9 against 3.7e-9 | passes (pieces with `conditioned_cuts`, sides through the rank n+2 engine); largest 3.8e-9 |
| `rank_n2_identity::{pieces_reproduce_the_rank_n2_references,rank_n2_integrals_equal_their_pieces}_binary64_default_geometry` (new, 414a57a) | — | ignored, "needs fast accuracy fix (default geometry)": with original Ninja's bubble reference and tadpole bases, the n = 7 pieces stay at 5.8e-9..1.6e-8 |

`pieces_reproduce_the_rank_n2_references` still asserts the default path for
n <= 6. The `rn2-id-pinch-n7` kinematics have propagator pairs whose shift
difference has a zero time component: (0,3), (1,5) and (2,6), none at
n <= 6. Those are the bubbles on which the default rank <= n+1 fast engine,
like original Ninja, loses accuracy. The new
`controls_fast_conditioned_binary64_match_references` checks the option on
the controls: 600 pass and 15 are report-only, as by default.

<!-- PASS2: confirm at the pass-2 head that these four tests pass and that the two *_default_geometry variants are the only ignored tests of the rank_n2_* files, and give the ignored-test count of the workspace run (8.2, V1). -->

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

The final review of the rank n+2 work and the work that followed it found the
limitations below. Four are fixed, and a fifth, a performance outlier, turned
out to be heap layout rather than code (9.1). The others are open (9.2-9.4).

### 9.1 Fixed or explained

**DoubleFloat and Float(128) bubbles at small |p²| with unequal masses**
(fixed in 6487309 and c3a6aba).
- B0, B1, B11 and B111 came from oneloopmaster's exact expressions. The
  Passarino-Veltman B1, B11 and B111 divide by powers of p². Even the scalar
  B0's `(m0²−m1²)/p²` logarithms cancel, losing about
  `log10(|m0²−m1²|/|p²|)` digits: 2.1e-21 relative in DoubleFloat on the
  review's case.
- In DoubleFloat and Float(128) these bubbles were silently wrong. B0..B1111
  now come from oneloopmaster's native tensor families for every carrier, as
  in binary64 (section 6).
- Scaled differences against the review's 50-digit references (p² = 1.86e-9,
  m² = 25 and 2025):

| rank | f64 (unchanged; generic / fast) | DoubleFloat: review → 6487309 → c3a6aba | Float(128): review → 6487309 → c3a6aba |
|---|---|---|---|
| 2 (`repro-smallp2-bubble-r2.json`) | 3.3e-13 / 4.8e-13 | 0.63 → 1.6e-18 → 8.3e-29 | 9.9e-8 → 3.3e-25 → 7.8e-36 |
| 4 (`repro-smallp2-bubble-r4.json`) | 4.3e-8 / 2.7e-8 | 1.3e13 → 2.2e-15 → 2.8e-24 | 2.0e6 → 5.4e-22 → 8.3e-31 |

- The rank-4 DoubleFloat floor, 2.8e-24, is 3.9 × 2^-104 times the lane's
  condition (1.5e7). Every intermediate of the DoubleFloat run is within
  1e-31 relative of Float(512), except the heavy line's tadpole c0 (2.6e-30):
  it is 4e5 and comes from bubble terms of order 1e9. The cut contributions
  (2.7e9) cancel to the lane's 2.6e3.
- Both reproducers are tests (`tests/rank_n2_float_bubbles.rs`, data in
  `tests/rank_n2_data`). So is B0..B111 of every carrier against the
  Feynman-parameter fixture (`tests/masters.rs`).
- On the review's 26-lane bubble scan, DoubleFloat passes every lane with a
  finite reference. Float(128) passes all but the threshold lanes (9.3,
  "Float(128) near thresholds").

**`--stability-test` and cancellation inside one cut contribution** (fixed in
22c332c and 9ff3d0a). `cancellation_stable` compares the result with the
summed moduli of every term of every cut contribution:
- each master integral times its weight, per Laurent coefficient
  (`ResidueIntegrator::term_moduli`);
- each additive term of the rational part, from a Symbolica kernel of the
  rational weight's terms that is compiled only with the test.

Without the test nothing extra is evaluated, and with it the results are
unchanged. The confusion counts below are against the generic engine in
Float(256), on the review's 170 lanes, their 166 twins, 32 scan and Euclidean
lanes, the suite and the controls (1286 lanes, measured at 9ff3d0a). "Wrong"
means an error above 2^-(p/2).

| precision | flagged / missed / flagged but right, before → after |
|---|---|
| binary64 | 54/30/0 → 59/25/2 |
| DoubleFloat | 11/10/0 → unchanged |

- The two flagged-but-right lanes are off by 1.1e-8 and 9.1e-9, just under
  1.5e-8.
- The review's `repro-stability-test-miss-n6-r8.json` (`fr-light-zero-n6-r8`
  lane 2, generic f64 24 % wrong) is `Unstable`.
- **The remaining misses** are off by 6 to 3000 times their term cancellation
  times the unit roundoff. Their cut coefficients are already inaccurate (for
  example `fr-mixed-complex-n4-r6` lane 1: 6.1e-6, with terms cancelling by
  4.6e7), which no measure of the summation can detect. A threshold factor or
  an a-posteriori test such as Ninja's N=N test would be the next step.
- **A false negative: `rn2-phys-real-n6-r8` lane 2.** With the generic
  selection (86b378b) its generic f64 error is 1.3e-6, above its binary64
  budget of 5.9e-7, yet it keeps more than half the bits between cuts. With
  the per-term measure merged,
  `case_check.rs::stability_test_flags_rank_n2_cancellations` still expects
  lane 1 of `rn2-phys-{real,complex}-n6-r8` to be the only `Unstable` lanes,
  so its loss is not in the summation that the test measures. <!-- PASS2: confirm that stability_test_flags_rank_n2_cancellations and stability_test_flags_cancellation_inside_cut_contributions pass at the pass-2 head (they were not run on the merged generic selection plus per-term test before pass 2). -->
- The fast engine has no equivalent test.

**Generic binary64 on ill-conditioned light-like lanes** (final review,
finding 5; fixed in 86b378b for rank n+2 and in 5ed3b22 for every rank class;
its cost reduced in f9a8b16).
- **Cause: the scores, not the candidate count.**
  - The generic bubble score
    `min((|k.r|/|k|)², min_v |G(k,r,v)|/((k.r)² v²))` grows as |k.r| shrinks.
  - Its tadpole score also penalised e2 and e4, which are never expanded in.
  - Forced to the generic choices, the fast engine loses as much:
    geometric-mean error ratio 3.4 on the review's 41 lanes, 2.83 on 830
    fresh lanes.
  - With the generic eight references but its own score, it does not (1.02).
- **Fix.** At every rank class the generic provider uses the fast engine's
  scores, over 24 bubble references and its six tadpole bases (section 7.2).
- **Rank n+2, review lanes.** At main 0d0d7e7, take the lanes with
  max(fast, generic before, generic after) > 1e-9: 42 lanes.
  - Generic f64 is worse than fast on 26 → 20 of them, and more than 10 times
    worse on 9 → 5.
  - The geometric-mean ratio goes from 2.56 to 1.07.
  - On fresh lanes from the review's generator the ratio after the fix is
    1.10 (830 lanes, seeds 101-105; before 2.95) to 1.30 (498 lanes, seeds
    301-303; before 2.68), depending on the seed set.
- **Rank n+2, after against before**, over all lanes with an error above
  1e-13: geometric mean 0.535 on the review lanes (89 lanes; 15 more than 10
  times better, 2 more than 10 times worse) and 0.531 on the fresh lanes (7
  more than 10 times worse). Single lanes can lose up to about 700 times:
  `fr-phys-complex-n6-r8-mu2` (seed 302) lane 2 goes from 4.8e-13 to 3.3e-10
  (fast 6.2e-13).
- **Rank <= n+1** (generic f64 against Float(256) of the same build; lanes
  with an error above 1e-13 before or after; `validation/regression/README.md`):
  - controls: 192 of 615 lanes, geometric mean 0.715; 8 more than 10 times
    better, 0 more than 10 times worse;
  - 1494 twins of the review's kinematics: 718 lanes, 0.661; 54 better, 11
    worse; median 5.6e-14 → 4.4e-14; p99 1.3e3 → 3.0e2;
  - the worst single lane, `fr-light-mixed-n4-r6-mu2-s301` lane 1 (twin),
    goes from 2.1e-3 to 7.1 (fast 1.5e-4). The other ten lose at most 26
    times.
- **Cost:** at most +1.5 % instructions per point on the generic bench after
  f9a8b16 (section 7.2).
- **What remains is not in the bases.**
  - On 14 of the review's 41 lanes the generic error sits in the box (7) or
    triangle (7) stage.
  - `fr-light-zero-n5-r7` lane 2 goes from 3.6e-8 to 1.7e-8 (fast 3.7e-10).
    Forcing each of the 24 references on its {0, 2} bubble gives 7e-11 to
    1e-6 in either engine, without order in the score, and the two engines
    differ by up to 50 times for the same reference.

**Declaring rank n+2 for a numerator of lower rank changed the fast engine's
results** (finding 6, fixed in `fast/declared_rank.rs`, dc8c7e7 and
baaa240).
- **Before**, such numerators went through the rank n+2 engine.
  - On the review's 166 rank n+1 twins with `case-check --declare-rank n+2`,
    6 lanes lost more than 10 times against the rank n+1 declaration
    (`fr-light-equal-n5-r7` lane 1: 2.56e-6 instead of 2.61e-9) and 28 gained
    more than 10 times (`fr-mixed-zero-n5-r7-muall` lane 0: 1.9e-13 instead
    of 1.7e-10). The geometric mean of the error ratio was 0.49.
  - On the 615 suite control lanes declared at n+2, 64 lanes gained more than
    10 times (`rn2-shifted-mixed-n3-r4` lane 2: 1.3e-14 instead of 3.2e-9)
    and none lost more than 7.0 times (`rn2-phys-complex-n6-r7` lane 1:
    2.21e-7 instead of 3.14e-8). The geometric mean was 0.49.
- **Cause.** A diagnostic build forced Ninja's bubble references and tadpole
  bases in the rank n+2 engine. With them, only 1 twin lane loses and 2 gain
  more than 10 times, so the differences come from the basis choices of
  `bases_x2.rs`, not from the declared rank itself. On `fr-light-equal-n5-r7`
  lane 1, the tadpole basis accounts for a factor 64 (4.0e-8 with Ninja's
  tadpole bases) and the rest of the rank n+2 algorithm for 12.
- **Now** a lane declared at rank n+2 is reduced bitwise as declared at its
  own rank (section 7.2). That equals the undeclared run when the case's rank
  is the lane's rank, as for every lane of the twins and controls: 166/166
  twin lanes and 615/615 control lanes are identical to the undeclared runs
  (control verdicts 600/0/15/0).
- **The gains above are given up by default**, because the rank <= n+1 engine
  does not use the conditioned bases by default (9.2, "Fast-engine
  conditioning").
  - `AtDeclaredRank` keeps the rank n+2 engine, at about 2 to 9 times the
    instructions (geometric mean 4.5 on 17 padded cases).
  - `FastOptions::conditioned_cuts` brings the conditioned bases to the
    lowered lanes without over-declaring; on the controls its error is 0.46
    times the default's in geometric mean.
- **Follow-up for the basis scoring:** on light-like equal-mass kinematics the
  tadpole score picks a basis 64 times worse than Ninja's
  (`fr-light-equal-n5-r7` lane 1).

**`syn-n2-r2` at width 1: heap layout, not code placement** (73f53d9). The
first guard measured this key at 1.061, and 1.11 rerun alone. The rerun of
2026-10-19 (`validation/performance/rank-n2-ab/`,
`results-0d0d7e7/layout-investigation.txt`) found that the cause is the heap
layout of the case-bench worker thread, not where the rank n+2 code is
linked.
- Placement changes did not help. Linking the 75 rank n+2 symbols first,
  `#[cold]` on 42 rank n+2 functions, and `-z keep-text-section-prefix` leave
  8346699 at 1.08–1.14 against dec3d55.
- Padding the case file, which shifts the worker's later heap allocations,
  moves 8346699 between 462 and 512 ns (dec3d55: 461–489 ns). With 64-byte
  aligned heap blocks it runs at 0.98 of dec3d55.
- Main 0d0d7e7 has the same hot-function layout as 8346699 and runs this key
  at 0.95–0.99 of d53dd08.
- Over heap layouts, the width-1 replayed reductions of all 27 cases move by
  2 % at the median and by up to 10 %.
- No remedy was adopted. Check a width-1 ratio above 1.03 over heap layouts
  (`scripts/layout_pad.sh`) before attributing it to code.

### 9.2 The engines

**Fast-engine conditioning.** The rank n+2 engine conditions its bubble
references and tadpole bases (79f306a) and its box geometry (611f6dc). By
default the rank <= n+1 path keeps original Ninja's bubble reference, tadpole
bases and box geometry. With `FastOptions::conditioned_cuts`
(`--conditioned-cuts`) it takes the rank n+2 engine's choices (section 7.2).
- **The default engine at n = 7.** Rank <= n+1 lanes with a propagator pair
  whose shift difference has a zero time component lose accuracy, as in
  original Ninja. The n = 7 pinch pieces are off by 5.8e-9..1.6e-8, against
  <= 1.5e-12 for generic f64. The ignored `*_default_geometry` V6 tests track
  this.
  - With the option they are off by 6.8e-14..7.6e-12.
  - The option costs +29.8 % instructions in the replayed rank <= n+1
    reduction, moves the cut-constructible/rational split away from original
    Ninja's, and makes 20 control lanes more than 2 times worse, up to
    1.1e-10 (11 times generic f64).
- **Over-declared lanes.** The same lanes declared at rank n+2 through
  `AtDeclaredRank` are often more accurate than at their own rank: 64 of the
  615 suite control lanes are more than 10 times more accurate (finding 6,
  9.1).
- **Remaining rank n+2 misses.** 13 of the 426 lanes of the rank n+2 set
  (section 7.2) are outside `max(10 × generic f64, 1e-13)`, at
  1.3e-13..7.9e-11.
  - The largest is `rn2-int-mixed-n5-r7` lane 2 (generic f64 7.0e-12). The
    largest ratio, 126, is on `rn2-exact-mu-n4-r6-perm` lane 5 (5.2e-13).
  - On `rn2-exact-mu-n4-r6-perm` lanes 1, 5 and 6 (4.8e-13..5.2e-13) the
    stage sums of 611f6dc put the loss in the triangle stage, whose `e3`,
    `e4` still come from Ninja's spinors.
  - The others lose them in the bubble and tadpole stages. All 13 were
    outside with the spinor box geometry too.
- **Ill-conditioned lanes.** Both binary64 engines lose digits there. On
  `rn2-shifted-mixed-n7-r9-mu2` lane 0 (tier A, condition 7.9e9) the fast
  engine is off by 1.3e-5 and generic f64 by 1.6e-4. DoubleFloat is off by
  5.9e-21 and Float(256) by 1.7e-30. For accuracy-critical points, use the
  generic engine in DoubleFloat or `Float`, or the stability test; the fast
  engine has no cancellation flag of its own.

**Generic binary64.**
- The three control outliers are gone with the common selection (5ed3b22).
  Generic f64 is now within 12.7 times original Ninja's error on the control
  lanes with a Ninja result, apart from the first lane below:
  - `rn2-phys-real-n6-r7` lane 2: from 4.26e-7 (633 times Ninja's error) to
    1.37e-8 (20 times); its known-inaccuracy entry is removed;
  - `syn-n2-r3` lane 1: from 3.0e-10 to 2.0e-12;
  - `rn2-phys-complex-n6-r7` lane 2: from 1.03e-7 to 7.3e-10.
- The new rank n+2 terms can cancel far beyond COLLIER's condition: 5e7 on
  `rn2-phys-real-n6-r8` lane 2 and 1.4e6 on `rn2-phys-complex-n6-r8` lane 2.
  - There, generic f64 is off by 1.3e-6 and 6.0e-6 (1.1e-3 and 5.4e-4 before
    86b378b), and Float(256) by about 5e-9.
  - `case-check --stability-test` marks lane 1 of both cases `Unstable` in
    binary64. Lane 2 keeps more than half the bits between cuts and passes;
    on the real case that is the false negative of 9.1.

**Bubbles with a small momentum in every component and unequal masses** (V9
`rn2-scan-bubp2-complex-n2-r4`: p = (5a, 0, 0, 3a), m² = 29929 − 259.5i and
6400 − 168i).
- On the bubble cut the loop momentum grows like (m0²−m1²)/(2 k·e2) as k → 0.
  At p² = 1.8e-3 (k24), |a0| = 1.9e5 and the bubble's c0 is 8e14. The cut
  contributions (4e14) then cancel to the lane's 197, while COLLIER's
  condition stays below 7.
- Every fixed precision loses the same relative accuracy from k8 to k24:
  generic f64 9.7e-13 → 8.0e-3, fast 8.5e-13 → 9.8e-4, DoubleFloat 3.2e-28 →
  1.4e-18, Float(128) 4.7e-35 → 7.6e-25.
- DoubleFloat/f64 is about 2^-51 on every lane, and every DoubleFloat
  intermediate is within 3e-31 of Float(512): the loss comes from the
  reduction, not the masters. Float(256) agrees with the references to
  <= 3e-40.
- Equal masses lose less: DoubleFloat 6.2e-29 and f64 1.5e-12 at k24.
- The p = 0 lanes fail with `DegenerateReference` / `RepeatedPropagator`.
- The scan is report-only. Use Float(256) for such points. A remedy needs
  another bubble parametrization for |k| → 0.

### 9.3 Master integrals

**Exact masters in Float(128).** `Float` evaluates C0/D0 with Symbolica's
precision-tracking interpreter of oneloopmaster's exact expressions, while
binary64 and DoubleFloat use its native code. At 128 bits that costs up to 10
bits on boxes: on the 139 boxes of the 15 lanes first listed as known
inaccuracies, D0 is off by a median 471 and at most 1.4e5 units in the last
place, against 34 and 4670 in binary64 (C0: at most 4 bits more).
- 3 suite lanes (`rn2-int-mixed-n4-r6` lane 2, `rn2-int-mixed-n6-r8` lane 0,
  `rn2-int-mixed-n7-r9` lane 1) and 11 control lanes miss the binary64 budget
  scaled to 128 bits by 1.0 to 10 times (worst 1.7e-31). They are listed as
  generic Float(128) known inaccuracies in `known-failures.json`.
  `rn2-irml-zero-n5-r6` lane 1 went from 4.3e-32 to 1.2e-32 with the common
  selection of 5ed3b22, inside its 2.7e-32 tolerance, and was removed.
  <!-- PASS2: the Float(128) known inaccuracies after known-failures.json is re-derived at the pass-2 head (oneloopmaster 8ec256f, Symbolica 9cdc2949): suite and control lane counts, worst miss, entries removed or added. -->
- From 160 bits on, the references' uncertainty hides this loss.
- The tracked-precision check of section 6 passes these masters: NinjaRS's
  recorded C0 and D0 calls keep at least 101 tracked bits at 128 bits.

**Float(128) near thresholds.**
- numerica's `Float` tracks significance through cancellation. For exact
  inputs near p² = (m0+m1)², the native family computes the discriminant
  (m1−m0−p²)² − 4p²m0 exactly, but keeps it (rounded) at about 106 of 128
  bits: at p² = 100.000019 with m² = 25 it tracks 106 bits, and B0's
  imaginary part keeps 110.
- The review's `scan-thr±` lanes (p² = 4m²(1 ± 1.9e-7)) are 1.0–3.0 times
  the binary64 budget scaled to 128 bits (4.6e-36 against 2.8e-36), before
  and after c3a6aba.
- DoubleFloat, which does not track significance, passes them (1.3e-31), and
  so does Float(256).

**Master integrals at small `p²`.**
- Every carrier takes B0..B1111 from oneloopmaster's native families.
  binary64 goes through `OneLoopNativeMasters`; DoubleFloat and `Float` go
  through `SymbolicaMasters`. `SymbolicaMasters<f64>` agrees with
  `OneLoopNativeMasters` bitwise.
- On the Feynman-parameter fixture (220 rows, |p²|/m² down to 1e-14),
  B0..B111 are within 7.8e-32 (DoubleFloat) and 1.5e-33, 4.3e-72, 2.7e-149
  (Float(128), Float(256), Float(512)) of the row scale. binary64 is within
  1e-15, or 7.2e-14 within 2^-40 of a (pseudo-)threshold.
- Dual carriers use the exact forms through the interpreter, with the
  small-p² loss of section 6; the exact B0 loses about
  `log10(|m0²−m1²|/|p²|)` digits.
- `SymbolicaDualMasters` keeps its existing derivative domains.

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

**Rescue at higher precision.** The production model is escalation: a point
whose binary64 result is flagged is recomputed in DoubleFloat or `Float`. The
property that matters most is that raising the precision rescues points
consistently. For oneloopmaster's C0 and D0 this was measured on the sweeps
fs, ns and ns2 of section 6 (100,680 rows; the OneLOop and COLLIER comparison
of 2026-10-21, with oneloopmaster 3a322d0, the code of 8ec256f, against
09b2c74). Per binary64 route, 236-250 referenced rows of fs, 627-746 of ns
and 367-391 of ns2 are wrong in binary64 (above 1e-6 or non-finite). Of
these:
- DoubleFloat gets 90-98 % right (fs 97.5-97.6 %, ns 94.3-95.2 %, ns2
  90.2-91.0 %), `Float(128)` 79-96 %, and `Float(256)` and `Float(512)`
  95-99.6 %, on the native and interpreted routes alike. At 09b2c74 all
  levels together rescued 15-33 %.
- 47 rows are rescued at no level up to 1024 bits (fs 1, ns 28, ns2 18).
  42 come from the planar dyadic family (light-like momenta in a plane, so a
  vanishing Gram determinant, some with a small dyadic transverse kick), 4
  have a vanishing channel (s12 or s23 = 0), and 1 is another box. 45 of them
  are NaN at every level, and OneLOop in quad precision is NaN on 43.
- A higher level can be wrong where a lower one was right. On 298 ns rows,
  179 ns2 rows and 1 fs row (09b2c74: 541, 317 and 42), DoubleFloat,
  `Float(128)` or `Float(256)` is within 1e-10 and a higher level up to 512
  bits is wrong. Most are `Float(128)` after a right DoubleFloat (249 ns and
  154 ns2 rows).
- Such wrong `Float` values usually track few bits. This is why
  `SymbolicaMasters` fails `Float` results below `max(53, bits/4)` tracked
  bits (section 6): a rescue loop then escalates further instead of accepting
  them. The check cannot see the 6 sweep rows that are wrong while reporting
  81 to 1023 tracked bits (section 6).

**OneLOop and COLLIER on the same sweeps** (D0 rows against COLLIER-based
references, C0 rows separately). OneLOop 3.6 is wrong (above 1e-6 or
non-finite) on 14-17 % of the referenced D0 rows in double precision and on
about 10 % in quad precision. Most of these are the soft boxes in non-cyclic
labellings, which 09b2c74 missed as well, and on 105 rows quad OneLOop
returns the same wrong value as 09b2c74 at 512 bits (3a322d0 is right on all
105). COLLIER 1.2.9's COLI is the most robust D0 (0 to 0.13 % of the rows
wrong, against values on which both oneloopmaster builds agree at 256 and 512
bits), but it is wrong on 14-16 % of the ns and ns2 C0 rows, mostly with a
vanishing Källén function, where it mostly raises its error flag; its DD
branch is wrong on 4-9 % of the D0 rows and 6-21 % of the C0 rows.

**oneloopmaster's open tail.** oneloopmaster main 8ec256f landed with its
remaining regressions against 09b2c74 documented as open work (decided on
2026-10-21; the counts are in section 6). None of NinjaRS's
recorded D0 calls is more than ten times worse than at 09b2c74. The
follow-ups of oneloopmaster's `STATUS.md` are:
- binary64 should return NaN where the second contour (oneloopmaster
  e72b404) is not resolved; the boxes of that chain turn 09b2c74's NaN into
  finite wrong values;
- `Float` should return NaN, or check the chart as binary64 does, where a
  contour's K or the Gram determinant is at the working precision's rounding
  level. These are the silent `Float` rows; NinjaRS's tracked-precision check
  now catches those that track few bits;
- a per-row `Float` comparison against the previous main in oneloopmaster's
  oracle-sweep gate;
- the D0 instruction gate's headroom, 0.04-0.06 % on syn-n6-r7;
- the rest of the tail: the regressions that are not silent, the seed-6 box
  and the soft boxes listed there.

Older open items of oneloopmaster (`STATUS.md`, "Open D0 limitations" and
"Open C0 limitations"):
- opposite-mass two-mass boxes next to collinear limits (6.2e-9 native,
  9.6e-9 interpreted);
- a real negative m², which is not the −i0 limit;
- SymJIT below |Im m²|/Re m² ≈ 1e-150;
- the corner-singular box (NaN), and a Gram determinant at the rounding level;
- `Float` three-mass boxes at 128 and 192 bits;
- C0 with equal masses, on-shell legs and a vanishing invariant (`Float`),
  and interpreted `Float` C0 at 112-224 bits;
- C0(0,0,0; m_complex, 0, m_real), which is wrong on every route, binary64
  included.

### 9.4 Coverage and scope

**Coverage.**
- In binary64 every n = 7 mu² suite lane is report-only (condition 1e9 to
  1e18), so binary64 checks 9 n = 7 suite lanes.
- The `phys` family at n >= 4 is mostly report-only in Float(256), because
  its uncertainty is the input ambiguity of snapped S-matrices. A change of
  the generic bubble reference alone moved Float(256) on such lanes by up to
  5.1e-8 (86b378b, section 8.3).
- Against original-Ninja references at the default 1e-8, case-check fails 5
  lanes of 4 synthetic cases where Ninja is the inaccurate code (e.g.
  `syn-n6-r6` lane 0 at 0.26). Judge NinjaRS with the high-precision
  references.

**Native kernels.** The `native-kernels` feature has no native degree-4
triangle-to-bubble correction. Rank n+2 bubbles keep the interpreted plan, and
`kernel_backends()` reports it.

**Original-Ninja export.** `export_ninja_cpp` and the CLI's reference export
reject rank n+2, because original Ninja cannot reduce it.

**Performance guard.** The guard of design §8 was rerun on 2026-10-19
against main d53dd08, with main 0d0d7e7 as the candidate and oneloopmaster
09b2c74 on both sides (`validation/performance/rank-n2-ab/`,
`results-0d0d7e7/`; 73f53d9). It passes:
- the geomean of the reduction and total ratios is 0.9996 (A/A 1.0003);
- 191 of 192 keys are <= 1.03; `gg-hhh-top-box-8` width-1 reduction is 1.038,
  and 1.012 rerun alone;
- instructions per point are within 0.5 %, except one point that only the
  probe binary shows (+0.503 %);
- the generic `bench` is at most 1.011, and the absolute gates against Ninja
  are unchanged (0.60 / 0.86 / 0.68).

The first guard, against dec3d55 on 8346699, failed one key (`syn-n2-r2`
width 1 at 1.061), a heap-layout effect (9.1). The benchmark report has a
rank n+2 section against COLLIER 1.2.10 (section 5,
`docs/benchmark/rank_n2.typ`; 7d9c01f, b992fbc).

<!-- PASS2: the cycle-timed guard of pass 2 (validation/performance/rank-n2-ab/scripts/run_rerun.sh, A = main 0d0d7e7, B = the pass-2 head): geomean of the reduction and total ratios, A/A, keys above 1.03 and their rerun-alone ratios, the generic bench maximum and the absolute gates against Ninja; or state that it was not rerun and why. -->

## 10. Errata and departures from the design

The design's §10 errata, and how the implementation settled them:

| # | erratum | status |
|---|---|---|
| 1 | OneLOop B1111 sign wrong in every swap branch | corrected in oneloopmaster's native family (fe550e3); swap-identity regression tests in oneloopmaster and NinjaRS (section 6) |
| 2 | Δ₀ depends on the tadpole basis: with per-tadpole bases it is non-zero already at r = n+1 (n >= 2) | integrals unaffected; reconstruction tests check per cut. The `tools/rank_n2/theory/common.py` docstring now states `decomposition_check.zero_point_bound` (e116813) |
| 3 | Ninja's `SizeHelper<2>` (`1 + i + i/2 + i/3`) is exact only for i <= 3; level 4 has 9 entries, not 8 | `t2_len = (i/2+1)(i+1-i/2)` (f7e8bad); `t2_len_le3` keeps Ninja's form at the rank <= n+1 call sites (61079d4) |
| 4 | the phase-1 references preprocess in binary64 (good to 1e-16..1e-13 on `-mu2` and shifted lanes) | regenerated with exact rational preprocessing and exact decimal driver inputs (498432f); the review's `hpref` differs by <= 1.4e-11, explained by its input rounding and COLLIER's error |
| 5a | native B1111 truncation tables | own rank-four radii 0.0085 and 0.17 (fe550e3; the design proposed 0.009 and 0.17) |
| 5b | small-p² loss of the exact B1111 | native family for binary64 and DoubleFloat (07395b2) and for `Float` at the row precision (bd2022b); B1..B111 (6487309) and B0 (c3a6aba) from the native families for every carrier as well; Dual keeps the exact forms |
| 5c | stability of the new divisions | PR #3's precision-aware checks in both engines (1524dbd); V9 scan; fast-engine basis conditioning (79f306a) and box geometry (611f6dc); the generic engine's selection (86b378b, 5ed3b22) |
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
- **Basis conditioning** (79f306a) and the **box geometry** (611f6dc) are
  new. The design kept original Ninja's bubble reference, tadpole bases and
  box normal.
- **`FastOptions::conditioned_cuts`** (d0819f0, ec2df0f) brings that geometry
  to rank <= n+1 as an opt-in, as a separate engine (section 7.2).
- **Lanes of a lower rank** declared at rank n+2 are reduced at their own rank
  (dc8c7e7, baaa240); the design reduced every lane at the declared rank.
- **The generic engine's bubble and tadpole selection** uses the fast
  engine's scores at every rank class (86b378b, 5ed3b22, f9a8b16); the design
  kept the generic engine's own selection.
- **`Float` B1111** is native at the row precision, not the exact expression
  (bd2022b); so are B0..B111 for every carrier (6487309, c3a6aba).
- **`Float` C0/D0** are relabelled for soft boxes (8e583f2) and fail when
  they track fewer than `max(53, bits/4)` bits (2a7df0d; section 6).
- **V0 baseline.** The fixtures were taken at e4c16a7, not 5f35b07 (96b3653).
  Above rank 5 the generic numerators keep a fixed sixteenth of the monomials.
  The file name `r-le-n1-d53dd08.json` is kept. The fixture was re-blessed for
  main's own changes from a clean checkout of main, and in place for the
  generic engine's accuracy changes and for dependency re-pins carried by the
  integration branch, each time with a written justification in
  `validation/regression/README.md`:
  - main dec3d55 (PR #3, merged in 1524dbd): the fast engine's
    `TensorNumerator` and `MuTensorNumerator` bits unchanged; generic binary64
    rounding moved by <= 2.1e-12 of the lane maximum, and by <= 1.0e-11 with
    explicit mu²;
  - main d53dd08 (oneloopmaster 264a6ce → 09b2c74, merged in e53c298): C0/D0
    changes move 69 of 237 fast width-1 lanes and 71 of 237 generic binary64
    lanes, median 2e-14, at most 8.8e-5 on the cancelling
    `syn-gghhh-ggHHH-n5-r5` lane 1 (a D0 next to a Gram determinant of 8.2e-8
    moved by 4e-11); no status changed;
  - B1, B11 and B111 from the native families (6487309, main): only the
    generic Float(256) section, 43 decimal components in their last printed
    digit (about 1e-76 relative);
  - B0 from the native family (c3a6aba): only the generic Float(256) section,
    37 of 170 components in 12 of 17 lanes, by at most 2.4e-69 of the lane's
    largest component;
  - the generic selection at every rank class (5ed3b22), justified by a
    symmetric statistic against Float(256) (controls: geometric mean 0.715, 8
    lanes more than 10 times better and 0 worse; 1494 twins: 0.661, 54 and
    11): generic binary64 moves on 203 of 237 lanes, explicit mu² on 23 of 26,
    Float(256) on 91 of 170 components; the fast sections are unchanged;
  - c24f976, once, for four causes together: (a) B0 from the native family
    (Float(256) only, 43 components, <= 1.8e-70); (b) the generic selection,
    already in the previous fixture; (c) oneloopmaster 09b2c74 → 8ec256f
    (every changed fast and binary64 entry and 23 Float(256) components,
    <= 1.8e-67); (d) Symbolica e679e8c1 → 9cdc2949 (Float(256) only, 82
    components, <= 5.4e-67). Against the previous fixture, the fast
    `TensorNumerator` moves on 62 of 237 width-1 lanes and 607 of 2112
    width-32 entries (median 4.5e-15), generic binary64 on 60 of 237 lanes,
    explicit mu² on 6 / 6 (fast) and 8 (generic) of 26, at most 4.8e-4 on
    `syn-gghhh-ggHHH-n5-r5` lane 1, whose error against the exact value goes
    from 4.6e-4 to 3.1e-4 (fast) and from 4.7e-4 to 7.8e-6 (generic). No
    status changed. The fast sections equal those of a fixture blessed on
    main 0d0d7e7 with only the two re-pins applied, so the fast engine moves
    only through (c). The relabelling and the tracked-precision check change
    nothing in V0.
  - **Next: a precision bound for the generic Float(256) section.** The
    generic Float(256) bits depend on the dependencies' `Float` rounding
    (cause (d) above) and, through oneloopmaster's Symbolica evaluators, on
    the process's symbol-registration history, not only on NinjaRS. V0
    will compare that section within a working-precision bound of the lane
    scale, with statuses exact, while the deterministic sections (fast
    engine, generic binary64 and explicit mu²) stay bitwise. The four causes
    of c24f976 together moved it by at most 3.6e-67 of the lane's largest
    coefficient (`syn-n6-r7` lane 1).
    <!-- PASS2: the commit that makes V0 compare the generic Float(256) section within a bound, the bound itself (e.g. 2^-200 of the lane scale) and its justification as written in regression_bitwise.rs and validation/regression/README.md. -->
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
| generic plans | `crates/ninjars-symbolica/src/{laurent,projection,subtraction,box_kernels,tadpole_kernels,contraction,integration}.rs`; bubble references and tadpole bases in `cuts.rs` |
| `MuShifted` | `crates/ninjars/src/numerator.rs`, `crates/ninjars-symbolica/src/{polynomial,dual_polynomial}.rs` |
| fast engine | `crates/ninjars/src/fast/{engine_x2,bases_x2,box_x2,declared_rank,subtractions_series_x2,subtractions_x2}.rs`; the rank <= n+1 opt-in `FastOptions::conditioned_cuts`: `ConditionedEngine` in `fast/engine.rs`, its stages in `fast/engine/conditioned.rs`, generated by `fast/tests_conditioned.rs` |
| fast numerators | `crates/ninjars/src/fast/{numerator,tensor,tensor_mu,tensor_jets,adapters,replay}.rs` |
| masters | `crates/ninjars-symbolica/src/{masters,oneloop_native}.rs` (the `Float` route's soft-box relabelling and tracked-precision check in `masters.rs`); oneloopmaster `src/native/tensor.rs` |
| CLI | `crates/ninjars-cli/src/case_check.rs`, `case_check/harness.rs` |
| derivations, review, master checks | `tools/rank_n2/` |
| fast generators | `tools/fast/{generate_jets,generate_subtractions_x2,make_subtraction_fixtures_x2}.py` |
| references, controls, identities, policy | `validation/rank-n2/` |
| rank <= n+1 regression fixtures | `validation/regression/` |
| COLLIER benchmark, rank n+2 | `validation/benchmark/collier/`, `docs/benchmark/rank_n2.typ`, `docs/benchmark/data/rank_n2.json`; COLLIER's cache defect: `COLLIER_CACHE_BUG.md` |

## Sources

Numbers cite the test or file that produces them, or the commit whose message
records them. Some come only from the implementation and verification reports
of the rank n+2 work (2026-10-18 to 2026-10-21), which are not in the
repository:

| report | content |
|---|---|
| design | the implementation design (decisions D1-D9, §1-§10) |
| theory, review | phase-1 derivation and independent referee; their artifacts are in `tools/rank_n2/` |
| S0, S2, S3, S4, S5b | stream reports; S2 → c066178, S3 → 1d5cb5f and ebdbc00, S4 → 07395b2, S5b → 3c6c88c |
| S2, S3, S4, S5b verifiers | independent re-checks of each stream |
| merge run | the integration head 8346699: workspace tests and `case-check` on the whole suite in four configurations |
| final review | the review of 8346699 that found the arbitrary-precision D0 bug and the limitations of section 9 (reproducers named there) |
| integration run | e116813, after merging main d53dd08 (oneloopmaster 09b2c74): V0 regenerated, known failures re-derived, `case-check` on the suite and the controls in five configurations, workspace tests |
| follow-up streams (2026-10-19/20) | generic conditioning (86b378b, 5ed3b22) and its cost (f9a8b16); over-declared rank (dc8c7e7, baaa240); code placement (73f53d9); stability test (9ff3d0a); fast n = 7 accuracy (611f6dc, d0819f0) and its compile-out (ec2df0f, a0e7b50); DoubleFloat at rank 4 (c3a6aba); COLLIER benchmark (7d9c01f, b992fbc) |
| follow-up verifiers | independent re-checks of each follow-up stream; their corrections are applied here |
| integration pass 1 | f33ff8d, before the generic merge: machine code and instruction counts against main 0d0d7e7, `case-check` in five configurations, all crate test suites |
| oneloopmaster landing | the oneloopmaster final candidate 3a322d0 and its verifiers (verify-olm-gates2: fs and the oracle-sweep comparisons; verify-olm-sweep2: ns and ns2), the documentation check of a8a3b8c (corrected in 8ec256f); oneloopmaster `STATUS.md`, "Adversarial sweeps and the remaining tail" |
| OneLOop and COLLIER comparison | 2026-10-21: OneLOop 3.6 (double and quad), COLLIER 1.2.9 (COLI and DD), oneloopmaster 09b2c74 and 3a322d0 on fs, ns and ns2, with the rescue ladder over DoubleFloat and `Float` 128 to 1024 bits; it also selected the tracked-precision rule (2a7df0d) |
| integration pass 2 | 8a8bc55..115fb46 (the oneloopmaster and Symbolica re-pins, 8e583f2, 2a7df0d, the V0 re-bless c24f976) and its checks: known failures and `case-check`, tests, performance <!-- PASS2: the pass-2 head and the checks' outcome in one line (what passed, what is documented as open). --> |
