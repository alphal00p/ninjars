// Rank n+2 section of the NinjaRS benchmark report. Numbers are read from
// data/rank_n2.json, produced by validation/benchmark/collier/collect.py.

#let rn2 = json("data/rank_n2.json")

// Helpers (the same conventions as benchmark.typ; an included file does not
// see the main file's definitions).
#let navy = rgb("#1f3b63")
#let pale = rgb("#eef3f9")
#let is-num(x) = type(x) == int or type(x) == float
#let fixed(x, d) = {
  let s = str(calc.round(x, digits: d))
  if d == 0 { return s }
  let parts = s.split(".")
  let frac = if parts.len() > 1 { parts.at(1) } else { "" }
  parts.at(0) + "." + frac + "0" * (d - frac.len())
}
#let fmt(x) = {
  if not is-num(x) { return [—] }
  let a = calc.abs(x)
  let d = if a >= 100 { 0 } else if a >= 10 { 1 } else if a >= 1 { 2 } else { 3 }
  let s = fixed(x, d)
  if a >= 10000 {
    let int-part = str(int(calc.round(x)))
    let groups = ()
    let rest = int-part
    while rest.len() > 3 {
      groups.insert(0, rest.slice(rest.len() - 3))
      rest = rest.slice(0, rest.len() - 3)
    }
    groups.insert(0, rest)
    s = groups.join(sym.space.thin)
  }
  s
}
#let ratio(a, b) = if is-num(a) and is-num(b) and b != 0 { a / b } else { none }
#let palette = gradient.linear(
  rgb("#5aae61"), rgb("#d9f0d3"), rgb("#ffffff"), rgb("#fddbc7"), rgb("#d6604d"),
)
#let ratio-fill(r) = {
  if r == none { return none }
  let t = calc.clamp(0.5 + calc.log(r, base: 2) / 4, 0, 1)
  palette.sample(t * 100%)
}
// Multipliers: two decimals below 10, otherwise significant figures.
#let fmt-ratio(r) = if r >= 10 { fixed(r, 0) } else if r >= 0.1 { fixed(r, 2) } else if r >= 0.01 { fixed(r, 3) } else { fixed(r, 4) }
#let ratio-cell(r, body: none) = table.cell(
  fill: ratio-fill(r),
  if r == none { [—] } else { text(weight: "semibold")[×#fmt-ratio(r)#body] },
)
#let cmp-cell(a, b, ref: "ref") = {
  let r = ratio(a, b)
  table.cell(fill: ratio-fill(r), {
    set par(leading: 0.35em)
    text(weight: "bold", size: 8.3pt, fmt(a))
    linebreak()
    text(size: 6.8pt, fill: luma(70))[#ref #fmt(b)]
    if r != none { text(size: 7pt, weight: "semibold")[#h(3pt)×#fmt-ratio(r)] }
  })
}
// Reference value and multiplier NinjaRS ÷ reference, coloured by the multiplier.
#let vs-cell(a, b) = {
  let r = ratio(a, b)
  table.cell(fill: ratio-fill(r), {
    set par(leading: 0.35em)
    text(size: 8pt, fmt(b))
    linebreak()
    if r != none { text(size: 7.5pt, weight: "semibold")[×#fmt-ratio(r)] }
  })
}
#let short(name) = if name.starts-with("rn2-") { name.slice(4) } else { name }
#let header-style(body) = {
  set text(8pt, font: "DejaVu Sans")
  set par(justify: false)
  show table.cell.where(y: 0): set text(fill: white, weight: "bold")
  body
}
#let table-stroke = (x, y) => (
  top: if y <= 1 { none } else { 0.4pt + luma(225) },
  left: if x >= 2 and y > 0 { 0.4pt + luma(225) } else { none },
)
#let sci(x, digits: 1) = {
  if not is-num(x) { return [—] }
  if x == 0 { return [0] }
  let e = calc.floor(calc.log(calc.abs(x), base: 10))
  let m = x / calc.pow(10.0, e)
  if calc.round(m, digits: digits) >= 10 { e += 1; m = m / 10 }
  [#fixed(m, digits)e#e]
}
#let case-by-name = (:)
#for c in rn2.cases { case-by-name.insert(c.case, c) }
#let get(c, ..path) = {
  let node = c
  for key in path.pos() {
    if node == none { return none }
    node = node.at(key, default: none)
  }
  node
}
#let suite = rn2.cases.filter(c => not c.twin)
#let nr(c) = [#c.n / #c.rank]
#let count-if(xs, f) = xs.filter(f).len()
#let geo(xs) = {
  let v = xs.filter(x => is-num(x) and x > 0)
  if v.len() == 0 { none } else { calc.exp(v.map(x => calc.ln(x)).sum() / v.len()) }
}
#let by-n = rn2.by_n
#let coli-range = (
  calc.min(..by-n.map(r => r.coli.geomean)),
  calc.max(..by-n.map(r => r.coli.geomean)),
)
#let mu-free-unshifted = suite.filter(c => c.mu_powers == 0 and not c.shifted).len()
#let flags = rn2.accuracy.collier_flags
#let acc = rn2.accuracy.by_n

= Numerator rank $n + 2$ <rank-n2>

NinjaRS also reduces numerators of rank $r = n + 2$, one beyond Ninja's
higher-rank extension @ninja-paper (the algorithm, the new integrand terms and
their validation are in `docs/rank-n2.md`). Original Ninja cannot serve as the
reference here, so this section compares NinjaRS with the tensor-integral
library COLLIER #rn2.collier.version @collier, with its COLI and DD branches
@denner-dittmaier-5pt @denner-dittmaier-reduction @denner-dittmaier-d0. All
inputs are the rank $n + 2$ bench cases of `validation/rank-n2`: the 105 suite
cases ($n = 1, dots, 7$; real, complex, mixed and zero masses; numerators with
powers of the integrand $mu^2$ up to $mu^8$; 3 phase-space points each) and
the six synthetic cases `syn-n1-r3` to `syn-n6-r8`, each lane with a
high-precision reference value.

*A different method, so a timing reference, not like for like.* COLLIER does
not reduce the integrand. It computes the tensor integrals
$T^(N, mu_1 dots mu_P)$ of every rank $P <= r$ by Passarino–Veltman reduction
and, where the Gram or Cayley determinants are small, by expansions in them;
the integral of a numerator is then the contraction of these tensor components
with the numerator's tensor coefficients. Three consequences shape the
comparison:
- *Inputs.* COLLIER takes the tensor in a loop momentum with $p_0 = 0$, and no
  integrand $mu^2$. The harness removes $mu^2$ exactly by pinching
  ($mu^2 = q^2 + 2 q dot V_a + V_a^2 - m_a^2 - D_a$; up to five pieces with
  fewer propagators per lane) and shifts each piece, in exact rational
  arithmetic, before any timing; the pieces are then rounded to binary64
  once. On the #mu-free-unshifted of #suite.len() cases without $mu^2$ and
  with $V_0 = 0$, COLLIER receives exactly the tensor NinjaRS receives and no
  preprocessing is left out of its time.
- *Output.* One COLLIER evaluation gives the finite part for one choice of its
  regulator parameters $Delta_"UV"$, $Delta_"IR"^((1))$, $Delta_"IR"^((2))$.
  The single and double poles take two more evaluations with shifted
  parameters (only the UV pole is returned separately). NinjaRS returns all
  three Laurent coefficients, and their split into cut-constructible and
  rational parts, in one evaluation. The tables compare NinjaRS with
  COLLIER's single evaluation ($epsilon^0$, its usual use) and with the three
  evaluations of the full Laurent expansion.
- *Configuration.* COLLIER #rn2.collier.version (sha256
  #raw(rn2.collier.sha256.slice(0, 16))…) is built twice in binary64:
  `precise`, its gfortran Release flags without `-ffast-math`, as every other
  code in this report; and `stock`, the shipped flags with `-ffast-math`. It
  runs with its default target accuracy $10^(-8)$ (`SetReqAcc_cll`), without
  its global cache, and with the renormalization scale set outside the timed
  region (`SetMuUV2_cll` costs #fixed(rn2.collier_overheads_us.scale_switch, 1)
  µs, the $Delta$ changes between the three evaluations
  #fixed(rn2.collier_overheads_us.delta_switch, 1) µs, both from
  DD's determination of the machine precision at every parameter change). DD
  covers $n <= 4$ at rank $n + 2$ (pentagons up to rank 5, hexagons up to 6).
  With one integrand per phase-space point, the global cache changes COLLIER's
  time by ×#fixed(rn2.collier_cache_over_nocache.geomean, 2) on average
  (×#fixed(rn2.collier_cache_over_nocache.min, 2)–#fixed(rn2.collier_cache_over_nocache.max, 2));
  the stock build takes ×#fixed(rn2.collier_stock_over_precise.geomean, 2) of
  the precise build's time.

NinjaRS runs the fast engine (`ninjars case-bench`: dense `TensorNumerator`,
`MuTensorNumerator` with $mu^2$ blocks, native masters) at batch widths 1 and
32, as in @tab:identical, and the generic engine in binary64 (`case-bench
--engines generic`: `SymbolicaAmplitudeKernels`, native masters and
`case-check`'s numerically expanded dense-tensor numerator).

*Measurement.* Every process is single-threaded and pinned to one core
(CPU #rn2.cpu, idle when chosen; the host has no SMT siblings). The
#rn2.rounds rounds run NinjaRS and every COLLIER variant on every case in
turn; each quantity is the best of the rounds' medians over blocks of at
least #rn2.blocks.min_block_ms ms (#rn2.blocks.ninjars blocks for NinjaRS,
#rn2.blocks.collier for COLLIER, #rn2.blocks.generic for the generic engine,
which runs once). Other processes took
#calc.round(100 * rn2.foreign_cpu_s / rn2.measured_cpu_s, digits: 1) % of that
core's time during the measurements. The core clock of this shared host
varies between 1.8 and 3.1 GHz, so the retired instructions per point
(@tab:rank-n2-instructions) are given as a clock-independent check. NinjaRS
at #raw(rn2.ninjars_head) (oneloopmaster #raw(str(rn2.oneloopmaster))), #rn2.date.

#let rn2-table-header = ([Case], [$n$ / $r$], [NinjaRS fast [µs]],
  [COLI $epsilon^0$ [µs]], [COLI Laurent [µs]], [DD $epsilon^0$ [µs]], [generic f64 [µs] ÷ COLI])
#let time-row(c) = {
  let a = get(c, "ninjars", "w32")
  let coli = get(c, "collier", "precise-coli")
  let dd = get(c, "collier", "precise-dd")
  let g = get(c, "generic", "w32", "total")
  (
    table.cell(text(size: 7.4pt, raw(short(c.case)))),
    nr(c),
    table.cell({
      set par(leading: 0.35em)
      text(weight: "bold", size: 8.3pt, fmt(a.total))
      linebreak()
      text(size: 6.5pt, fill: luma(70))[#fmt(a.numerator) + #fmt(a.reduction) + #fmt(a.masters)]
    }),
    vs-cell(a.total, get(coli, "total_finite")),
    vs-cell(a.total, get(coli, "total_laurent")),
    if dd == none { table.cell(text(fill: luma(140))[—]) } else { vs-cell(a.total, get(dd, "total_finite")) },
    if g == none { table.cell(text(fill: luma(140))[—]) } else {
      table.cell(fill: ratio-fill(ratio(g, get(coli, "total_finite"))), {
        set par(leading: 0.35em)
        text(size: 8pt, fmt(g))
        linebreak()
        text(size: 7.5pt, weight: "semibold")[×#fmt-ratio(ratio(g, get(coli, "total_finite")))]
      })
    },
  )
}

#figure(
  kind: table,
  caption: [*Time per phase-space point at rank $n + 2$, NinjaRS versus
    COLLIER.* Binary64, one integrand per point, all codes pinned to the same
    core (CPU #rn2.cpu). NinjaRS fast engine at batch width 32: total in bold
    and, below, tensor-numerator expansions + pure reduction + masters.
    COLLIER (precise build): tensor integrals of every piece plus their
    contraction with the numerator coefficients, for the finite part
    ($epsilon^0$, one evaluation) and for the full Laurent expansion (three
    evaluations); COLI and, where it applies, DD. Each COLLIER cell gives its
    time and the multiplier NinjaRS ÷ COLLIER (green: NinjaRS faster). The
    last column is NinjaRS's generic engine (binary64, width 32) and its
    multiplier against COLI. `syn-*` are the synthetic cases, the others suite
    cases without their `rn2-` prefix; `-mu2` and `-muall` carry $mu^2$
    numerators (COLLIER: several pieces); `shifted` has $V_0 != 0$.],
  header-style(table(
    columns: (1fr, auto, auto, auto, auto, auto, auto),
    align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
    inset: (x: 4pt, y: 3.5pt),
    stroke: table-stroke,
    fill: (x, y) => if y == 0 { navy } else { none },
    table.header(..rn2-table-header),
    ..for name in rn2.display { time-row(case-by-name.at(name)) },
  )),
) <tab:rank-n2-time>

#figure(
  kind: table,
  caption: [*All #suite.len() rank $n + 2$ cases by $n$.* Geometric means
    over the cases of each $n$ of NinjaRS's fast-engine total (width 32) and
    COLLIER's COLI time ($epsilon^0$), and of their ratio, with its range;
    the same ratio at width 1, against COLI's full Laurent expansion, against
    DD and against the stock build. The last column is the generic engine's
    total over the fast engine's (width 32; synthetic and `rn2-int-complex`
    cases).],
  header-style(table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    align: center + horizon,
    inset: (x: 3.5pt, y: 3.5pt),
    stroke: table-stroke,
    fill: (x, y) => if y == 0 { navy } else { none },
    table.header([$n$], [cases], [NinjaRS [µs]], [COLI [µs]], [÷ COLI], [range],
      [÷ COLI (w1)], [÷ COLI Laurent], [÷ DD], [÷ COLI stock], [generic ÷ fast]),
    ..for r in by-n {
      (
        [#r.n], [#r.cases], fmt(r.ninjars_w32_total_geomean), fmt(r.collier_coli_geomean),
        ratio-cell(r.coli.geomean),
        text(size: 6.5pt)[×#fmt-ratio(r.coli.min)–#fmt-ratio(r.coli.max)],
        ratio-cell(get(r, "coli_w1", "geomean")),
        ratio-cell(get(r, "coli_laurent", "geomean")),
        if r.dd == none { [—] } else { ratio-cell(r.dd.geomean) },
        ratio-cell(get(r, "stock_coli", "geomean")),
        if is-num(r.generic_over_fast_w32) { [×#fmt-ratio(r.generic_over_fast_w32)] } else { [—] },
      )
    },
  )),
) <tab:rank-n2-by-n>

#let instr = rn2.at("instructions", default: none)
#let minstr(x) = if x == none { [—] } else { fmt(x.instructions / 1e6) }
#if instr != none [
  #figure(
    kind: table,
    caption: [*Retired user-space instructions per phase-space point*
      (millions; `perf stat -e instructions:u`, the difference of two
      fixed-work runs of each code, so start-up and recording are excluded).
      Unlike wall-clock time they do not depend on the core clock, which
      varies between 1.8 and 3.1 GHz on the shared host. NinjaRS fast engine
      total at widths 1 and 32; COLLIER finite part (precise COLI and DD,
      stock COLI). The ratio is NinjaRS (width 32) ÷ COLLIER COLI.],
    header-style(table(
      columns: (1fr, auto, auto, auto, auto, auto, auto, auto),
      align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
      inset: (x: 4pt, y: 3.5pt),
      stroke: table-stroke,
      fill: (x, y) => if y == 0 { navy } else { none },
      table.header([Case], [$n$ / $r$], [NinjaRS w1], [NinjaRS w32], [COLI], [DD], [COLI stock], [÷ COLI]),
      ..for r in instr {
        let c = case-by-name.at(r.case, default: none)
        let a = r.at("ninjars_fast_w32", default: none)
        let b = r.at("collier_precise_coli", default: none)
        (
          table.cell(text(size: 7.4pt, raw(short(r.case)))), if c == none { [—] } else { nr(c) },
          minstr(r.at("ninjars_fast_w1", default: none)), minstr(a), minstr(b),
          minstr(r.at("collier_precise_dd", default: none)), minstr(r.at("collier_stock_coli", default: none)),
          ratio-cell(if a != none and b != none { a.instructions / b.instructions } else { none }),
        )
      },
    )),
  ) <tab:rank-n2-instructions>
]
#let bn(n) = by-n.find(r => r.n == n)
#let speed(r) = if r == none { [—] } else { [×#fmt-ratio(r)] }
#let pct(x) = [#calc.round(x * 100) %]
#let low = by-n.filter(r => r.n <= 3)
#let high = by-n.filter(r => r.n >= 4)
#let instr-ratio(name) = {
  let r = if instr == none { none } else { instr.find(x => x.case == name) }
  if r == none { none } else {
    let a = r.at("ninjars_fast_w32", default: none)
    let b = r.at("collier_precise_coli", default: none)
    if a == none or b == none { none } else { a.instructions / b.instructions }
  }
}

*Time.* Up to $n = 3$ the two codes take similar times: per $n$,
NinjaRS's total is #low.map(r => speed(r.coli.geomean)).join([, ], last: [ and ])
of COLLIER's for $n = 1, 2, 3$ (geometric means, @tab:rank-n2-by-n), between
#speed(calc.min(..low.map(r => r.coli.min))) and
#speed(calc.max(..low.map(r => r.coli.max))) on single cases. Here
COLLIER's tensor integrals are cheap, while NinjaRS spends a median
#pct(calc.min(..low.map(r => r.numerator_share_w32.median)))–#pct(calc.max(..low.map(r => r.numerator_share_w32.median)))
of its time in the tensor-numerator expansions of the dense rank $n + 2$
tensor, and only
#pct(calc.min(..low.map(r => r.reduction_share_w32.median)))–#pct(calc.max(..low.map(r => r.reduction_share_w32.median)))
in the reduction itself. From $n = 4$ on, COLLIER's cost grows much faster
with the rank: NinjaRS takes #high.map(r => speed(r.coli.geomean)).join([, ], last: [ and ])
of COLLIER's time for $n = 4, dots, 7$, where COLLIER needs
#fmt(bn(7).collier_coli_geomean / 1000) ms per point at $n = 7$ against
NinjaRS's #fmt(bn(7).ninjars_w32_total_geomean / 1000) ms. A `perf` profile
of COLLIER on `syn-n5-r7` and `syn-n6-r8` attributes 69 % and 57 % of its
time to building the tensor components from the Lorentz-covariant
coefficients (`addtotensortn` in `BuildTensors.F90`); at rank $n + 2$ the
number of components and of their covariant structures grows steeply. The
retired instructions (@tab:rank-n2-instructions) show the same picture
independently of the core clock: from
#speed(instr-ratio("syn-n3-r5")) at $n = 3$ to #speed(instr-ratio("rn2-int-complex-n7-r9"))
at $n = 7$. COLLIER's full Laurent expansion costs it
×#fixed(geo(by-n.map(r => r.collier_laurent_over_finite)), 1) of its finite
part, so against it NinjaRS gains another factor of about three. DD, where it
applies ($n <= 4$), takes
#by-n.filter(r => r.dd != none).map(r => [×#fixed(r.coli.geomean / r.dd.geomean, 2)]).join([, ], last: [ and ])
of COLI's time for $n = 1, dots, 4$, and the stock build
×#fixed(rn2.collier_stock_over_precise.geomean, 2) of the precise build's.
COLLIER's global cache, which can only reuse integrals within one
phase-space point, changes its time by
#by-n.map(r => [×#fixed(r.collier_cache_over_nocache, 2)]).join([, ], last: [ and ])
for $n = 1, dots, 7$.
#let g-total = geo(by-n.map(r => r.generic_over_fast_w32))
#let g-red = geo(by-n.map(r => r.generic_reduction_over_fast_w32))
#if g-total != none and g-red != none [
  The generic engine in binary64 takes ×#fixed(g-total, 0) of the fast
  engine's time, and ×#fixed(g-red, 0) for the pure reduction alone. Its
  numerator here is `case-check`'s numerical expansion of the dense tensor,
  not the production path of the generic engine, which prepares the
  numerator symbolically.
]

#let twin-rows = rn2.twins
#figure(
  kind: table,
  caption: [*Cost of rank $n + 2$ against the rank $n + 1$ twins*
    (`rn2-int-complex`, the same kinematics and complex masses; refresh of
    the table in `docs/rank-n2.md` §7.2, now with the conditioned bases).
    NinjaRS fast engine: pure reduction (replayed numerator and masters) at
    width 32, tensor-numerator expansions, masters and total at width 32;
    COLLIER COLI ($epsilon^0$). Each cell gives rank $n + 2$ in bold, the
    rank $n + 1$ twin below it and their ratio; here the colour reads
    rank $n + 2$ ÷ rank $n + 1$.],
  header-style(table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: center + horizon,
    inset: (x: 4pt, y: 3.5pt),
    stroke: table-stroke,
    fill: (x, y) => if y == 0 { navy } else { none },
    table.header([$n$], [reduction [µs]], [numerator [µs]], [masters [µs]], [total [µs]], [COLLIER COLI [µs]]),
    ..for r in twin-rows {
      (
        [#r.n],
        cmp-cell(r.reduction_w32.at(0), r.reduction_w32.at(1), ref: "r = n+1"),
        cmp-cell(r.numerator_w32.at(0), r.numerator_w32.at(1), ref: "r = n+1"),
        cmp-cell(r.masters_w32.at(0), r.masters_w32.at(1), ref: "r = n+1"),
        cmp-cell(r.total_w32.at(0), r.total_w32.at(1), ref: "r = n+1"),
        if r.at("collier", default: none) == none { [—] } else {
          cmp-cell(r.collier.at(0), r.collier.at(1), ref: "r = n+1")
        },
      )
    },
  )),
) <tab:rank-n2-twins>
#let tw(key) = twin-rows.map(r => r.at(key).at(2))
#let tw2(key) = twin-rows.filter(r => r.n >= 2).map(r => r.at(key).at(2))
#let tw1 = twin-rows.find(r => r.n == 1)

*Cost of rank $n + 2$.* Against their rank $n + 1$ twins, the rank $n + 2$
lanes cost NinjaRS ×#fixed(calc.min(..tw2("reduction_w32")), 1)–#fixed(calc.max(..tw2("reduction_w32")), 1)
in the pure reduction for $n = 2, dots, 7$ (`docs/rank-n2.md` measured ×2.1–2.9 at
$n = 2, dots, 6$ before the basis conditioning, which added 3–6 % instructions;
the tadpole at $n = 1$: ×#fixed(tw1.reduction_w32.at(2), 2)) and
×#fixed(calc.min(..tw("total_w32")), 1)–#fixed(calc.max(..tw("total_w32")), 1)
in total, where the tensor-numerator expansions
(×#fixed(calc.min(..tw("numerator_w32")), 1)–#fixed(calc.max(..tw("numerator_w32")), 1))
dominate. COLLIER's time grows by
×#fixed(calc.min(..twin-rows.filter(r => r.at("collier", default: none) != none).map(r => r.collier.at(2))), 1)–#fixed(calc.max(..twin-rows.filter(r => r.at("collier", default: none) != none).map(r => r.collier.at(2))), 1)
over the same step.

#let acc-cell(s) = if s == none or s.lanes == 0 { [—] } else {
  set par(leading: 0.35em)
  [#text(size: 7.6pt, sci(s.median)) \ #text(size: 6.8pt, fill: luma(70), sci(s.max))]
}
#figure(
  kind: table,
  caption: [*Binary64 accuracy against the high-precision references*, on
    the tier A and B lanes with a condition of at most $10^6$, the lanes on
    which `case-check` judges binary64 (in brackets: the excluded lanes,
    tier C, whose published value is COLLIER's own quadruple-precision result
    or lacks an independent check, and conditions above $10^6$, on which no
    binary64 code is accurate). Scaled difference as `ninjars case-check` computes
    it: the largest $|x - x_"ref"|$ over the three Laurent coefficients,
    divided by the lane's largest reference coefficient. Each cell gives the
    median and, below it, the maximum. NinjaRS fast and generic engines at
    the case's own batch width (case-check); COLLIER precise build. The last
    column counts the lanes on which the fast engine is at least as accurate
    as COLI.],
  header-style(table(
    columns: (auto, auto, 1fr, 1fr, 1fr, 1fr, 1fr, auto),
    align: center + horizon,
    inset: (x: 4pt, y: 3.5pt),
    stroke: table-stroke,
    fill: (x, y) => if y == 0 { navy } else { none },
    table.header([$n$], [lanes], [NinjaRS fast], [NinjaRS generic f64], [COLI], [DD], [COLI stock],
      [fast ≤ COLI]),
    ..for r in acc {
      (
        [#r.n], [#r.lanes#if r.tier_c + r.ill_conditioned > 0 { text(size: 6.8pt, fill: luma(90))[ (+#(r.tier_c + r.ill_conditioned))] }],
        acc-cell(r.fast), acc-cell(r.generic), acc-cell(r.at("precise-coli")), acc-cell(r.at("precise-dd")),
        acc-cell(r.at("stock-coli")), [#r.fast_better_than_coli / #r.compared],
      )
    },
  )),
) <tab:rank-n2-accuracy>
#let acc-all = acc.map(r => r.compared).sum()
#let acc-better = acc.map(r => r.fast_better_than_coli).sum()

*Accuracy.* On the #acc.map(r => r.lanes).sum() suite lanes that
`case-check` judges in binary64 (@tab:rank-n2-accuracy), the fast engine's
error is at most COLI's on #acc-better of #acc-all lanes. Both lose accuracy
with $n$, for different reasons: NinjaRS through cancellations between the
cut contributions and in the subtractions, COLLIER through its Gram and
Cayley expansions and the size of its rank $n + 2$ tensor reduction. The
largest errors for $n = 5, 6, 7$ are
#acc.filter(r => r.n >= 5).map(r => sci(r.fast.max)).join([, ]) for the
fast engine and
#acc.filter(r => r.n >= 5).map(r => sci(r.at("precise-coli").max)).join([, ])
for COLI. On the
#acc.map(r => r.ill_conditioned).sum() excluded tier A and B lanes with a
condition above $10^6$ (all at $n >= #acc.find(r => r.ill_conditioned > 0).n$:
COLLIER's terms or the new rank $n + 2$ terms cancel) neither code is
accurate in binary64: up to
#sci(calc.max(..acc.map(r => r.at("fast_all_ab").at("max", default: 0)))) for
the fast engine and
#sci(calc.max(..acc.map(r => r.at("precise-coli_all_ab").at("max", default: 0)))) for COLI.

#let fresh = rn2.accuracy.at("final_review", default: none)
#let fresh-cell(s) = if s == none or s.lanes == 0 { [—] } else {
  set par(leading: 0.35em)
  [#text(size: 7.6pt, sci(s.median)) \ #text(size: 6.8pt, fill: luma(70))[#sci(s.max) #if s.at("above_1e-6") > 0 [(#s.at("above_1e-6") > 1e-6)]]]
}
#if fresh != none [
  #figure(
    kind: table,
    caption: [*Binary64 accuracy on the final review's fresh cases*
      (#fresh.cases cases, #fresh.lanes lanes, $n = 1, dots, 7$ at rank $n + 2$;
      light-like, equal, mixed, zero and complex masses, $mu^2$
      numerators; references to 60–100 digits). Median and, below, maximum
      scaled difference, with the number of lanes above $10^(-6)$; NinjaRS
      fast and generic engines (`case-check`), COLLIER precise build.],
    header-style(table(
      columns: (auto, auto, 1fr, 1fr, 1fr, 1fr),
      align: center + horizon,
      inset: (x: 4pt, y: 3.5pt),
      stroke: table-stroke,
      fill: (x, y) => if y == 0 { navy } else { none },
      table.header([$n$], [lanes], [NinjaRS fast], [NinjaRS generic f64], [COLI], [DD]),
      ..for r in fresh.by_n {
        ([#r.n], [#r.lanes], fresh-cell(r.fast), fresh-cell(r.generic), fresh-cell(r.coli), fresh-cell(r.dd))
      },
    )),
  ) <tab:rank-n2-fresh>
]
#let coli-bad = fresh.at("coli_above_1e-6")
*Fresh cases.* The final review generated #fresh.cases further rank $n + 2$
cases (#fresh.lanes lanes) with references to 60–100 digits, including
deliberately ill-conditioned light-like kinematics
(@tab:rank-n2-fresh). There NinjaRS's binary64 error exceeds $10^(-6)$ on
#fresh.at("ninjars_above_1e-6") lanes, all of whose cut contributions cancel by
more than #sci(fresh.at("above_1e-6_condition_min"), digits: 0) relative to
the result: an instability of integrand reduction in fixed precision that a
tensor-integral code does not share, and COLI is accurate to a median
#sci(fresh.at("above_1e-6_coli").median) on these lanes. The generic engine's
opt-in cancellation test (`--stability-test`) reports
#fresh.at("ninjars_above_1e-6_flagged") of them as `Unstable`; it leaves
#fresh.at("ninjars_above_1e-5_unflagged") lanes with an error above
$10^(-5)$ unflagged and flags #fresh.at("flagged_below_1e-6") lanes whose
error is below $10^(-6)$. Flagged points can be recomputed in DoubleFloat,
which recovers these lanes (final review). COLI in turn exceeds $10^(-6)$ on
#coli-bad.len() lanes, #coli-bad.filter(l => l.flags.at(1) < 0).len() of them
with accuracy flag $-1$.

#let flag-row(name, f) = (
  [#name], [#f.lanes], [#f.errflag_negative],
  [#f.accflag.at("0") / #f.accflag.at("-1") / #f.accflag.at("-2")],
  [#f.at("error_above_1e-8")], [#f.at("error_above_1e-8_with_accflag_0")],
  [#f.at("accflag_negative_error_below_1e-8")],
  [#f.estimate_covers_error], [#sci(f.worst_underestimate)],
)
#figure(
  kind: table,
  caption: [*COLLIER's own accuracy information against its actual error*
    (tier A and B lanes, precise build). Error flag (negative: an internal
    error) and accuracy flag (0: target $10^(-8)$ reached, $-1$: not reached,
    $-2$: critical $10^(-1)$ not reached), the most negative over the lane's
    calls; lanes whose actual scaled error exceeds $10^(-8)$, how many of those
    report accuracy flag 0, and how many lanes report a missed target while
    being accurate to $10^(-8)$. The error estimate is
    $sum_k |c_k| Delta T(P_k)$ over the numerator coefficients $c_k$ and
    COLLIER's per-rank absolute error estimates of the tensor components
    (`TNerr`), divided by the lane scale; the last columns count the lanes
    whose actual error it covers (an estimate below $10^(-15)$ counts as
    $10^(-15)$) and give the worst ratio of actual error to estimate.],
  header-style(table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
    align: center + horizon,
    inset: (x: 4pt, y: 3.5pt),
    stroke: table-stroke,
    fill: (x, y) => if y == 0 { navy } else { none },
    table.header([branch], [lanes], [errflag < 0], [accflag 0 / −1 / −2], [error > 1e-8],
      [… with accflag 0], [accflag < 0, error ≤ 1e-8], [estimate ≥ error], [worst error ÷ estimate]),
    ..flag-row([COLI], flags.at("precise-coli")),
    ..flag-row([DD], flags.at("precise-dd")),
  )),
) <tab:rank-n2-flags>
#let fc = flags.at("precise-coli")
#let fd = flags.at("precise-dd")

*COLLIER's own accuracy information* (@tab:rank-n2-flags) is a warning, not a
bound: COLI reports a missed target ($sigma_"acc" = -1$) on
#fc.accflag.at("-1") of #fc.lanes lanes, but on
#fc.at("error_above_1e-8_with_accflag_0") of the
#fc.at("error_above_1e-8") lanes whose actual error exceeds $10^(-8)$ it
reports the target as reached, and its error estimate falls short of the
actual error on #(fc.lanes - fc.estimate_covers_error) lanes, by up to
×#sci(fc.worst_underestimate). The quadruple-precision builds showed the
same in the reference harness (`validation/rank-n2/README.md`, finding 1).

#let cc = rn2.case_check
#let cc-runs = (
  ("fast/f64", [fast (f64)]),
  ("generic/f64", [generic f64]),
  ("generic/double", [generic DoubleFloat]),
  ("generic/float:128", [generic Float(128)]),
  ("generic/float:256", [generic Float(256)]),
).filter(((k, _)) => k in cc)
#let cc-cell(t) = if t == none { [—] } else {
  text(size: 7.5pt)[#t.pass / #t.fail / #t.at("report-only") / #t.at("known-failure")]
}
#let cc-ns = cc.at(cc-runs.at(0).at(0)).by_n.keys().sorted(key: k => int(k))
#figure(
  kind: table,
  caption: [*`ninjars case-check` on the #suite.len() rank $n + 2$ cases*
    (suite and synthetic, #cc.at(cc-runs.at(0).at(0)).by_n.values().map(t => t.lanes).sum() lanes), as
    pass / fail / report-only / known failure per lane, with the tolerance
    policy of `docs/rank-n2.md` §8.3: in binary64 a budget from original
    Ninja's error on the rank twins (30× for the fast engine, 100× for the
    generic engine) plus the references' uncertainty; at high precision three
    times the uncertainty. Report-only lanes are tier C, conditions above
    $10^6$ in binary64, DoubleFloat and Float(128), tier-B uncertainties
    taken from the generic engine's own twins at high precision, and lanes
    whose smallest new rank $n + 2$ term lies below the tolerance.],
  header-style(table(
    columns: (auto, auto) + cc-runs.map(_ => 1fr),
    align: center + horizon,
    inset: (x: 4pt, y: 3.5pt),
    stroke: table-stroke,
    fill: (x, y) => if y == 0 { navy } else { none },
    table.header([$n$], [lanes], ..cc-runs.map(((k, label)) => label)),
    ..for n in cc-ns {
      ([#n], [#cc.at(cc-runs.at(0).at(0)).by_n.at(n).lanes],
        ..cc-runs.map(((k, _)) => cc-cell(cc.at(k).by_n.at(n, default: none))))
    },
    table.cell(colspan: 2)[largest checked],
    ..cc-runs.map(((k, _)) => text(size: 7.5pt, sci(cc.at(k).largest_checked))),
    table.cell(colspan: 2)[largest tier A],
    ..cc-runs.map(((k, _)) => text(size: 7.5pt, sci(cc.at(k).largest_tier_a))),
  )),
) <tab:rank-n2-case-check>
#let agree = rn2.accuracy.at("collier_vs_1.2.9")

*Correctness of NinjaRS at rank $n + 2$* is checked lane by lane by `ninjars
case-check` (@tab:rank-n2-case-check): no lane fails in any of the five
runs, which reproduce `docs/rank-n2.md` §8.3. *The COLLIER driver* reproduces
the binary64 COLLIER 1.2.9 diagnostics stored with the references (exact
decimal inputs and a 90-digit contraction there, binary64 here): on the
#agree.at("precise-coli").at("above_1e-14") COLI lanes with an error above
$10^(-14)$ the ratio of the errors has median
#fixed(agree.at("precise-coli").median_ratio, 3), and the
#agree.at("precise-coli").differ_2x.len() lanes that differ by more than a
factor of two are $mu^2$ lanes, whose pieces this harness rounds to binary64
before the contraction. COLLIER 1.2.10 changes, against 1.2.9, complex-mass
$C_0$ root prescriptions and two error-estimate guards. Building and running
COLLIER showed one defect in the reference harness and one cost to know
about: its driver aggregated COLLIER's accuracy flag with `max` from 0 and so
never reported a negative flag (the references do not use it), and every
change of a regulator or scale parameter runs DD's machine-precision loop
(`DDsetparam`), also in COLI mode.
