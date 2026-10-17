// NinjaRS benchmark report. Numbers are read from data/results.json, which is
// produced by validation/benchmark/collect.py from the raw measurement files.

#let data = json("data/results.json")

#set document(title: "NinjaRS benchmark", author: "Valentin Hirschi")
#set page(
  paper: "a4",
  margin: (x: 1.6cm, y: 1.8cm),
  numbering: "1 / 1",
  header: context {
    if counter(page).get().first() > 1 [
      #set text(8pt, fill: luma(110))
      NinjaRS benchmark #h(1fr) #data.date
    ]
  },
)
#set text(font: "Libertinus Serif", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => {
  v(0.6em)
  block(text(14pt, fill: rgb("#1f3b63"), it))
  v(0.2em)
}
#show heading.where(level: 2): it => block(text(11.5pt, fill: rgb("#1f3b63"), it))
#show raw: set text(font: "DejaVu Sans Mono", size: 8.5pt)
#show link: set text(fill: rgb("#1f5fa8"))
#show figure.where(kind: table): set block(breakable: true)
#show figure.where(kind: table): set figure.caption(position: top)

// ---------------------------------------------------------------- helpers --

#let navy = rgb("#1f3b63")
#let pale = rgb("#eef3f9")
#let is-num(x) = type(x) == int or type(x) == float

// Fixed-decimal formatting (keeps trailing zeros: 0.30, 1.05).
#let fixed(x, d) = {
  let s = str(calc.round(x, digits: d))
  if d == 0 { return s }
  let parts = s.split(".")
  let frac = if parts.len() > 1 { parts.at(1) } else { "" }
  parts.at(0) + "." + frac + "0" * (d - frac.len())
}

// Significant-figure formatting suited to timings.
#let fmt(x) = {
  if not is-num(x) { return [—] }
  let a = calc.abs(x)
  let d = if a >= 100 { 0 } else if a >= 10 { 1 } else if a >= 1 { 2 } else { 3 }
  let s = fixed(x, d)
  if a >= 10000 {
    // thin-space thousands separator
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

// Diverging palette: green (NinjaRS faster) — white (parity) — red (slower).
#let palette = gradient.linear(
  rgb("#5aae61"), rgb("#d9f0d3"), rgb("#ffffff"), rgb("#fddbc7"), rgb("#d6604d"),
)
#let ratio-fill(r) = {
  if r == none { return none }
  // log2 scale: ×1/4 → 0 %, ×1 → 50 %, ×4 → 100 %
  let t = calc.clamp(0.5 + calc.log(r, base: 2) / 4, 0, 1)
  palette.sample(t * 100%)
}
#let ratio-cell(r) = table.cell(
  fill: ratio-fill(r),
  if r == none { [—] } else { text(weight: "semibold")[×#fmt(r)] },
)

// ------------------------------------------------------------------ title --

#align(center)[
  #v(0.4em)
  #text(22pt, weight: "bold", fill: navy)[NinjaRS benchmark]
  #v(-0.3em)
  #text(12pt, fill: luma(80))[Batched one-loop integrand reduction in Rust versus MadLoop (Ninja + OneLOop)]
  #v(0.2em)
  #text(9.5pt, fill: luma(110))[#data.date · #data.host]
]

#v(0.6em)

#block(fill: pale, inset: 10pt, radius: 4pt, width: 100%)[
  #set text(9.5pt)
  *Reading guide.* Every multiplier is _NinjaRS time ÷ reference time_ for
  the same work on the same inputs: generation is compared with MadLoop
  output plus compilation; the numerator with MadLoop's loop-coefficient
  computation (including Ninja's tensor-numerator expansions); the pure
  reduction with original Ninja; the master integrals with OneLOop; the total
  with the complete MadLoop pipeline. Green cells mean NinjaRS is faster, red
  slower; white is parity. NinjaRS evaluates batches of
  #data.batch_width phase-space points; times are per point.
]

#let range-of(values) = {
  let v = values.filter(is-num)
  if v.len() == 0 { return [—] }
  [×#fixed(calc.min(..v), 2)–#fixed(calc.max(..v), 2)]
}
#let gate-row(title, target, achieved) = (
  text(weight: "semibold", title), target, text(weight: "bold", fill: rgb("#1b7a3e"), achieved),
)
#block(stroke: 0.6pt + luma(200), inset: 9pt, radius: 4pt, width: 100%)[
  #set text(9pt)
  *Performance gates* (binary64, NinjaRS batched; ranges over the physical
  diagrams of @tab:summary, identical-input values from @tab:identical)
  #v(2pt)
  #table(
    columns: (auto, 1fr, auto),
    stroke: none,
    inset: (x: 4pt, y: 3pt),
    ..gate-row([(a) Pure reduction],
      [NinjaRS ≤ original Ninja on identical inputs],
      range-of(data.cases.map(c => ratio(c.ninjars_identical_us.reduction, c.ninja_identical_us.reduction)))),
    ..gate-row([(b) Master integrals],
      [oneloopmaster native ≤ 1.3 × OneLOop on the same calls],
      range-of(data.cases.map(c => ratio(c.ninjars_identical_us.masters, c.ninja_identical_us.masters)))),
    ..gate-row([(c) End to end],
      [NinjaRS (SymJIT O2 numerator + reduction + masters) ≤ MadLoop pipeline],
      range-of(data.cases.map(c => ratio(c.ninjars_us.total, c.madloop_us.total)))),
  )
]

= Summary of all test cases <summary>

#let cases = data.cases

#let fmt-ratio(r) = if r >= 10 { fixed(r, 0) } else { fixed(r, 2) }
// One comparison cell: NinjaRS value, reference value and the multiplier.
#let cmp-cell(a, b, unit: none, ref: "ref") = {
  let r = ratio(a, b)
  table.cell(fill: ratio-fill(r), {
    set par(leading: 0.35em)
    text(weight: "bold", size: 8.5pt, fmt(a))
    if unit != none { text(size: 7pt)[#h(1pt)#unit] }
    linebreak()
    text(size: 6.8pt, fill: luma(70))[#ref #fmt(b)]
    if r != none { text(size: 7pt, weight: "semibold")[#h(3pt)×#fmt-ratio(r)] }
  })
}

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
#let sci(x) = {
  if not is-num(x) { return [—] }
  if x == 0 { return [0] }
  let e = calc.floor(calc.log(calc.abs(x), base: 10))
  let m = x / calc.pow(10.0, e)
  [#fixed(m, 1)e#e]
}

#figure(
  kind: table,
  caption: [*Physical diagrams versus the MadLoop pipeline.* Generation
    time and per-phase-space-point evaluation time (binary64, one helicity
    configuration, a single loop diagram). Each cell shows the NinjaRS value
    in bold, the reference value below it and the multiplier
    NinjaRS ÷ reference. Generation: NinjaRS CLI graph preparation,
    coefficient planning and SymJIT O2 compilation versus MadLoop process
    output plus compilation (ML). Numerator: SymJIT O2 coefficient heads and
    tensor-numerator expansions versus MadLoop's loop-coefficient
    construction and Ninja's tensor-numerator expansions (ML). Reduction:
    pure reduction versus original Ninja inside MadLoop (N). Masters:
    oneloopmaster's native code versus OneLOop (O). Total: complete
    evaluation versus the complete MadLoop pipeline (ML). The rank column
    gives NinjaRS's rank and, in brackets, the rank MadLoop passes to Ninja.],
  header-style(table(
    columns: (1fr, auto, auto, auto, auto, auto, auto),
    align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
    inset: (x: 5pt, y: 4.5pt),
    stroke: table-stroke,
    fill: (x, y) => if y == 0 { navy } else { none },
    table.header(
      [Test case], [$n$ / $r$], [Generation [s]], [Numerator [µs]],
      [Reduction [µs]], [Masters [µs]], [Total [µs]],
    ),
    ..for c in cases {
      let g = c.generation_s
      let a = c.ninjars_us
      let b = c.madloop_us
      (
        table.cell(text(size: 8.3pt)[#c.process \ #text(fill: luma(90), c.diagram)]),
        [#c.n / #c.rank #text(fill: luma(110))[(#c.madloop_rank)]],
        cmp-cell(g.ninjars, g.madloop, ref: "ML"),
        cmp-cell(a.numerator, b.numerator, ref: "ML"),
        cmp-cell(a.reduction, b.reduction, ref: "N"),
        cmp-cell(a.masters, b.masters, ref: "O"),
        cmp-cell(a.total, b.total, ref: "ML"),
      )
    },
  )),
) <tab:summary>

#v(0.2em)
#align(center)[
  #set text(8pt)
  #box(width: 9cm, height: 0.35cm, fill: palette, radius: 2pt)
  #h(0.4em)
  #box(width: 9cm)[×1/4 #h(1fr) ×1/2 #h(1fr) ×1 #h(1fr) ×2 #h(1fr) ×4]
]

// Composition figure: per diagram, NinjaRS and MadLoop bars normalized to the
// MadLoop total and stacked by component. Categorical slots 1-3 of the
// validated default palette (blue, orange, aqua), 2pt surface gaps.
#let component-colors = (rgb("#2a78d6"), rgb("#eb6834"), rgb("#1baf7a"))
#let component-names = ("Numerator", "Reduction", "Masters")
#let stack-bar(parts, reference, width) = {
  let total = parts.sum()
  box(width: width, height: 8pt, {
    let x = 0pt
    for (i, value) in parts.enumerate() {
      let w = width * value / reference
      place(dx: x, rect(
        width: calc.max(w - 1pt, 0.5pt), height: 8pt, fill: component-colors.at(i),
        radius: if i == parts.len() - 1 { (right: 2pt) } else { 0pt },
      ))
      x += w
    }
  })
}
#figure(
  kind: image,
  caption: [Per-point evaluation time of each diagram, NinjaRS (upper bar,
    batched) and the MadLoop pipeline (lower bar), stacked by component and
    normalized to the MadLoop total. The label gives NinjaRS's total as a
    multiple of MadLoop's; @tab:summary lists the values.],
  {
    set text(8pt, font: "DejaVu Sans")
    let bar-width = 10cm
    grid(
      columns: (3.3cm, auto, 2.2cm),
      column-gutter: 6pt,
      row-gutter: range(2 * cases.len()).map(i => if calc.even(i) { 2pt } else { 9pt }),
      align: (right + horizon, left + horizon, left + horizon),
      ..for c in cases {
        let a = c.ninjars_us
        let b = c.madloop_us
        let mine = (a.numerator, a.reduction, a.masters)
        let theirs = (b.numerator, b.reduction, b.masters)
        let reference = theirs.sum()
        (
          grid.cell(rowspan: 2, [#c.process \ #text(fill: luma(90), c.diagram)]),
          stack-bar(mine, reference, bar-width),
          [NinjaRS #text(weight: "bold")[×#fixed(mine.sum() / reference, 2)]],
          stack-bar(theirs, reference, bar-width),
          text(fill: luma(110))[MadLoop],
        )
      },
    )
    v(4pt)
    for (i, name) in component-names.enumerate() {
      box(rect(width: 8pt, height: 8pt, fill: component-colors.at(i), radius: 1pt))
      h(3pt)
      name
      h(12pt)
    }
  },
) <fig:composition>

#let identical-rows(entries, physical: true) = for c in entries {
  let a = if physical { c.ninjars_identical_us } else { c.ninjars_us }
  let b = if physical { c.ninja_identical_us } else { c.ninja_us }
  if a == none or b == none { continue }
  (
    table.cell(text(size: 8pt, if physical { [#c.process, #c.diagram] } else { raw(c.id) })),
    [#c.n / #c.rank],
    cmp-cell(a.numerator, b.numerator, ref: "N"),
    cmp-cell(a.reduction, b.reduction, ref: "N"),
    cmp-cell(a.masters, b.masters, ref: "O"),
    cmp-cell(a.total, b.total, ref: "N+O"),
    table.cell(text(size: 7.5pt, sci(c.max_scaled_difference_vs_ninja))),
  )
}

#figure(
  kind: table,
  caption: [*Identical inputs versus original Ninja and OneLOop.* Both
    implementations receive the same tensor-numerator coefficients,
    propagators and invariants (binary64), and evaluate the same rank.
    Tensor numerator: Ninja-layout tensor expansions in NinjaRS's batched
    `TensorNumerator` versus Ninja's `TensorNumerator` (N); pure reduction
    with replayed numerator and master values (N); master integrals on the
    identical call stream, oneloopmaster native versus OneLOop (O); total
    with live numerator and masters. Original Ninja is built with
    `g++ -O3 -fcx-fortran-rules` (MadLoop's flag; 20 % faster than plain
    `-O3`). The last column is the maximal difference of the three Laurent
    coefficients relative to the largest, NinjaRS versus Ninja.],
  header-style(table(
    columns: (1fr, auto, auto, auto, auto, auto, auto),
    align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
    inset: (x: 4pt, y: 3.5pt),
    stroke: table-stroke,
    fill: (x, y) => if y == 0 { navy } else { none },
    table.header(
      [Case], [$n$ / $r$], [Tensor num. [µs]], [Reduction [µs]],
      [Masters [µs]], [Total [µs]], [Δ vs Ninja],
    ),
    ..identical-rows(cases, physical: true),
    table.cell(colspan: 7, fill: pale, text(size: 7.5pt, weight: "semibold")[Synthetic topologies (random kinematics, real and complex masses, dense random tensors)]),
    ..identical-rows(data.synthetic, physical: false),
  )),
) <tab:identical>

#include "sections.typ"
