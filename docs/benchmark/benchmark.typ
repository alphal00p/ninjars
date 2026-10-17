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

// ---------------------------------------------------------------- helpers --

#let navy = rgb("#1f3b63")
#let pale = rgb("#eef3f9")
#let is-num(x) = type(x) == int or type(x) == float

// Significant-figure formatting suited to timings.
#let fmt(x) = {
  if not is-num(x) { return [—] }
  let a = calc.abs(x)
  let d = if a >= 1000 { 0 } else if a >= 100 { 0 } else if a >= 10 { 1 } else if a >= 1 { 2 } else { 3 }
  let s = str(calc.round(x, digits: d))
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

= Summary of all test cases <summary>

#let cases = data.cases

// Fixed-decimal formatting (keeps trailing zeros: 0.30, 1.05).
#let fixed(x, d) = {
  let s = str(calc.round(x, digits: d))
  if d == 0 { return s }
  let parts = s.split(".")
  let frac = if parts.len() > 1 { parts.at(1) } else { "" }
  parts.at(0) + "." + frac + "0" * (d - frac.len())
}
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

#figure(
  kind: table,
  caption: [Generation time and per-phase-space-point evaluation time
    (binary64, one helicity configuration). Each cell shows the NinjaRS value
    in bold, the reference value below it and the multiplier
    NinjaRS ÷ reference. Generation: NinjaRS CLI preparation including the
    SymJIT O2 compilation of the numerator coefficients, versus MadLoop
    process output plus compilation (ML). Numerator versus MadLoop's loop
    coefficients and tensor expansions (ML); pure reduction versus original
    Ninja (N); master integrals versus OneLOop (O); total versus the complete
    MadLoop pipeline (ML).],
  {
    set text(8pt, font: "DejaVu Sans")
    set par(justify: false)
    show table.cell.where(y: 0): set text(fill: white, weight: "bold")
    table(
      columns: (1fr, auto, auto, auto, auto, auto, auto),
      align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
      inset: (x: 5pt, y: 4.5pt),
      stroke: (x, y) => (
        top: if y <= 1 { none } else { 0.4pt + luma(225) },
        left: if x >= 2 and y > 0 { 0.4pt + luma(225) } else { none },
      ),
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
          table.cell(text(size: 8.3pt, c.label)),
          [#c.n / #c.rank],
          cmp-cell(g.ninjars, g.madloop, ref: "ML"),
          cmp-cell(a.numerator, b.numerator, ref: "ML"),
          cmp-cell(a.reduction, b.reduction, ref: "N"),
          cmp-cell(a.masters, b.masters, ref: "O"),
          cmp-cell(a.total, b.total, ref: "ML"),
        )
      },
    )
  },
) <tab:summary>

#v(0.3em)
#align(center)[
  #set text(8pt)
  #box(width: 9cm, height: 0.35cm, fill: palette, radius: 2pt)
  #h(0.4em)
  #box(width: 9cm)[×1/4 #h(1fr) ×1/2 #h(1fr) ×1 #h(1fr) ×2 #h(1fr) ×4]
]

#include "sections.typ"
