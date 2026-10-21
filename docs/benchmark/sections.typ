// Prose sections of the NinjaRS benchmark report.

= The benchmarked diagrams <diagrams>

#let drawing(path, title, detail) = box(width: 100%, {
  set align(center)
  box(height: 3.4cm, align(horizon, image(path, width: 88%, height: 3.4cm, fit: "contain")))
  v(-2pt)
  text(8.5pt, weight: "semibold", title)
  linebreak()
  text(7.5pt, fill: luma(90), detail)
})

The physical diagrams are drawn from the same FeynKit DOT files that NinjaRS
prepares (`ninjars draw`, laid out and styled by GammaLoop's Linnest
package): gluons as coils, photons as waves, top quarks as solid lines with
fermion-flow arrows and Higgs bosons dashed. Each diagram is matched one to
one with the MadLoop loop diagram given by its loop-filter identifier.

#figure(
  kind: image,
  caption: [The seven physical one-loop diagrams with a top-quark loop.
    Below each drawing: number of propagators $n$, numerator rank $r$ in
    NinjaRS (rank passed to Ninja by MadLoop in brackets) and the MadLoop
    loop-diagram identifier.],
  grid(
    columns: 4,
    gutter: 10pt,
    drawing("linnet/hgg/top-triangle-0.svg", [gg → H], [$n = 3$, $r = 2$ (3), ML \#3]),
    drawing("linnet/four-photon/top-box-0.svg", [γγ → γγ], [$n = 4$, $r = 4$ (4), ML \#182]),
    drawing("linnet/gg-hh/top-triangle-0.svg", [gg → HH], [$n = 3$, $r = 2$ (3), ML \#11]),
    drawing("linnet/gg-hh/top-box-2.svg", [gg → HH], [$n = 4$, $r = 4$ (4), ML \#10]),
    drawing("linnet/gg-hhh/top-triangle-0.svg", [gg → HHH], [$n = 3$, $r = 2$ (3), ML \#66]),
    drawing("linnet/gg-hhh/top-box-8.svg", [gg → HHH], [$n = 4$, $r = 4$ (4), ML \#58]),
    drawing("linnet/gg-hhh/top-pentagon-26.svg", [gg → HHH], [$n = 5$, $r = 4$ (5), ML \#55]),
  ),
) <fig:physical-diagrams>

#figure(
  kind: image,
  caption: [Scalar topologies of the synthetic cases ($n = 2, dots, 6$).
    Each synthetic case attaches a dense random tensor numerator of rank
    $n - 1$, $n$ or $n + 1$, random real external kinematics with time-like
    and space-like invariants, and real or complex internal masses to one of
    these topologies.],
  grid(
    columns: 5,
    gutter: 8pt,
    drawing("linnet/synthetic/bubble.svg", [$n = 2$], [ranks 1–3]),
    drawing("linnet/synthetic/triangle.svg", [$n = 3$], [ranks 2–4]),
    drawing("linnet/synthetic/box.svg", [$n = 4$], [ranks 3–5]),
    drawing("linnet/synthetic/pentagon-rank4.svg", [$n = 5$], [ranks 4–6]),
    drawing("linnet/synthetic/hexagon.svg", [$n = 6$], [ranks 5–7]),
  ),
) <fig:synthetic-diagrams>

= Correctness <correctness>

Speed is only meaningful at equal accuracy. The binary64 results of the
benchmark were checked as follows.

- *Reduction.* With identical master values, `ninjars::fast` agrees with
  original Ninja to $7 times 10^(-13)$ (relative to the largest modulus among
  Ninja's Laurent coefficients and its cut-constructible and rational parts)
  on every phase-space point of the physical cases and to
  $4 times 10^(-12)$ on 30 dedicated edge cases (light-like and zero
  invariants, mixed and complex masses, infrared-divergent triangles and
  boxes, all tadpole reference bases, ranks $n - 5, dots, n + 1$ and
  $n = 7$), including the split into cut-constructible and rational parts.
  Where the two differ more (the synthetic pentagons and hexagons
  `syn-n5-r6`, `syn-n6-r5` and `syn-n6-r7` at $2 times 10^(-9)$ to
  $1.1 times 10^(-7)$, and the ill-conditioned point of
  `syn-gghhh-ggHHH-n5-r5` discussed below), both are compared with the
  generic NinjaRS engine at 256 bits: over these and the edge-case points
  (42 in total) the error ratio NinjaRS ÷ Ninja has median 1.1, i.e. the
  deviation is inherited binary64 conditioning, not a porting error. Results are bitwise identical for every
  batch width and lane order, and the warmed path performs no heap
  allocation.
- *End to end.* On identical inputs every physical diagram matches original
  Ninja to $10^(-12)$ or better (last column of @tab:identical), and the
  complete NinjaRS pipeline, with its own SymJIT numerator, matches Ninja fed
  with the exported tensors to $2 times 10^(-12)$. Original Ninja gives the
  same result on MadLoop's own inputs as on the NinjaRS-exported tensors up
  to one constant factor at all three points ($-2$ for the QCD diagrams,
  $-1/3$ for $gamma gamma -> gamma gamma$, spread
  $lt.eq 1.2 times 10^(-14)$). Together with a diagram check and identical
  propagator invariants and masses, this fixes the diagram and its helicity
  configuration; the propagator order differs (MadLoop's is cyclic).
- *Masters.* On all 1,998 master calls that original Ninja makes in the
  benchmark (34 call streams: the physical diagrams as exported by NinjaRS
  and by MadLoop, and the synthetic topologies), oneloopmaster's native code
  agrees with OneLOop 3.6 to $8 times 10^(-14)$ on the physical diagrams
  and to $2.7 times 10^(-12)$ on the synthetic ones (a $D_0$ of
  `syn-n6-r7`), except for one rank-3 tensor bubble
  ($2.5 times 10^(-9)$). The native bubble tensors were
  validated against 595 direct Feynman-parameter integrations, including
  thresholds and massless limits.

*Ill-conditioned lanes.* Two synthetic cases differ from Ninja by more than
$10^(-6)$ (marked † in @tab:identical). In each, one lane's finite part is a
tiny remainder of cancelling master-integral contributions. The exact values
were computed with the generic NinjaRS engine at 256–512 bits, where the
evaluations agree to 40 digits (`syn-n6-r6`, for its snapped binary64
inputs) and to binary64 rounding (`syn-gghhh-ggHHH-n5-r5`); the per-point
data are in `validation/benchmark/ill-conditioned/`. Against them, neither
binary64 code is accurate on these lanes:

- `syn-gghhh-ggHHH-n5-r5`, lane 1, borders a box with normalized Gram
  determinant $8 times 10^(-8)$. Ninja is off by $3.0 times 10^(-4)$ and
  NinjaRS by $3.1 times 10^(-4)$. With exact masters rounded to binary64,
  NinjaRS is also off by $3.1 times 10^(-4)$, so the loss is in the binary64
  reduction itself, in both codes. oneloopmaster's binary64 $D_0$ at that
  box is off by $2.6 times 10^(-14)$ (OneLOop $7.9 times 10^(-15)$). From
  9c5490b (exact Källén edge roots) until 8ec256fc it was off by
  $3.6 times 10^(-11)$, which put this lane at $4.6 times 10^(-4)$;
  8ec256fc's contour box for vanishing Gram determinants fixes it.
- `syn-n6-r6`, lane 0, cancels by a factor of about $10^(12)$. Ninja is off
  by 2.3 % and NinjaRS by 1.3 % (1.2 % with Ninja's own master values).
  NinjaRS was off by 24 % before oneloopmaster conditioned the dilogarithm
  differences of nearly coincident triangle roots: the triangle
  $C_0(-237088.3, -236743.3, 0; 29929 - 259.5 i, 0, 6464.16 - 168.84 i)$
  was off by $2.4 times 10^(-13)$ against OneLOop's $3.3 times 10^(-15)$
  and is now off by $2.2 times 10^(-15)$, with the divided-difference
  expansion OneLOop uses for such roots. The lane's exact value is fixed by
  its binary64 inputs only to about $10^(-4)$: taking the invariants from the
  snapped S-matrix entries or from the momenta changes it by
  $9.9 times 10^(-5)$, far less than the binary64 errors.

Neither code's stability checks flag these lanes. All other lanes of these
cases agree with the exact values to $10^(-9)$ or better.

*Defects found on the way.* The comparison exposed and fixed the following
defects.

- *Symbolica:*
  - in binary64, the dilogarithm went through an MPFR callback (12 µs to
    2 ms per call, making native $C_0$/$D_0$ 25–430× slower than OneLOop);
    a binary64 series kernel now serves it (PR \#89), and oneloopmaster's
    binary64 code calls it directly;
  - in arbitrary precision, `Float` products were rounded twice
    ($a b != b a$) and $a - b$ was formed through a working-precision $-1$,
    so that exact sign tests in $C_0$/$D_0$ flipped at 256 bits; they are
    now rounded once (PR \#97, carried by the pinned 9cdc2949; binary64 and
    double-double are bitwise unchanged).
- *oneloopmaster* (fixed in the exact expressions, so every route — native,
  hand-written, SymJIT, interpreter, arbitrary precision and Dual numbers —
  carries the fix, except where noted; NinjaRS pins oneloopmaster 8ec256fc.
  The tenth to seventeenth items and the full extent of the $B_(1111)$
  error were found after the revision of this report that used 264a6ce, the
  last three items and the present accuracy of the seventh after the
  previous revision, which used 3023fa6):
  - $B_0$/$B_0'$ returning NaN with a massless line;
  - inexact unit roots in massless bubble tensors (fixed in the native port
    of OneLOop's bubble routines), and wrong lips at $p^2 = 0$ with masses of
    opposite sign;
  - a sign error in OneLOop's own $B_(1111)$: its mass-swap reconstruction
    is negated in all three swap branches, not only at $p^2 = 0$ (the
    native rank-four bubble family, added for numerator rank $n + 2$, uses
    the corrected one);
  - wrong lips in two- and three-mass and 't Hooft–Veltman triangles;
  - errors in infrared boxes and in four-mass boxes with
    $s_(12) s_(23) < 0$;
  - in real-mass contour boxes, analytically real logarithms whose lip was
    set by rounding (off by multiples of $2 pi i$ on about a quarter of such
    points). OneLOop 3.6 itself departs from its own vanishing-width limit at
    some of these points;
  - a box near a vanishing Gram determinant (accuracy $7 times 10^(-10)$,
    then $4 times 10^(-12)$, and $3.6 times 10^(-11)$ in binary64 after the
    exact Källén edge roots below; now $2.6 times 10^(-14)$: the contour's
    line edges take a pole's offset from its root from Vieta's product, and
    where one contour is ill-conditioned the other is evaluated too,
    8ec256fc);
  - exact-zero branch tests decided by rounding noise in arbitrary precision
    ($C_0$ wrong at 256, 512 and 1024 bits for some arguments);
  - an expression-construction blow-up introduced and removed during these
    fixes (shared definitions built in 80 s with 16 GB of memory; now 59 MB,
    built in 0.35 s);
  - dilogarithm differences of nearly coincident triangle roots cancelling
    in binary64 (the $C_0$ of `syn-n6-r6` above), now expanded as OneLOop
    does; and box coefficients formed by cancellation, which cost bits in
    arbitrary precision (a $D_0$ with an invariant of $2^(-114)$ kept 7 to
    55 correct bits at 132 to 184 bits);
  - real-mass three-mass boxes in OneLOop's `boxf33` chart, wrong in some
    propagator labellings (in 39 of 154 boxes of a sweep, OneLOop 3.6
    included); they now take the contour box in every labelling;
  - contour boxes losing 8 to 10 digits in binary64 next to a
    pseudo-threshold beside a massless line, and up to $5.7 times 10^(-4)$
    from far real roots of an edge quadratic; the edge roots now come from
    exact Källén functions and far roots from a series in $1 slash z$;
  - real four-mass boxes in OneLOop's `boxf4` chart, wrong in some
    labellings (OneLOop 3.6 in 118 of 6,040 relabellings of a sweep); they
    now use eq. (3.61) of Denner and Dittmaier in a labelling that meets its
    conditions, with exact branch decisions, and agree in every labelling of
    the sweeps;
  - one- and two-mass boxes next to collinear limits (up to
    $2.9 times 10^(-7)$ off in binary64, as in OneLOop's double precision,
    and $O(1)$ at invariants of $10^(-14)$); now within $10^(-11)$ on a
    sweep of 400 such boxes;
  - massless boxes next to collinear limits, and massless, one- and two-mass
    boxes whose discriminant also nearly vanishes there (up to
    $2 times 10^(-2)$ off in binary64);
  - in the native double-double code only, products whose low word depended
    on the operand order, which let exact branch tests read a $10^(-35)$
    residue as a sign: an equal-mass four-mass $D_0$ lost its real part in
    one of eight labellings (in NinjaRS a 12-point double-double amplitude
    was off by $1.6 times 10^(-3)$). The native products are now symmetric;
    the interpreter's double-double route (tensor families, Dual numbers)
    keeps Symbolica's product;
  - in the interpreter only, the cancellation test of the conditioned
    dilogarithm differences above evaluating both dilogarithms two or three
    times, slowing the arbitrary-precision runs of @arbprec by 1.4–1.5×;
    they are now evaluated once (09b2c74);
  - one-mass boxes with a complex mass, at any width, on a wrong sheet,
    OneLOop's `boxf1` included (131 of 3,000 random such boxes, by 0.3 to 7
    of the value), and two-mass boxes with complex masses in the same way
    (0.4 % to 2.8 % of random such boxes); they now take Denner and
    Dittmaier's sheets;
  - two-mass boxes with both massive lines at or next to their shells, whose
    dilogarithm arguments' complements, formed from the rounded arguments,
    lost up to 0.16 of the value in binary64; they now come from the
    channels' complements;
  - soft boxes, wrong or non-finite in many labellings of their propagators
    (on oneloopmaster's adversarial sweeps, 59 % of the binary64
    evaluations of 14,400 soft-box relabellings at 09b2c74); they are now
    relabelled at the numerical $D_0$ entry points and right on every
    route (8ec256fc).
- *NinjaRS:*
  - rounded light-like invariants entering the masters without OneLOop's
    on-shell threshold (a $10^(-7)$ error in $gamma gamma -> gamma gamma$
    below threshold);
  - in arbitrary precision, invariants differenced from rounded momenta (the
    four-mass $D_0$ was wrong by orders of magnitude at 128 bits while
    reporting 8 correct bits); they are now derived exactly from the prepared
    momenta;
  - a degenerate rational bubble reference vector (an $O(10^13)$ blow-up at
    exactly coplanar rational kinematics);
  - bubble reference vectors tied in score, chosen by rounding: at some
    precisions this moved $plus.minus 1375$ between the cut-constructible and
    rational parts while their sum stayed exact;
  - inaccurate Passarino–Veltman bubble tensors in an intermediate provider.
- *OneLOop reference copies:* two wrong $D_0$ values, in versions 3.4 and a
  2024 snapshot. OneLOop 3.6 and COLLIER agree with each other there.

Apart from the on-shell threshold, none of these defects affected the
accuracy of the physical benchmark diagrams in binary64; the dilogarithm
affected only their speed. Since the previous revision of this report
(oneloopmaster 3023fa6), the binary64 results of the physical diagrams have
moved by at most $4 times 10^(-15)$. oneloopmaster's own adversarial $D_0$
sweeps (three sweeps, 99,119 referenced rows each) still find a tail that
its `STATUS.md` lists as open work. Over them the four binary64 routes return
2,798 finite but wrong values, against 18,377 at 09b2c74; but on the two
sweeps generated after the fixes, 28 binary64 boxes that 09b2c74 got right
or returned as NaN are now finite and wrong by more than $10^(-6)$ (by up to
0.36), and so are 195 `Float` rows, mostly at 1 to 4 tracked bits. None of
the benchmark's master calls is among them (*Masters*, above).

= Arbitrary precision end to end <arbprec>

#let arb = json("data/arbitrary_precision.json")
#let one-decimal(x) = {
  let s = str(calc.round(x, digits: 1))
  if s.contains(".") { s } else { s + ".0" }
}

The same CLI evaluates a physical diagram at any working precision, from the
DOT graph to the integrated result, without a separate quadruple-precision
build. For the $g g -> H H H$ top pentagon (graph 26) at its three
benchmark points,

#align(center, raw(lang: "sh", "ninjars reduce examples/graphs/physical/gg-hhh/top-pentagon-26-pool.json \\\n  --precision float --bits 700"))

prepares the numerator exactly (FeynKit, exact radical momenta and
polarization vectors), runs the generic reduction engine on Symbolica's
arbitrary-precision `Float` (cut solutions, Laurent divisions and
subtractions at 700 bits) and evaluates oneloopmaster's exact master
expressions at the same precision. All three points take 7.9 seconds in
total at 700 bits and 9.1 seconds at 1000 bits (CPU 60), including the
one-time preparation, and 7.8–8.2 and 9.2–9.4 seconds in three further
rounds on other cores. Run alternately on the same cores with the CLI of the
previous revision (oneloopmaster 09b2c74, Symbolica e679e8c1), the current
one executes 0.7 % (700 bits) and 0.9 % (1000 bits) more instructions. The 700-bit results
agree with a 1000-bit run to
#calc.floor(calc.min(..arb.points.map(p => p.digits_700_vs_1000))) digits;
the poles and the rational part vanish to working precision (below
$10^(-211)$ at 700 bits and $10^(-301)$ at 1000 bits; the double pole is
exactly zero), as they must for this finite pentagon. The table compares the finite part across precisions.
Accuracy is always measured this way, by comparison with a higher
precision: the significant precision that Symbolica's `Float` tracks counts
cancellations but not accumulated rounding, and overstates the accuracy by
a few bits (4–7 bits in the 700-bit runs above, up to 13 bits in precision
scans of the $gamma gamma -> gamma gamma$ boxes).

#figure(
  kind: table,
  caption: [Correct digits of the finite part of graph 26, measured against
    the 1000-bit evaluation.],
  {
    set text(8.5pt, font: "DejaVu Sans")
    show table.cell.where(y: 0): set text(fill: white, weight: "bold")
    table(
      columns: (auto, auto, auto, auto),
      inset: (x: 7pt, y: 4pt),
      align: center + horizon,
      fill: (x, y) => if y == 0 { rgb("#1f3b63") } else if calc.even(y) { luma(248) } else { none },
      stroke: none,
      table.header([$sqrt(s)$], [binary64 (fast engine)], [double-double], [700 bits]),
      ..for (point, energy) in arb.points.zip(("450 GeV", "525 GeV", "600 GeV")) {
        (
          energy,
          [#one-decimal(point.digits_binary64_fast)],
          [#one-decimal(point.digits_double_float)],
          [#one-decimal(point.digits_700_vs_1000)],
        )
      },
    )
  },
) <tab:arbprec>

Finite part of graph 26 at $sqrt(s) = 450$ GeV to 200 significant digits
(1000-bit evaluation, normalization of the NinjaRS fixture):

#block(inset: (x: 8pt, y: 6pt), fill: rgb("#eef3f9"), radius: 3pt, width: 100%, {
  set text(7.6pt, font: "DejaVu Sans Mono")
  set par(justify: false, leading: 0.5em)
  // Mantissa in groups of ten digits, five groups per line, then the exponent.
  let show-number(name, s) = {
    let (mantissa, exponent) = s.split("e")
    let (lead, digits) = mantissa.split(".")
    let groups = range(0, digits.len(), step: 10).map(i => digits.slice(i, calc.min(i + 10, digits.len())))
    let lines = range(0, groups.len(), step: 5).map(i => groups.slice(i, calc.min(i + 5, groups.len())).join(" "))
    grid(
      columns: (auto, auto),
      column-gutter: 4pt,
      row-gutter: 3pt,
      [#name = #lead.], lines.first(),
      ..lines.slice(1).map(l => ([], l)).flatten(),
      [], [× 10#super(exponent)],
    )
  }
  show-number("Re", arb.points.at(0).re)
  v(4pt)
  show-number("Im", arb.points.at(0).im)
})

#include "rank_n2.typ"

= How the numbers are measured <methodology>

*Test cases.* Seven physical one-loop diagrams with a top-quark loop,
generated independently by FeynKit (NinjaRS) and MadGraph5_aMC\@NLO 3.7.0
(MadLoop), and matched graph by graph: the internal masses agree exactly,
the propagator invariants to binary64 rounding, and the integrated results
up to a constant normalization over all phase-space points. Each diagram is evaluated at
three phase-space points (exact rational/radical momenta rounded once to
binary64), one fixed helicity configuration, zero widths and
$mu^2 = 100^2$ GeV#super[2] ($125^2$ GeV#super[2] for $g g -> H$). Twenty synthetic topologies ($n = 2, dots, 6$
propagators, ranks $n - 1$, $n$ and $n + 1$, random real kinematics with
time-like and space-like invariants, real and complex masses, dense random
tensor numerators) complete the suite.

*NinjaRS.* `ninjars pipeline-bench` builds the physical numerator from the
DOT graph and the UFO-style model (Feynman rules, wavefunctions, colour
projection, Dirac and Lorentz algebra, exact polynomial in the loop
momentum), plans and compiles its coefficient heads with SymJIT O2 and then
times, per batch of phase-space points: the numerator preparation
(wavefunctions, couplings, coefficient heads, Ninja tensor layout and
propagator data); the pure reduction (`ninjars::fast` with the numerator
outputs and master values of the same evaluation replayed at O(1) cost); the
tensor-numerator expansions (live tensor numerator minus pure reduction); the
master integrals (the recorded calls replayed through oneloopmaster's native
code); and the uninstrumented total. At the tabulated batch width the
components add up to the total within 1 %. `ninjars case-bench` performs the same breakdown on the shared
bench-case files, which contain exactly the propagators and tensor
coefficients given to original Ninja.

*MadLoop pipeline.* Optimized standalone MadLoop output (`loop_optimized_output`),
restricted to the same single loop diagram, Ninja as the only reduction
library, no stability rotations, linked against original Ninja 1.2.1
(`g++ -O3 -fcx-fortran-rules`, integral cache disabled) and OneLOop 3.6
(`gfortran -O3`). An instrumented build splits each evaluation into
MadLoop's loop-coefficient construction, its Ninja interface (tensor copy,
momentum inversion, S-matrix), Ninja's tensor-numerator callbacks, the
OneLOop master integrals and the rest of Ninja's entry point (the reduction,
including Ninja's per-call setup). The numerator column sums the first three.
Replaying MadLoop's exact Ninja inputs in the C++ driver below reproduces
the master and tensor-numerator times within 5 %; the instrumented
reduction is 14–17 % above Ninja's pure reduction measured there (replayed
numerator and masters, as for NinjaRS), which is the reference of
@tab:identical and of gate (a). The table uses the faster MadLoop variant for every quantity:
`GLOBAL_FLAG=-O3` for evaluation and the default flags for compilation
time.

*Identical-input comparison.* The C++ driver feeds original Ninja the same
bench cases (`Amplitude<ComplexMasses>`, `TensorNumerator`, explicit
S-matrix) and replays recorded numerator outputs and OneLOop results to
isolate each component, exactly like the NinjaRS harness. Every replay
reproduces the direct evaluation bit for bit.

*Environment.* A shared AMD EPYC 9754 host; every timed process runs
single-threaded and pinned to one core. The NinjaRS timings are from
2026-10-21 (NinjaRS aa6a630, oneloopmaster 8ec256fc, Symbolica 9cdc2949) on
cores of the first socket, which has simultaneous multithreading: a run was
kept only if the core's other hardware thread stayed at most 5–10 % busy
(after all retries, two rounds kept their least disturbed attempt, at 19 %
and 5.3 %). Every value is the most representative of three rounds
interleaved with the previous revision on the same core
(`validation/benchmark/rerun-8ec256fc/`). The reference timings are from
2026-10-18 on CPU 250 (second socket, no SMT); an earlier cross-check on
CPU 29 agreed within about 1 %, except OneLOop on the $n = 5$ cases, 5–8 %
faster there (`validation/benchmark/cpu-crosscheck.md`). Re-timed on the
cores used now, with the same executables, original Ninja's replay lies
within −6.3 to +3.5 % of them (pure reduction −0.3 to +3.0 %, OneLOop −6.3
to +3.5 %), and the MadLoop pipeline, rebuilt for three diagrams, within −3
to −1 %. This moves no gate: against these re-timed references the gates are
(a) ×0.48–0.56, (b) ×0.27–0.61 and (c) ×0.19–0.39 (three diagrams). Against
the previous revision on the same core, the reduction is unchanged
(×0.99–1.02; instructions within 0.25 %), and oneloopmaster 8ec256fc's
master integrals take 6–7 % longer on the gg → HH(H) box and pentagon
diagrams (+1.4–1.6 % instructions). The arbitrary-precision runs of
@arbprec used CPU 60, whose other hardware thread was 2.4 % busy. Rust 1.98
(`--profile perf`: opt-level 3, one codegen unit for the hot crates), GCC and
gfortran 15.3, no `-march`/`-ffast-math` on either side. Timings are medians
of nine blocks of at least 20 ms; the three rounds agree within 6 %
(pipeline totals within 2 %), components below 1.5 µs within 22 %.

= What is new in NinjaRS <innovations>

NinjaRS reimplements the integrand-reduction method of Ninja
@ninja-laurent @ninja-paper in Rust and embeds it in an end-to-end pipeline
built on shared physics and algebra libraries. The original stack used for
comparison — MadLoop's numerator @madloop, Ninja's reduction and OneLOop's
master integrals @oneloop — is a set of independently written Fortran and
C++ codes connected through a tensor-numerator interface
@ninja-tensor. NinjaRS replaces each component while keeping Ninja's
semantics and results.

== One exact source, from Feynman graph to integrand

Physical numerators are prepared by FeynKit from Feynman-graph DOT files and a
UFO-style model: Feynman rules, external wavefunctions, colour projection and
Dirac/Lorentz algebra are applied symbolically with Spenso/Idenso and
Symbolica. The result is an *exact* polynomial in the loop momentum (and
$mu^2$) whose coefficients are exact expressions in the external kinematics
and model parameters. Its distinct coefficient "heads" are deduplicated,
optionally factored to remove numerical cancellations, and compiled together
into a single multi-output SymJIT evaluator at optimization level O2. Per
phase-space point NinjaRS evaluates these heads once and scatters them into
Ninja's tensor layout with a precomputed, allocation-free map; MadLoop instead
rebuilds its loop polynomial through an open-loop recursion over loop
wavefunctions at every point.

Because the polynomial is exact, NinjaRS knows the *true* rank of each
numerator. The odd-$gamma$ trace removes the $q^5$ terms of the top-quark
pentagons of $g g -> H H H$ and the $q^3$ terms of the triangles, which NinjaRS
therefore reduces at rank 4 and 2, while MadLoop declares ranks 5 and 3 and
drives Ninja through more expensive expansions (rank $r = n$ instead of the
true $r = n - 1$).

== Batching throughout the pipeline

Every NinjaRS interface — numerator coefficients, cut-numerator callbacks,
reduction and master integrals — processes a batch of phase-space points
together. Callbacks receive one compact batch per cut instead of one call per
point and cut; master integrals are requested once per integral family for
all cuts and points of a batch. Failures are isolated per phase-space lane: a
lane failing a stability check or a master evaluation keeps its previous
result while its neighbours are accumulated.

== Generic numerical domains and derivatives

The generic reduction engine is written once for any Symbolica numerical
domain: binary64, double-double (`DoubleFloat`), arbitrary-precision `Float`
with tracked significant precision, and Symbolica's Dual numbers for first,
higher and mixed derivatives with respect to kinematic or model parameters.
Precision rescue and derivatives therefore need neither a separate
quadruple-precision build nor numerical differencing. Its arithmetic kernels
are generated mechanically from Symbolica instruction graphs.

== A dedicated binary64 engine

For ordinary double-precision evaluation, `ninjars::fast` is a monomorphic
specialization of the same Laurent reduction, batched over phase-space
points, with plain complex arithmetic, precomputed topology data and
allocation-free workspaces. It covers every cut level, any number of
propagators and ranks up to $n + 1$, with Ninja's default options, skip rules
and stability checks, and reproduces original Ninja's results including the
split into cut-constructible and rational parts. It is the engine measured in
@tab:summary and @tab:identical.

== Master integrals from exact expressions: oneloopmaster

oneloopmaster encodes OneLOop's scalar integrals as exact Symbolica
expressions, including analytic-continuation repairs, and generates direct
Rust implementations from them. One source therefore serves binary64,
double-double and arbitrary precision (and Dual numbers through the
interpreter), with lazy conditional branches preserved. The bubble and
tadpole tensor coefficients are native ports of OneLOop's routines. The
corrections to the scalar integrals found while benchmarking (@correctness)
were made in the exact expressions, so all of these routes, and the
hand-written binary64 $C_0$/$D_0$ kernels transcribed from them, carry them. Replayed call by call on the master calls
original Ninja makes in the benchmark, the native masters take 0.24–0.88× the
time of OneLOop.

= Algorithmic improvements <algorithms>

The speed-ups in @tab:identical come from changes in how the work is
organized, not from different mathematics: NinjaRS and Ninja agree to
$10^(-12)$ or better on the physical diagrams, and the larger deviations of
some synthetic cases are inherited binary64 conditioning (@correctness).

- *Topology precomputation.* Cut partitions, complements, parent lists for
  the coefficient-level subtractions and tadpole reference candidates are
  built once per topology. Ninja allocates its cut containers and rescans
  sub-partition relations at every phase-space point.
- *Batched callbacks and masters.* One numerator call per cut (both
  orientations of a triangle share it) and one master call per family per
  batch, instead of per point and cut; all master requests of all cuts and
  points are collected before a single contraction pass.
- *Truncated-jet tensor expansions.* Ninja's `TensorNumerator` materializes
  symmetric tensor powers of the cut vectors ($e_3^r$, $e_3^(r - 1) a$, ...)
  and contracts them with the numerator. NinjaRS instead evaluates the
  homogenized numerator $t^(-R) N(q)$ as one transposed Horner sweep in
  $u = 1 / t$ truncated at the requested order, with exact polynomial
  dependence on $mu^2$ and $x$: every tensor coefficient is touched once per
  expansion. This makes the tensor numerator 2.0–2.5× faster than Ninja's on
  the physical diagrams (1.9–2.7× over all cases).
- *Rank-specialized subtractions.* Ninja's FORM-generated coefficient-level
  subtractions are translated mechanically, and the parent coefficients that
  vanish identically for the given rank are folded away at compile time
  (Ninja evaluates every term and discards some afterwards).
- *Shared per-propagator data.* Products of cut-basis vectors with the
  propagator shifts and their reciprocals are computed once per cut and lane
  and reused by the Laurent divisions of both orientations and by the
  subtractions; each uncut denominator costs one reciprocal.
- *Plain complex arithmetic.* Rust's complex multiply and divide have no
  NaN/infinity recovery, which is what `-fcx-fortran-rules` gives C++; we
  measure original Ninja with that flag so the comparison is not inflated by
  `__muldc3`/`__divdc3` calls (worth 20 % on Ninja's side).
- *True numerator rank.* As described above, the exact numerator avoids the
  inflated ranks MadLoop declares, which makes Ninja's tensor expansions
  3.6× more expensive for the pentagon (81 µs at rank 5 against 23 µs at
  rank 4) and about 5× for the triangles in the MadLoop pipeline.
- *Single-dispatch master kernels.* The generated oneloopmaster code
  evaluated each Laurent order of $C_0$ and $D_0$ separately, repeating the
  channel and sector selection three times. Hand-transcribed binary64
  kernels for the dominant sectors (three-mass triangles, contour boxes and
  the four-mass Denner–Nierste–Scharf box) perform one dispatch per call,
  never evaluate inactive branches, and keep the exact expressions as their
  validated reference; they are 2–3.5× faster than the generated code and,
  on the physical call streams, 1.5–1.8× (boxes) and 1.9–3.7× (triangles)
  faster than OneLOop.
- *Fast binary64 dilogarithm and native tensor bubbles.* oneloopmaster's
  binary64 dilogarithm now uses a dedicated series kernel (17–110 ns per
  call instead of an MPFR detour; contributed to Symbolica as PR \#89), and the
  bubble tensor coefficients $B_0, B_1, B_(11), B_(111)$ come from one
  native evaluation sharing the roots and logarithms (a binary64
  specialization of oneloopmaster's port of OneLOop's bubble routines,
  validated against the exact expressions and Feynman-parameter integrals),
  4–12× faster than OneLOop's `olo_b11` on the physical call streams.
- *On-shell thresholds in one place.* Exact-zero dispatch in oneloopmaster
  is combined with OneLOop's on-shell threshold as Ninja configures it,
  applied once to every master request, and MadLoop's on-shell treatment of
  the kinematic matrix is reproduced when propagator data are prepared.

*Further possible improvements.* The per-point arithmetic of the reduction
is scalar; vectorizing it across phase-space points (structure-of-arrays) is
the natural next step. Master requests could be
deduplicated across diagrams of a process, and the box and triangle masters
could select the cheapest equivalent argument ordering, as OneLOop does
internally. For the generic engine, staged fusion of the reduction stages up
to master weights reduces the instruction count of an $n = 3$, $r = 3$
reduction from 1,884 to 1,071 operations in an exact inventory.

#bibliography("refs.yml", title: "References")
