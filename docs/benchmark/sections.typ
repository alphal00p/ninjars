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
  original Ninja to $7 times 10^(-13)$ (relative to the largest Laurent
  coefficient) on every lane of the physical cases and to
  $4 times 10^(-12)$ on 30 dedicated edge cases (light-like and zero
  invariants, mixed and complex masses, infrared-divergent triangles and
  boxes, all tadpole reference bases, ranks $n - 5, dots, n + 1$ and
  $n = 7$), including the split into cut-constructible and rational parts.
  Where the two differ more (rank $n + 1$ hexagons, up to $10^(-7)$), both
  are compared with the generic NinjaRS engine at 256 bits: the error ratio
  NinjaRS ÷ Ninja has median 1.1, i.e. the deviation is inherited binary64
  conditioning, not a porting error. Results are bitwise identical for every
  batch width and lane order, and the warmed path performs no heap
  allocation.
- *End to end.* Every physical diagram matches original Ninja fed with the
  same tensors to $2 times 10^(-12)$ or better (last column of
  @tab:identical), and the NinjaRS and MadLoop integrated results differ by
  a single constant normalization over all phase-space points ($2$ for the
  QCD diagrams, $1/3$ for $gamma gamma -> gamma gamma$, spread
  $lt.eq 10^(-14)$), which fixes diagram identity, routing and helicity.
- *Masters.* oneloopmaster's native $C_0$/$D_0$ agree with OneLOop 3.6 on
  all 783 distinct calls of the benchmark streams (median $2 times
  10^(-15)$); the native bubble tensors were validated against 595 direct
  Feynman-parameter integrations, including thresholds and massless limits.

*Defects found on the way.* The comparison exposed and fixed: an MPFR detour
in oneloopmaster's binary64 dilogarithm (performance), inexact unit roots in
massless bubble tensors, rounded light-like invariants entering the masters
without OneLOop's on-shell threshold (a $10^(-7)$ error in
$gamma gamma -> gamma gamma$ below threshold), a degenerate rational bubble
reference vector in the generic NinjaRS engine (an $O(10^13)$ blow-up at
exactly coplanar rational kinematics), and inaccurate Passarino–Veltman
bubble tensors in an intermediate provider. It also exposed defects in
oneloopmaster's exact expressions for $B_0$ with a massless second line,
$C_0$ with one massless internal line and some real-mass $D_0$ sectors,
none of which occur in the physical benchmark diagrams — and two wrong
$D_0$ values in reference copies of OneLOop itself (versions 3.4 and a 2024
snapshot; OneLOop 3.6 and COLLIER agree with each other there).

= How the numbers are measured <methodology>

*Test cases.* Seven physical one-loop diagrams with a top-quark loop,
generated independently by FeynKit (NinjaRS) and MadGraph5_aMC\@NLO 3.7.0
(MadLoop), and matched graph by graph: the propagator invariants and masses
agree exactly and the integrated results agree up to a constant
normalization over all phase-space points. Each diagram is evaluated at
three phase-space points (exact rational/radical momenta rounded once to
binary64), one fixed helicity configuration, zero widths and
$mu^2 = 100^2$ GeV#super[2]. Twenty synthetic topologies ($n = 2, dots, 6$
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
code); and the uninstrumented total. The components add up to the total
within 2 %. `ninjars case-bench` performs the same breakdown on the shared
bench-case files, which contain exactly the propagators and tensor
coefficients given to original Ninja.

*MadLoop pipeline.* Optimized standalone MadLoop output (`loop_optimized_output`),
restricted to the same single loop diagram, Ninja as the only reduction
library, no stability rotations, linked against original Ninja 1.2.1
(`g++ -O3 -fcx-fortran-rules`, integral cache disabled) and OneLOop 3.6
(`gfortran -O3`). The numerator time is MadLoop's loop-coefficient
construction plus Ninja's tensor-numerator expansions; reduction and master
time come from an instrumented Ninja build and are cross-checked by
replaying MadLoop's exact Ninja inputs in a C++ driver. The table uses the
faster MadLoop variant for every quantity: `GLOBAL_FLAG=-O3` for evaluation
and the default flags for compilation time.

*Identical-input comparison.* The C++ driver feeds original Ninja the same
bench cases (`Amplitude<ComplexMasses>`, `TensorNumerator`, explicit
S-matrix) and replays recorded numerator outputs and OneLOop results to
isolate each component, exactly like the NinjaRS harness. Every replay
reproduces the direct evaluation bit for bit.

*Environment.* One pinned core of a shared AMD EPYC 9754 host, Rust 1.98
(`--profile perf`: opt-level 3, one codegen unit for the hot crates), GCC and
gfortran 15.3, no `-march`/`-ffast-math` on either side. Timings are medians
of at least seven blocks of at least 20 ms; repeated runs agree within
1–2 %.

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
drives Ninja through more expensive expansions (for the pentagon through its
higher-rank $r = n + 1$ path).

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
tadpole tensor coefficients are native ports of OneLOop's routines.

= Algorithmic improvements <algorithms>

The speed-ups in @tab:identical come from changes in how the work is
organized, not from different mathematics; NinjaRS and Ninja agree to
$10^(-12)$ or better on well-conditioned points.

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
  expansion. This makes the tensor numerator 2–2.6× faster than Ninja's.
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
  inflated ranks MadLoop declares, which roughly triples the cost of the
  tensor expansions for the pentagon in the MadLoop pipeline.
- *Single-dispatch master kernels.* The generated oneloopmaster code
  evaluated each Laurent order of $C_0$ and $D_0$ separately, repeating the
  channel and sector selection three times. Hand-transcribed binary64
  kernels for the dominant sectors (three-mass triangles, contour boxes and
  the four-mass Denner–Nierste–Scharf box) perform one dispatch per call,
  never evaluate inactive branches, and keep the exact expressions as their
  validated reference; they are 2–3.5× faster than the generated code and
  about twice as fast as OneLOop.
- *Fast binary64 dilogarithm and native tensor bubbles.* oneloopmaster's
  binary64 dilogarithm now uses a dedicated series kernel (12–79 ns per call
  instead of an MPFR detour; contributed to Symbolica as PR \#89), and the
  bubble tensor coefficients $B_0, B_1, B_(11), B_(111)$ come from one
  native evaluation sharing the roots and logarithms, 3–5× faster than
  OneLOop's `olo_b11`.
- *On-shell thresholds in one place.* Exact-zero dispatch in oneloopmaster
  is combined with OneLOop's on-shell threshold as Ninja configures it,
  applied once to every master request, and MadLoop's on-shell treatment of
  the kinematic matrix is reproduced when propagator data are prepared.

*Further possible improvements.* The reduction is now latency-bound per
lane; vectorizing the per-lane arithmetic across phase-space points
(structure-of-arrays) is the natural next step. Master requests could be
deduplicated across diagrams of a process, and the box and triangle masters
could select the cheapest equivalent argument ordering, as OneLOop does
internally. For the generic engine, staged fusion of the reduction stages up
to master weights reduces the instruction count of an $n = 3$, $r = 3$
reduction from 1,884 to 1,071 operations in an exact inventory.

#bibliography("refs.yml", title: "References")
