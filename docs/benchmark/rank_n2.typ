// Rank n+2 section of the NinjaRS benchmark report: validation of NinjaRS's
// rank n+2 results against the high-precision references and COLLIER (no
// timing comparison). Numbers are read from data/rank_n2.json, produced by
// validation/benchmark/collier/collect.py.

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
#let suite = rn2.cases.filter(c => not c.twin)
// "rn2-x-lane1" -> `rn2-x` lane 1
#let lane-name(id) = {
  let parts = id.split("-lane")
  [#raw(parts.at(0)) lane #parts.at(1)]
}
#let mu-free-unshifted = suite.filter(c => c.mu_powers == 0 and not c.shifted).len()
#let flags = rn2.accuracy.collier_flags
#let acc = rn2.accuracy.by_n

= Validation at numerator rank $n + 2$ <rank-n2>

NinjaRS also reduces numerators of rank $r = n + 2$, one beyond Ninja's
higher-rank extension @ninja-paper (the algorithm, the new integrand terms and
their validation are in `docs/rank-n2.md`). Original Ninja cannot serve as the
reference here, so this section validates NinjaRS's rank $n + 2$ results
against high-precision reference values and against the tensor-integral
library COLLIER #rn2.collier.version @collier, with its COLI and DD branches
@denner-dittmaier-5pt @denner-dittmaier-reduction @denner-dittmaier-d0. It
compares accuracy only, not time. All inputs are the rank $n + 2$ bench cases
of `validation/rank-n2`: the 105 suite cases ($n = 1, dots, 7$; real,
complex, mixed and zero masses; numerators with powers of the integrand
$mu^2$ up to $mu^8$; 3 phase-space points each) and the six synthetic cases
`syn-n1-r3` to `syn-n6-r8`, each lane with a high-precision reference value.

*A different method, so an independent check.* COLLIER does not reduce the
integrand. It computes the tensor integrals $T^(N, mu_1 dots mu_P)$ of every
rank $P <= r$ by Passarino–Veltman reduction and, where the Gram or Cayley
determinants are small, by expansions in them; the integral of a numerator is
then the contraction of these tensor components with the numerator's tensor
coefficients. Three consequences shape the comparison:
- *Inputs.* COLLIER takes the tensor in a loop momentum with $p_0 = 0$, and no
  integrand $mu^2$. The harness removes $mu^2$ exactly by pinching
  ($mu^2 = q^2 + 2 q dot V_a + V_a^2 - m_a^2 - D_a$; up to five pieces with
  fewer propagators per lane) and shifts each piece in exact rational
  arithmetic; the pieces are then rounded to binary64 once. On the
  #mu-free-unshifted of #suite.len() cases without $mu^2$ and with
  $V_0 = 0$, COLLIER receives exactly the tensor NinjaRS receives.
- *Output.* One COLLIER evaluation gives the finite part for one choice of its
  regulator parameters $Delta_"UV"$, $Delta_"IR"^((1))$, $Delta_"IR"^((2))$.
  The single and double poles take two more evaluations with shifted
  parameters (only the UV pole is returned separately). NinjaRS returns all
  three Laurent coefficients, and their split into cut-constructible and
  rational parts, in one evaluation. Every comparison below takes the largest
  difference over the three Laurent coefficients, divided by the lane's
  largest reference coefficient (the lane scale).
- *Configuration.* COLLIER #rn2.collier.version (sha256
  #raw(rn2.collier.sha256.slice(0, 16))…) is built twice in binary64:
  `precise`, its gfortran Release flags without `-ffast-math`, as every other
  code in this report; and `stock`, the shipped flags with `-ffast-math`. It
  runs with its default target accuracy $10^(-8)$ (`SetReqAcc_cll`) and
  without its optional global cache, which returns wrong 7-point results at
  rank 9 (see below). DD covers $n <= 4$ at rank $n + 2$ (pentagons up to
  rank 5, hexagons up to 6). The references also hold COLLIER 1.2.9's COLI
  result in quadruple precision (target accuracy $10^(-20)$), a third COLLIER
  evaluation.

NinjaRS's values are those of `ninjars case-check`: the fast engine (dense
`TensorNumerator`, `MuTensorNumerator` with $mu^2$ blocks, native masters) in
binary64, and the generic engine (`SymbolicaAmplitudeKernels`, native
masters) in binary64, DoubleFloat, Float(128) and Float(256), each at the
case's own batch width.

*Provenance.* NinjaRS #raw(rn2.accuracy_head) (oneloopmaster
#raw(rn2.accuracy_oneloopmaster), Symbolica #raw(rn2.accuracy_symbolica)),
#rn2.accuracy_date. Two complete `case-check` runs gave bitwise identical
lanes in all five modes. COLLIER's binary64 values are those of its uncached
validation pass, which a fresh build reproduced bitwise on all 1,122 lanes
(precise and stock, COLI and DD) on #rn2.accuracy_date. The previous
revision of this report also compared the times of the two codes (NinjaRS
#raw(rn2.ninjars_head), #rn2.date); that comparison is no longer maintained,
and its raw files in `validation/benchmark/collier/results/` are kept for
reference only.

#let acc-cell(s) = if s == none or s.lanes == 0 { [—] } else {
  set par(leading: 0.35em)
  [#text(size: 7.6pt, sci(s.median)) \ #text(size: 6.8pt, fill: luma(70), sci(s.max))]
}
#figure(
  kind: table,
  caption: [*Binary64 accuracy against the high-precision references*, on
    the tier A and B lanes (suite and synthetic) with a condition of at most
    $10^6$; `case-check` judges #acc.map(r => r.case_check_judged).sum() of
    them in binary64 and reports the other
    #acc.map(r => r.case_check_report_only).sum() as report-only (in brackets:
    the excluded lanes, tier C, whose published value is COLLIER's own
    quadruple-precision result or lacks an independent check, and conditions
    above $10^6$, on which the binary64 results of at least one of the two
    codes are not reliable). Scaled difference as `ninjars case-check` computes
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

*Accuracy.* On the #acc.map(r => r.lanes).sum() tier A and B lanes (suite
and synthetic) with a condition of at most $10^6$ (@tab:rank-n2-accuracy;
`case-check` judges #acc.map(r => r.case_check_judged).sum() of them in
binary64 and reports #acc.map(r => r.case_check_report_only).sum(), whose
tolerance cannot resolve the new rank $n + 2$ term, as report-only), the
fast engine's error is at most COLI's on #acc-better of #acc-all lanes. Both lose accuracy
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
COLLIER's contraction terms cancel, which hurts COLI, or the new rank $n + 2$
terms cancel, which hurts NinjaRS) the binary64 results are not reliable for at least one of
the two codes: the errors reach
#sci(calc.max(..acc.map(r => r.at("fast_all_ab").at("max", default: 0)))) for
the fast engine and
#sci(calc.max(..acc.map(r => r.at("precise-coli_all_ab").at("max", default: 0)))) for COLI,
while on #acc.map(r => r.ill_conditioned_one_below_1e-8).sum() of them one of
the two stays below $10^(-8)$.

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
#let hp = fresh.at("high_precision", default: (:))
#let fvc = fresh.at("ninjars_vs_collier", default: (:)).at("fast/coli", default: none)
#let fresh-unflagged = fresh.at("ninjars_above_1e-6") - fresh.at("ninjars_above_1e-6_flagged")
*Fresh cases.* The final review generated #fresh.cases further rank $n + 2$
cases (#fresh.lanes lanes) with references to 60–100 digits, including
deliberately ill-conditioned light-like kinematics
(@tab:rank-n2-fresh). There NinjaRS's binary64 error exceeds $10^(-6)$ on
#fresh.at("ninjars_above_1e-6") lanes, all of whose cut contributions cancel by
more than #sci(fresh.at("above_1e-6_condition_min"), digits: 0) relative to
the result, an instability of integrand reduction in fixed precision. COLI
stays below $10^(-6)$ on #fresh.at("above_1e-6_coli_below_1e-6") of these
#fresh.at("ninjars_above_1e-6") lanes: its error on them has median
#sci(fresh.at("above_1e-6_coli").median) and reaches
#sci(fresh.at("above_1e-6_coli").max) on the worst. The generic engine's
opt-in cancellation test (`--stability-test`) reports
#fresh.at("ninjars_above_1e-6_flagged") of them as `Unstable`; it leaves
#fresh.at("ninjars_above_1e-5_unflagged") lanes with an error above
$10^(-5)$ unflagged (it misses
#(fresh-unflagged - fresh.at("ninjars_above_1e-5_unflagged")) between
$10^(-6)$ and $10^(-5)$) and flags #fresh.at("flagged_below_1e-6") lanes whose
error is below $10^(-6)$. COLI in turn exceeds $10^(-6)$ on
#coli-bad.len() lanes, #coli-bad.filter(l => l.flags.at(1) < 0).len() of them
with accuracy flag $-1$.
#if fvc != none [
  Lane by lane, with NinjaRS's tolerance set to $10^(-6)$, the fast engine
  and COLI disagree on #fvc.disagree of #fvc.compared lanes; the references
  put COLLIER right on #fvc.disagreements_right.at("collier", default: 0),
  NinjaRS on #fvc.disagreements_right.at("ninjars", default: 0) and neither
  on #fvc.disagreements_right.at("neither", default: 0).
]
#if hp.len() > 0 [
  Flagged points can be recomputed at higher precision. The generic engine
  returns every fresh lane as `Success` in DoubleFloat, Float(128) and
  Float(256), so the tracked-precision check of its `Float` masters fails
  none, and its errors are at most #sci(hp.double.max) in DoubleFloat,
  #sci(hp.float128.max) in Float(128) and #sci(hp.float256.max) in
  Float(256). The last, on #lane-name(hp.float256.largest_id),
  is the references' own floor: their scalar $C_0$/$D_0$ table carries 45
  digits, which the lane's cancellation amplifies, and Float(512) agrees
  with Float(256) to $10^(-55)$ there.
]

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

#let cache = rn2.accuracy.at("collier_cache", default: ())
#let cache7 = cache.find(r => r.n == 7)
#let cache-low = cache.filter(r => r.n <= 6)
#if cache7 != none [
  *COLLIER's global cache returns wrong results at $n = 7$.*
  COLLIER can reuse integrals across calls through an optional global cache
  (`InitCacheSystem_cll`, `InitEvent_cll` before every phase-space point).
  Validated with one cache per lane, in the cache mode of an event loop (15
  warm-up events per lane cache), it returns wrong rank-9 heptagons: all #cache7.lanes $n = 7$ lanes
  differ from the uncached values, the
  #cache7.judged.lanes compared ones have scaled errors from
  #sci(cache7.judged.cache_min) to #sci(cache7.judged.cache.max) (at most
  #sci(cache7.judged.nocache.max) without the cache), and COLLIER reports
  accuracy flag $-2$ on #cache7.judged.at("cache_accflag_-2") of them. The
  manual's usage alone reproduces it
  (`validation/benchmark/collier/collier_cache_repro.f90`: one cache, one
  `TNten_cll` call with $N = 7$, $R = 9$ per event, the same point in every
  event, no parameter change): from the third event on, the tensor is off by
  3.9 times its largest component, with flag $-2$; at rank 8 the cached
  tensor is bitwise identical. At $n <= 6$ the cache changes
  #cache-low.map(r => r.differ_from_nocache).sum() of
  #cache-low.map(r => r.lanes).sum() lanes at the level of COLLIER's own
  accuracy. The comparisons therefore use COLLIER without the cache, its
  default (`COLLIER_CACHE_BUG.md` describes the defect).
]

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

*Correctness of NinjaRS at rank $n + 2$* is checked lane by lane by
`ninjars case-check` (@tab:rank-n2-case-check): no lane fails in any of the five
runs, which reproduce `docs/rank-n2.md` §8.3, every lane's status is
`Success` (the tracked-precision check of the `Float` masters flags none),
and the three known failures that Float(128) had in the previous revision are
gone with Symbolica's corrected `Float` products.

#let vc = rn2.accuracy.at("ninjars_vs_collier", default: none)
#if vc != none [
  #let right-label = (ninjars: [N], collier: [C], neither: [—])
  #let vc-cell(b) = {
    set par(leading: 0.35em)
    text(size: 8pt)[#b.agree / #b.compared]
    if b.disagree > 0 {
      linebreak()
      text(size: 6.8pt, fill: luma(70))[#b.disagree: #b.disagreements_right.pairs().map(((k, v)) => [#right-label.at(k) #v]).join([, ])]
    }
  }
  #let v64 = vc.at("fast/f64")
  #figure(
    kind: table,
    caption: [*NinjaRS against COLLIER, lane by lane*, per `case-check` run
      (the #v64.lanes lanes of @tab:rank-n2-case-check). A lane agrees when
      NinjaRS and COLLIER differ by at most the sum of the two codes' own
      claims: `case-check`'s tolerance for NinjaRS, and for COLLIER
      $max(10^(-8) max(1, A), E)$ in binary64 ($A$ the sum of its
      contraction terms' magnitudes, $E$ its error estimate, both over the
      lane scale) or $10^(-20) max(1, "condition")$ in quadruple precision
      (COLI quad, COLLIER 1.2.9). Each cell gives agreeing / compared lanes
      and, below, the number of disagreements with who is right according to
      the references (N: NinjaRS, C: COLLIER, —: neither). The other
      #v64.coli.no_ninjars_tolerance lanes, tier C at $n = 7$, have no
      NinjaRS tolerance. "Within tolerance" counts the lanes within
      `case-check`'s tolerance of the references (every lane outside it is
      report-only); the last column, the lanes within $10^(-12)$ of COLI quad.],
    header-style(table(
      columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr),
      align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
      inset: (x: 4pt, y: 3.5pt),
      stroke: table-stroke,
      fill: (x, y) => if y == 0 { navy } else { none },
      table.header([NinjaRS run], [within tolerance], [COLI binary64],
        [DD binary64], [COLI quad], [within $10^(-12)$ of COLI quad]),
      ..for (k, label) in cc-runs {
        let v = vc.at(k)
        (
          label,
          text(size: 8pt)[#v.reference.within_tolerance / #v.reference.with_tolerance],
          vc-cell(v.coli), vc-cell(v.dd), vc-cell(v.coli_quad),
          text(size: 8pt)[#v.coli_quad.within.at("1e-12") / #v.coli_quad.compared],
        )
      },
    )),
  ) <tab:rank-n2-vs-collier>
  #let dis = v64.coli.disagreements.at(0)
  #let gen-dis = vc.at("generic/f64").coli.disagreements.find(d => d.id == dis.id)
  #let miss = v64.coli.collier_misses.at(0)

  *NinjaRS against COLLIER, lane by lane* (@tab:rank-n2-vs-collier). With
  each code's own claim as its tolerance, NinjaRS agrees with binary64 COLI
  on #v64.coli.agree of #v64.coli.compared lanes in every run, and with DD
  on all #v64.dd.compared lanes DD can evaluate, by at most
  #sci(v64.dd.largest_difference). COLLIER's binary64 claim is loose: it
  exceeds $10^(-6)$ on #v64.coli.at("collier_claim_above_1e-6") of the
  #v64.coli.compared COLI lanes and #v64.dd.at("collier_claim_above_1e-6") of
  the #v64.dd.compared DD lanes. At a fixed threshold the fast engine and COLI
  differ by at most #sci(v64.coli.comparison_set.largest_difference) on the
  #v64.coli.comparison_set.lanes lanes of @tab:rank-n2-accuracy, and by at
  most $10^(-8)$ on #v64.coli.comparison_set.at("within_1e-8") of them. Besides the
  #v64.coli.no_ninjars_tolerance tier C lanes, the
  #v64.coli.disagree COLI disagreements are COLLIER errors with accuracy
  flag 0:
  - #lane-name(miss.at(0)): COLI is off by #calc.round(miss.at(1), digits: 2)
    (its own claim #sci(miss.at(2))), NinjaRS by $5 times 10^(-10)$.
  - #lane-name(dis.id): COLI is off by #sci(dis.collier_error) (its claim
    #sci(dis.collier_claim)). In binary64 NinjaRS misses its own budget of
    #sci(dis.ninjars_tolerance) there too, by #sci(dis.ninjars_error)
    (fast) and #sci(gen-dis.ninjars_error) (generic, flagged by
    `--stability-test`; a report-only lane). From DoubleFloat on it is
    right.

  Against COLI quad, binary64 NinjaRS is the one off only on three
  report-only lanes with a cancellation above $10^6$ (the condition, or for the
  lane above the new rank $n + 2$ terms' $5 times 10^7$): the lane above, and with
  the generic engine `rn2-int-real-n7-r9-mu2` lane 2 ($2.3 times 10^(-4)$)
  and `rn2-shifted-mixed-n7-r9-mu2` lane 0 ($1.6 times 10^(-4)$). From
  DoubleFloat on, NinjaRS is within $10^(-12)$ of COLI quad on
  #vc.at("generic/double").coli_quad.within.at("1e-12") of
  #vc.at("generic/double").coli_quad.compared lanes, and where they disagree,
  the references show that COLLIER misses its $10^(-20)$ target
  (`validation/rank-n2/README.md`, finding 1), with one exception:
  `rn2-phys-real-n6-r8` lane 0 in Float(256), $7.5 times 10^(-12)$ from its
  reference against a tolerance of $2.8 times 10^(-12)$ (report-only).
  DoubleFloat, Float(128) and Float(512) agree with that Float(256) value to
  $3 times 10^(-25)$: the deviation is the input ambiguity of the lane's
  snapped S-matrix (`docs/rank-n2.md` §8.3), not a loss of precision. On
  `rn2-int-real-n7-r9-muall` lanes 0 and 1, whose only reference is COLLIER's
  quad value, NinjaRS differs from that value by 0.86 and 0.14 in DoubleFloat,
  Float(128) and Float(256), and these agree with a Float(512) evaluation to
  $1.2 times 10^(-14)$ or better (binary64 COLI is off by $2.4 times 10^4$ and
  $2.3 times 10^3$): COLLIER is wrong there, as `validation/rank-n2/README.md`
  found.
]

*The COLLIER driver* reproduces
the binary64 COLLIER 1.2.9 diagnostics stored with the references (exact
decimal inputs and a 90-digit contraction there, binary64 here): on the
#agree.at("precise-coli").at("above_1e-14") COLI lanes with an error above
$10^(-14)$ the ratio of the errors has median
#fixed(agree.at("precise-coli").median_ratio, 3), and the
#agree.at("precise-coli").differ_2x.len() lanes that differ by more than a
factor of two are $mu^2$ lanes, whose pieces this harness rounds to binary64
before the contraction. COLLIER 1.2.10 changes, against 1.2.9, complex-mass
$C_0$ root prescriptions and two error-estimate guards. Building and running
COLLIER showed, besides the cache defect, one defect in the reference
harness: the harness combines COLLIER's accuracy flag with `max` from 0, in
its Fortran driver (`collier_tensor_driver.f90`) and again over a lane's
pieces (`collier_reference.py`), and so never reports a negative flag (no
reference value uses it).
