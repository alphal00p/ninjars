# Rank n+2 performance guard: main dec3d55 against claude/rank-n2-integration 8346699

This is the final A/B of design section 8 for rank n+2 support.
- **A (baseline):** main `dec3d55`.
- **B (candidate):** `claude/rank-n2-integration` `8346699`. It contains main `dec3d55`, merged in `1524dbd`.

Both were built with `cargo build --locked --profile perf -p ninjars-cli --features numerator-jit`. Each was built from an exported tree into its own target directory, and both ran on identical inputs. Host, toolchain, binary hashes and cores are listed in [results/environment.txt](results/environment.txt).

**Rerun.** On 2026-10-19 the guard was rerun against main `d53dd08`, with current main `0d0d7e7` as the candidate. It passes. Its investigation shows that the `syn-n2-r2` outlier below is a heap-layout effect, not code placement: see [Rerun against main d53dd08](#rerun-against-main-d53dd08-and-the-syn-n2-r2-outlier-todo-item-23).

## Verdict

| Check (design 8) | Criterion | Result | |
|---|---|---|---|
| (a) every ratio | ≤ 1.03 | 190 of 192 keys pass. Over: `syn-n2-r2` width 1 reduction 1.061, `syn-n2-r1` width 1 reduction 1.040 | see (c), (d) |
| (b) geomean of the reduction and total ratios | ≤ 1.01 | **0.9975** (164 keys); all ratios 0.9983 | pass |
| (c) ratio in (1.03, 1.05]: rerun alone, 9 rounds | rerun ≤ 1.03 | `syn-n2-r1` w1 reduction: **1.018** (3 cores × 9 + 9 runs) | pass |
| (d) any ratio > 1.05 | fail, investigate | `syn-n2-r2` w1 reduction: 1.061, rerun **1.11** on 3 cores. Same instructions; it is a code-layout effect, see below | **fail by the letter** |
| (e) instructions per point (27 cases × widths 1, 32; replay and live) | within 0.5 % | 107 of 108 within. Worst: `syn-n2-r1` w1 replay +0.503 % (+24 instructions per call in the probe binary; the CLI's replay executes the same count, see below) | pass; one point 0.003 percentage points over |
| (f) absolute gates against Ninja, width 32 | as today | red× ≤ 0.60, mast× ≤ 0.88, tot× ≤ 0.66 (A: 0.61, 0.87, 0.67) | pass |
| (g) generic `ninjars bench` | ≤ 1.05 | max **1.019**, geomean 1.005; instructions per point within ±0.4 % | pass |
| A/A noise (same binary under two names) | spread ≤ 2 % | geomean 0.9993, max ratio 1.022; run-to-run spread 0.33–0.43 % (median), ≤ 2.3 % | ok |

- **Lane results:** the per-lane results of every pipeline-bench point (`lane_results`) and every case-bench reference check are identical across all A and B runs.
- **What remains:** one key, `syn-n2-r2` (n = 2, rank 2) at width 1, replayed reduction only, is 35–45 ns slower per call: 460 against 505 ns.
  - It is consistent across cores and binaries, and it is not caused by extra work.
  - Its total (live numerator and masters) is unchanged: 1.003, reruns 0.98–0.99. So is its width-32 reduction: 1.007, reruns 1.018.

## Protocol

The commands are those of `run_final2.sh` ([scripts/run_one.sh](scripts/run_one.sh)):
- pipeline-bench on the 7 physical graphs, widths 1, 8, 32 and 128;
- case-bench on the 7 physical and 20 synthetic cases, widths 1 and 32;
- options `--blocks 9 --min-block-ms 20` and `--reference`, everything pinned with `taskset`.

Runs were interleaved on two idle cores in different L3 complexes, CPU 189 and CPU 197:
- each core did one A/B (`ABBA…`, 10 + 10 runs) and one A/A (8 + 8 runs);
- the order was swapped between the cores (core 189: A/B then A/A; core 197: A/A then A/B).

Per key (case, width, metric), the statistic is the median over runs of the per-run median ns/point; the ratio is B/A. The two cores' ratios are combined by geometric mean. The metrics are those of the design: case-bench reduction and total; pipeline-bench reduction, masters and total. Script: [scripts/analyze.py](scripts/analyze.py).

## Fast engine (192 keys)

| Metric | Keys | Geomean B/A | Min | Max | A/A max |
|---|---|---|---|---|---|
| case-bench reduction (replay) | 54 | 0.9916 | 0.920 | 1.061 | 1.004 |
| case-bench total | 54 | 1.0011 | 0.988 | 1.013 | 1.004 |
| pipeline reduction | 28 | 0.9971 | 0.971 | 1.005 | 1.010 |
| pipeline masters | 28 | 1.0033 | 0.988 | 1.011 | 1.022 |
| pipeline total | 28 | 1.0026 | 0.996 | 1.007 | 1.005 |

Pipeline at width 32 (B/A):

| Graph | Reduction | Masters | Total |
|---|---|---|---|
| hgg top-triangle-0 | 0.996 | 0.999 | 1.000 |
| four-photon top-box-0 | 1.002 | 1.004 | 1.003 |
| gg-hh top-triangle-0 | 0.997 | 1.003 | 0.997 |
| gg-hh top-box-2 | 1.004 | 1.009 | 1.006 |
| gg-hhh top-triangle-0 | 0.997 | 1.001 | 1.003 |
| gg-hhh top-box-8 | 1.004 | 1.010 | 1.006 |
| gg-hhh top-pentagon-26 | 0.997 | 1.006 | 1.006 |

- **Masters:** these include the oneloopmaster pin bump 264a6ce → fe550e3 (B1111 added). The masters geomean is 1.003.
- **Replay reductions faster in B:** several replayed reductions at width 32 are 5–8 % faster in B, for example `syn-n4-r5` at 0.92. The code is the same, so this is the same kind of placement effect in the other direction.

Full tables:
- [results/fast-ab.tsv](results/fast-ab.tsv) and [results/fast-ab.txt](results/fast-ab.txt);
- A/A: [results/fast-aa.tsv](results/fast-aa.tsv);
- reruns: [results/rerun-syn-n2.txt](results/rerun-syn-n2.txt).

### The `syn-n2-r2` width-1 reduction ([results/layout-syn-n2-r2.txt](results/layout-syn-n2-r2.txt))

- **Same work.** callgrind on the CLI binaries counts, per replayed call:
  - `evaluate`/`evaluate_up_to_n1`: 5038 instructions in A, 5037 in B;
  - the timed closure: 5072 in A, 5082 in B. The rank dispatch is inlined there, and `measure()` is no longer inlined (+7);
  - all callees: identical call counts and instructions.

  The machine code of the replay evaluate, `bubbles`, `prepare` and `integrate` is the same, apart from 7 register renames and one moved store ([results/dis-compare-cli.txt](results/dis-compare-cli.txt)).
- **Extra cycles, not instructions.** Per 1000 instructions, B has:
  - +8 % loads crossing a cache line (4.68 → 5.05);
  - +7–10 % op-cache misses;
  - +7 % "load not complete" stalls.

  IPC drops from 3.41 to 3.30. L1i misses, iTLB misses and branch mispredictions are unchanged.
- **Layout controls tried; none removed it:**
  - 64-byte function and loop alignment, both sides rebuilt;
  - eight glibc malloc configurations, the default and seven tunables (the gap ranged from about +3 to +12 %);
  - every stack offset of the engine frame, 0–112 bytes, in a separate probe binary. There B is +8 % cycles for +0.85 % instructions at every offset ([results/layout-stackprobe-syn-n2-r2.tsv](results/layout-stackprobe-syn-n2-r2.tsv)).
- **Placement.** The non-generic hot functions of this path come from the single `ninjars` codegen unit. They span 136 KiB in A and 475 KiB in B, because the rank n+2 modules (`engine_x2`, `bases_x2`, `subtractions_x2`, `subtractions_series_x2`) sit between them. The effect is front-end and load-side, and it shows only where one call is about 460 ns.
- **Not tried:** moving the rank n+2 modules into their own codegen unit or crate, or ordering functions at link time. That is the remedy if (d) is to be met literally.
- **Corrected by the rerun.** The placement explanation does not hold.
  - With the rank n+2 code linked out of the span, 8346699 stays 8 % slower. It is as fast as dec3d55 at other heap layouts, or with 64-byte aligned heap blocks.
  - See [the rerun's investigation](#the-syn-n2-r2-outlier-is-heap-layout-not-code-placement-results-0d0d7e7layout-investigationtxt).

## Instruction counts ([results/icount-fast.txt](results/icount-fast.txt))

- **Probe:** [scripts/ab_icount.rs](scripts/ab_icount.rs), compiled into both trees. It counts user-space instructions and cycles with `perf_event_open`, on case-bench's replayed reduction and on the live path (TensorNumerator + OneLoopNativeMasters).
- **Repeatability:** three repetitions per side, identical to ≤ 1.2e-6.
- **Replay:** ratios 0.9997–1.0050. The +24–75 instructions per call at width 1 come from register-allocation differences in this test binary's `evaluate`, which inlines `integrate` (the CLI's replay `evaluate` executes the same instruction count in A and B). They are amortised at width 32, where the ratio is ≤ 1.0023.
- **Live:** ratios 0.9996–1.0019.
- **Generic engine** ([results/generic-icount.tsv](results/generic-icount.tsv)): instructions per timed point, from the difference of 2-round and 12-round runs. Ratios 0.996–1.004.

## Generic engine ([results/generic-ab.txt](results/generic-ab.txt))

- **Setup:** `ninjars bench` on `box` (n=4), `pentagon-rank4` (n=5), `heptagon-rank8` (n=7, r=n+1) and the physical gg-hh `top-box-2` pool.
  - These run in binary64; box and pentagon also in DoubleFloat and Float(128).
  - Two interleaved replicates on CPU 205, 16 + 16 runs.
- **Results:**
  - geomean 1.005, max 1.019 (box f64 width 32);
  - DoubleFloat and Float(128): 0.993–1.002;
  - representative results identical.
- **Bimodal f64 runs.** The binary64 runs are bimodal from run to run, by up to 15 % on both sides. With 8 runs per side one replicate showed 1.06. That is why 16 runs per side and the instruction counts are reported.

## Rank n+2 cost against rank n+1 twins (B only; [results/twin-cost.txt](results/twin-cost.txt))

### Fast engine, case-bench

The cases are 57 pairs with the same kinematics:
- `syn-n{2..6}-r{n+2}` against `syn-n-r{n+1}`;
- the rank n+2 suite `rn2-{int-complex,int-real,int-mixed,shifted-mixed,phys-complex,phys-real}` with `-mu2` variants, against `validation/rank-n2/controls`.

The table gives the median over families of the ratio rank n+2 / rank n+1. Reduction is replayed, total is live.

| n | Pairs (q-only / mu2) | Red w1 | Red w32 | Total w1 | Total w32 | Instructions red w32 | Instructions live w32 |
|---|---|---|---|---|---|---|---|
| 2 | 7 / 3 | 2.01 | 2.40 | 2.39 | 2.87 | 2.30 | 2.85 |
| 3 | 7 / 3 | 2.43 | 2.93 | 2.62 | 2.71 | 2.50 | 2.67 |
| 4 | 7 / 3 | 2.39 | 2.88 | 2.20 | 2.26 | 2.46 | 2.28 |
| 5 | 7 / 3 | 2.36 | 2.72 | 1.93 | 1.99 | 2.39 | 2.06 |
| 6 | 7 / 3 | 2.33 | 2.59 | 1.77 | 1.76 | 2.33 | 1.89 |
| 7 | 4 / 3 | 2.27 | 2.47 | 1.60 | 1.59 | 2.26 | 1.73 |

- **mu2 twins:** their reduction ratios agree with the q-only ones within 0.03; their total ratios differ by up to 0.18.
- **Absolute, width 32, `rn2-int-complex`:** reduction 1950 / 11835 / 35322 / 77943 / 149407 / 255293 ns per point for n = 2…7, against 810 / 4045 / 12245 / 28684 / 57804 / 103473 for the twins.
- **Soft target:** the design's target is ≤ 2.5× reduction. It is met at n = 2 and 7, and exceeded at n = 3–6 (2.6–2.9× at width 32).
  - The instruction ratio is 2.3–2.5. The rest is lower IPC in the rank n+2 engine.
  - The S3 report measured 2.1–2.9× (on `rn2-int-complex`, before the conditioning fix 79f306a).

### Generic engine and the fast engine on n-gon fixtures

`ninjars bench` runs on n-gon twins with the same kinematics ([scripts/make_twin_fixtures.py](scripts/make_twin_fixtures.py)). The numerator is dense with mu² terms, of rank n+1 and n+2. The table gives the ratio rank n+2 / rank n+1.

| n | Generic f64 w1 | Generic f64 w32 | DoubleFloat w1 | Float(128) w1 | Fast (`--engine fast`, MuTensorNumerator) w32 |
|---|---|---|---|---|---|
| 2 | 2.47 | 2.26 | 2.53 | 2.91 | 3.02 |
| 3 | 2.10 | 2.15 | 1.30 | 2.09 | 2.85 |
| 4 | 2.41 | 2.45 | 1.29 | 1.97 | 2.37 |
| 5 | 2.58 | 2.67 | 1.34 | 2.29 | 2.11 |
| 6 | 2.70 | 2.75 | 1.31 | 2.17 | 1.96 |
| 7 | 3.05 | 2.72 | 1.31 | 2.12 | 1.88 |

All lanes succeeded. For absolute values, see [results/twin-cost-ngon.tsv](results/twin-cost-ngon.tsv).

### Native bubble masters ([results/bubble-masters.tsv](results/bubble-masters.tsv))

The rows are the 576 propagator pairs of the 27 cases, timed through `FastMasters` ([scripts/ab_bubble_masters.rs](scripts/ab_bubble_masters.rs)). Values are ns per row, batch 1 / batch 576.

| Family | A | B |
|---|---|---|
| BubbleRank2 (B0, B1, B11) | 150 / 103 | 151 / 104 |
| BubbleRank3 (+ B111) | 163 / 114 | 165 / 113 |
| BubbleRank4 (+ B1111) | – | 174 / 125 |

B1111 costs 1.06× (batch 1) and 1.10× (batch 576) the rank-3 family. OneLOop's rank-4 call was not timed here.

## Rerun against main d53dd08, and the `syn-n2-r2` outlier (TODO item 2.3)

On 2026-10-19 the guard of design section 8 was rerun against current main. The `syn-n2-r2` width-1 outlier of the first guard was also investigated with interleaved, pinned runs.
- **A:** main `d53dd08`, the last main before the rank n+2 merge (oneloopmaster `09b2c74`, gammaloop `474478a7`).
- **B:** main `0d0d7e7` (oneloopmaster `09b2c74`, gammaloop `1e8063a7`).
- **B':** none. No placement remedy changed the outlier (below), so none was adopted, and the crates are unchanged.

Results are in [results-0d0d7e7/](results-0d0d7e7/) and the sequence in [scripts/run_rerun.sh](scripts/run_rerun.sh). Builds, binary hashes and cores are listed in [results-0d0d7e7/environment.txt](results-0d0d7e7/environment.txt).

**Each side reads its own checkout** (`INPUTS_A`, `INPUTS_B` in `ab.sh`). B's 70 physical DOT graphs carry the model fingerprint of the new gammaloop pin, and A's pipeline-bench rejects them; the reverse also fails. Cases, points and Ninja references are identical.

### Verdict

| Check (design 8) | Criterion | Result | |
|---|---|---|---|
| (a) every ratio | ≤ 1.03 | 191 of 192 keys pass. The only key over: `gg-hhh-top-box-8` width-1 case-bench reduction, 1.038 (1.039 and 1.037 on the two cores) | see (c) |
| (b) geomean of the reduction and total ratios | ≤ 1.01 | **0.9996** (164 keys); all ratios 0.9990 | pass |
| (c) ratio in (1.03, 1.05]: rerun alone, 9 rounds | rerun ≤ 1.03 | `gg-hhh-top-box-8` w1 reduction: **1.012** (9 + 9 runs on each of 2 cores) | pass |
| (d) any ratio > 1.05 | fail | none; `syn-n2-r2` w1 reduction **0.993** (rerun alone 0.973) | pass |
| (e) instructions per point (27 cases × widths 1, 32; replay and live) | within 0.5 % | 107 of 108 within. Worst: `syn-n2-r1` w1 replay +0.50 % (+24 instructions per call), only in the probe binary; the CLI's replay `evaluate` executes 4716 (A) and 4715 (B) per call | pass; one point 0.003 percentage points over, in the probe only, as in the first guard |
| (f) absolute gates against Ninja, width 32 | as today | B: red× ≤ 0.60, mast× ≤ 0.86, tot× ≤ 0.68 (A: 0.60, 0.87, 0.67) | pass |
| (g) generic `ninjars bench` | ≤ 1.05 | max **1.011**, geomean 0.976 and 0.990 (two replicates) | pass, but see the results note below |
| A/A noise (same binary under two names) | spread ≤ 2 % | geomean 1.0003, max ratio 1.022; run-to-run spread 0.65–0.70 % (median), ≤ 3.2 % | ok |

- **Fast-engine lane results:** the per-lane results of every pipeline-bench point and every case-bench reference check are identical across all A and B runs.
- **Generic representative results:** they differ between A and B in the last digits for pentagon-rank4 (binary64, DoubleFloat and Float(128)), heptagon-rank8 and gg-hh `top-box-2`. For example, the pentagon-rank4 binary64 finite part reads `-4.383624845145409066e-2` in A and `...406290e-2` in B. Each side is identical across all of its runs, and `box` is identical in A and B. Commits between `d53dd08` and `0d0d7e7` change the numerator preparation (`bc03412`, gammaloop pin and ninjars-graph) and the generic masters (`6487309`). This rerun did not find which commit causes the difference.
- **Pipeline at width 32 (B/A):** reduction 0.995–1.005, masters 0.992–1.000, total 0.996–1.006.

Tables:
- [results-0d0d7e7/fast-ab.txt](results-0d0d7e7/fast-ab.txt) and `.tsv`;
- A/A: [results-0d0d7e7/fast-aa.txt](results-0d0d7e7/fast-aa.txt);
- rerun: [results-0d0d7e7/rerun-gg-hhh-top-box-8.txt](results-0d0d7e7/rerun-gg-hhh-top-box-8.txt);
- instruction counts: [results-0d0d7e7/icount-fast.txt](results-0d0d7e7/icount-fast.txt), from 6 repetitions per side, identical to ≤ 0.013 %;
- [results-0d0d7e7/generic-ab.txt](results-0d0d7e7/generic-ab.txt);
- gates: [results-0d0d7e7/gate-check-A.txt](results-0d0d7e7/gate-check-A.txt) and `-B.txt`.

### The `syn-n2-r2` outlier is heap layout, not code placement ([results-0d0d7e7/layout-investigation.txt](results-0d0d7e7/layout-investigation.txt))

The first guard attributed `syn-n2-r2`'s +6 % to the rank n+2 modules sitting among the hot rank ≤ n+1 functions of the `ninjars` codegen unit. That attribution does not hold.

1. **Current main is not slower.** For `syn-n2-r2` w1 reduction, the screens (`ab_screen.sh`, 12 + 12 runs on each of two cores) give 0.952 and 0.972, and the full guard 0.993. Yet B places its hot functions exactly like the first guard's B (8346699): a 475 KiB span with 211 KiB of rank n+2 code inside ([hot-span.txt](results-0d0d7e7/hot-span.txt)). The fast-engine code is also the same, except 4 lines of `engine_x2.rs`.
2. **The first guard's pair still shows it on this host.** dec3d55 against 8346699 gives 1.060 and 1.072: 473–475 ns against 503–507 ns.
3. **Placement remedies applied to 8346699 do not remove it.** The table lists the `syn-n2-r2` w1 reduction against dec3d55, from `multi_screen.sh` (6 binaries round-robin, 8 rounds on each of two cores).

   | 8346699 variant | Hot span | Rank n+2 code inside | Ratio |
   |---|---|---|---|
   | as built | 475 KiB | 211 KiB | 1.072 |
   | the 75 rank n+2 symbols linked first (LLD `--symbol-ordering-file`) | 264 KiB | 0 | 1.084 (and 1.080 / 1.084 in `ab_screen.sh`) |
   | `#[cold]` on 42 rank n+2 functions ([coldmark.py](scripts/coldmark.py)) | 472 KiB | 218 KiB | 1.081 |
   | the same, linked with `-z keep-text-section-prefix` | 252 KiB | 0 | 1.089 |
   | `-z keep-text-section-prefix` alone | 473 KiB | 211 KiB | 1.141 |

   - **Why `#[cold]` alone does not move code.** rustc emits a codegen unit's functions sorted by mangled symbol name; the 122 `.text` sections of B's `ninjars` object are in byte-wise name order. rust-lld then concatenates `.text.unlikely.*` with the other `.text.*` sections in input order. `evaluate_x2`, already `#[cold]`, sits inside the hot span.
   - **A separate crate or codegen unit** for the rank n+2 engine can only produce the layouts of rows 2 and 4, with the rank n+2 code out of the span.
4. **Heap layout decides it.** [layout_pad.sh](scripts/layout_pad.sh) pads the case file with 0–256 bytes of whitespace. This changes only the size of the buffer the case-bench worker thread reads it into, and with it the addresses of the buffers allocated later in that thread's malloc arena.
   - 8346699 then takes 462–512 ns, and dec3d55 461–489 ns. At 4 of 12 layouts 8346699 is within 1 % of dec3d55, and at 2 more within 3 %.
   - A and B each move by up to 9 %.
   - A longer `-o` path, which the main thread allocates, changes nothing. With a single malloc arena (`GLIBC_TUNABLES=glibc.malloc.arena_max=1`), 8346699 matches dec3d55 at 7 of 8 path lengths.
5. **With every heap block of ≥ 64 bytes aligned to 64 bytes** (`LD_PRELOAD` of [align64.c](scripts/align64.c)), the outlier disappears: 8346699 takes 454–472 ns at 11 of 12 layouts (at the twelfth, its two runs took 462 and 553 ns), against 461–493 ns for dec3d55, and B 439–466 ns against 457–485 ns for A.
6. **Counters.** Per 1000 instructions, the line-crossing loads (`ls_misal_loads.ma64`) and load stalls rise with the slow layout and return to dec3d55's values with the fast one: 5.05–5.66 at the slow layout and 4.60 at the fast one, against 4.63–4.71 for dec3d55. The op-cache misses stay higher in 8346699 at the fast layout too (33.9–34.2 against 32.5), so they do not cause the extra time.
   - In a cycles profile, glibc's `memset` grows from 1.2–1.3 % to 3.3–3.6 % of the process.
   - 34 % of memset's samples fall on its 64-byte tail store, against 4 % in dec3d55.

**Conclusion.** At width 1, a replayed reduction of the smallest cases is about 450 ns and 5000 instructions per call. It moves by up to 10 % with the cache-line offsets of the worker thread's heap buffers, which depend on each binary's allocation history. Where the rank n+2 code is linked does not matter.

**The other width-1 keys move with layout as well.** [layout_sweep.sh](scripts/layout_sweep.sh) ran all 27 cases at 4 layouts, 3 runs each ([layout-sweep-ab.txt](results-0d0d7e7/layout-sweep-ab.txt)).
- The width-1 reduction ratios B/A have a geomean of 0.996 and a max of 1.044 over the layouts.
- A key's spread over layouts is 2 % at the median and up to 4.7 % (A) and 9.9 % (B).
- `gg-hhh-top-box-8` w1 reduction is 1.013 as the median over 12 layouts ([layout-pad-gg-hhh-top-box-8.txt](results-0d0d7e7/layout-pad-gg-hhh-top-box-8.txt)).

**Before attributing a width-1 ratio above 1.03 to code**, run that key over heap layouts (`layout_pad.sh`), or run both sides under `align64.c`.

**Probe cycles are too noisy to judge layout.** The in-process probe's cycle counts vary by ±10 % between repetitions on this host. `ab_icount.rs` times 400 calls per point at width 1. They are not used for timing here.

## Integration pass 2 (115fb46): machine code and instruction counts ([results/zero-overhead-115fb46.txt](results/zero-overhead-115fb46.txt))

Release CLIs and in-process probes of main `0d0d7e7` (A), pass 1's head `f33ff8d` (F) and `claude/todo-next-integration` `115fb46` (B), which carries the oneloopmaster pin `8ec256fc` and the Symbolica pin `9cdc2949`. Instruction counts only; the cycle-timed guard (`run_rerun.sh`) is run separately.
- **Machine code** (`dis_compare.py --no-alias`):
  - all 28 default stage instances (`boxes`, `bubbles`, `tadpoles`, `triangles`, `evaluate_up_to_n1`) are identical A -> B;
  - F -> B, every function outside oneloopmaster is identical, also with struct offsets kept (`offsets_cmp.py`);
  - A -> B outside oneloopmaster, the 33 differences are pass 1's;
  - oneloopmaster's native code differs in 1287 functions, from its pin.
- **`lane_rank`** is still inlined into `evaluate_declared_n2`: no out-of-line `TensorNumerator`/`MuTensorNumerator::lane_rank`. At width 32, n = 1..3, F -> B moves the live path without masters by at most 2.3 instructions per point.
- **Rank <= n+1** (54 keys), B against A: replayed reduction -0.001 % (-0.021 % to +0.016 %), live path +0.305 %. The live change is the masters' (+0.674 %); the live path without masters is +0.000 % (-0.035 % to +0.032 %).
- **Rank n+2** (222 keys), B against F: replay +0.000 %, live +0.116 %, all from the masters (+0.883 %).
- **Masters per family** (`ab_masters_family.rs`), A -> B: D0 +1.6 % instructions per row, C0 -0.3 % to -0.5 %, A0 +0.6 %, bubbles ±0.1 %.

Per-key numbers: [results/icount-115fb46.tsv](results/icount-115fb46.tsv). Probes: `ab_icount.rs` (unchanged), `ab_icount_split.rs` (adds a run with the replayed numerator and live masters, which separates the masters from the reduction and the numerator) and `ab_masters_family.rs`.

## Files

**scripts/**

| Script | Purpose |
|---|---|
| `build.sh` | the perf CLI build |
| `build_probes.sh` | builds the in-process probes `ab_icount.rs`, `ab_stackprobe.rs`, `ab_bubble_masters.rs` |
| `run_one.sh`, `ab.sh` | one pass of run_final2.sh, and its interleaved driver |
| `run_cases.sh`, `ab_cases.sh` | single-case reruns |
| `run_generic_one.sh`, `ab_generic.sh` | the generic bench A/B |
| `analyze.py`, `analyze_generic.py` | ratios and acceptance |
| `run_icount.sh`, `icount_compare.py`, `generic_icount.sh` | instruction counts |
| `ab_icount_split.rs`, `ab_masters_family.rs` | instruction counts of the live path split into reduction, masters and numerator, and per master family (integration pass 2) |
| `dis_compare.py` | machine-code comparison |
| `twin_pairs.py`, `twin_cost.sh`, `twin_report.py`, `make_twin_fixtures.py` | rank n+2 cost |
| `gate_check.py` | the absolute gates |
| `idle_l3.py` | core selection |
| `run_all.sh` | the sequence as run |
| `run_rerun.sh` | the rerun against main d53dd08, as run (`ab.sh`, `ab_cases.sh`, `ab_generic.sh` take `INPUTS_A`/`INPUTS_B`) |
| `ab_screen.sh`, `screen_analyze.py` | interleaved A/B on a few cases under `perf stat` (cycles, instructions) |
| `multi_screen.sh`, `multi_analyze.py` | the same, round-robin over several binaries |
| `hot_span.py` | address span of the hot rank <= n+1 functions and the rank n+2 code inside it |
| `x2_symbols.sh`, `relink_cli.sh`, `coldmark.py` | placement experiments: link order, linker options, `#[cold]` |
| `layout_pad.sh`, `pad_table.py`, `layout_sweep.sh`, `layout_analyze.py` | heap-layout sweeps (padded case files) |
| `align64.c` | `LD_PRELOAD` shim: heap blocks of >= 64 bytes aligned to 64 bytes |

**results/:** the tables cited above, plus [results/environment.txt](results/environment.txt).

**results-0d0d7e7/:** the rerun's tables, plus [results-0d0d7e7/environment.txt](results-0d0d7e7/environment.txt).

Raw per-run JSONs (about 47 MB) are not committed.

## Reproduce

Inside the dev shell, with the Symbolica licence set, export both revisions and run the whole sequence:

```sh
git archive dec3d55 | tar -x -C /tmp/a
git archive 8346699 | tar -x -C /tmp/b
validation/performance/rank-n2-ab/scripts/run_all.sh /tmp/a /tmp/b /tmp/w <cpu-x> <cpu-y> <cpu-z>
```

Pick the cores with `scripts/idle_l3.py 15`: two idle cores on different L3 complexes, and a third. The steps after the A/B (instruction counts, twin costs, gates) are listed at the end of `run_all.sh`.

For the rerun, use detached worktrees (`git worktree add --detach <dir> <rev>`) of `d53dd08`, `0d0d7e7`, `dec3d55` and `8346699`:

```sh
validation/performance/rank-n2-ab/scripts/run_rerun.sh <a> <b> <old-a> <old-b> <work> <cpu-x> <cpu-y> <cpu-z> <cpu-w>
```

The experiment commands are listed at its end.
