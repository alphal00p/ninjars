# Rank n+2 performance guard: main dec3d55 against claude/rank-n2-integration 8346699

This is the final A/B of design section 8 for rank n+2 support.
- **A (baseline):** main `dec3d55`.
- **B (candidate):** `claude/rank-n2-integration` `8346699`. It contains main `dec3d55`, merged in `1524dbd`.

Both were built with `cargo build --locked --profile perf -p ninjars-cli --features numerator-jit`. Each was built from an exported tree into its own target directory, and both ran on identical inputs. Host, toolchain, binary hashes and cores are listed in [results/environment.txt](results/environment.txt).

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
| `dis_compare.py` | machine-code comparison |
| `twin_pairs.py`, `twin_cost.sh`, `twin_report.py`, `make_twin_fixtures.py` | rank n+2 cost |
| `gate_check.py` | the absolute gates |
| `idle_l3.py` | core selection |
| `run_all.sh` | the sequence as run |

**results/:** the tables cited above, plus [results/environment.txt](results/environment.txt).

Raw per-run JSONs (about 47 MB) are not committed.

## Reproduce

Inside the dev shell, with the Symbolica licence set, export both revisions and run the whole sequence:

```sh
git archive dec3d55 | tar -x -C /tmp/a
git archive 8346699 | tar -x -C /tmp/b
validation/performance/rank-n2-ab/scripts/run_all.sh /tmp/a /tmp/b /tmp/w <cpu-x> <cpu-y> <cpu-z>
```

Pick the cores with `scripts/idle_l3.py 15`: two idle cores on different L3 complexes, and a third. The steps after the A/B (instruction counts, twin costs, gates) are listed at the end of `run_all.sh`.
